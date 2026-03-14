#!/usr/bin/env python3
"""procy CLI — system-level entry point.

Usage (from anywhere):
    procy                  # wraps `claude` with procy, starts UI on :7862
    procy --resume-procy 7c52a9b1   # continue procy trace + auto resume claude
    procy --agent codex    # wrap codex instead
    procy --no-ui          # skip auto-starting the monitor UI
    procy --no-tunnel      # skip SSH tunnel to EXP07 for Qwen
"""
from __future__ import annotations

import argparse
import atexit
import ast
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import re
from pathlib import Path

from .io import write_stdout
# ── Paths ──

PROCY_HOME = Path.home() / ".procy"
DEFAULT_DB = PROCY_HOME / "traces.db"
TUNNEL_PID_FILE = PROCY_HOME / "tunnel.pid"
UI_PID_FILE = PROCY_HOME / "ui.pid"

# SSH tunnel config
QWEN_SSH_HOST = "EXP07"        # from ~/.ssh/config
QWEN_REMOTE_PORT = 8000        # vLLM instance 0 port
QWEN_LOCAL_PORT = 18000        # local tunnel port (high to avoid conflicts)

# GPU inference / training config
QWEN_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
QWEN_SERVED_NAME = "qwen35"
QWEN_NUM_GPUS = 4
QWEN_INFERENCE_PORTS = list(range(8000, 8000 + QWEN_NUM_GPUS))
LORA_PATH = "/dev/shm/procy_lora"  # on EXP07, fast storage


def ensure_home():
    PROCY_HOME.mkdir(parents=True, exist_ok=True)


# ── ANSI helpers ──

def _info(msg: str):
    write_stdout(f"\r\033[2K\033[35m[procy]\033[0m {msg}\r\n")

def _dim(msg: str):
    write_stdout(f"\r\033[2K\033[2m  {msg}\033[0m\r\n")

def _err(msg: str):
    write_stdout(f"\r\033[2K\033[31m[procy]\033[0m {msg}\r\n")


# ── SSH Tunnel ──

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_tunnel() -> subprocess.Popen | None:
    """Start SSH tunnel: localhost:QWEN_LOCAL_PORT -> EXP07:QWEN_REMOTE_PORT."""
    if _port_in_use(QWEN_LOCAL_PORT):
        _dim(f"tunnel already active on :{QWEN_LOCAL_PORT}")
        return None

    _info(f"opening SSH tunnel to {QWEN_SSH_HOST}:{QWEN_REMOTE_PORT} -> localhost:{QWEN_LOCAL_PORT}")
    proc = subprocess.Popen(
        [
            "ssh", "-N", "-L",
            f"{QWEN_LOCAL_PORT}:127.0.0.1:{QWEN_REMOTE_PORT}",
            QWEN_SSH_HOST,
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ConnectTimeout=10",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait a moment and check it's alive
    time.sleep(2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode() if proc.stderr else ""
        _err(f"SSH tunnel failed: {err.strip()}")
        return None

    # Verify the port is now open
    if not _port_in_use(QWEN_LOCAL_PORT):
        _err("tunnel started but port not reachable — check SSH config")
        proc.kill()
        return None

    TUNNEL_PID_FILE.write_text(str(proc.pid))
    _dim(f"tunnel pid={proc.pid}")
    return proc


def stop_tunnel(proc: subprocess.Popen | None):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    TUNNEL_PID_FILE.unlink(missing_ok=True)


# ── GPU inference management ──

def _gpu_stop_inference(host: str):
    """Stop all procy_gpu* inference containers on host."""
    names = " ".join(f"procy_gpu{i}" for i in range(QWEN_NUM_GPUS))
    subprocess.run(
        ["ssh", host, f"docker stop {names} 2>/dev/null; docker rm {names} 2>/dev/null"],
        capture_output=True, timeout=30,
    )
    # Wait for GPUs to free
    time.sleep(3)


def _gpu_start_inference(host: str, use_lora: bool = False):
    """Start one vLLM inference container per GPU on host."""
    for i in range(QWEN_NUM_GPUS):
        port = QWEN_INFERENCE_PORTS[i]
        name = f"procy_gpu{i}"

        lora_flags = ""
        lora_mount = ""
        if use_lora:
            lora_mount = f"-v {LORA_PATH}:/lora"
            lora_flags = f"--enable-lora --lora-modules trained=/lora/latest"

        cmd = (
            f"docker run -d --name {name}"
            f" --gpus '\"device={i}\"'"
            f" --shm-size 4g"
            f" -e VLLM_DISABLE_PYNCCL=1"
            f" -e CUDA_VISIBLE_DEVICES=0"
            f" -v /dev/shm/hf_cache:/root/.cache/huggingface"
            f" {lora_mount}"
            f" -p {port}:8000"
            f" vllm/vllm-openai:latest"
            f" --model {QWEN_MODEL_ID}"
            f" --served-model-name {QWEN_SERVED_NAME}"
            f" --tensor-parallel-size 1"
            f" --gpu-memory-utilization 0.85"
            f" --max-model-len 4096"
            f" --max-num-seqs 2"
            f" --max-num-batched-tokens 512"
            f" --enforce-eager"
            f" --skip-mm-profiling"
            f" --limit-mm-per-prompt '{{\"image\":0,\"video\":0}}'"
            f" --port 8000"
            f" {lora_flags}"
        )
        subprocess.run(["ssh", host, cmd], capture_output=True, timeout=30)
        _dim(f"  started {name} on GPU {i}, port {port}")

    # Wait for servers to initialize
    _dim("waiting for inference servers to start...")
    time.sleep(90)
    # Quick health check on first instance
    result = subprocess.run(
        ["ssh", host, f"curl -sf http://localhost:{QWEN_INFERENCE_PORTS[0]}/v1/models"],
        capture_output=True, timeout=15,
    )
    if result.returncode == 0:
        _dim("inference servers ready")
    else:
        _dim("warning: servers may still be starting up")


# ── UI Server ──

def start_ui(db_path: str, port: int = 7862) -> subprocess.Popen | None:
    """Start the monitor UI as a background process."""
    if _port_in_use(port):
        _dim(f"UI already running on :{port}")
        return None

    _info(f"starting monitor UI on http://localhost:{port}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "procy.ui", "--db", db_path, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    if proc.poll() is not None:
        _err("UI failed to start")
        return None

    UI_PID_FILE.write_text(str(proc.pid))
    _dim(f"UI pid={proc.pid}")
    return proc


def stop_ui(proc: subprocess.Popen | None):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    UI_PID_FILE.unlink(missing_ok=True)


# ── Control-sequence cleanup ──
_RESIDUAL_CTRL_RE = re.compile(r"\[\?[0-9;]*[A-Za-z]")
# Match all Warp terminal integration markers:
#   |Warp(v0.2026.02.04.08.20.stable_03)>  or  P>|Warp(...)  etc.
_WARP_MARKER_RE = re.compile(r"\|?P?>\|?Warp\([^)]*\)>?|Warp\(v[^)]*\)>?")
# Stateless ANSI/control-sequence stripper (for read-only checks)
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[A-Za-z]"   # CSI sequences
    r"|\x1b\][^\x07]*\x07"     # OSC sequences (BEL terminated)
    r"|\x1b\][^\x1b]*\x1b\\"   # OSC sequences (ST terminated)
    r"|\x1bP[^\x1b]*\x1b\\"    # DCS sequences
    r"|\x1b[NOc()][^\x1b]?"    # SS2, SS3, RIS, charset
    r"|\x1b[\x20-\x2f]."       # 2-byte private sequences
    r"|\r"                      # carriage return
)
# Claude TUI spinner characters (each animation frame overwrites the previous)
_SPINNER_CHARS = set("✻✽✶✳✢·⏺●○◉◎☀★✦✧✹✸✷✺⊛⊙◐◑◒◓▪▫")
# Box-drawing / rule characters
_RULE_CHARS = set("─━═┄┈╌╍│┃║┆┊╎┌┐└┘├┤┬┴┼╔╗╚╝╠╣╦╩╬▬▐▌░▒▓")
# Spinner label fragments rendered char-by-char
_SPINNER_FRAGMENTS = {
    "Spinning…", "Spinning", "Sp", "Sn", "in", "ni", "ng", "g…",
    "pn", "ii", "nn", "pi", "i…",
    "Stewing…", "Stewing...", "Stew", "tew", "ewi", "win", "ing", "g..", "g...",
}
# Prompt hint patterns — must be the whole line (not embedded in content)
_PROMPT_HINT_RE = re.compile(
    r"^\s*[?!]\s+for\s+(shortcuts|bash\s+mode)\s*$"
    r"|^\s*esc\s*to\s*interrupt\s*$",
    re.IGNORECASE,
)
_STATUS_NOISE_RE = re.compile(
    r"(?:Stewing\.\.\.)+"
    r"|checking\s+for\s+updates"
    r"|bypass\s+permissions\s+on"
    r"|ctrl\+b\s+to\s+run\s+in\s+background"
    r"|running\s+in\s+the\s+background"
    r"|shift\+tab\s+to\s+cycle"
    r"|esc\s+to\s+interrupt"
    r"|\(thinking\s+with\s+\w+\s+effort\)"
    r"|No\s+recent\s+activity"
    r"|Claude\s+Code\s+v[\d.]+"
    r"|Claude\s+Max"
    r"|Haiku\s+[\d.]+\s+with\s+\w+\s+effort"
    r"|Sonnet\s+[\d.]+\s+with\s+\w+\s+effort"
    r"|Opus\s+[\d.]+\s+with\s+\w+\s+effort"
    r"|@\S+\.ai['\u2019]s\s+Organization"
    r"|plan\s+mode\s+on"
    r"|Entered\s+plan\s+mode",
    re.IGNORECASE,
)
_TOOL_CALL_RE = re.compile(r"^\s*(?:[•*-]\s*)?([A-Za-z][A-Za-z0-9_-]*)\((.*)\)\s*$")
_TOOL_RESULT_PREFIX_RE = re.compile(r"^\s*(?:[⎿└│]|\.\.\.\s*\+\d+\s+lines|\(timeout\b)", re.IGNORECASE)


def _compact_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _is_noise_line(stripped: str) -> bool:
    """Return True if the line is TUI decoration noise, not real content."""
    if not stripped:
        return True
    # Lines that are entirely rule/whitespace chars
    if all(ch in _RULE_CHARS or ch in " \t" for ch in stripped):
        return True
    # Lines that are entirely spinner chars
    if all(ch in _SPINNER_CHARS or ch in " \t" for ch in stripped):
        return True
    # Prompt hints
    if _PROMPT_HINT_RE.match(stripped):
        return True
    # Claude transient status lines
    if _STATUS_NOISE_RE.search(stripped):
        return True
    # Bare prompt chars: ❯ › > (possibly with whitespace)
    if re.match(r"^[❯›>]\s*$", stripped):
        return True
    # Spinner label fragments
    if stripped in _SPINNER_FRAGMENTS:
        return True
    # Lines starting with a spinner char followed by a short fragment (≤ 3 chars)
    # e.g. "✻nn", "✽Sp", "✶i…" — these are partial spinner renders
    if stripped and stripped[0] in _SPINNER_CHARS:
        rest = stripped[1:].strip()
        if len(rest) <= 3:
            return True
    # Very short non-alnum lines (likely animation residue)
    if len(stripped) <= 2 and not stripped.isalnum():
        return True
    if stripped == "...":
        return True
    # Single lowercase letter (spinner fragments like "i", "n", "g")
    if len(stripped) == 1 and stripped.islower():
        return True
    # Lines that are >60% rule/spinner chars (mixed noise)
    noise_count = sum(1 for ch in stripped if ch in _RULE_CHARS or ch in _SPINNER_CHARS)
    if len(stripped) > 3 and noise_count / len(stripped) > 0.6:
        return True
    return False


def _clean_for_db(text: str) -> str:
    """Remove TUI noise (spinners, rules, prompt hints, banner) before storing to DB."""
    # First pass: strip corrupted spinner text (overlapping cursor writes)
    # Pattern: repeated fragments of "Elucidating", "Thinking", etc. jumbled together
    text = re.sub(
        r"(?:Elu(?:ci)?(?:dat)?(?:ing)?|Ecud|cadiatn|Eliudid|Elung|cidating){2,}[.…]*",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:Think|Thin|Thi|nkin|king|inki|hink){2,}[.…]*",
        "", text, flags=re.IGNORECASE,
    )
    # Strip Claude TUI banner blocks (logo chars)
    text = re.sub(r"[▐▌▛▜▙▟█▀▄▝▘▗▖]+", "", text)
    # Strip "(thinking with X effort)" repeated
    text = re.sub(r"(?:\(thinking\s+with\s+\w+\s+effort\)\s*)+", "", text, flags=re.IGNORECASE)
    # Strip "⏵⏵bypass permissions on..." status lines
    text = re.sub(r"⏵+\s*bypass\s+permissions\s+on[^\n]*", "", text, flags=re.IGNORECASE)
    # Strip status mode indicators
    text = re.sub(r"⏵+\s*\w+\s+mode\s+on[^\n]*", "", text, flags=re.IGNORECASE)
    # Strip thinking/working indicators with ellipsis
    text = re.sub(r"(?:Elucidating|Stewing|Thinking|Working|Processing)[.…]+", "", text, flags=re.IGNORECASE)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _is_noise_line(stripped):
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        # Clean on original line (preserving indentation)
        # Strip spinner chars and labels: "hi ✻ Spinning…" → "hi", "⏺Hi" → "Hi"
        line_clean = re.sub(r"\s*[✻✽✶✳✢·⏺]\s*(?:Spinning…?)?", "", line)
        line_clean = re.sub(r"(?:Stewing\.\.\.)+", "", line_clean, flags=re.IGNORECASE)
        # Strip leading prompt chars from content: "❯ hi" → "hi"
        line_clean = re.sub(r"^[❯›>]\s+", "", line_clean)
        # Strip trailing prompt hints: "...today?❯ ? for shortcuts" → "...today?"
        line_clean = re.sub(r"[❯›>]\s*[?!]\s+for\s+(shortcuts|bash\s+mode)\s*$", "", line_clean, flags=re.IGNORECASE)
        line_clean = line_clean.rstrip()
        if not line_clean:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(line_clean)
    result = "\n".join(cleaned)
    # Collapse runs of 3+ blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ── Procy Core (inlined to avoid circular import issues) ──

from .terminal import ProxySession
from .store import TraceStore


class Procy:
    def __init__(
        self,
        agent_cmd: list[str],
        cwd: str | None = None,
        db_path: str = str(DEFAULT_DB),
        qwen_url: str | None = None,
        resume_procy: str | None = None,
        evolve_policy: str = "fixed",
    ):
        self.agent_cmd = agent_cmd
        self.cwd = cwd or os.getcwd()
        self.store = TraceStore(db_path)
        self.qwen_url = qwen_url
        self.resume_procy = (resume_procy or "").strip() or None
        self.evolve_policy = evolve_policy  # "fixed" or "proxy"
        self._resume_agent_session_id: str | None = None

        self.session_id: str | None = None
        self.turn_num = 0
        self.last_human_prompt: str = ""
        self._command_mode = False
        self._command_buffer = ""
        self._command_cursor = 0  # cursor position within buffer
        self._command_silent = False  # hide local command overlay while agent is actively streaming
        self._command_esc_mode = 0  # 0=none, 1=after ESC, 2=CSI, 3=OSC, 4=OSC_ESC
        self._command_csi_buf = ""
        self._command_in_paste = False
        self._evolving = False
        self._stop_evolve = False
        self._proxy: ProxySession | None = None
        self._capture_output = False
        self._captured_output = b""
        self._saved_terminal = None
        self._state_lock = threading.RLock()
        self._last_input_at = 0.0
        self._last_agent_output_at = 0.0
        self._typed_line_buffer = ""
        self._typed_esc_mode = 0  # 0=none, 1=after ESC, 2=CSI/SS3, 3=OSC, 4=OSC_ESC
        self._typed_csi_buf = ""
        self._typed_saw_cr = False
        self._in_bracketed_paste = False
        self._output_esc_mode = 0  # 0=none, 1=after ESC, 2=CSI/SS3, 3=OSC, 4=OSC_ESC
        self._output_seq = 0
        self._agent_log_buffer = ""
        self._agent_log_turn = 0
        self._agent_log_last_flush = time.time()
        self._action_parse_carry = ""
        self._pending_action: dict | None = None
        self._last_action_sig = ""
        self._evolve_thread: threading.Thread | None = None
        self._evolve_state = "idle"
        self._evolve_progress = (0, 0)
        self._evolve_note = ""
        self._evolve_response_buf = ""  # captures agent output during evolve iteration
        self._eval_autoset_generation = 0

    def _echo(self, text: str):
        """Echo text directly to the real terminal (bypasses PTY, works in raw mode)."""
        write_stdout(text)

    def _refresh_agent_screen(self, *, force: bool = False) -> None:
        """Ask the wrapped TUI to repaint after local procy overlays."""
        if not force:
            return
        if not self._proxy or not self._proxy.child_pid:
            return
        try:
            os.kill(self._proxy.child_pid, signal.SIGWINCH)
        except OSError:
            return
        # Intentionally avoid injecting Ctrl+L redraws here; they can cause
        # visible full-screen refresh flicker in the wrapped terminal UI.

    def _on_input(self, data: bytes) -> bytes | None:
        now = time.time()
        with self._state_lock:
            self._last_input_at = now
            if self._command_mode:
                # Recovery path: if command mode was hidden (silent) and empty,
                # don't swallow normal typing forever.
                if (
                    self._command_silent
                    and not self._command_buffer
                    and data
                    and len(data) == 1
                ):
                    b0 = data[0]
                    if 32 <= b0 <= 126 and b0 != ord("!"):
                        self._command_mode = False
                        self._command_silent = False
                        self._command_cursor = 0
                        self._command_esc_mode = 0
                        self._command_csi_buf = ""
                        self._command_in_paste = False
                        # Pass through to agent as normal input.
                        return None
                return self._handle_command_mode_input(data)
            if (
                not self._typed_line_buffer
                and data
                and data[0] == ord("!")
            ):
                quiet_for = time.time() - self._last_agent_output_at
                streaming = self._output_seq > 0 and quiet_for < 0.25
                self._command_mode = True
                self._command_buffer = ""
                self._command_cursor = 0
                # While evolve is running, keep command input visible so user can
                # type and get an explicit "rejected while running" response.
                self._command_silent = streaming and not self._evolving
                self._command_esc_mode = 0
                self._command_csi_buf = ""
                self._command_in_paste = False
                if not self._command_silent:
                    self._echo("\033[?25h")  # show cursor (Claude TUI hides it)
                return self._handle_command_mode_input(data)
            if self.session_id and data:
                try:
                    self.store.log_terminal_event(self.session_id, self.turn_num, "stdin", data)
                except Exception:
                    pass
            self._update_typed_line_locked(data)
            return None

    def _can_enter_command_mode_locked(self) -> bool:
        if self._output_seq == 0:
            return True
        quiet_for = time.time() - self._last_agent_output_at
        return self._is_agent_prompt_visible() and quiet_for >= 0.25

    def _update_typed_line_locked(self, data: bytes) -> None:
        for byte in data:
            if self._typed_esc_mode == 1:  # after ESC
                if byte in (ord("["), ord("O")):
                    self._typed_esc_mode = 2
                    self._typed_csi_buf = ""
                elif byte == ord("]"):
                    self._typed_esc_mode = 3
                else:
                    self._typed_esc_mode = 0
                continue
            if self._typed_esc_mode == 2:  # CSI/SS3
                if byte < 64:
                    self._typed_csi_buf += chr(byte)
                    if len(self._typed_csi_buf) > 16:
                        self._typed_csi_buf = self._typed_csi_buf[-16:]
                    continue
                if 64 <= byte <= 126:
                    seq = self._typed_csi_buf + chr(byte)
                    if seq == "200~":
                        self._in_bracketed_paste = True
                    elif seq == "201~":
                        self._in_bracketed_paste = False
                    self._typed_csi_buf = ""
                    self._typed_esc_mode = 0
                continue
            if self._typed_esc_mode == 3:  # OSC; end with BEL or ESC \
                if byte == 7:
                    self._typed_esc_mode = 0
                elif byte == 27:
                    self._typed_esc_mode = 4
                continue
            if self._typed_esc_mode == 4:  # OSC ESC \
                self._typed_esc_mode = 0
                continue
            if byte == 27:  # ANSI escape sequence
                self._typed_esc_mode = 1
                continue
            if byte in (8, 127):  # Backspace/Delete
                if self._typed_line_buffer:
                    self._typed_line_buffer = self._typed_line_buffer[:-1]
                continue
            if byte in (10, 13):  # Enter
                if byte == 10 and self._typed_saw_cr:
                    self._typed_saw_cr = False
                    continue
                self._typed_saw_cr = byte == 13
                if self._in_bracketed_paste:
                    if not self._typed_line_buffer.endswith("\n"):
                        self._typed_line_buffer += "\n"
                    continue
                line = self._sanitize_input_line(self._typed_line_buffer)
                self._typed_line_buffer = ""
                if line:
                    self._flush_pending_action_locked()
                    self._action_parse_carry = ""
                    self._last_action_sig = ""
                    self.turn_num += 1
                    self.last_human_prompt = line
                    self.store.log_turn(self.session_id, self.turn_num, "human", line)
                continue
            self._typed_saw_cr = False
            if byte < 32:
                continue
            ch = chr(byte)
            if ch.isprintable():
                self._typed_line_buffer += ch

    def _sanitize_input_line(self, line: str) -> str:
        if not line:
            return ""
        text = line.replace("\r", "\n")
        if not text:
            return ""
        text = _WARP_MARKER_RE.sub(" ", text)
        text = _RESIDUAL_CTRL_RE.sub(" ", text)
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if not text:
            return ""
        first_line = text.split("\n", 1)[0].lstrip()
        # Some terminal integrations prepend prompt markers like "|" or "P>|".
        # Strip these before command detection so "!..." doesn't get logged as
        # a normal human prompt.
        first_line = re.sub(r"^(?:[|>›❯\\\[\]\s]|P>)+", "", first_line)
        if first_line.startswith("!"):
            return ""
        cleaned_lines: list[str] = []
        for ln in text.split("\n"):
            s = ln.strip()
            if not s:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            low = s.lower()
            # Drop pure prompt marker residue.
            if re.match(r"^[|>›❯\\\[\]]+$", s):
                continue
            if s.startswith(("P>|", "|Warp", "❯", "›")):
                continue
            if "for shortcuts" in low or "for bash mode" in low:
                continue
            if re.match(r"^[Ww]arp\(", s):
                continue
            cleaned_lines.append(ln.rstrip())
        text = "\n".join(cleaned_lines).strip()
        return text

    def _sanitize_output_chunk(self, data: bytes) -> str:
        out = bytearray()
        for byte in data:
            if self._output_esc_mode == 1:  # after ESC
                if byte in (ord("["), ord("O")):
                    self._output_esc_mode = 2
                elif byte == ord("]"):
                    self._output_esc_mode = 3
                else:
                    self._output_esc_mode = 0
                continue
            if self._output_esc_mode == 2:  # CSI/SS3
                if 64 <= byte <= 126:
                    self._output_esc_mode = 0
                continue
            if self._output_esc_mode == 3:  # OSC
                if byte == 7:
                    self._output_esc_mode = 0
                elif byte == 27:
                    self._output_esc_mode = 4
                continue
            if self._output_esc_mode == 4:  # OSC ESC \
                self._output_esc_mode = 0
                continue
            if byte == 27:
                self._output_esc_mode = 1
                continue
            if byte == 13:
                continue
            if byte < 32 and byte not in (9, 10):
                continue
            out.append(byte)
        text = out.decode("utf-8", errors="replace")
        text = _WARP_MARKER_RE.sub("", text)
        text = _RESIDUAL_CTRL_RE.sub("", text)
        # Collapse stray pipe/angle-bracket noise left by terminal integrations
        text = re.sub(r"(?:^|\n)[|>P]+(?:\s*$|\s*\n)", "\n", text)
        return text

    def _handle_command_mode_input(self, data: bytes) -> bytes:
        if self._command_mode and self._command_silent:
            # Output is still moving; avoid local redraw flicker.
            quiet_for = time.time() - self._last_agent_output_at
            if self._output_seq == 0 or quiet_for >= 0.35:
                self._command_silent = False
                self._echo("\033[?25h")
                self._render_command_line_locked()
        needs_render = False
        for byte in data:
            # ── Escape sequence state machine ──
            if self._command_esc_mode == 1:  # after ESC
                if byte == ord("["):
                    self._command_esc_mode = 2  # CSI
                    self._command_csi_buf = ""
                elif byte == ord("O"):
                    self._command_esc_mode = 2  # SS3 (same handling)
                    self._command_csi_buf = ""
                elif byte == ord("]"):
                    self._command_esc_mode = 3  # OSC
                else:
                    self._command_esc_mode = 0
                continue
            if self._command_esc_mode == 2:  # CSI: params then final byte
                if byte < 64:  # parameter/intermediate bytes
                    self._command_csi_buf += chr(byte)
                    if len(self._command_csi_buf) > 16:
                        self._command_csi_buf = self._command_csi_buf[-16:]
                    continue
                if 64 <= byte <= 126:  # final byte
                    seq = self._command_csi_buf + chr(byte)
                    if seq == "200~":
                        self._command_in_paste = True
                    elif seq == "201~":
                        self._command_in_paste = False
                        needs_render = True
                    elif seq == "D":  # Left arrow
                        if self._command_cursor > 0:
                            self._command_cursor -= 1
                            needs_render = True
                    elif seq == "C":  # Right arrow
                        if self._command_cursor < len(self._command_buffer):
                            self._command_cursor += 1
                            needs_render = True
                    elif seq in ("H", "1~"):  # Home
                        self._command_cursor = 0
                        needs_render = True
                    elif seq in ("F", "4~"):  # End
                        self._command_cursor = len(self._command_buffer)
                        needs_render = True
                    elif seq == "3~":  # Delete key (forward delete)
                        if self._command_cursor < len(self._command_buffer):
                            self._command_buffer = (
                                self._command_buffer[:self._command_cursor]
                                + self._command_buffer[self._command_cursor + 1:]
                            )
                            needs_render = True
                    self._command_csi_buf = ""
                    self._command_esc_mode = 0
                continue
            if self._command_esc_mode == 3:  # OSC: end with BEL or ST
                if byte == 7:
                    self._command_esc_mode = 0
                elif byte == 27:
                    self._command_esc_mode = 4
                continue
            if self._command_esc_mode == 4:  # OSC ST (ESC \)
                self._command_esc_mode = 0
                continue

            # ── Normal bytes ──
            if byte == 27:
                self._command_esc_mode = 1
                continue
            if byte in (10, 13):
                if self._command_in_paste:
                    self._command_buffer += "\n"
                    self._command_cursor += 1
                    continue
                line = self._command_buffer.strip()
                self._command_mode = False
                self._command_buffer = ""
                self._command_cursor = 0
                self._command_esc_mode = 0
                self._command_in_paste = False
                was_silent = self._command_silent
                self._command_silent = False
                if not was_silent:
                    self._echo("\033[?25l\r\033[2K")
                if line:
                    self._handle_command(line)
                return b""
            if byte == 3:  # Ctrl-C
                self._command_mode = False
                self._command_buffer = ""
                self._command_cursor = 0
                self._command_esc_mode = 0
                self._command_in_paste = False
                if not self._command_silent:
                    self._echo("\033[?25l\r\033[2K^C")
                self._command_silent = False
                return b""
            if byte in (8, 127):  # Backspace
                if self._command_cursor > 0:
                    self._command_buffer = (
                        self._command_buffer[:self._command_cursor - 1]
                        + self._command_buffer[self._command_cursor:]
                    )
                    self._command_cursor -= 1
                    needs_render = True
                elif not self._command_buffer:
                    # Empty buffer + backspace = exit command mode
                    self._command_mode = False
                    self._command_esc_mode = 0
                    self._command_in_paste = False
                    if not self._command_silent:
                        self._echo("\033[?25l\r\033[2K")
                    self._command_silent = False
                    time.sleep(0.05)
                continue
            if byte == 1:  # Ctrl-A (Home)
                self._command_cursor = 0
                needs_render = True
                continue
            if byte == 5:  # Ctrl-E (End)
                self._command_cursor = len(self._command_buffer)
                needs_render = True
                continue
            if byte == 21:  # Ctrl-U (clear line)
                self._command_buffer = ""
                self._command_cursor = 0
                needs_render = True
                continue
            if byte == 11:  # Ctrl-K (kill to end)
                self._command_buffer = self._command_buffer[:self._command_cursor]
                needs_render = True
                continue
            ch = chr(byte)
            if ch.isprintable():
                self._command_buffer = (
                    self._command_buffer[:self._command_cursor]
                    + ch
                    + self._command_buffer[self._command_cursor:]
                )
                self._command_cursor += 1
                if not self._command_in_paste:
                    needs_render = True
        if needs_render and self._command_mode:
            self._render_command_line_locked()
        return b""

    def _render_command_line_locked(self) -> None:
        if not self._command_mode or self._command_silent:
            return
        display_full = self._command_buffer.replace("\n", " ↵ ")
        prefix_colored = "\033[35m[procy-cmd]\033[0m "
        prefix_visible_len = len("[procy-cmd] ")

        # Keep command overlay to a single clipped line so long inputs do not
        # wrap and leave repeated/split artifacts on screen.
        cols = 120
        try:
            cols = os.get_terminal_size(sys.stdout.fileno()).columns
        except OSError:
            pass
        cols = max(cols, 20)
        max_display = max(1, cols - prefix_visible_len - 1)

        buf_before = self._command_buffer[:self._command_cursor]
        newline_count = buf_before.count("\n")
        display_cursor_full = self._command_cursor + newline_count * 2
        display_cursor_full = max(0, min(display_cursor_full, len(display_full)))

        start = 0
        if len(display_full) > max_display:
            start = max(0, display_cursor_full - (max_display - 1))
            end = start + max_display
            if end > len(display_full):
                end = len(display_full)
                start = max(0, end - max_display)
            display = display_full[start:end]
        else:
            display = display_full

        display_cursor = max(0, min(display_cursor_full - start, len(display)))

        self._echo("\r\033[2K\033[?25h" + prefix_colored + display + "\033[0m")
        chars_after = len(display) - display_cursor
        if chars_after > 0:
            self._echo(f"\033[{chars_after}D")

    def _on_output(self, data: bytes):
        now = time.time()
        with self._state_lock:
            if self._command_mode and not self._command_silent and not self._evolving:
                # Agent resumed output while local command editor is open.
                # Hide overlay to prevent interleaving/flicker.
                self._command_silent = True
                self._echo("\r\033[2K\033[?25l")
            self._output_seq += 1
            if self.session_id and data:
                try:
                    self.store.log_terminal_event(self.session_id, self.turn_num, "stdout", data)
                except Exception:
                    pass
            if self._capture_output:
                self._captured_output += data
                if len(self._captured_output) > 500_000:
                    self._captured_output = self._captured_output[-250_000:]

            text = self._sanitize_output_chunk(data)
            if not text:
                return
            clean_text = _clean_for_db(text)
            if not clean_text:
                return
            # Only count meaningful (non-noise) output for quiet detection.
            # Spinner/status animations must not reset the timer, otherwise
            # _wait_for_agent_response_done never sees "quiet" and hangs.
            self._last_agent_output_at = now
            # Capture response text during evolve iterations
            if self._evolving and self._evolve_state == "waiting_response":
                self._evolve_response_buf += clean_text
            if self._agent_log_turn != self.turn_num:
                self._flush_agent_log_locked(force=True)
                self._flush_pending_action_locked()
                self._action_parse_carry = ""
                self._last_action_sig = ""
                self._agent_log_turn = self.turn_num
            self._extract_actions_from_text_locked(clean_text)
            self._agent_log_buffer += clean_text
            if (
                len(self._agent_log_buffer) >= 3000
                or (now - self._agent_log_last_flush) >= 1.0
                or "\n" in clean_text
            ):
                self._flush_agent_log_locked(force=False)

    def _on_resize(self, cols: int, rows: int):
        if not self.session_id:
            return
        payload = json.dumps({
            "type": "resize",
            "cols": int(cols),
            "rows": int(rows),
        }).encode("utf-8")
        try:
            self.store.log_terminal_event(self.session_id, self.turn_num, "meta", payload)
        except Exception:
            pass

    def _extract_actions_from_text_locked(self, text: str) -> None:
        stream = self._action_parse_carry + text
        lines = stream.split("\n")
        if stream and not stream.endswith("\n"):
            self._action_parse_carry = lines.pop()
        else:
            self._action_parse_carry = ""
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            m = _TOOL_CALL_RE.match(line)
            if m:
                self._flush_pending_action_locked()
                self._pending_action = {
                    "tool": m.group(1).lower(),
                    "input": m.group(2).strip(),
                    "result": [],
                }
                continue
            if not self._pending_action:
                continue
            if line.startswith(("└", "│", "...")):
                line = re.sub(r"^[└│]+\s*", "", line)
            if line == "...":
                continue
            # Ignore transient UI hints in tool output
            if _STATUS_NOISE_RE.search(line) or _PROMPT_HINT_RE.match(line):
                continue
            self._pending_action["result"].append(line)
            if len(self._pending_action["result"]) > 12:
                self._pending_action["result"] = self._pending_action["result"][-12:]

    def _flush_pending_action_locked(self) -> None:
        if not self._pending_action:
            return
        tool_name = self._pending_action["tool"]
        tool_input = self._pending_action["input"]
        tool_result = "\n".join(self._pending_action["result"]).strip()
        sig = f"{self.turn_num}|{tool_name}|{tool_input}"
        if tool_input and sig != self._last_action_sig:
            self.store.log_action(
                self.session_id,
                self.turn_num,
                tool_name,
                tool_input[:4000],
                tool_result[:8000],
            )
            self._last_action_sig = sig
        self._pending_action = None

    def _flush_agent_log_locked(self, force: bool) -> None:
        if self.turn_num <= 0:
            # Ignore pre-conversation terminal chatter (banner, UI chrome).
            self._agent_log_buffer = ""
            self._agent_log_last_flush = time.time()
            if force:
                self._flush_pending_action_locked()
            return
        if not self._agent_log_buffer.strip():
            self._agent_log_buffer = ""
            self._agent_log_last_flush = time.time()
            if force:
                self._flush_pending_action_locked()
            return
        if not force and (time.time() - self._agent_log_last_flush) < 0.2 and len(self._agent_log_buffer) < 3000:
            return
        chunk = self._agent_log_buffer[:2000]
        self._agent_log_buffer = self._agent_log_buffer[2000:]
        self._agent_log_last_flush = time.time()
        self.store.append_turn_content(
            self.session_id,
            self.turn_num,
            "agent",
            chunk,
        )
        if force and self._agent_log_buffer:
            self._flush_agent_log_locked(force=True)
        if force and not self._agent_log_buffer:
            if self._action_parse_carry.strip():
                self._extract_actions_from_text_locked("\n")
            self._flush_pending_action_locked()

    def _handle_command(self, line: str):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            _err(f"invalid command syntax: {exc}")
            return
        if not parts:
            return
        cmd = parts[0].lower()
        if self._evolving and cmd not in {
            "!status", "!help", "!stop", "!reset-evolve", "!evolve-status", "!estatus", "!history"
        }:
            _err("evolve is running; command rejected. Use !status or !stop.")
            return
        if cmd == "!help":
            self._show_help()
        elif cmd == "!evolve":
            try:
                n = int(parts[1]) if len(parts) > 1 else 3
            except ValueError:
                _err("usage: !evolve N")
                return
            self._start_evolve(n)
        elif cmd == "!correct":
            self._do_correct()
        elif cmd == "!status":
            self._show_status()
        elif cmd == "!history":
            self._show_history()
        elif cmd == "!train":
            self._do_train()
        elif cmd in ("!evolve-status", "!estatus"):
            self._show_status()
        elif cmd == "!stop":
            with self._state_lock:
                self._stop_evolve = True
                if self._evolving:
                    self._evolve_state = "stopping"
                    self._evolve_note = "stop requested by user"
            _info("stopping evolve...")
        elif cmd == "!reset-evolve":
            with self._state_lock:
                alive = bool(self._evolve_thread and self._evolve_thread.is_alive())
                if alive:
                    self._stop_evolve = True
                    self._evolve_state = "stopping"
                    self._evolve_note = "stop requested by reset"
                else:
                    self._stop_evolve = False
                    self._evolving = False
                    self._evolve_state = "idle"
                    self._evolve_progress = (0, 0)
                    self._evolve_note = "manually reset"
            if alive:
                _info("evolve thread is alive; requested stop. Run !status.")
            else:
                _info("evolve state reset")
        elif cmd == "!eval":
            self._handle_eval_command(parts[1:])
        elif cmd == "!deploy":
            self._handle_deploy_command(parts[1:])
        else:
            _err(f"unknown command: {cmd}. Type !help")

    def _show_help(self):
        _info("Commands:")
        _dim(f"!evolve N    — run N evolution iterations (policy: {self.evolve_policy})")
        _dim("!estatus     — show evolve status/progress")
        _dim("!correct     — correct the last prompt (logs for SFT training)")
        _dim("!status      — session info")
        _dim("!history     — prompt/correction history")
        _dim("!train       — train LoRA on all GPUs (stops/restarts inference)")
        _dim("!deploy trained   — switch to trained LoRA model")
        _dim("!deploy base      — switch to base model (no LoRA)")
        _dim("!deploy status    — show running containers & LoRA availability")
        _dim("!stop        — stop ongoing evolve")
        _dim("!reset-evolve— clear stuck evolve state")
        _dim("!eval set <path>         — set evaluator script for this session")
        _dim("!eval generate [desc]    — ask agent to write ./eval.py, then auto-register")
        _dim("!eval show               — show current evaluator")
        _dim("!eval run                — run evaluator manually")
        _dim("!eval metrics            — show eval results history")
        _dim("!help        — this message")

    def _show_status(self):
        turns = self.store.get_turns(self.session_id) if self.session_id else []
        corrections = self.store.get_corrections(self.session_id) if self.session_id else []
        _info(f"session: {(self.session_id or 'none')[:8]}")
        _dim(f"turns: {len(turns)}  corrections: {len(corrections)}")
        _dim(f"last prompt: {self.last_human_prompt[:80]}")
        _dim(f"db: {self.store.db_path}")
        if self.qwen_url:
            _dim(f"qwen: {self.qwen_url}")
        _dim(f"evolve policy: {self.evolve_policy}")
        with self._state_lock:
            cur, total = self._evolve_progress
            alive = bool(self._evolve_thread and self._evolve_thread.is_alive())
            _dim(f"evolve: state={self._evolve_state} running={self._evolving} thread_alive={alive} progress={cur}/{total}")
            if self._evolve_note:
                _dim(f"evolve-note: {self._evolve_note}")

    def _show_history(self):
        turns = self.store.get_turns(self.session_id) if self.session_id else []
        corrections = self.store.get_corrections(self.session_id) if self.session_id else []
        _info("History:")
        for t in turns:
            if t["role"] == "human":
                _dim(f"\033[32m[human t{t['turn_num']}]\033[0m {t['content'][:120]}")
        if corrections:
            _dim("\033[33mCorrections:\033[0m")
            for c in corrections:
                _dim(f"  original: {c['original_prompt'][:80]}")
                _dim(f"  corrected: {c['corrected_prompt'][:80]}")

    def _handle_eval_command(self, args: list[str]):
        if not self.session_id:
            _err("no active session")
            return
        if not args:
            _info("usage: !eval set <path> | !eval generate [desc] | !eval show | !eval run | !eval metrics")
            return
        sub = args[0].lower()

        if sub == "set":
            if len(args) < 2:
                _err("usage: !eval set <script_path> [--name <name>]")
                return
            rest = args[1:]
            name = "default"
            name_idx = None
            for i, tok in enumerate(rest):
                if tok == "--name":
                    name_idx = i
                    break
                if tok.startswith("--name="):
                    name = tok.split("=", 1)[1] or "default"
                    name_idx = i
                    break
            if name_idx is None:
                path_tokens = rest
            else:
                path_tokens = rest[:name_idx]
                if rest[name_idx] == "--name":
                    if name_idx + 1 >= len(rest):
                        _err("usage: !eval set <script_path> [--name <name>]")
                        return
                    name = rest[name_idx + 1]
            script_path = " ".join(path_tokens).strip()
            if not script_path:
                _err("usage: !eval set <script_path> [--name <name>]")
                return
            p = Path(script_path)
            if not p.is_absolute():
                p = Path(self.cwd) / p
            if not p.exists():
                _err(f"file not found: {p}")
                return
            suffix = p.suffix.lower()
            if suffix == ".py":
                run_cmd = f"python3 {{script}}"
            elif suffix == ".sh":
                run_cmd = f"bash {{script}}"
            elif suffix == ".js":
                run_cmd = f"node {{script}}"
            else:
                run_cmd = f"{{script}}"
            content = p.read_text()
            metrics_schema = self._detect_metrics_schema(content)
            eid = self.store.set_evaluator(
                self.session_id, name,
                script_path=str(p),
                script_content=content,
                run_command=run_cmd,
                metrics_schema=metrics_schema,
                created_by="human",
            )
            _info(f"evaluator '{name}' set: {p.name}")
            if metrics_schema:
                metric_names = [m["name"] for m in metrics_schema]
                _dim(f"detected metrics: {', '.join(metric_names)}")
            _dim(f"run command: {run_cmd}")
            _dim(f"evaluator id: {eid}")

        elif sub == "generate":
            desc = " ".join(args[1:]) if len(args) > 1 else ""
            task_context = self.last_human_prompt or "the current task"
            eval_desc = desc if desc else f"the task: {task_context[:200]}"
            eval_path = (Path(self.cwd) / "eval.py").resolve()
            prompt = (
                f"Write a Python evaluator script and save it to {eval_path}. "
                f"{eval_desc} "
                "Output format: the script must print exactly one JSON line to stdout with the metrics described above as numeric values, "
                "e.g. print(json.dumps({{\"metric1\": 0.95, \"metric2\": 1200}})). "
                "Exit 0 on success. Keep it self-contained. "
                "Define METRICS_SCHEMA at the top listing each metric with name, type, and goal (maximize or minimize). "
                "Write the file directly, do not print it in chat. "
                "After writing, briefly say what it measures and end with: WROTE eval.py"
            )
            # Wait for agent to be ready, then inject
            _info("waiting for Claude to be ready...")
            if not self._wait_for_agent_prompt(timeout=30):
                _err("Claude doesn't seem to be at a prompt. Try again when Claude is idle.")
                return
            self._inject_prompt(prompt)
            self._start_eval_autoset(eval_path, name="default", timeout_s=240)
            _info(f"prompt sent to Claude; waiting for {eval_path.name} to appear...")
            _dim("once written, procy auto-registers it. Use !eval show to confirm.")

        elif sub == "show":
            ev = self.store.get_evaluator(self.session_id)
            if not ev:
                _info("no evaluator set. Use: !eval set <path>")
                return
            _info(f"evaluator: {ev['name']}")
            _dim(f"script: {ev.get('script_path', 'inline')}")
            _dim(f"run: {ev.get('run_command', '?')}")
            _dim(f"created by: {ev.get('created_by', '?')}")
            schema = ev.get("metrics_schema")
            if schema:
                for m in schema:
                    _dim(f"  metric: {m['name']} ({m.get('type','?')}, goal={m.get('goal','?')})")

        elif sub == "run":
            ev = self.store.get_evaluator(self.session_id)
            if not ev:
                _err("no evaluator set. Use: !eval set <path>")
                return
            _info(f"running evaluator '{ev['name']}'...")
            result = self._run_evaluator(ev)
            raw = (result.get("raw_output") or "").strip()
            last_line = self._last_nonempty_line(raw) or self._last_eval_log_line()
            tail_msg = last_line[:220] if last_line else "(no evaluator output)"
            if result["exit_code"] == 0:
                _info("eval complete")
                _dim(f"log tail: {tail_msg}")
                metrics = result.get("metrics") or {}
                if isinstance(metrics, dict) and metrics:
                    for k, v in metrics.items():
                        _dim(f"{k} = {v}")
                else:
                    _dim("no JSON metrics parsed from evaluator output")
            else:
                _err(f"eval failed (exit {result['exit_code']})")
                _dim(f"log tail: {tail_msg}")
                if not last_line and result.get("raw_output"):
                    _dim(result["raw_output"][:500])

        elif sub == "metrics":
            results = self.store.get_eval_results(self.session_id)
            if not results:
                _info("no eval results yet")
                return
            _info(f"eval results ({len(results)}):")
            for r in results:
                tag = f"#{r.get('iteration', '?')}" if r.get('iteration') else "manual"
                metrics = r.get("metrics", {})
                metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) if isinstance(metrics, dict) else str(metrics)
                _dim(f"  [{tag}] {metric_str} (exit={r.get('exit_code', '?')}, {r.get('duration_s', 0):.1f}s)")
                trace = r.get("trace_metrics")
                if trace:
                    _dim(f"         trace: {json.dumps(trace)}")

        else:
            _err(f"unknown: !eval {sub}. Try: set, generate, show, run, metrics")

    def _start_eval_autoset(self, eval_path: Path, name: str = "default", timeout_s: float = 240.0) -> None:
        """Watch for eval_path creation/update and auto-register evaluator."""
        with self._state_lock:
            self._eval_autoset_generation += 1
            gen = self._eval_autoset_generation
        t = threading.Thread(
            target=self._watch_eval_file_and_set,
            args=(gen, eval_path, name, timeout_s),
            daemon=True,
            name="procy-eval-autoset",
        )
        t.start()

    def _watch_eval_file_and_set(self, generation: int, eval_path: Path, name: str, timeout_s: float) -> None:
        start = time.time()
        initial_mtime = None
        if eval_path.exists():
            try:
                initial_mtime = eval_path.stat().st_mtime
            except OSError:
                initial_mtime = None
        while (time.time() - start) < timeout_s:
            with self._state_lock:
                if generation != self._eval_autoset_generation:
                    return  # superseded by newer !eval generate
            if eval_path.exists() and eval_path.is_file():
                try:
                    st = eval_path.stat()
                    mtime = st.st_mtime
                    size = st.st_size
                except OSError:
                    mtime = None
                    size = 0
                changed = (initial_mtime is None) or (mtime is not None and mtime != initial_mtime)
                if size > 0 and changed:
                    try:
                        content = eval_path.read_text()
                    except Exception as exc:
                        _err(f"failed reading evaluator file: {exc}")
                        return
                    run_cmd = "python3 {script}" if eval_path.suffix.lower() == ".py" else "{script}"
                    metrics_schema = self._detect_metrics_schema(content)
                    eid = self.store.set_evaluator(
                        self.session_id,
                        name,
                        script_path=str(eval_path),
                        script_content=content,
                        run_command=run_cmd,
                        metrics_schema=metrics_schema,
                        created_by="claude",
                    )
                    _info(f"evaluator auto-set: {eval_path.name} (id={eid})")
                    if metrics_schema:
                        _dim("metrics: " + ", ".join(m["name"] for m in metrics_schema))
                    return
            time.sleep(0.5)
        _dim(f"auto-set timeout: {eval_path.name} not created/updated yet")

    def _eval_generate(self, description: str):
        """Ask Claude to write an evaluator script, capture it, and auto-register."""
        try:
            self._eval_generate_inner(description)
        except Exception as exc:
            import traceback
            _err(f"eval generate crashed: {exc}")
            _dim(traceback.format_exc()[:500])
            with self._state_lock:
                self._evolving = False
                self._evolve_state = "idle"
                self._evolve_response_buf = ""

    def _eval_generate_inner(self, description: str):
        if not self._proxy or not self._proxy.master_fd:
            _err("no agent running")
            return
        task_context = self.last_human_prompt or "the current task"
        if description:
            eval_desc = description
        else:
            eval_desc = f"the task: {task_context[:200]}"
        prompt = (
            f"Write a Python evaluator script for {eval_desc}.\n\n"
            "Requirements:\n"
            "- The script should evaluate the quality/correctness of the work done\n"
            "- It must print a single JSON line to stdout with numeric metrics\n"
            "  e.g. print(json.dumps({\"accuracy\": 0.95, \"latency_ms\": 42.1}))\n"
            "- The first metric in the JSON is the primary score\n"
            "- Exit code 0 = success, non-zero = failure\n"
            "- The script runs from the project root directory\n"
            "- Available env vars: $PROCY_SESSION, $PROCY_ITERATION, $PROCY_CWD\n"
            "- Keep it self-contained (no procy imports needed)\n\n"
            "Output ONLY the Python script in a single ```python code block, nothing else."
        )
        _info("asking Claude to generate evaluator...")
        _dim(f"context: {eval_desc[:100]}")
        with self._state_lock:
            self._evolve_response_buf = ""
            old_state = self._evolve_state
            old_evolving = self._evolving
            self._stop_evolve = False  # reset in case left over from previous run
            self._evolving = True
            self._evolve_state = "waiting_response"
        before_seq = self._output_seq
        self._inject_prompt(prompt)
        _dim(f"waiting for Claude to respond... (before_seq={before_seq}, stop_evolve={self._stop_evolve})")
        if not self._wait_for_agent_response_done(before_seq, timeout=120):
            with self._state_lock:
                buf_len = len(self._evolve_response_buf)
                cur_seq = self._output_seq
                stop = self._stop_evolve
                self._evolving = old_evolving
                self._evolve_state = old_state
                self._evolve_response_buf = ""
            _err(f"timeout waiting for Claude's response (seq={cur_seq}, buf={buf_len}, stop={stop})")
            return
        with self._state_lock:
            buf_len = len(self._evolve_response_buf)
        _dim(f"response captured ({buf_len} chars), extracting code...")
        with self._state_lock:
            response_text = _clean_for_db(self._evolve_response_buf)
            self._evolve_response_buf = ""
            self._evolving = old_evolving
            self._evolve_state = old_state
        if not response_text:
            _err("empty response from Claude")
            _dim(f"raw buf len was: {len(self._evolve_response_buf)}")
            return
        script_content = self._extract_code_block(response_text, "python")
        if not script_content:
            _err("no ```python code block found in response")
            _dim(f"response preview: {response_text[:300]}")
            return
        eval_name = "eval_generated"
        if description:
            slug = re.sub(r'[^a-z0-9]+', '_', description.lower().strip())[:30].strip('_')
            if slug:
                eval_name = f"eval_{slug}"
        eval_path = Path(self.cwd) / f"{eval_name}.py"
        counter = 1
        while eval_path.exists():
            eval_path = Path(self.cwd) / f"{eval_name}_{counter}.py"
            counter += 1
        eval_path.write_text(script_content)
        _info(f"saved evaluator: {eval_path.name}")
        metrics_schema = self._detect_metrics_schema(script_content)
        run_cmd = f"python3 {{script}}"
        eid = self.store.set_evaluator(
            self.session_id, "default",
            script_path=str(eval_path),
            script_content=script_content,
            run_command=run_cmd,
            metrics_schema=metrics_schema,
            created_by="claude",
        )
        _info(f"evaluator registered (id={eid})")
        if metrics_schema:
            metric_names = [m["name"] for m in metrics_schema]
            _dim(f"detected metrics: {', '.join(metric_names)}")
        _dim("run !eval run to test it, or !evolve N to start optimizing")

    def _extract_code_block(self, text: str, lang: str = "python") -> str | None:
        """Extract content of a fenced code block from text.

        Robust to language tags like ```py, ```python3, or no tag.
        """
        blocks: list[tuple[int, str]] = []
        for m in re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL):
            raw_tag = (m.group(1) or "").strip().lower()
            code = (m.group(2) or "").strip()
            if not code:
                continue
            score = 0
            if raw_tag:
                if raw_tag == lang or raw_tag.startswith(lang):
                    score += 6
                elif raw_tag in ("py", "python3", "py3"):
                    score += 5
                else:
                    score -= 1
            # Prefer blocks that look like Python source.
            if re.search(r"^\s*(import\s+\w+|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+|if __name__\s*==)", code, re.MULTILINE):
                score += 4
            if "json.dumps" in code:
                score += 2
            score += min(len(code), 6000) // 1200
            blocks.append((score, code))
        if blocks:
            blocks.sort(key=lambda x: x[0], reverse=True)
            return blocks[0][1]

        # Fallback: if model forgot fences, try to recover raw Python-ish text.
        raw = text.strip()
        if re.search(r"^\s*(import\s+\w+|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+|if __name__\s*==)", raw, re.MULTILINE):
            lines = []
            for ln in raw.splitlines():
                s = ln.strip()
                if re.match(r"^(here(?:'s| is)?|sure[,!:]?|below is|this script)", s, re.IGNORECASE):
                    continue
                if s.startswith("```"):
                    continue
                lines.append(ln)
            candidate = "\n".join(lines).strip()
            return candidate or None
        return None

    def _detect_metrics_schema(self, script_content: str) -> list[dict]:
        """Try to detect metric names from evaluator script output JSON keys."""
        m_schema = re.search(r"METRICS_SCHEMA\s*=\s*(\[[\s\S]*?\])", script_content)
        if m_schema:
            try:
                parsed = ast.literal_eval(m_schema.group(1))
                if isinstance(parsed, list):
                    normalized: list[dict] = []
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name") or "").strip()
                        if not name:
                            continue
                        goal = str(item.get("goal") or "maximize").strip().lower()
                        if goal not in ("maximize", "minimize"):
                            goal = "maximize"
                        mtype = str(item.get("type") or "float").strip().lower() or "float"
                        normalized.append({"name": name, "type": mtype, "goal": goal})
                    if normalized:
                        return normalized[:20]
            except Exception:
                pass
        keys_found = set()
        for m in re.finditer(r'json\.dumps\s*\(\s*\{([^}]+)\}', script_content):
            block = m.group(1)
            for km in re.finditer(r'["\'](\w+)["\']\s*:', block):
                keys_found.add(km.group(1))
        for m in re.finditer(r'(?:results?|metrics?|output)\s*=\s*\{([^}]+)\}', script_content):
            block = m.group(1)
            for km in re.finditer(r'["\'](\w+)["\']\s*:', block):
                keys_found.add(km.group(1))
        skip = {"type", "name", "error", "status", "message", "version",
                "model", "data", "input", "output", "content", "role",
                "__main__", "format", "help", "default", "action",
                "dim", "shape", "size", "length", "count", "index",
                "n_base", "n_query", "n_train", "n_test", "k"}
        keys_found -= skip
        schema = []
        for key in sorted(keys_found):
            goal = "maximize"
            if any(w in key.lower() for w in ("time", "latency", "error", "loss",
                                                "cost", "memory", "mem", "size")):
                goal = "minimize"
            schema.append({"name": key, "type": "float", "goal": goal})
        return schema[:10]

    def _infer_metrics_schema_from_metrics(self, metrics: dict) -> list[dict]:
        schema = []
        for key, val in metrics.items():
            if not isinstance(key, str):
                continue
            mtype = "float" if isinstance(val, (int, float)) else "str"
            goal = "maximize"
            if any(w in key.lower() for w in ("time", "latency", "error", "loss",
                                              "cost", "memory", "mem", "size")):
                goal = "minimize"
            schema.append({"name": key, "type": mtype, "goal": goal})
        return schema

    def _run_evaluator(self, evaluator: dict, evolve_run_id: int | None = None,
                        iteration: int | None = None) -> dict:
        """Run an evaluator script and parse its JSON output."""
        import subprocess as sp
        script_path = evaluator.get("script_path")
        run_cmd = evaluator.get("run_command", "python3 {script}")
        if not script_path or not Path(script_path).exists():
            content = evaluator.get("script_content")
            if not content:
                return {"metrics": {}, "raw_output": "no script", "exit_code": -1, "duration_s": 0}
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(content)
                script_path = f.name
        script_path_str = str(script_path)
        # Default placeholder expansion uses shell-safe quoting so paths with
        # spaces work (e.g. "/Users/.../New project/eval_generated.py").
        script_quoted = shlex.quote(script_path_str)
        if "{script_quoted}" in run_cmd:
            cmd = run_cmd.replace("{script_quoted}", script_quoted).replace("{script}", script_path_str)
        else:
            cmd = run_cmd.replace("{script}", script_quoted)
        env = os.environ.copy()
        env["PROCY_SESSION"] = self.session_id or ""
        env["PROCY_CWD"] = self.cwd or ""
        env["PROCY_DB"] = str(self.store.db_path) if self.store else ""
        if iteration is not None:
            env["PROCY_ITERATION"] = str(iteration)
        if evolve_run_id is not None:
            env["PROCY_EVOLVE_RUN_ID"] = str(evolve_run_id)
        changed = self._get_changed_files()
        if changed:
            env["PROCY_FILES_CHANGED"] = "\n".join(changed)
        log_path = Path(self.cwd) / "eval.log"
        start = time.time()
        try:
            # Stream output to eval.log so user can tail it
            with open(log_path, "w") as log_f:
                proc = sp.Popen(
                    cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT,
                    text=True, cwd=self.cwd, env=env,
                )
                output_lines = []
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        output_lines.append(line)
                        log_f.write(line)
                        log_f.flush()
                    elapsed = time.time() - start
                    if elapsed > 600:
                        proc.kill()
                        output_lines.append("KILLED: timeout 600s\n")
                        break
                proc.wait(timeout=5)
            duration = time.time() - start
            raw_output = "".join(output_lines).strip()
            metrics = {}
            for line in reversed(raw_output.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        metrics = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            trace_metrics = self._compute_trace_metrics()
            self.store.log_eval_result(
                self.session_id, evaluator["id"],
                metrics=metrics, raw_output=raw_output[:8000],
                exit_code=proc.returncode, duration_s=duration,
                trace_metrics=trace_metrics,
                evolve_run_id=evolve_run_id, iteration=iteration,
            )
            if metrics and not evaluator.get("metrics_schema"):
                inferred = self._infer_metrics_schema_from_metrics(metrics)
                if inferred:
                    try:
                        self.store.update_evaluator_metrics_schema(evaluator["id"], inferred)
                        evaluator["metrics_schema"] = inferred
                    except Exception:
                        pass
            if evolve_run_id and metrics:
                schema = evaluator.get("metrics_schema", [])
                primary = schema[0]["name"] if schema else next(iter(metrics), None)
                if primary and primary in metrics:
                    score = float(metrics[primary])
                    self.store.update_evolve_score(evolve_run_id, json.dumps(metrics), score)
            return {
                "metrics": metrics,
                "trace_metrics": trace_metrics,
                "raw_output": raw_output,
                "exit_code": proc.returncode,
                "duration_s": duration,
            }
        except Exception as e:
            return {"metrics": {}, "raw_output": str(e), "exit_code": -1, "duration_s": 0}

    def _last_nonempty_line(self, text: str) -> str:
        if not text:
            return ""
        for ln in reversed(text.splitlines()):
            ln = ln.strip()
            if ln:
                return ln
        return ""

    def _last_eval_log_line(self) -> str:
        """Best-effort tail(1) for eval.log in cwd."""
        try:
            p = Path(self.cwd) / "eval.log"
            if not p.exists():
                return ""
            data = p.read_text(errors="replace")
            return self._last_nonempty_line(data)
        except Exception:
            return ""

    def _get_changed_files(self) -> list[str]:
        """Return deduplicated list of files Claude wrote/edited this session."""
        if not self.session_id:
            return []
        actions = self.store.get_actions(self.session_id) if hasattr(self.store, "get_actions") else []
        seen = set()
        files = []
        for a in actions:
            if a.get("tool_name") in ("write", "edit"):
                path = (a.get("tool_input") or "").strip()
                if path and path not in seen:
                    seen.add(path)
                    files.append(path)
        return files

    def _compute_trace_metrics(self) -> dict:
        """Compute metrics from procy's own traces (tool calls, turn count, etc.)."""
        if not self.session_id:
            return {}
        turns = self.store.get_turns(self.session_id) or []
        actions = self.store.get_actions(self.session_id) if hasattr(self.store, "get_actions") else []
        return {
            "total_turns": len(turns),
            "total_tool_calls": len(actions),
            "tool_names": list(set(a.get("tool_name", "") for a in actions)) if actions else [],
        }

    def _do_correct(self):
        if not self.last_human_prompt:
            _err("no prompt to correct")
            return
        _info(f"Original: {self.last_human_prompt[:200]}")
        _info("Type corrected prompt (or empty to cancel):")
        import termios, tty
        if not sys.stdin.isatty() or self._saved_terminal is None:
            _err("interactive correction requires a TTY")
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_terminal)
        try:
            corrected = input("\033[33m[correct]> \033[0m")
        except (EOFError, KeyboardInterrupt):
            corrected = ""
        finally:
            tty.setraw(sys.stdin.fileno())
        if not corrected.strip():
            _dim("cancelled")
            return
        with self._state_lock:
            base_prompt = self.last_human_prompt
            turn_num = self.turn_num
        self.store.log_correction(
            self.session_id, turn_num,
            base_prompt, corrected.strip(),
        )
        _info("Correction logged. Use !train to export.")

    def _do_train(self):
        """Full training lifecycle: stop inference → train LoRA → restart inference."""
        pairs = self.store.get_training_pairs()
        all_data = self.store.get_training_data_all()
        sft_rows = [d for d in all_data if d.get("category") in ("human", "corrected")]
        if not sft_rows and not pairs:
            _err("no training data (need human prompts or corrections)")
            return
        _info(f"training data: {len(sft_rows)} SFT examples, {len(pairs)} correction pairs")

        def run():
            try:
                self._train_lifecycle(sft_rows)
            except Exception as exc:
                import traceback
                _err(f"train failed: {exc}")
                _dim(traceback.format_exc()[:500])

        t = threading.Thread(target=run, daemon=True, name="procy-train")
        t.start()
        _info("training started in background. Use !deploy status to check.")

    def _train_lifecycle(self, sft_rows):
        """Stop inference → export data → train LoRA → restart with LoRA."""
        host = QWEN_SSH_HOST

        # 1. Export training data
        _dim("exporting training data...")
        sft_lines = []
        for d in sft_rows:
            sft_lines.append(json.dumps({
                "instruction": d.get("context") or "Generate the next prompt.",
                "input": d.get("agent_response", "") or "",
                "output": d["prompt"],
                "category": d["category"],
            }))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(sft_lines))
            tmp_path = f.name

        script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_proxy.py"

        # 2. Upload data
        _dim(f"uploading to {host}:/tmp/procy_train/ ...")
        subprocess.run(["ssh", host, "mkdir -p /tmp/procy_train"], check=True, capture_output=True, timeout=10)
        subprocess.run(["scp", tmp_path, f"{host}:/tmp/procy_train/train.jsonl"], check=True, capture_output=True, timeout=30)
        if script_path.exists():
            subprocess.run(["scp", str(script_path), f"{host}:/tmp/procy_train/train_proxy.py"], check=True, capture_output=True, timeout=30)
        Path(tmp_path).unlink(missing_ok=True)

        # 3. Stop inference containers to free GPUs
        _info("stopping inference containers...")
        _gpu_stop_inference(host)

        # 4. Run training on all GPUs
        _info(f"training LoRA on {QWEN_NUM_GPUS} GPUs...")
        docker_cmd = (
            f"docker run --gpus all --rm --name procy_train_job"
            f" --entrypoint bash"
            f" -v /tmp/procy_train:/data"
            f" -v /dev/shm/hf_cache:/root/.cache/huggingface"
            f" -v {LORA_PATH}:/output"
            f" -e HF_HOME=/root/.cache/huggingface"
            f" vllm/vllm-openai:latest"
            f" -c 'pip install peft trl datasets accelerate 2>&1 | tail -3"
            f" && python3 /data/train_proxy.py"
            f" --data /data/train.jsonl"
            f" --output /output/latest"
            f" --model {QWEN_MODEL_ID}"
            f" --epochs 3"
            f" --lr 2e-4"
            f" 2>&1'"
        )
        subprocess.run(["ssh", host, "docker rm -f procy_train_job"], capture_output=True, timeout=10)
        result = subprocess.run(
            ["ssh", host, docker_cmd],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            _err(f"training failed (exit {result.returncode})")
            if result.stderr:
                _dim(result.stderr[-500:])
            # Restart base inference even if training failed
            _info("restarting base inference...")
            _gpu_start_inference(host, use_lora=False)
            return
        _info("training complete!")
        _dim(result.stdout[-300:] if result.stdout else "")

        # 5. Restart inference with trained LoRA
        _info("restarting inference with trained model...")
        _gpu_start_inference(host, use_lora=True)
        _info("deployed trained model on all GPUs")

    def _handle_deploy_command(self, args):
        if not args:
            _err("usage: !deploy [trained|base|status]")
            return
        sub = args[0].lower()
        host = QWEN_SSH_HOST
        if sub == "status":
            self._show_deploy_status(host)
        elif sub == "trained":
            _info("deploying trained model...")
            t = threading.Thread(target=self._deploy_model, args=(host, True), daemon=True)
            t.start()
        elif sub == "base":
            _info("deploying base model...")
            t = threading.Thread(target=self._deploy_model, args=(host, False), daemon=True)
            t.start()
        else:
            _err(f"unknown: !deploy {sub}. Use trained|base|status")

    def _deploy_model(self, host, use_lora):
        try:
            _gpu_stop_inference(host)
            _gpu_start_inference(host, use_lora=use_lora)
            label = "trained (LoRA)" if use_lora else "base"
            _info(f"deployed {label} model on {QWEN_NUM_GPUS} GPUs")
        except Exception as exc:
            _err(f"deploy failed: {exc}")

    def _show_deploy_status(self, host):
        try:
            result = subprocess.run(
                ["ssh", host, "docker ps --format '{{.Names}} {{.Status}}' --filter name=procy_gpu"],
                capture_output=True, text=True, timeout=10,
            )
            containers = result.stdout.strip()
            if containers:
                _info(f"running containers:")
                for line in containers.splitlines():
                    _dim(line)
            else:
                _dim("no inference containers running")

            # Check if LoRA exists
            result = subprocess.run(
                ["ssh", host, f"ls -la {LORA_PATH}/latest/adapter_config.json 2>/dev/null"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                _dim(f"trained LoRA available at {LORA_PATH}/latest/")
            else:
                _dim("no trained LoRA found")
        except Exception as exc:
            _err(f"status check failed: {exc}")

    def _start_evolve(self, n: int):
        if n < 1:
            _err("N must be >= 1")
            return
        with self._state_lock:
            last_prompt = self.last_human_prompt
            already_running = self._evolving
            quiet_for = time.time() - self._last_agent_output_at
            has_output = self._output_seq > 0
            if already_running and self._evolve_thread and not self._evolve_thread.is_alive():
                self._evolving = False
                self._evolve_state = "idle"
                already_running = False
        if already_running:
            cur, total = self._evolve_progress
            _err(f"evolve already running ({cur}/{total}, state={self._evolve_state}). Use !stop.")
            return

        if self.evolve_policy == "fixed":
            # Fixed policy: templates + Claude via PTY (no Qwen needed)
            with self._state_lock:
                self._evolving = True
                self._stop_evolve = False
                self._evolve_state = "starting"
                self._evolve_progress = (0, n)
                self._evolve_note = f"fixed policy, {n} iterations"
            t = threading.Thread(target=self._evolve_loop_fixed, args=(n,), daemon=True, name="procy-evolve")
            self._evolve_thread = t
            t.start()
        else:
            # Proxy policy: generate prompts via Qwen, inject into agent PTY
            if has_output and quiet_for < 0.25:
                _err("agent is still outputting; wait a moment, then run !evolve again")
                return
            if not last_prompt or not last_prompt.strip():
                _err("no previous prompt to evolve from")
                return
            if not self.qwen_url:
                _err("no Qwen available. Need --tunnel or --qwen-url")
                return
            with self._state_lock:
                self._evolving = True
                self._stop_evolve = False
                self._evolve_state = "starting"
                self._evolve_progress = (0, n)
                self._evolve_note = "proxy policy"
            t = threading.Thread(target=self._evolve_loop, args=(n,), daemon=True, name="procy-evolve")
            self._evolve_thread = t
            t.start()
        # Avoid local status prints here; they can shift rows relative to child PTY.

    def _evolve_loop(self, n: int):
        completed = 0
        try:
            corrections = self.store.get_corrections(self.session_id) or []
            evolve_history = self.store.get_evolve_runs(self.session_id) or []
            for i in range(1, n + 1):
                with self._state_lock:
                    self._evolve_progress = (i - 1, n)
                    if self._stop_evolve:
                        self._evolve_state = "stopped"
                        self._evolve_note = "stopped by user"
                        break
                    self._evolve_state = "waiting_prompt"
                    self._evolve_note = f"waiting for agent prompt (iteration {i}/{n})"
                if not self._wait_for_agent_prompt(timeout=60):
                    with self._state_lock:
                        self._evolve_state = "prompt_timeout"
                        self._evolve_note = "agent prompt not detected"
                    break
                with self._state_lock:
                    if self._stop_evolve:
                        self._evolve_state = "stopped"
                        self._evolve_note = "stopped by user"
                        break
                    base_prompt = self.last_human_prompt
                    self._evolve_state = "generating_prompt"
                    self._evolve_note = f"generating prompt (iteration {i}/{n})"
                if not base_prompt:
                    with self._state_lock:
                        self._evolve_state = "no_base_prompt"
                        self._evolve_note = "no base prompt available"
                    break

                new_prompt = self._generate_prompt(base_prompt, corrections, evolve_history)
                if not new_prompt:
                    with self._state_lock:
                        self._evolve_state = "generate_failed"
                        if not self._evolve_note:
                            self._evolve_note = "failed to generate prompt"
                    break
                with self._state_lock:
                    self.turn_num += 1
                    turn_num = self.turn_num
                    self.last_human_prompt = new_prompt
                    self._evolve_state = "injecting"
                    self._evolve_note = f"injecting prompt (iteration {i}/{n})"
                self.store.log_turn(self.session_id, turn_num, "procy", new_prompt)
                evolve_id = self.store.log_evolve(self.session_id, i, new_prompt, None, None, None, "procy")

                before_output_seq = self._output_seq
                with self._state_lock:
                    self._evolve_response_buf = ""  # clear before injection
                self._inject_prompt(new_prompt)
                with self._state_lock:
                    self._evolve_state = "waiting_response"
                    self._evolve_note = f"[#{i}] waiting for response"
                if not self._wait_for_agent_response_done(before_output_seq, timeout=120):
                    with self._state_lock:
                        self._evolve_state = "response_timeout"
                        self._evolve_note = f"[#{i}] response timeout"
                    break

                # Capture and store the response
                with self._state_lock:
                    response_text = _clean_for_db(self._evolve_response_buf)
                    self._evolve_response_buf = ""
                # Truncate for storage (keep first 4000 chars)
                response_summary = response_text[:4000] if response_text else ""
                self.store.update_evolve_response(evolve_id, response_summary)

                # Run evaluator if one is set
                score = None
                evaluator = self.store.get_evaluator(self.session_id)
                if evaluator:
                    with self._state_lock:
                        self._evolve_state = "evaluating"
                        self._evolve_note = f"[#{i}] running evaluator..."
                    eval_result = self._run_evaluator(evaluator, evolve_run_id=evolve_id, iteration=i)
                    raw = (eval_result.get("raw_output") or "").strip()
                    last_line = self._last_nonempty_line(raw) or self._last_eval_log_line()
                    if eval_result["exit_code"] == 0 and eval_result["metrics"]:
                        schema = evaluator.get("metrics_schema", [])
                        primary = schema[0]["name"] if schema else next(iter(eval_result["metrics"]), None)
                        if primary and primary in eval_result["metrics"]:
                            score = float(eval_result["metrics"][primary])
                        with self._state_lock:
                            self._evolve_note = f"[#{i}] eval: {json.dumps(eval_result['metrics'])}"
                        if last_line:
                            _dim(f"[#{i}] eval log tail: {last_line[:220]}")
                    else:
                        with self._state_lock:
                            self._evolve_note = f"[#{i}] eval failed (exit {eval_result['exit_code']})"
                        if last_line:
                            _dim(f"[#{i}] eval log tail: {last_line[:220]}")

                evolve_history.append({"iteration": i, "prompt": new_prompt, "response_summary": response_summary, "score": score, "source": "procy"})
                completed = i
                with self._state_lock:
                    self._evolve_progress = (completed, n)
                    self._evolve_state = "running"
                    self._evolve_note = f"[#{i}/{n}] complete" + (f" score={score:.4f}" if score is not None else "")
                time.sleep(0.2)
        except Exception as exc:
            with self._state_lock:
                self._evolve_state = "error"
                self._evolve_note = f"crashed: {exc}"
        finally:
            with self._state_lock:
                if completed >= n and n > 0:
                    self._evolve_state = "idle"
                    self._evolve_note = "completed"
                elif self._evolve_state in ("prompt_timeout", "response_timeout", "error", "stopped", "generate_failed", "no_base_prompt"):
                    pass
                else:
                    self._evolve_state = "idle"
                    if not self._evolve_note:
                        self._evolve_note = f"finished early ({completed}/{n})"
                self._evolving = False
                self._stop_evolve = False
                self._evolve_progress = (completed, n)

    def _evolve_loop_fixed(self, n: int):
        """Fixed-policy evolve loop: build prompt from templates, inject into Claude via PTY."""
        from .evolve_engine import EvolveEngine
        completed = 0
        try:
            engine = EvolveEngine(
                self.store, self.session_id,
                language="python",
            )

            # Seed population if empty — snapshot current git state
            if self.store.count_programs(self.session_id) == 0:
                commit = self._git_current_commit()
                engine.seed_population(commit_hash=commit)

            for i in range(1, n + 1):
                with self._state_lock:
                    self._evolve_progress = (i - 1, n)
                    if self._stop_evolve:
                        self._evolve_state = "stopped"
                        self._evolve_note = "stopped by user"
                        break
                    self._evolve_state = "waiting_prompt"
                    self._evolve_note = f"waiting for Claude (iteration {i}/{n})"

                # Wait for Claude to be at a prompt
                if not self._wait_for_agent_prompt(timeout=60):
                    with self._state_lock:
                        self._evolve_state = "prompt_timeout"
                        self._evolve_note = "Claude not at prompt"
                    break

                # Select parent and build prompt
                parent = engine.sample_parent()
                parent_hash = parent.get("commit_hash") if parent else None
                parent_fitness = parent.get("fitness_score", 0) if parent else 0

                eval_cmd = "python3 eval.py"
                evaluator = self.store.get_evaluator(self.session_id)
                if evaluator:
                    eval_cmd = evaluator.get("run_command", "python3 {script}").replace(
                        "{script}", evaluator.get("script_path", "eval.py"))

                focus = ""
                if parent_hash:
                    focus = f"Start from commit {parent_hash[:8]} (fitness {parent_fitness:.4f}). Run: git checkout {parent_hash[:8]}"

                _system_msg, user_msg = engine.build_prompt(
                    eval_command=eval_cmd, focus=focus,
                )

                # Inject into Claude
                with self._state_lock:
                    self.turn_num += 1
                    turn_num = self.turn_num
                    self.last_human_prompt = user_msg[:500]
                    self._evolve_state = "injecting"
                    self._evolve_note = f"[#{i}/{n}] injecting prompt into Claude"
                    self._evolve_response_buf = ""
                self.store.log_turn(self.session_id, turn_num, "procy", user_msg[:2000])
                evolve_id = self.store.log_evolve(self.session_id, i, user_msg[:500], None, None, None, "evolve_engine")

                before_seq = self._output_seq
                self._inject_prompt(user_msg)

                # Wait for Claude's response (longer timeout — Claude writes files)
                with self._state_lock:
                    self._evolve_state = "waiting_response"
                    self._evolve_note = f"[#{i}/{n}] Claude is working..."
                if not self._wait_for_agent_response_done(before_seq, timeout=300):
                    with self._state_lock:
                        self._evolve_state = "response_timeout"
                        self._evolve_note = f"[#{i}/{n}] response timeout"
                    break

                # Capture response summary
                with self._state_lock:
                    response_text = _clean_for_db(self._evolve_response_buf)
                    self._evolve_response_buf = ""
                response_summary = response_text[:4000] if response_text else ""
                self.store.update_evolve_response(evolve_id, response_summary)

                # Commit Claude's changes
                commit_hash = self._git_commit_evolve(i)

                # Run evaluator
                score = None
                metrics = {}
                evaluator = self.store.get_evaluator(self.session_id)
                if evaluator:
                    with self._state_lock:
                        self._evolve_state = "evaluating"
                        self._evolve_note = f"[#{i}/{n}] running evaluator..."
                    eval_result = self._run_evaluator(evaluator, evolve_run_id=evolve_id, iteration=i)
                    raw = (eval_result.get("raw_output") or "").strip()
                    last_line = self._last_nonempty_line(raw) or self._last_eval_log_line()
                    if eval_result["exit_code"] == 0 and eval_result["metrics"]:
                        metrics = eval_result["metrics"]
                        schema = evaluator.get("metrics_schema", [])
                        primary = schema[0]["name"] if schema else next(iter(metrics), None)
                        if primary and primary in metrics:
                            score = float(metrics[primary])
                        pass
                    # Don't print to stdout during evolve — it corrupts Claude's TUI

                # Add to population
                if commit_hash:
                    fitness = score if score is not None else 0.0
                    best = self.store.get_best_program(self.session_id)
                    best_fitness = best.get("fitness_score", 0) if best else 0
                    improved = fitness > best_fitness
                    engine.add_to_population(
                        commit_hash=commit_hash,
                        parent_id=parent["id"] if parent else None,
                        metrics=metrics, fitness_score=fitness,
                        changes_description=response_summary[:200] if response_summary else f"iter {i}",
                    )

                self.store.update_evolve_score(evolve_id, metrics if metrics else None, score)
                engine.maybe_migrate()

                completed = i
                with self._state_lock:
                    self._evolve_progress = (completed, n)
                    self._evolve_state = "running"
                    note = f"[#{i}/{n}]"
                    if score is not None:
                        note += f" score={score:.4f}"
                    if response_summary:
                        note += f" ({len(response_summary)} chars)"
                    self._evolve_note = note
                time.sleep(0.2)

        except Exception as exc:
            import traceback
            with self._state_lock:
                self._evolve_state = "error"
                self._evolve_note = f"crashed: {exc}"
            pass  # error stored in _evolve_note
        finally:
            with self._state_lock:
                if completed >= n and n > 0:
                    self._evolve_state = "idle"
                    best = self.store.get_best_program(self.session_id)
                    bf = best.get("fitness_score", 0) if best else 0
                    self._evolve_note = f"completed {completed}/{n}, best={bf:.4f}"
                elif self._evolve_state not in ("error", "stopped", "prompt_timeout", "response_timeout"):
                    self._evolve_state = "idle"
                    if not self._evolve_note:
                        self._evolve_note = f"finished ({completed}/{n})"
                self._evolving = False
                self._stop_evolve = False
                self._evolve_progress = (completed, n)

    def _git_current_commit(self) -> str | None:
        """Get current HEAD commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=self.cwd,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _git_commit_evolve(self, iteration: int) -> str | None:
        """Commit all changes and create a branch to anchor the commit."""
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=self.cwd,
            )
            if not status.stdout.strip():
                return self._git_current_commit()

            subprocess.run(
                ["git", "add", "-A"],
                capture_output=True, timeout=5, cwd=self.cwd,
            )
            sid = (self.session_id or "")[:8]
            result = subprocess.run(
                ["git", "commit", "-m", f"procy evolve {sid} iter {iteration}"],
                capture_output=True, text=True, timeout=10, cwd=self.cwd,
            )
            if result.returncode != 0:
                return None
            commit = self._git_current_commit()
            if commit:
                # Create a branch so the commit is never GC'd
                branch = f"evolve/{sid}/{iteration}"
                subprocess.run(
                    ["git", "branch", "-f", branch, commit],
                    capture_output=True, timeout=5, cwd=self.cwd,
                )
            return commit
        except Exception:
            return None

    def _is_agent_prompt_visible(self) -> bool:
        tail = self._captured_output[-4000:] if self._captured_output else b""
        text = tail.decode("utf-8", errors="replace")
        text = _ANSI_RE.sub("", text)
        text = _WARP_MARKER_RE.sub("", text)
        text = _RESIDUAL_CTRL_RE.sub("", text)
        if "for bash mode" in text.lower():
            return True
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        last = lines[-1]
        # Match common prompt chars (possibly preceded by path/context)
        return bool(re.search(r"[❯›>$%]\s*$", last))

    def _wait_for_agent_prompt(self, timeout: float = 120) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with self._state_lock:
                if self._stop_evolve:
                    return False
                quiet_for = time.time() - self._last_agent_output_at
                prompt_visible = self._is_agent_prompt_visible()
                user_typing = (time.time() - self._last_input_at) < 0.6
                command_mode = self._command_mode
                has_output = self._output_seq > 0
            if prompt_visible and quiet_for > 0.25 and not user_typing and not command_mode:
                return True
            # Fallback when prompt glyph can't be parsed for a given theme/agent.
            if has_output and quiet_for > 1.5 and not user_typing and not command_mode:
                return True
            time.sleep(0.1)
        return False

    def _wait_for_agent_response_done(self, before_output_seq: int, timeout: float = 120) -> bool:
        start = time.time()
        saw_new_output = False
        while time.time() - start < timeout:
            with self._state_lock:
                if self._stop_evolve:
                    return False
                if self._output_seq > before_output_seq:
                    saw_new_output = True
                quiet_for = time.time() - self._last_agent_output_at
                prompt_visible = self._is_agent_prompt_visible()
                user_typing = (time.time() - self._last_input_at) < 0.6
                command_mode = self._command_mode
            if saw_new_output and prompt_visible and quiet_for > 0.25 and not user_typing and not command_mode:
                return True
            if saw_new_output and quiet_for > 1.5 and not user_typing and not command_mode:
                return True
            time.sleep(0.1)
        return False

    def _generate_prompt(self, base_prompt: str, corrections: list, history: list) -> str | None:
        """Call the 14B proxy model (with LoRA) to generate the next prompt."""
        import urllib.request

        # System prompt: the proxy's role
        system_msg = (
            "You are a prompt optimization proxy. Given a task and the history of "
            "previous attempts with their scores, generate the next directional prompt "
            "that will improve results. Balance exploration (try new approaches) with "
            "exploitation (refine what works). Be specific and concise."
        )

        # Build user message with task + history
        user_parts = [f"Task: {base_prompt}"]

        if corrections:
            user_parts.append("\nHuman corrections:")
            for c in corrections[-5:]:
                user_parts.append(f"  Before: {c['original_prompt'][:200]}")
                user_parts.append(f"  After: {c['corrected_prompt'][:200]}")

        if history:
            user_parts.append("\nPrevious attempts:")
            best_score = 0
            for h in history[-8:]:
                tag = f"#{h.get('iteration', '?')}"
                resp = h.get("response_summary", "") or ""
                score = h.get("score")
                score_str = f", score={score:.4f}" if score is not None else ""
                user_parts.append(f"  [{tag}] Prompt: {h['prompt'][:150]}{score_str}")
                if resp:
                    user_parts.append(f"  [{tag}] Response: {resp[:200]}")
                if score is not None and score > best_score:
                    best_score = score
            user_parts.append(f"\nCurrent best score: {best_score:.4f}")

        user_parts.append("\nWhat should we try next?")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        # Use the proxy model (base or LoRA-adapted)
        payload = json.dumps({
            "model": QWEN_SERVED_NAME,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        }).encode()
        try:
            url = f"{self.qwen_url.rstrip('/')}/v1/chat/completions"
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                generated = data["choices"][0]["message"]["content"].strip()
                # The proxy outputs directional suggestions — use as the prompt
                return generated
        except Exception as e:
            with self._state_lock:
                self._evolve_note = f"proxy model error: {e}"
            return None

    def _inject_prompt(self, prompt: str):
        if self._proxy:
            # Flatten to single line, send via injection pipe (thread-safe,
            # forwarded to master_fd by the proxy_loop's main thread).
            flat = prompt.replace("\n", " ").replace("\r", "")
            # Use bracketed paste for single TUI redraw
            data = b"\x1b[200~" + flat.encode("utf-8", errors="replace") + b"\x1b[201~"
            self._proxy.inject(data)
            # Delay then Enter — must be a separate inject so the TUI
            # processes the paste first, then sees Enter as a new event.
            time.sleep(0.3)
            self._proxy.inject(b"\r")
            os.write(fd, b"\r")

    def _is_claude_command(self) -> bool:
        if not self.agent_cmd:
            return False
        cmd0 = os.path.basename(self.agent_cmd[0]).lower()
        return cmd0.startswith("claude")

    def _extract_resume_flag_value(self) -> str | None:
        for i, token in enumerate(self.agent_cmd):
            if token == "--resume":
                if i + 1 < len(self.agent_cmd):
                    return self.agent_cmd[i + 1]
                return None
            if token.startswith("--resume="):
                return token.split("=", 1)[1]
        return None

    def _has_agent_resume_flags(self) -> bool:
        if "--continue" in self.agent_cmd:
            return True
        return self._extract_resume_flag_value() is not None

    def _configure_agent_resume(self) -> None:
        if not self.session_id:
            return

        resume_value = self._extract_resume_flag_value()
        if resume_value:
            try:
                self.store.set_agent_session(self.session_id, resume_value)
            except Exception:
                pass
            return

        if self._has_agent_resume_flags():
            return

        if not self.resume_procy or not self._is_claude_command():
            return

        if self._resume_agent_session_id:
            self.agent_cmd += ["--resume", self._resume_agent_session_id]
            _dim(f"claude resume: --resume {self._resume_agent_session_id[:12]}")
            return

        self.agent_cmd += ["--continue"]
        _dim("claude resume: --continue (no stored session id)")

    def _initialize_or_resume_session(self) -> None:
        if not self.resume_procy:
            self.session_id = self.store.new_session(goal=f"procy @ {os.getcwd()}")
            return
        resolved = self.store.resolve_session_id(self.resume_procy)
        if not resolved:
            raise ValueError(
                f"unable to resolve procy session '{self.resume_procy}' "
                "(use full id or unique prefix from UI/!status)"
            )
        self.session_id = resolved
        self.store.mark_session_running(self.session_id)
        session = self.store.get_session(self.session_id) or {}
        stored_agent_session = str(session.get("agent_session_id") or "").strip()
        self._resume_agent_session_id = stored_agent_session or None
        turns = self.store.get_turns(self.session_id)
        if turns:
            self.turn_num = max(int(t.get("turn_num") or 0) for t in turns)
            for t in reversed(turns):
                if t.get("role") in ("human", "procy"):
                    self.last_human_prompt = str(t.get("content") or "").strip()
                    break
        self._agent_log_turn = self.turn_num

    def run(self) -> int:
        self._initialize_or_resume_session()
        self._configure_agent_resume()

        if self.resume_procy:
            _info(f"resumed session {self.session_id} (turn={self.turn_num})")
        else:
            _info(f"session {self.session_id}")
        _dim(f"resume: procy --resume-procy {self.session_id[:8]}")
        _dim(f"agent: {' '.join(self.agent_cmd)}")
        _dim(f"db: {self.store.db_path}")
        if self.qwen_url:
            _dim(f"qwen: {self.qwen_url}")
        _dim("type !help for procy commands")
        write_stdout("─" * 60 + "\r\n")

        import termios
        if sys.stdin.isatty():
            self._saved_terminal = termios.tcgetattr(sys.stdin.fileno())

        self._capture_output = True
        self._proxy = ProxySession(
            cmd=self.agent_cmd,
            cwd=self.cwd,
            on_output=self._on_output,
            on_input=self._on_input,
            on_resize=self._on_resize,
        )
        exit_code = self._proxy.run()
        with self._state_lock:
            self._flush_agent_log_locked(force=True)
        self.store.end_session(self.session_id)
        _info(f"session ended: {self.session_id}")
        _dim(f"resume: procy --resume-procy {self.session_id[:8]}")
        _dim(f"traces: {self.store.db_path}")
        return exit_code


# ── CLI entry point ──

def main():
    ensure_home()

    parser = argparse.ArgumentParser(
        prog="procy",
        description="procy — transparent prompt proxy for AI agents",
    )
    parser.add_argument("--agent", default="claude --dangerously-skip-permissions",
                        help="agent CLI command (default: claude --dangerously-skip-permissions)")
    parser.add_argument("--cwd", default=None,
                        help="working directory for the agent")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help=f"trace database (default: {DEFAULT_DB})")
    parser.add_argument("--resume-procy", default=None,
                        help="resume an existing procy trace session (and auto-resume claude via stored --resume or --continue)")
    parser.add_argument("--qwen-url", default=None,
                        help="Qwen API URL (default: auto-tunnel to EXP07)")
    parser.add_argument("--no-ui", action="store_true",
                        help="don't auto-start the monitor UI")
    parser.add_argument("--no-tunnel", action="store_true",
                        help="don't auto-start SSH tunnel to EXP07")
    parser.add_argument("--ui-port", type=int, default=7862,
                        help="monitor UI port (default: 7862)")
    parser.add_argument("--evolve-policy", default="fixed",
                        choices=["fixed", "proxy"],
                        help="evolve policy: 'fixed' (OpenEvolve-style templates) or 'proxy' (Qwen LLM policy)")
    args = parser.parse_args()

    tunnel_proc = None
    ui_proc = None

    def cleanup():
        stop_tunnel(tunnel_proc)
        stop_ui(ui_proc)

    atexit.register(cleanup)

    # SSH tunnel for Qwen
    qwen_url = args.qwen_url
    if not args.no_tunnel and not qwen_url:
        tunnel_proc = start_tunnel()
        if tunnel_proc or _port_in_use(QWEN_LOCAL_PORT):
            qwen_url = f"http://127.0.0.1:{QWEN_LOCAL_PORT}"

    # Monitor UI
    if not args.no_ui:
        ui_proc = start_ui(args.db, args.ui_port)

    # Run procy
    agent_cmd = shlex.split(args.agent)
    procy = Procy(
        agent_cmd=agent_cmd,
        cwd=args.cwd,
        db_path=args.db,
        qwen_url=qwen_url,
        resume_procy=args.resume_procy,
        evolve_policy=args.evolve_policy,
    )

    try:
        exit_code = procy.run()
    except ValueError as exc:
        _err(str(exc))
        exit_code = 2
    finally:
        cleanup()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
