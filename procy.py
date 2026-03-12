#!/usr/bin/env python3
"""procy — transparent prompt proxy between human and AI agents.

Wraps Claude Code (or any CLI agent) as a PTY proxy. The TUI works
perfectly — procy is invisible unless you use ! commands.

! commands (intercepted by procy, not forwarded to agent):
  !evolve N        Procy-evolve: auto-generate N prompts via local Qwen
  !correct         Mark last prompt as wrong, enter correction
  !status          Show procy session info
  !history         Show prompt/correction history
  !train           Export SFT training data
  !stop            Stop ongoing evolve
  !help            Show commands

Everything else goes straight to the agent.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from terminal import ProxySession, _copy_terminal_size
from trace import TraceStore

# ── ANSI helpers (for writing directly to the real terminal) ──
def _raw_print(msg: str):
    """Print to stderr (bypasses the PTY, shows on the real terminal)."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _strip_ansi(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\r", "", text)


class Procy:
    def __init__(
        self,
        agent_cmd: list[str],
        cwd: str | None = None,
        db_path: str = "procy_traces.db",
        qwen_url: str | None = None,
    ):
        self.agent_cmd = agent_cmd
        self.cwd = cwd or os.getcwd()
        self.store = TraceStore(db_path)
        self.qwen_url = qwen_url  # e.g. http://10.89.7.3:8000 for Qwen on EXP06/07

        self.session_id: str | None = None
        self.turn_num = 0
        self.last_human_prompt: str = ""
        self._input_buffer = b""
        self._output_buffer = b""
        self._evolving = False
        self._stop_evolve = False
        self._proxy: ProxySession | None = None

        # Output capture state
        self._capture_output = False
        self._captured_output = b""

    # ── Input interception ──

    def _on_input(self, data: bytes) -> bytes | None:
        """Called for every chunk of user input. Return None to pass through."""
        # In raw mode, we get bytes one at a time (or small chunks).
        # Buffer until Enter (0x0d) to detect ! commands.
        self._input_buffer += data

        # Check for Enter key
        if b"\r" in data or b"\n" in data:
            line = self._input_buffer.decode("utf-8", errors="replace").strip()
            self._input_buffer = b""

            if line.startswith("!"):
                # Procy command — don't forward to agent
                self._handle_command(line)
                return b""  # swallow the input

            # Regular input — log it and pass through
            if line:
                self.turn_num += 1
                self.last_human_prompt = line
                self.store.log_turn(self.session_id, self.turn_num, "human", line)

            return None  # pass through unchanged

        # Not Enter yet — check if buffer looks like it might be a ! command
        buf_str = self._input_buffer.decode("utf-8", errors="replace")
        if buf_str.startswith("!") and len(buf_str) < 100:
            # Might be a procy command being typed — still pass through
            # (we'll intercept on Enter). But we need to show the typing
            # in the terminal. Since we're in raw mode and the child's
            # line editor would echo it, just pass through.
            return None

        return None  # pass through

    # ── Output observation ──

    def _on_output(self, data: bytes):
        """Called for every chunk of agent output. Just observes."""
        # Always log raw output
        if self._capture_output:
            self._captured_output += data

        # Parse for tool calls and responses (best effort)
        text = _strip_ansi(data)
        if text.strip():
            self.store.log_turn(
                self.session_id, self.turn_num, "agent_chunk", text[:2000],
            )

    # ── ! command handling ──

    def _handle_command(self, line: str):
        """Handle a procy ! command."""
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "!help":
            self._show_help()
        elif cmd == "!evolve":
            n = int(parts[1]) if len(parts) > 1 else 3
            self._start_evolve(n)
        elif cmd == "!correct":
            self._do_correct()
        elif cmd == "!status":
            self._show_status()
        elif cmd == "!history":
            self._show_history()
        elif cmd == "!train":
            self._do_train()
        elif cmd == "!stop":
            self._stop_evolve = True
            _raw_print("\033[35m[procy] Stopping evolve...\033[0m")
        else:
            _raw_print(f"\033[31m[procy] Unknown command: {cmd}. Type !help\033[0m")

    def _show_help(self):
        _raw_print("\033[35m[procy] Commands:\033[0m")
        _raw_print("  !evolve N    — procy-evolve: auto-generate N prompts")
        _raw_print("  !correct     — correct the last prompt (logs for training)")
        _raw_print("  !status      — session info")
        _raw_print("  !history     — prompt/correction history")
        _raw_print("  !train       — export SFT training pairs")
        _raw_print("  !stop        — stop ongoing evolve")
        _raw_print("  !help        — this message")

    def _show_status(self):
        sess = self.store.get_session(self.session_id) if self.session_id else {}
        turns = self.store.get_turns(self.session_id) if self.session_id else []
        corrections = self.store.get_corrections(self.session_id) if self.session_id else []
        _raw_print(f"\033[35m[procy] Session: {(self.session_id or 'none')[:8]}\033[0m")
        _raw_print(f"  turns: {len(turns)}  corrections: {len(corrections)}")
        _raw_print(f"  last prompt: {self.last_human_prompt[:80]}")
        if self.qwen_url:
            _raw_print(f"  qwen: {self.qwen_url}")

    def _show_history(self):
        turns = self.store.get_turns(self.session_id) if self.session_id else []
        corrections = self.store.get_corrections(self.session_id) if self.session_id else []
        _raw_print("\033[35m[procy] History:\033[0m")
        seen_turns = set()
        for t in turns:
            if t["role"] == "human":
                _raw_print(f"  \033[32m[human t{t['turn_num']}]\033[0m {t['content'][:120]}")
                seen_turns.add(t["turn_num"])
        if corrections:
            _raw_print("\033[33m  Corrections:\033[0m")
            for c in corrections:
                _raw_print(f"    original: {c['original_prompt'][:80]}")
                _raw_print(f"    corrected: {c['corrected_prompt'][:80]}")

    def _do_correct(self):
        """Let user correct the last prompt. Temporarily exit raw mode for input."""
        if not self.last_human_prompt:
            _raw_print("\033[31m[procy] No prompt to correct.\033[0m")
            return

        _raw_print(f"\033[33m[procy] Original: {self.last_human_prompt[:200]}\033[0m")
        _raw_print("\033[33m[procy] Type corrected prompt (or empty to cancel):\033[0m")

        # We need to read a line from the user. In raw mode this is tricky.
        # Temporarily restore terminal to get a clean readline.
        import termios
        old = termios.tcgetattr(sys.stdin.fileno())
        import tty
        # Restore cooked mode temporarily
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_terminal)
        try:
            corrected = input("\033[33m[correct]> \033[0m")
        except (EOFError, KeyboardInterrupt):
            corrected = ""
        finally:
            # Back to raw mode
            tty.setraw(sys.stdin.fileno())

        if not corrected.strip():
            _raw_print("\033[2m[procy] Cancelled.\033[0m")
            return

        self.store.log_correction(
            self.session_id, self.turn_num,
            self.last_human_prompt, corrected.strip(),
        )
        _raw_print(f"\033[32m[procy] Correction logged. Use !train to export.\033[0m")

    def _do_train(self):
        """Export SFT training pairs from corrections."""
        pairs = self.store.get_training_pairs()
        if not pairs:
            _raw_print("\033[31m[procy] No corrections to train from.\033[0m")
            return

        out_path = Path(self.cwd) / "procy_train.jsonl"
        with open(out_path, "w") as f:
            for p in pairs:
                entry = {
                    "instruction": p["original_prompt"],
                    "output": p["corrected_prompt"],
                }
                f.write(json.dumps(entry) + "\n")

        _raw_print(f"\033[32m[procy] Exported {len(pairs)} pairs to {out_path}\033[0m")

    # ── Evolve ──

    def _start_evolve(self, n: int):
        """Procy-evolve: generate N prompts and inject them into the agent."""
        if not self.last_human_prompt:
            _raw_print("\033[31m[procy] No previous prompt to evolve from.\033[0m")
            return

        if not self.qwen_url:
            _raw_print("\033[31m[procy] No --qwen-url set. Need local Qwen for evolve.\033[0m")
            _raw_print("\033[31m  Usage: procy --qwen-url http://EXP06:8000\033[0m")
            return

        self._evolving = True
        self._stop_evolve = False

        # Run evolve in a thread so the proxy loop continues
        t = threading.Thread(target=self._evolve_loop, args=(n,), daemon=True)
        t.start()

    def _evolve_loop(self, n: int):
        """Generate N prompt variants via Qwen and inject them."""
        corrections = self.store.get_corrections(self.session_id) or []
        evolve_history = self.store.get_evolve_runs(self.session_id) or []

        for i in range(1, n + 1):
            if self._stop_evolve:
                _raw_print(f"\033[35m[procy] Evolve stopped at iteration {i-1}/{n}\033[0m")
                break

            _raw_print(f"\033[35m[procy] ═══ evolve {i}/{n} ═══\033[0m")

            # Generate new prompt via Qwen
            new_prompt = self._generate_prompt(corrections, evolve_history)
            if not new_prompt:
                _raw_print(f"\033[31m[procy] Failed to generate prompt, skipping.\033[0m")
                continue

            _raw_print(f"\033[36m[procy] Injecting: {new_prompt[:120]}\033[0m")

            # Log it
            self.turn_num += 1
            self.last_human_prompt = new_prompt
            self.store.log_turn(self.session_id, self.turn_num, "procy", new_prompt)
            self.store.log_evolve(
                self.session_id, i, new_prompt,
                None, None, None, "procy",
            )

            # Inject into agent's stdin (type it as if the user typed it)
            self._inject_prompt(new_prompt)

            # Wait for agent to respond
            time.sleep(2)  # Give agent time to start processing
            self._wait_for_agent_idle(timeout=120)

            evolve_history.append({
                "iteration": i, "prompt": new_prompt, "score": None, "source": "procy",
            })

            _raw_print(f"\033[35m[procy] ═══ evolve {i}/{n} done ═══\033[0m")
            time.sleep(1)

        self._evolving = False
        _raw_print(f"\033[35m[procy] Evolve complete. Use !correct to fix any, !train to export.\033[0m")

    def _generate_prompt(self, corrections: list, history: list) -> str | None:
        """Call local Qwen to generate a new prompt variant."""
        import urllib.request

        meta_parts = [
            "You are a prompt engineer. Generate ONE improved prompt for a coding agent.",
            f"Base prompt:\n{self.last_human_prompt}\n",
        ]
        if corrections:
            meta_parts.append("Human corrections:")
            for c in corrections[-5:]:
                meta_parts.append(f"  Before: {c['original_prompt'][:200]}")
                meta_parts.append(f"  After: {c['corrected_prompt'][:200]}")
        if history:
            meta_parts.append("Previous evolve attempts:")
            for h in history[-5:]:
                meta_parts.append(f"  {h['prompt'][:150]}")
        meta_parts.append(
            "\nOutput ONLY the new prompt. No explanation, no markdown, just the prompt text."
        )

        payload = json.dumps({
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "messages": [{"role": "user", "content": "\n".join(meta_parts)}],
            "max_tokens": 1000,
            "temperature": 0.7,
        }).encode()

        try:
            url = f"{self.qwen_url.rstrip('/')}/v1/chat/completions"
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            _raw_print(f"\033[31m[procy] Qwen error: {e}\033[0m")
            return None

    def _inject_prompt(self, prompt: str):
        """Type a prompt into the agent's stdin as if the user typed it."""
        if self._proxy and self._proxy.master_fd:
            # Send the prompt followed by Enter
            os.write(self._proxy.master_fd, (prompt + "\r").encode())

    def _wait_for_agent_idle(self, timeout: float = 120):
        """Wait until agent seems to be done (heuristic: no output for a while)."""
        # Simple heuristic: wait until we see the prompt character (❯ or >)
        # or no output for 5 seconds after some output
        start = time.time()
        while time.time() - start < timeout:
            if self._stop_evolve:
                return
            time.sleep(1)
            # Check if the output buffer has the prompt indicator
            # (This is a rough heuristic — works for Claude Code's ❯ prompt)
            recent = self._captured_output[-500:] if self._captured_output else b""
            text = recent.decode("utf-8", errors="replace")
            if "❯" in text or "\n> " in text:
                # Agent is waiting for input
                time.sleep(1)  # Small buffer
                return
        _raw_print("\033[33m[procy] Timeout waiting for agent.\033[0m")

    # ── Main entry ──

    def run(self) -> int:
        """Start the procy proxy session."""
        self.session_id = self.store.new_session(goal="procy session")
        self._saved_terminal = None

        _raw_print("\033[1;35m  procy v0.2 — transparent prompt proxy\033[0m")
        _raw_print(f"\033[2m  session: {self.session_id[:8]}\033[0m")
        _raw_print(f"\033[2m  agent: {' '.join(self.agent_cmd)}\033[0m")
        if self.qwen_url:
            _raw_print(f"\033[2m  qwen: {self.qwen_url}\033[0m")
        _raw_print(f"\033[2m  type !help for procy commands\033[0m")
        _raw_print("─" * 60)

        # Save terminal settings before proxy takes over
        import termios
        if sys.stdin.isatty():
            self._saved_terminal = termios.tcgetattr(sys.stdin.fileno())

        # Start output capture
        self._capture_output = True

        self._proxy = ProxySession(
            cmd=self.agent_cmd,
            cwd=self.cwd,
            on_output=self._on_output,
            on_input=self._on_input,
        )
        exit_code = self._proxy.run()

        self.store.end_session(self.session_id)
        _raw_print(f"\n\033[35m[procy] Session ended. Trace: {self.store.db_path}\033[0m")
        return exit_code


def main():
    import argparse
    parser = argparse.ArgumentParser(description="procy — transparent prompt proxy")
    parser.add_argument("--agent", default="claude", help="Agent CLI command (default: claude)")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--db", default="procy_traces.db", help="Trace database")
    parser.add_argument("--qwen-url", default=None, help="Local Qwen API URL for evolve (e.g. http://10.89.7.3:8000)")
    args = parser.parse_args()

    agent_cmd = args.agent.split()
    procy = Procy(
        agent_cmd=agent_cmd,
        cwd=args.cwd,
        db_path=args.db,
        qwen_url=args.qwen_url,
    )
    sys.exit(procy.run())


if __name__ == "__main__":
    main()
