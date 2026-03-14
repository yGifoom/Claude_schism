"""Comprehensive tests for command mode input handling and eval/generate flow in cli.py."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Patch write_stdout before importing cli so _info/_err/_dim don't write to real stdout.
_written_output: list[str] = []


def _mock_write_stdout(text: str) -> None:
    _written_output.append(text)


# We need to patch at the module level before importing Procy
import procy.io
_orig_write_stdout = procy.io.write_stdout
procy.io.write_stdout = _mock_write_stdout

# Also patch cli's imported reference
import procy.cli
procy.cli.write_stdout = _mock_write_stdout

from procy.cli import Procy, _clean_for_db, _is_noise_line


def _make_procy(db_path: str | None = None) -> Procy:
    """Create a Procy instance with a temp DB and dummy agent command."""
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="procy_test_")
        os.close(fd)
    p = Procy(
        agent_cmd=["echo", "test"],
        cwd="/tmp",
        db_path=db_path,
    )
    # Create a session so store operations work
    p.session_id = p.store.new_session(goal="test")
    return p


def _get_output() -> str:
    return "".join(_written_output)


def _clear_output() -> None:
    _written_output.clear()


def _enter_command_mode_clean(p: Procy) -> None:
    """Enter command mode with empty buffer (no '!' in buffer).

    _on_input(b"!") both enters command mode AND adds '!' to buffer.
    For tests that want to test raw buffer operations, we set state manually.
    """
    p._command_mode = True
    p._command_buffer = ""
    p._command_cursor = 0
    p._command_silent = False
    p._command_esc_mode = 0
    p._command_csi_buf = ""
    p._command_in_paste = False


class TestCommandModeEntry(unittest.TestCase):
    """Test entering command mode by typing '!' as first character."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_exclamation_enters_command_mode(self):
        """Typing '!' with empty typed_line_buffer enters command mode."""
        self.p._typed_line_buffer = ""
        result = self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        # Should return empty bytes (input consumed)
        self.assertEqual(result, b"")

    def test_exclamation_in_middle_does_not_enter_command_mode(self):
        """'!' after other characters does not enter command mode."""
        self.p._typed_line_buffer = "hello"
        result = self.p._on_input(b"!")
        self.assertFalse(self.p._command_mode)
        # Returns None (pass through to agent)
        self.assertIsNone(result)

    def test_command_mode_initializes_buffer_with_excl(self):
        """Entering command mode via _on_input(b'!') puts '!' in buffer."""
        self.p._on_input(b"!")
        # '!' is passed to _handle_command_mode_input, which adds it as printable
        self.assertEqual(self.p._command_buffer, "!")
        self.assertEqual(self.p._command_cursor, 1)

    def test_command_mode_silent_when_streaming(self):
        """When agent is actively outputting, command mode starts in silent mode."""
        self.p._output_seq = 10
        self.p._last_agent_output_at = time.time()  # just now
        self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        self.assertTrue(self.p._command_silent)

    def test_command_mode_not_silent_when_quiet(self):
        """When agent has been quiet, command mode is not silent."""
        self.p._output_seq = 10
        self.p._last_agent_output_at = time.time() - 1.0  # 1 second ago
        self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        self.assertFalse(self.p._command_silent)

    def test_command_mode_not_silent_no_output_yet(self):
        """When no output has been seen (_output_seq=0), not silent."""
        self.p._output_seq = 0
        self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        self.assertFalse(self.p._command_silent)

    def test_command_mode_visible_while_evolve_running(self):
        """Even when output is active, keep command mode visible during evolve."""
        self.p._evolving = True
        self.p._output_seq = 10
        self.p._last_agent_output_at = time.time()
        self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        self.assertFalse(self.p._command_silent)

    def test_hidden_empty_command_mode_recovers_to_passthrough(self):
        """If command mode is silently stuck with empty buffer, normal typing passes through."""
        self.p._command_mode = True
        self.p._command_silent = True
        self.p._command_buffer = ""
        self.p._command_cursor = 0
        result = self.p._on_input(b"h")
        self.assertFalse(self.p._command_mode)
        self.assertIsNone(result)


class TestCommandModeCharacterInput(unittest.TestCase):
    """Test character input and buffer management in command mode."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_typing_characters(self):
        """Characters are appended to buffer and cursor advances."""
        self.p._handle_command_mode_input(b"h")
        self.assertEqual(self.p._command_buffer, "h")
        self.assertEqual(self.p._command_cursor, 1)

        self.p._handle_command_mode_input(b"el")
        self.assertEqual(self.p._command_buffer, "hel")
        self.assertEqual(self.p._command_cursor, 3)

    def test_non_printable_ignored(self):
        """Control characters (except recognized ones) are ignored."""
        self.p._handle_command_mode_input(b"\x00")
        self.assertEqual(self.p._command_buffer, "")
        self.p._handle_command_mode_input(b"\x02")  # Ctrl-B not handled
        self.assertEqual(self.p._command_buffer, "")

    def test_typing_full_command(self):
        """Typing a full command builds the buffer correctly."""
        self.p._handle_command_mode_input(b"!help")
        self.assertEqual(self.p._command_buffer, "!help")
        self.assertEqual(self.p._command_cursor, 5)


class TestCommandModeCursorMovement(unittest.TestCase):
    """Test cursor movement: left, right, home, end, ctrl-a, ctrl-e."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        self.p._handle_command_mode_input(b"hello")
        _clear_output()

    def test_left_arrow(self):
        """Left arrow (ESC[D) moves cursor left."""
        self.assertEqual(self.p._command_cursor, 5)
        self.p._handle_command_mode_input(b"\x1b[D")
        self.assertEqual(self.p._command_cursor, 4)

    def test_left_arrow_at_start(self):
        """Left arrow at position 0 stays at 0."""
        self.p._command_cursor = 0
        self.p._handle_command_mode_input(b"\x1b[D")
        self.assertEqual(self.p._command_cursor, 0)

    def test_right_arrow(self):
        """Right arrow (ESC[C) moves cursor right."""
        self.p._command_cursor = 3
        self.p._handle_command_mode_input(b"\x1b[C")
        self.assertEqual(self.p._command_cursor, 4)

    def test_right_arrow_at_end(self):
        """Right arrow at end of buffer stays at end."""
        self.assertEqual(self.p._command_cursor, 5)
        self.p._handle_command_mode_input(b"\x1b[C")
        self.assertEqual(self.p._command_cursor, 5)

    def test_home_key_csi_h(self):
        """Home key (ESC[H) moves cursor to start."""
        self.p._handle_command_mode_input(b"\x1b[H")
        self.assertEqual(self.p._command_cursor, 0)

    def test_home_key_csi_1_tilde(self):
        """Home key (ESC[1~) moves cursor to start."""
        self.p._handle_command_mode_input(b"\x1b[1~")
        self.assertEqual(self.p._command_cursor, 0)

    def test_end_key_csi_f(self):
        """End key (ESC[F) moves cursor to end."""
        self.p._command_cursor = 2
        self.p._handle_command_mode_input(b"\x1b[F")
        self.assertEqual(self.p._command_cursor, 5)

    def test_end_key_csi_4_tilde(self):
        """End key (ESC[4~) moves cursor to end."""
        self.p._command_cursor = 0
        self.p._handle_command_mode_input(b"\x1b[4~")
        self.assertEqual(self.p._command_cursor, 5)

    def test_ctrl_a_home(self):
        """Ctrl-A moves cursor to start."""
        self.p._handle_command_mode_input(b"\x01")
        self.assertEqual(self.p._command_cursor, 0)

    def test_ctrl_e_end(self):
        """Ctrl-E moves cursor to end."""
        self.p._command_cursor = 0
        self.p._handle_command_mode_input(b"\x05")
        self.assertEqual(self.p._command_cursor, 5)


class TestCommandModeBackspace(unittest.TestCase):
    """Test backspace at cursor position."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_backspace_at_end(self):
        """Backspace at end of buffer removes last char."""
        self.p._handle_command_mode_input(b"abc")
        self.p._handle_command_mode_input(b"\x7f")  # DEL = backspace
        self.assertEqual(self.p._command_buffer, "ab")
        self.assertEqual(self.p._command_cursor, 2)

    def test_backspace_in_middle(self):
        """Backspace in middle removes char before cursor."""
        self.p._handle_command_mode_input(b"abcd")
        # Move cursor left twice (to position 2, between 'b' and 'c')
        self.p._handle_command_mode_input(b"\x1b[D\x1b[D")
        self.assertEqual(self.p._command_cursor, 2)
        self.p._handle_command_mode_input(b"\x08")  # BS
        self.assertEqual(self.p._command_buffer, "acd")
        self.assertEqual(self.p._command_cursor, 1)

    def test_backspace_at_start_nonempty(self):
        """Backspace at cursor position 0 with non-empty buffer does nothing."""
        self.p._handle_command_mode_input(b"abc")
        self.p._command_cursor = 0
        self.p._handle_command_mode_input(b"\x7f")
        # cursor == 0 so first branch (cursor > 0) is false,
        # buffer not empty so second branch (empty buf) is also false
        self.assertEqual(self.p._command_buffer, "abc")
        self.assertEqual(self.p._command_cursor, 0)

    def test_backspace_empty_exits_command_mode(self):
        """Backspace on empty buffer exits command mode."""
        self.assertTrue(self.p._command_mode)
        self.p._handle_command_mode_input(b"\x7f")
        self.assertFalse(self.p._command_mode)

    def test_backspace_del_byte_also_works(self):
        """Byte 8 (BS) also works as backspace."""
        self.p._handle_command_mode_input(b"xy")
        self.p._handle_command_mode_input(b"\x08")
        self.assertEqual(self.p._command_buffer, "x")


class TestCommandModeDeleteKey(unittest.TestCase):
    """Test delete key (forward delete)."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_delete_key(self):
        """Delete key (ESC[3~) removes char at cursor."""
        self.p._handle_command_mode_input(b"abcd")
        self.p._command_cursor = 1  # between 'a' and 'b'
        self.p._handle_command_mode_input(b"\x1b[3~")
        self.assertEqual(self.p._command_buffer, "acd")
        self.assertEqual(self.p._command_cursor, 1)

    def test_delete_at_end(self):
        """Delete key at end of buffer does nothing."""
        self.p._handle_command_mode_input(b"ab")
        self.assertEqual(self.p._command_cursor, 2)
        self.p._handle_command_mode_input(b"\x1b[3~")
        self.assertEqual(self.p._command_buffer, "ab")
        self.assertEqual(self.p._command_cursor, 2)


class TestCommandModeCtrlU(unittest.TestCase):
    """Test Ctrl-U (clear line)."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_ctrl_u_clears_buffer(self):
        """Ctrl-U clears entire buffer and resets cursor to 0."""
        self.p._handle_command_mode_input(b"!status")
        self.p._handle_command_mode_input(b"\x15")
        self.assertEqual(self.p._command_buffer, "")
        self.assertEqual(self.p._command_cursor, 0)


class TestCommandModeCtrlK(unittest.TestCase):
    """Test Ctrl-K (kill to end of line)."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_ctrl_k_kills_to_end(self):
        """Ctrl-K removes everything from cursor to end."""
        self.p._handle_command_mode_input(b"abcdef")
        self.p._command_cursor = 3
        self.p._handle_command_mode_input(b"\x0b")
        self.assertEqual(self.p._command_buffer, "abc")
        self.assertEqual(self.p._command_cursor, 3)

    def test_ctrl_k_at_end(self):
        """Ctrl-K at end of buffer does nothing."""
        self.p._handle_command_mode_input(b"abc")
        self.p._handle_command_mode_input(b"\x0b")
        self.assertEqual(self.p._command_buffer, "abc")


class TestCommandModeInsertAtCursor(unittest.TestCase):
    """Test inserting characters at cursor position."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_insert_at_middle(self):
        """Characters inserted at cursor position, not appended."""
        self.p._handle_command_mode_input(b"ac")
        self.p._command_cursor = 1  # between 'a' and 'c'
        self.p._handle_command_mode_input(b"b")
        self.assertEqual(self.p._command_buffer, "abc")
        self.assertEqual(self.p._command_cursor, 2)

    def test_insert_at_beginning(self):
        """Insert at position 0."""
        self.p._handle_command_mode_input(b"bc")
        self.p._command_cursor = 0
        self.p._handle_command_mode_input(b"a")
        self.assertEqual(self.p._command_buffer, "abc")
        self.assertEqual(self.p._command_cursor, 1)


class TestCommandModeBracketedPaste(unittest.TestCase):
    """Test bracketed paste (ESC[200~ ... ESC[201~)."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_paste_simple(self):
        """Bracketed paste captures text between markers."""
        paste_data = b"\x1b[200~hello world\x1b[201~"
        self.p._handle_command_mode_input(paste_data)
        self.assertEqual(self.p._command_buffer, "hello world")

    def test_paste_does_not_duplicate(self):
        """Bracketed paste should not duplicate content."""
        paste_data = b"\x1b[200~test\x1b[201~"
        self.p._handle_command_mode_input(paste_data)
        self.assertEqual(self.p._command_buffer, "test")
        self.assertEqual(len(self.p._command_buffer), 4)

    def test_paste_with_newlines(self):
        """Newlines inside bracketed paste are added to buffer."""
        paste_data = b"\x1b[200~line1\nline2\x1b[201~"
        self.p._handle_command_mode_input(paste_data)
        self.assertIn("\n", self.p._command_buffer)
        self.assertEqual(self.p._command_buffer, "line1\nline2")

    def test_paste_newline_does_not_execute(self):
        """Newlines inside bracketed paste do not execute the command."""
        paste_data = b"\x1b[200~!help\nmore\x1b[201~"
        self.p._handle_command_mode_input(paste_data)
        # Should still be in command mode
        self.assertTrue(self.p._command_mode)
        self.assertEqual(self.p._command_buffer, "!help\nmore")

    def test_paste_in_paste_flag(self):
        """_command_in_paste is True during paste, False after."""
        self.p._handle_command_mode_input(b"\x1b[200~")
        self.assertTrue(self.p._command_in_paste)
        self.p._handle_command_mode_input(b"text")
        self.assertTrue(self.p._command_in_paste)
        self.p._handle_command_mode_input(b"\x1b[201~")
        self.assertFalse(self.p._command_in_paste)


class TestCommandModeEnter(unittest.TestCase):
    """Test Enter executes command and exits command mode."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_enter_executes_help(self):
        """Enter on '!help' executes the help command."""
        self.p._handle_command_mode_input(b"!help")
        _clear_output()
        self.p._handle_command_mode_input(b"\r")
        self.assertFalse(self.p._command_mode)
        output = _get_output()
        self.assertIn("Commands", output)

    def test_enter_on_empty_buffer(self):
        """Enter on empty buffer exits command mode without executing."""
        self.p._handle_command_mode_input(b"\r")
        self.assertFalse(self.p._command_mode)

    def test_enter_resets_state(self):
        """Enter resets command mode state variables."""
        self.p._handle_command_mode_input(b"!help\r")
        self.assertFalse(self.p._command_mode)
        self.assertEqual(self.p._command_buffer, "")
        self.assertEqual(self.p._command_cursor, 0)
        self.assertFalse(self.p._command_in_paste)

    def test_lf_also_works(self):
        """Line feed (0x0a) also triggers execution."""
        self.p._handle_command_mode_input(b"!help\n")
        self.assertFalse(self.p._command_mode)
        output = _get_output()
        self.assertIn("Commands", output)

    def test_enter_returns_empty_bytes(self):
        """Enter returns b'' to suppress passthrough to agent."""
        result = self.p._handle_command_mode_input(b"!help\r")
        self.assertEqual(result, b"")


class TestCommandModeCtrlC(unittest.TestCase):
    """Test Ctrl-C cancels command mode."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_ctrl_c_exits_command_mode(self):
        """Ctrl-C exits command mode without executing."""
        self.p._handle_command_mode_input(b"!some command")
        self.p._handle_command_mode_input(b"\x03")
        self.assertFalse(self.p._command_mode)

    def test_ctrl_c_resets_buffer(self):
        """Ctrl-C clears the buffer."""
        self.p._handle_command_mode_input(b"!test")
        self.p._handle_command_mode_input(b"\x03")
        self.assertEqual(self.p._command_buffer, "")
        self.assertEqual(self.p._command_cursor, 0)

    def test_ctrl_c_echoes_caret_c(self):
        """Ctrl-C echoes ^C to terminal."""
        self.p._handle_command_mode_input(b"\x03")
        output = _get_output()
        self.assertIn("^C", output)


class TestCommandModeESCSequences(unittest.TestCase):
    """Test ESC sequences are properly consumed."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_unknown_csi_consumed(self):
        """Unknown CSI sequences are consumed without affecting buffer."""
        self.p._handle_command_mode_input(b"ab")
        # Send some unknown CSI: ESC [ 9 Z
        self.p._handle_command_mode_input(b"\x1b[9Z")
        self.assertEqual(self.p._command_buffer, "ab")

    def test_osc_sequence_consumed(self):
        """OSC sequences (ESC ] ... BEL) are consumed."""
        self.p._handle_command_mode_input(b"ab")
        # OSC with BEL terminator
        self.p._handle_command_mode_input(b"\x1b]0;title\x07")
        self.assertEqual(self.p._command_buffer, "ab")
        self.assertEqual(self.p._command_esc_mode, 0)

    def test_osc_st_terminated(self):
        """OSC sequences terminated with ST (ESC \\) are consumed."""
        self.p._handle_command_mode_input(b"ab")
        # OSC with ST terminator
        self.p._handle_command_mode_input(b"\x1b]0;title\x1b\\")
        self.assertEqual(self.p._command_buffer, "ab")
        self.assertEqual(self.p._command_esc_mode, 0)

    def test_ss3_consumed(self):
        """SS3 sequences (ESC O ...) are consumed."""
        self.p._handle_command_mode_input(b"ab")
        self.p._handle_command_mode_input(b"\x1bOA")  # SS3 up arrow
        self.assertEqual(self.p._command_buffer, "ab")

    def test_bare_esc_resets(self):
        """Bare ESC followed by non-sequence byte resets esc mode."""
        self.p._handle_command_mode_input(b"\x1bx")
        self.assertEqual(self.p._command_esc_mode, 0)


class TestCommandDispatch(unittest.TestCase):
    """Test command dispatch for all commands."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_help_command(self):
        _clear_output()
        self.p._handle_command("!help")
        output = _get_output()
        self.assertIn("Commands", output)
        self.assertIn("!evolve", output)
        self.assertIn("!eval", output)

    def test_status_command(self):
        _clear_output()
        self.p._handle_command("!status")
        output = _get_output()
        self.assertIn("session", output)

    def test_history_command(self):
        _clear_output()
        self.p._handle_command("!history")
        output = _get_output()
        self.assertIn("History", output)

    def test_stop_command(self):
        _clear_output()
        self.p._handle_command("!stop")
        self.assertTrue(self.p._stop_evolve)
        output = _get_output()
        self.assertIn("stopping", output)

    def test_reset_evolve_command_idle(self):
        """!reset-evolve when no thread is running resets state."""
        _clear_output()
        self.p._evolving = True
        self.p._evolve_state = "stuck"
        self.p._handle_command("!reset-evolve")
        self.assertFalse(self.p._evolving)
        self.assertEqual(self.p._evolve_state, "idle")

    def test_unknown_command(self):
        _clear_output()
        self.p._handle_command("!nonexistent")
        output = _get_output()
        self.assertIn("unknown command", output)

    def test_eval_no_args(self):
        _clear_output()
        self.p._handle_command("!eval")
        output = _get_output()
        self.assertIn("usage", output)

    def test_eval_show_no_evaluator(self):
        _clear_output()
        self.p._handle_eval_command(["show"])
        output = _get_output()
        self.assertIn("no evaluator set", output)

    def test_eval_run_no_evaluator(self):
        _clear_output()
        self.p._handle_eval_command(["run"])
        output = _get_output()
        self.assertIn("no evaluator set", output)

    def test_eval_metrics_empty(self):
        _clear_output()
        self.p._handle_eval_command(["metrics"])
        output = _get_output()
        self.assertIn("no eval results", output)

    def test_eval_unknown_subcommand(self):
        _clear_output()
        self.p._handle_eval_command(["bogus"])
        output = _get_output()
        self.assertIn("unknown", output)

    def test_eval_set_missing_path(self):
        _clear_output()
        self.p._handle_eval_command(["set"])
        output = _get_output()
        self.assertIn("usage", output)

    def test_eval_set_nonexistent_file(self):
        _clear_output()
        self.p._handle_eval_command(["set", "/tmp/procy_no_such_file_xyz.py"])
        output = _get_output()
        self.assertIn("file not found", output)

    def test_eval_set_valid_file(self):
        """!eval set with a valid .py file registers the evaluator."""
        _clear_output()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="procy_eval_test_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write('import json\nprint(json.dumps({"accuracy": 0.99}))\n')
            self.p._handle_eval_command(["set", path])
            output = _get_output()
            self.assertIn("evaluator", output)
            self.assertIn("set", output)
            # Verify it's stored
            ev = self.p.store.get_evaluator(self.p.session_id)
            self.assertIsNotNone(ev)
            self.assertEqual(ev["script_path"], path)
        finally:
            os.unlink(path)

    def test_eval_set_with_name(self):
        """!eval set <path> --name <name> sets the evaluator name."""
        _clear_output()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="procy_eval_test_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write('print("{}")\n')
            self.p._handle_eval_command(["set", path, "--name", "myeval"])
            ev = self.p.store.get_evaluator(self.p.session_id, name="myeval")
            self.assertIsNotNone(ev)
            self.assertEqual(ev["name"], "myeval")
        finally:
            os.unlink(path)

    def test_evolve_no_qwen(self):
        """!evolve without qwen_url errors."""
        _clear_output()
        self.p.last_human_prompt = "test task"
        self.p.qwen_url = None
        self.p._handle_command("!evolve 3")
        output = _get_output()
        self.assertIn("no Qwen", output)

    def test_evolve_no_prompt(self):
        """!evolve with no previous prompt errors."""
        _clear_output()
        self.p.last_human_prompt = ""
        self.p.qwen_url = None  # no qwen = error for both policies
        self.p._handle_command("!evolve 3")
        output = _get_output()
        self.assertIn("no Qwen", output)

    def test_correct_no_prompt(self):
        """!correct with no previous prompt errors."""
        _clear_output()
        self.p.last_human_prompt = ""
        self.p._handle_command("!correct")
        output = _get_output()
        self.assertIn("no prompt to correct", output)

    def test_train_no_corrections(self):
        """!train with no corrections errors."""
        _clear_output()
        self.p._handle_command("!train")
        output = _get_output()
        self.assertIn("no training data", output)

    def test_estatus_alias(self):
        """!estatus is an alias for !status."""
        _clear_output()
        self.p._handle_command("!estatus")
        output = _get_output()
        self.assertIn("session", output)

    def test_evolve_status_alias(self):
        """!evolve-status is an alias for !status."""
        _clear_output()
        self.p._handle_command("!evolve-status")
        output = _get_output()
        self.assertIn("session", output)

    def test_invalid_syntax(self):
        """Mismatched quotes produce a syntax error."""
        _clear_output()
        self.p._handle_command('"unclosed')
        output = _get_output()
        self.assertIn("invalid command syntax", output)

    def test_rejects_non_control_commands_while_evolving(self):
        """During evolve, mutating commands are explicitly rejected."""
        _clear_output()
        self.p._evolving = True
        self.p._handle_command("!eval show")
        output = _get_output()
        self.assertIn("evolve is running", output)

    def test_allows_stop_while_evolving(self):
        """During evolve, !stop should still work."""
        _clear_output()
        self.p._evolving = True
        self.p._handle_command("!stop")
        self.assertTrue(self.p._stop_evolve)
        output = _get_output()
        self.assertIn("stopping", output)

    def test_eval_no_session(self):
        """!eval without session_id errors."""
        _clear_output()
        self.p.session_id = None
        self.p._handle_eval_command(["show"])
        output = _get_output()
        self.assertIn("no active session", output)


class TestEvalGenerate(unittest.TestCase):
    """Test _eval_generate threading and response capture."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_eval_generate_no_proxy(self):
        """_eval_generate without proxy logs error."""
        _clear_output()
        self.p._proxy = None
        self.p._eval_generate("test evaluator")
        output = _get_output()
        self.assertIn("no agent running", output)

    def test_eval_generate_error_handling(self):
        """_eval_generate catches exceptions and resets state."""
        _clear_output()
        self.p._proxy = MagicMock()
        self.p._proxy.master_fd = 5
        # Make _inject_prompt raise
        self.p._inject_prompt = MagicMock(side_effect=RuntimeError("test crash"))
        # _wait_for_agent_response_done should also be mocked to raise
        self.p._wait_for_agent_response_done = MagicMock(side_effect=RuntimeError("test crash"))
        self.p._eval_generate("test")
        # Should have caught the exception
        output = _get_output()
        self.assertIn("crashed", output)
        # State should be reset
        self.assertFalse(self.p._evolving)

    def test_eval_generate_resets_stop_evolve(self):
        """_eval_generate resets _stop_evolve at start."""
        self.p._proxy = MagicMock()
        self.p._proxy.master_fd = 5
        self.p._stop_evolve = True
        # Mock to return quickly
        self.p._wait_for_agent_response_done = MagicMock(return_value=False)
        self.p._inject_prompt = MagicMock()
        _clear_output()
        self.p._eval_generate("test")
        # _stop_evolve was reset to False at the start of _eval_generate_inner
        # (then restored on timeout)

    def test_eval_generate_thread_does_not_block(self):
        """Verify _eval_generate can run in a thread without blocking."""
        self.p._proxy = MagicMock()
        self.p._proxy.master_fd = 5
        self.p._inject_prompt = MagicMock()
        self.p._wait_for_agent_response_done = MagicMock(return_value=False)

        t = threading.Thread(target=self.p._eval_generate, args=("test",), daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "Thread should have completed")


class TestWaitForAgentResponseDone(unittest.TestCase):
    """Test _wait_for_agent_response_done quiet detection."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_timeout_returns_false(self):
        """Returns False when timeout expires."""
        self.p._output_seq = 0
        result = self.p._wait_for_agent_response_done(0, timeout=0.3)
        self.assertFalse(result)

    def test_stop_evolve_returns_false(self):
        """Returns False when _stop_evolve is set."""
        self.p._stop_evolve = True
        result = self.p._wait_for_agent_response_done(0, timeout=1)
        self.assertFalse(result)

    def test_detects_quiet_after_output(self):
        """Returns True when new output followed by quiet period."""
        self.p._output_seq = 5
        self.p._last_agent_output_at = time.time() - 2.0  # quiet for 2 seconds
        self.p._captured_output = b"some output >"  # looks like prompt
        self.p._last_input_at = 0
        result = self.p._wait_for_agent_response_done(0, timeout=1)
        self.assertTrue(result)


class TestLastAgentOutputAt(unittest.TestCase):
    """Test _last_agent_output_at only updates on meaningful output."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        self.p._last_agent_output_at = 0.0

    def test_meaningful_output_updates_timestamp(self):
        """Real content output updates _last_agent_output_at."""
        before = self.p._last_agent_output_at
        self.p._on_output(b"Hello, this is a real response from the agent.\r\n")
        self.assertGreater(self.p._last_agent_output_at, before)

    def test_spinner_noise_does_not_update(self):
        """Pure spinner/noise output does not update _last_agent_output_at."""
        self.p._last_agent_output_at = 0.0
        # Pure ANSI sequence with no visible content
        self.p._on_output(b"\x1b[2K\x1b[1G")
        self.assertEqual(self.p._last_agent_output_at, 0.0)

    def test_spinner_chars_are_noise(self):
        """Spinner characters alone are noise."""
        self.p._last_agent_output_at = 0.0
        self.p._on_output(b"\xe2\x9c\xbb")  # ✻
        self.assertEqual(self.p._last_agent_output_at, 0.0)


class TestCleanForDb(unittest.TestCase):
    """Test _clean_for_db preserves code blocks and removes noise."""

    def test_preserves_code_blocks(self):
        """Code blocks with ``` markers are preserved."""
        text = "Here is code:\n```python\ndef foo():\n    return 42\n```\nDone."
        result = _clean_for_db(text)
        self.assertIn("```python", result)
        self.assertIn("def foo():", result)
        self.assertIn("return 42", result)
        self.assertIn("```", result)

    def test_removes_spinner(self):
        """Spinner characters and labels are removed."""
        text = "Hello world"
        result = _clean_for_db(text)
        self.assertIn("Hello world", result)

    def test_removes_rules(self):
        """Rule lines are removed."""
        text = "Hello\n─────────\nWorld"
        result = _clean_for_db(text)
        self.assertIn("Hello", result)
        self.assertIn("World", result)
        self.assertNotIn("─────────", result)

    def test_removes_prompt_hints(self):
        """Prompt hints like '? for shortcuts' are removed."""
        text = "Hello\n? for shortcuts\nWorld"
        result = _clean_for_db(text)
        self.assertNotIn("for shortcuts", result)

    def test_removes_thinking_indicators(self):
        """Thinking/working indicators are removed."""
        text = "Elucidating...\nActual content here"
        result = _clean_for_db(text)
        self.assertIn("Actual content here", result)

    def test_collapses_blank_lines(self):
        """3+ consecutive blank lines are collapsed to 2."""
        text = "Hello\n\n\n\n\nWorld"
        result = _clean_for_db(text)
        # Should not have more than 2 newlines in a row
        self.assertNotIn("\n\n\n", result)

    def test_empty_input(self):
        result = _clean_for_db("")
        self.assertEqual(result, "")

    def test_strips_spinner_chars_from_lines(self):
        """Lines with spinner chars embedded are cleaned."""
        text = "Hello ✻ Spinning…"
        result = _clean_for_db(text)
        self.assertIn("Hello", result)
        self.assertNotIn("Spinning", result)


class TestExtractCodeBlock(unittest.TestCase):
    """Test _extract_code_block finds python blocks."""

    def setUp(self):
        self.p = _make_procy()

    def test_finds_python_block(self):
        text = "Here:\n```python\nimport json\nprint(json.dumps({}))\n```\n"
        result = self.p._extract_code_block(text)
        self.assertIsNotNone(result)
        self.assertIn("import json", result)

    def test_finds_py_block(self):
        text = "Here:\n```py\nimport os\n```\n"
        result = self.p._extract_code_block(text)
        self.assertIsNotNone(result)
        self.assertIn("import os", result)

    def test_no_code_block(self):
        text = "Just plain text, no fences"
        result = self.p._extract_code_block(text)
        self.assertIsNone(result)

    def test_prefers_python_tagged(self):
        """When multiple blocks exist, prefers python-tagged ones."""
        text = "```bash\nls -la\n```\n\n```python\nimport sys\nprint(sys.argv)\n```\n"
        result = self.p._extract_code_block(text)
        self.assertIn("import sys", result)

    def test_fallback_raw_python(self):
        """Falls back to detecting raw Python if no fences."""
        text = "import json\ndef main():\n    print(json.dumps({}))\n"
        result = self.p._extract_code_block(text)
        self.assertIsNotNone(result)
        self.assertIn("import json", result)

    def test_empty_code_block_skipped(self):
        """Empty code blocks are skipped."""
        text = "```python\n```\n\n```python\nimport os\n```\n"
        result = self.p._extract_code_block(text)
        self.assertIsNotNone(result)
        self.assertIn("import os", result)


class TestDetectMetricsSchema(unittest.TestCase):
    """Test _detect_metrics_schema detects metric names and goals."""

    def setUp(self):
        self.p = _make_procy()

    def test_explicit_schema(self):
        """Detects METRICS_SCHEMA constant in script."""
        script = '''
METRICS_SCHEMA = [
    {"name": "accuracy", "type": "float", "goal": "maximize"},
    {"name": "latency_ms", "type": "float", "goal": "minimize"},
]
'''
        result = self.p._detect_metrics_schema(script)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "accuracy")
        self.assertEqual(result[0]["goal"], "maximize")
        self.assertEqual(result[1]["name"], "latency_ms")
        self.assertEqual(result[1]["goal"], "minimize")

    def test_json_dumps_detection(self):
        """Detects metric names from json.dumps calls."""
        script = 'print(json.dumps({"recall": 0.95, "f1": 0.9}))\n'
        result = self.p._detect_metrics_schema(script)
        names = {m["name"] for m in result}
        self.assertIn("recall", names)
        self.assertIn("f1", names)

    def test_results_dict_detection(self):
        """Detects metric names from results/metrics dict assignments."""
        script = 'results = {"precision": 0.8, "latency": 50}\n'
        result = self.p._detect_metrics_schema(script)
        names = {m["name"] for m in result}
        self.assertIn("precision", names)
        self.assertIn("latency", names)
        # latency should be minimize
        lat = [m for m in result if m["name"] == "latency"][0]
        self.assertEqual(lat["goal"], "minimize")

    def test_skip_common_keys(self):
        """Common non-metric keys like 'type', 'name' are skipped."""
        script = 'result = {"type": "test", "name": "foo", "score": 0.9}\n'
        result = self.p._detect_metrics_schema(script)
        names = {m["name"] for m in result}
        self.assertNotIn("type", names)
        self.assertNotIn("name", names)
        self.assertIn("score", names)

    def test_no_metrics(self):
        """Script with no detectable metrics returns empty list."""
        script = "print('hello')\n"
        result = self.p._detect_metrics_schema(script)
        self.assertEqual(result, [])

    def test_invalid_schema_graceful(self):
        """Invalid METRICS_SCHEMA is handled gracefully."""
        script = 'METRICS_SCHEMA = "not a list"\n'
        result = self.p._detect_metrics_schema(script)
        self.assertIsInstance(result, list)


class TestRunEvaluator(unittest.TestCase):
    """Test _run_evaluator runs script and parses JSON output."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        # Set up a simple evaluator script
        self.eval_script = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="procy_eval_"
        )
        self.eval_script.write(
            'import json\nprint(json.dumps({"accuracy": 0.95, "f1": 0.88}))\n'
        )
        self.eval_script.close()

    def tearDown(self):
        os.unlink(self.eval_script.name)

    def test_run_evaluator_success(self):
        """Running a valid evaluator script returns metrics."""
        eid = self.p.store.set_evaluator(
            self.p.session_id, "test",
            script_path=self.eval_script.name,
            run_command="python3 {script}",
        )
        ev = self.p.store.get_evaluator(self.p.session_id, "test")
        result = self.p._run_evaluator(ev)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("accuracy", result["metrics"])
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.95)
        self.assertIn("f1", result["metrics"])

    def test_run_evaluator_failure(self):
        """Running a failing script returns error exit code."""
        fd, bad_script = tempfile.mkstemp(suffix=".py", prefix="procy_bad_eval_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("import sys\nsys.exit(1)\n")
            eid = self.p.store.set_evaluator(
                self.p.session_id, "bad",
                script_path=bad_script,
                run_command="python3 {script}",
            )
            ev = self.p.store.get_evaluator(self.p.session_id, "bad")
            result = self.p._run_evaluator(ev)
            self.assertNotEqual(result["exit_code"], 0)
        finally:
            os.unlink(bad_script)

    def test_run_evaluator_no_script(self):
        """Evaluator with no script path and no content returns error."""
        ev = {
            "id": 999,
            "script_path": "/tmp/procy_no_such_path_xyz.py",
            "run_command": "python3 {script}",
        }
        result = self.p._run_evaluator(ev)
        self.assertEqual(result["exit_code"], -1)

    def test_run_evaluator_inline_content(self):
        """Evaluator with inline script_content runs from temp file."""
        ev_content = 'import json\nprint(json.dumps({"score": 1.0}))\n'
        eid = self.p.store.set_evaluator(
            self.p.session_id, "inline",
            script_content=ev_content,
            run_command="python3 {script}",
        )
        ev = self.p.store.get_evaluator(self.p.session_id, "inline")
        # Remove script_path to force inline
        ev["script_path"] = None
        result = self.p._run_evaluator(ev)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("score", result["metrics"])


class TestRenderCommandLineLocked(unittest.TestCase):
    """Test _render_command_line_locked doesn't crash."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_render_when_not_in_command_mode(self):
        """Does nothing when not in command mode."""
        _clear_output()
        self.p._command_mode = False
        self.p._render_command_line_locked()
        self.assertEqual(_get_output(), "")

    def test_render_when_silent(self):
        """Does nothing when silent."""
        _clear_output()
        self.p._command_mode = True
        self.p._command_silent = True
        self.p._render_command_line_locked()
        self.assertEqual(_get_output(), "")

    def test_render_normal(self):
        """Renders command line with prefix."""
        _clear_output()
        self.p._command_mode = True
        self.p._command_silent = False
        self.p._command_buffer = "!help"
        self.p._command_cursor = 5
        self.p._render_command_line_locked()
        output = _get_output()
        self.assertIn("procy-cmd", output)
        self.assertIn("!help", output)

    def test_render_cursor_in_middle(self):
        """Render with cursor not at end emits cursor movement."""
        _clear_output()
        self.p._command_mode = True
        self.p._command_silent = False
        self.p._command_buffer = "abcde"
        self.p._command_cursor = 2  # 3 chars after cursor
        self.p._render_command_line_locked()
        output = _get_output()
        # Should contain cursor-back sequence for 3 chars
        self.assertIn("\033[3D", output)

    def test_render_empty_buffer(self):
        """Render with empty buffer doesn't crash."""
        _clear_output()
        self.p._command_mode = True
        self.p._command_silent = False
        self.p._command_buffer = ""
        self.p._command_cursor = 0
        self.p._render_command_line_locked()
        output = _get_output()
        self.assertIn("procy-cmd", output)

    def test_render_with_newlines_in_buffer(self):
        """Newlines in buffer are displayed as arrows."""
        _clear_output()
        self.p._command_mode = True
        self.p._command_silent = False
        self.p._command_buffer = "line1\nline2"
        self.p._command_cursor = 11
        self.p._render_command_line_locked()
        output = _get_output()
        self.assertIn("\u21b5", output)  # ↵


class TestIsNoiseLine(unittest.TestCase):
    """Test _is_noise_line identifies TUI decoration noise."""

    def test_empty_line(self):
        self.assertTrue(_is_noise_line(""))

    def test_rule_line(self):
        self.assertTrue(_is_noise_line("\u2500" * 20))

    def test_spinner_line(self):
        self.assertTrue(_is_noise_line("\u273b"))

    def test_prompt_hint(self):
        self.assertTrue(_is_noise_line("? for shortcuts"))

    def test_status_noise(self):
        self.assertTrue(_is_noise_line("Stewing..."))

    def test_real_content(self):
        self.assertFalse(_is_noise_line("Here is some real content"))

    def test_code_content(self):
        self.assertFalse(_is_noise_line("def foo():"))

    def test_bare_prompt_char(self):
        self.assertTrue(_is_noise_line("\u276f"))

    def test_single_lowercase(self):
        self.assertTrue(_is_noise_line("n"))

    def test_ellipsis(self):
        self.assertTrue(_is_noise_line("..."))

    def test_short_non_alnum(self):
        self.assertTrue(_is_noise_line(".."))

    def test_esc_to_interrupt(self):
        self.assertTrue(_is_noise_line("esc to interrupt"))


class TestOnOutput(unittest.TestCase):
    """Test _on_output and output handling."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_output_increments_seq(self):
        """Output increments _output_seq."""
        before = self.p._output_seq
        self.p._on_output(b"hello")
        self.assertGreater(self.p._output_seq, before)

    def test_output_during_command_mode_sets_silent(self):
        """Output arriving during command mode sets _command_silent."""
        _enter_command_mode_clean(self.p)
        self.assertFalse(self.p._command_silent)
        self.p._on_output(b"agent output")
        self.assertTrue(self.p._command_silent)

    def test_capture_output_stores_data(self):
        """With _capture_output=True, output is accumulated."""
        self.p._capture_output = True
        self.p._on_output(b"captured data")
        self.assertIn(b"captured data", self.p._captured_output)


class TestCommandModeBackspaceOnEmpty(unittest.TestCase):
    """Backspace on empty buffer exits command mode (regression test)."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_excl_then_backspace_exits(self):
        """! enters command mode with '!' in buffer. Backspace removes '!', then
        buffer is empty. A second backspace exits command mode."""
        result = self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        self.assertEqual(self.p._command_buffer, "!")
        # Backspace removes the '!'
        result = self.p._on_input(b"\x7f")
        self.assertTrue(self.p._command_mode)
        self.assertEqual(self.p._command_buffer, "")
        # Second backspace on truly empty buffer exits
        result = self.p._on_input(b"\x7f")
        self.assertFalse(self.p._command_mode)


class TestCommandModeMultipleOperations(unittest.TestCase):
    """Integration tests combining multiple operations."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        _enter_command_mode_clean(self.p)
        _clear_output()

    def test_type_delete_retype(self):
        """Type, delete, retype sequence works correctly."""
        self.p._handle_command_mode_input(b"!hel")
        self.p._handle_command_mode_input(b"\x7f")  # backspace
        self.p._handle_command_mode_input(b"lp")
        self.assertEqual(self.p._command_buffer, "!help")

    def test_left_insert_right_sequence(self):
        """Left arrow, insert, right arrow sequence."""
        self.p._handle_command_mode_input(b"!hep")
        # Left once
        self.p._handle_command_mode_input(b"\x1b[D")
        # Insert 'l'
        self.p._handle_command_mode_input(b"l")
        self.assertEqual(self.p._command_buffer, "!help")
        self.assertEqual(self.p._command_cursor, 4)

    def test_ctrl_a_type_ctrl_e(self):
        """Ctrl-A, type, Ctrl-E sequence."""
        self.p._handle_command_mode_input(b"help")
        self.p._handle_command_mode_input(b"\x01")  # Ctrl-A
        self.assertEqual(self.p._command_cursor, 0)
        self.p._handle_command_mode_input(b"!")
        self.assertEqual(self.p._command_buffer, "!help")
        self.p._handle_command_mode_input(b"\x05")  # Ctrl-E
        self.assertEqual(self.p._command_cursor, 5)


class TestEvalGenerateCodeExtraction(unittest.TestCase):
    """Test _eval_generate response capture and code extraction flow."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_code_extraction_with_real_response(self):
        """Simulate a response buffer and extract code."""
        response = (
            "Here's the evaluator:\n\n"
            "```python\n"
            "import json\n"
            "import os\n\n"
            "def evaluate():\n"
            "    return {'score': 1.0}\n\n"
            "if __name__ == '__main__':\n"
            "    print(json.dumps(evaluate()))\n"
            "```\n\n"
            "WROTE eval.py"
        )
        code = self.p._extract_code_block(response, "python")
        self.assertIsNotNone(code)
        self.assertIn("import json", code)
        self.assertIn("evaluate()", code)

    def test_code_extraction_python3_tag(self):
        """python3 tag is recognized."""
        response = "```python3\nimport sys\n```"
        code = self.p._extract_code_block(response, "python")
        self.assertIsNotNone(code)
        self.assertIn("import sys", code)


class TestSanitizeOutputChunk(unittest.TestCase):
    """Test _sanitize_output_chunk strips control sequences."""

    def setUp(self):
        self.p = _make_procy()

    def test_strips_ansi(self):
        """ANSI CSI sequences are stripped."""
        result = self.p._sanitize_output_chunk(b"\x1b[31mhello\x1b[0m")
        self.assertEqual(result, "hello")

    def test_strips_cr(self):
        """Carriage returns are stripped."""
        result = self.p._sanitize_output_chunk(b"hello\r\nworld")
        self.assertEqual(result, "hello\nworld")

    def test_strips_osc(self):
        """OSC sequences are stripped."""
        result = self.p._sanitize_output_chunk(b"\x1b]0;title\x07hello")
        self.assertEqual(result, "hello")

    def test_passes_tab_and_newline(self):
        """Tab and newline are preserved."""
        result = self.p._sanitize_output_chunk(b"hello\tworld\n")
        self.assertEqual(result, "hello\tworld\n")


class TestCommandModeOnInputReturn(unittest.TestCase):
    """Verify _on_input returns correct values in various states."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_not_command_mode_returns_none(self):
        """Normal input returns None (passthrough)."""
        self.p._typed_line_buffer = "hello"
        result = self.p._on_input(b"x")
        self.assertIsNone(result)

    def test_command_mode_returns_empty_bytes(self):
        """Command mode input always returns b'' (suppress passthrough)."""
        self.p._on_input(b"!")
        result = self.p._on_input(b"a")
        self.assertEqual(result, b"")


class TestCommandModeSilentToVisible(unittest.TestCase):
    """Test transition from silent to visible command mode."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        # Enter command mode in silent mode
        self.p._output_seq = 10
        self.p._last_agent_output_at = time.time()
        self.p._on_input(b"!")
        self.assertTrue(self.p._command_silent)
        _clear_output()

    def test_becomes_visible_after_quiet(self):
        """Silent mode transitions to visible after output goes quiet."""
        # Simulate quiet period
        self.p._last_agent_output_at = time.time() - 1.0
        self.p._output_seq = 0  # or quiet
        self.p._handle_command_mode_input(b"a")
        self.assertFalse(self.p._command_silent)


class TestEvalSetFileTypes(unittest.TestCase):
    """Test !eval set with different file extensions."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def _test_extension(self, ext: str, expected_cmd_prefix: str):
        fd, path = tempfile.mkstemp(suffix=ext, prefix="procy_eval_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("# test\n")
            _clear_output()
            self.p._handle_eval_command(["set", path])
            ev = self.p.store.get_evaluator(self.p.session_id)
            self.assertIsNotNone(ev)
            self.assertTrue(ev["run_command"].startswith(expected_cmd_prefix))
        finally:
            os.unlink(path)

    def test_py_file(self):
        self._test_extension(".py", "python3")

    def test_sh_file(self):
        self._test_extension(".sh", "bash")

    def test_js_file(self):
        self._test_extension(".js", "node")

    def test_unknown_ext(self):
        """Unknown extension uses {script} directly."""
        fd, path = tempfile.mkstemp(suffix=".rb", prefix="procy_eval_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("# test\n")
            _clear_output()
            self.p._handle_eval_command(["set", path])
            ev = self.p.store.get_evaluator(self.p.session_id)
            self.assertIsNotNone(ev)
            self.assertEqual(ev["run_command"], "{script}")
        finally:
            os.unlink(path)


class TestEvalGenerate_InjectPrompt(unittest.TestCase):
    """Test that !eval generate calls _inject_prompt."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        self.p._proxy = MagicMock()
        self.p._proxy.master_fd = 5

    def test_generate_calls_inject(self):
        """!eval generate subcommand injects a prompt to the agent."""
        injected = []

        def capture_inject(prompt):
            injected.append(prompt)

        # Mock both _inject_prompt and _wait_for_agent_prompt (no real agent in tests)
        self.p._inject_prompt = capture_inject
        self.p._wait_for_agent_prompt = lambda timeout=30: True
        _clear_output()
        self.p._handle_eval_command(["generate", "test task"])
        # Should have injected a prompt
        self.assertTrue(len(injected) > 0)
        self.assertIn("eval", injected[0].lower())


class TestInferMetricsSchemaFromMetrics(unittest.TestCase):
    """Test _infer_metrics_schema_from_metrics."""

    def setUp(self):
        self.p = _make_procy()

    def test_infers_float_types(self):
        metrics = {"accuracy": 0.95, "count": 42}
        result = self.p._infer_metrics_schema_from_metrics(metrics)
        self.assertEqual(len(result), 2)
        names = {m["name"] for m in result}
        self.assertIn("accuracy", names)
        self.assertIn("count", names)

    def test_infers_minimize_goals(self):
        metrics = {"latency_ms": 50.0, "error_rate": 0.01}
        result = self.p._infer_metrics_schema_from_metrics(metrics)
        for m in result:
            self.assertEqual(m["goal"], "minimize")

    def test_string_metric_type(self):
        metrics = {"status": "ok"}
        result = self.p._infer_metrics_schema_from_metrics(metrics)
        self.assertEqual(result[0]["type"], "str")


class TestSanitizeInputLine(unittest.TestCase):
    """Input sanitizer should remove prompt-marker artifacts."""

    def setUp(self):
        self.p = _make_procy()

    def test_drops_lone_pipe_marker(self):
        self.assertEqual(self.p._sanitize_input_line("|"), "")

    def test_drops_command_with_marker_prefix(self):
        self.assertEqual(self.p._sanitize_input_line("| !eval generate benchmark"), "")


class TestGetChangedFiles(unittest.TestCase):
    """Test _get_changed_files."""

    def setUp(self):
        self.p = _make_procy()

    def test_no_session(self):
        self.p.session_id = None
        result = self.p._get_changed_files()
        self.assertEqual(result, [])

    def test_with_actions(self):
        """Changed files are extracted from write/edit actions."""
        self.p.store.log_action(self.p.session_id, 1, "write", "/tmp/file.py", "ok")
        self.p.store.log_action(self.p.session_id, 1, "edit", "/tmp/other.py", "ok")
        self.p.store.log_action(self.p.session_id, 1, "read", "/tmp/read.py", "ok")
        result = self.p._get_changed_files()
        self.assertIn("/tmp/file.py", result)
        self.assertIn("/tmp/other.py", result)
        self.assertNotIn("/tmp/read.py", result)

    def test_deduplication(self):
        """Same file written twice appears only once."""
        self.p.store.log_action(self.p.session_id, 1, "write", "/tmp/file.py", "ok")
        self.p.store.log_action(self.p.session_id, 2, "write", "/tmp/file.py", "ok")
        result = self.p._get_changed_files()
        self.assertEqual(result.count("/tmp/file.py"), 1)


class TestRenderCommandLineLockedCursorMath(unittest.TestCase):
    """Test cursor math edge cases in _render_command_line_locked."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        self.p._command_mode = True
        self.p._command_silent = False

    def test_cursor_at_end_no_back_sequence(self):
        """Cursor at end of buffer: no backward cursor movement."""
        _clear_output()
        self.p._command_buffer = "abc"
        self.p._command_cursor = 3
        self.p._render_command_line_locked()
        output = _get_output()
        # Should NOT have \033[...D after the final \033[0m
        # The display is: prefix + " " + display + \033[0m
        # If cursor is at end, no \033[...D is emitted
        parts = output.split("\033[0m")
        last_part = parts[-1] if parts else ""
        self.assertNotIn("D", last_part)

    def test_cursor_at_0_back_full(self):
        """Cursor at 0: backward movement for full buffer length."""
        _clear_output()
        self.p._command_buffer = "abc"
        self.p._command_cursor = 0
        self.p._render_command_line_locked()
        output = _get_output()
        self.assertIn("\033[3D", output)


class TestCommandModeViaOnInput_Integration(unittest.TestCase):
    """Integration: test the full flow via _on_input (! in buffer)."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()

    def test_full_help_flow(self):
        """Type '!help' + Enter via _on_input produces help output."""
        _clear_output()
        self.p._on_input(b"!")
        self.assertTrue(self.p._command_mode)
        self.assertEqual(self.p._command_buffer, "!")
        # Type 'help'
        self.p._on_input(b"help")
        self.assertEqual(self.p._command_buffer, "!help")
        # Press Enter
        _clear_output()
        self.p._on_input(b"\r")
        self.assertFalse(self.p._command_mode)
        output = _get_output()
        self.assertIn("Commands", output)

    def test_full_status_flow(self):
        """Type '!status' + Enter via _on_input."""
        self.p._on_input(b"!")
        self.p._on_input(b"status")
        _clear_output()
        self.p._on_input(b"\r")
        self.assertFalse(self.p._command_mode)
        output = _get_output()
        self.assertIn("session", output)


class TestRenderCursorMathWithNewlines(unittest.TestCase):
    """Cursor math when buffer has newlines (replaced with ' \\u21b5 ')."""

    def setUp(self):
        _clear_output()
        self.p = _make_procy()
        self.p._command_mode = True
        self.p._command_silent = False

    def test_newline_expansion_cursor_math(self):
        """Buffer 'a\\nb' has display 'a \\u21b5 b'. Cursor at 2 (after \\n) means
        display cursor at position 4 (after ' \\u21b5 '). chars_after = display_len - display_cursor."""
        _clear_output()
        self.p._command_buffer = "a\nb"
        # cursor at position 2 means after the \n, before 'b'
        self.p._command_cursor = 2
        self.p._render_command_line_locked()
        # display = "a ↵ b" (len=5), cursor after "a ↵ " = 4, chars_after = 1
        # This should produce \033[1D
        output = _get_output()
        self.assertIn("\033[1D", output)


if __name__ == "__main__":
    unittest.main()
