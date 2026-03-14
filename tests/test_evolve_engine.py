"""Tests for the OpenEvolve-style evolution engine."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from procy.store import TraceStore
from procy.evolve_engine import EvolveEngine
from procy import evolve_prompts as prompts


class TestEvolvePrompts(unittest.TestCase):
    """Test prompt template formatting."""

    def test_format_metrics(self):
        m = {"recall": 0.95, "qps": 1200}
        out = prompts.format_metrics(m)
        self.assertIn("recall", out)
        self.assertIn("0.95", out)

    def test_format_metrics_empty(self):
        out = prompts.format_metrics({})
        self.assertIn("no metrics", out)

    def test_format_history(self):
        history = [
            {"iteration": 1, "fitness_score": 0.5, "changes_description": "baseline"},
            {"iteration": 2, "fitness_score": 0.7, "changes_description": "improved recall"},
        ]
        out = prompts.format_history(history)
        self.assertIn("Iteration 1", out)
        self.assertIn("Iteration 2", out)
        self.assertIn("+0.2000", out)

    def test_format_history_empty(self):
        out = prompts.format_history([])
        self.assertIn("no history", out)

    def test_build_prompt(self):
        parent = {"code": "def foo(): pass", "fitness_score": 0.5, "metrics": {"recall": 0.5}}
        sys_msg, user_msg = prompts.build_prompt(parent, [], [], [])
        self.assertIn("expert software developer", sys_msg)
        self.assertIn("def foo(): pass", user_msg)
        self.assertIn("0.5000", user_msg)

    def test_build_prompt_with_top_programs(self):
        parent = {"code": "pass", "fitness_score": 0.5, "metrics": {}}
        top = [{"code": "def best(): return 1", "fitness_score": 0.9, "metrics": {"recall": 0.9}}]
        _, user_msg = prompts.build_prompt(parent, top, [], [])
        self.assertIn("#1", user_msg)
        self.assertIn("0.9000", user_msg)

    def test_build_prompt_with_inspirations(self):
        parent = {"code": "pass", "fitness_score": 0.5, "metrics": {}}
        diverse = [{"code": "import random", "fitness_score": 0.3, "metrics": {}, "island_id": 2}]
        _, user_msg = prompts.build_prompt(parent, [], diverse, [])
        self.assertIn("Inspiration", user_msg)
        self.assertIn("island 2", user_msg)


class TestStorePopulation(unittest.TestCase):
    """Test population-related DB operations."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = TraceStore(self.tmp.name)
        self.session_id = self.store.new_session(goal="test evolve")

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_add_and_get_program(self):
        pid = self.store.add_program(
            self.session_id, "def foo(): pass",
            metrics={"recall": 0.5}, fitness_score=0.5,
        )
        prog = self.store.get_program(pid)
        self.assertIsNotNone(prog)
        self.assertEqual(prog["code"], "def foo(): pass")
        self.assertEqual(prog["fitness_score"], 0.5)
        self.assertEqual(prog["metrics"], {"recall": 0.5})

    def test_get_top_programs(self):
        for i in range(5):
            self.store.add_program(
                self.session_id, f"code_{i}",
                fitness_score=i * 0.1, island_id=0,
            )
        top = self.store.get_top_programs(self.session_id, 3)
        self.assertEqual(len(top), 3)
        self.assertGreaterEqual(top[0]["fitness_score"], top[1]["fitness_score"])

    def test_get_top_programs_by_island(self):
        self.store.add_program(self.session_id, "a", fitness_score=0.9, island_id=0)
        self.store.add_program(self.session_id, "b", fitness_score=0.8, island_id=1)
        top = self.store.get_top_programs(self.session_id, 5, island_id=0)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0]["code"], "a")

    def test_get_diverse_programs(self):
        pids = []
        for i in range(5):
            pid = self.store.add_program(
                self.session_id, f"code_{i}", fitness_score=0.1,
            )
            pids.append(pid)
        diverse = self.store.get_diverse_programs(self.session_id, 2, exclude_ids=[pids[0]])
        self.assertEqual(len(diverse), 2)
        self.assertTrue(all(p["id"] != pids[0] for p in diverse))

    def test_count_programs(self):
        self.store.add_program(self.session_id, "a", island_id=0)
        self.store.add_program(self.session_id, "b", island_id=0)
        self.store.add_program(self.session_id, "c", island_id=1)
        self.assertEqual(self.store.count_programs(self.session_id), 3)
        self.assertEqual(self.store.count_programs(self.session_id, island_id=0), 2)
        self.assertEqual(self.store.count_programs(self.session_id, island_id=1), 1)

    def test_get_best_program(self):
        self.store.add_program(self.session_id, "low", fitness_score=0.1)
        self.store.add_program(self.session_id, "high", fitness_score=0.9)
        best = self.store.get_best_program(self.session_id)
        self.assertEqual(best["code"], "high")

    def test_island_config(self):
        self.store.set_island_config(self.session_id, num_islands=3, exploration_ratio=0.3)
        cfg = self.store.get_island_config(self.session_id)
        self.assertEqual(cfg["num_islands"], 3)
        self.assertEqual(cfg["exploration_ratio"], 0.3)

    def test_island_config_update(self):
        self.store.set_island_config(self.session_id, num_islands=5)
        self.store.set_island_config(self.session_id, num_islands=3)
        cfg = self.store.get_island_config(self.session_id)
        self.assertEqual(cfg["num_islands"], 3)

    def test_update_program_metrics(self):
        pid = self.store.add_program(self.session_id, "code", fitness_score=0.1)
        self.store.update_program_metrics(pid, {"recall": 0.9}, 0.9, [3, 7])
        prog = self.store.get_program(pid)
        self.assertEqual(prog["fitness_score"], 0.9)
        self.assertEqual(prog["feature_coords"], [3, 7])

    def test_get_recent_programs(self):
        import time
        for i in range(5):
            self.store.add_program(self.session_id, f"code_{i}", fitness_score=i * 0.1)
            time.sleep(0.01)
        recent = self.store.get_recent_programs(self.session_id, 3)
        self.assertEqual(len(recent), 3)
        # Most recent first
        self.assertEqual(recent[0]["code"], "code_4")


class TestEvolveEngine(unittest.TestCase):
    """Test the evolution engine."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = TraceStore(self.tmp.name)
        self.session_id = self.store.new_session(goal="test")
        self.engine = EvolveEngine(
            self.store, self.session_id,
            num_islands=3,
        )

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_seed_population(self):
        self.engine.seed_population("def baseline(): pass", {"recall": 0.1}, 0.1)
        total = self.store.count_programs(self.session_id)
        self.assertEqual(total, 3)  # one per island
        for i in range(3):
            progs = self.store.get_island_programs(self.session_id, i)
            self.assertEqual(len(progs), 1)
            self.assertEqual(progs[0]["code"], "def baseline(): pass")

    def test_sample_parent(self):
        self.engine.seed_population("code", {"r": 0.5}, 0.5)
        parent = self.engine.sample_parent()
        self.assertIsNotNone(parent)
        self.assertEqual(parent["code"], "code")

    def test_sample_parent_empty(self):
        parent = self.engine.sample_parent()
        self.assertIsNone(parent)

    def test_sample_inspirations(self):
        self.engine.seed_population("seed", {"r": 0.5}, 0.5)
        # Add more programs
        for i in range(5):
            self.store.add_program(
                self.session_id, f"prog_{i}",
                fitness_score=i * 0.1, island_id=0,
            )
        parent = self.engine.sample_parent()
        top, diverse = self.engine.sample_inspirations(parent["id"])
        self.assertIsInstance(top, list)
        self.assertIsInstance(diverse, list)

    def test_parse_code_fenced(self):
        response = "Here's the code:\n```python\ndef foo():\n    return 42\n```\nDone."
        code = self.engine.parse_code(response)
        self.assertEqual(code, "def foo():\n    return 42")

    def test_parse_code_no_fence(self):
        code = self.engine.parse_code("no code here")
        self.assertIsNone(code)

    def test_evaluate_no_evaluator(self):
        metrics, fitness = self.engine.evaluate("line1\nline2\nline3")
        self.assertEqual(metrics["lines"], 3)
        self.assertEqual(fitness, 0.0)

    def test_evaluate_with_fn(self):
        def eval_fn(code):
            return {"recall": 0.8, "qps": 100}
        metrics, fitness = self.engine.evaluate("code", eval_fn)
        self.assertEqual(metrics["recall"], 0.8)
        self.assertEqual(fitness, 0.8)  # first metric

    def test_feature_coords(self):
        coords = self.engine.calculate_feature_coords("x" * 500, {"recall": 0.5})
        self.assertEqual(len(coords), 2)
        self.assertTrue(0 <= coords[0] < 10)
        self.assertTrue(0 <= coords[1] < 10)

    def test_add_to_population(self):
        seed_id = self.engine.seed_population("seed", {"r": 0.1}, 0.1)
        child_id = self.engine.add_to_population(
            "improved_code", parent_id=seed_id,
            metrics={"r": 0.5}, fitness_score=0.5,
        )
        child = self.store.get_program(child_id)
        self.assertEqual(child["code"], "improved_code")
        self.assertEqual(child["generation"], 1)

    def test_maybe_migrate(self):
        self.engine.seed_population("code", {"r": 0.5}, 0.5)
        self.engine.iteration = 50
        self.engine.maybe_migrate()
        # Check that migration created new programs
        total = self.store.count_programs(self.session_id)
        self.assertGreater(total, 3)

    def test_status(self):
        self.engine.seed_population("code", {"r": 0.5}, 0.5)
        st = self.engine.status()
        self.assertEqual(st["total_programs"], 3)
        self.assertEqual(st["best_fitness"], 0.5)

    def test_stop(self):
        self.engine.seed_population("code", {}, 0)
        self.engine.stop()
        results = self.engine.run(10)
        self.assertEqual(len(results), 0)

    def test_run_iteration_no_llm(self):
        """Without LLM URL, iteration should return error."""
        self.engine.seed_population("code", {"r": 0.5}, 0.5)
        result = self.engine.run_iteration()
        self.assertIn("error", result)
        self.assertIn("LLM call failed", result["error"])

    def test_full_iteration_with_mock_llm(self):
        """Test a full iteration with a mocked LLM that returns code."""
        self.engine.seed_population("def baseline(): return 0", {"recall": 0.1}, 0.1)

        # Mock call_llm to return a code block
        original_call = self.engine.call_llm
        self.engine.call_llm = lambda sys, usr: "```python\ndef improved(): return 1\n```"

        def eval_fn(code):
            return {"recall": 0.5}

        result = self.engine.run_iteration(evaluator_fn=eval_fn)
        self.engine.call_llm = original_call

        self.assertNotIn("error", result)
        self.assertEqual(result["fitness"], 0.5)
        self.assertTrue(result["improved"])
        self.assertEqual(result["iteration"], 1)

    def test_run_multiple_iterations_mock(self):
        """Run 3 iterations with mock LLM."""
        self.engine.seed_population("def v0(): pass", {"recall": 0.1}, 0.1)

        call_count = [0]
        def mock_llm(sys, usr):
            call_count[0] += 1
            return f"```python\ndef v{call_count[0]}(): pass\n```"

        self.engine.call_llm = mock_llm

        def eval_fn(code):
            return {"recall": 0.1 + call_count[0] * 0.1}

        results = self.engine.run(3, evaluator_fn=eval_fn)
        self.assertEqual(len(results), 3)
        # Fitness should increase
        self.assertGreater(results[-1]["fitness"], results[0]["fitness"])

    def test_callback(self):
        """Test that callback is called each iteration."""
        self.engine.seed_population("code", {"r": 0.1}, 0.1)
        self.engine.call_llm = lambda s, u: "```python\npass\n```"

        collected = []
        results = self.engine.run(2, callback=collected.append)
        self.assertEqual(len(collected), 2)

    def test_target_score(self):
        """Test early stopping on target score."""
        self.engine.seed_population("code", {"r": 0.1}, 0.1)

        call_count = [0]
        def mock_llm(sys, usr):
            call_count[0] += 1
            return "```python\npass\n```"

        self.engine.call_llm = mock_llm

        def eval_fn(code):
            return {"recall": 0.5 + call_count[0] * 0.2}

        results = self.engine.run(100, evaluator_fn=eval_fn, target_score=0.9)
        self.assertLess(len(results), 100)
        self.assertGreaterEqual(results[-1]["fitness"], 0.9)


if __name__ == "__main__":
    unittest.main()
