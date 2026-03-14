"""OpenEvolve-style evolution engine with MAP-Elites population and island model.

Two modes:
  - Headless: calls LLM directly (Qwen or any OpenAI-compatible endpoint)
  - Interactive: builds prompt, caller injects into Claude via PTY

The only method that changes when swapping fixed→proxy policy is build_prompt().
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from typing import Callable

from . import evolve_prompts as prompts
from .store import TraceStore


class EvolveEngine:
    """Core evolution engine — population management, parent selection, MAP-Elites."""

    def __init__(
        self,
        store: TraceStore,
        session_id: str,
        llm_url: str | None = None,
        llm_model: str = "qwen35",
        language: str = "python",
        num_islands: int = 5,
        n_top: int = 3,
        n_diverse: int = 2,
    ):
        self.store = store
        self.session_id = session_id
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.language = language
        self.num_islands = num_islands
        self.n_top = n_top
        self.n_diverse = n_diverse
        self.iteration = 0
        self.current_island = 0
        self.stopped = False

        # Ensure island config exists
        cfg = store.get_island_config(session_id)
        if not cfg:
            store.set_island_config(session_id, num_islands=num_islands)
        self._config = store.get_island_config(session_id)

    @property
    def config(self) -> dict:
        return self._config or {}

    # ── Seed ─────────────────────────────────────────────────────────────

    def seed_population(
        self,
        commit_hash: str | None = None,
        metrics: dict | None = None,
        fitness_score: float = 0,
    ) -> str:
        """Add the initial commit to all islands as the seed."""
        seed_id = None
        for island in range(self.num_islands):
            pid = self.store.add_program(
                self.session_id,
                commit_hash=commit_hash,
                island_id=island,
                iteration=0,
                metrics=metrics,
                fitness_score=fitness_score,
                changes_description="initial seed",
            )
            if seed_id is None:
                seed_id = pid
        return seed_id

    # ── Parent Selection (3-strategy) ────────────────────────────────────

    def sample_parent(self) -> dict | None:
        """Select a parent using exploration/exploitation/weighted strategy."""
        cfg = self.config
        exploration_ratio = cfg.get("exploration_ratio", 0.2)
        exploitation_ratio = cfg.get("exploitation_ratio", 0.7)

        r = random.random()

        if r < exploration_ratio:
            # Random from current island
            programs = self.store.get_island_programs(self.session_id, self.current_island)
            if programs:
                return random.choice(programs)

        elif r < exploration_ratio + exploitation_ratio:
            # Top programs globally (exploitation)
            top = self.store.get_top_programs(self.session_id, 5)
            if top:
                return random.choice(top)

        # Fitness-weighted random
        all_progs = self.store.get_top_programs(self.session_id, 50)
        if not all_progs:
            return None
        weights = [max(p.get("fitness_score", 0), 0.001) for p in all_progs]
        return random.choices(all_progs, weights=weights, k=1)[0]

    def sample_inspirations(self, parent_id: str) -> tuple[list[dict], list[dict]]:
        """Get top programs + diverse inspiration programs."""
        top = self.store.get_top_programs(
            self.session_id, self.n_top, island_id=self.current_island,
        )
        exclude = [parent_id] + [p["id"] for p in top]
        diverse = self.store.get_diverse_programs(
            self.session_id, self.n_diverse, exclude_ids=exclude,
        )
        return top, diverse

    # ── Prompt Building (fixed policy — swap point for Qwen) ─────────

    def build_prompt(self, eval_command: str = "python3 eval.py",
                     focus: str = "") -> tuple[str, str]:
        """Build the evolution prompt. THIS IS THE METHOD QWEN REPLACES."""
        best = self.store.get_best_program(self.session_id)
        best_fitness = best.get("fitness_score", 0) if best else 0
        return prompts.build_prompt(
            best_fitness=best_fitness,
            db_path=self.store.db_path,
            session_id=self.session_id,
            eval_command=eval_command,
            focus=focus,
        )

    # ── LLM Call ─────────────────────────────────────────────────────────

    def call_llm(self, system_msg: str, user_msg: str) -> str | None:
        """Call the LLM endpoint. Returns response text or None."""
        if not self.llm_url:
            return None
        payload = json.dumps({
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 4000,
            "temperature": 0.8,
        }).encode()
        try:
            url = f"{self.llm_url.rstrip('/')}/v1/chat/completions"
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None

    # ── Response Parsing ─────────────────────────────────────────────────

    def parse_code(self, response: str) -> str | None:
        """Extract code from LLM response (fenced code block)."""
        if not response:
            return None
        # Try fenced code block
        pattern = r"```(?:\w+)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the longest match (likely the main code)
            return max(matches, key=len).strip()
        # If no fenced block, try the whole response as code
        stripped = response.strip()
        if stripped and not stripped.startswith("#") and "def " in stripped or "class " in stripped:
            return stripped
        return None

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self, code: str,
        evaluator_fn: Callable[[str], dict] | None = None,
    ) -> tuple[dict, float]:
        """Evaluate a program. Returns (metrics_dict, fitness_score).

        evaluator_fn: takes code string, returns {"metric_name": value, ...}
        The first metric is treated as the primary fitness score.
        """
        if evaluator_fn is None:
            # No evaluator — use code length as a dummy metric
            return {"lines": code.count("\n") + 1}, 0.0

        metrics = evaluator_fn(code)
        if not metrics:
            return {}, 0.0

        # Primary fitness = first metric value
        first_key = next(iter(metrics))
        fitness = float(metrics[first_key])
        return metrics, fitness

    # ── Feature Coordinates (simple heuristics) ──────────────────────────

    def calculate_feature_coords(self, metrics: dict) -> list[int]:
        """Compute MAP-Elites bin indices from metrics.

        Uses first two metrics as the two dimensions.
        """
        bins = self.config.get("feature_bins", 10)
        values = list(metrics.values()) if metrics else []

        def to_bin(val):
            norm = max(0, min(1, float(val)))
            return min(int(norm * bins), bins - 1)

        bin_x = to_bin(values[0]) if len(values) > 0 else 0
        bin_y = to_bin(values[1]) if len(values) > 1 else 0
        return [bin_x, bin_y]

    # ── Population Management ────────────────────────────────────────────

    def add_to_population(
        self, *,
        commit_hash: str | None = None,
        parent_id: str | None = None,
        metrics: dict, fitness_score: float,
        changes_description: str = "",
    ) -> str:
        """Add a new iteration result to the population."""
        parent = self.store.get_program(parent_id) if parent_id else None
        generation = (parent.get("generation", 0) + 1) if parent else 0
        feature_coords = self.calculate_feature_coords(metrics)

        pid = self.store.add_program(
            self.session_id,
            commit_hash=commit_hash,
            parent_id=parent_id,
            generation=generation,
            island_id=self.current_island,
            iteration=self.iteration,
            metrics=metrics,
            feature_coords=feature_coords,
            fitness_score=fitness_score,
            changes_description=changes_description,
        )
        return pid

    def maybe_migrate(self):
        """Migrate top programs between islands (ring topology)."""
        interval = self.config.get("migration_interval", 50)
        if self.iteration == 0 or self.iteration % interval != 0:
            return

        rate = self.config.get("migration_rate", 0.1)
        for src_island in range(self.num_islands):
            dst_island = (src_island + 1) % self.num_islands
            top = self.store.get_top_programs(self.session_id, 3, island_id=src_island)
            n_migrate = max(1, int(len(top) * rate))
            for prog in top[:n_migrate]:
                # Copy to destination island
                self.store.add_program(
                    self.session_id,
                    commit_hash=prog.get("commit_hash"),
                    parent_id=prog["id"],
                    generation=prog.get("generation", 0),
                    island_id=dst_island,
                    iteration=self.iteration,
                    metrics=prog.get("metrics"),
                    feature_coords=prog.get("feature_coords"),
                    fitness_score=prog.get("fitness_score", 0),
                    changes_description=f"migrated from island {src_island}",
                    metadata={"migrant": True, "source_island": src_island},
                )

    # ── Single Iteration ─────────────────────────────────────────────────

    def run_iteration(
        self,
        evaluator_fn: Callable[[str], dict] | None = None,
        callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run one evolution iteration.

        Returns dict with: iteration, parent_id, child_id, fitness, metrics,
                          improved (bool), code (str)
        """
        self.iteration += 1
        self.current_island = (self.iteration - 1) % self.num_islands

        # 1. Sample parent
        parent = self.sample_parent()
        if parent is None:
            return {"iteration": self.iteration, "error": "no parent available"}

        # 2. Sample inspirations
        top_progs, diverse_progs = self.sample_inspirations(parent["id"])

        # 3. Build prompt
        system_msg, user_msg = self.build_prompt(parent, top_progs, diverse_progs)

        # 4. Call LLM
        response = self.call_llm(system_msg, user_msg)
        if not response:
            return {"iteration": self.iteration, "error": "LLM call failed"}

        # 5. Parse code
        new_code = self.parse_code(response)
        if not new_code:
            return {"iteration": self.iteration, "error": "no code in response"}

        # 6. Evaluate
        metrics, fitness = self.evaluate(new_code, evaluator_fn)

        # 7. Add to population
        parent_fitness = parent.get("fitness_score", 0)
        improved = fitness > parent_fitness
        child_id = self.add_to_population(
            new_code,
            parent_id=parent["id"],
            metrics=metrics,
            fitness_score=fitness,
            changes_description=f"{'improved' if improved else 'variant'} from {parent['id'][:8]}",
        )

        # 8. Log to evolve_runs
        self.store.log_evolve(
            self.session_id,
            self.iteration,
            prompt=user_msg[:500],
            response_summary=response[:300],
            eval_result=metrics,
            score=fitness,
            source="evolve_engine",
        )

        # 9. Maybe migrate
        self.maybe_migrate()

        result = {
            "iteration": self.iteration,
            "parent_id": parent["id"],
            "child_id": child_id,
            "fitness": fitness,
            "parent_fitness": parent_fitness,
            "metrics": metrics,
            "improved": improved,
            "island": self.current_island,
            "code": new_code,
        }

        if callback:
            callback(result)

        return result

    # ── Main Loop ────────────────────────────────────────────────────────

    def run(
        self,
        n_iterations: int,
        evaluator_fn: Callable[[str], dict] | None = None,
        callback: Callable[[dict], None] | None = None,
        target_score: float | None = None,
    ) -> list[dict]:
        """Run N iterations of evolution.

        Args:
            n_iterations: number of iterations
            evaluator_fn: function(code) -> metrics dict
            callback: called after each iteration with result dict
            target_score: stop early if fitness reaches this

        Returns list of result dicts.
        """
        results = []

        for _ in range(n_iterations):
            if self.stopped:
                break

            result = self.run_iteration(evaluator_fn, callback)
            results.append(result)

            if target_score is not None and result.get("fitness", 0) >= target_score:
                break

        return results

    def stop(self):
        """Signal the engine to stop after the current iteration."""
        self.stopped = True

    # ── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Get current evolution status."""
        total = self.store.count_programs(self.session_id)
        best = self.store.get_best_program(self.session_id)
        island_counts = {}
        for i in range(self.num_islands):
            island_counts[i] = self.store.count_programs(self.session_id, island_id=i)

        return {
            "iteration": self.iteration,
            "total_programs": total,
            "island_counts": island_counts,
            "best_fitness": best.get("fitness_score", 0) if best else 0,
            "best_program_id": best["id"] if best else None,
            "current_island": self.current_island,
        }
