# Tutorial: Optimize a Text Search Engine with ProCy

This walks through a full procy session — from task to evaluator to evolve.
No proxy model needed (works with fixed-policy 14B or even without it for manual iteration).

## Prerequisites

```bash
pip install -e .   # install procy
```

## Step 1: Start procy

```bash
cd ~/my-project   # or any working directory
procy --no-tunnel  # skip GPU tunnel if not using proxy model
```

You're now inside a procy session wrapping Claude Code.

## Step 2: Give Claude the task

Type this as your first prompt to Claude:

```
Write a Python text search engine in baseline.py. It should have two functions:
- build_index(documents: list[str]) -> dict — builds a search index
- search(index: dict, query: str, k: int = 10) -> list[int] — returns top-k doc indices

Start with a simple TF-IDF implementation. Keep it self-contained, no external dependencies.
```

Claude writes `baseline.py`. This is the artifact that will be optimized.

## Step 3: Generate the evaluator

Once Claude has written `baseline.py`, type the procy command:

```
!eval generate benchmark text search: measure recall@10 against brute-force ground truth, queries per second, and build time. use 10000 synthetic documents and 100 queries.
```

Procy injects a prompt into Claude asking it to write an evaluator script.
Claude writes it, procy extracts the code, saves it as `eval_benchmark_text_search.py`, and registers it.

You'll see:
```
[procy] saved evaluator: eval_benchmark_text_search.py
[procy] evaluator registered (id=1)
  detected metrics: recall_at_10, qps, build_time_s
  run: !eval run to test it, or !evolve N to start optimizing
```

## Step 4: Test the evaluator

```
!eval run
```

You should see something like:
```
[procy] eval complete: {"recall_at_10": 1.0, "qps": 45.2, "build_time_s": 0.38}
```

If it errors, tell Claude to fix it:
```
The evaluator at eval_benchmark_text_search.py failed with: <paste error>. Fix it.
```

Then `!eval set eval_benchmark_text_search.py` to re-register and `!eval run` again.

## Step 5: Start optimizing (manual)

Now ask Claude to improve:

```
The baseline gets 45 qps. Optimize baseline.py to get at least 500 qps while keeping recall@10 above 0.95. Try an inverted index with posting lists.
```

After Claude modifies `baseline.py`, test again:

```
!eval run
```

Repeat — look at the metrics, tell Claude what to try next.

## Step 5 (alternative): Auto-evolve with proxy model

If you have the proxy model running (vLLM on EXP07):

```bash
procy   # starts with tunnel to proxy model
```

Then after steps 2-4:

```
!evolve 5
```

Procy's proxy model generates optimization prompts, injects them into Claude,
and runs the evaluator after each iteration. Watch progress with:

```
!status
```

See results:

```
!eval metrics
```

## Step 6: Review in the UI

Open http://localhost:7862

- **Evolve tab**: see the results grid — each iteration's prompt, response, metrics
- **Interactions tab**: full prompt/response history
- **Training tab**: export the session data for fine-tuning the proxy model

## Step 7 (optional): Correct and train

If the proxy model generated a bad prompt, click it in the UI and write a correction.
That correction becomes DPO training data (proxy prompt = rejected, your correction = chosen).

## Summary of procy commands used

| Command | What it does |
|---------|-------------|
| `!eval generate <desc>` | Ask Claude to write an evaluator, auto-register it |
| `!eval set <path>` | Register an existing script as evaluator |
| `!eval run` | Run evaluator manually, see metrics |
| `!eval show` | Show current evaluator info |
| `!eval metrics` | Show all eval results history |
| `!evolve N` | Run N auto-optimization iterations |
| `!status` | Check evolve progress |
| `!correct` | Correct the last prompt |
| `!stop` | Stop an ongoing evolve |

Type `!` as the first character to enter procy command mode. All commands start with `!`.
