# ProCy

Transparent prompt proxy between human and AI coding agents. ProCy sits between you and your AI agent (e.g. Claude Code), recording every interaction and learning to generate better prompts over time.

## What it does

- **Wraps** any CLI agent (Claude Code, etc.) in a PTY proxy — zero changes to the agent
- **Records** every prompt, response, and tool call to a local SQLite database
- **Evolves** prompts: a local proxy model (Qwen 14B + LoRA) generates improved prompts based on past results
- **Learns** from human corrections: when you fix a prompt, that becomes training data
- **Trains** the proxy model from the UI — one click to fine-tune on your corrections
- **Replays** terminal sessions via xterm.js in the web UI

## Install

```bash
pip install -e .
```

Requires Python 3.10+. The only runtime dependency is Flask (for the monitor UI).

## Usage

```bash
# Basic: wrap Claude Code
procy

# With a specific agent command
procy --agent "claude --dangerously-skip-permissions"

# Skip the SSH tunnel to GPU server (if not using proxy model)
procy --no-tunnel

# Custom Qwen API URL (if proxy model is hosted elsewhere)
procy --qwen-url http://localhost:18000
```

## ProCy Commands

Inside a procy session, type `!` at the start of a line to enter command mode:

| Command | Description |
|---------|-------------|
| `!evolve N` | Run N iterations of prompt evolution using the proxy model |
| `!eval set <path>` | Set evaluator script for this session |
| `!eval generate [desc]` | Ask Claude to write an evaluator, auto-register it |
| `!eval show` | Show current evaluator info |
| `!eval run` | Run evaluator manually |
| `!eval metrics` | Show all eval results history |
| `!correct` | Correct the last prompt (opens editor, logs for training) |
| `!train` | Export correction pairs as JSONL for fine-tuning |
| `!status` | Show current session status, evolve progress |
| `!history` | Show prompt and correction history |
| `!stop` | Stop an ongoing evolve |
| `!reset-evolve` | Clear stuck evolve state |
| `!help` | Show available commands |

## End-to-End Tutorial

This walks through a full procy cycle: give a task, set up an evaluator, iterate manually, correct the proxy, train, reload the finetuned model, and verify improvement.

### Step 1: Start procy

```bash
# Without proxy model (manual iteration only)
procy --no-tunnel

# With proxy model (needs vLLM serving Qwen on a GPU server)
procy
```

You're now inside a procy session wrapping Claude Code.
The monitor UI is at http://localhost:7862.

### Step 2: Give Claude a task

Type a prompt to Claude as normal. Procy records it transparently.

```
Write a Python text search engine in baseline.py. It should have two functions:
- build_index(documents: list[str]) -> dict — builds a search index
- search(index: dict, query: str, k: int = 10) -> list[int] — returns top-k doc indices

Start with a simple TF-IDF implementation. Keep it self-contained, no external dependencies.
```

Claude writes `baseline.py`. This is the artifact that will be optimized.

### Step 3: Generate an evaluator

Once Claude has written the code, ask procy to generate an evaluator:

```
!eval generate benchmark text search: measure recall@10 against brute-force ground truth, queries per second, and build time. use 10000 synthetic documents and 100 queries.
```

Procy injects a prompt into Claude asking it to write an evaluator script.
Claude writes it, procy extracts the code, saves it, and registers it:

```
[procy] saved evaluator: eval_benchmark_text_search.py
[procy] evaluator registered (id=1)
  detected metrics: recall_at_10, qps, build_time_s
```

### Step 4: Test the evaluator

```
!eval run
```

You should see:
```
[procy] eval complete: {"recall_at_10": 1.0, "qps": 45.2, "build_time_s": 0.38}
```

If it errors, tell Claude to fix it, then re-register with `!eval set <path>`.

### Step 5: Iterate (manual or auto)

**Manual:** Ask Claude to improve, then run the evaluator again.

```
The baseline gets 45 qps. Optimize baseline.py to get at least 500 qps while keeping recall@10 above 0.95. Try an inverted index with posting lists.
```

After Claude modifies the code:

```
!eval run
```

Repeat — look at the metrics, tell Claude what to try next.

**Auto-evolve (requires proxy model):**

```
!evolve 5
```

The proxy model generates optimization prompts, injects them into Claude,
and runs the evaluator after each iteration. Check progress with `!status`.

### Step 6: Correct the proxy

If the proxy model generated a bad prompt during evolve, correct it:

```
!correct
```

This opens your `$EDITOR` with the last proxy prompt. Edit it, save, and quit.
The correction is stored as a training pair: original (rejected) vs your version (chosen).

You can also click any prompt in the monitor UI (http://localhost:7862, Interactions tab)
and write a correction there.

### Step 7: Export training data

```
!train
```

This exports all correction pairs as `procy_train.jsonl` in the working directory.
Each line has `instruction` (original proxy prompt) and `output` (your corrected version).

You can also export from the Training tab in the monitor UI, which shows all three categories:
- **Human** — your prompts (gold standard for SFT)
- **Corrected** — proxy prompts you fixed (SFT + DPO pairs)
- **Proxy** — proxy prompts you accepted without correction (implicit approval)

### Step 8: Fine-tune the proxy model

On your GPU server (inside a vLLM Docker container or any env with CUDA):

```bash
pip install peft trl datasets accelerate
python3 scripts/train_proxy.py \
    --data procy_train.jsonl \
    --model Qwen/Qwen2.5-14B-Instruct \
    --output /data/proxy_lora \
    --epochs 3
```

Or click **Start Training** in the Training tab of the monitor UI — it SSHes to your
GPU server and runs the same script inside Docker.

The training script supports multi-GPU with fp16 and gradient checkpointing.
Output: a LoRA adapter directory at `--output`.

### Step 9: Reload vLLM with the finetuned adapter

Restart vLLM with the LoRA adapter so procy's `!evolve` uses the finetuned model:

```bash
# On the GPU server, restart vLLM with the LoRA adapter enabled:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --enable-lora \
    --lora-modules proxy=/data/proxy_lora \
    --port 8000
```

The `--lora-modules proxy=/data/proxy_lora` flag registers the adapter under the name `proxy`,
which is the model name procy uses when calling the API (`"model": "proxy"` in the request).

### Step 10: Verify improvement

Start a new procy session (or resume with `procy --resume-procy <session-id>`),
give it the same task, and run evolve again:

```
!evolve 5
```

Compare the new evolve results against the previous session. The finetuned proxy
should generate better directional prompts from the start, based on what it learned
from your corrections. Check both sessions in the monitor UI to compare metrics side by side.

## Web Monitor UI

The monitor UI starts automatically on port 7862 when you launch procy.

Open `http://localhost:7862` to see:

- **Interactions** — Human prompts (green), agent responses (gold), evolve prompts (violet). Click any prompt to correct it.
- **Terminal** — Full PTY replay of the session via xterm.js
- **Evolve** — Tagged tries (#1, #2, ...) with scores, prompts, and responses
- **Corrections** — All human corrections, with add/delete/export
- **Training** — Three categories of training data (human/corrected/proxy), export as SFT or DPO, and a "Start Training" button to fine-tune the proxy model on a remote GPU server

## Architecture

```
Human  <-->  ProCy (PTY proxy)  <-->  AI Agent (Claude Code)
               |                           |
               |  records to SQLite        |  raw terminal I/O
               |  generates prompts        |
               v                           v
          Proxy Model              Agent does the work
        (Qwen 14B + LoRA)
               |
               v
          Monitor UI (Flask + xterm.js)
```

## Project Structure

```
procy/
  cli.py        — Main entry point, Procy class, evolve loop
  store.py      — SQLite trace store (sessions, turns, corrections, evolves)
  terminal.py   — PTY proxy session management
  ui.py         — Flask web UI with xterm.js terminal replay
  io.py         — Thread-safe terminal I/O
  agent.py      — Agent process management
  assets/       — xterm.js CSS/JS for terminal replay
scripts/
  train_proxy.py         — LoRA fine-tuning script (multi-GPU)
  run_evolve_pipeline.py — Standalone evolve pipeline (no procy needed)
  eval_ann.py            — ANN search benchmark evaluator
  ann_search.py          — Baseline ANN implementation
tutorials/
  text-search-optimization.md — Extended tutorial with text search example
```

## License

MIT
