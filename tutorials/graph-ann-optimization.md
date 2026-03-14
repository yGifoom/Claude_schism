# Tutorial: Optimize a Graph-Based ANN Index with Procy

Build and iteratively optimize a graph-based Approximate Nearest Neighbor
search engine using procy's evolution loop. We measure **recall@10** (accuracy)
and **queries per second** (speed) — the classic ANN tradeoff.

## What you'll do

1. Ask Claude to write a baseline NSW (Navigable Small World) graph index
2. Generate an evaluator that measures recall and speed
3. Run `!evolve` to automatically find better graph construction and search strategies

No external libraries needed — the implementation is pure Python + NumPy.

## Step 1: Start procy

```bash
procy
```

This starts procy wrapping Claude Code, with the SSH tunnel to the Qwen model
and the monitor UI on http://localhost:7862.

## Step 2: Ask Claude to write the baseline

Paste this as your first prompt:

```
Write a graph-based approximate nearest neighbor (ANN) search engine in ann.py.
Pure Python + NumPy only. It should implement a Navigable Small World (NSW) graph
with these functions:

  build_index(vectors: np.ndarray, M: int = 16) -> dict
    Build an NSW graph over the given vectors (N x D float32 array).
    M is the max number of neighbors per node.
    Return an index dict containing the graph adjacency lists and the vectors.

  search(index: dict, query: np.ndarray, k: int = 10, ef: int = 50) -> np.ndarray
    Search the graph for k nearest neighbors of the query vector.
    ef is the size of the dynamic candidate list (higher = more accurate, slower).
    Return array of k indices into the original vectors.

Start simple: random insertion order, greedy search with a single entry point.
Keep the code under 150 lines.
```

Claude writes `ann.py`. This is the code that evolution will optimize.

## Step 3: Generate the evaluator

Once `ann.py` exists, tell procy to create a benchmark:

```
!eval generate ANN benchmark: measure recall@10 (fraction of true 10-NN found) against brute-force ground truth, and queries per second. Generate 10000 random 128-dim float32 vectors for the index and 200 query vectors. Report recall_at_10 and qps as the metrics. Import ann.py and call build_index / search.
```

Procy asks Claude to write the evaluator. You'll see:

```
[procy] saved evaluator: eval_ann_benchmark.py
[procy] evaluator registered (id=1)
  detected metrics: recall_at_10, qps
```

## Step 4: Run the evaluator

```
!eval run
```

Typical baseline numbers for a naive NSW:

```
[procy] eval complete: {"recall_at_10": 0.62, "qps": 312.5}
```

~62% recall is poor. A good graph ANN should reach 95%+ recall while staying
fast. That's what we'll optimize.

## Step 5: Evolve — fixed policy (automatic)

```
!evolve 10
```

Procy's evolution engine runs 10 iterations. Each iteration:

1. Selects a parent program from the population (MAP-Elites)
2. Shows the LLM the current code + metrics + top/diverse programs
3. LLM produces an improved version
4. Evaluator measures recall and speed
5. New program enters the population if it's good

Watch progress:

```
!status
```

You'll see output like:

```
[procy] evolve: state=running progress=4/10
  evolve-note: [#4/10] improved! fitness=0.8700
```

After it finishes, check what happened:

```
!eval metrics
```

## Step 6: Keep pushing — manual iteration

If evolve stalls, you can intervene. Look at what the best version does:

```
Show me the current ann.py and explain its graph construction strategy.
```

Then give Claude a specific direction:

```
The recall is 0.87 but we need 0.95+. The issue is the graph has poor
connectivity — nodes inserted early have few long-range connections.
Try adding a hierarchical layer structure like HNSW: insert nodes at
random levels, search from the top layer down. Keep M=16 per layer.
```

Run the evaluator:

```
!eval run
```

If recall improves, great. If the proxy would have generated a better prompt,
correct it:

```
!correct
```

This logs a training pair (proxy's prompt → your better prompt) for later
fine-tuning.

## Step 7: Evolve again with the improved code

```
!evolve 10
```

The population now includes both the baseline and your improved version.
The engine uses both as parents for further mutation, combining the best
ideas from each lineage.

## Step 8: Check the population

```
!status
```

Shows total programs, best fitness, and island distribution.

Open the UI at http://localhost:7862 for the full picture:
- **Evolve tab**: results grid with metrics per iteration
- **Interactions tab**: full prompt/response history

## What the evaluator measures

| Metric | Goal | What it means |
|--------|------|--------------|
| `recall_at_10` | maximize | Fraction of true 10 nearest neighbors found by the graph search |
| `qps` | maximize | Queries per second (search throughput) |

The primary fitness score is `recall_at_10` (first metric). The engine
optimizes for recall while the prompt templates also show speed metrics so
the LLM considers the tradeoff.

## Ideas the evolution might discover

- **Better neighbor selection**: using RNG (Relative Neighborhood Graph) pruning
  instead of keeping all candidates
- **Multi-layer HNSW**: hierarchical structure with exponential layer assignment
- **Improved search**: beam search with backtracking instead of simple greedy
- **Construction heuristics**: inserting high-degree hub nodes first
- **Distance caching**: avoiding redundant distance computations during search
- **Entry point selection**: using multiple random entry points or the centroid

## Switching to proxy policy

If you want the Qwen model to generate directional prompts (injected into
Claude via PTY) instead of the fixed OpenEvolve-style templates:

```bash
procy --evolve-policy proxy
```

Same DB, same session — just a different strategy for generating mutations.
The proxy policy is better once the Qwen model has been fine-tuned on
evolution data from many sessions.

## Command reference

| Command | What it does |
|---------|-------------|
| `!eval generate <desc>` | Ask Claude to write an evaluator |
| `!eval run` | Run evaluator, see metrics |
| `!eval metrics` | Show all eval results history |
| `!evolve N` | Run N evolution iterations |
| `!status` | Check evolve progress & population |
| `!stop` | Stop an ongoing evolve |
| `!correct` | Correct the last prompt (training data) |
| `!deploy status` | Check GPU inference containers |
| `!help` | List all commands |
