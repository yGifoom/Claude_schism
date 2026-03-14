# Tutorial: Optimize a JPEG Decoder RTL Design with Procy

Use procy to iteratively improve a Verilog JPEG decoder — reducing cycle
count and buffer size while maintaining functional correctness. The evaluator
runs Verilator simulations on a remote server.

## What you'll do

1. Clone an open-source JPEG decoder RTL design
2. Set up a Verilator-based evaluator that measures cycles, buffer size, and correctness
3. Use `!evolve` to have Claude iterate on the Verilog, guided by eval results

## Prerequisites

- SSH access to a server with Verilator installed (e.g., `ssh D6` or `ssh D7`)
- The JPEG decoder repo: `git@github.com:ultraembedded/core_jpeg_decoder.git`

## Step 1: Start procy in the project directory

```bash
cd ~/core_jpeg_decoder
procy
```

## Step 2: Give Claude the task

```
I need to improve a JPEG decoder design: git@github.com:ultraembedded/core_jpeg_decoder.git.
Use Verilator for simulation. I'm targeting cycle count (performance) and buffer size reduction.
Generate some test JPEG images — note the design only supports certain formats.
Use ssh D6 to set up experiments and run tests.
```

Claude will:
- Clone the repo (if needed)
- Set up Verilator on the remote server
- Generate compatible test JPEG images
- Build and run initial simulations

## Step 3: Generate the evaluator

```
!eval generate JPEG decoder benchmark: measure functional_correctness (fraction of pixels matching baseline design), avg_cycles_per_pixel (average simulation cycles per pixel, lower is better), and total_buffer_bits (total RAM/register storage in the Verilog, lower is better). Build with Verilator on ssh D6. Compare candidate (working tree) against baseline (git HEAD). Run on generated test images.
```

Claude writes `eval.py` which:
1. Snapshots the original Verilog as baseline
2. Builds both baseline and candidate designs with Verilator
3. Runs simulations on test JPEG images
4. Compares pixel outputs for functional correctness
5. Parses cycle counts from simulation
6. Counts buffer bits from Verilog source

Typical baseline output:
```json
{"functional_correctness": 1.0, "avg_cycles_per_pixel": 2.2308, "total_buffer_bits": 29312}
```

## Step 4: Register and test the evaluator

```
!eval set eval.py
!eval run
```

The evaluator takes ~50 seconds (remote Verilator build + simulation).
You'll see live progress:
```
[procy] running evaluator 'default'...
  eval: Running baseline simulation...
  eval: Running candidate simulation...
  eval: {"functional_correctness": 1.0, "avg_cycles_per_pixel": 2.2308, "total_buffer_bits": 29312}
[procy] eval complete
  functional_correctness = 1.0
  avg_cycles_per_pixel = 2.2308
  total_buffer_bits = 29312
```

## Step 5: Evolve

```
!evolve 5
```

Procy's evolution engine will:
1. Select a parent approach from the population
2. Prompt Claude to improve the Verilog (Claude reads files, queries the
   population DB, and makes changes)
3. Run the evaluator after each iteration
4. Commit changes and record results

Check progress:
```
!status
```

## Step 6: Manual iteration

If evolve stalls, intervene with specific directions:

```
The buffer bits are 29312. Look at the IDCT buffer — it uses 32-bit storage
for coefficients that only need 12 bits. Reduce the width and adjust the
pipeline accordingly. Make sure functional_correctness stays at 1.0.
```

Then run the evaluator:
```
!eval run
```

## What the evaluator measures

| Metric | Goal | Baseline | Description |
|--------|------|----------|-------------|
| `functional_correctness` | maximize (=1.0) | 1.0 | Pixel-exact match with baseline |
| `avg_cycles_per_pixel` | minimize | 2.23 | Simulation cycles per output pixel |
| `total_buffer_bits` | minimize | 29312 | Total register/RAM storage in RTL |

The primary fitness score is `functional_correctness` — any regression here
means the optimization broke the decoder.

## Example results from a real session

| Iteration | correctness | cycles/pixel | buffer bits | What changed |
|-----------|-------------|-------------|-------------|--------------|
| baseline  | 1.0 | 2.2308 | 29312 | original design |
| manual    | 1.0 | 2.2306 | 29312 | minor pipeline tweak |
| manual    | 1.0 | 2.2306 | 17024 | reduced IDCT buffer widths |

The buffer size was reduced by **42%** (29312 → 17024 bits) while maintaining
full functional correctness and nearly identical cycle count.

## Tips for RTL optimization with procy

- **Keep correctness at 1.0** — always check `functional_correctness` first.
  If it drops, revert immediately.
- **Target one metric at a time** — first reduce cycles, then optimize buffers,
  or vice versa. Don't try both at once.
- **Use `!correct`** when Claude makes a change that breaks correctness — this
  teaches the proxy model what not to do.
- **Remote evaluation** — the evaluator SSHs to the simulation server. Make
  sure SSH keys are set up for passwordless access.
- **Verilator build time** — each eval takes ~50s due to Verilator compilation.
  Consider using `verilator --build -j` for parallel builds.

## Command reference

| Command | What it does |
|---------|-------------|
| `!eval generate <desc>` | Ask Claude to write an evaluator |
| `!eval set <path>` | Register an evaluator script |
| `!eval run` | Run evaluator, see metrics |
| `!eval metrics` | Show all eval results history |
| `!evolve N` | Run N evolution iterations |
| `!status` | Check evolve progress & population |
| `!stop` | Stop an ongoing evolve |
| `!correct` | Correct the last prompt (training data) |
| `!help` | List all commands |
