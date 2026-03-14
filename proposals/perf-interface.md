# Proposal: Learning to Interpret Performance — Agent-Driven Training Framework Optimization via GPU Simulation and Human-in-the-Loop Proxy

## What We Have

### 1. Transparent GPU-Cluster Simulator

We built a GPU-cluster simulator that runs the **real training stack** (Megatron-LM, PyTorch, NCCL, RDMA) end-to-end, replacing only the hardware layer with lightweight, timing-synchronized simulators for GPUs, NICs, and the network. Unlike trace-based or analytical approaches, our simulator preserves the full software stack and captures cross-layer CPU–GPU–Network interactions that drive real performance behavior. A CompressT mechanism keeps wall-clock cost proportional to simulated time, making rapid iteration practical without a real cluster.

With this simulator, we can:

- **Answer what-if questions** — What happens if we switch from TP=4 to TP=8? What if we add an optimized kernel? What if we upgrade to faster interconnects?
- **Profile at full fidelity** — See compute/communication overlap, collective stalls, memory pressure, and pipeline bubble ratios as they would appear on real hardware.
- **Iterate without GPUs** — Test framework changes, parallelism strategies, and kernel optimizations on a CPU-only machine in minutes, not hours on a contested cluster.

### 2. Procy — Transparent Prompt Proxy for AI Agents

We built Procy, a PTY-level proxy that sits between a human and an AI coding agent (e.g., Claude Code). Procy:

- **Traces everything** — every prompt, response, tool call, and code change is recorded to a structured SQLite database.
- **Collects human corrections** — when the human corrects the agent's approach, that correction becomes SFT/DPO training data.
- **Runs an evolve loop** — a local proxy model (Qwen 14B + LoRA) generates improved prompts based on past results and human feedback, injects them into the agent, and measures outcomes with custom evaluator scripts.
- **Learns from the loop** — correction pairs are used to fine-tune the proxy model, so it gets better at generating effective prompts over time.

---

## The Gap

Today, optimizing a distributed training framework is a deeply manual, expert-driven process:

1. An engineer profiles a training run (or uses the simulator).
2. They stare at timelines, traces, and counters — interpreting what the numbers mean.
3. They form a hypothesis ("the all-reduce is blocking compute because launch latency is too high").
4. They modify the framework code.
5. They re-profile and check if it helped.
6. Repeat.

**Step 2 is the bottleneck.** Interpreting performance data requires years of expertise in GPU architecture, collective communication, and framework internals. Even experts struggle with novel configurations. And this interpretation — the mapping from "what the profiler shows" to "what code change to try" — is entirely locked in human heads. It is not written down, not formalized, and not learnable by current AI systems.

---

## What We Propose

### A Performance Interface for Agent-Driven Framework Optimization

We propose closing the loop between the simulator and an AI agent, with Procy as the learning layer in between.

```
┌──────────────────────────────────────────────────────┐
│                    Human Expert                       │
│         (corrects, guides, approves)                  │
└────────────────────┬─────────────────────────────────┘
                     │ corrections / feedback
                     ▼
┌──────────────────────────────────────────────────────┐
│              Proxy Model                      │
│  Interprets perf data → generates optimization prompt │
│  Trained on human interaction traces                  │
└──────┬───────────────────────────────────┬───────────┘
       │ optimization prompt               │ learn from
       ▼                                   │ outcomes
┌──────────────────┐              ┌────────┴───────────┐
│   AI Agent       │              │   Evaluator        │
│  (Claude Code)   │──  code  ──▶│  (Simulator-based)  │
│  modifies the    │   changes   │  runs simulated     │
│  training code   │             │  training, outputs  │
│                  │             │  perf metrics        │
└──────────────────┘              └─────────────────────┘
```

#### The Performance Interface

The core contribution is a structured **performance interface** — a representation of simulator output that an agent (or proxy model) can consume and act on. This is not a dashboard for humans; it is a machine-readable summary of:

- **Bottleneck classification**: compute-bound vs. communication-bound vs. memory-bound, per pipeline stage
- **Overlap efficiency**: what fraction of communication is hidden behind compute
- **Critical path**: which operations are on the critical path and by how much
- **Comparative delta**: what changed (and by how much) relative to the previous iteration

The interface transforms raw simulation traces into a structured context that can be prepended to an agent prompt: "Here is what the profiler found. The all-reduce on layer 23 is the bottleneck, adding 12ms to the critical path. Communication overlap is 34%. Suggest a code change."

#### The Learning Loop

1. **Human + Agent + Simulator**: A human expert uses Procy to direct an AI agent to optimize a training framework. The simulator provides fast feedback (metrics, profiles). Procy records every interaction: the performance data the human looked at, the interpretation they gave, the code change they requested, and whether it helped.

2. **Training the Proxy Model**: The collected traces are used to fine-tune the proxy model to perform the expert's role:
   - **Input**: performance interface (structured simulator output)
   - **Output**: optimization prompt (what to tell the agent to try next)
   - **Signal**: evaluator score delta (did the suggestion improve throughput?)

3. **Self-Evolve**: Once the proxy model has learned basic performance interpretation, it can drive the evolve loop autonomously — reading simulator output, generating optimization prompts, injecting them into the agent, and measuring results. Human experts only intervene to correct bad suggestions, and those corrections further improve the model.

#### What This Achieves

- **Democratize performance optimization**: A non-expert can use the system to optimize their training setup. The proxy model encodes expert knowledge about interpreting performance data.
- **Scale expert iteration**: An expert can supervise multiple optimization runs simultaneously. The proxy model handles routine optimizations; the expert focuses on novel bottlenecks.
- **Capture tacit knowledge**: The interpretation of performance data — which is currently locked in expert heads — becomes formalized as training data and embedded in a model.
- **No real GPUs needed**: The entire loop (agent writes code → simulator evaluates → proxy model interprets → agent tries again) runs on CPU, making it accessible and cheap to iterate.

---

## Concrete Plan

### Phase 1: Performance Interface Design (Weeks 1–3)

- Define the structured performance interface schema (JSON/text format)
- Build extractors that convert raw simulator traces into the interface format
- Validate that the interface captures the information experts actually use when making optimization decisions

### Phase 2: Data Collection (Weeks 3–6)

- Expert sessions: 3–5 experts use Procy + Simulator to optimize real training configurations (different models, different parallelism strategies, different cluster sizes)
- Procy records all interactions: performance data seen, interpretation given, code changes made, resulting metrics
- Target: 200+ optimization steps with human-annotated reasoning

### Phase 3: Proxy Model Training (Weeks 6–8)

- Structure collected data as (performance_interface, expert_prompt) pairs
- Fine-tune Qwen 14B with LoRA on interpretation + suggestion generation
- Evaluate: given simulator output, does the model suggest changes that actually improve performance?

### Phase 4: Autonomous Evolve Loop (Weeks 8–12)

- Deploy the trained proxy model in Procy's evolve loop
- Run end-to-end: proxy model reads simulator output → generates optimization prompt → agent modifies code → simulator re-evaluates
- Measure: how many iterations to reach X% of expert-level throughput, with and without the trained proxy model
- Collect new corrections from experts reviewing autonomous runs → retrain

---

## Expected Outcomes

1. **A performance interface specification** that bridges the gap between raw simulator output and actionable optimization guidance.
2. **A trained proxy model** that can interpret performance data and generate effective optimization prompts, reducing the need for expert involvement by an estimated 60–80% of routine optimization steps.
3. **A self-improving system** where every human correction makes the proxy model better, creating a flywheel: more usage → more data → better model → less human effort → more usage.
4. **Reproducible benchmarks** comparing human-only, agent-only, and proxy-guided optimization across standard training configurations (GPT-3 style models at 1B/7B/13B scale, various parallelism strategies).
