# Async Rollout + Speculative Decoding Online Simulator

This directory contains an experiment harness for studying speculative decoding
in an asynchronous RLHF-style rollout workload.

The key design choice is:

- `workload`, `queueing`, `dynamic batching`, `KV admission`, and `accept/reject`
  are simulated at the system level.
- `compute` can be executed in two modes:
  - `proxy`: lightweight analytical cost model for rapid iteration
  - `real_hf`: real Hugging Face forward passes driven online by the scheduler

The second mode is the important one for your use case: the scheduler decides
the batch shape at each step, and that exact shape is then executed by HF so
you can measure real latency and memory behavior during a dynamic scheduling
process.

## What This Simulator Is Trying To Answer

The target question is not just:

- "Does SD speed up one isolated decode call?"

It is:

- "When rollout requests arrive asynchronously and are packed by a dynamic
  batching scheduler, what real system-level gains or regressions does SD
  create?"

Concretely, the simulator is intended to help answer:

- Does SD improve end-to-end throughput under async rollout traffic?
- How does SD change queue growth and stability under load?
- When does a larger `chunk_size` help, and when does it hurt?
- Under low acceptance, does SD create negative system-level returns?
- How do KV pressure and dynamic batching interact with SD?
- What happens to tail latency and backlog as load increases?

## What Is Simulated vs What Is Real

### Simulated

- request arrivals
- prompt length / generation length distributions
- rollout-burst style asynchronous traffic
- scheduler logic
- KV cache accounting
- acceptance / rejection outcomes
- request lifecycle transitions

### Real when `use_real_compute=True`

- HF `prefill` forward cost for the scheduled batch shape
- HF `decode` forward cost for the scheduled batch shape
- wall-clock latency per scheduling step
- CUDA memory allocated / peak memory allocated / reserved memory

### Important Limitation

This simulator is **not** a drop-in replacement for vLLM internals.

It approximates a vLLM-like serving loop, but the actual compute engine is HF.
That means:

- it is useful for validating whether SD introduces system-level performance
  problems in this scenario
- it is not a perfect prediction of production numbers from vLLM

The simulator is best viewed as a fast, controllable experimental platform for
finding trends, failure regions, and candidate explanations before building or
modifying a full serving stack.

## Core Files

- [simulator.py](d:/phd/git/hf_infer/sim/simulator.py)
  Main event loop. Owns requests, queue, scheduler, KV tracking, batch
  processing, acceptance, and summary metrics.

- [executor.py](d:/phd/git/hf_infer/sim/executor.py)
  Compute backends.
  `ProxyExecutor` is analytical.
  `HFRealExecutor` executes real HF forwards online inside the scheduling loop.

- [run_online.py](d:/phd/git/hf_infer/sim/run_online.py)
  CLI entrypoint for running a single simulation with either proxy or real HF
  compute.

- [hardware_benchmark.py](d:/phd/git/hf_infer/sim/hardware_benchmark.py)
  Offline shape benchmark utility. Useful if you want a calibrated proxy mode,
  but it does **not** replace the online real compute mode.

- [experiments.py](d:/phd/git/hf_infer/sim/experiments.py)
  Parameter sweeps for quick trend analysis.

- [visualizer.py](d:/phd/git/hf_infer/sim/visualizer.py)
  Post-processing plotting tool. Reads saved raw JSON outputs and generates
  comparison plots and time-series plots while preserving the original data.

- [quickstart.py](d:/phd/git/hf_infer/sim/quickstart.py)
  A few preset scenarios.

- [test_simulator.py](d:/phd/git/hf_infer/sim/test_simulator.py)
  Tests for queueing, KV tracking, metrics, and benchmark-table fallback logic.

## System Architecture

Each simulation step follows this loop:

1. `RequestGenerator` injects new rollout-like requests.
2. Requests enter the waiting queue.
3. `KVCacheManager` decides whether requests can be admitted.
4. `BatchScheduler` forms the next batch from active requests.
5. The chosen batch is executed:
   - `prefill` for newly admitted requests
   - `decode` for requests already in generation
6. After compute, `SDAccepter` applies speculative accept/reject behavior.
7. Request state, sequence length, and KV usage are updated.
8. Completed requests are removed and memory is released.
9. The event loop advances with the measured compute latency.

This is the key property of the framework:

- scheduling and compute are in the same closed loop

That is what allows the simulator to expose system-level side effects such as:

- larger chunk sizes increasing per-step cost
- low acceptance creating wasted work and extra backlog
- higher throughput reducing queueing delay
- KV growth shifting the scheduler’s feasible batch composition over time

## Compute Modes

### 1. Proxy Mode

Use this when you want quick iteration and broad sweeps.

Characteristics:

- fast
- no GPU dependency
- good for logic validation and rough trends
- not sufficient by itself for credible hardware conclusions

### 2. Real HF Mode

Use this when you want the scheduler to drive actual HF work online.

Characteristics:

- slower
- requires `torch`, `transformers`, and a usable device
- measures latency and CUDA memory inside the event loop
- best mode for validating whether SD causes real regressions under dynamic
  scheduling

### How `HFRealExecutor` Works

The executor uses the scheduler-selected batch shape each step:

- `prefill`: batch size and padded prompt length
- `decode`: batch size, padded sequence length, and chunk size

The content itself is synthetic token IDs, because the goal is to study:

- batching shape
- KV growth
- scheduling interaction
- hardware cost

This is intentional. For this workload study, shape and timing matter more than
semantic correctness of tokens.

## Metrics

The simulator reports system metrics such as:

- `throughput_tokens_per_sec`
- `avg_request_latency_sec`
- `p95_request_latency_sec`
- `avg_queue_wait_sec`
- `stability_ratio`
- `queue_backlog_ratio`
- `peak_kv_utilization`
- `avg_batch_size`
- `draft_acceptance_rate`
- `fallback_share`
- per-step `step_traces`
- windowed time-series metrics via `get_windowed_metrics()`

When compute is executed online with HF, it also reports:

- `avg_step_memory_allocated_mb`
- `peak_step_memory_allocated_mb`
- `executor_mode`

## Example Usage

### Quick Proxy Run

```bash
python sim/run_online.py --duration 20 --workload-mode mixed
```

Export summary + per-step trace + windowed metrics:

```bash
python sim/run_online.py \
  --duration 20 \
  --output-json sim/run_summary.json \
  --output-trace-json sim/step_trace.json \
  --output-window-json sim/window_metrics.json
```

Generate plots from those raw files:

```bash
python sim/visualizer.py \
  --window-json sim/window_metrics.json \
  --trace-json sim/step_trace.json \
  --output-dir sim/plots \
  --prefix rollout_run
```

### Async Rollout Style Traffic

```bash
python sim/run_online.py \
  --duration 30 \
  --workload-mode rollout_burst \
  --rollout-burst-size 12 \
  --arrival-rate 16 \
  --batch-size 16 \
  --chunk-size 4 \
  --accept-rate 0.85
```

### Real HF Online Compute

```bash
python sim/run_online.py \
  --use-real-compute \
  --model /path/to/your/model \
  --device cuda \
  --dtype float16 \
  --duration 30 \
  --workload-mode rollout_burst \
  --arrival-rate 16 \
  --batch-size 16 \
  --max-concurrent 64 \
  --chunk-size 4 \
  --accept-rate 0.85 \
  --verbose
```

### Disable Speculative Decoding for Baseline

```bash
python sim/run_online.py \
  --disable-speculative \
  --chunk-size 1 \
  --accept-rate 1.0
```

### Quick Comparison Preset

```bash
python sim/quickstart.py compare
```

## Offline Calibration Mode

If you want a better proxy mode but do not want full online HF execution for
every experiment, you can benchmark a grid of shapes and then load it into
`ComputeModel`.

Generate a table:

```bash
python sim/hardware_benchmark.py --model /path/to/your/model --output sim/benchmark_table.json
```

Use it:

```bash
python sim/run_online.py --benchmark-table-path sim/benchmark_table.json
```

This is useful for:

- large sweeps
- quick iteration after one calibration pass

But it is still less faithful than the online real compute mode because it
cannot capture all time-varying interactions between scheduler decisions and
runtime behavior.

## Recommended Experimental Workflow

### Phase 1: Logic and Stability

Run proxy mode first.

Goals:

- verify the scheduler loop behaves sensibly
- check whether queueing and KV pressure trends make sense
- identify rough regions of interest

### Phase 2: Online Real Compute

Switch to `use_real_compute=True`.

Goals:

- measure real latency under dynamic batching
- measure peak memory behavior under different `chunk_size`
- validate whether proxy conclusions survive real execution

### Phase 3: Comparative Studies

Compare:

- `SD on` vs `SD off`
- low vs high `accept_rate`
- small vs large `chunk_size`
- low vs high arrival pressure
- poisson vs rollout-burst workloads

The experiment runner now stores, for each condition:

- final summary metrics
- per-window time-series aggregates
- baseline metrics from `SD off`
- `delta_vs_baseline` fields such as throughput speedup and latency change

And you can generate aggregate comparison figures from the saved experiment file:

```bash
python sim/visualizer.py --results-json sim_results.json --output-dir sim/plots
```

If you specifically want to understand the impact of `chunk_size`, run a focused
chunk sweep first:

```bash
python sim/experiments.py ^
  --suite chunk ^
  --duration 20 ^
  --arrival-rate 10 ^
  --batch-size 16 ^
  --accept-rate 0.85 ^
  --output-json sim/chunk_results.json ^
  --output-md sim/chunk_report.md ^
  --output-csv sim/chunk_summary.csv
```

And then visualize it:

```bash
python sim/visualizer.py --results-json sim/chunk_results.json --output-dir sim/plots
```

This is where you should look for:

- throughput gain regions
- latency blow-up regions
- backlog / instability regions
- negative-return SD regions

## Why This Can Still Be Useful Even Though It Is Not vLLM

This framework can reflect real system behavior well enough to be useful if
your question is:

- "Is there likely a system-level performance issue when SD is introduced into
  an async rollout scheduler?"

It is especially helpful because it preserves the most important coupling:

- dynamic queueing
- KV pressure
- variable batch composition
- real online compute cost
- acceptance-driven request evolution

That coupling is the heart of your research question.

## What It Still Does Not Capture Perfectly

- vLLM’s exact paged-attention kernels
- kernel-level memory bandwidth counters
- exact cache management details of production servers
- networking / RPC overhead
- tokenizer and application-layer preprocessing cost
- semantic differences in real generated tokens

So the right interpretation is:

- good for system behavior trends
- good for stress-testing SD under async rollout dynamics
- not a final replacement for production benchmarking

## Extending The Framework

Natural next steps:

- integrate NVML sampling for SM / memory utilization traces
- add per-step trace export to CSV or JSONL
- add better packing policies
- add a more vLLM-like admission policy
- add explicit TTFT and TPOT metrics
- add speculative verify implementations beyond probabilistic acceptance
- plug in your history-based SD path from `engine/sd_core.py`

## Current Status

The repository currently supports:

- online scheduling loop
- async rollout-like workload generation
- KV-aware dynamic batching
- proxy compute mode
- real HF online compute mode
- offline calibration mode
- acceptance-driven request updates
- summary metrics for throughput, latency, backlog, and memory

## Validation

Local test status in the current workspace:

```bash
pytest sim/test_simulator.py -q
```

Current result:

```text
26 passed
```

## Practical Advice

If your immediate goal is to decide whether SD is risky for async rollout
serving, the most convincing workflow is:

1. use this simulator in `real_hf` mode
2. sweep `chunk_size`, `accept_rate`, and `arrival_rate`
3. compare against `SD off`
4. identify the region where throughput gain disappears or backlog grows

That will give you a strong answer quickly, even before you invest in a full
serving-system implementation.
