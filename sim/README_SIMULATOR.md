# Batch Request + Speculative Decoding (SD) Rollout Simulator

A comprehensive discrete-event simulator for modeling LLM serving systems with dynamic batching and speculative decoding. This simulator helps research and understand the performance characteristics of batch serving with SD token acceptance strategy.

## Overview

This simulator models a queuing system with the following characteristics:
- **Memory Constraint**: KV cache memory budget limits concurrent requests
- **Probabilistic Rejection**: Speculative decoding with configurable acceptance rates
- **Dynamic Batching**: Continuous batching scheduler selects requests based on available resources
- **Discrete Time**: Event-driven simulation with configurable time steps

## System Architecture

```
┌─────────────────┐
│ Request         │  Generates requests with:
│ Generator       │  - Prompt length
│                 │  - Max generation length
│                 │  - Arrival time (Poisson process)
│                 │  - SD accept rate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch           │  Selects active requests:
│ Scheduler       │  - Respects memory constraints
│                 │  - Greedy: prioritize short jobs
│                 │  - Max batch size limit
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compute         │  Estimates:
│ Model (HF       │  - Latency per compute step
│ Proxy)          │  - Throughput
│                 │  - GPU utilization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SD Accepter     │  Token acceptance logic:
│                 │  - Accept tokens with probability
│                 │  - Stop on rejection (normal SD behavior)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ KV Cache        │  Memory tracking:
│ Manager         │  - Allocate cache for new requests
│                 │  - Update on sequence growth
│                 │  - Release on completion
└────────┬────────┘
         │
         ▼
    [Queue Update]
```

## Files

### Core Simulator
- **`simulator.py`**: Main simulation engine with all components
  - `RequestGenerator`: Poisson arrival process
  - `KVCacheManager`: Memory management with paging
  - `ComputeModel`: Performance estimation for HF models
  - `SDAccepter`: Token-level acceptance logic
  - `BatchScheduler`: Request scheduling with constraints
  - `BatchSDSimulator`: Main orchestrator

### Experiments
- **`experiments.py`**: Experiment runners for systematic evaluation
  - Parameter sweeps (accept rate, batch size, chunk size, load)
  - Baseline vs SD comparison
  - Customizable experiment harness

### Visualization
- **`visualizer.py`**: Analysis and visualization tools
  - Throughput curves
  - Memory utilization plots
  - Performance dashboards
  - PNG export for reports

## Quick Start

### 1. Run Basic Simulation

```bash
python simulator.py
```

This runs a 30-second simulation with default parameters:
- 10 requests/sec arrival rate
- Batch size: 16
- Chunk size: 4 (tokens per step)
- SD accept rate: 0.85

Expected output:
```
Starting simulation for 30.0 seconds
================================================================================
[0.000s] Request 0 arrived (prompt=45, max_tokens=210, accept_rate=0.82)
[0.000s] Request 1 arrived (prompt=52, max_tokens=195, accept_rate=0.87)
...
================================================================================
SIMULATION SUMMARY
================================================================================
Completed Requests: 245
Total Tokens Generated: 45230
Total Tokens Accepted: 38445
SD Acceptance Rate: 85.04%
Overall Throughput: 1850.5 tokens/sec
```

### 2. Run Comprehensive Experiments

```bash
python experiments.py
```

This runs five experiment suites:
1. **SD Speedup Curve**: Accept rate sweep (0.5 to 1.0)
2. **Batch Size Impact**: Sweep batch sizes (4, 8, 16, 32)
3. **Chunk Size Granularity**: Verify granularity tradeoffs
4. **Load Impact**: Arrival rate sweep (5 to 40 req/sec)
5. **Baseline Comparison**: With vs without SD

### 3. Generate Visualizations

```bash
python visualizer.py
```

Generates performance dashboards and comparison plots (PNG format):
- `throughput_vs_accept_rate.png`: SD quality impact
- `batch_size_scaling.png`: Batch size scaling curves
- `chunk_size_granularity.png`: Verification granularity tradeoff
- `load_sensitivity.png`: System behavior under load
- `memory_pressure.png`: KV cache utilization
- `summary_dashboard.png`: Comprehensive dashboard

## Configuration

### System Parameters (SystemConfig)

```python
config = SystemConfig(
    # Model configuration
    model_hidden_size=1152,      # Qwen2.5-1.5B
    num_layers=28,
    
    # Memory configuration
    gpu_memory_mb=12000,          # Total GPU memory
    model_weights_mb=3000,        # Model weights size
    kv_budget_mb=6000,            # Available for KV cache (auto-calculated)
    
    # Batching configuration
    max_batch_size=32,            # Max requests per batch
    chunk_size=4,                 # Tokens per generation step
    
    # Workload configuration
    arrival_rate=10.0,            # Requests per second
    avg_prompt_len=50,            # Average prompt tokens
    avg_max_tokens=200,           # Average generation length
    avg_accept_rate=0.85,         # SD token acceptance rate
)
```

### Custom Simulation

```python
from simulator import BatchSDSimulator, SystemConfig

# Create custom configuration
config = SystemConfig(
    arrival_rate=20.0,           # Higher load
    max_batch_size=32,           # Larger batches
    chunk_size=8,                # Larger chunk
    avg_accept_rate=0.9,         # Better SD quality
    gpu_memory_mb=20000,         # More memory
)

# Run simulation
simulator = BatchSDSimulator(config, seed=42)
simulator.run_simulation(duration_seconds=60.0)
```

## Key Metrics

The simulator tracks and reports:

### Request Metrics
- **Completed Requests**: Successfully processed requests
- **Request Latency**: Time from arrival to completion
- **Queue Size**: Active requests awaiting processing

### Token Metrics
- **Total Tokens Generated**: Proposed tokens from SD
- **Total Tokens Accepted**: Tokens confirmed by acceptance
- **SD Acceptance Rate**: Ratio of accepted to generated tokens
- **Overall Throughput**: Tokens/sec (accepted tokens only)

### System Metrics
- **Batch Count**: Number of batches processed
- **KV Cache Utilization**: Memory usage percentage
- **Average Batch Latency**: Time per batch step
- **GPU Utilization**: Model compute efficiency (estimate)

## Performance Characteristics

### Expected Results (Default Config)
- **Throughput**: 1,500-2,000 tokens/sec
- **SD Acceptance Rate**: 80-90% (depends on quality)
- **Latency**: 50-200ms per request
- **Memory Utilization**: 60-85% of KV budget

### Scalability Observations
1. **Batch Size**: Near-linear throughput improvement up to batch=32
2. **Accept Rate**: Lower acceptance (poor SD) → reduced effective throughput
3. **Chunk Size**: Larger chunks reduce overhead but increase rejection likelihood
4. **Memory**: System saturates when KV budget exceeded (blocking new requests)

## Research Questions

The simulator helps answer:

1. **What is the optimal batch size for my hardware/model?**
   - Run `experiments.py` batch size sweep

2. **How does SD quality (accept_rate) affect throughput?**
   - Run `experiments.py` accept rate sweep → generates speedup curves

3. **What is the best chunk size for verification?**
   - Trade-off: Larger chunks → better compute efficiency, but higher rejection
   - Run `experiments.py` chunk size sweep

4. **What happens under high load?**
   - Run `experiments.py` arrival rate sweep
   - Check KV cache saturation point

5. **What's the speedup from using SD?**
   - Run baseline comparison with chunk_size=1, accept_rate=1.0 vs your config

## Advanced Usage

### Custom Request Patterns

```python
# Create custom request generator
class CustomRequestGenerator(RequestGenerator):
    def generate_request(self, current_time):
        # Your custom distribution logic
        return Request(...)

# Use in simulator
simulator.request_gen = CustomRequestGenerator(config)
simulator.run_simulation(30.0)
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

simulator.run_simulation(30.0)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Sensitivity Analysis

```python
# Sweep multiple parameters
for batch_size in [8, 16, 32]:
    for accept_rate in [0.7, 0.85, 1.0]:
        config = SystemConfig(
            max_batch_size=batch_size,
            avg_accept_rate=accept_rate
        )
        sim = BatchSDSimulator(config)
        sim.run_simulation(20.0)
        # Log results
```

## Limitations & Future Work

### Current Limitations
1. **Simplified Compute Model**: Linear throughput scaling (real GPU has non-linear behavior)
2. **No Prefill/Decode Distinction**: Treats all compute uniformly (no prefill batching)
3. **No Attention Pattern Impact**: Ignores sequence length impact on attention
4. **Simple Scheduling**: Greedy job selection (no FIFO or priority queues)
5. **No Token Variance**: All sequences generate same chunk size

### Potential Enhancements
- [ ] Realistic GPU performance model (roofline model)
- [ ] Separate prefill vs decode phases
- [ ] Variable chunk sizes per request
- [ ] FCFS/priority scheduling policies
- [ ] Preemption and request swapping
- [ ] Multi-GPU simulation
- [ ] Exact throughput benchmarking integration
- [ ] Token-level latency tracking (time-to-first-token)

## Troubleshooting

### Issue: Low throughput
- Check batch size (should be > 8 for good efficiency)
- Verify KV budget not exceeded
- Check SD acceptance rate (too low means wasted compute)

### Issue: High latency
- Reduce arrival_rate (system overloaded)
- Increase GPU memory or batch scheduling window
- Check for request rejection due to memory

### Issue: Requests not completing
- Check if max_new_tokens is reasonable
- Verify accept_rate > 0 (tokens must be accepted)
- Check simulator duration is long enough

## Citation

If you use this simulator in research, please cite:

```bibtex
@misc{sd_batch_simulator,
  title={Batch Request + Speculative Decoding LLM Serving Simulator},
  year={2024}
}
```

## License

MIT

## Questions?

Check the README.md in the parent directory for project context and related work.
