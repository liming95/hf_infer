"""Quick presets for common simulator scenarios."""

from __future__ import annotations

import sys

from simulator import BatchSDSimulator, SystemConfig


def run_scenario(name: str, config: SystemConfig, duration: float = 30.0) -> None:
    print("\n" + "=" * 80)
    print(f"SCENARIO: {name}")
    print("=" * 80)
    summary = BatchSDSimulator(config, seed=42).run_simulation(duration_seconds=duration, verbose=True)
    print(f"Scenario throughput: {summary['throughput_tokens_per_sec']:.2f} tokens/sec")


def scenario_light_load() -> None:
    run_scenario(
        "Light Load",
        SystemConfig(
            arrival_rate=5.0,
            max_batch_size=8,
            max_concurrent_requests=24,
            chunk_size=4,
            avg_accept_rate=0.85,
            workload_mode="poisson",
        ),
    )


def scenario_async_rollout() -> None:
    run_scenario(
        "Async RLHF Rollout",
        SystemConfig(
            arrival_rate=16.0,
            max_batch_size=16,
            max_concurrent_requests=64,
            chunk_size=4,
            avg_accept_rate=0.83,
            workload_mode="rollout_burst",
            rollout_burst_size=12,
        ),
    )


def scenario_high_quality_sd() -> None:
    run_scenario(
        "High Quality Speculative",
        SystemConfig(
            arrival_rate=10.0,
            max_batch_size=16,
            max_concurrent_requests=48,
            chunk_size=4,
            avg_accept_rate=0.95,
            workload_mode="mixed",
        ),
    )


def scenario_low_quality_sd() -> None:
    run_scenario(
        "Low Quality Speculative",
        SystemConfig(
            arrival_rate=10.0,
            max_batch_size=16,
            max_concurrent_requests=48,
            chunk_size=8,
            avg_accept_rate=0.60,
            workload_mode="mixed",
        ),
    )


def scenario_memory_constrained() -> None:
    run_scenario(
        "Memory Constrained",
        SystemConfig(
            gpu_memory_mb=6000,
            max_batch_size=8,
            max_concurrent_requests=16,
            arrival_rate=10.0,
            chunk_size=4,
            avg_accept_rate=0.85,
            workload_mode="mixed",
        ),
    )


def scenario_compare() -> None:
    print("\n" + "=" * 80)
    print("SCENARIO: SD vs No-SD")
    print("=" * 80)

    baseline = BatchSDSimulator(
        SystemConfig(
            arrival_rate=10.0,
            max_batch_size=16,
            max_concurrent_requests=48,
            chunk_size=1,
            avg_accept_rate=1.0,
            enable_speculative=False,
            workload_mode="mixed",
        ),
        seed=42,
    ).run_simulation(duration_seconds=20.0, verbose=False)

    sd = BatchSDSimulator(
        SystemConfig(
            arrival_rate=10.0,
            max_batch_size=16,
            max_concurrent_requests=48,
            chunk_size=4,
            avg_accept_rate=0.85,
            enable_speculative=True,
            workload_mode="mixed",
        ),
        seed=42,
    ).run_simulation(duration_seconds=20.0, verbose=False)

    speedup = (
        sd["throughput_tokens_per_sec"] / baseline["throughput_tokens_per_sec"]
        if baseline["throughput_tokens_per_sec"] > 0
        else 0.0
    )

    print(f"Baseline throughput: {baseline['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"SD throughput:       {sd['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Baseline latency:    {baseline['avg_request_latency_sec']:.3f}s")
    print(f"SD latency:          {sd['avg_request_latency_sec']:.3f}s")


PRESETS = {
    "light": scenario_light_load,
    "async_rollout": scenario_async_rollout,
    "high_quality_sd": scenario_high_quality_sd,
    "low_quality_sd": scenario_low_quality_sd,
    "memory_constrained": scenario_memory_constrained,
    "compare": scenario_compare,
}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in PRESETS:
        PRESETS[sys.argv[1]]()
        return

    print("Usage: python quickstart.py [scenario]")
    print(f"Available scenarios: {', '.join(PRESETS.keys())}")
    for name, fn in PRESETS.items():
        try:
            fn()
        except Exception as exc:
            print(f"Scenario {name} failed: {exc}")


if __name__ == "__main__":
    main()
