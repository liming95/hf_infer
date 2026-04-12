"""Experiment runners for the async rollout + speculative decoding simulator."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

from simulator import BatchSDSimulator, SystemConfig


class ExperimentRunner:
    """Run repeatable sweeps and collect compact result dictionaries."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results: List[Dict] = []

    def run_experiment(self, name: str, config: SystemConfig, duration: float = 30.0) -> Dict:
        simulator = BatchSDSimulator(config, seed=self.seed)
        summary = simulator.run_simulation(duration_seconds=duration, verbose=False)
        result = {
            "name": name,
            "duration": duration,
            "config": {
                "workload_mode": config.workload_mode,
                "arrival_rate": config.arrival_rate,
                "max_batch_size": config.max_batch_size,
                "max_concurrent_requests": config.max_concurrent_requests,
                "chunk_size": config.chunk_size,
                "avg_accept_rate": config.avg_accept_rate,
                "enable_speculative": config.enable_speculative,
            },
            "metrics": summary,
        }
        self.results.append(result)
        return result

    def _run_sweep(
        self,
        title: str,
        values: Iterable,
        config_builder,
        duration: float = 30.0,
    ) -> List[Dict]:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {title}")
        print("=" * 80)

        results: List[Dict] = []
        for value in values:
            config = config_builder(value)
            result = self.run_experiment(f"{title}:{value}", config, duration)
            metrics = result["metrics"]
            results.append(result)
            print(
                f"value={value} throughput={metrics['throughput_tokens_per_sec']:.2f} "
                f"lat={metrics['avg_request_latency_sec']:.3f}s "
                f"stability={metrics['stability_ratio']:.2%} "
                f"peak_kv={metrics['peak_kv_utilization']:.1f}%"
            )
        return results

    def run_sweep_accept_rate(self, duration: float = 25.0) -> List[Dict]:
        return self._run_sweep(
            "SD Speedup Curve",
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.98],
            lambda ar: SystemConfig(
                arrival_rate=10.0,
                max_batch_size=16,
                max_concurrent_requests=48,
                chunk_size=4,
                avg_accept_rate=ar,
                workload_mode="mixed",
            ),
            duration,
        )

    def run_sweep_batch_size(self, duration: float = 25.0) -> List[Dict]:
        return self._run_sweep(
            "Batch Size Impact",
            [4, 8, 16, 32],
            lambda bs: SystemConfig(
                arrival_rate=10.0,
                max_batch_size=bs,
                max_concurrent_requests=max(3 * bs, 16),
                chunk_size=4,
                avg_accept_rate=0.85,
                workload_mode="mixed",
            ),
            duration,
        )

    def run_sweep_chunk_size(self, duration: float = 25.0) -> List[Dict]:
        return self._run_sweep(
            "Chunk Size Granularity",
            [1, 2, 4, 8],
            lambda cs: SystemConfig(
                arrival_rate=10.0,
                max_batch_size=16,
                max_concurrent_requests=48,
                chunk_size=cs,
                avg_accept_rate=0.85,
                enable_speculative=cs > 1,
                workload_mode="mixed",
            ),
            duration,
        )

    def run_sweep_arrival_rate(self, duration: float = 25.0) -> List[Dict]:
        return self._run_sweep(
            "Load Impact",
            [5.0, 10.0, 20.0, 30.0, 40.0],
            lambda ar: SystemConfig(
                arrival_rate=ar,
                max_batch_size=16,
                max_concurrent_requests=64,
                chunk_size=4,
                avg_accept_rate=0.85,
                workload_mode="mixed",
            ),
            duration,
        )

    def run_stability_boundary(self, duration: float = 30.0) -> List[Dict]:
        points = [
            (8, 2),
            (8, 4),
            (16, 4),
            (16, 8),
            (24, 8),
        ]
        return self._run_sweep(
            "Stability Boundary",
            points,
            lambda item: SystemConfig(
                arrival_rate=20.0,
                max_batch_size=16,
                max_concurrent_requests=64,
                chunk_size=item[1],
                avg_accept_rate=0.82,
                avg_prompt_len=50,
                avg_max_tokens=200,
                page_size=item[0],
                workload_mode="rollout_burst",
            ),
            duration,
        )

    def save_results(self, filename: str = "sim_results.json") -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        print(f"\nSaved results to {filename}")


def baseline_comparison(duration: float = 25.0) -> Dict[str, Dict]:
    print("\n" + "=" * 80)
    print("EXPERIMENT: SD vs Baseline")
    print("=" * 80)

    sd_config = SystemConfig(
        arrival_rate=10.0,
        max_batch_size=16,
        max_concurrent_requests=48,
        chunk_size=4,
        avg_accept_rate=0.85,
        enable_speculative=True,
        workload_mode="mixed",
    )
    baseline_config = SystemConfig(
        arrival_rate=10.0,
        max_batch_size=16,
        max_concurrent_requests=48,
        chunk_size=1,
        avg_accept_rate=1.0,
        enable_speculative=False,
        workload_mode="mixed",
    )

    sd_summary = BatchSDSimulator(sd_config, seed=42).run_simulation(duration, verbose=False)
    baseline_summary = BatchSDSimulator(baseline_config, seed=42).run_simulation(duration, verbose=False)

    speedup = (
        sd_summary["throughput_tokens_per_sec"] / baseline_summary["throughput_tokens_per_sec"]
        if baseline_summary["throughput_tokens_per_sec"] > 0
        else 0.0
    )

    print(f"SD throughput:       {sd_summary['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Baseline throughput: {baseline_summary['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"SD stability:        {sd_summary['stability_ratio']:.2%}")
    print(f"Baseline stability:  {baseline_summary['stability_ratio']:.2%}")

    return {
        "sd": sd_summary,
        "baseline": baseline_summary,
        "speedup": {"throughput_speedup": speedup},
    }


def main() -> None:
    runner = ExperimentRunner(seed=42)
    runner.run_sweep_accept_rate()
    runner.run_sweep_batch_size()
    runner.run_sweep_chunk_size()
    runner.run_sweep_arrival_rate()
    runner.run_stability_boundary()
    baseline_comparison()
    runner.save_results()


if __name__ == "__main__":
    main()
