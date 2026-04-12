"""Experiment runners for the async rollout + speculative decoding simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from simulator import BatchSDSimulator, SystemConfig


class ExperimentRunner:
    """Run repeatable sweeps and collect compact result dictionaries."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results: List[Dict] = []

    def run_experiment(
        self,
        name: str,
        config: SystemConfig,
        duration: float = 30.0,
        baseline_config: Optional[SystemConfig] = None,
        window_sec: float = 1.0,
    ) -> Dict:
        simulator = BatchSDSimulator(config, seed=self.seed)
        summary = simulator.run_simulation(duration_seconds=duration, verbose=False)
        windowed_metrics = simulator.get_windowed_metrics(window_sec=window_sec)
        baseline_summary = None
        deltas = None
        if baseline_config is not None:
            baseline_summary = BatchSDSimulator(baseline_config, seed=self.seed).run_simulation(
                duration_seconds=duration,
                verbose=False,
            )
            deltas = self._build_delta(summary, baseline_summary)
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
            "windowed_metrics": windowed_metrics,
            "baseline_metrics": baseline_summary,
            "delta_vs_baseline": deltas,
        }
        self.results.append(result)
        return result

    def _build_delta(self, metrics: Dict, baseline_metrics: Dict) -> Dict[str, float]:
        def pct_change(key: str) -> float:
            base = baseline_metrics.get(key, 0.0)
            current = metrics.get(key, 0.0)
            if abs(base) < 1e-9:
                return 0.0
            return (current - base) / base

        return {
            "throughput_speedup": (
                metrics["throughput_tokens_per_sec"] / baseline_metrics["throughput_tokens_per_sec"]
                if baseline_metrics["throughput_tokens_per_sec"] > 0
                else 0.0
            ),
            "latency_change_pct": pct_change("avg_request_latency_sec"),
            "p95_latency_change_pct": pct_change("p95_request_latency_sec"),
            "queue_wait_change_pct": pct_change("avg_queue_wait_sec"),
            "stability_change_pct": pct_change("stability_ratio"),
            "kv_peak_change_pct": pct_change("peak_kv_utilization"),
        }

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
            baseline_config = self._baseline_from_config(config)
            result = self.run_experiment(
                f"{title}:{value}",
                config,
                duration,
                baseline_config=baseline_config,
            )
            metrics = result["metrics"]
            delta = result["delta_vs_baseline"] or {}
            results.append(result)
            print(
                f"value={value} throughput={metrics['throughput_tokens_per_sec']:.2f} "
                f"lat={metrics['avg_request_latency_sec']:.3f}s "
                f"stability={metrics['stability_ratio']:.2%} "
                f"peak_kv={metrics['peak_kv_utilization']:.1f}% "
                f"speedup={delta.get('throughput_speedup', 0.0):.2f}x"
            )
        return results

    def _baseline_from_config(self, config: SystemConfig) -> SystemConfig:
        return SystemConfig(
            arrival_rate=config.arrival_rate,
            max_batch_size=config.max_batch_size,
            max_concurrent_requests=config.max_concurrent_requests,
            chunk_size=1,
            avg_accept_rate=1.0,
            workload_mode=config.workload_mode,
            rollout_burst_size=config.rollout_burst_size,
            gpu_memory_mb=config.gpu_memory_mb,
            page_size=config.page_size,
            enable_speculative=False,
            benchmark_table_path=config.benchmark_table_path,
            use_real_compute=config.use_real_compute,
            model_name_or_path=config.model_name_or_path,
            real_compute_device=config.real_compute_device,
            real_compute_dtype=config.real_compute_dtype,
        )

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

    def save_markdown_report(self, filename: str = "sim_report.md") -> None:
        lines = [
            "# Simulation Report",
            "",
            "## Metric Guide",
            "",
            "- `throughput_tokens_per_sec`: system-level committed token throughput. Higher is better.",
            "- `avg_request_latency_sec`: average end-to-end request latency. Lower is better.",
            "- `p95_request_latency_sec`: tail latency. This is often more important than average latency.",
            "- `stability_ratio`: completed_requests / arrived_requests. Lower means the system is falling behind.",
            "- `avg_queue_wait_sec`: average waiting time before admission. Higher means scheduling pressure.",
            "- `peak_kv_utilization`: peak KV memory pressure. Near saturation means the scheduler is constrained by memory.",
            "- `throughput_speedup`: SD throughput divided by baseline throughput. Above 1 means SD helps.",
            "",
            "## How To Read The Results",
            "",
            "- If `throughput_speedup > 1` and latency does not grow too much, SD is helping.",
            "- If `throughput_speedup` is near 1 but latency or queue wait increases, SD is likely not worth it.",
            "- If `stability_ratio` drops or `peak_kv_utilization` rises sharply, SD may be hurting scheduling quality.",
            "",
        ]
        for result in self.results:
            metrics = result["metrics"]
            lines.append(f"## {result['name']}")
            lines.append("")
            lines.append(f"- Throughput: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
            lines.append(f"- Avg latency: {metrics['avg_request_latency_sec']:.3f}s")
            lines.append(f"- P95 latency: {metrics['p95_request_latency_sec']:.3f}s")
            lines.append(f"- Stability: {metrics['stability_ratio']:.2%}")
            lines.append(f"- Peak KV: {metrics['peak_kv_utilization']:.2f}%")
            if result.get("delta_vs_baseline"):
                delta = result["delta_vs_baseline"]
                lines.append(f"- Baseline throughput speedup: {delta['throughput_speedup']:.2f}x")
                lines.append(f"- Baseline avg latency change: {delta['latency_change_pct']:.2%}")
                lines.append(f"- Baseline queue wait change: {delta['queue_wait_change_pct']:.2%}")
            if result.get("windowed_metrics"):
                peak_window = max(
                    result["windowed_metrics"],
                    key=lambda item: item["throughput_tokens_per_sec"],
                )
                worst_queue = max(
                    result["windowed_metrics"],
                    key=lambda item: item["avg_queue_size"],
                )
                lines.append(
                    f"- Peak window throughput: {peak_window['throughput_tokens_per_sec']:.2f} tokens/sec "
                    f"at [{peak_window['window_start']:.1f}, {peak_window['window_end']:.1f}]s"
                )
                lines.append(
                    f"- Worst queue window: avg queue {worst_queue['avg_queue_size']:.2f} "
                    f"at [{worst_queue['window_start']:.1f}, {worst_queue['window_end']:.1f}]s"
                )
            lines.append("")
        Path(filename).write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved markdown report to {filename}")

    def save_csv_summary(self, filename: str = "sim_summary.csv") -> None:
        header = [
            "name",
            "chunk_size",
            "accept_rate",
            "arrival_rate",
            "batch_size",
            "throughput_tokens_per_sec",
            "avg_request_latency_sec",
            "p95_request_latency_sec",
            "avg_queue_wait_sec",
            "stability_ratio",
            "peak_kv_utilization",
            "throughput_speedup",
            "latency_change_pct",
        ]
        rows = [",".join(header)]
        for result in self.results:
            config = result["config"]
            metrics = result["metrics"]
            delta = result.get("delta_vs_baseline") or {}
            values = [
                result["name"],
                str(config["chunk_size"]),
                str(config["avg_accept_rate"]),
                str(config["arrival_rate"]),
                str(config["max_batch_size"]),
                f"{metrics['throughput_tokens_per_sec']:.6f}",
                f"{metrics['avg_request_latency_sec']:.6f}",
                f"{metrics['p95_request_latency_sec']:.6f}",
                f"{metrics['avg_queue_wait_sec']:.6f}",
                f"{metrics['stability_ratio']:.6f}",
                f"{metrics['peak_kv_utilization']:.6f}",
                f"{delta.get('throughput_speedup', 0.0):.6f}",
                f"{delta.get('latency_change_pct', 0.0):.6f}",
            ]
            rows.append(",".join(values))
        Path(filename).write_text("\n".join(rows), encoding="utf-8")
        print(f"Saved CSV summary to {filename}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment sweeps for the async rollout simulator.")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "accept", "batch", "chunk", "arrival", "stability"],
    )
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--output-json", default="sim_results.json")
    parser.add_argument("--output-md", default="sim_report.md")
    parser.add_argument("--output-csv", default="sim_summary.csv")
    parser.add_argument("--use-real-compute", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=48)
    parser.add_argument("--accept-rate", type=float, default=0.85)
    parser.add_argument("--workload-mode", default="mixed", choices=["poisson", "rollout_burst", "mixed"])
    return parser


def _override_runner_configs(runner: ExperimentRunner, args) -> None:
    original_baseline = runner._baseline_from_config

    def wrapped_baseline(config: SystemConfig) -> SystemConfig:
        cfg = original_baseline(config)
        cfg.use_real_compute = args.use_real_compute
        cfg.model_name_or_path = args.model
        cfg.real_compute_device = args.device
        cfg.real_compute_dtype = args.dtype
        return cfg

    runner._baseline_from_config = wrapped_baseline


def main() -> None:
    args = build_parser().parse_args()
    runner = ExperimentRunner(seed=42)
    _override_runner_configs(runner, args)

    if args.suite in ("all", "accept"):
        runner.run_sweep_accept_rate(duration=args.duration)
    if args.suite in ("all", "batch"):
        runner.run_sweep_batch_size(duration=args.duration)
    if args.suite in ("all", "chunk"):
        runner._run_sweep(
            "Chunk Size Granularity",
            [1, 2, 4, 8],
            lambda cs: SystemConfig(
                arrival_rate=args.arrival_rate,
                max_batch_size=args.batch_size,
                max_concurrent_requests=args.max_concurrent,
                chunk_size=cs,
                avg_accept_rate=args.accept_rate,
                enable_speculative=cs > 1,
                workload_mode=args.workload_mode,
                use_real_compute=args.use_real_compute,
                model_name_or_path=args.model,
                real_compute_device=args.device,
                real_compute_dtype=args.dtype,
            ),
            args.duration,
        )
    if args.suite in ("all", "arrival"):
        runner.run_sweep_arrival_rate(duration=args.duration)
    if args.suite in ("all", "stability"):
        runner.run_stability_boundary(duration=args.duration)
    if args.suite == "all":
        baseline_comparison(duration=args.duration)

    runner.save_results(args.output_json)
    runner.save_markdown_report(args.output_md)
    runner.save_csv_summary(args.output_csv)


if __name__ == "__main__":
    main()
