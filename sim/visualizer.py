"""Visualization utilities for experiment summaries and time-series traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


class SimulationVisualizer:
    """Generate comparison figures and time-series plots from saved raw data."""

    def __init__(self, output_dir: str = "sim_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename: str) -> Path:
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_experiment_comparisons(self, results_path: str) -> List[Path]:
        results = _load_json(results_path)
        grouped = defaultdict(list)
        for item in results:
            name = item["name"].split(":")[0]
            grouped[name].append(item)

        saved: List[Path] = []
        for group_name, items in grouped.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = axes.flatten()

            def _sort_key(item):
                suffix = item["name"].split(":", 1)[1]
                try:
                    return float(suffix)
                except ValueError:
                    return str(suffix)

            items = sorted(items, key=_sort_key)
            x_labels = [item["name"].split(":", 1)[1] for item in items]
            throughput = [item["metrics"]["throughput_tokens_per_sec"] for item in items]
            latency = [item["metrics"]["avg_request_latency_sec"] for item in items]
            stability = [item["metrics"]["stability_ratio"] for item in items]
            kv_peak = [item["metrics"]["peak_kv_utilization"] for item in items]
            speedup = [
                (item.get("delta_vs_baseline") or {}).get("throughput_speedup", 1.0)
                for item in items
            ]

            axes[0].plot(x_labels, throughput, marker="o", linewidth=2)
            axes[0].set_title(f"{group_name}: Throughput")
            axes[0].set_ylabel("tokens/sec")
            axes[0].grid(alpha=0.3)

            axes[1].plot(x_labels, latency, marker="o", color="tab:red", linewidth=2)
            axes[1].set_title(f"{group_name}: Avg Latency")
            axes[1].set_ylabel("sec")
            axes[1].grid(alpha=0.3)

            axes[2].plot(x_labels, stability, marker="o", color="tab:green", linewidth=2)
            axes[2].set_title(f"{group_name}: Stability")
            axes[2].set_ylabel("ratio")
            axes[2].grid(alpha=0.3)

            axes[3].plot(x_labels, speedup, marker="o", color="tab:purple", linewidth=2, label="speedup vs baseline")
            axes[3].plot(x_labels, kv_peak, marker="x", color="tab:orange", linewidth=2, label="peak kv util")
            axes[3].set_title(f"{group_name}: Baseline Delta / KV")
            axes[3].legend()
            axes[3].grid(alpha=0.3)

            for ax in axes:
                ax.tick_params(axis="x", rotation=25)

            saved.append(self._save(fig, f"{_slugify(group_name)}_comparison.png"))

        return saved

    def plot_summary_vs_baseline(self, results_path: str) -> Path:
        results = _load_json(results_path)
        names = [item["name"] for item in results]
        speedup = [(item.get("delta_vs_baseline") or {}).get("throughput_speedup", 1.0) for item in results]
        latency_change = [
            (item.get("delta_vs_baseline") or {}).get("latency_change_pct", 0.0) * 100.0
            for item in results
        ]
        queue_change = [
            (item.get("delta_vs_baseline") or {}).get("queue_wait_change_pct", 0.0) * 100.0
            for item in results
        ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 11))
        axes[0].bar(names, speedup, color="steelblue")
        axes[0].axhline(1.0, linestyle="--", color="black", linewidth=1)
        axes[0].set_title("Throughput Speedup vs Baseline")
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(names, latency_change, color="indianred")
        axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1)
        axes[1].set_title("Average Latency Change vs Baseline (%)")
        axes[1].grid(alpha=0.3, axis="y")

        axes[2].bar(names, queue_change, color="darkseagreen")
        axes[2].axhline(0.0, linestyle="--", color="black", linewidth=1)
        axes[2].set_title("Queue Wait Change vs Baseline (%)")
        axes[2].grid(alpha=0.3, axis="y")

        for ax in axes:
            ax.tick_params(axis="x", rotation=35)

        return self._save(fig, "baseline_comparison_overview.png")

    def plot_window_timeseries(self, window_metrics_path: str, prefix: str = "window") -> List[Path]:
        windows = _load_json(window_metrics_path)
        if not windows:
            return []

        x = [item["window_end"] for item in windows]
        saved: List[Path] = []

        metric_groups = [
            (
                "throughput_and_acceptance.png",
                [
                    ("throughput_tokens_per_sec", "throughput"),
                    ("accepted_tokens_per_sec", "accepted"),
                    ("rejected_tokens_per_sec", "rejected"),
                    ("fallback_tokens_per_sec", "fallback"),
                ],
                "tokens/sec",
            ),
            (
                "queue_and_batch.png",
                [
                    ("avg_queue_size", "queue"),
                    ("avg_active_requests", "active"),
                    ("avg_batch_size", "batch"),
                ],
                "count",
            ),
            (
                "latency_and_kv.png",
                [
                    ("avg_step_latency_ms", "step latency"),
                    ("avg_kv_utilization", "avg kv util"),
                    ("peak_kv_utilization", "peak kv util"),
                ],
                "mixed",
            ),
            (
                "gpu_and_memory.png",
                [
                    ("avg_gpu_utilization", "gpu util"),
                    ("avg_memory_allocated_mb", "avg mem"),
                    ("peak_memory_allocated_mb", "peak mem"),
                ],
                "mixed",
            ),
        ]

        for filename, metrics, ylabel in metric_groups:
            fig, ax = plt.subplots(figsize=(12, 5))
            for metric_name, label in metrics:
                ax.plot(x, [item.get(metric_name, 0.0) for item in windows], marker="o", linewidth=2, label=label)
            ax.set_xlabel("time (sec)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{prefix}: {filename.replace('_', ' ').replace('.png', '')}")
            ax.grid(alpha=0.3)
            ax.legend()
            saved.append(self._save(fig, f"{prefix}_{filename}"))

        return saved

    def plot_step_trace_timeseries(self, trace_path: str, prefix: str = "step") -> List[Path]:
        traces = _load_json(trace_path)
        if not traces:
            return []

        x = [item["end_time"] for item in traces]
        saved: List[Path] = []

        configs = [
            (
                "step_batch_and_queue.png",
                [("batch_size", "batch"), ("queued_requests", "queue"), ("active_requests", "active")],
                "count",
            ),
            (
                "step_latency.png",
                [("prefill_latency_ms", "prefill"), ("decode_latency_ms", "decode"), ("total_latency_ms", "total")],
                "ms",
            ),
            (
                "step_kv_and_gpu.png",
                [("kv_utilization", "kv util"), ("peak_kv_utilization", "peak kv"), ("gpu_utilization", "gpu util")],
                "mixed",
            ),
            (
                "step_memory.png",
                [("memory_allocated_mb", "mem"), ("peak_memory_allocated_mb", "peak mem")],
                "MB",
            ),
        ]

        for filename, metrics, ylabel in configs:
            fig, ax = plt.subplots(figsize=(12, 5))
            for metric_name, label in metrics:
                ax.plot(x, [item.get(metric_name, 0.0) for item in traces], linewidth=1.5, label=label)
            ax.set_xlabel("time (sec)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{prefix}: {filename.replace('_', ' ').replace('.png', '')}")
            ax.grid(alpha=0.3)
            ax.legend()
            saved.append(self._save(fig, f"{prefix}_{filename}"))

        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from saved simulator raw data.")
    parser.add_argument("--results-json", help="Path to sim_results.json")
    parser.add_argument("--window-json", help="Path to window_metrics.json")
    parser.add_argument("--trace-json", help="Path to step_trace.json")
    parser.add_argument("--output-dir", default="sim_plots")
    parser.add_argument("--prefix", default="run")
    args = parser.parse_args()

    visualizer = SimulationVisualizer(output_dir=args.output_dir)
    saved: List[Path] = []

    if args.results_json:
        saved.extend(visualizer.plot_experiment_comparisons(args.results_json))
        saved.append(visualizer.plot_summary_vs_baseline(args.results_json))
    if args.window_json:
        saved.extend(visualizer.plot_window_timeseries(args.window_json, prefix=args.prefix))
    if args.trace_json:
        saved.extend(visualizer.plot_step_trace_timeseries(args.trace_json, prefix=args.prefix))

    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
