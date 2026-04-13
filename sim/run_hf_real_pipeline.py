"""One-shot pipeline for HF-real experiments, raw data export, and plotting.

This script creates a self-contained results directory under ``sim/results/``
and stores:
- one online run with raw traces
- one chunk-size sweep with baseline comparison
- markdown/csv/json summaries
- comparison plots and time-series plots
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from experiments import ExperimentRunner
from simulator import BatchSDSimulator, SystemConfig
from visualizer import SimulationVisualizer

SIM_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full HF-real experiment pipeline into sim/results/.")
    parser.add_argument("--run-name")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--chunk-sweep", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--accept-rate", type=float, default=0.85)
    parser.add_argument("--avg-prompt-len", type=int, default=50)
    parser.add_argument("--avg-max-tokens", type=int, default=200)
    parser.add_argument("--workload-mode", default="mixed", choices=["poisson", "rollout_burst", "mixed"])
    parser.add_argument("--rollout-burst-size", type=int, default=8)
    parser.add_argument("--gpu-memory-mb", type=float, default=12000)
    parser.add_argument("--use-real-compute", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _default_run_name(prefix: str = "hf_real") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_config(args, chunk_size: int, enable_speculative: bool) -> SystemConfig:
    return SystemConfig(
        arrival_rate=args.arrival_rate,
        max_batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrent,
        chunk_size=chunk_size,
        avg_accept_rate=args.accept_rate if enable_speculative else 1.0,
        avg_prompt_len=args.avg_prompt_len,
        avg_max_tokens=args.avg_max_tokens,
        workload_mode=args.workload_mode,
        rollout_burst_size=args.rollout_burst_size,
        gpu_memory_mb=args.gpu_memory_mb,
        enable_speculative=enable_speculative,
        use_real_compute=args.use_real_compute,
        model_name_or_path=args.model,
        real_compute_device=args.device,
        real_compute_dtype=args.dtype,
    )


def run_pipeline(args) -> Path:
    run_name = args.run_name or _default_run_name()
    result_dir = SIM_DIR / "results" / run_name
    plot_dir = result_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "mode": "real_hf" if args.use_real_compute else "proxy",
        "config": vars(args),
    }
    _write_json(result_dir / "manifest.json", manifest)

    online_config = _base_config(args, chunk_size=args.chunk_size, enable_speculative=(args.chunk_size > 1))
    simulator = BatchSDSimulator(online_config, seed=args.seed)
    summary = simulator.run_simulation(duration_seconds=args.duration, verbose=False)
    windowed_metrics = simulator.get_windowed_metrics(window_sec=args.window_sec)

    _write_json(result_dir / "online_summary.json", summary)
    _write_json(result_dir / "online_window_metrics.json", windowed_metrics)
    simulator.export_step_traces(str(result_dir / "online_step_trace.json"))
    simulator.export_oom_events(str(result_dir / "online_oom_events.json"))

    runner = ExperimentRunner(seed=args.seed)
    runner._run_sweep(
        "Chunk Size Granularity",
        args.chunk_sweep,
        lambda cs: _base_config(args, chunk_size=cs, enable_speculative=(cs > 1)),
        duration=args.duration,
    )
    runner.save_results(str(result_dir / "chunk_results.json"))
    runner.save_markdown_report(str(result_dir / "chunk_report.md"))
    runner.save_csv_summary(str(result_dir / "chunk_summary.csv"))

    overview = {
        "online_summary": summary,
        "chunk_sweep_best_throughput": max(
            runner.results,
            key=lambda item: item["metrics"]["throughput_tokens_per_sec"],
        )["name"] if runner.results else None,
        "chunk_sweep_best_speedup": max(
            runner.results,
            key=lambda item: (item.get("delta_vs_baseline") or {}).get("throughput_speedup", 0.0),
        )["name"] if runner.results else None,
    }
    _write_json(result_dir / "overview.json", overview)

    visualizer = SimulationVisualizer(output_dir=str(plot_dir))
    visualizer.plot_experiment_comparisons(str(result_dir / "chunk_results.json"))
    visualizer.plot_chunk_sweep_focus(str(result_dir / "chunk_results.json"))
    visualizer.plot_summary_vs_baseline(str(result_dir / "chunk_results.json"))
    visualizer.plot_window_timeseries(str(result_dir / "online_window_metrics.json"), prefix="online")
    visualizer.plot_step_trace_timeseries(str(result_dir / "online_step_trace.json"), prefix="online")

    return result_dir


def main() -> None:
    args = build_parser().parse_args()
    result_dir = run_pipeline(args)
    print(f"Saved full pipeline outputs to: {result_dir}")


if __name__ == "__main__":
    main()
