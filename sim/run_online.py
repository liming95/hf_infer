"""CLI entrypoint for the online async rollout simulator."""

from __future__ import annotations

import argparse
import json

from simulator import BatchSDSimulator, SystemConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run async rollout + SD simulator with proxy or real HF compute.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--accept-rate", type=float, default=0.85)
    parser.add_argument("--avg-prompt-len", type=int, default=50)
    parser.add_argument("--avg-max-tokens", type=int, default=200)
    parser.add_argument("--workload-mode", default="mixed", choices=["poisson", "rollout_burst", "mixed"])
    parser.add_argument("--rollout-burst-size", type=int, default=8)
    parser.add_argument("--gpu-memory-mb", type=float, default=12000)
    parser.add_argument("--use-real-compute", action="store_true")
    parser.add_argument("--model", dest="model_name_or_path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--benchmark-table-path")
    parser.add_argument("--disable-speculative", action="store_true")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--output-json")
    parser.add_argument("--output-trace-json")
    parser.add_argument("--output-window-json")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SystemConfig(
        arrival_rate=args.arrival_rate,
        max_batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrent,
        chunk_size=args.chunk_size,
        avg_accept_rate=args.accept_rate,
        avg_prompt_len=args.avg_prompt_len,
        avg_max_tokens=args.avg_max_tokens,
        workload_mode=args.workload_mode,
        rollout_burst_size=args.rollout_burst_size,
        gpu_memory_mb=args.gpu_memory_mb,
        benchmark_table_path=args.benchmark_table_path,
        enable_speculative=not args.disable_speculative,
        use_real_compute=args.use_real_compute,
        model_name_or_path=args.model_name_or_path,
        real_compute_device=args.device,
        real_compute_dtype=args.dtype,
    )
    simulator = BatchSDSimulator(config, seed=42)
    summary = simulator.run_simulation(duration_seconds=args.duration, verbose=args.verbose)
    windowed_metrics = simulator.get_windowed_metrics(window_sec=args.window_sec)
    payload = {
        "summary": summary,
        "windowed_metrics": windowed_metrics,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    if args.output_trace_json:
        simulator.export_step_traces(args.output_trace_json)
    if args.output_window_json:
        with open(args.output_window_json, "w", encoding="utf-8") as handle:
            json.dump(windowed_metrics, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
