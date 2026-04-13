"""Rollout-pull real-HF sweep over batch, generation length, chunk, and SD rate."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

from simulator import BatchSDSimulator, SystemConfig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM_DIR = Path(__file__).resolve().parent


def _int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _default_run_name() -> str:
    return f"rollout_hf_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rollout_pull + real HF sweeps for batch/max_tokens/chunk/SD acceptance rate."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--proxy-debug", action="store_true", help="Use proxy executor for local plotting/debug only.")
    parser.add_argument("--target-completed-requests", type=int, default=64)
    parser.add_argument("--duration", type=float, default=None, help="Optional safety cap. Usually omit for rollout mode.")
    parser.add_argument("--window-sec", type=float, default=1.0)

    parser.add_argument("--prompt-len", type=int, default=1024, help="Backward-compatible single prompt length.")
    parser.add_argument("--prompt-lens", default=None, help="Comma-separated prompt lengths, e.g. 256,1024,2048.")
    parser.add_argument("--batch-sizes", default="4,8,16")
    parser.add_argument("--max-token-lens", default="256,1024,2048")
    parser.add_argument("--chunk-sizes", default="1,2,4,8,16")
    parser.add_argument("--sd-rates", default="0.60,0.75,0.90")

    parser.add_argument("--rollout-pull-batch-size", type=int, default=8)
    parser.add_argument("--rollout-pull-target-outstanding", type=int, default=32)
    parser.add_argument("--max-concurrent-factor", type=int, default=2)
    parser.add_argument("--gpu-memory-mb", type=float, default=48000)
    parser.add_argument("--compute-memory-margin-mb", type=float, default=4096)
    parser.add_argument("--chunk1-as-baseline", action="store_true", default=True)
    parser.add_argument("--random-lengths", action="store_true", help="Use Poisson prompt/max-token sampling instead of fixed sweep lengths.")

    parser.add_argument("--limit", type=int, help="Run only the first N configurations.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def iter_configs(args):
    batch_sizes = _int_list(args.batch_sizes)
    prompt_lens = _int_list(args.prompt_lens) if args.prompt_lens else [args.prompt_len]
    max_token_lens = _int_list(args.max_token_lens)
    chunk_sizes = _int_list(args.chunk_sizes)
    sd_rates = _float_list(args.sd_rates)

    configs = []
    for batch_size in batch_sizes:
        for prompt_len in prompt_lens:
            for max_tokens in max_token_lens:
                for chunk_size in chunk_sizes:
                    rates = [1.0] if chunk_size == 1 and args.chunk1_as_baseline else sd_rates
                    for sd_rate in rates:
                        configs.append(
                            {
                                "batch_size": batch_size,
                                "prompt_len": prompt_len,
                                "max_tokens": max_tokens,
                                "chunk_size": chunk_size,
                                "sd_rate": sd_rate,
                                "enable_speculative": chunk_size > 1,
                            }
                        )
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def build_system_config(args, item: dict) -> SystemConfig:
    batch_size = item["batch_size"]
    return SystemConfig(
        workload_mode="rollout_pull",
        rollout_pull_batch_size=args.rollout_pull_batch_size,
        rollout_pull_target_outstanding=args.rollout_pull_target_outstanding,
        max_batch_size=batch_size,
        max_concurrent_requests=max(batch_size, batch_size * args.max_concurrent_factor),
        chunk_size=item["chunk_size"],
        avg_accept_rate=item["sd_rate"],
        avg_prompt_len=item["prompt_len"],
        avg_max_tokens=item["max_tokens"],
        fixed_prompt_len=not args.random_lengths,
        fixed_max_tokens=not args.random_lengths,
        enable_speculative=item["enable_speculative"],
        gpu_memory_mb=args.gpu_memory_mb,
        compute_memory_margin_mb=args.compute_memory_margin_mb,
        use_real_compute=not args.proxy_debug,
        model_name_or_path=args.model,
        real_compute_device=args.device,
        real_compute_dtype=args.dtype,
    )


def estimate_decode_steps(args, item: dict) -> int:
    chunk = max(1, item["chunk_size"])
    batch = max(1, item["batch_size"])
    total_decode_tokens = args.target_completed_requests * item["max_tokens"]
    return max(1, (total_decode_tokens + batch * chunk - 1) // (batch * chunk))


def flatten_summary(item: dict, summary: dict) -> dict:
    return {
        "batch_size": item["batch_size"],
        "prompt_len": item["prompt_len"],
        "max_tokens": item["max_tokens"],
        "chunk_size": item["chunk_size"],
        "sd_rate": item["sd_rate"],
        "enable_speculative": item["enable_speculative"],
        "throughput_tokens_per_sec": summary["throughput_tokens_per_sec"],
        "avg_request_latency_sec": summary["avg_request_latency_sec"],
        "p95_request_latency_sec": summary["p95_request_latency_sec"],
        "avg_queue_wait_sec": summary["avg_queue_wait_sec"],
        "stability_ratio": summary["stability_ratio"],
        "peak_kv_utilization": summary["peak_kv_utilization"],
        "avg_batch_size": summary["avg_batch_size"],
        "draft_acceptance_rate": summary["draft_acceptance_rate"],
        "fallback_share": summary["fallback_share"],
        "oom_events_count": summary["oom_events_count"],
        "oom_retry_count": summary["oom_retry_count"],
        "avg_executed_chunk_size": summary["avg_executed_chunk_size"],
        "completed_requests": summary["completed_requests"],
        "arrived_requests": summary["arrived_requests"],
        "simulation_time": summary["simulation_time"],
        "stop_condition": summary["stop_condition"],
        "status": "ok",
        "error": "",
    }


def failed_row(item: dict, error: Exception) -> dict:
    return {
        "batch_size": item["batch_size"],
        "prompt_len": item.get("prompt_len"),
        "max_tokens": item["max_tokens"],
        "chunk_size": item["chunk_size"],
        "sd_rate": item["sd_rate"],
        "enable_speculative": item["enable_speculative"],
        "throughput_tokens_per_sec": 0.0,
        "avg_request_latency_sec": 0.0,
        "p95_request_latency_sec": 0.0,
        "avg_queue_wait_sec": 0.0,
        "stability_ratio": 0.0,
        "peak_kv_utilization": 0.0,
        "avg_batch_size": 0.0,
        "draft_acceptance_rate": 0.0,
        "fallback_share": 0.0,
        "oom_events_count": 0,
        "oom_retry_count": 0,
        "avg_executed_chunk_size": 0.0,
        "completed_requests": 0,
        "arrived_requests": 0,
        "simulation_time": 0.0,
        "stop_condition": "error",
        "status": "failed",
        "error": repr(error).replace("\n", " "),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: List[dict]) -> None:
    lines = [
        "# Rollout HF Sweep Report",
        "",
        "This report summarizes rollout_pull + real_hf sweeps over batch size, max token length, chunk size, and SD rate.",
        "",
        "## How To Read",
        "",
        "- Higher `throughput_tokens_per_sec` is better.",
        "- Lower `avg_request_latency_sec` and `p95_request_latency_sec` are better.",
        "- `oom_events_count` and `oom_retry_count` show whether a config is memory fragile.",
        "- `avg_executed_chunk_size < chunk_size` means OOM retry reduced the actual chunk size.",
        "",
    ]
    if rows:
        best = max(rows, key=lambda item: item["throughput_tokens_per_sec"])
        worst_oom = max(rows, key=lambda item: item["oom_events_count"])
        lines.extend(
            [
                "## Key Points",
                "",
                f"- Best throughput config: batch={best['batch_size']}, prompt_len={best['prompt_len']}, max_tokens={best['max_tokens']}, "
                f"chunk={best['chunk_size']}, sd_rate={best['sd_rate']} -> "
                f"{best['throughput_tokens_per_sec']:.2f} tokens/sec",
                f"- Most OOM-prone config: batch={worst_oom['batch_size']}, prompt_len={worst_oom['prompt_len']}, max_tokens={worst_oom['max_tokens']}, "
                f"chunk={worst_oom['chunk_size']}, sd_rate={worst_oom['sd_rate']} -> "
                f"{worst_oom['oom_events_count']} OOM events",
                "",
                "## Rows",
                "",
            ]
        )
        for row in rows:
            lines.append(
                f"- batch={row['batch_size']}, prompt_len={row['prompt_len']}, max_tokens={row['max_tokens']}, chunk={row['chunk_size']}, "
                f"sd_rate={row['sd_rate']}: throughput={row['throughput_tokens_per_sec']:.2f}, "
                f"lat={row['avg_request_latency_sec']:.3f}s, p95={row['p95_request_latency_sec']:.3f}s, "
                f"oom={row['oom_events_count']}, executed_chunk={row['avg_executed_chunk_size']:.2f}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def unique_values(rows: List[dict], key: str) -> List:
    return sorted({row[key] for row in rows})


def filter_rows(rows: List[dict], **conditions) -> List[dict]:
    filtered = []
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            filtered.append(row)
    return filtered


def _save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pick_middle(values: List):
    return values[len(values) // 2]


def plot_metric_vs_chunk(
    rows: List[dict],
    plot_dir: Path,
    fixed_batch_size: int,
    fixed_prompt_len: int,
    metric: str,
    ylabel: str,
    filename: str,
) -> str:
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        return f"{filename}: skipped because all rows failed."
    max_tokens_values = unique_values(rows, "max_tokens")
    sd_rates = unique_values(rows, "sd_rate")
    sd_rates = [value for value in sd_rates if value < 1.0] or sd_rates
    fixed_sd_rate = _pick_middle(sd_rates)

    fig, ax = plt.subplots(figsize=(10, 6))
    for max_tokens in max_tokens_values:
        points = sorted(
            filter_rows(
                rows,
                batch_size=fixed_batch_size,
                prompt_len=fixed_prompt_len,
                max_tokens=max_tokens,
                sd_rate=fixed_sd_rate,
            ),
            key=lambda item: item["chunk_size"],
        )
        if not points:
            continue
        ax.plot(
            [item["chunk_size"] for item in points],
            [item[metric] for item in points],
            marker="o",
            linewidth=2,
            label=f"max_tokens={max_tokens}",
        )
    ax.set_title(f"{ylabel} vs chunk size")
    ax.set_xlabel("chunk_size")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    _save_plot(fig, plot_dir / filename)
    return (
        f"{filename}: x-axis varies `chunk_size`; lines vary `max_tokens`. "
        f"Fixed constants: batch_size={fixed_batch_size}, prompt_len={fixed_prompt_len}, sd_rate={fixed_sd_rate}."
    )


def plot_metric_vs_batch(rows: List[dict], plot_dir: Path, fixed_prompt_len: int, fixed_max_tokens: int, fixed_sd_rate: float) -> str:
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        return "throughput_vs_batch_by_chunk.png: skipped because all rows failed."
    chunk_sizes = unique_values(rows, "chunk_size")
    fig, ax = plt.subplots(figsize=(10, 6))
    for chunk_size in chunk_sizes:
        points = sorted(
            filter_rows(
                rows,
                prompt_len=fixed_prompt_len,
                max_tokens=fixed_max_tokens,
                chunk_size=chunk_size,
                sd_rate=fixed_sd_rate,
            ),
            key=lambda item: item["batch_size"],
        )
        if not points:
            continue
        ax.plot(
            [item["batch_size"] for item in points],
            [item["throughput_tokens_per_sec"] for item in points],
            marker="o",
            linewidth=2,
            label=f"chunk={chunk_size}",
        )
    ax.set_title("Throughput vs batch size")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("throughput_tokens_per_sec")
    ax.grid(alpha=0.3)
    ax.legend()
    filename = "throughput_vs_batch_by_chunk.png"
    _save_plot(fig, plot_dir / filename)
    return (
        f"{filename}: x-axis varies `batch_size`; lines vary `chunk_size`. "
        f"Fixed constants: prompt_len={fixed_prompt_len}, max_tokens={fixed_max_tokens}, sd_rate={fixed_sd_rate}."
    )


def plot_heatmap(
    rows: List[dict],
    plot_dir: Path,
    x_key: str,
    y_key: str,
    metric: str,
    fixed: dict,
    filename: str,
    title: str,
) -> str:
    selected = filter_rows(rows, **fixed)
    selected = [row for row in selected if row.get("status") == "ok"]
    x_values = unique_values(selected, x_key)
    y_values = unique_values(selected, y_key)
    if not selected or not x_values or not y_values:
        return (
            f"{filename}: skipped because no rows matched fixed constants "
            f"{', '.join(f'{key}={value}' for key, value in fixed.items())}."
        )
    matrix = []
    for y in y_values:
        row_values = []
        for x in x_values:
            matches = filter_rows(selected, **{x_key: x, y_key: y})
            row_values.append(matches[0][metric] if matches else float("nan"))
        matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels(y_values)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=metric)
    _save_plot(fig, plot_dir / filename)
    fixed_text = ", ".join(f"{key}={value}" for key, value in fixed.items())
    return (
        f"{filename}: heatmap varies x=`{x_key}` and y=`{y_key}`; color=`{metric}`. "
        f"Fixed constants: {fixed_text}."
    )


def generate_plots(result_dir: Path, rows: List[dict], prompt_len: int) -> None:
    if not rows:
        return
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        plot_dir = result_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "README_PLOTS.md").write_text(
            "# Plot Guide\n\nNo plots were generated because all configurations failed.",
            encoding="utf-8",
        )
        return
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = unique_values(ok_rows, "batch_size")
    prompt_lens = unique_values(ok_rows, "prompt_len")
    max_tokens_values = unique_values(ok_rows, "max_tokens")
    sd_rates = [value for value in unique_values(ok_rows, "sd_rate") if value < 1.0]
    fixed_batch_size = _pick_middle(batch_sizes)
    fixed_prompt_len = _pick_middle(prompt_lens)
    fixed_max_tokens = _pick_middle(max_tokens_values)
    fixed_sd_rate = _pick_middle(sd_rates) if sd_rates else unique_values(rows, "sd_rate")[0]

    descriptions = [
        f"Default prompt_len argument={prompt_len}. The sweep may include multiple prompt_len values.",
        plot_metric_vs_chunk(
            ok_rows,
            plot_dir,
            fixed_batch_size=fixed_batch_size,
            fixed_prompt_len=fixed_prompt_len,
            metric="throughput_tokens_per_sec",
            ylabel="throughput_tokens_per_sec",
            filename="throughput_vs_chunk_by_max_tokens.png",
        ),
        plot_metric_vs_chunk(
            ok_rows,
            plot_dir,
            fixed_batch_size=fixed_batch_size,
            fixed_prompt_len=fixed_prompt_len,
            metric="avg_request_latency_sec",
            ylabel="avg_request_latency_sec",
            filename="latency_vs_chunk_by_max_tokens.png",
        ),
        plot_metric_vs_batch(
            ok_rows,
            plot_dir,
            fixed_prompt_len=fixed_prompt_len,
            fixed_max_tokens=fixed_max_tokens,
            fixed_sd_rate=fixed_sd_rate,
        ),
        plot_heatmap(
            ok_rows,
            plot_dir,
            x_key="chunk_size",
            y_key="batch_size",
            metric="oom_events_count",
            fixed={"prompt_len": fixed_prompt_len, "max_tokens": fixed_max_tokens, "sd_rate": fixed_sd_rate},
            filename="oom_count_heatmap_batch_chunk.png",
            title="OOM count by batch size and chunk size",
        ),
        plot_heatmap(
            ok_rows,
            plot_dir,
            x_key="chunk_size",
            y_key="sd_rate",
            metric="throughput_tokens_per_sec",
            fixed={"batch_size": fixed_batch_size, "prompt_len": fixed_prompt_len, "max_tokens": fixed_max_tokens},
            filename="throughput_heatmap_chunk_sd_rate.png",
            title="Throughput by chunk size and SD rate",
        ),
        plot_heatmap(
            ok_rows,
            plot_dir,
            x_key="chunk_size",
            y_key="max_tokens",
            metric="throughput_tokens_per_sec",
            fixed={"batch_size": fixed_batch_size, "prompt_len": fixed_prompt_len, "sd_rate": fixed_sd_rate},
            filename="throughput_heatmap_max_tokens_chunk.png",
            title="Throughput by max tokens and chunk size",
        ),
        plot_heatmap(
            ok_rows,
            plot_dir,
            x_key="chunk_size",
            y_key="prompt_len",
            metric="throughput_tokens_per_sec",
            fixed={"batch_size": fixed_batch_size, "max_tokens": fixed_max_tokens, "sd_rate": fixed_sd_rate},
            filename="throughput_heatmap_prompt_len_chunk.png",
            title="Throughput by prompt length and chunk size",
        ),
    ]
    (plot_dir / "README_PLOTS.md").write_text(
        "# Plot Guide\n\n" + "\n".join(f"- {item}" for item in descriptions),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    run_name = args.run_name or _default_run_name()
    result_dir = SIM_DIR / "results" / run_name
    raw_dir = result_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    configs = iter_configs(args)
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "config_count": len(configs),
        "args": vars(args),
        "mode": "proxy_debug" if args.proxy_debug else "real_hf",
        "configs": configs,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"Dry run only. Planned {len(configs)} configs.")
        return

    rows = []
    full_results = []
    for index, item in enumerate(configs):
        item = dict(item)
        print(
            f"[{index + 1}/{len(configs)}] batch={item['batch_size']} prompt_len={item['prompt_len']} max_tokens={item['max_tokens']} "
            f"chunk={item['chunk_size']} sd_rate={item['sd_rate']} "
            f"est_decode_steps~{estimate_decode_steps(args, item)}"
        )
        try:
            sim = BatchSDSimulator(build_system_config(args, item), seed=args.seed)
            summary = sim.run(
                duration_seconds=args.duration,
                target_completed_requests=args.target_completed_requests,
                verbose=False,
            )
            window_metrics = sim.get_windowed_metrics(window_sec=args.window_sec)
            row = flatten_summary(item, summary)
            result = {
                "config": item,
                "status": "ok",
                "summary": summary,
                "window_metrics": window_metrics,
                "oom_events": [asdict(event) for event in sim.oom_events],
            }
            if args.save_step_traces:
                result["step_trace"] = [asdict(step) for step in sim.step_traces]
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"Config failed, continuing: {repr(exc)}")
            row = failed_row(item, exc)
            result = {
                "config": item,
                "status": "failed",
                "error": repr(exc),
                "summary": None,
                "window_metrics": [],
                "oom_events": [],
            }
        rows.append(row)
        full_results.append(result)
        raw_path = raw_dir / (
            f"batch{item['batch_size']}_prompt{item['prompt_len']}_max{item['max_tokens']}_chunk{item['chunk_size']}_sd{item['sd_rate']}.json"
        )
        raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    (result_dir / "sweep_results.json").write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    write_csv(result_dir / "sweep_summary.csv", rows)
    write_report(result_dir / "sweep_report.md", rows)
    generate_plots(result_dir, rows, prompt_len=args.prompt_len)
    print(f"Saved rollout HF sweep results to: {result_dir}")


if __name__ == "__main__":
    main()
