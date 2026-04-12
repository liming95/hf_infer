"""Real-model hardware benchmarks for calibrating the async simulator.

This module is the bridge between your system simulator and real GPU behavior:
- run actual HF forward passes for prefill/decode shapes
- record wall-clock latency and CUDA memory statistics
- export a benchmark table that `sim/simulator.py` can consume

The current environment in Codex may not have torch/GPU available, so imports are
kept lazy and the file is designed to run on the target machine where the model is installed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class BenchmarkPoint:
    phase: str
    batch_size: int
    prompt_len: int = 0
    seq_len: int = 0
    chunk_size: int = 0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    memory_allocated_mb: float = 0.0
    max_memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    repeat: int = 0


def _import_torch():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


class TorchBenchmarkRunner:
    """Runs real HF model forwards and stores measured hardware characteristics."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        torch, AutoModelForCausalLM, AutoTokenizer = _import_torch()
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        model_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=model_dtype,
        ).to(device)
        self.model.eval()
        self.vocab_size = int(self.model.config.vocab_size)

    def _rand_ids(self, batch_size: int, seq_len: int):
        torch = self.torch
        ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, seq_len),
            device=self.device,
            dtype=torch.long,
        )
        mask = torch.ones((batch_size, seq_len), device=self.device, dtype=torch.long)
        return ids, mask

    def _synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def _measure(self, fn, repeat: int, warmup: int) -> tuple[float, float, float]:
        torch = self.torch
        for _ in range(warmup):
            fn()
            self._synchronize()

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        self._synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(repeat, 1)

        if self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        else:
            allocated = 0.0
            peak_allocated = 0.0
            reserved = 0.0

        return latency_ms, allocated, peak_allocated if peak_allocated > 0 else reserved

    @property
    def eos_token_id(self) -> int:
        if self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return 0

    def benchmark_prefill(
        self,
        batch_size: int,
        prompt_len: int,
        repeat: int = 5,
        warmup: int = 2,
    ) -> BenchmarkPoint:
        torch = self.torch
        input_ids, attention_mask = self._rand_ids(batch_size, prompt_len)

        @torch.no_grad()
        def run():
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        latency_ms, allocated_mb, peak_mb = self._measure(run, repeat=repeat, warmup=warmup)
        tokens = batch_size * prompt_len
        return BenchmarkPoint(
            phase="prefill",
            batch_size=batch_size,
            prompt_len=prompt_len,
            latency_ms=latency_ms,
            tokens_per_sec=(tokens / latency_ms * 1000.0) if latency_ms > 0 else 0.0,
            memory_allocated_mb=allocated_mb,
            max_memory_allocated_mb=peak_mb,
            memory_reserved_mb=peak_mb,
            repeat=repeat,
        )

    def benchmark_decode(
        self,
        batch_size: int,
        seq_len: int,
        chunk_size: int,
        repeat: int = 5,
        warmup: int = 2,
    ) -> BenchmarkPoint:
        torch = self.torch
        prefix_ids, prefix_mask = self._rand_ids(batch_size, seq_len)
        next_ids, _ = self._rand_ids(batch_size, chunk_size)

        with torch.no_grad():
            prefill = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
            )
            past_key_values = prefill.past_key_values

        @torch.no_grad()
        def run():
            self.model(
                input_ids=next_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        latency_ms, allocated_mb, peak_mb = self._measure(run, repeat=repeat, warmup=warmup)
        tokens = batch_size * chunk_size
        return BenchmarkPoint(
            phase="decode",
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=chunk_size,
            latency_ms=latency_ms,
            tokens_per_sec=(tokens / latency_ms * 1000.0) if latency_ms > 0 else 0.0,
            memory_allocated_mb=allocated_mb,
            max_memory_allocated_mb=peak_mb,
            memory_reserved_mb=peak_mb,
            repeat=repeat,
        )

    def benchmark_grid(
        self,
        prefill_shapes: Sequence[tuple[int, int]],
        decode_shapes: Sequence[tuple[int, int, int]],
        repeat: int = 5,
        warmup: int = 2,
    ) -> dict:
        prefill_points = [
            asdict(self.benchmark_prefill(batch_size, prompt_len, repeat=repeat, warmup=warmup))
            for batch_size, prompt_len in prefill_shapes
        ]
        decode_points = [
            asdict(self.benchmark_decode(batch_size, seq_len, chunk_size, repeat=repeat, warmup=warmup))
            for batch_size, seq_len, chunk_size in decode_shapes
        ]
        return {
            "meta": {
                "device": self.device,
                "model_name_or_path": str(self.model.name_or_path),
                "repeat": repeat,
                "warmup": warmup,
            },
            "prefill": prefill_points,
            "decode": decode_points,
        }


def save_benchmark_table(
    output_path: str,
    model_name_or_path: str,
    prefill_shapes: Iterable[tuple[int, int]],
    decode_shapes: Iterable[tuple[int, int, int]],
    repeat: int = 5,
    warmup: int = 2,
    device: str = "cuda",
    dtype: str = "float16",
) -> Path:
    runner = TorchBenchmarkRunner(model_name_or_path, device=device, dtype=dtype)
    payload = runner.benchmark_grid(
        list(prefill_shapes),
        list(decode_shapes),
        repeat=repeat,
        warmup=warmup,
    )
    output = Path(output_path)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run real-model hardware benchmarks for the simulator.")
    parser.add_argument("--model", required=True, help="Local HF model path or model id")
    parser.add_argument("--output", default="benchmark_table.json", help="Where to write the benchmark table")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    prefill_shapes = [(1, 64), (4, 64), (8, 128), (16, 256)]
    decode_shapes = [(1, 128, 1), (4, 128, 4), (8, 256, 4), (16, 512, 8)]

    output = save_benchmark_table(
        output_path=args.output,
        model_name_or_path=args.model,
        prefill_shapes=prefill_shapes,
        decode_shapes=decode_shapes,
        repeat=args.repeat,
        warmup=args.warmup,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"Saved benchmark table to {output}")


if __name__ == "__main__":
    main()
