"""Execution backends for the async rollout simulator."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class ExecutionMetrics:
    """Measured or estimated cost for one compute step."""

    phase: str
    latency_ms: float
    batch_size: int
    max_seq_len: int
    total_tokens: int
    memory_allocated_mb: float = 0.0
    max_memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    gpu_utilization: Optional[float] = None
    note: str = ""


class ExecutorBackend(Protocol):
    """Interface used by the scheduler loop to execute one compute step."""

    def prefill(self, prompt_lengths: List[int]) -> ExecutionMetrics:
        ...

    def decode(self, seq_lengths: List[int], chunk_size: int) -> ExecutionMetrics:
        ...


class RealComputeOOM(RuntimeError):
    """Raised when the real HF backend hits CUDA OOM."""

    def __init__(self, phase: str, batch_size: int, max_seq_len: int, chunk_size: int, message: str):
        super().__init__(message)
        self.phase = phase
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size


class ProxyExecutor:
    """Cheap analytical executor used when no real backend is available."""

    def __init__(self, compute_model):
        self.compute_model = compute_model

    def prefill(self, prompt_lengths: List[int]) -> ExecutionMetrics:
        latency_ms = self.compute_model.estimate_prefill_latency_from_lengths(prompt_lengths)
        batch_size = len(prompt_lengths)
        total_tokens = sum(prompt_lengths)
        max_seq_len = max(prompt_lengths) if prompt_lengths else 0
        gpu_util = self.compute_model.estimate_gpu_utilization(batch_size, max_seq_len, 1.0)
        return ExecutionMetrics(
            phase="prefill",
            latency_ms=latency_ms,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            total_tokens=total_tokens,
            gpu_utilization=gpu_util,
            note="proxy",
        )

    def decode(self, seq_lengths: List[int], chunk_size: int) -> ExecutionMetrics:
        latency_ms = self.compute_model.estimate_decode_latency_from_lengths(seq_lengths, chunk_size)
        batch_size = len(seq_lengths)
        total_tokens = batch_size * chunk_size
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        gpu_util = self.compute_model.estimate_gpu_utilization(batch_size, max_seq_len, chunk_size)
        return ExecutionMetrics(
            phase="decode",
            latency_ms=latency_ms,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            total_tokens=total_tokens,
            gpu_utilization=gpu_util,
            note="proxy",
        )


class HFRealExecutor:
    """Real HF executor that measures hardware behavior online for each step.

    This backend intentionally uses synthetic token tensors with the same shapes
    as the scheduler-selected batch. That makes it suitable for measuring the
    hardware cost of dynamic batching itself, even when the request content is
    simulated.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
        ).to(device)
        self.model.eval()
        self.vocab_size = int(self.model.config.vocab_size)
        self.forward_signature = inspect.signature(self.model.forward)

    def _last_logits_kwargs(self) -> dict:
        kwargs = {}
        parameters = self.forward_signature.parameters
        if "logits_to_keep" in parameters:
            kwargs["logits_to_keep"] = 1
        elif "num_logits_to_keep" in parameters:
            kwargs["num_logits_to_keep"] = 1
        return kwargs

    def _rand_ids(self, batch_size: int, seq_len: int):
        torch = self.torch
        if batch_size <= 0 or seq_len <= 0:
            return None, None
        ids = torch.randint(
            0,
            self.vocab_size,
            (batch_size, seq_len),
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.device)
        return ids, mask

    def _measure(self, fn, phase: str, batch_size: int, max_seq_len: int, chunk_size: int):
        torch = self.torch
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            fn()
        except torch.OutOfMemoryError as exc:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            raise RealComputeOOM(
                phase=phase,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                chunk_size=chunk_size,
                message=(
                    "HFRealExecutor hit CUDA OOM during real compute. "
                    "Try reducing batch_size, max_concurrent, avg_prompt_len, avg_max_tokens, or chunk_size. "
                    "For long-sequence runs, start with smaller values and scale up gradually."
                ),
            ) from exc
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        else:
            allocated = 0.0
            peak_alloc = 0.0
            reserved = 0.0
        latency_ms = (time.perf_counter() - start) * 1000.0
        return latency_ms, allocated, peak_alloc, reserved

    def prefill(self, prompt_lengths: List[int]) -> ExecutionMetrics:
        if not prompt_lengths:
            return ExecutionMetrics("prefill", 0.0, 0, 0, 0, note="real")
        batch_size = len(prompt_lengths)
        max_prompt = max(prompt_lengths)
        input_ids, attention_mask = self._rand_ids(batch_size, max_prompt)
        torch = self.torch
        logits_kwargs = self._last_logits_kwargs()

        @torch.inference_mode()
        def run():
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **logits_kwargs,
            )

        latency_ms, allocated, peak_alloc, reserved = self._measure(
            run,
            phase="prefill",
            batch_size=batch_size,
            max_seq_len=max_prompt,
            chunk_size=0,
        )
        return ExecutionMetrics(
            phase="prefill",
            latency_ms=latency_ms,
            batch_size=batch_size,
            max_seq_len=max_prompt,
            total_tokens=sum(prompt_lengths),
            memory_allocated_mb=allocated,
            max_memory_allocated_mb=peak_alloc,
            memory_reserved_mb=reserved,
            note="real",
        )

    def decode(self, seq_lengths: List[int], chunk_size: int) -> ExecutionMetrics:
        if not seq_lengths:
            return ExecutionMetrics("decode", 0.0, 0, 0, 0, note="real")
        batch_size = len(seq_lengths)
        max_seq = max(seq_lengths)
        prefix_ids, prefix_mask = self._rand_ids(batch_size, max_seq)
        next_ids, _ = self._rand_ids(batch_size, max(1, chunk_size))
        torch = self.torch
        logits_kwargs = self._last_logits_kwargs()

        with torch.inference_mode():
            prefill_outputs = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
                **logits_kwargs,
            )
            past_key_values = prefill_outputs.past_key_values
            del prefill_outputs

        @torch.inference_mode()
        def run():
            self.model(
                input_ids=next_ids,
                past_key_values=past_key_values,
                use_cache=True,
                **logits_kwargs,
            )

        latency_ms, allocated, peak_alloc, reserved = self._measure(
            run,
            phase="decode",
            batch_size=batch_size,
            max_seq_len=max_seq,
            chunk_size=max(1, chunk_size),
        )
        del past_key_values
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return ExecutionMetrics(
            phase="decode",
            latency_ms=latency_ms,
            batch_size=batch_size,
            max_seq_len=max_seq,
            total_tokens=batch_size * max(1, chunk_size),
            memory_allocated_mb=allocated,
            max_memory_allocated_mb=peak_alloc,
            memory_reserved_mb=reserved,
            note="real",
        )
