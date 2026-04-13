"""
Async rollout + continuous batching + speculative decoding simulator.

The simulator is intentionally lightweight, but it models the interactions that
matter for the research questions in ``sim/readme.md``:
- continuously arriving RLHF-style requests
- KV-cache-constrained admission
- prefill/decode separation
- speculative verification with accepted and wasted draft work
- queue build-up and stability under sustained load
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Deque, Dict, List, Optional

import numpy as np

from executor import ExecutionMetrics, HFRealExecutor, ProxyExecutor, RealComputeOOM


@dataclass
class Request:
    """Represents one rollout request flowing through the system."""

    request_id: int
    prompt_len: int
    max_new_tokens: int
    arrival_time: float
    accept_rate: float

    generated_tokens: int = 0
    current_seq_len: int = field(init=False)
    kv_cache_size: float = 0.0
    prefill_done: bool = False
    admitted_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    completed_time: Optional[float] = None

    def __post_init__(self) -> None:
        self.current_seq_len = self.prompt_len

    def is_completed(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens

    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - self.generated_tokens)

    def queue_wait(self) -> float:
        if self.admitted_time is None:
            return 0.0
        return self.admitted_time - self.arrival_time

    def end_to_end_latency(self) -> float:
        if self.completed_time is None:
            return 0.0
        return self.completed_time - self.arrival_time


@dataclass
class StepTrace:
    """Per-step execution trace for temporal analysis."""

    step_id: int
    start_time: float
    end_time: float
    batch_size: int
    prefill_batch_size: int
    decode_batch_size: int
    verify_width: float
    new_arrivals: int
    new_admissions: int
    new_completions: int
    queued_requests: int
    active_requests: int
    kv_utilization: float
    peak_kv_utilization: float
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    draft_tokens: int
    accepted_draft_tokens: int
    rejected_draft_tokens: int
    fallback_tokens: int
    committed_tokens: int
    step_draft_tokens: int
    step_accepted_draft_tokens: int
    step_rejected_draft_tokens: int
    step_fallback_tokens: int
    step_committed_tokens: int
    avg_seq_len: float
    max_seq_len: int
    avg_prompt_len: float
    avg_remaining_tokens: float
    executor_mode: str
    requested_chunk_size: int
    executed_chunk_size: int
    oom_retries: int = 0
    memory_allocated_mb: float = 0.0
    peak_memory_allocated_mb: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class OOMEvent:
    """Records one OOM event and the retry decision that followed."""

    step_id: int
    time: float
    phase: str
    requested_batch_size: int
    requested_chunk_size: int
    max_seq_len: int
    retry_batch_size: int
    retry_chunk_size: int
    message: str


@dataclass
class SystemConfig:
    """System configuration parameters."""

    model_hidden_size: int = 1152
    num_layers: int = 28
    num_heads: int = 12
    head_dim: int = 96
    dtype_bytes: int = 2

    gpu_memory_mb: float = 12000
    model_weights_mb: float = 3000
    activation_buffer_mb: float = 2000
    runtime_overhead_mb: float = 1000

    max_batch_size: int = 16
    max_concurrent_requests: int = 64
    chunk_size: int = 4
    page_size: int = 128

    arrival_rate: float = 10.0
    avg_prompt_len: int = 50
    avg_max_tokens: int = 200
    avg_accept_rate: float = 0.8
    workload_mode: str = "poisson"  # poisson | rollout_burst | rollout_pull | mixed
    rollout_burst_size: int = 8
    rollout_pull_batch_size: int = 8
    rollout_pull_target_outstanding: int = 16
    long_request_ratio: float = 0.2
    benchmark_table_path: Optional[str] = None
    use_real_compute: bool = False
    model_name_or_path: Optional[str] = None
    real_compute_device: str = "cuda"
    real_compute_dtype: str = "float16"
    compute_memory_margin_mb: float = 1024.0
    prefill_activation_mb_per_token: float = 0.0025
    decode_activation_mb_per_token: float = 0.0015
    activation_scaling_exponent: float = 1.05
    oom_retry_limit: int = 6

    enable_speculative: bool = True
    verify_parallelism: float = 0.32
    draft_cost_ratio: float = 0.08
    rejection_penalty: float = 0.15
    prefill_token_cost_ms: float = 0.012
    decode_token_cost_ms: float = 0.0018
    scheduler_overhead_ms: float = 0.12
    batch_overhead_ms: float = 0.22
    attention_seq_scaling: float = 1.08

    def __post_init__(self) -> None:
        reserved = (
            self.model_weights_mb
            + self.activation_buffer_mb
            + self.runtime_overhead_mb
        )
        if reserved >= self.gpu_memory_mb:
            reserved = min(reserved, self.gpu_memory_mb * 0.7)
        self.kv_budget_mb = max(1.0, self.gpu_memory_mb - reserved)

        self.kv_per_token_bytes = (
            2 * self.num_layers * self.model_hidden_size * self.dtype_bytes
        )
        self.kv_per_token_mb = self.kv_per_token_bytes / (1024 * 1024)


class RequestGenerator:
    """Generates incoming requests for different workload shapes."""

    def __init__(self, config: SystemConfig, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.request_counter = 0
        self._burst_remaining = 0
        self._pull_remaining = 0

    def _sample_prompt_len(self) -> int:
        base = self.config.avg_prompt_len
        if self.np_rng.rand() < self.config.long_request_ratio:
            base *= 3
        return max(8, int(self.np_rng.poisson(base)))

    def _sample_generation_len(self) -> int:
        base = self.config.avg_max_tokens
        if self.np_rng.rand() < self.config.long_request_ratio:
            base *= 2
        return max(16, int(self.np_rng.poisson(base)))

    def _sample_accept_rate(self, prompt_len: int) -> float:
        mean_accept = self.config.avg_accept_rate
        if self.config.enable_speculative and self.config.chunk_size > 1:
            mean_accept -= 0.015 * (self.config.chunk_size - 1)
        if prompt_len > self.config.avg_prompt_len * 2:
            mean_accept -= 0.04
        return float(np.clip(self.np_rng.normal(mean_accept, 0.08), 0.05, 1.0))

    def generate_request(self, current_time: float) -> Request:
        prompt_len = self._sample_prompt_len()
        max_tokens = self._sample_generation_len()
        accept_rate = self._sample_accept_rate(prompt_len)
        request = Request(
            request_id=self.request_counter,
            prompt_len=prompt_len,
            max_new_tokens=max_tokens,
            arrival_time=current_time,
            accept_rate=accept_rate,
        )
        self.request_counter += 1
        return request

    def next_arrival_time(self, last_arrival: float) -> float:
        mode = self.config.workload_mode
        if mode == "rollout_pull":
            # External scheduler decides when to pull the next rollout batch.
            return float("inf")
        if mode == "rollout_burst":
            if self._burst_remaining <= 0:
                self._burst_remaining = max(1, self.config.rollout_burst_size - 1)
                burst_gap = self.config.rollout_burst_size / max(self.config.arrival_rate, 1e-6)
                return last_arrival + burst_gap
            self._burst_remaining -= 1
            return last_arrival + self.np_rng.uniform(0.0, 0.004)

        if mode == "mixed":
            if self.np_rng.rand() < 0.35:
                return last_arrival + self.np_rng.uniform(0.0, 0.01)
            inter_arrival = self.np_rng.exponential(1.0 / max(self.config.arrival_rate, 1e-6))
            return last_arrival + inter_arrival

        inter_arrival = self.np_rng.exponential(1.0 / max(self.config.arrival_rate, 1e-6))
        return last_arrival + inter_arrival


class KVCacheManager:
    """Tracks KV usage and enforces memory limits."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.total_kv_used_mb = 0.0
        self.peak_kv_used_mb = 0.0
        self.request_kv: Dict[int, float] = {}

    def _round_to_pages(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        pages = math.ceil(seq_len / self.config.page_size)
        return pages * self.config.page_size

    def estimate_request_kv_mb(self, seq_len: int) -> float:
        rounded_seq_len = self._round_to_pages(seq_len)
        return rounded_seq_len * self.config.kv_per_token_mb

    def can_allocate_for_seq_len(self, seq_len: int, extra_mb: float = 0.0) -> bool:
        needed = self.estimate_request_kv_mb(seq_len)
        return self.total_kv_used_mb + needed + extra_mb <= self.config.kv_budget_mb + 1e-9

    def allocate(self, request: Request) -> bool:
        required_mb = self.estimate_request_kv_mb(request.current_seq_len)
        if self.total_kv_used_mb + required_mb > self.config.kv_budget_mb + 1e-9:
            return False
        self.request_kv[request.request_id] = required_mb
        self.total_kv_used_mb += required_mb
        self.peak_kv_used_mb = max(self.peak_kv_used_mb, self.total_kv_used_mb)
        request.kv_cache_size = required_mb
        return True

    def update(self, request: Request, new_seq_len: int) -> bool:
        new_size = self.estimate_request_kv_mb(new_seq_len)
        old_size = self.request_kv.get(request.request_id, 0.0)
        delta = new_size - old_size
        if self.total_kv_used_mb + delta > self.config.kv_budget_mb + 1e-9:
            return False
        self.total_kv_used_mb += delta
        self.peak_kv_used_mb = max(self.peak_kv_used_mb, self.total_kv_used_mb)
        self.request_kv[request.request_id] = new_size
        request.kv_cache_size = new_size
        request.current_seq_len = new_seq_len
        return True

    def release(self, request: Request) -> None:
        kv_size = self.request_kv.pop(request.request_id, 0.0)
        self.total_kv_used_mb = max(0.0, self.total_kv_used_mb - kv_size)
        request.kv_cache_size = 0.0

    def get_utilization(self) -> float:
        return 100.0 * self.total_kv_used_mb / max(self.config.kv_budget_mb, 1e-9)

    def get_peak_utilization(self) -> float:
        return 100.0 * self.peak_kv_used_mb / max(self.config.kv_budget_mb, 1e-9)


class ComputeModel:
    """A small compute proxy for prefill and decode."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.prefill_points: List[Dict[str, float]] = []
        self.decode_points: List[Dict[str, float]] = []
        self._load_benchmark_table()

    def _load_benchmark_table(self) -> None:
        path = self.config.benchmark_table_path
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.prefill_points = list(payload.get("prefill", []))
        self.decode_points = list(payload.get("decode", []))

    def _nearest_prefill_latency(self, batch_size: int, prompt_tokens: int) -> Optional[float]:
        if not self.prefill_points:
            return None
        point = min(
            self.prefill_points,
            key=lambda item: abs(item["batch_size"] - batch_size) + abs(item["prompt_len"] - prompt_tokens),
        )
        return float(point["latency_ms"])

    def _nearest_decode_latency(self, batch_size: int, seq_len: float, verify_width: float) -> Optional[float]:
        if not self.decode_points:
            return None
        point = min(
            self.decode_points,
            key=lambda item: (
                abs(item["batch_size"] - batch_size)
                + abs(item["seq_len"] - seq_len)
                + abs(item["chunk_size"] - verify_width)
            ),
        )
        return float(point["latency_ms"])

    def estimate_prefill_latency(self, requests: List[Request]) -> float:
        if not requests:
            return 0.0
        return self.estimate_prefill_latency_from_lengths([r.prompt_len for r in requests])

    def estimate_prefill_latency_from_lengths(self, prompt_lengths: List[int]) -> float:
        if not prompt_lengths:
            return 0.0
        total_prompt_tokens = sum(prompt_lengths)
        measured = self._nearest_prefill_latency(len(prompt_lengths), total_prompt_tokens)
        if measured is not None:
            return measured
        batch_gain = math.sqrt(len(prompt_lengths))
        latency = (
            self.config.scheduler_overhead_ms
            + self.config.batch_overhead_ms
            + self.config.prefill_token_cost_ms * total_prompt_tokens / max(batch_gain, 1.0)
        )
        return latency

    def estimate_decode_latency(self, requests: List[Request], verify_width: float) -> float:
        if not requests:
            return 0.0
        return self.estimate_decode_latency_from_lengths([r.current_seq_len for r in requests], verify_width)

    def estimate_decode_latency_from_lengths(self, seq_lengths: List[int], verify_width: float) -> float:
        if not seq_lengths:
            return 0.0
        batch_size = len(seq_lengths)
        avg_seq_len = mean(seq_lengths)
        measured = self._nearest_decode_latency(batch_size, avg_seq_len, verify_width)
        if measured is not None:
            return measured
        seq_factor = max(1.0, (avg_seq_len / 128.0) ** self.config.attention_seq_scaling)
        parallel_width = 1.0 + max(0.0, verify_width - 1.0) * self.config.verify_parallelism
        draft_cost = verify_width * self.config.draft_cost_ratio
        latency = (
            self.config.scheduler_overhead_ms
            + self.config.batch_overhead_ms
            + self.config.decode_token_cost_ms * batch_size * seq_factor * parallel_width * 100.0
            + draft_cost
        )
        return latency

    def estimate_prefill_extra_memory_mb(self, prompt_lengths: List[int]) -> float:
        if not prompt_lengths:
            return 0.0
        batch_size = len(prompt_lengths)
        max_prompt = max(prompt_lengths)
        measured = None
        if self.prefill_points:
            point = min(
                self.prefill_points,
                key=lambda item: abs(item["batch_size"] - batch_size) + abs(item["prompt_len"] - max_prompt),
            )
            measured = float(point.get("max_memory_allocated_mb", 0.0) or point.get("memory_reserved_mb", 0.0) or 0.0)
        if measured and measured > 0:
            return measured
        return (
            self.config.compute_memory_margin_mb
            + self.config.prefill_activation_mb_per_token
            * batch_size
            * max_prompt
            * ((max_prompt / 128.0) ** max(0.0, self.config.activation_scaling_exponent - 1.0))
        )

    def estimate_decode_extra_memory_mb(self, seq_lengths: List[int], chunk_size: int) -> float:
        if not seq_lengths:
            return 0.0
        batch_size = len(seq_lengths)
        max_seq_len = max(seq_lengths)
        measured = None
        if self.decode_points:
            point = min(
                self.decode_points,
                key=lambda item: (
                    abs(item["batch_size"] - batch_size)
                    + abs(item["seq_len"] - max_seq_len)
                    + abs(item["chunk_size"] - chunk_size)
                ),
            )
            measured = float(point.get("max_memory_allocated_mb", 0.0) or point.get("memory_reserved_mb", 0.0) or 0.0)
        if measured and measured > 0:
            return measured
        return (
            self.config.compute_memory_margin_mb
            + self.config.decode_activation_mb_per_token
            * batch_size
            * max(chunk_size, 1)
            * max_seq_len
            * ((max_seq_len / 128.0) ** max(0.0, self.config.activation_scaling_exponent - 1.0))
        )

    def estimate_step_extra_memory_mb(
        self,
        prefill_prompt_lengths: List[int],
        decode_seq_lengths: List[int],
        chunk_size: int,
    ) -> float:
        return max(
            self.estimate_prefill_extra_memory_mb(prefill_prompt_lengths),
            self.estimate_decode_extra_memory_mb(decode_seq_lengths, chunk_size),
        )

    def estimate_latency(self, batch_size: int, avg_seq_len: int) -> float:
        dummy_requests = [
            Request(
                request_id=i,
                prompt_len=avg_seq_len,
                max_new_tokens=self.config.chunk_size,
                arrival_time=0.0,
                accept_rate=1.0,
            )
            for i in range(batch_size)
        ]
        for req in dummy_requests:
            req.prefill_done = True
            req.current_seq_len = avg_seq_len
        return self.estimate_decode_latency(dummy_requests, self.config.chunk_size)

    def estimate_throughput(self, batch_size: int, latency_ms: float) -> float:
        if latency_ms <= 0:
            return 0.0
        tokens_per_step = batch_size * max(1, self.config.chunk_size)
        return tokens_per_step / latency_ms * 1000.0

    def estimate_gpu_utilization(self, batch_size: int, avg_seq_len: float, verify_width: float) -> float:
        work = batch_size * max(avg_seq_len, 1.0) * max(verify_width, 1.0)
        saturation = work / (work + 2000.0)
        return min(98.0, 25.0 + 73.0 * saturation)


class SDAccepter:
    """Token acceptance for speculative decoding."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def accept_tokens(self, request: Request, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        accepted = 0
        for _ in range(num_tokens):
            if self.rng.random() < request.accept_rate:
                accepted += 1
            else:
                break
        return accepted


class BatchScheduler:
    """Selects a decode/prefill batch under batch and memory constraints."""

    def __init__(self, config: SystemConfig, kv_manager: KVCacheManager, compute_model: ComputeModel):
        self.config = config
        self.kv_manager = kv_manager
        self.compute_model = compute_model
        self.queue: Deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        if request not in self.queue:
            self.queue.append(request)

    def remove_request(self, request: Request) -> None:
        try:
            self.queue.remove(request)
        except ValueError:
            return

    def schedule(self, active_requests: Optional[List[Request]] = None) -> List[Request]:
        candidates = list(active_requests) if active_requests is not None else list(self.queue)
        if not candidates:
            return []

        def priority_key(req: Request) -> tuple:
            return (
                0 if req.prefill_done else 1,
                req.remaining_tokens(),
                req.current_seq_len,
                req.arrival_time,
            )

        available_mb = max(0.0, self.config.kv_budget_mb - self.kv_manager.total_kv_used_mb)
        reserved_growth_mb = 0.0
        batch: List[Request] = []

        for request in sorted(candidates, key=priority_key):
            if len(batch) >= self.config.max_batch_size:
                break

            projected_tokens = 0 if not request.prefill_done else min(
                request.remaining_tokens(),
                max(1, self.config.chunk_size if self.config.enable_speculative else 1),
            )
            growth_mb = (
                self.kv_manager.estimate_request_kv_mb(request.current_seq_len + projected_tokens)
                - request.kv_cache_size
            )
            candidate_batch = batch + [request]
            prefill_prompt_lengths = [r.prompt_len for r in candidate_batch if not r.prefill_done]
            decode_seq_lengths = [r.current_seq_len for r in candidate_batch if r.prefill_done and not r.is_completed()]
            compute_extra_mb = self.compute_model.estimate_step_extra_memory_mb(
                prefill_prompt_lengths=prefill_prompt_lengths,
                decode_seq_lengths=decode_seq_lengths,
                chunk_size=max(1, self.config.chunk_size if self.config.enable_speculative else 1),
            )
            if reserved_growth_mb + max(0.0, growth_mb) + compute_extra_mb <= available_mb + 1e-9:
                batch.append(request)
                reserved_growth_mb += max(0.0, growth_mb)

        return batch


class BatchSDSimulator:
    """Main orchestrator for the async serving simulation."""

    def __init__(self, config: SystemConfig, seed: int = 42):
        self.config = config
        self.request_gen = RequestGenerator(config, seed)
        self.kv_manager = KVCacheManager(config)
        self.compute_model = ComputeModel(config)
        self.scheduler = BatchScheduler(config, self.kv_manager, self.compute_model)
        self.sd_accepter = SDAccepter(seed)
        self.executor = self._build_executor()

        self.current_time = 0.0
        self.waiting_requests: Deque[Request] = deque()
        self.active_requests: List[Request] = []
        self.completed_requests: List[Request] = []

        self.total_arrived_requests = 0
        self.total_admitted_requests = 0
        self.total_batches = 0
        self.total_compute_steps = 0
        self.total_compute_time_ms = 0.0
        self.total_prefill_time_ms = 0.0
        self.total_decode_time_ms = 0.0
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        self.total_rejected_tokens = 0
        self.total_fallback_tokens = 0
        self.total_completed_tokens = 0
        self.total_blocked_admissions = 0
        self.total_batch_size_acc = 0
        self.utilization_samples: List[float] = []
        self.batch_size_trace: List[int] = []
        self.queue_size_trace: List[int] = []
        self.execution_trace: List[ExecutionMetrics] = []
        self.step_traces: List[StepTrace] = []
        self.oom_events: List[OOMEvent] = []
        self.stop_reason: str = "unknown"

    def _build_executor(self):
        if self.config.use_real_compute:
            if not self.config.model_name_or_path:
                raise ValueError("model_name_or_path must be set when use_real_compute=True")
            return HFRealExecutor(
                model_name_or_path=self.config.model_name_or_path,
                device=self.config.real_compute_device,
                dtype=self.config.real_compute_dtype,
            )
        return ProxyExecutor(self.compute_model)

    def run_simulation(self, duration_seconds: float = 60.0, verbose: bool = False) -> Dict[str, float]:
        return self.run(
            duration_seconds=duration_seconds,
            target_completed_requests=None,
            verbose=verbose,
        )

    def run(
        self,
        duration_seconds: Optional[float] = 60.0,
        target_completed_requests: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        if duration_seconds is None and target_completed_requests is None:
            raise ValueError("Either duration_seconds or target_completed_requests must be provided")

        next_arrival_time = self.request_gen.next_arrival_time(0.0)

        while True:
            if target_completed_requests is not None and len(self.completed_requests) >= target_completed_requests:
                self.stop_reason = "completed_requests"
                break
            if duration_seconds is not None and self.current_time >= duration_seconds:
                self.stop_reason = "duration"
                break

            arrivals_this_step = self._ingest_rollout_pull_arrivals(verbose)
            next_arrival_time, streamed_arrivals = self._ingest_arrivals(
                next_arrival_time,
                duration_seconds,
                verbose,
            )
            arrivals_this_step += streamed_arrivals
            admissions_this_step = self._admit_requests()

            batch = self.scheduler.schedule(self.active_requests)
            if batch:
                latency_ms = self._process_batch(
                    batch,
                    verbose,
                    arrivals_this_step=arrivals_this_step,
                    admissions_this_step=admissions_this_step,
                )
                self.current_time += latency_ms / 1000.0
                self.queue_size_trace.append(len(self.waiting_requests))
                continue

            if self.config.workload_mode == "rollout_pull" and target_completed_requests is not None:
                if not self.waiting_requests and not self.active_requests:
                    self._ingest_rollout_pull_arrivals(verbose)
                    admissions_this_step = self._admit_requests()
                    batch = self.scheduler.schedule(self.active_requests)
                    if batch:
                        continue

            if duration_seconds is None:
                if target_completed_requests is not None and not self.waiting_requests and not self.active_requests:
                    self.stop_reason = "idle_before_target"
                    break
                self.current_time += 0.001
                continue

            if next_arrival_time >= duration_seconds:
                self.stop_reason = "duration_idle"
                break
            self.current_time = max(self.current_time, next_arrival_time)

        summary = self.get_summary(duration_seconds if duration_seconds is not None else self.current_time)
        if verbose:
            self._print_summary(summary)
        return summary

    def _ingest_rollout_pull_arrivals(self, verbose: bool) -> int:
        if self.config.workload_mode != "rollout_pull":
            return 0
        outstanding = len(self.waiting_requests) + len(self.active_requests)
        if outstanding >= self.config.rollout_pull_target_outstanding:
            return 0

        pulled = 0
        while (
            outstanding + pulled < self.config.rollout_pull_target_outstanding
            and pulled < self.config.rollout_pull_batch_size
        ):
            request = self.request_gen.generate_request(self.current_time)
            self.waiting_requests.append(request)
            self.total_arrived_requests += 1
            pulled += 1
            if verbose:
                print(
                    f"[{self.current_time:7.3f}s] rollout_pull request={request.request_id} arrived "
                    f"prompt={request.prompt_len} max_new={request.max_new_tokens} "
                    f"accept={request.accept_rate:.2f}"
                )
        return pulled

    def _ingest_arrivals(
        self,
        next_arrival_time: float,
        duration_seconds: Optional[float],
        verbose: bool,
    ) -> tuple[float, int]:
        arrivals = 0
        if duration_seconds is None:
            return next_arrival_time, arrivals
        while next_arrival_time <= self.current_time and next_arrival_time < duration_seconds:
            request = self.request_gen.generate_request(next_arrival_time)
            self.waiting_requests.append(request)
            self.total_arrived_requests += 1
            arrivals += 1
            if verbose:
                print(
                    f"[{self.current_time:7.3f}s] request={request.request_id} arrived "
                    f"prompt={request.prompt_len} max_new={request.max_new_tokens} "
                    f"accept={request.accept_rate:.2f}"
                )
            next_arrival_time = self.request_gen.next_arrival_time(next_arrival_time)
        return next_arrival_time, arrivals

    def _admit_requests(self) -> int:
        admitted = 0
        while self.waiting_requests and len(self.active_requests) < self.config.max_concurrent_requests:
            request = self.waiting_requests[0]
            projected_kv_mb = self.kv_manager.estimate_request_kv_mb(request.current_seq_len)
            projected_extra_mb = self.compute_model.estimate_step_extra_memory_mb(
                prefill_prompt_lengths=[request.prompt_len],
                decode_seq_lengths=[],
                chunk_size=max(1, self.config.chunk_size if self.config.enable_speculative else 1),
            )
            total_if_admitted = self.kv_manager.total_kv_used_mb + projected_kv_mb + projected_extra_mb
            if total_if_admitted > self.config.kv_budget_mb + 1e-9:
                self.total_blocked_admissions += 1
                break
            if not self.kv_manager.allocate(request):
                self.total_blocked_admissions += 1
                break

            self.waiting_requests.popleft()
            request.admitted_time = self.current_time
            self.active_requests.append(request)
            self.scheduler.add_request(request)
            self.total_admitted_requests += 1
            admitted += 1
        return admitted

    def _process_batch(self, batch: List[Request], verbose: bool, arrivals_this_step: int = 0, admissions_this_step: int = 0) -> float:
        step_start_time = self.current_time
        effective_batch = list(batch)
        requested_chunk_size = max(1, self.config.chunk_size if self.config.enable_speculative else 1)
        executed_chunk_size = requested_chunk_size
        oom_retries = 0

        while True:
            prefill_requests = [r for r in effective_batch if not r.prefill_done]
            decode_requests = [r for r in effective_batch if r.prefill_done and not r.is_completed()]

            try:
                prefill_metrics = self.executor.prefill([r.prompt_len for r in prefill_requests])
                prefill_latency_ms = prefill_metrics.latency_ms
                verify_width = 1.0
                if decode_requests:
                    verify_width = mean(
                        min(r.remaining_tokens(), executed_chunk_size)
                        for r in decode_requests
                    )
                decode_metrics = self.executor.decode(
                    [r.current_seq_len for r in decode_requests],
                    max(1, int(round(verify_width))),
                )
                decode_latency_ms = decode_metrics.latency_ms
                break
            except RealComputeOOM as exc:
                oom_retries += 1
                retry_batch_size = len(effective_batch)
                retry_chunk_size = executed_chunk_size
                if executed_chunk_size > 1:
                    retry_chunk_size = max(1, executed_chunk_size // 2)
                    if retry_chunk_size == executed_chunk_size:
                        retry_chunk_size = max(1, executed_chunk_size - 1)
                    executed_chunk_size = retry_chunk_size
                elif len(effective_batch) > 1:
                    retry_batch_size = max(1, len(effective_batch) // 2)
                    effective_batch = sorted(
                        effective_batch,
                        key=lambda r: (r.current_seq_len, r.remaining_tokens(), r.arrival_time),
                    )[:retry_batch_size]
                else:
                    raise

                self.oom_events.append(
                    OOMEvent(
                        step_id=self.total_compute_steps + 1,
                        time=self.current_time,
                        phase=exc.phase,
                        requested_batch_size=exc.batch_size,
                        requested_chunk_size=exc.chunk_size if exc.chunk_size > 0 else requested_chunk_size,
                        max_seq_len=exc.max_seq_len,
                        retry_batch_size=len(effective_batch),
                        retry_chunk_size=executed_chunk_size,
                        message=str(exc),
                    )
                )
                if oom_retries >= self.config.oom_retry_limit:
                    raise RuntimeError(
                        f"Exceeded oom_retry_limit={self.config.oom_retry_limit} while trying to execute a real HF step"
                    ) from exc

        batch = effective_batch
        prefill_requests = [r for r in batch if not r.prefill_done]
        decode_requests = [r for r in batch if r.prefill_done and not r.is_completed()]
        self.total_batches += 1
        self.total_compute_steps += 1
        self.total_batch_size_acc += len(batch)
        self.batch_size_trace.append(len(batch))
        if prefill_requests:
            self.total_prefill_time_ms += prefill_latency_ms
            self.execution_trace.append(prefill_metrics)
            for request in prefill_requests:
                request.prefill_done = True
                if request.decode_start_time is None:
                    request.decode_start_time = self.current_time
        self.total_decode_time_ms += decode_latency_ms
        if decode_requests:
            self.execution_trace.append(decode_metrics)

        step_draft_tokens = 0
        step_accepted_draft_tokens = 0
        step_rejected_draft_tokens = 0
        step_fallback_tokens = 0
        step_committed_tokens = 0

        for request in decode_requests:
            remaining = request.remaining_tokens()
            proposed_tokens = min(remaining, executed_chunk_size)
            if self.config.enable_speculative:
                accepted_tokens = self.sd_accepter.accept_tokens(request, proposed_tokens)
                fallback_token = 1 if accepted_tokens < proposed_tokens and remaining > accepted_tokens else 0
                committed_tokens = min(remaining, accepted_tokens + fallback_token)
                rejected_tokens = max(0, proposed_tokens - accepted_tokens)
            else:
                accepted_tokens = proposed_tokens
                fallback_token = 0
                committed_tokens = proposed_tokens
                rejected_tokens = 0

            if committed_tokens > 0:
                new_seq_len = request.prompt_len + request.generated_tokens + committed_tokens
                updated = self.kv_manager.update(request, new_seq_len)
                if not updated:
                    continue
                request.generated_tokens += committed_tokens
                if request.decode_start_time is None:
                    request.decode_start_time = self.current_time

            self.total_tokens_generated += proposed_tokens
            self.total_tokens_accepted += accepted_tokens
            self.total_rejected_tokens += rejected_tokens
            self.total_fallback_tokens += fallback_token
            self.total_completed_tokens += committed_tokens
            step_draft_tokens += proposed_tokens
            step_accepted_draft_tokens += accepted_tokens
            step_rejected_draft_tokens += rejected_tokens
            step_fallback_tokens += fallback_token
            step_committed_tokens += committed_tokens

        batch_latency_ms = prefill_latency_ms + decode_latency_ms
        self.total_compute_time_ms += batch_latency_ms
        step_end_time = self.current_time + batch_latency_ms / 1000.0

        if decode_requests:
            if decode_metrics.gpu_utilization is not None:
                self.utilization_samples.append(decode_metrics.gpu_utilization)
            else:
                avg_seq_len = mean(r.current_seq_len for r in decode_requests)
                self.utilization_samples.append(
                    self.compute_model.estimate_gpu_utilization(len(decode_requests), avg_seq_len, verify_width)
                )
        elif prefill_requests:
            if prefill_metrics.gpu_utilization is not None:
                self.utilization_samples.append(prefill_metrics.gpu_utilization)
            else:
                avg_prompt = mean(r.prompt_len for r in prefill_requests)
                self.utilization_samples.append(
                    self.compute_model.estimate_gpu_utilization(len(prefill_requests), avg_prompt, 1.0)
                )

        step_gpu_utilization = self.utilization_samples[-1] if self.utilization_samples else 0.0
        step_memory_allocated_mb = max(prefill_metrics.memory_allocated_mb, decode_metrics.memory_allocated_mb)
        step_peak_memory_allocated_mb = max(
            prefill_metrics.max_memory_allocated_mb,
            decode_metrics.max_memory_allocated_mb,
        )

        completed_now: List[Request] = []
        for request in batch:
            if request.is_completed():
                request.completed_time = self.current_time + batch_latency_ms / 1000.0
                completed_now.append(request)

        for request in completed_now:
            self.scheduler.remove_request(request)
            if request in self.active_requests:
                self.active_requests.remove(request)
            self.completed_requests.append(request)
            self.kv_manager.release(request)
            if verbose:
                print(
                    f"[{request.completed_time:7.3f}s] request={request.request_id} completed "
                    f"gen={request.generated_tokens}/{request.max_new_tokens} "
                    f"latency={request.end_to_end_latency():.3f}s"
                )

        self.step_traces.append(
            StepTrace(
                step_id=self.total_compute_steps,
                start_time=step_start_time,
                end_time=step_end_time,
                batch_size=len(batch),
                prefill_batch_size=len(prefill_requests),
                decode_batch_size=len(decode_requests),
                verify_width=verify_width,
                new_arrivals=arrivals_this_step,
                new_admissions=admissions_this_step,
                new_completions=len(completed_now),
                queued_requests=len(self.waiting_requests),
                active_requests=len(self.active_requests),
                kv_utilization=self.kv_manager.get_utilization(),
                peak_kv_utilization=self.kv_manager.get_peak_utilization(),
                prefill_latency_ms=prefill_latency_ms,
                decode_latency_ms=decode_latency_ms,
                total_latency_ms=batch_latency_ms,
                draft_tokens=self.total_tokens_generated,
                accepted_draft_tokens=self.total_tokens_accepted,
                rejected_draft_tokens=self.total_rejected_tokens,
                fallback_tokens=self.total_fallback_tokens,
                committed_tokens=self.total_completed_tokens,
                step_draft_tokens=step_draft_tokens,
                step_accepted_draft_tokens=step_accepted_draft_tokens,
                step_rejected_draft_tokens=step_rejected_draft_tokens,
                step_fallback_tokens=step_fallback_tokens,
                step_committed_tokens=step_committed_tokens,
                avg_seq_len=float(mean([r.current_seq_len for r in batch])) if batch else 0.0,
                max_seq_len=max([r.current_seq_len for r in batch], default=0),
                avg_prompt_len=float(mean([r.prompt_len for r in prefill_requests])) if prefill_requests else 0.0,
                avg_remaining_tokens=float(mean([r.remaining_tokens() for r in batch])) if batch else 0.0,
                executor_mode="real_hf" if self.config.use_real_compute else "proxy",
                requested_chunk_size=requested_chunk_size,
                executed_chunk_size=executed_chunk_size,
                oom_retries=oom_retries,
                memory_allocated_mb=step_memory_allocated_mb,
                peak_memory_allocated_mb=step_peak_memory_allocated_mb,
                gpu_utilization=step_gpu_utilization,
            )
        )

        return max(batch_latency_ms, 0.05)

    def export_step_traces(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in self.step_traces], handle, indent=2)

    def export_oom_events(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in self.oom_events], handle, indent=2)

    def _counter_delta(self, traces: List[StepTrace], field_name: str) -> float:
        if not traces:
            return 0.0
        previous_value = 0.0
        first_index = self.step_traces.index(traces[0])
        if first_index > 0:
            previous_value = getattr(self.step_traces[first_index - 1], field_name)
        end_value = getattr(traces[-1], field_name)
        return float(end_value - previous_value)

    def get_windowed_metrics(self, window_sec: float = 1.0) -> List[Dict[str, float]]:
        if not self.step_traces:
            return []

        max_time = max(trace.end_time for trace in self.step_traces)
        window_start = 0.0
        windows: List[Dict[str, float]] = []

        while window_start < max_time + 1e-9:
            window_end = window_start + window_sec
            traces = [
                trace
                for trace in self.step_traces
                if trace.end_time > window_start and trace.start_time <= window_end
            ]
            if traces:
                duration = max(window_end - window_start, 1e-9)
                windows.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "steps": len(traces),
                        "avg_batch_size": float(np.mean([trace.batch_size for trace in traces])),
                        "avg_prefill_batch_size": float(np.mean([trace.prefill_batch_size for trace in traces])),
                        "avg_decode_batch_size": float(np.mean([trace.decode_batch_size for trace in traces])),
                        "avg_queue_size": float(np.mean([trace.queued_requests for trace in traces])),
                        "avg_active_requests": float(np.mean([trace.active_requests for trace in traces])),
                        "avg_kv_utilization": float(np.mean([trace.kv_utilization for trace in traces])),
                        "peak_kv_utilization": float(max(trace.peak_kv_utilization for trace in traces)),
                        "avg_step_latency_ms": float(np.mean([trace.total_latency_ms for trace in traces])),
                        "avg_prefill_latency_ms": float(np.mean([trace.prefill_latency_ms for trace in traces])),
                        "avg_decode_latency_ms": float(np.mean([trace.decode_latency_ms for trace in traces])),
                        "avg_requested_chunk_size": float(np.mean([trace.requested_chunk_size for trace in traces])),
                        "avg_executed_chunk_size": float(np.mean([trace.executed_chunk_size for trace in traces])),
                        "oom_retries": int(sum(trace.oom_retries for trace in traces)),
                        "throughput_tokens_per_sec": self._counter_delta(traces, "committed_tokens") / duration,
                        "accepted_tokens_per_sec": self._counter_delta(traces, "accepted_draft_tokens") / duration,
                        "rejected_tokens_per_sec": self._counter_delta(traces, "rejected_draft_tokens") / duration,
                        "fallback_tokens_per_sec": self._counter_delta(traces, "fallback_tokens") / duration,
                        "avg_gpu_utilization": float(np.mean([trace.gpu_utilization for trace in traces])),
                        "avg_memory_allocated_mb": float(np.mean([trace.memory_allocated_mb for trace in traces])),
                        "peak_memory_allocated_mb": float(max(trace.peak_memory_allocated_mb for trace in traces)),
                    }
                )
            window_start = window_end

        return windows

    def get_summary(self, duration_seconds: float) -> Dict[str, float]:
        completed_latencies = [r.end_to_end_latency() for r in self.completed_requests]
        queue_waits = [r.queue_wait() for r in self.completed_requests]
        decode_latencies = [
            (r.completed_time - r.decode_start_time)
            for r in self.completed_requests
            if r.completed_time is not None and r.decode_start_time is not None
        ]
        simulation_time = max(self.current_time, duration_seconds, 1e-9)
        throughput = self.total_completed_tokens / simulation_time
        acceptance_rate = (
            self.total_tokens_accepted / self.total_tokens_generated
            if self.total_tokens_generated > 0
            else 0.0
        )
        fallback_share = (
            self.total_fallback_tokens / self.total_completed_tokens
            if self.total_completed_tokens > 0
            else 0.0
        )

        return {
            "simulation_time": simulation_time,
            "stop_condition": self.stop_reason,
            "completed_requests": len(self.completed_requests),
            "active_requests": len(self.active_requests),
            "queued_requests": len(self.waiting_requests),
            "arrived_requests": self.total_arrived_requests,
            "admitted_requests": self.total_admitted_requests,
            "blocked_admissions": self.total_blocked_admissions,
            "total_batches": self.total_batches,
            "total_compute_steps": self.total_compute_steps,
            "total_compute_time_ms": self.total_compute_time_ms,
            "total_prefill_time_ms": self.total_prefill_time_ms,
            "total_decode_time_ms": self.total_decode_time_ms,
            "draft_tokens": self.total_tokens_generated,
            "accepted_draft_tokens": self.total_tokens_accepted,
            "rejected_draft_tokens": self.total_rejected_tokens,
            "fallback_tokens": self.total_fallback_tokens,
            "committed_tokens": self.total_completed_tokens,
            "draft_acceptance_rate": acceptance_rate,
            "fallback_share": fallback_share,
            "throughput_tokens_per_sec": throughput,
            "avg_batch_size": self.total_batch_size_acc / max(self.total_batches, 1),
            "avg_request_latency_sec": float(np.mean(completed_latencies)) if completed_latencies else 0.0,
            "p95_request_latency_sec": float(np.percentile(completed_latencies, 95)) if completed_latencies else 0.0,
            "avg_queue_wait_sec": float(np.mean(queue_waits)) if queue_waits else 0.0,
            "avg_decode_service_sec": float(np.mean(decode_latencies)) if decode_latencies else 0.0,
            "peak_kv_utilization": self.kv_manager.get_peak_utilization(),
            "final_kv_utilization": self.kv_manager.get_utilization(),
            "avg_gpu_utilization": float(np.mean(self.utilization_samples)) if self.utilization_samples else 0.0,
            "avg_step_memory_allocated_mb": float(
                np.mean([m.memory_allocated_mb for m in self.execution_trace])
            ) if self.execution_trace else 0.0,
            "peak_step_memory_allocated_mb": max(
                [m.max_memory_allocated_mb for m in self.execution_trace],
                default=0.0,
            ),
            "oom_events_count": len(self.oom_events),
            "oom_retry_count": int(sum(event.retry_chunk_size != event.requested_chunk_size or event.retry_batch_size != event.requested_batch_size for event in self.oom_events)),
            "oom_events_by_requested_chunk": {
                str(chunk): sum(1 for event in self.oom_events if event.requested_chunk_size == chunk)
                for chunk in sorted({event.requested_chunk_size for event in self.oom_events})
            },
            "avg_executed_chunk_size": float(np.mean([trace.executed_chunk_size for trace in self.step_traces])) if self.step_traces else 0.0,
            "executor_mode": "real_hf" if self.config.use_real_compute else "proxy",
            "stability_ratio": len(self.completed_requests) / max(self.total_arrived_requests, 1),
            "queue_backlog_ratio": len(self.waiting_requests) / max(self.total_arrived_requests, 1),
        }

    def _print_summary(self, summary: Dict[str, float]) -> None:
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Arrived Requests:       {summary['arrived_requests']}")
        print(f"Completed Requests:     {summary['completed_requests']}")
        print(f"Active Requests:        {summary['active_requests']}")
        print(f"Queued Requests:        {summary['queued_requests']}")
        print(f"Throughput:             {summary['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Draft Acceptance Rate:  {summary['draft_acceptance_rate']:.2%}")
        print(f"Average Batch Size:     {summary['avg_batch_size']:.2f}")
        print(f"Average Request Latency:{summary['avg_request_latency_sec']:.3f}s")
        print(f"P95 Request Latency:    {summary['p95_request_latency_sec']:.3f}s")
        print(f"Average Queue Wait:     {summary['avg_queue_wait_sec']:.3f}s")
        print(f"Peak KV Utilization:    {summary['peak_kv_utilization']:.2f}%")
        print(f"Average GPU Utilization:{summary['avg_gpu_utilization']:.2f}%")
        print(f"Stability Ratio:        {summary['stability_ratio']:.2%}")


def main() -> None:
    config = SystemConfig(
        arrival_rate=10.0,
        max_batch_size=16,
        max_concurrent_requests=48,
        chunk_size=4,
        avg_prompt_len=50,
        avg_max_tokens=200,
        avg_accept_rate=0.85,
        workload_mode="mixed",
    )
    simulator = BatchSDSimulator(config, seed=42)
    simulator.run_simulation(duration_seconds=30.0, verbose=True)


if __name__ == "__main__":
    main()
