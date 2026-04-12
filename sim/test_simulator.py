"""
Unit tests and validation for the SD simulator
"""

import unittest
import json
import tempfile
import os
from simulator import (
    Request, SystemConfig, RequestGenerator, KVCacheManager,
    ComputeModel, SDAccepter, BatchScheduler, BatchSDSimulator
)


class TestRequest(unittest.TestCase):
    """Test Request data structure"""
    
    def test_request_creation(self):
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        self.assertEqual(req.request_id, 0)
        self.assertEqual(req.current_seq_len, 50)
        self.assertFalse(req.is_completed())
    
    def test_request_completion(self):
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        req.generated_tokens = 100
        self.assertTrue(req.is_completed())
    
    def test_remaining_tokens(self):
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        self.assertEqual(req.remaining_tokens(), 100)
        req.generated_tokens = 50
        self.assertEqual(req.remaining_tokens(), 50)


class TestSystemConfig(unittest.TestCase):
    """Test SystemConfig calculations"""
    
    def test_config_defaults(self):
        config = SystemConfig()
        self.assertGreater(config.kv_budget_mb, 0)
        self.assertGreater(config.kv_per_token_mb, 0)
    
    def test_kv_budget_calculation(self):
        config = SystemConfig(
            gpu_memory_mb=10000,
            model_weights_mb=3000,
            activation_buffer_mb=2000,
            runtime_overhead_mb=1000
        )
        expected_budget = 10000 - 3000 - 2000 - 1000
        self.assertEqual(config.kv_budget_mb, expected_budget)


class TestRequestGenerator(unittest.TestCase):
    """Test RequestGenerator"""
    
    def test_request_generation(self):
        config = SystemConfig()
        gen = RequestGenerator(config, seed=42)
        
        req = gen.generate_request(0.0)
        self.assertIsNotNone(req)
        self.assertGreater(req.prompt_len, 0)
        self.assertGreater(req.max_new_tokens, 0)
        self.assertEqual(req.arrival_time, 0.0)
    
    def test_next_arrival_time(self):
        config = SystemConfig(arrival_rate=10.0)
        gen = RequestGenerator(config, seed=42)
        
        t1 = gen.next_arrival_time(0.0)
        self.assertGreater(t1, 0.0)
    
    def test_request_counter(self):
        config = SystemConfig()
        gen = RequestGenerator(config, seed=42)
        
        req1 = gen.generate_request(0.0)
        req2 = gen.generate_request(1.0)
        
        self.assertEqual(req1.request_id, 0)
        self.assertEqual(req2.request_id, 1)


class TestKVCacheManager(unittest.TestCase):
    """Test KVCacheManager"""
    
    def test_allocation(self):
        config = SystemConfig(gpu_memory_mb=100)
        manager = KVCacheManager(config)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        
        result = manager.allocate(req)
        self.assertTrue(result)
        self.assertGreater(manager.total_kv_used_mb, 0)
    
    def test_release(self):
        config = SystemConfig(gpu_memory_mb=100)
        manager = KVCacheManager(config)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        
        manager.allocate(req)
        initial_size = manager.total_kv_used_mb
        
        manager.release(req)
        self.assertEqual(manager.total_kv_used_mb, 0)
    
    def test_memory_constraint(self):
        config = SystemConfig(gpu_memory_mb=1)  # Very small
        manager = KVCacheManager(config)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        
        result = manager.allocate(req)
        # Should fail due to memory constraint
        self.assertFalse(result)


class TestComputeModel(unittest.TestCase):
    """Test ComputeModel"""
    
    def test_latency_estimation(self):
        config = SystemConfig()
        model = ComputeModel(config)
        
        latency = model.estimate_latency(batch_size=8, avg_seq_len=100)
        self.assertGreater(latency, 0)
    
    def test_throughput_estimation(self):
        config = SystemConfig()
        model = ComputeModel(config)
        
        throughput = model.estimate_throughput(batch_size=8, latency_ms=10)
        self.assertGreater(throughput, 0)
    
    def test_latency_increases_with_batch(self):
        config = SystemConfig()
        model = ComputeModel(config)
        
        latency_1 = model.estimate_latency(batch_size=1, avg_seq_len=100)
        latency_8 = model.estimate_latency(batch_size=8, avg_seq_len=100)
        
        self.assertGreater(latency_8, latency_1)
    
    def test_benchmark_table_lookup(self):
        payload = {
            "prefill": [
                {"batch_size": 4, "prompt_len": 128, "latency_ms": 12.5},
            ],
            "decode": [
                {"batch_size": 4, "seq_len": 256, "chunk_size": 4, "latency_ms": 3.5},
            ],
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as handle:
            json.dump(payload, handle)
            path = handle.name
        try:
            config = SystemConfig(benchmark_table_path=path)
            model = ComputeModel(config)
            reqs = [
                Request(i, prompt_len=32, max_new_tokens=64, arrival_time=0.0, accept_rate=1.0)
                for i in range(4)
            ]
            for req in reqs:
                req.prefill_done = True
                req.current_seq_len = 256
            self.assertEqual(model._nearest_prefill_latency(4, 128), 12.5)
            self.assertEqual(model.estimate_decode_latency(reqs, 4), 3.5)
        finally:
            os.unlink(path)


class TestSDAccepter(unittest.TestCase):
    """Test SDAccepter"""
    
    def test_token_acceptance(self):
        accepter = SDAccepter(seed=42)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=1.0  # Always accept
        )
        
        accepted = accepter.accept_tokens(req, num_tokens=4)
        self.assertEqual(accepted, 4)
    
    def test_token_rejection(self):
        accepter = SDAccepter(seed=42)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.0  # Never accept
        )
        
        accepted = accepter.accept_tokens(req, num_tokens=4)
        self.assertEqual(accepted, 0)
    
    def test_statistical_acceptance(self):
        """Test that acceptance rate is statistically correct"""
        accepter = SDAccepter(seed=42)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.5
        )
        
        # Run many trials
        total_proposed = 0
        total_accepted = 0
        
        for _ in range(100):
            total_proposed += 4
            total_accepted += accepter.accept_tokens(req, num_tokens=4)
        
        rate = total_accepted / total_proposed
        # Should be roughly 0.5 (may not stop early due to test design)
        self.assertGreater(rate, 0.2)
        self.assertLess(rate, 0.8)


class TestBatchScheduler(unittest.TestCase):
    """Test BatchScheduler"""
    
    def test_scheduler_creation(self):
        config = SystemConfig()
        kv_manager = KVCacheManager(config)
        scheduler = BatchScheduler(config, kv_manager)
        
        self.assertEqual(len(scheduler.queue), 0)
    
    def test_add_request(self):
        config = SystemConfig()
        kv_manager = KVCacheManager(config)
        scheduler = BatchScheduler(config, kv_manager)
        
        req = Request(
            request_id=0,
            prompt_len=50,
            max_new_tokens=100,
            arrival_time=0.0,
            accept_rate=0.85
        )
        
        scheduler.add_request(req)
        self.assertEqual(len(scheduler.queue), 1)
    
    def test_schedule_batch(self):
        config = SystemConfig(max_batch_size=16, gpu_memory_mb=1000)
        kv_manager = KVCacheManager(config)
        scheduler = BatchScheduler(config, kv_manager)
        
        # Add multiple requests
        for i in range(10):
            req = Request(
                request_id=i,
                prompt_len=50,
                max_new_tokens=100,
                arrival_time=0.0,
                accept_rate=0.85
            )
            kv_manager.allocate(req)
            scheduler.add_request(req)
        
        batch = scheduler.schedule()
        self.assertGreater(len(batch), 0)
        self.assertLessEqual(len(batch), config.max_batch_size)


class TestSimulator(unittest.TestCase):
    """Test the full simulator"""
    
    def test_simulator_creation(self):
        config = SystemConfig()
        sim = BatchSDSimulator(config, seed=42)
        
        self.assertIsNotNone(sim.request_gen)
        self.assertIsNotNone(sim.kv_manager)
        self.assertIsNotNone(sim.scheduler)
    
    def test_short_simulation(self):
        """Test a short simulation runs without errors"""
        config = SystemConfig(
            arrival_rate=5.0,
            max_batch_size=8,
        )
        sim = BatchSDSimulator(config, seed=42)
        
        # This should complete without error
        sim.run_simulation(duration_seconds=2.0)
        
        # Check that some requests were processed
        total_requests = len(sim.completed_requests) + len(sim.active_requests)
        self.assertGreater(total_requests, 0)
    
    def test_simulator_metrics(self):
        """Test that metrics are tracked correctly"""
        config = SystemConfig(
            arrival_rate=5.0,
            max_batch_size=8,
        )
        sim = BatchSDSimulator(config, seed=42)
        summary = sim.run_simulation(duration_seconds=2.0)
        
        self.assertGreaterEqual(sim.total_batches, 0)
        self.assertGreaterEqual(sim.total_compute_steps, 0)
        self.assertGreaterEqual(sim.total_tokens_generated, 0)
        self.assertGreaterEqual(sim.total_tokens_accepted, 0)
        self.assertIn("throughput_tokens_per_sec", summary)
        self.assertIn("peak_kv_utilization", summary)
    
    def test_non_speculative_mode(self):
        config = SystemConfig(
            arrival_rate=5.0,
            max_batch_size=8,
            chunk_size=1,
            enable_speculative=False,
            avg_accept_rate=1.0,
        )
        sim = BatchSDSimulator(config, seed=42)
        summary = sim.run_simulation(duration_seconds=2.0)
        
        self.assertEqual(summary["rejected_draft_tokens"], 0)
        self.assertEqual(summary["fallback_tokens"], 0)
    
    def test_rollout_burst_workload(self):
        config = SystemConfig(
            arrival_rate=8.0,
            max_batch_size=8,
            workload_mode="rollout_burst",
            rollout_burst_size=4,
        )
        sim = BatchSDSimulator(config, seed=42)
        summary = sim.run_simulation(duration_seconds=2.0)
        
        self.assertGreaterEqual(summary["arrived_requests"], 1)
        self.assertGreaterEqual(summary["admitted_requests"], 1)


def run_all_tests():
    """Run all unit tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    print("Running SD Simulator Unit Tests")
    print("="*80)
    run_all_tests()
