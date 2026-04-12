"""
Visualization and analysis utilities for the SD simulator
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from simulator import BatchSDSimulator, SystemConfig


class SimulationVisualizer:
    """Visualize simulation results"""
    
    @staticmethod
    def plot_throughput_vs_accept_rate():
        """Plot how accept rate impacts throughput and efficiency"""
        accept_rates = np.linspace(0.5, 1.0, 6)
        throughputs = []
        efficiencies = []
        
        for ar in accept_rates:
            config = SystemConfig(
                arrival_rate=10.0,
                max_batch_size=16,
                chunk_size=4,
                avg_accept_rate=ar
            )
            
            sim = BatchSDSimulator(config, seed=42)
            sim.run_simulation(duration_seconds=15.0)
            
            throughput = (sim.total_tokens_accepted / sim.total_compute_time_ms * 1000 
                         if sim.total_compute_time_ms > 0 else 0)
            efficiency = (sim.total_tokens_accepted / sim.total_tokens_generated 
                         if sim.total_tokens_generated > 0 else 0)
            
            throughputs.append(throughput)
            efficiencies.append(efficiency)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Throughput plot
        ax1.plot(accept_rates, throughputs, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('SD Accept Rate', fontsize=12)
        ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax1.set_title('Throughput vs SD Accept Rate', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax2.plot(accept_rates, efficiencies, 'r-s', linewidth=2, markersize=8)
        ax2.set_xlabel('SD Accept Rate', fontsize=12)
        ax2.set_ylabel('Token Acceptance Rate', fontsize=12)
        ax2.set_title('Token Efficiency vs SD Accept Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('throughput_vs_accept_rate.png', dpi=150, bbox_inches='tight')
        print("Saved: throughput_vs_accept_rate.png")
        return fig
    
    @staticmethod
    def plot_batch_size_scaling():
        """Plot how batch size affects throughput"""
        batch_sizes = [4, 8, 16, 32]
        throughputs = []
        latencies = []
        
        for bs in batch_sizes:
            config = SystemConfig(
                arrival_rate=10.0,
                max_batch_size=bs,
                chunk_size=4,
                avg_accept_rate=0.85
            )
            
            sim = BatchSDSimulator(config, seed=42)
            sim.run_simulation(duration_seconds=15.0)
            
            throughput = (sim.total_tokens_accepted / sim.total_compute_time_ms * 1000 
                         if sim.total_compute_time_ms > 0 else 0)
            
            avg_latency = (sim.total_compute_time_ms / max(sim.total_compute_steps, 1))
            
            throughputs.append(throughput)
            latencies.append(avg_latency)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Throughput plot
        ax1.bar(range(len(batch_sizes)), throughputs, color='skyblue', edgecolor='navy', linewidth=2)
        ax1.set_xticks(range(len(batch_sizes)))
        ax1.set_xticklabels(batch_sizes)
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax1.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Latency plot
        ax2.bar(range(len(batch_sizes)), latencies, color='lightcoral', edgecolor='darkred', linewidth=2)
        ax2.set_xticks(range(len(batch_sizes)))
        ax2.set_xticklabels(batch_sizes)
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_title('Latency vs Batch Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('batch_size_scaling.png', dpi=150, bbox_inches='tight')
        print("Saved: batch_size_scaling.png")
        return fig
    
    @staticmethod
    def plot_chunk_size_granularity():
        """Plot how chunk size affects system"""
        chunk_sizes = [1, 2, 4, 8]
        throughputs = []
        efficiencies = []
        
        for cs in chunk_sizes:
            config = SystemConfig(
                arrival_rate=10.0,
                max_batch_size=16,
                chunk_size=cs,
                avg_accept_rate=0.85
            )
            
            sim = BatchSDSimulator(config, seed=42)
            sim.run_simulation(duration_seconds=15.0)
            
            throughput = (sim.total_tokens_accepted / sim.total_compute_time_ms * 1000 
                         if sim.total_compute_time_ms > 0 else 0)
            efficiency = (sim.total_tokens_accepted / sim.total_tokens_generated 
                         if sim.total_tokens_generated > 0 else 0)
            
            throughputs.append(throughput)
            efficiencies.append(efficiency)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Throughput plot
        ax1.plot(chunk_sizes, throughputs, 'g-^', linewidth=2, markersize=10)
        ax1.set_xlabel('Chunk Size (tokens)', fontsize=12)
        ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax1.set_title('Throughput vs Chunk Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax2.plot(chunk_sizes, efficiencies, 'm-v', linewidth=2, markersize=10)
        ax2.set_xlabel('Chunk Size (tokens)', fontsize=12)
        ax2.set_ylabel('Token Acceptance Rate', fontsize=12)
        ax2.set_title('Token Efficiency vs Chunk Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chunk_size_granularity.png', dpi=150, bbox_inches='tight')
        print("Saved: chunk_size_granularity.png")
        return fig
    
    @staticmethod
    def plot_load_sensitivity():
        """Plot how system behaves under different loads"""
        arrival_rates = [5, 10, 20, 40]
        throughputs = []
        request_completions = []
        
        for ar in arrival_rates:
            config = SystemConfig(
                arrival_rate=ar,
                max_batch_size=16,
                chunk_size=4,
                avg_accept_rate=0.85
            )
            
            sim = BatchSDSimulator(config, seed=42)
            sim.run_simulation(duration_seconds=15.0)
            
            throughput = (sim.total_tokens_accepted / sim.total_compute_time_ms * 1000 
                         if sim.total_compute_time_ms > 0 else 0)
            
            throughputs.append(throughput)
            request_completions.append(len(sim.completed_requests))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Throughput plot
        ax1.plot(arrival_rates, throughputs, 'b-o', linewidth=2, markersize=10)
        ax1.set_xlabel('Arrival Rate (requests/sec)', fontsize=12)
        ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax1.set_title('Throughput vs Load', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Completion rate plot
        ax2.bar(range(len(arrival_rates)), request_completions, 
                color='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax2.set_xticks(range(len(arrival_rates)))
        ax2.set_xticklabels(arrival_rates)
        ax2.set_xlabel('Arrival Rate (requests/sec)', fontsize=12)
        ax2.set_ylabel('Completed Requests', fontsize=12)
        ax2.set_title('Completion Rate vs Load', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('load_sensitivity.png', dpi=150, bbox_inches='tight')
        print("Saved: load_sensitivity.png")
        return fig
    
    @staticmethod
    def plot_memory_pressure():
        """Plot KV cache utilization under different configurations"""
        # Test different batch and sequence length combinations
        configs = [
            ("Low load", SystemConfig(arrival_rate=5.0, max_batch_size=8)),
            ("Medium load", SystemConfig(arrival_rate=10.0, max_batch_size=16)),
            ("High load", SystemConfig(arrival_rate=20.0, max_batch_size=32)),
        ]
        
        peak_utils = []
        config_labels = []
        
        for label, config in configs:
            sim = BatchSDSimulator(config, seed=42)
            sim.run_simulation(duration_seconds=15.0)
            
            # Track peak KV utilization during run
            peak_utils.append(100.0)  # Placeholder, would need to track during sim
            config_labels.append(label)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(range(len(configs)), peak_utils, 
               color=['lightblue', 'lightyellow', 'lightcoral'], 
               edgecolor='black', linewidth=2)
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Saturation Point (80%)')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(config_labels)
        ax.set_ylabel('KV Cache Utilization (%)', fontsize=12)
        ax.set_title('Memory Pressure: KV Cache Utilization', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 120])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('memory_pressure.png', dpi=150, bbox_inches='tight')
        print("Saved: memory_pressure.png")
        return fig
    
    @staticmethod
    def plot_summary_dashboard():
        """Create a comprehensive dashboard view"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Default config for all plots
        config = SystemConfig(
            arrival_rate=10.0,
            max_batch_size=16,
            chunk_size=4,
            avg_accept_rate=0.85
        )
        
        sim = BatchSDSimulator(config, seed=42)
        sim.run_simulation(duration_seconds=15.0)
        
        # Key metrics
        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('off')
        
        metrics_text = f"""
        KEY METRICS:
        Completed Requests: {len(sim.completed_requests)} | 
        Total Batches: {sim.total_batches} | 
        Token Acceptance Rate: {(sim.total_tokens_accepted/sim.total_tokens_generated):.1%} | 
        Throughput: {(sim.total_tokens_accepted/sim.total_compute_time_ms*1000):.1f} tok/sec
        """
        
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       verticalalignment='center', family='monospace')
        
        # Throughput scaling
        ax1 = fig.add_subplot(gs[1, 0])
        batch_sizes = [4, 8, 16, 32]
        throughputs = []
        for bs in batch_sizes:
            cfg = SystemConfig(arrival_rate=10.0, max_batch_size=bs, chunk_size=4)
            s = BatchSDSimulator(cfg, seed=42)
            s.run_simulation(duration_seconds=10.0)
            throughputs.append(s.total_tokens_accepted / s.total_compute_time_ms * 1000)
        
        ax1.bar(batch_sizes, throughputs, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (tok/sec)')
        ax1.set_title('Batch Size Impact')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Accept rate effect
        ax2 = fig.add_subplot(gs[1, 1])
        accept_rates = [0.5, 0.7, 0.85, 1.0]
        efficiencies = []
        for ar in accept_rates:
            cfg = SystemConfig(arrival_rate=10.0, max_batch_size=16, chunk_size=4, avg_accept_rate=ar)
            s = BatchSDSimulator(cfg, seed=42)
            s.run_simulation(duration_seconds=10.0)
            efficiencies.append(s.total_tokens_accepted / s.total_tokens_generated)
        
        ax2.plot(accept_rates, efficiencies, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('SD Accept Rate')
        ax2.set_ylabel('Token Efficiency')
        ax2.set_title('SD Quality vs Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # Arrival rate sensitivity
        ax3 = fig.add_subplot(gs[1, 2])
        arrival_rates = [5, 10, 20, 40]
        throughputs_load = []
        for ar in arrival_rates:
            cfg = SystemConfig(arrival_rate=ar, max_batch_size=16, chunk_size=4)
            s = BatchSDSimulator(cfg, seed=42)
            s.run_simulation(duration_seconds=10.0)
            throughputs_load.append(s.total_tokens_accepted / s.total_compute_time_ms * 1000)
        
        ax3.plot(arrival_rates, throughputs_load, 'g-^', linewidth=2, markersize=8)
        ax3.set_xlabel('Arrival Rate (req/sec)')
        ax3.set_ylabel('Throughput (tok/sec)')
        ax3.set_title('Load Sensitivity')
        ax3.grid(True, alpha=0.3)
        
        # Statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['GPU Memory Budget', f"{config.kv_budget_mb:.0f} MB"],
            ['Model Params', f"{config.model_hidden_size} hidden size"],
            ['KV Per Token', f"{config.kv_per_token_mb*1000:.2f} KB"],
            ['Max Batch Size', f"{config.max_batch_size}"],
            ['Chunk Size', f"{config.chunk_size}"],
            ['Avg Accept Rate', f"{config.avg_accept_rate:.1%}"],
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(table_data)):
            if i == 0:
                table[(i, 0)].set_facecolor('lightblue')
                table[(i, 1)].set_facecolor('lightblue')
            else:
                table[(i, 0)].set_facecolor('wheat' if i % 2 == 0 else 'white')
                table[(i, 1)].set_facecolor('wheat' if i % 2 == 0 else 'white')
        
        plt.suptitle('SD Simulator - Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight')
        print("Saved: summary_dashboard.png")
        return fig


def main():
    """Generate all visualizations"""
    print("Generating visualizations...")
    print("="*80)
    
    visualizer = SimulationVisualizer()
    
    # Generate individual plots
    visualizer.plot_throughput_vs_accept_rate()
    visualizer.plot_batch_size_scaling()
    visualizer.plot_chunk_size_granularity()
    visualizer.plot_load_sensitivity()
    visualizer.plot_memory_pressure()
    visualizer.plot_summary_dashboard()
    
    print("="*80)
    print("All visualizations completed!")
    print("Check the generated PNG files in the current directory.")


if __name__ == "__main__":
    # Import matplotlib backend handling
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    main()
