#!/usr/bin/env python3
"""
Benchmarking Suite for Prefix Sum with Odd-Positive Masking

Measures actual performance across implementations and problem sizes,
generating roofline and scaling plots with real data.

Usage:
    python benchmark.py                    # Run full benchmark
    python benchmark.py --quick            # Quick smoke test
    python benchmark.py --sizes 1024 4096  # Specific sizes only
    python benchmark.py --output results/  # Custom output directory
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np

# Check for CUDA availability before importing torch
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

if CUDA_AVAILABLE:
    try:
        import triton
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False
        print("WARNING: Triton not installed. GPU kernels will not be available.")
else:
    TRITON_AVAILABLE = False
    print("WARNING: CUDA not available. Running CPU-only benchmarks.")

from prefix_sum import (
    prefix_sum_reference,
    prefix_sum_single_block,
    prefix_sum,
)


# =============================================================================
# Hardware Detection
# =============================================================================

@dataclass
class GPUInfo:
    name: str
    compute_capability: tuple
    total_memory_gb: float
    sm_count: int
    clock_mhz: int
    memory_bandwidth_gb_s: float  # Estimated
    peak_fp32_tflops: float       # Estimated


def detect_gpu() -> Optional[GPUInfo]:
    """Detect GPU and estimate its capabilities."""
    if not CUDA_AVAILABLE:
        return None

    props = torch.cuda.get_device_properties(0)

    # Estimate memory bandwidth based on known GPUs
    # These are approximate theoretical peaks
    bandwidth_estimates = {
        "A10G": 600,
        "A10": 600,
        "A100": 2039,
        "H100": 3350,
        "V100": 900,
        "T4": 300,
        "RTX 3090": 936,
        "RTX 4090": 1008,
    }

    # Estimate peak FP32 TFLOPS
    tflops_estimates = {
        "A10G": 31.2,
        "A10": 31.2,
        "A100": 19.5,  # FP32 without tensor cores
        "H100": 67.0,
        "V100": 15.7,
        "T4": 8.1,
        "RTX 3090": 35.6,
        "RTX 4090": 82.6,
    }

    # Find matching GPU
    gpu_name = props.name
    bandwidth = 600  # Default estimate
    tflops = 30.0    # Default estimate

    for known_gpu, bw in bandwidth_estimates.items():
        if known_gpu.lower() in gpu_name.lower():
            bandwidth = bw
            tflops = tflops_estimates.get(known_gpu, 30.0)
            break

    return GPUInfo(
        name=props.name,
        compute_capability=(props.major, props.minor),
        total_memory_gb=props.total_memory / (1024**3),
        sm_count=props.multi_processor_count,
        clock_mhz=props.clock_rate // 1000,
        memory_bandwidth_gb_s=bandwidth,
        peak_fp32_tflops=tflops,
    )


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    implementation: str
    problem_size: int
    iterations: int
    warmup_iterations: int

    # Timing statistics (milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float

    # Derived metrics
    throughput_gflops: float
    memory_bandwidth_gb_s: float
    operational_intensity: float

    # Stability metrics
    coefficient_of_variation: float
    is_stable: bool

    # Metadata
    timestamp: str
    gpu_name: str


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: list
    gpu_info: Optional[GPUInfo]
    timestamp: str
    python_version: str
    torch_version: str
    triton_version: Optional[str]


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_implementation(
    fn,
    x: torch.Tensor,
    warmup: int = 50,
    iterations: int = 200,
    target_cv: float = 0.05,
    max_iterations: int = 1000,
    adaptive: bool = True,
) -> BenchmarkResult:
    """
    Benchmark a single implementation with adaptive iteration count.

    Args:
        fn: Function to benchmark
        x: Input tensor
        warmup: Number of warmup iterations
        iterations: Initial number of measurement iterations
        target_cv: Target coefficient of variation for stability
        max_iterations: Maximum iterations if adaptive
        adaptive: Whether to adaptively increase iterations for stability

    Returns:
        BenchmarkResult with timing statistics
    """
    n = x.numel()
    is_cuda = x.is_cuda

    # Warmup phase
    for _ in range(warmup):
        _ = fn(x)
        if is_cuda:
            torch.cuda.synchronize()

    # Measurement phase
    times_ms = []
    current_iterations = iterations

    while True:
        # Clear any cached allocations
        if is_cuda:
            torch.cuda.empty_cache()

        batch_times = []
        for _ in range(current_iterations):
            if is_cuda:
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = fn(x)

            if is_cuda:
                torch.cuda.synchronize()

            end = time.perf_counter()
            batch_times.append((end - start) * 1000)  # Convert to ms

        times_ms.extend(batch_times)

        # Check stability
        times_array = np.array(times_ms)
        mean = np.mean(times_array)
        std = np.std(times_array)
        cv = std / mean if mean > 0 else 0

        if not adaptive or cv <= target_cv or len(times_ms) >= max_iterations:
            break

        # Need more iterations
        current_iterations = min(100, max_iterations - len(times_ms))
        if current_iterations <= 0:
            break

    # Compute statistics
    times_array = np.array(times_ms)
    mean_ms = float(np.mean(times_array))
    std_ms = float(np.std(times_array))
    min_ms = float(np.min(times_array))
    max_ms = float(np.max(times_array))
    median_ms = float(np.median(times_array))
    cv = std_ms / mean_ms if mean_ms > 0 else 0

    # Compute derived metrics
    # Operations per element: ~6 (comparison, 2 cumsums, subtract, mask, select)
    ops_per_element = 6
    total_ops = n * ops_per_element
    throughput_gflops = (total_ops / (mean_ms / 1000)) / 1e9 if mean_ms > 0 else 0

    # Memory: read input (4 bytes) + write output (4 bytes) = 8 bytes per element
    bytes_per_element = 8
    total_bytes = n * bytes_per_element
    memory_bw = (total_bytes / (mean_ms / 1000)) / 1e9 if mean_ms > 0 else 0

    # Operational intensity
    oi = ops_per_element / bytes_per_element

    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    return BenchmarkResult(
        implementation=fn.__name__,
        problem_size=n,
        iterations=len(times_ms),
        warmup_iterations=warmup,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        median_ms=median_ms,
        throughput_gflops=throughput_gflops,
        memory_bandwidth_gb_s=memory_bw,
        operational_intensity=oi,
        coefficient_of_variation=cv,
        is_stable=cv <= target_cv,
        timestamp=datetime.now().isoformat(),
        gpu_name=gpu_name,
    )


def run_benchmarks(
    sizes: list[int],
    warmup: int = 50,
    iterations: int = 200,
    adaptive: bool = True,
    verbose: bool = True,
) -> BenchmarkSuite:
    """
    Run benchmarks across all implementations and sizes.

    Args:
        sizes: List of problem sizes to benchmark
        warmup: Warmup iterations per benchmark
        iterations: Measurement iterations per benchmark
        adaptive: Whether to use adaptive iteration count
        verbose: Print progress

    Returns:
        BenchmarkSuite with all results
    """
    results = []
    gpu_info = detect_gpu()

    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK CONFIGURATION")
        print("=" * 70)
        if gpu_info:
            print(f"GPU: {gpu_info.name}")
            print(f"  Compute Capability: {gpu_info.compute_capability}")
            print(f"  SMs: {gpu_info.sm_count}")
            print(f"  Memory: {gpu_info.total_memory_gb:.1f} GB")
            print(f"  Est. Bandwidth: {gpu_info.memory_bandwidth_gb_s} GB/s")
            print(f"  Est. Peak FP32: {gpu_info.peak_fp32_tflops} TFLOPS")
        else:
            print("GPU: None (CPU only)")
        print(f"Problem sizes: {sizes}")
        print(f"Warmup: {warmup}, Iterations: {iterations}, Adaptive: {adaptive}")
        print("=" * 70 + "\n")

    # Define implementations to benchmark
    implementations = [
        ("reference", prefix_sum_reference, False),  # CPU only
    ]

    if CUDA_AVAILABLE and TRITON_AVAILABLE:
        implementations.extend([
            ("single_block", prefix_sum_single_block, True),
            ("multi_block", prefix_sum, True),
        ])

    for size in sizes:
        if verbose:
            print(f"\n--- Problem size: n = {size:,} ---")

        for name, fn, needs_cuda in implementations:
            # Skip GPU implementations if no CUDA
            if needs_cuda and not CUDA_AVAILABLE:
                continue

            # Single-block only works for n <= 1024
            if name == "single_block" and size > 1024:
                if verbose:
                    print(f"  {name}: SKIPPED (n > 1024)")
                continue

            # Create input tensor
            if needs_cuda:
                x = torch.randn(size, device='cuda', dtype=torch.float32)
            else:
                x = torch.randn(size, dtype=torch.float32)

            if verbose:
                print(f"  {name}: ", end="", flush=True)

            try:
                result = benchmark_implementation(
                    fn, x,
                    warmup=warmup,
                    iterations=iterations,
                    adaptive=adaptive,
                )
                results.append(result)

                if verbose:
                    stability = "STABLE" if result.is_stable else f"UNSTABLE (CV={result.coefficient_of_variation:.1%})"
                    print(f"{result.mean_ms:.4f} ms (±{result.std_ms:.4f}), "
                          f"{result.throughput_gflops:.1f} GFLOP/s, "
                          f"{result.memory_bandwidth_gb_s:.1f} GB/s, "
                          f"{stability}")

            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")

    # Build suite
    suite = BenchmarkSuite(
        results=results,
        gpu_info=gpu_info,
        timestamp=datetime.now().isoformat(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        triton_version=triton.__version__ if TRITON_AVAILABLE else None,
    )

    return suite


# =============================================================================
# Results Export
# =============================================================================

def save_results(suite: BenchmarkSuite, output_dir: str = "results"):
    """Save benchmark results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results as JSON
    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, 'w') as f:
        data = {
            "gpu_info": asdict(suite.gpu_info) if suite.gpu_info else None,
            "timestamp": suite.timestamp,
            "python_version": suite.python_version,
            "torch_version": suite.torch_version,
            "triton_version": suite.triton_version,
            "results": [asdict(r) for r in suite.results],
        }
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {json_path}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    with open(csv_path, 'w') as f:
        headers = [
            "implementation", "problem_size", "mean_ms", "std_ms",
            "throughput_gflops", "memory_bandwidth_gb_s", "cv", "stable"
        ]
        f.write(",".join(headers) + "\n")

        for r in suite.results:
            row = [
                r.implementation,
                str(r.problem_size),
                f"{r.mean_ms:.6f}",
                f"{r.std_ms:.6f}",
                f"{r.throughput_gflops:.2f}",
                f"{r.memory_bandwidth_gb_s:.2f}",
                f"{r.coefficient_of_variation:.4f}",
                str(r.is_stable),
            ]
            f.write(",".join(row) + "\n")

    print(f"Summary saved to: {csv_path}")

    return json_path, csv_path


# =============================================================================
# Plotting
# =============================================================================

def plot_results(suite: BenchmarkSuite, output_dir: str = "results"):
    """Generate plots from benchmark results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed. Skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Organize results by implementation
    by_impl = {}
    for r in suite.results:
        if r.implementation not in by_impl:
            by_impl[r.implementation] = []
        by_impl[r.implementation].append(r)

    # Sort by problem size
    for impl in by_impl:
        by_impl[impl].sort(key=lambda x: x.problem_size)

    # Color scheme
    colors = {
        "prefix_sum_reference": "#e74c3c",
        "prefix_sum_single_block": "#3498db",
        "prefix_sum": "#2ecc71",
    }

    markers = {
        "prefix_sum_reference": "X",
        "prefix_sum_single_block": "o",
        "prefix_sum": "^",
    }

    labels = {
        "prefix_sum_reference": "Reference (CPU)",
        "prefix_sum_single_block": "Single-Block Triton",
        "prefix_sum": "Multi-Block Triton",
    }

    # =========================================================================
    # Plot 1: Performance vs Problem Size
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for impl, results in by_impl.items():
        sizes = [r.problem_size for r in results]
        throughputs = [r.throughput_gflops for r in results]
        stds = [r.std_ms / r.mean_ms * r.throughput_gflops for r in results]  # Propagate error

        color = colors.get(impl, "gray")
        marker = markers.get(impl, "s")
        label = labels.get(impl, impl)

        ax1.errorbar(sizes, throughputs, yerr=stds, fmt=f'{marker}-',
                     color=color, label=label, capsize=3, markersize=8, linewidth=2)

    ax1.set_xscale('log')
    ax1.set_xlabel('Problem Size (n)', fontsize=11)
    ax1.set_ylabel('Throughput (GFLOP/s)', fontsize=11)
    ax1.set_title('Throughput vs Problem Size (Measured)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Add block limit line
    ax1.axvline(x=1024, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(1024, ax1.get_ylim()[1] * 0.9, 'n=1024\n(block limit)',
             ha='center', fontsize=9, color='gray')

    # =========================================================================
    # Plot 2: Memory Bandwidth Achieved
    # =========================================================================
    for impl, results in by_impl.items():
        sizes = [r.problem_size for r in results]
        bandwidths = [r.memory_bandwidth_gb_s for r in results]

        color = colors.get(impl, "gray")
        marker = markers.get(impl, "s")
        label = labels.get(impl, impl)

        ax2.semilogx(sizes, bandwidths, f'{marker}-', color=color, label=label,
                     markersize=8, linewidth=2)

    # Add theoretical peak bandwidth line
    if suite.gpu_info:
        ax2.axhline(y=suite.gpu_info.memory_bandwidth_gb_s, color='black',
                    linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Peak BW ({suite.gpu_info.memory_bandwidth_gb_s} GB/s)')

    ax2.set_xlabel('Problem Size (n)', fontsize=11)
    ax2.set_ylabel('Memory Bandwidth (GB/s)', fontsize=11)
    ax2.set_title('Memory Bandwidth Achieved (Measured)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add block limit line
    ax2.axvline(x=1024, color='gray', linestyle=':', linewidth=2, alpha=0.7)

    plt.tight_layout()

    # Save
    plot_path = os.path.join(output_dir, f"benchmark_scaling_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Scaling plot saved to: {plot_path}")
    plt.close()

    # =========================================================================
    # Plot 3: Roofline Model with Measured Data
    # =========================================================================
    if suite.gpu_info:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw roofline
        peak_gflops = suite.gpu_info.peak_fp32_tflops * 1000
        mem_bw = suite.gpu_info.memory_bandwidth_gb_s
        ridge_point = peak_gflops / mem_bw

        x = np.logspace(-2, 2, 1000)
        memory_bound = mem_bw * x
        compute_bound = np.full_like(x, peak_gflops)
        roofline = np.minimum(memory_bound, compute_bound)

        ax.loglog(x, roofline, 'k-', linewidth=3, label=f'{suite.gpu_info.name} Roofline')
        ax.fill_between(x, roofline, 0.01, alpha=0.1, color='gray')

        # Plot measured points
        for impl, results in by_impl.items():
            ois = [r.operational_intensity for r in results]
            throughputs = [r.throughput_gflops for r in results]
            sizes = [r.problem_size for r in results]

            color = colors.get(impl, "gray")
            marker = markers.get(impl, "s")
            label = labels.get(impl, impl)

            # Plot with size annotations
            scatter = ax.scatter(ois, throughputs, c=color, marker=marker, s=100,
                                label=label, edgecolors='black', linewidths=1, zorder=5)

            # Annotate with problem sizes
            for oi, tp, sz in zip(ois, throughputs, sizes):
                ax.annotate(f'n={sz}', (oi, tp), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, alpha=0.7)

        ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.5)
        ax.text(ridge_point, peak_gflops * 0.1, f'Ridge\n({ridge_point:.1f})',
                ha='center', fontsize=9, color='gray')

        ax.set_xlim(0.1, 10)
        ax.set_ylim(0.1, peak_gflops * 2)
        ax.set_xlabel('Operational Intensity (FLOP/byte)', fontsize=11)
        ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
        ax.set_title(f'Roofline Model: Measured Performance on {suite.gpu_info.name}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, which='both', alpha=0.3)

        roofline_path = os.path.join(output_dir, f"benchmark_roofline_{timestamp}.png")
        plt.savefig(roofline_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Roofline plot saved to: {roofline_path}")
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark prefix sum implementations")

    parser.add_argument('--sizes', type=int, nargs='+',
                        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
                        help='Problem sizes to benchmark')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Measurement iterations')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive iteration count')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick benchmark with fewer sizes/iterations')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.sizes = [512, 1024, 4096]
        args.warmup = 10
        args.iterations = 50

    # Run benchmarks
    suite = run_benchmarks(
        sizes=args.sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        adaptive=not args.no_adaptive,
    )

    # Save results
    save_results(suite, args.output)

    # Generate plots
    if not args.no_plot:
        plot_results(suite, args.output)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
