# Prefix Sum with Odd-Positive Masking: GPU Optimization

Triton kernel optimization for the METR MLPuzzles benchmark.

## Motivation

This project originated from [3b1b.co/talent](https://3b1b.co/talent) (Grant Sanderson's talent search), which led to [mlpuzzles.com](https://mlpuzzles.com/) - a GPU kernel optimization challenge created by [METR](https://metr.org/) (Model Evaluation & Threat Research).

The challenge tests ability to optimize GPU kernels on real hardware (NVIDIA A10G), with submissions scored on correctness, execution time, and reviewed by Claude Opus for reward hacking detection.

This repo contains:
- Working implementations (reference + Triton)
- Benchmarking infrastructure for measured performance
- Theoretical analysis tools (roofline model, work-depth diagrams)
- CI pipeline for automated submission

## Problem

Compute a prefix sum where position `i` is accumulated only if the count of positive values in `x[0:i]` (exclusive) is odd.

```python
# Example
x = [3, -1, 2, 5, -3]
# Position 0: positive_count_before = 0 (even) → exclude
# Position 1: positive_count_before = 1 (odd)  → include (-1)
# Position 2: positive_count_before = 1 (odd)  → include (2)
# Position 3: positive_count_before = 2 (even) → exclude
# Position 4: positive_count_before = 3 (odd)  → include (-3)
# Output: [0, -1, 1, 1, -2]
```

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd metr
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Run tests
python prefix_sum.py

# Run benchmarks (requires CUDA GPU)
python benchmark.py --quick        # Quick test
python benchmark.py                # Full benchmark
python benchmark.py --help         # See all options
```

## Submission to mlpuzzles.com

### Option 1: GitHub Action (Automatic)

Push to `main` or `trunk` branch triggers automatic submission:
- Runs CPU correctness tests
- Submits to mlpuzzles.com A10G
- Posts results to commit/PR summary
- Updates leaderboard

### Option 2: Local Submission

```bash
# Submit prefix_sum.py
python submit.py

# Submit custom file
python submit.py --file my_solution.py

# View leaderboard
python submit.py --leaderboard

# Raw JSON output
python submit.py --json
```

### Option 3: Direct curl

```bash
curl -X POST https://puzzle.metr-dev.org/api/submit \
  -H "Content-Type: application/json" \
  -d "{\"code\": \"$(cat prefix_sum.py | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')\"}"
```

### Option 4: SSH Sandbox

```bash
ssh -p 2222 sandbox@puzzle.metr-dev.org  # password: puzzle
# Then: submit solution.py
```

## Benchmarking

The benchmark suite measures:
- **Throughput** (GFLOP/s)
- **Memory bandwidth** (GB/s)
- **Stability** (coefficient of variation < 5%)

Results are saved to `results/` as JSON and CSV, with plots.

```bash
# Full benchmark across sizes
python benchmark.py --sizes 256 512 1024 2048 4096 8192 16384 32768 65536

# Custom iterations for stability
python benchmark.py --warmup 100 --iterations 500

# Output to specific directory
python benchmark.py --output my_results/
```

## Implementations

| Implementation | Size Range | Description |
|---------------|------------|-------------|
| `prefix_sum_reference` | any n | CPU sequential (baseline) |
| `prefix_sum_single_block` | n ≤ 1024 | Single Triton block, optimal for small inputs |
| `prefix_sum` | any n | Multi-block with CPU Phase 2 (current bottleneck) |

## Analysis Tools

```bash
# Generate theoretical roofline model
python roofline_model.py

# Generate work-depth diagram
python work_depth_diagram.py
```

## Optimization Status

### Current Bottleneck

The multi-block implementation has a **CPU Phase 2** that processes elements sequentially, destroying GPU parallelism for n > 1024.

```
Phase 1 (GPU) → Phase 2 (CPU, O(n) sequential) → Phase 3 (GPU)
                        ↑
                   BOTTLENECK
```

### Optimization Priority

1. **Move Phase 2 to GPU** - Expected ~100x speedup for large n
2. **Operator fusion** - Reduce memory traffic
3. **Autotune block sizes** - Optimize for specific GPU

## Hardware Targets

Primary: **NVIDIA A10G** (AWS G5 instances, METR MLPuzzles)
- 31.2 TFLOPS FP32
- 600 GB/s memory bandwidth
- 84 SMs, 24GB GDDR6

## Files

```
├── prefix_sum.py           # Main implementation
├── benchmark.py            # Benchmarking suite
├── roofline_model.py       # Theoretical performance analysis
├── work_depth_diagram.py   # Parallelism visualization
├── requirements.txt        # Dependencies
└── results/                # Benchmark outputs (gitignored)
```

## References

- [METR MLPuzzles](https://mlpuzzles.com/)
- [Triton Documentation](https://triton-lang.org/)
- [Parallel Prefix Sum (Blelloch)](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
