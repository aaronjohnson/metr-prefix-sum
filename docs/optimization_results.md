# Optimization Results - MLPuzzles Prefix Sum

**Date:** 2026-02-01
**Hardware:** NVIDIA A10G (24GB GDDR6, 600 GB/s, 31.2 TFLOPS FP32)
**Author:** Aaron Johnson

## Summary

Best configuration: **BS=256 baseline** (no extra parameters)

## Results Table

| Config | Score (elem/sec) | Time (ms) | vs Best | Status |
|--------|------------------|-----------|---------|--------|
| **BS=256 baseline** | **429,769 - 433,770** | **0.221-0.223** | **baseline** | ✓ BEST |
| BS=1024 baseline | 401,006 | 0.2394 | -7% | ✓ |
| BS=512 | 428,015 | 0.2236 | -1% | ✓ |
| BS=128 | 422,837 | 0.2265 | -2% | ✓ |
| BS=256 + num_stages=4 | 424,979 | 0.2253 | -2% | ✓ |
| BS=256 + num_stages=3 | N/A | 868ms | N/A | ✗ Compile fail |
| BS=256 + num_stages=2 | 423,263 | 0.2263 | -2% | ⚠ False positive flag |
| BS=256 + num_warps=2 | 394,000 - 411,000 | 0.23-0.24 | -5% | ✓ |
| BS=256 + num_warps=8 | 373,842 | 0.2575 | -13% | ✓ |
| Single-block ≤1024 | 396,458 | 0.2422 | -8% | ✓ |
| BS=256 + preallocated | 425,842 | 0.2248 | -1.5% | ✓ |

## Key Findings

### Block Size (BLOCK_SIZE)
- **BS=256 is optimal** - best balance of parallelism vs overhead
- BS=512 is close second (-1%)
- BS=1024 too large, reduces parallelism (-7%)
- BS=128 too small, increases overhead (-2%)

### num_stages (Memory Latency Hiding)
- **Does not help** for this problem size
- num_stages=3 causes compilation failure
- num_stages=2 triggers false positive reward hacking detection (see bug report)
- num_stages=4 works but slightly slower

### num_warps (Thread Parallelism)
- **Default (4) is optimal**
- num_warps=2: -5% (not enough parallelism)
- num_warps=8: -13% (too much contention)

### Single-Block Threshold
- Extending to 1024 elements hurts performance (-8%)
- Suggests test input is >1024 elements OR multi-block with BS=256 is more efficient

### Pre-allocated Buffers
- **Does not help** (-1.5% vs baseline)
- `torch.empty()` is already fast on GPU
- Dictionary lookup overhead negates any savings
- Would only help with repeated calls (benchmark runs once)

## Algorithm Overview

Three-phase parallel prefix sum:
1. **Phase 1**: Each block computes local aggregates (pos count, even sum, odd sum)
2. **Phase 2**: torch.cumsum for block prefix sums (GPU, no CPU bottleneck)
3. **Phase 3**: Each block computes final result with global offsets

Key insight: Precompute both even-start and odd-start sums, select based on parity.

## Lessons Learned

1. Triton's defaults are well-tuned - don't over-optimize
2. Small problem sizes are dominated by overhead, not compute
3. Some optimizations trigger false positive detection (document and avoid)
4. Systematic testing with version control is essential

## Files

- `solution.py` - Best configuration (BS=256 baseline)
- `variants/solution_bs256_baseline.py` - Explicit baseline
- `variants/solution_bs256_stages2.py` - num_stages=2 (triggers false positive)
- `variants/solution_bs256_stages4.py` - num_stages=4
- `variants/solution_bs256_warps8.py` - num_warps=8
- `variants/solution_single_block_1024.py` - Extended single-block threshold
- `variants/solution_preallocated.py` - Pre-allocated buffers

## Final Conclusion

After exhaustive testing, **the baseline BS=256 configuration is optimal**. All attempted optimizations either:
- Made performance worse (num_warps, single-block threshold)
- Made marginal difference (num_stages, preallocated)
- Triggered bugs (num_stages=2/3 false positive detection)

Triton's defaults are well-tuned for this workload. The ~430K elem/sec score represents near-optimal performance for this algorithm on A10G hardware.
