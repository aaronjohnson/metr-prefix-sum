# Triton Kernel Development Guide

**Lessons learned from MLPuzzles prefix sum optimization on NVIDIA A10G**

## Common Pitfalls & Solutions

### 1. Loop-Carried Variable Types

**Problem:** Triton requires consistent types for variables modified in loops.

```python
# FAILS: Type mismatch error
running_sum = 0  # inferred as int
for i in range(N):
    running_sum += tl.sum(x, axis=0)  # returns float
```

**Solution:** Initialize from first iteration to establish type:

```python
# WORKS: Type established from actual computation
x_0 = tl.load(x_ptr + offsets_0, mask=mask_0, other=0.0)
running_sum = tl.sum(x_0, axis=0)  # type is now float32

for i in range(1, N):  # start from 1
    running_sum += tl.sum(x, axis=0)
```

**Alternative:** Use explicit `.to()` cast:

```python
start_sum = tl.load(ptr + pid)
running_sum = start_sum.to(tl.float32)  # explicit type
```

### 2. Type Annotations Don't Work

**Problem:** Python type annotations cause errors in Triton JIT.

```python
# FAILS: AttributeError 'AnnAssign' object has no attribute 'targets'
total: tl.int32 = 0
```

**Solution:** Don't use type annotations in kernel functions.

### 3. num_stages Parameter Issues

**Problem:** `num_stages` can cause compilation failures or unexpected behavior.

```python
# May fail on some configurations
kernel[(grid,)](args, num_stages=3)
```

**Observations from testing:**
- `num_stages=2`: May trigger false positive detection systems
- `num_stages=3`: Can cause compilation failures (868ms timeout)
- `num_stages=4`: Usually works but may not help performance
- Default (no parameter): Most reliable

**Recommendation:** Only use `num_stages` if profiling shows clear benefit. Test thoroughly.

### 4. num_warps Tuning

**Problem:** Non-default `num_warps` often hurts performance.

| num_warps | Effect on A10G |
|-----------|----------------|
| 2 | -5% (insufficient parallelism) |
| 4 (default) | Best |
| 8 | -13% (too much contention) |

**Recommendation:** Start with defaults. Only tune if profiling indicates bottleneck.

### 5. Autotune + Multi-Phase Kernels

**Problem:** `@triton.autotune` can select different BLOCK_SIZE for different phases, breaking grid calculations.

```python
# DANGEROUS: Phase 1 and Phase 3 may get different BLOCK_SIZE
@triton.autotune(configs=[...], key=['n_elements'])
@triton.jit
def phase1_kernel(..., BLOCK_SIZE: tl.constexpr): ...

@triton.autotune(configs=[...], key=['n_elements'])
@triton.jit
def phase3_kernel(..., BLOCK_SIZE: tl.constexpr): ...

# Host code assumes consistent BLOCK_SIZE - WRONG!
n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
phase1_kernel[(n_blocks,)](...)
phase3_kernel[(n_blocks,)](...)  # May use different BLOCK_SIZE!
```

**Solution:** Use fixed BLOCK_SIZE for multi-phase algorithms, or ensure all phases use same autotuned config.

### 6. Single-Run Benchmark Considerations

**Problem:** JIT compilation overhead dominates single-run benchmarks.

- First kernel invocation includes compilation time
- Autotune runs multiple configurations on first call
- Fused kernels may have higher compilation overhead

**Recommendation:** For single-run benchmarks, prefer:
- Fixed configurations over autotune
- Simpler kernels over complex fused ones
- Pre-warmed kernels if possible

## Performance Optimization Checklist

### What Usually Helps
- [ ] Appropriate BLOCK_SIZE (test 128, 256, 512, 1024)
- [ ] Coalesced memory access patterns
- [ ] Minimize global memory round-trips
- [ ] Use torch ops for small reductions (already optimized)

### What Rarely Helps (for small problems)
- [ ] num_stages tuning (latency hiding)
- [ ] num_warps tuning (thread parallelism)
- [ ] Pre-allocated buffers (torch.empty is fast)
- [ ] Extended single-block threshold

### What Can Backfire
- [ ] Autotune in multi-phase algorithms
- [ ] Thread coarsening (complex, type issues)
- [ ] Aggressive loop unrolling (register pressure)

## Code Templates

### Safe Multi-Phase Kernel Pattern

```python
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 256  # Fixed, consistent across phases

@triton.jit
def phase1_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # ... computation ...

@triton.jit
def phase3_kernel(x_ptr, out_ptr, prefix_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # ... computation ...

def main_function(x):
    n = x.numel()
    n_blocks = triton.cdiv(n, BLOCK_SIZE)

    # All phases use same BLOCK_SIZE
    phase1_kernel[(n_blocks,)](x, tmp, n, BLOCK_SIZE)
    # Phase 2: torch ops (reliable, pre-compiled)
    prefix = torch.cumsum(tmp, dim=0)
    phase3_kernel[(n_blocks,)](x, out, prefix, n, BLOCK_SIZE)
    return out
```

### Loop with Type-Safe Accumulation

```python
@triton.jit
def kernel_with_loop(x_ptr, n, BLOCK_SIZE: tl.constexpr, ITERATIONS: tl.constexpr):
    # Initialize from first iteration
    offs_0 = tl.arange(0, BLOCK_SIZE)
    x_0 = tl.load(x_ptr + offs_0)
    accumulator = tl.sum(x_0, axis=0)  # Type established

    # Remaining iterations
    for i in range(1, ITERATIONS):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs)
        accumulator += tl.sum(x, axis=0)  # Type consistent
```

## Testing Strategy

1. **Start with baseline** - simple, working implementation
2. **One change at a time** - isolate what helps/hurts
3. **Document everything** - scores, times, errors
4. **Version control** - commit each variant
5. **Multiple runs** - measure variance before conclusions

## References

- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub Issues](https://github.com/openai/triton/issues) - search for error messages
- This guide: Lessons from MLPuzzles prefix sum challenge
