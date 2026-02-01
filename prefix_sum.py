"""
Prefix Sum with Odd-Positive Masking - Triton Implementation
https://mlpuzzles.com/

Problem: Compute prefix sum where position i is accumulated only if
the count of positive values in x[0:i] (exclusive) is odd.

Author: Aaron Johnson
GitHub: https://github.com/aaronjohnson/metr-prefix-sum
LinkedIn: https://www.linkedin.com/in/aaronmarkjohnson/
"""

import numpy as np

# Optional torch/triton imports (for GPU acceleration)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


# -----------------------------------------------------------------------------
# NumPy Reference Implementation (no torch/triton required)
# -----------------------------------------------------------------------------

def prefix_sum_numpy(x: np.ndarray) -> np.ndarray:
    """Pure NumPy reference implementation - works without torch."""
    n = len(x)
    out = np.zeros_like(x)

    positive_count = 0
    running_sum = 0.0

    for i in range(n):
        if positive_count % 2 == 1:
            running_sum += x[i]
        out[i] = running_sum

        if x[i] > 0:
            positive_count += 1

    return out


# -----------------------------------------------------------------------------
# Reference Implementation (works with torch.Tensor or numpy.ndarray)
# -----------------------------------------------------------------------------

def prefix_sum_reference(x):
    """
    Reference implementation - works with torch.Tensor or numpy.ndarray.
    """
    if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
        return _prefix_sum_reference_torch(x)
    else:
        return prefix_sum_numpy(np.asarray(x))


def _prefix_sum_reference_torch(x):
    """PyTorch reference implementation."""
    n = x.numel()
    out = torch.zeros_like(x)

    positive_count = 0
    running_sum = 0.0

    for i in range(n):
        if positive_count % 2 == 1:
            running_sum += x[i].item()
        out[i] = running_sum

        if x[i] > 0:
            positive_count += 1

    return out


# -----------------------------------------------------------------------------
# Triton Kernels (require torch + triton + CUDA)
# -----------------------------------------------------------------------------

if TRITON_AVAILABLE and triton is not None:

    @triton.jit
    def _prefix_sum_single_block_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Single-block kernel for small inputs (n <= BLOCK_SIZE)."""
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load input
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Step 1: Compute INCLUSIVE prefix count of positives
        is_positive = (x > 0).to(tl.int32)
        pos_count_inclusive = tl.cumsum(is_positive, axis=0)

        # Step 2: Convert to EXCLUSIVE prefix
        pos_count_exclusive = pos_count_inclusive - is_positive

        # Step 3: Determine mask (odd count = include)
        include_mask = (pos_count_exclusive & 1) == 1

        # Step 4: Apply mask and compute prefix sum
        masked_x = tl.where(include_mask, x, 0.0)
        result = tl.cumsum(masked_x, axis=0)

        # Store output
        tl.store(out_ptr + offsets, result, mask=mask)


    def prefix_sum_single_block(x):
        """Wrapper for single-block kernel."""
        assert x.is_cuda, "Input must be on CUDA"
        out = torch.empty_like(x)
        n_elements = x.numel()

        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

        if n_elements > BLOCK_SIZE:
            raise ValueError(f"Input size {n_elements} exceeds single block capacity {BLOCK_SIZE}")

        _prefix_sum_single_block_kernel[(1,)](x, out, n_elements, BLOCK_SIZE)
        return out


    @triton.jit
    def _prefix_sum_phase1_kernel(
        x_ptr,
        block_pos_counts_ptr,
        block_sums_even_ptr,
        block_sums_odd_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Phase 1: Each block computes local aggregates.

        Key insight: The mask depends on parity of (global_start + local_exclusive_count).
        - If global_start is EVEN: include where local_exclusive_count is ODD
        - If global_start is ODD:  include where local_exclusive_count is EVEN (flipped!)

        We compute both sums so Phase 2 can select the correct one.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        is_positive = (x > 0).to(tl.int32)
        total_positives = tl.sum(is_positive, axis=0)

        pos_count_inclusive = tl.cumsum(is_positive, axis=0)
        pos_count_exclusive = pos_count_inclusive - is_positive

        # Even start: include where local count is odd
        include_mask_even = (pos_count_exclusive & 1) == 1
        block_sum_even = tl.sum(tl.where(include_mask_even, x, 0.0), axis=0)

        # Odd start: include where local count is even (FLIPPED)
        include_mask_odd = (pos_count_exclusive & 1) == 0
        block_sum_odd = tl.sum(tl.where(include_mask_odd, x, 0.0), axis=0)

        tl.store(block_pos_counts_ptr + pid, total_positives)
        tl.store(block_sums_even_ptr + pid, block_sum_even)
        tl.store(block_sums_odd_ptr + pid, block_sum_odd)


    @triton.jit
    def _prefix_sum_phase3_kernel(
        x_ptr,
        out_ptr,
        block_pos_prefix_ptr,
        block_sum_prefix_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Phase 3: Each block recomputes with correct starting offset."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        start_pos_count = tl.load(block_pos_prefix_ptr + pid)

        is_positive = (x > 0).to(tl.int32)
        pos_count_inclusive = tl.cumsum(is_positive, axis=0)
        pos_count_exclusive = pos_count_inclusive - is_positive

        global_pos_count = pos_count_exclusive + start_pos_count
        include_mask = (global_pos_count & 1) == 1

        masked_x = tl.where(include_mask, x, 0.0)
        local_result = tl.cumsum(masked_x, axis=0)

        start_sum = tl.load(block_sum_prefix_ptr + pid)
        result = local_result + start_sum

        tl.store(out_ptr + offsets, result, mask=mask)


    def prefix_sum(x):
        """
        Main entry point - handles arbitrary input sizes.

        v1-gpu-phase2: All phases now run on GPU, eliminating the O(n) CPU bottleneck.
        """
        assert x.is_cuda, "Input must be on CUDA"
        n_elements = x.numel()

        if n_elements == 0:
            return x.clone()

        BLOCK_SIZE = 1024

        if n_elements <= BLOCK_SIZE:
            return prefix_sum_single_block(x)

        # Multi-block path
        out = torch.empty_like(x)
        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

        block_pos_counts = torch.empty(n_blocks, dtype=torch.int32, device=x.device)
        block_sums_even = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
        block_sums_odd = torch.empty(n_blocks, dtype=x.dtype, device=x.device)

        # Phase 1: Compute block aggregates (GPU)
        _prefix_sum_phase1_kernel[(n_blocks,)](
            x, block_pos_counts, block_sums_even, block_sums_odd, n_elements, BLOCK_SIZE
        )

        # Phase 2: Compute prefix sums of block aggregates (ALL GPU - no CPU loops!)
        block_pos_prefix = torch.zeros(n_blocks, dtype=torch.int32, device=x.device)
        if n_blocks > 1:
            block_pos_prefix[1:] = torch.cumsum(block_pos_counts[:-1], dim=0)

        # Select correct block sum based on parity
        start_is_even = (block_pos_prefix % 2) == 0
        block_sums_selected = torch.where(start_is_even, block_sums_even, block_sums_odd)

        block_sum_prefix = torch.zeros(n_blocks, dtype=x.dtype, device=x.device)
        if n_blocks > 1:
            block_sum_prefix[1:] = torch.cumsum(block_sums_selected[:-1], dim=0)

        # Phase 3: Final computation with global offsets (GPU)
        _prefix_sum_phase3_kernel[(n_blocks,)](
            x, out, block_pos_prefix, block_sum_prefix, n_elements, BLOCK_SIZE
        )

        return out

else:
    # Stubs when Triton is not available
    def prefix_sum_single_block(x):
        raise RuntimeError("Triton not available. Install with: pip install triton")

    def prefix_sum(x):
        raise RuntimeError("Triton not available. Install with: pip install triton")


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

def test_correctness():
    """Test against reference implementation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, testing with NumPy only")
        x = np.array([3.0, -1.0, 2.0, 5.0, -3.0])
        result = prefix_sum_numpy(x)
        expected = np.array([0.0, -1.0, 1.0, 1.0, 1.0])
        print(f"  NumPy test: {'PASS' if np.allclose(result, expected) else 'FAIL'}")
        return

    torch.manual_seed(42)

    test_cases = [
        torch.tensor([3.0, -1.0, 2.0, 5.0, -3.0]),
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        torch.tensor([1.0]),
        torch.randn(100),
        torch.randn(1000),
        torch.randn(2048),   # Multi-block test
        torch.randn(8192),   # Larger multi-block
    ]

    print("Testing correctness...")
    for i, x_cpu in enumerate(test_cases):
        if not torch.cuda.is_available():
            print(f"  Test {i+1}: SKIP (CUDA not available)")
            continue

        x = x_cpu.cuda()
        expected = prefix_sum_reference(x_cpu).cuda()

        result = prefix_sum(x)
        max_diff = (result - expected).abs().max().item()
        status = "PASS" if max_diff < 1e-4 else "FAIL"
        print(f"  Test {i+1} (n={x.numel():5d}): {status} (max_diff={max_diff:.2e})")

    print()


def benchmark():
    """Simple benchmark."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    import time

    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print("Benchmarking...")
    for n in sizes:
        x = torch.randn(n, device='cuda')

        # Warmup
        for _ in range(10):
            _ = prefix_sum(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = prefix_sum(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) / iterations * 1000
        print(f"  n={n:6d}: {avg_ms:.4f} ms")


if __name__ == "__main__":
    test_correctness()
    benchmark()
