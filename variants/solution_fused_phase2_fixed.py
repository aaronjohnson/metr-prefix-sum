"""
Prefix Sum with Odd-Positive Masking
Author: Aaron Johnson
GitHub: https://github.com/aaronjohnson/metr-prefix-sum
LinkedIn: https://www.linkedin.com/in/aaronmarkjohnson/

Optimization: Fused Phase 2 with fixed BLOCK_SIZE (no autotune)
- Single Triton kernel for Phase 2 instead of torch ops
- Avoids autotune JIT overhead that broke previous attempt
- Reduces kernel launch overhead
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_sum_single_block_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    is_positive = (x > 0).to(tl.int32)
    pos_count_inclusive = tl.cumsum(is_positive, axis=0)
    pos_count_exclusive = pos_count_inclusive - is_positive
    include_mask = (pos_count_exclusive & 1) == 1
    masked_x = tl.where(include_mask, x, 0.0)
    result = tl.cumsum(masked_x, axis=0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _prefix_sum_phase1_kernel(x_ptr, block_pos_counts_ptr, block_sums_even_ptr, block_sums_odd_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    is_positive = (x > 0).to(tl.int32)
    total_positives = tl.sum(is_positive, axis=0)
    pos_count_inclusive = tl.cumsum(is_positive, axis=0)
    pos_count_exclusive = pos_count_inclusive - is_positive
    include_mask_even = (pos_count_exclusive & 1) == 1
    block_sum_even = tl.sum(tl.where(include_mask_even, x, 0.0), axis=0)
    include_mask_odd = (pos_count_exclusive & 1) == 0
    block_sum_odd = tl.sum(tl.where(include_mask_odd, x, 0.0), axis=0)
    tl.store(block_pos_counts_ptr + pid, total_positives)
    tl.store(block_sums_even_ptr + pid, block_sum_even)
    tl.store(block_sums_odd_ptr + pid, block_sum_odd)


@triton.jit
def _prefix_sum_phase2_kernel(
    block_pos_counts_ptr, block_sums_even_ptr, block_sums_odd_ptr,
    block_pos_prefix_ptr, block_sum_prefix_ptr,
    n_blocks, BLOCK_SIZE: tl.constexpr
):
    """Fused Phase 2: compute both prefix sums in a single kernel."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    # Load block aggregates
    pos_counts = tl.load(block_pos_counts_ptr + offsets, mask=mask, other=0)
    sums_even = tl.load(block_sums_even_ptr + offsets, mask=mask, other=0.0)
    sums_odd = tl.load(block_sums_odd_ptr + offsets, mask=mask, other=0.0)

    # Exclusive prefix sum of position counts
    pos_prefix_inclusive = tl.cumsum(pos_counts, axis=0)
    pos_prefix_exclusive = pos_prefix_inclusive - pos_counts

    # Select correct sum based on parity of starting position
    start_is_odd = (pos_prefix_exclusive & 1) == 1
    selected_sums = tl.where(start_is_odd, sums_odd, sums_even)

    # Exclusive prefix sum of selected sums
    sum_prefix_inclusive = tl.cumsum(selected_sums, axis=0)
    sum_prefix_exclusive = sum_prefix_inclusive - selected_sums

    tl.store(block_pos_prefix_ptr + offsets, pos_prefix_exclusive, mask=mask)
    tl.store(block_sum_prefix_ptr + offsets, sum_prefix_exclusive, mask=mask)


@triton.jit
def _prefix_sum_phase3_kernel(x_ptr, out_ptr, block_pos_prefix_ptr, block_sum_prefix_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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


def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    if n_elements == 0:
        return x.clone()

    BLOCK_SIZE = 256

    if n_elements <= BLOCK_SIZE:
        out = torch.empty_like(x)
        BS = min(triton.next_power_of_2(n_elements), 1024)
        _prefix_sum_single_block_kernel[(1,)](x, out, n_elements, BS)
        return out

    out = torch.empty_like(x)
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    block_pos_counts = torch.empty(n_blocks, dtype=torch.int32, device=x.device)
    block_sums_even = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    block_sums_odd = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    block_pos_prefix = torch.empty(n_blocks, dtype=torch.int32, device=x.device)
    block_sum_prefix = torch.empty(n_blocks, dtype=x.dtype, device=x.device)

    # Phase 1
    _prefix_sum_phase1_kernel[(n_blocks,)](
        x, block_pos_counts, block_sums_even, block_sums_odd, n_elements, BLOCK_SIZE
    )

    # Phase 2 - Fused Triton kernel (fixed BLOCK_SIZE, no autotune)
    phase2_block_size = triton.next_power_of_2(n_blocks)
    if phase2_block_size <= 1024:
        _prefix_sum_phase2_kernel[(1,)](
            block_pos_counts, block_sums_even, block_sums_odd,
            block_pos_prefix, block_sum_prefix,
            n_blocks, phase2_block_size
        )
    else:
        # Fallback to torch for very large n_blocks
        block_pos_prefix.zero_()
        block_sum_prefix.zero_()
        if n_blocks > 1:
            block_pos_prefix[1:] = torch.cumsum(block_pos_counts[:-1], dim=0)
            start_is_even = (block_pos_prefix % 2) == 0
            selected = torch.where(start_is_even, block_sums_even, block_sums_odd)
            block_sum_prefix[1:] = torch.cumsum(selected[:-1], dim=0)

    # Phase 3
    _prefix_sum_phase3_kernel[(n_blocks,)](
        x, out, block_pos_prefix, block_sum_prefix, n_elements, BLOCK_SIZE
    )
    return out
