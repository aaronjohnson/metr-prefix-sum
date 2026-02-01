"""
Prefix Sum with Odd-Positive Masking
Author: Aaron Johnson
GitHub: https://github.com/aaronjohnson/metr-prefix-sum
LinkedIn: https://www.linkedin.com/in/aaronmarkjohnson/

Optimization: Thread coarsening (ELEMENTS_PER_THREAD=4)
- Each thread processes 4 elements sequentially
- Reduces thread count, improves instruction-level parallelism
- Better register utilization
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
def _prefix_sum_phase1_coarsened_kernel(
    x_ptr, block_pos_counts_ptr, block_sums_even_ptr, block_sums_odd_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr, ELEMENTS_PER_THREAD: tl.constexpr
):
    pid = tl.program_id(0)
    # Each block processes BLOCK_SIZE * ELEMENTS_PER_THREAD elements
    block_start = pid * BLOCK_SIZE * ELEMENTS_PER_THREAD

    # Use first iteration to establish types, then accumulate
    # First chunk
    offsets_0 = block_start + tl.arange(0, BLOCK_SIZE)
    mask_0 = offsets_0 < n_elements
    x_0 = tl.load(x_ptr + offsets_0, mask=mask_0, other=0.0)
    is_positive_0 = (x_0 > 0).to(tl.int32)

    total_positives = tl.sum(is_positive_0, axis=0)
    running_pos_count = total_positives

    pos_exc_0 = tl.cumsum(is_positive_0, axis=0) - is_positive_0
    block_sum_even = tl.sum(tl.where((pos_exc_0 & 1) == 1, x_0, 0.0), axis=0)
    block_sum_odd = tl.sum(tl.where((pos_exc_0 & 1) == 0, x_0, 0.0), axis=0)

    # Process remaining chunks (first chunk handled above to establish types)
    for i in range(1, ELEMENTS_PER_THREAD):
        chunk_start = block_start + i * BLOCK_SIZE
        offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        is_positive = (x > 0).to(tl.int32)
        chunk_positives = tl.sum(is_positive, axis=0)

        pos_count_inclusive = tl.cumsum(is_positive, axis=0)
        pos_count_exclusive = pos_count_inclusive - is_positive + running_pos_count

        include_mask_even = (pos_count_exclusive & 1) == 1
        include_mask_odd = (pos_count_exclusive & 1) == 0

        block_sum_even += tl.sum(tl.where(include_mask_even, x, 0.0), axis=0)
        block_sum_odd += tl.sum(tl.where(include_mask_odd, x, 0.0), axis=0)

        total_positives += chunk_positives
        running_pos_count += chunk_positives

    tl.store(block_pos_counts_ptr + pid, total_positives)
    tl.store(block_sums_even_ptr + pid, block_sum_even)
    tl.store(block_sums_odd_ptr + pid, block_sum_odd)


@triton.jit
def _prefix_sum_phase3_coarsened_kernel(
    x_ptr, out_ptr, block_pos_prefix_ptr, block_sum_prefix_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr, ELEMENTS_PER_THREAD: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * ELEMENTS_PER_THREAD

    start_pos_count = tl.load(block_pos_prefix_ptr + pid)
    start_sum = tl.load(block_sum_prefix_ptr + pid)

    # Explicit type initialization to avoid loop-carried type mismatch
    running_pos_count = start_pos_count.to(tl.int32)
    running_sum = start_sum.to(tl.float32)

    for i in range(ELEMENTS_PER_THREAD):
        chunk_start = block_start + i * BLOCK_SIZE
        offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        is_positive = (x > 0).to(tl.int32)

        pos_count_inclusive = tl.cumsum(is_positive, axis=0)
        pos_count_exclusive = pos_count_inclusive - is_positive + running_pos_count

        include_mask = (pos_count_exclusive & 1) == 1
        masked_x = tl.where(include_mask, x, 0.0)
        local_result = tl.cumsum(masked_x, axis=0)

        tl.store(out_ptr + offsets, local_result + running_sum, mask=mask)

        # Update running state for next chunk
        chunk_positives = tl.sum(is_positive, axis=0)
        chunk_sum = tl.sum(masked_x, axis=0)
        running_pos_count += chunk_positives
        running_sum += chunk_sum


def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    if n_elements == 0:
        return x.clone()

    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4
    EFFECTIVE_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD  # 1024 elements per block

    if n_elements <= BLOCK_SIZE:
        out = torch.empty_like(x)
        BS = min(triton.next_power_of_2(n_elements), 1024)
        _prefix_sum_single_block_kernel[(1,)](x, out, n_elements, BS)
        return out

    out = torch.empty_like(x)
    n_blocks = triton.cdiv(n_elements, EFFECTIVE_BLOCK)

    block_pos_counts = torch.empty(n_blocks, dtype=torch.int32, device=x.device)
    block_sums_even = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    block_sums_odd = torch.empty(n_blocks, dtype=x.dtype, device=x.device)

    _prefix_sum_phase1_coarsened_kernel[(n_blocks,)](
        x, block_pos_counts, block_sums_even, block_sums_odd,
        n_elements, BLOCK_SIZE, ELEMENTS_PER_THREAD
    )

    block_pos_prefix = torch.zeros(n_blocks, dtype=torch.int32, device=x.device)
    if n_blocks > 1:
        block_pos_prefix[1:] = torch.cumsum(block_pos_counts[:-1], dim=0)
    start_is_even = (block_pos_prefix % 2) == 0
    block_sums_selected = torch.where(start_is_even, block_sums_even, block_sums_odd)
    block_sum_prefix = torch.zeros(n_blocks, dtype=x.dtype, device=x.device)
    if n_blocks > 1:
        block_sum_prefix[1:] = torch.cumsum(block_sums_selected[:-1], dim=0)

    _prefix_sum_phase3_coarsened_kernel[(n_blocks,)](
        x, out, block_pos_prefix, block_sum_prefix,
        n_elements, BLOCK_SIZE, ELEMENTS_PER_THREAD
    )
    return out
