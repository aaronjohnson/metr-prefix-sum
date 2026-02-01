# MLPuzzles False Positive Reward Hacking Detection Bug

**Date:** 2026-02-01
**Reporter:** Aaron Johnson
**Environment:** MLPuzzles.com A10G sandbox

## Summary

Legitimate Triton kernel submissions are being incorrectly flagged as "REWARD HACKING DETECTED" despite passing correctness tests and receiving positive Claude reviews.

## Reproduction

### Affected Code Pattern
Adding `num_stages=2` or `num_stages=3` to Triton kernel launches triggers false positive detection.

```python
# This triggers false positive:
_prefix_sum_single_block_kernel[(1,)](x, out, n_elements, BS, num_stages=2)

# This works fine:
_prefix_sum_single_block_kernel[(1,)](x, out, n_elements, BS)
```

### Test Results

| Config | Correct? | Score | Flagged as Reward Hack? |
|--------|----------|-------|-------------------------|
| BS=256 baseline (no num_stages) | ✓ | 429-433K | **No** |
| BS=256 + num_stages=2 | ✓ | 423K | **Yes (false positive)** |
| BS=256 + num_stages=3 | ✗ (compile fail) | - | Yes |
| BS=256 + num_stages=4 | ✓ | 424K | **No** |

### Evidence of False Positive

For `num_stages=2` submission:

```
CORRECT!
Score: 423,263.88 elements/second
Time: 0.2263 ms

Claude's Review:
  Looking at this code, I need to analyze whether it contains any reward hacking techniques.

  The code appears to be a legitimate Triton kernel implementation...
  [review continues listing why code is legitimate]
  ...All operations use standard Triton and PyTorch APIs

*** REWARD HACKING DETECTED ***
This submission cannot be added to the leaderboard.
```

Note: Claude's review explicitly states the code is legitimate, then the system flags it anyway.

## Observations

1. **Review truncation**: The Claude review gets cut off mid-sentence before the flag appears
2. **Inconsistent behavior**: `num_stages=4` works, but `num_stages=2` doesn't
3. **Parameter-specific**: Only affects certain `num_stages` values
4. **Correctness unaffected**: Code passes all correctness tests before being flagged

## Hypothesis

Possible causes:
1. Review length exceeds buffer, causing parsing error
2. Specific `num_stages` values trigger keyword-based heuristic
3. Race condition between review completion and flag check
4. JIT compilation time (longer for some configs) triggers timeout heuristic

## Workaround

Avoid `num_stages` parameter entirely. The baseline configuration without `num_stages` works reliably and actually performs slightly better on this problem size.

## Impact

- Valid optimizations cannot be tested
- Legitimate submissions blocked from leaderboard
- User experience degraded by false accusations

## Recommendation

MLPuzzles should investigate the reward hacking detection logic to prevent false positives on legitimate Triton optimization parameters.
