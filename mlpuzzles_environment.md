# MLPuzzles.com Environment

## System Info
- **CUDA**: 12.4.1
- **Container**: NVIDIA Deep Learning Container
- **GPU**: NVIDIA A10G (24GB GDDR6, 600 GB/s, 31.2 TFLOPS FP32)

## Workflow
```bash
# Edit your solution
nano solution.py

# Test locally (runs correctness checks)
python solution.py

# Submit for scoring (correctness + timing + AI review)
submit solution.py
```

## File Structure
The challenge expects a file named `solution.py` with a function:
```python
def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    ...
```

## Access Methods (as of 2026-02)
- **Web terminal**: https://mlpuzzles.com/ (paste code directly)
- **SSH**: `ssh -p 2222 sandbox@puzzle.metr-dev.org` (password: puzzle) - *may timeout*
- **API**: `curl -X POST https://puzzle.metr-dev.org/api/submit` - *may timeout*

## Notes
- First Triton kernel run triggers JIT compilation (slower)
- Subsequent runs are cached
- Scoring based on: correctness, execution time (ms), AI review (Claude Opus)
