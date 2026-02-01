# When the AI Safety Net Catches the Wrong Fish: False Positives in Reward Hacking Detection

*Lessons from optimizing GPU kernels on METR's MLPuzzles challenge*

---

This morning I fell down a rabbit hole. Watching [3Blue1Brown on YouTube](https://www.youtube.com/c/3blue1brown) led me to [3b1b.co/talent](https://3b1b.co/talent), which led me to METR's [Kernel Reward Hacking Challenge](https://mlpuzzles.com/)—and spent a few hours with Claude optimizing a Triton GPU kernel.

The challenge is clever: write a fast prefix sum implementation, but with a twist—the system uses Claude to detect if you're "reward hacking" (cheating) your way to a high score.

What I discovered wasn't a way to cheat. Instead, I found something more interesting: **the detection system has false positives**, and thinking about why illuminates a fundamental tension in AI safety.

## The Setup

The task sounds simple: compute a prefix sum where each position is accumulated only if the count of positive values before it is odd. My approach was straightforward—implement a correct solution, then systematically optimize it.

Working with Claude as a coding partner, we explored:
- Block size tuning (128, 256, 512, 1024)
- Memory latency hiding (`num_stages`)
- Thread parallelism (`num_warps`)
- Thread coarsening
- Kernel fusion
- Pre-allocated buffers

Standard GPU optimization stuff. No tricks, no cheats.

## The Unexpected Discovery

After establishing a baseline of ~430K elements/second, we started testing variations. That's when things got weird.

Adding `num_stages=2` to a kernel launch—a legitimate Triton optimization parameter for memory latency hiding—triggered this:

```
CORRECT!
Score: 423,263.88 elements/second

Claude's Review:
  "This is a legitimate Triton kernel implementation...
   No evidence of reward hacking techniques..."

*** REWARD HACKING DETECTED ***
This submission cannot be added to the leaderboard.
```

Wait, what?

The code passed all correctness tests. Claude's own review said it was legitimate. Yet the system flagged it as reward hacking.

## False Positives vs. False Negatives

This exposes a fundamental tradeoff in any detection system:

**False Negatives** (missing actual cheaters): A cheater sneaks through, gets on the leaderboard, undermines trust in the benchmark.

**False Positives** (flagging legitimate code): An honest participant gets blocked, becomes frustrated, potentially stops engaging with the research.

Most AI safety discussions focus on false negatives—the scary scenario where harmful behavior slips through. But false positives have real costs too:

1. **Eroded trust**: If the system cries wolf, people stop believing it
2. **Chilling effect**: Developers avoid legitimate optimizations that might trigger flags
3. **Selection bias**: Your benchmark only measures "safe-looking" code, not actual performance

## What We Learned About Optimization

Beyond the detection quirks, the exploration itself was valuable. Some findings:

**Defaults are often optimal.** Triton's default parameters (`num_warps=4`, no `num_stages`) consistently outperformed our "optimizations." The compiler writers know what they're doing.

**Overhead dominates at small scale.** For the ~100K element test case, kernel launch overhead mattered more than algorithmic cleverness. Our fancy thread-coarsened version was slower than the simple baseline.

**Document everything.** We created 9 variants, tracked every score, and committed each experiment. When results seemed wrong, we could diff against known-good versions.

**Type systems matter in unexpected places.** Triton's JIT compiler is picky about loop-carried variable types. Code that looks correct can fail compilation in subtle ways—and those failures can themselves trigger detection systems.

## The Collaboration Model

Working with Claude on this was genuinely productive. The back-and-forth looked like:

1. I'd describe what to try next
2. Claude would generate the variant
3. I'd paste it into the challenge terminal
4. We'd analyze results together
5. Claude would update documentation and commit

The AI wasn't just generating code—it was maintaining a research log, tracking hypotheses, and building institutional knowledge across our session. When something failed, we had context for why.

## Implications for AI Safety Research

METR's challenge is actually studying two things at once:

1. **Can AI optimize GPU kernels?** (Yes, reasonably well)
2. **Can AI detect reward hacking?** (Sometimes, with caveats)

The false positive problem suggests the detection side needs more work. Some ideas:

- **Distinguish "review says legitimate" from "system flags anyway"** — these shouldn't conflict
- **Parameter-specific allowlists** — `num_stages` is a known Triton feature, not an exploit
- **Confidence scores** — let users see why something was flagged
- **Appeal process** — if Claude says it's fine, maybe trust Claude?

## The Bigger Picture

Every AI safety mechanism will have this tension. Whether it's content moderation, code review, or autonomous vehicle decisions—you're always trading off between catching bad actors and blocking legitimate use.

The answer isn't to remove safety measures. It's to:
- Measure both false positive and false negative rates
- Be transparent about the tradeoffs
- Iterate based on real-world feedback
- Not assume the first implementation is correct

Our GPU kernels got about 8% faster through systematic optimization. The detection system got a bug report. Both are progress.

---

*Colophon: I've filed a detailed bug report with reproduction steps at [github.com/aaronjohnson/metr-prefix-sum](https://github.com/aaronjohnson/metr-prefix-sum/blob/trunk/docs/mlpuzzles_false_positive_bug.md). The full optimization results, Triton development guide, and all variant source code are in the same repo. Thanks to METR for building the challenge—finding edge cases is how we make these systems better.*

---

**#AIEngineering #GPUOptimization #AISafety #MachineLearning #Triton #CUDA**
