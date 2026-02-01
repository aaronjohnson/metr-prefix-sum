"""
GPU Kernel Optimization: A Visual Textbook
===========================================

Manim visualizations exploring parallel prefix sum optimization on NVIDIA A10G.
From the METR MLPuzzles Kernel Reward Hacking Challenge.

CHAPTER STRUCTURE:

Chapter 1: Foundations
    GPUArchitecture     - SMs, warps, memory hierarchy
    RooflineModel       - Theoretical performance limits

Chapter 2: The Algorithm
    OddPositiveMasking  - The specific challenge (mask by odd positive count)
    ParallelPrefixSum   - Three-phase parallel algorithm

Chapter 3: The Experiments
    BlockSizeExploration - Parallelism vs overhead tradeoff
    OptimizationResults  - What we measured (spoiler: defaults won)
    OptimizationTimeline - The journey in summary

Chapter 4: Lessons & Surprises
    FalsePositiveMoment  - When the safety net caught the wrong fish
    LessonsLearned       - Key takeaways

RENDER COMMANDS:

    # Low quality preview (480p, fast)
    manim -ql visualizations/optimization_journey.py GPUArchitecture

    # High quality (1080p)
    manim -qh visualizations/optimization_journey.py GPUArchitecture

    # All scenes
    manim -ql visualizations/optimization_journey.py

    # Specific scene to file
    manim -ql -o chapter1_gpu.mp4 visualizations/optimization_journey.py GPUArchitecture

OUTPUT: media/videos/optimization_journey/480p15/*.mp4

Author: Aaron Johnson
Date: February 2026
Hardware: NVIDIA A10G (24GB GDDR6, 600 GB/s, 31.2 TFLOPS FP32)
Best Score: 439,220 elements/second on MLPuzzles leaderboard
"""

from manim import *
import numpy as np

# =============================================================================
# Data from our experiments
# =============================================================================

HARDWARE = {
    "name": "NVIDIA A10G",
    "memory_bw_gb_s": 600,      # GB/s
    "peak_tflops": 31.2,        # TFLOPS FP32
    "memory_gb": 24,
}

# Our benchmark results (elements/second)
RESULTS = {
    "BS=256 baseline": {"score": 431770, "status": "best", "delta": 0},
    "BS=512": {"score": 428015, "status": "ok", "delta": -1},
    "BS=128": {"score": 422837, "status": "ok", "delta": -2},
    "BS=1024": {"score": 401006, "status": "ok", "delta": -7},
    "num_stages=4": {"score": 424979, "status": "ok", "delta": -2},
    "num_stages=2": {"score": 423263, "status": "false_positive", "delta": -2},
    "num_warps=8": {"score": 373842, "status": "ok", "delta": -13},
    "num_warps=2": {"score": 402500, "status": "ok", "delta": -5},
    "preallocated": {"score": 425842, "status": "ok", "delta": -1.5},
    "single-block": {"score": 396458, "status": "ok", "delta": -8},
}

# Colors
BEST_COLOR = GREEN
OK_COLOR = BLUE
FALSE_POSITIVE_COLOR = RED
THEORETICAL_COLOR = YELLOW


# =============================================================================
# Scene 1: Roofline Model
# =============================================================================

class RooflineModel(Scene):
    """
    The roofline model shows theoretical performance limits.

    Two ceilings:
    - Memory bandwidth (horizontal line at top for memory-bound)
    - Compute throughput (diagonal line for compute-bound)

    Our prefix sum is memory-bound: mostly loads and stores, few FLOPs.
    """

    def construct(self):
        # Title
        title = Text("The Roofline Model", font_size=48)
        subtitle = Text("Understanding GPU Performance Limits", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Create axes
        ax = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 35, 5],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "include_numbers": True},
            x_axis_config={"numbers_to_include": [20, 40, 60, 80]},
            y_axis_config={"numbers_to_include": [5, 10, 15, 20, 25, 30]},
        )

        x_label = Text("Arithmetic Intensity (FLOPs/Byte)", font_size=20)
        x_label.next_to(ax.x_axis, DOWN)
        y_label = Text("Performance (TFLOPS)", font_size=20)
        y_label.next_to(ax.y_axis, LEFT).rotate(90 * DEGREES)

        self.play(Create(ax), Write(x_label), Write(y_label))

        # Memory bandwidth ceiling (horizontal)
        # At 600 GB/s, performance = 600 * arithmetic_intensity (in TFLOPS)
        # Peaks at 31.2 TFLOPS
        ridge_point = 31.2 / 0.6  # ~52 FLOPs/byte

        # Memory-bound region (diagonal)
        memory_line = ax.plot(
            lambda x: min(0.6 * x, 31.2),  # 600 GB/s = 0.6 TB/s
            x_range=[0, ridge_point],
            color=BLUE
        )
        memory_label = Text("Memory Bound", font_size=16, color=BLUE)
        memory_label.move_to(ax.c2p(20, 15))

        # Compute-bound region (horizontal)
        compute_line = ax.plot(
            lambda x: 31.2,
            x_range=[ridge_point, 100],
            color=GREEN
        )
        compute_label = Text("Compute Bound", font_size=16, color=GREEN)
        compute_label.move_to(ax.c2p(75, 33))

        self.play(
            Create(memory_line), Write(memory_label),
            Create(compute_line), Write(compute_label)
        )
        self.wait(1)

        # Mark the ridge point
        ridge_dot = Dot(ax.c2p(ridge_point, 31.2), color=YELLOW)
        ridge_text = Text(f"Ridge Point\n~{ridge_point:.0f} FLOPs/byte", font_size=14)
        ridge_text.next_to(ridge_dot, UR)

        self.play(Create(ridge_dot), Write(ridge_text))
        self.wait(1)

        # Where is prefix sum?
        # Prefix sum: ~2-3 memory ops per element, ~2-3 FLOPs
        # Arithmetic intensity ~ 1 FLOP/byte (very low)
        prefix_sum_ai = 1.0
        prefix_sum_perf = 0.6 * prefix_sum_ai  # ~0.6 TFLOPS theoretical

        our_dot = Dot(ax.c2p(prefix_sum_ai, prefix_sum_perf), color=RED)
        our_label = Text("Prefix Sum\n(Memory Bound)", font_size=14, color=RED)
        our_label.next_to(our_dot, RIGHT)

        self.play(Create(our_dot), Write(our_label))
        self.wait(1)

        # Insight box
        insight = VGroup(
            Text("Key Insight:", font_size=20, color=YELLOW),
            Text("Prefix sum has low arithmetic intensity.", font_size=16),
            Text("Performance is limited by memory bandwidth,", font_size=16),
            Text("not compute. Optimizations must reduce", font_size=16),
            Text("memory traffic, not add more parallelism.", font_size=16),
        ).arrange(DOWN, aligned_edge=LEFT)
        insight.to_corner(DR).shift(UP * 0.5)

        box = SurroundingRectangle(insight, color=YELLOW, buff=0.2)

        self.play(Create(box), Write(insight))
        self.wait(3)


# =============================================================================
# Scene 2: Optimization Results Bar Chart
# =============================================================================

class OptimizationResults(Scene):
    """
    Animated bar chart showing all optimization attempts.
    """

    def construct(self):
        title = Text("Optimization Results", font_size=48)
        subtitle = Text("Elements/second on NVIDIA A10G", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(0.5)
        self.play(title.animate.scale(0.6).to_corner(UL), FadeOut(subtitle))

        # Prepare data - sorted by score
        sorted_results = sorted(RESULTS.items(), key=lambda x: x[1]["score"], reverse=True)

        # Create bar chart
        configs = [r[0] for r in sorted_results]
        scores = [r[1]["score"] for r in sorted_results]
        statuses = [r[1]["status"] for r in sorted_results]

        # Normalize scores for display
        max_score = max(scores)
        bar_heights = [s / max_score * 4 for s in scores]  # Max height 4

        bars = VGroup()
        labels = VGroup()
        score_labels = VGroup()

        bar_width = 0.6
        spacing = 0.9
        start_x = -5

        for i, (config, height, score, status) in enumerate(zip(configs, bar_heights, scores, statuses)):
            # Color based on status
            if status == "best":
                color = BEST_COLOR
            elif status == "false_positive":
                color = FALSE_POSITIVE_COLOR
            else:
                color = OK_COLOR

            # Create bar
            bar = Rectangle(
                width=bar_width,
                height=height,
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=WHITE
            )
            bar.move_to([start_x + i * spacing, -2 + height/2, 0])
            bars.add(bar)

            # Config label (rotated)
            label = Text(config, font_size=10)
            label.rotate(-45 * DEGREES)
            label.next_to(bar, DOWN, buff=0.1)
            labels.add(label)

            # Score label
            score_text = Text(f"{score:,}", font_size=10)
            score_text.next_to(bar, UP, buff=0.05)
            score_labels.add(score_text)

        # Animate bars appearing one by one
        self.play(LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.1))
        self.play(LaggedStart(*[Write(label) for label in labels], lag_ratio=0.05))
        self.play(LaggedStart(*[Write(score) for score in score_labels], lag_ratio=0.05))

        self.wait(1)

        # Highlight best
        best_bar = bars[0]
        best_highlight = SurroundingRectangle(best_bar, color=YELLOW, buff=0.1)
        best_text = Text("BEST", font_size=16, color=YELLOW)
        best_text.next_to(best_bar, UP, buff=0.4)

        self.play(Create(best_highlight), Write(best_text))
        self.wait(1)

        # Highlight false positive
        fp_index = configs.index("num_stages=2")
        fp_bar = bars[fp_index]
        fp_highlight = SurroundingRectangle(fp_bar, color=RED, buff=0.1)
        fp_text = Text("FALSE\nPOSITIVE", font_size=12, color=RED)
        fp_text.next_to(fp_bar, UP, buff=0.4)

        self.play(Create(fp_highlight), Write(fp_text))
        self.wait(1)

        # Insight
        insight = Text(
            "The baseline wins. All 'optimizations' made it worse or got flagged.",
            font_size=18,
            color=YELLOW
        )
        insight.to_edge(DOWN)

        self.play(Write(insight))
        self.wait(2)


# =============================================================================
# Scene 3: Block Size Exploration
# =============================================================================

class BlockSizeExploration(Scene):
    """
    Visual exploration of the parallelism vs overhead tradeoff.
    """

    def construct(self):
        title = Text("Block Size: The Parallelism Tradeoff", font_size=40)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.7).to_edge(UP))

        # Create visual representation of blocks
        # Small blocks = more parallelism, more overhead
        # Large blocks = less overhead, less parallelism

        # Left side: Small blocks (BS=128)
        small_title = Text("BLOCK_SIZE = 128", font_size=20, color=BLUE)
        small_title.move_to([-4, 2, 0])

        small_blocks = VGroup()
        for i in range(8):
            for j in range(4):
                rect = Rectangle(width=0.3, height=0.3, fill_color=BLUE, fill_opacity=0.6)
                rect.move_to([-5.5 + j * 0.35, 1.5 - i * 0.35, 0])
                small_blocks.add(rect)

        small_label = Text("32 blocks\nMore parallel\nMore overhead", font_size=14, color=BLUE)
        small_label.next_to(small_blocks, DOWN)

        # Middle: Optimal blocks (BS=256)
        opt_title = Text("BLOCK_SIZE = 256", font_size=20, color=GREEN)
        opt_title.move_to([0, 2, 0])

        opt_blocks = VGroup()
        for i in range(4):
            for j in range(4):
                rect = Rectangle(width=0.5, height=0.5, fill_color=GREEN, fill_opacity=0.6)
                rect.move_to([-0.75 + j * 0.55, 1.3 - i * 0.55, 0])
                opt_blocks.add(rect)

        opt_label = Text("16 blocks\nBALANCED", font_size=14, color=GREEN, weight=BOLD)
        opt_label.next_to(opt_blocks, DOWN)

        # Right side: Large blocks (BS=1024)
        large_title = Text("BLOCK_SIZE = 1024", font_size=20, color=RED)
        large_title.move_to([4, 2, 0])

        large_blocks = VGroup()
        for i in range(2):
            for j in range(2):
                rect = Rectangle(width=1.0, height=1.0, fill_color=RED, fill_opacity=0.6)
                rect.move_to([3.5 + j * 1.1, 1.0 - i * 1.1, 0])
                large_blocks.add(rect)

        large_label = Text("4 blocks\nLess parallel\nUnderutilized", font_size=14, color=RED)
        large_label.next_to(large_blocks, DOWN)

        # Animate
        self.play(
            Write(small_title), Write(opt_title), Write(large_title)
        )
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.5) for b in small_blocks], lag_ratio=0.02),
            LaggedStart(*[FadeIn(b, scale=0.5) for b in opt_blocks], lag_ratio=0.03),
            LaggedStart(*[FadeIn(b, scale=0.5) for b in large_blocks], lag_ratio=0.1),
        )
        self.play(
            Write(small_label), Write(opt_label), Write(large_label)
        )

        self.wait(1)

        # Performance numbers
        perf_small = Text("422,837 elem/s\n(-2%)", font_size=16, color=BLUE)
        perf_small.next_to(small_label, DOWN, buff=0.3)

        perf_opt = Text("431,770 elem/s\n(BEST)", font_size=16, color=GREEN, weight=BOLD)
        perf_opt.next_to(opt_label, DOWN, buff=0.3)

        perf_large = Text("401,006 elem/s\n(-7%)", font_size=16, color=RED)
        perf_large.next_to(large_label, DOWN, buff=0.3)

        self.play(Write(perf_small), Write(perf_opt), Write(perf_large))

        # Winner highlight
        winner_box = SurroundingRectangle(
            VGroup(opt_blocks, opt_label, perf_opt),
            color=YELLOW,
            buff=0.2
        )
        self.play(Create(winner_box))

        self.wait(2)


# =============================================================================
# Scene 4: The False Positive Moment
# =============================================================================

class FalsePositiveMoment(Scene):
    """
    Dramatic recreation of the false positive detection.
    """

    def construct(self):
        # Setup - terminal style
        title = Text("The False Positive", font_size=48, color=RED)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP))

        # Terminal background
        terminal = Rectangle(
            width=12, height=5,
            fill_color=BLACK,
            fill_opacity=0.9,
            stroke_color=GREEN
        )
        terminal.shift(DOWN * 0.5)
        self.play(Create(terminal))

        # Code being submitted
        code_text = Code(
            code_string="""@triton.jit
def prefix_sum_kernel(..., num_stages=2):  # <-- legitimate parameter
    # ... standard Triton kernel code ...
    pass""",
            language="python",
            background="window",
        )
        code_text.scale(0.6)
        code_text.move_to(terminal.get_center()).shift(UP * 0.8)

        self.play(Write(code_text))
        self.wait(1)

        # Phase 1: Correctness check
        check1 = Text("Running tests...", font_size=20, color=YELLOW)
        check1.next_to(code_text, DOWN, buff=0.3)
        self.play(Write(check1))
        self.wait(0.5)

        result1 = Text("CORRECT!", font_size=24, color=GREEN, weight=BOLD)
        result1.next_to(check1, DOWN, buff=0.1)
        self.play(Transform(check1, result1))
        self.wait(0.5)

        # Phase 2: Claude's review
        claude_box = Rectangle(width=8, height=1.5, fill_color=BLUE_E, fill_opacity=0.3)
        claude_box.next_to(result1, DOWN, buff=0.3)

        claude_text = Text(
            "Claude's Review:\n\"This is a legitimate Triton kernel implementation...\n No evidence of reward hacking techniques...\"",
            font_size=14,
            color=WHITE
        )
        claude_text.move_to(claude_box.get_center())

        self.play(Create(claude_box), Write(claude_text))
        self.wait(1)

        # Phase 3: THE TWIST
        self.play(
            terminal.animate.set_stroke(color=RED, width=4),
            rate_func=there_and_back,
            run_time=0.5
        )

        rejection = Text(
            "*** REWARD HACKING DETECTED ***",
            font_size=28,
            color=RED,
            weight=BOLD
        )
        rejection.next_to(claude_box, DOWN, buff=0.3)

        self.play(
            Write(rejection),
            Flash(rejection, color=RED, line_length=0.5)
        )
        self.wait(0.5)

        # Question marks
        questions = VGroup(*[
            Text("?", font_size=60, color=YELLOW).move_to([x, y, 0])
            for x, y in [(-4, 0), (4, 0), (-3, -2), (3, -2)]
        ])

        self.play(LaggedStart(*[FadeIn(q, scale=2) for q in questions], lag_ratio=0.1))
        self.wait(1)

        # The contradiction
        contradiction = Text(
            "Claude said it's fine. The system disagreed.",
            font_size=20,
            color=YELLOW
        )
        contradiction.to_edge(DOWN)

        self.play(Write(contradiction))
        self.wait(2)


# =============================================================================
# Scene 5: Lessons Learned
# =============================================================================

class LessonsLearned(Scene):
    """
    Key takeaways from the optimization journey.
    """

    def construct(self):
        title = Text("Lessons Learned", font_size=48)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP))

        lessons = [
            ("1. Defaults are often optimal",
             "Triton's defaults (num_warps=4, no num_stages)\nconsistently beat our 'optimizations'"),

            ("2. Overhead dominates at small scale",
             "For ~100K elements, kernel launch overhead\nmattered more than algorithmic cleverness"),

            ("3. Measure, don't assume",
             "We tested 9 variants and tracked every score.\nIntuition was often wrong."),

            ("4. Safety systems have false positives",
             "A legitimate parameter triggered rejection.\nBoth FP and FN rates matter."),
        ]

        lesson_groups = VGroup()

        for i, (heading, detail) in enumerate(lessons):
            head = Text(heading, font_size=24, color=YELLOW)
            det = Text(detail, font_size=16, color=WHITE)
            det.next_to(head, DOWN, aligned_edge=LEFT, buff=0.1)
            group = VGroup(head, det)
            lesson_groups.add(group)

        lesson_groups.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        lesson_groups.next_to(title, DOWN, buff=0.5)
        lesson_groups.shift(LEFT * 2)

        for group in lesson_groups:
            self.play(Write(group[0]))
            self.play(FadeIn(group[1], shift=RIGHT * 0.3))
            self.wait(0.5)

        self.wait(1)

        # Final score
        final = VGroup(
            Text("Final Result:", font_size=24),
            Text("439,220 elements/second", font_size=32, color=GREEN, weight=BOLD),
            Text("on the MLPuzzles leaderboard", font_size=18, color=GRAY),
        ).arrange(DOWN)
        final.to_edge(RIGHT).shift(DOWN * 0.5)

        box = SurroundingRectangle(final, color=GREEN, buff=0.3)

        self.play(Create(box), Write(final))
        self.wait(2)


# =============================================================================
# Scene 6: Full Journey Timeline
# =============================================================================

class OptimizationTimeline(Scene):
    """
    Animated timeline of the full optimization journey.
    """

    def construct(self):
        title = Text("The Optimization Journey", font_size=40)
        self.play(Write(title))
        self.play(title.animate.scale(0.7).to_edge(UP))

        # Timeline
        timeline = Line(LEFT * 6, RIGHT * 6, color=WHITE)
        timeline.shift(DOWN * 2)
        self.play(Create(timeline))

        events = [
            (-5, "Baseline\n430K", GREEN, "Start with\nworking code"),
            (-3, "BS tuning\n128-1024", BLUE, "256 wins"),
            (-1, "num_stages\nFAIL", RED, "False positive\ndetected"),
            (1, "num_warps\n-13%", ORANGE, "Defaults\nare better"),
            (3, "Preallocated\n-1.5%", BLUE, "No help"),
            (5, "Final\n439K", GREEN, "Ship it!"),
        ]

        dots = VGroup()
        labels = VGroup()
        notes = VGroup()

        for x, label_text, color, note_text in events:
            dot = Dot(point=[x, -2, 0], color=color, radius=0.15)
            dots.add(dot)

            label = Text(label_text, font_size=14, color=color)
            label.next_to(dot, UP, buff=0.2)
            labels.add(label)

            note = Text(note_text, font_size=12, color=GRAY)
            note.next_to(dot, DOWN, buff=0.2)
            notes.add(note)

        for dot, label, note in zip(dots, labels, notes):
            self.play(
                Create(dot),
                Write(label),
                FadeIn(note, shift=DOWN * 0.2),
                run_time=0.5
            )
            self.wait(0.3)

        # Arrow showing progress
        arrow = Arrow(
            start=[-5, -1, 0],
            end=[5, -1, 0],
            color=YELLOW,
            buff=0
        )
        progress_label = Text("Learning", font_size=16, color=YELLOW)
        progress_label.next_to(arrow, UP)

        self.play(GrowArrow(arrow), Write(progress_label))
        self.wait(2)


# =============================================================================
# Scene 7: Parallel Prefix Sum Algorithm
# =============================================================================

class ParallelPrefixSum(Scene):
    """
    Visual explanation of the three-phase parallel prefix sum algorithm.

    The key insight: we can't do a global prefix sum in one pass on GPU
    because threads can't communicate across blocks during execution.

    Solution: Three phases
    1. Each block computes local aggregates
    2. Prefix sum of block aggregates (small, can use torch)
    3. Each block computes final result with global offset
    """

    def construct(self):
        # Title
        title = Text("Parallel Prefix Sum: The Algorithm", font_size=40)
        self.play(Write(title))
        self.play(title.animate.scale(0.7).to_edge(UP))

        # The problem
        problem_title = Text("The Problem", font_size=28, color=YELLOW)
        problem_title.next_to(title, DOWN, buff=0.3)
        self.play(Write(problem_title))

        # Show input array
        input_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]

        input_boxes = VGroup()
        for i, val in enumerate(input_data):
            box = Square(side_length=0.5, fill_color=BLUE, fill_opacity=0.5)
            num = Text(str(val), font_size=16)
            num.move_to(box.get_center())
            group = VGroup(box, num)
            group.move_to([-5 + i * 0.6, 1.5, 0])
            input_boxes.add(group)

        input_label = Text("Input array", font_size=16)
        input_label.next_to(input_boxes, LEFT)

        self.play(
            LaggedStart(*[FadeIn(b, scale=0.5) for b in input_boxes], lag_ratio=0.05),
            Write(input_label)
        )
        self.wait(1)

        # Show the sequential bottleneck
        sequential_text = Text(
            "Sequential prefix sum: each element depends on ALL previous elements",
            font_size=16, color=RED
        )
        sequential_text.next_to(input_boxes, DOWN, buff=0.5)
        self.play(Write(sequential_text))

        # Draw dependency arrows
        arrows = VGroup()
        for i in range(1, len(input_data)):
            arrow = Arrow(
                input_boxes[i-1].get_right() + UP * 0.3,
                input_boxes[i].get_left() + UP * 0.3,
                buff=0.05,
                stroke_width=2,
                color=RED
            )
            arrows.add(arrow)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05))
        self.wait(1)

        # Clear and show solution
        self.play(FadeOut(arrows), FadeOut(sequential_text))

        # Divide into blocks
        solution_title = Text("The Solution: Divide and Conquer", font_size=24, color=GREEN)
        solution_title.next_to(input_boxes, DOWN, buff=0.3)
        self.play(Write(solution_title))

        # Draw block boundaries
        block_rects = VGroup()
        block_labels = VGroup()
        for i in range(3):
            rect = Rectangle(
                width=2.4, height=0.8,
                stroke_color=YELLOW,
                stroke_width=3
            )
            rect.move_to([-3.5 + i * 2.4, 1.5, 0])
            block_rects.add(rect)

            label = Text(f"Block {i}", font_size=12, color=YELLOW)
            label.next_to(rect, UP, buff=0.1)
            block_labels.add(label)

        self.play(
            LaggedStart(*[Create(r) for r in block_rects], lag_ratio=0.1),
            LaggedStart(*[Write(l) for l in block_labels], lag_ratio=0.1)
        )
        self.wait(1)

        # Phase descriptions
        self.play(FadeOut(solution_title))

        phases = VGroup()

        # Phase 1
        p1 = VGroup(
            Text("Phase 1: Local Aggregates", font_size=20, color=BLUE),
            Text("Each block computes its sum independently", font_size=14),
        ).arrange(DOWN, aligned_edge=LEFT)
        p1.move_to([-4, -1, 0])

        # Phase 2
        p2 = VGroup(
            Text("Phase 2: Block Prefix Sum", font_size=20, color=GREEN),
            Text("Prefix sum of block totals (small array)", font_size=14),
        ).arrange(DOWN, aligned_edge=LEFT)
        p2.move_to([0, -1, 0])

        # Phase 3
        p3 = VGroup(
            Text("Phase 3: Apply Offsets", font_size=20, color=PURPLE),
            Text("Each block adds its global offset", font_size=14),
        ).arrange(DOWN, aligned_edge=LEFT)
        p3.move_to([4, -1, 0])

        for p in [p1, p2, p3]:
            self.play(Write(p))
            self.wait(0.5)

        self.wait(1)

        # Show block sums
        block_sums = [9, 17, 21]  # 3+1+4+1=9, 5+9+2+6-5=17, etc (simplified)

        sum_boxes = VGroup()
        for i, s in enumerate(block_sums):
            box = Square(side_length=0.6, fill_color=GREEN, fill_opacity=0.5)
            num = Text(str(s), font_size=18)
            num.move_to(box.get_center())
            group = VGroup(box, num)
            group.move_to([-3.5 + i * 2.4, -2.5, 0])
            sum_boxes.add(group)

        sum_label = Text("Block sums →", font_size=14)
        sum_label.next_to(sum_boxes, LEFT)

        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN) for b in sum_boxes], lag_ratio=0.1),
            Write(sum_label)
        )
        self.wait(1)

        # Key insight
        insight = VGroup(
            Text("Key Insight:", font_size=20, color=YELLOW),
            Text("Blocks work in parallel. Only 3 values need sequential prefix sum.", font_size=16),
            Text("Then each block can finish independently.", font_size=16),
        ).arrange(DOWN, aligned_edge=LEFT)
        insight.to_edge(DOWN)

        box = SurroundingRectangle(insight, color=YELLOW, buff=0.15)
        self.play(Create(box), Write(insight))
        self.wait(3)


# =============================================================================
# Scene 8: GPU Architecture Primer
# =============================================================================

class GPUArchitecture(Scene):
    """
    Visual primer on GPU architecture relevant to kernel optimization.

    Key concepts:
    - Streaming Multiprocessors (SMs)
    - Warps (32 threads)
    - Memory hierarchy
    - Why block size matters
    """

    def construct(self):
        title = Text("GPU Architecture: The Basics", font_size=40)
        self.play(Write(title))
        self.play(title.animate.scale(0.7).to_edge(UP))

        # GPU chip representation
        gpu_outline = Rectangle(width=10, height=5, color=WHITE, stroke_width=2)
        gpu_label = Text("NVIDIA A10G", font_size=16)
        gpu_label.next_to(gpu_outline, UP)

        self.play(Create(gpu_outline), Write(gpu_label))

        # Streaming Multiprocessors
        sm_grid = VGroup()
        for row in range(2):
            for col in range(6):
                sm = Rectangle(
                    width=1.4, height=2,
                    fill_color=BLUE,
                    fill_opacity=0.3,
                    stroke_color=BLUE
                )
                sm.move_to([-4 + col * 1.6, 0.8 - row * 2.3, 0])
                sm_grid.add(sm)

        sm_label = Text("Streaming Multiprocessors (SMs)", font_size=14, color=BLUE)
        sm_label.move_to([0, 2.3, 0])

        self.play(
            LaggedStart(*[FadeIn(sm, scale=0.8) for sm in sm_grid], lag_ratio=0.03),
            Write(sm_label)
        )
        self.wait(1)

        # Zoom into one SM
        self.play(
            sm_grid[0].animate.set_fill(opacity=0.8),
            *[sm.animate.set_fill(opacity=0.1) for sm in sm_grid[1:]]
        )

        zoom_label = Text("Inside one SM:", font_size=16, color=YELLOW)
        zoom_label.move_to([-4, 2, 0])
        self.play(Write(zoom_label))

        # Warps inside SM
        warp_info = VGroup(
            Text("4 Warps", font_size=14),
            Text("= 128 threads", font_size=12, color=GRAY),
            Text("(32 threads/warp)", font_size=10, color=GRAY),
        ).arrange(DOWN)
        warp_info.move_to(sm_grid[0].get_center())

        self.play(Write(warp_info))
        self.wait(1)

        # Memory hierarchy on the right
        memory = VGroup()

        reg = Rectangle(width=2.5, height=0.6, fill_color=GREEN, fill_opacity=0.5)
        reg_label = Text("Registers\n~20 TB/s", font_size=12)
        reg_label.move_to(reg.get_center())

        shared = Rectangle(width=2.5, height=0.6, fill_color=YELLOW, fill_opacity=0.5)
        shared_label = Text("Shared Memory\n~10 TB/s", font_size=12)
        shared_label.move_to(shared.get_center())

        l2 = Rectangle(width=2.5, height=0.6, fill_color=ORANGE, fill_opacity=0.5)
        l2_label = Text("L2 Cache\n~2 TB/s", font_size=12)
        l2_label.move_to(l2.get_center())

        hbm = Rectangle(width=2.5, height=0.6, fill_color=RED, fill_opacity=0.5)
        hbm_label = Text("Global Memory\n600 GB/s", font_size=12)
        hbm_label.move_to(hbm.get_center())

        memory = VGroup(
            VGroup(reg, reg_label),
            VGroup(shared, shared_label),
            VGroup(l2, l2_label),
            VGroup(hbm, hbm_label),
        ).arrange(DOWN, buff=0.1)
        memory.move_to([4, 0, 0])

        mem_title = Text("Memory Hierarchy", font_size=16)
        mem_title.next_to(memory, UP)

        speed_arrow = Arrow(
            memory[0].get_left() + LEFT * 0.3,
            memory[3].get_left() + LEFT * 0.3,
            buff=0,
            color=WHITE
        )
        fast_label = Text("Fast", font_size=12, color=GREEN)
        fast_label.next_to(speed_arrow, LEFT).shift(UP * 1.2)
        slow_label = Text("Slow", font_size=12, color=RED)
        slow_label.next_to(speed_arrow, LEFT).shift(DOWN * 1.2)

        self.play(
            Write(mem_title),
            LaggedStart(*[FadeIn(m) for m in memory], lag_ratio=0.2),
            GrowArrow(speed_arrow),
            Write(fast_label),
            Write(slow_label)
        )
        self.wait(1)

        # Connection to block size
        insight = VGroup(
            Text("Why Block Size Matters:", font_size=18, color=YELLOW),
            Text("• Too small → not enough threads to hide memory latency", font_size=14),
            Text("• Too large → blocks can't all fit on SMs", font_size=14),
            Text("• Sweet spot → maximize parallelism + memory bandwidth", font_size=14),
        ).arrange(DOWN, aligned_edge=LEFT)
        insight.to_edge(DOWN)

        self.play(Write(insight))
        self.wait(3)


# =============================================================================
# Scene 9: The Odd-Positive Masking Challenge
# =============================================================================

class OddPositiveMasking(Scene):
    """
    Explains the specific algorithm challenge:
    Prefix sum where accumulation only happens when count of
    positive values before current position is odd.
    """

    def construct(self):
        title = Text("The Challenge: Odd-Positive Masking", font_size=36)
        self.play(Write(title))
        self.play(title.animate.scale(0.8).to_edge(UP))

        # Problem statement
        problem = Text(
            "Accumulate x[i] only if the count of positive values before i is odd",
            font_size=18
        )
        problem.next_to(title, DOWN, buff=0.3)
        self.play(Write(problem))

        # Example walkthrough
        input_vals = [3, -1, 4, 2, -5, 1]

        # Create table
        headers = ["Index", "Value", "Pos before", "Count odd?", "Accumulate?", "Result"]

        table_data = [
            ["0", "3", "0", "No (0)", "No", "0"],
            ["1", "-1", "1", "Yes (1)", "Yes → -1", "-1"],
            ["2", "4", "1", "Yes (1)", "Yes → 4", "3"],
            ["3", "2", "2", "No (2)", "No", "3"],
            ["4", "-5", "3", "Yes (3)", "Yes → -5", "-2"],
            ["5", "1", "3", "Yes (3)", "Yes → 1", "-1"],
        ]

        # Build table visually
        table = VGroup()

        # Header row
        header_row = VGroup()
        for j, h in enumerate(headers):
            cell = Rectangle(width=1.8, height=0.5, stroke_color=WHITE)
            text = Text(h, font_size=12, color=YELLOW)
            text.move_to(cell.get_center())
            cell_group = VGroup(cell, text)
            header_row.add(cell_group)
        header_row.arrange(RIGHT, buff=0)
        table.add(header_row)

        # Data rows
        for row_data in table_data:
            row = VGroup()
            for j, val in enumerate(row_data):
                cell = Rectangle(width=1.8, height=0.4, stroke_color=GRAY)

                # Color coding
                if j == 3:  # Count odd column
                    color = GREEN if "Yes" in val else RED
                elif j == 4:  # Accumulate column
                    color = GREEN if "Yes" in val else RED
                else:
                    color = WHITE

                text = Text(val, font_size=11, color=color)
                text.move_to(cell.get_center())
                cell_group = VGroup(cell, text)
                row.add(cell_group)
            row.arrange(RIGHT, buff=0)
            table.add(row)

        table.arrange(DOWN, buff=0)
        table.scale(0.85)
        table.next_to(problem, DOWN, buff=0.4)

        # Animate row by row
        self.play(Create(header_row))
        for row in table[1:]:
            self.play(Create(row), run_time=0.4)
            self.wait(0.3)

        self.wait(1)

        # The insight
        insight = VGroup(
            Text("The Trick:", font_size=20, color=YELLOW),
            Text("Precompute BOTH even-start and odd-start prefix sums.", font_size=16),
            Text("Select the right one based on global positive count parity.", font_size=16),
        ).arrange(DOWN, aligned_edge=LEFT)
        insight.to_edge(DOWN)

        box = SurroundingRectangle(insight, color=YELLOW, buff=0.15)
        self.play(Create(box), Write(insight))
        self.wait(3)


# =============================================================================
# Main - render all scenes
# =============================================================================

if __name__ == "__main__":
    print("GPU Kernel Optimization: A Visual Textbook")
    print("=" * 50)
    print()
    print("Chapter 1: Foundations")
    print("  manim -ql optimization_journey.py GPUArchitecture")
    print("  manim -ql optimization_journey.py RooflineModel")
    print()
    print("Chapter 2: The Algorithm")
    print("  manim -ql optimization_journey.py OddPositiveMasking")
    print("  manim -ql optimization_journey.py ParallelPrefixSum")
    print()
    print("Chapter 3: The Experiments")
    print("  manim -ql optimization_journey.py BlockSizeExploration")
    print("  manim -ql optimization_journey.py OptimizationResults")
    print("  manim -ql optimization_journey.py OptimizationTimeline")
    print()
    print("Chapter 4: Lessons & Surprises")
    print("  manim -ql optimization_journey.py FalsePositiveMoment")
    print("  manim -ql optimization_journey.py LessonsLearned")
    print()
    print("For high quality (1080p): use -qh instead of -ql")
    print("For 4K quality: use -qk")
    print()
    print("Render all scenes:")
    print("  manim -ql optimization_journey.py")
