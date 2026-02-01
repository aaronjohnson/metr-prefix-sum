"""
Complete Algorithm Explanation - Unified Manim Animation
Combines algorithm walkthrough, parallelization challenges, and Triton execution.

Run with:
    manim -pql explain_algorithm_complete.py CompleteOverview
"""

from manim import *


class CompleteOverview(Scene):
    def construct(self):
        # ===== PART 1: INTRODUCTION =====
        self.intro_section()

        # ===== PART 2: ALGORITHM WALKTHROUGH =====
        self.algorithm_section()

        # ===== TRANSITION 1 =====
        self.transition_to_parallel()

        # ===== PART 3: PARALLELIZATION CHALLENGE =====
        self.parallel_section()

        # ===== TRANSITION 2 =====
        self.transition_to_triton()

        # ===== PART 4: TRITON EXECUTION =====
        self.triton_section()

        # ===== CONCLUSION =====
        self.conclusion_section()

    # -------------------------------------------------------------------------
    # PART 1: INTRODUCTION
    # -------------------------------------------------------------------------
    def intro_section(self):
        title = Text("Prefix Sum with Odd-Positive Masking", font_size=36)
        subtitle = Text("A GPU Kernel Optimization Challenge", font_size=24, color=GRAY)

        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.move_to(ORIGIN)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1)

        # Problem statement
        self.play(title_group.animate.to_edge(UP, buff=0.4))

        problem = VGroup(
            Text("The Challenge:", font_size=24, color=YELLOW),
            Text("Compute a prefix sum, but only accumulate", font_size=20),
            Text("values at positions where the count of", font_size=20),
            Text("positive numbers before it is ODD.", font_size=20, color=GREEN),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        problem.next_to(title_group, DOWN, buff=0.5)

        self.play(Write(problem), run_time=2)
        self.wait(1.5)

        # Clear for next section
        self.play(FadeOut(problem), FadeOut(title_group))

    # -------------------------------------------------------------------------
    # PART 2: ALGORITHM WALKTHROUGH
    # -------------------------------------------------------------------------
    def algorithm_section(self):
        section_title = Text("Part 1: The Algorithm", font_size=28, color=BLUE)
        section_title.to_edge(UP, buff=0.3)
        self.play(Write(section_title))

        # Example input
        values = [3, -1, 2, 5, -3]
        n = len(values)

        array_group = self.create_array(values, "x")
        array_group.next_to(section_title, DOWN, buff=0.5)
        self.play(FadeIn(array_group))
        self.wait(0.5)

        # Step 1: Identify positives
        step_text = Text("Step 1: Identify positive values", font_size=18, color=YELLOW)
        step_text.next_to(array_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        cells = array_group[0]
        positive_mask = [v > 0 for v in values]

        highlights = []
        for i, is_pos in enumerate(positive_mask):
            if is_pos:
                highlight = cells[i].copy().set_fill(GREEN, opacity=0.3)
                highlights.append(highlight)

        self.play(*[FadeIn(h) for h in highlights])

        is_pos_values = [1 if v > 0 else 0 for v in values]
        is_pos_group = self.create_array(is_pos_values, "is_pos", color=GREEN)
        is_pos_group.next_to(step_text, DOWN, buff=0.2)
        self.play(FadeIn(is_pos_group))
        self.wait(0.5)

        # Step 2: Count positives before each position
        self.play(FadeOut(step_text), *[FadeOut(h) for h in highlights])
        step_text = Text("Step 2: Count positives BEFORE each position", font_size=18, color=YELLOW)
        step_text.next_to(array_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        prefix_counts = []
        count = 0
        for v in values:
            prefix_counts.append(count)
            if v > 0:
                count += 1

        count_group = self.create_array(prefix_counts, "count", color=BLUE)
        count_group.next_to(is_pos_group, DOWN, buff=0.2)
        self.play(FadeIn(count_group))
        self.wait(0.5)

        # Step 3: Determine odd/even
        self.play(FadeOut(step_text), FadeOut(is_pos_group))
        self.play(count_group.animate.next_to(array_group, DOWN, buff=0.4))

        step_text = Text("Step 3: Include only where count is ODD", font_size=18, color=YELLOW)
        step_text.next_to(count_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        odd_even = ["E" if c % 2 == 0 else "O" for c in prefix_counts]
        include = [c % 2 == 1 for c in prefix_counts]

        odd_even_group = self.create_array_text(odd_even, "odd?", include)
        odd_even_group.next_to(step_text, DOWN, buff=0.2)
        self.play(FadeIn(odd_even_group))
        self.wait(0.5)

        # Step 4: Mask values
        self.play(FadeOut(step_text), FadeOut(count_group))
        self.play(odd_even_group.animate.next_to(array_group, DOWN, buff=0.4))

        step_text = Text("Step 4: Zero out excluded positions", font_size=18, color=YELLOW)
        step_text.next_to(odd_even_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        masked_values = [v if inc else 0 for v, inc in zip(values, include)]
        masked_group = self.create_array(masked_values, "masked", color=ORANGE)
        masked_group.next_to(step_text, DOWN, buff=0.2)
        self.play(FadeIn(masked_group))
        self.wait(0.5)

        # Step 5: Prefix sum
        self.play(FadeOut(step_text), FadeOut(odd_even_group))
        self.play(masked_group.animate.next_to(array_group, DOWN, buff=0.4))

        step_text = Text("Step 5: Compute prefix sum", font_size=18, color=YELLOW)
        step_text.next_to(masked_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        prefix_sum = []
        running = 0
        for v in masked_values:
            running += v
            prefix_sum.append(running)

        result_group = self.create_array(prefix_sum, "output", color=RED)
        result_group.next_to(step_text, DOWN, buff=0.2)
        self.play(FadeIn(result_group))
        self.wait(1)

        # Clear
        self.play(
            FadeOut(section_title), FadeOut(array_group), FadeOut(masked_group),
            FadeOut(step_text), FadeOut(result_group)
        )

    # -------------------------------------------------------------------------
    # TRANSITION 1: To Parallelization
    # -------------------------------------------------------------------------
    def transition_to_parallel(self):
        transition = VGroup(
            Text("This algorithm is simple sequentially...", font_size=24),
            Text("But we need to run it on a GPU with", font_size=24),
            Text("thousands of parallel threads!", font_size=28, color=YELLOW),
        ).arrange(DOWN, buff=0.3)

        self.play(Write(transition[0]))
        self.wait(0.5)
        self.play(Write(transition[1]))
        self.play(Write(transition[2]))
        self.wait(1.5)

        question = Text("How do we parallelize a sequential dependency?",
                       font_size=24, color=RED)
        question.next_to(transition, DOWN, buff=0.5)
        self.play(Write(question))
        self.wait(1)

        self.play(FadeOut(transition), FadeOut(question))

    # -------------------------------------------------------------------------
    # PART 3: PARALLELIZATION CHALLENGE
    # -------------------------------------------------------------------------
    def parallel_section(self):
        section_title = Text("Part 2: The Parallelization Challenge", font_size=28, color=BLUE)
        section_title.to_edge(UP, buff=0.3)
        self.play(Write(section_title))

        # The problem
        problem = VGroup(
            Text("The Problem:", font_size=22, color=RED),
            Text("Each position needs ALL previous values", font_size=18),
            Text("→ Inherently sequential!", font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        problem.next_to(section_title, DOWN, buff=0.4)
        self.play(Write(problem))
        self.wait(1)

        # The solution
        solution = VGroup(
            Text("The Solution: Parallel Scan", font_size=22, color=GREEN),
            Text("• Divide input into blocks", font_size=16),
            Text("• Each block computes locally (parallel)", font_size=16),
            Text("• Combine block results", font_size=16),
            Text("• Adjust with global offsets (parallel)", font_size=16),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        solution.next_to(problem, DOWN, buff=0.4)
        self.play(Write(solution), run_time=2)
        self.wait(1)

        # Complexity
        complexity = Text("Complexity: O(n) work, O(log n) depth", font_size=20, color=ORANGE)
        complexity.next_to(solution, DOWN, buff=0.4)
        self.play(Write(complexity))
        self.wait(1.5)

        self.play(FadeOut(section_title), FadeOut(problem), FadeOut(solution), FadeOut(complexity))

    # -------------------------------------------------------------------------
    # TRANSITION 2: To Triton
    # -------------------------------------------------------------------------
    def transition_to_triton(self):
        transition = VGroup(
            Text("Triton makes this easier!", font_size=28, color=GREEN),
            Text("It provides high-level primitives for GPU programming", font_size=20),
            Text("Let's see how a Triton kernel processes data...", font_size=20, color=GRAY),
        ).arrange(DOWN, buff=0.3)

        self.play(Write(transition[0]))
        self.wait(0.3)
        self.play(Write(transition[1]))
        self.play(Write(transition[2]))
        self.wait(1.5)

        self.play(FadeOut(transition))

    # -------------------------------------------------------------------------
    # PART 4: TRITON EXECUTION
    # -------------------------------------------------------------------------
    def triton_section(self):
        section_title = Text("Part 3: Triton Kernel Execution", font_size=28, color=BLUE)
        section_title.to_edge(UP, buff=0.3)
        self.play(Write(section_title))

        # Create blocks visualization
        n = 16
        block_size = 4
        n_blocks = n // block_size

        blocks = VGroup()
        for b in range(n_blocks):
            block_group = VGroup()
            for _ in range(block_size):
                cell = Square(side_length=0.4)
                cell.set_stroke(WHITE, width=1)
                block_group.add(cell)
            block_group.arrange(RIGHT, buff=0.03)

            border = SurroundingRectangle(block_group, color=BLUE, buff=0.08)
            block_label = Text(f"B{b}", font_size=12, color=BLUE)
            block_label.next_to(border, UP, buff=0.08)

            blocks.add(VGroup(block_group, border, block_label))

        blocks.arrange(RIGHT, buff=0.25)
        blocks.next_to(section_title, DOWN, buff=0.6)
        self.play(FadeIn(blocks))

        # Phase 1
        phase1 = Text("Phase 1: Each block computes locally (parallel)", font_size=18, color=YELLOW)
        phase1.next_to(blocks, DOWN, buff=0.4)
        self.play(Write(phase1))
        self.play(*[block[1].animate.set_color(GREEN) for block in blocks])
        self.wait(0.8)

        # Phase 2
        self.play(FadeOut(phase1))
        phase2 = Text("Phase 2: Combine block aggregates", font_size=18, color=YELLOW)
        phase2.next_to(blocks, DOWN, buff=0.4)
        self.play(Write(phase2))

        arrows = VGroup()
        for i in range(n_blocks - 1):
            arrow = Arrow(
                blocks[i].get_right(),
                blocks[i+1].get_left(),
                buff=0.08,
                color=ORANGE,
                stroke_width=3
            )
            arrows.add(arrow)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2))
        self.wait(0.8)

        # Phase 3
        self.play(FadeOut(phase2), FadeOut(arrows))
        phase3 = Text("Phase 3: Add global offsets (parallel)", font_size=18, color=YELLOW)
        phase3.next_to(blocks, DOWN, buff=0.4)
        self.play(Write(phase3))
        self.play(*[block[1].animate.set_color(RED) for block in blocks])
        self.wait(0.8)

        # Result
        self.play(FadeOut(phase3))
        result = Text("Complete prefix sum in O(log n) parallel steps!", font_size=20, color=GREEN)
        result.next_to(blocks, DOWN, buff=0.4)
        self.play(Write(result))
        self.wait(1)

        self.play(FadeOut(section_title), FadeOut(blocks), FadeOut(result))

    # -------------------------------------------------------------------------
    # CONCLUSION
    # -------------------------------------------------------------------------
    def conclusion_section(self):
        title = Text("Summary", font_size=32, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        summary = VGroup(
            Text("1. Algorithm: Prefix sum with conditional masking", font_size=18),
            Text("   based on odd count of prior positives", font_size=16, color=GRAY),
            Text("", font_size=10),
            Text("2. Challenge: Sequential dependency requires", font_size=18),
            Text("   parallel scan techniques", font_size=16, color=GRAY),
            Text("", font_size=10),
            Text("3. Solution: Block-wise computation with", font_size=18),
            Text("   global offset adjustment in Triton", font_size=16, color=GRAY),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        summary.next_to(title, DOWN, buff=0.5)

        self.play(Write(summary), run_time=3)
        self.wait(1)

        # Final message
        final = VGroup(
            Text("Now optimize it for an NVIDIA A10G!", font_size=24, color=YELLOW),
            Text("mlpuzzles.com", font_size=20, color=BLUE),
        ).arrange(DOWN, buff=0.2)
        final.next_to(summary, DOWN, buff=0.5)

        self.play(Write(final))
        self.wait(2)

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------
    def create_array(self, values, label, color=WHITE):
        cells = VGroup()
        labels = VGroup()

        for v in values:
            cell = Square(side_length=0.55)
            cell.set_stroke(color, width=2)
            cells.add(cell)

            if isinstance(v, float):
                text = Text(f"{v:.0f}" if v == int(v) else f"{v:.1f}", font_size=16)
            else:
                text = Text(str(v), font_size=16)
            labels.add(text)

        cells.arrange(RIGHT, buff=0.06)

        for cell, lbl in zip(cells, labels):
            lbl.move_to(cell)

        row_label = Text(f"{label}:", font_size=14, color=GRAY)
        row_label.next_to(cells, LEFT, buff=0.2)

        return VGroup(cells, labels, row_label)

    def create_array_text(self, texts, label, highlights=None):
        cells = VGroup()
        labels = VGroup()

        for i, t in enumerate(texts):
            cell = Square(side_length=0.55)
            color = GREEN if highlights and highlights[i] else RED
            cell.set_stroke(color, width=2)
            if highlights and highlights[i]:
                cell.set_fill(GREEN, opacity=0.2)
            cells.add(cell)

            text = Text(t, font_size=14, color=color)
            labels.add(text)

        cells.arrange(RIGHT, buff=0.06)

        for cell, lbl in zip(cells, labels):
            lbl.move_to(cell)

        row_label = Text(f"{label}:", font_size=14, color=GRAY)
        row_label.next_to(cells, LEFT, buff=0.2)

        return VGroup(cells, labels, row_label)
