"""
Manim animation explaining the Prefix Sum with Odd-Positive Masking algorithm.

Run with:
    manim -pql explain_algorithm.py OddPositivePrefixSum

For higher quality:
    manim -pqh explain_algorithm.py OddPositivePrefixSum
"""

from manim import *


class OddPositivePrefixSum(Scene):
    def construct(self):
        # Title
        title = Text("Prefix Sum with Odd-Positive Masking", font_size=32)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.5)

        # Example input
        values = [3, -1, 2, 5, -3]
        n = len(values)

        # Create array visualization - smaller cells
        array_group = self.create_array(values, "x")
        array_group.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(array_group))
        self.wait(0.5)

        # Step 1: Show positive detection
        step_text = Text("Step 1: Identify positive values", font_size=20, color=YELLOW)
        step_text.next_to(array_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        # Highlight positives
        cells = array_group[0]  # The rectangles
        positive_mask = [v > 0 for v in values]

        highlights = []
        for i, is_pos in enumerate(positive_mask):
            if is_pos:
                highlight = cells[i].copy().set_fill(GREEN, opacity=0.3)
                highlights.append(highlight)

        self.play(*[FadeIn(h) for h in highlights])
        self.wait(0.5)

        # Create is_positive row
        is_pos_values = [1 if v > 0 else 0 for v in values]
        is_pos_group = self.create_array(is_pos_values, "is_pos", color=GREEN)
        is_pos_group.next_to(step_text, DOWN, buff=0.25)
        self.play(FadeIn(is_pos_group))
        self.wait(0.5)

        # Step 2: Exclusive prefix count - fade out step 1 elements
        self.play(
            FadeOut(step_text),
            *[FadeOut(h) for h in highlights],
        )
        step_text = Text("Step 2: Count positives BEFORE each position",
                          font_size=20, color=YELLOW)
        step_text.next_to(array_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        # Animate counting
        prefix_counts = []
        count = 0
        for i, v in enumerate(values):
            prefix_counts.append(count)
            if v > 0:
                count += 1

        count_group = self.create_array(prefix_counts, "count", color=BLUE)
        count_group.next_to(is_pos_group, DOWN, buff=0.2)

        # Animate each count appearing
        count_cells = count_group[0]
        count_labels = count_group[1]

        self.play(FadeIn(count_group[2]))  # Row label

        running_count = 0
        for i in range(n):
            self.play(
                FadeIn(count_cells[i]),
                FadeIn(count_labels[i]),
                run_time=0.3
            )
            if values[i] > 0:
                running_count += 1

        self.wait(0.5)

        # Step 3: Odd/Even determination - fade out is_pos row to save space
        self.play(FadeOut(step_text), FadeOut(is_pos_group))

        # Move count_group up
        self.play(count_group.animate.next_to(array_group, DOWN, buff=0.4))

        step_text = Text("Step 3: Include only where count is ODD",
                          font_size=20, color=YELLOW)
        step_text.next_to(count_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        # Create odd/even labels
        odd_even = ["even" if c % 2 == 0 else "odd" for c in prefix_counts]
        include = [c % 2 == 1 for c in prefix_counts]

        odd_even_group = self.create_array_text(odd_even, "odd?", include)
        odd_even_group.next_to(step_text, DOWN, buff=0.25)
        self.play(FadeIn(odd_even_group))
        self.wait(0.5)

        # Step 4: Create masked values - fade out count row
        self.play(FadeOut(step_text), FadeOut(count_group))

        # Move odd_even_group up
        self.play(odd_even_group.animate.next_to(array_group, DOWN, buff=0.4))

        step_text = Text("Step 4: Zero out excluded positions",
                          font_size=20, color=YELLOW)
        step_text.next_to(odd_even_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        masked_values = [v if inc else 0 for v, inc in zip(values, include)]
        masked_group = self.create_array(masked_values, "masked", color=ORANGE)
        masked_group.next_to(step_text, DOWN, buff=0.25)
        self.play(FadeIn(masked_group))
        self.wait(0.5)

        # Step 5: Compute prefix sum - fade out odd_even row
        self.play(FadeOut(step_text), FadeOut(odd_even_group))

        # Move masked_group up
        self.play(masked_group.animate.next_to(array_group, DOWN, buff=0.4))

        step_text = Text("Step 5: Compute prefix sum of masked values",
                          font_size=20, color=YELLOW)
        step_text.next_to(masked_group, DOWN, buff=0.3)
        self.play(Write(step_text))

        # Calculate prefix sum
        prefix_sum = []
        running = 0
        for v in masked_values:
            running += v
            prefix_sum.append(running)

        result_group = self.create_array(prefix_sum, "output", color=RED)
        result_group.next_to(step_text, DOWN, buff=0.25)

        # Animate prefix sum building
        result_cells = result_group[0]
        result_labels = result_group[1]

        self.play(FadeIn(result_group[2]))  # Row label

        for i in range(n):
            if i == 0:
                arrow = Arrow(
                    masked_group[0][0].get_bottom(),
                    result_cells[0].get_top(),
                    buff=0.05,
                    color=RED,
                    stroke_width=3
                )
            else:
                arrow = Arrow(
                    result_cells[i-1].get_right() + RIGHT * 0.05,
                    result_cells[i].get_left() + LEFT * 0.05,
                    buff=0.02,
                    color=RED,
                    stroke_width=3
                )

            self.play(
                GrowArrow(arrow),
                FadeIn(result_cells[i]),
                FadeIn(result_labels[i]),
                run_time=0.4
            )
            self.play(FadeOut(arrow), run_time=0.15)

        self.wait(0.5)

        # Final summary
        self.play(FadeOut(step_text))

        summary = VGroup(
            Text("Result: [0, -1, 1, 1, 1]", font_size=24, color=RED),
            Text("Positions 1,2 contributed (odd positive count before them)",
                 font_size=16, color=GRAY)
        ).arrange(DOWN, buff=0.15)
        summary.next_to(result_group, DOWN, buff=0.4)

        self.play(Write(summary))
        self.wait(2)

    def create_array(self, values, label, color=WHITE):
        """Create a row of cells with values."""
        cells = VGroup()
        labels = VGroup()

        for v in values:
            cell = Square(side_length=0.6)  # Smaller cells
            cell.set_stroke(color, width=2)
            cells.add(cell)

            if isinstance(v, float):
                text = Text(f"{v:.0f}" if v == int(v) else f"{v:.1f}", font_size=18)
            else:
                text = Text(str(v), font_size=18)
            labels.add(text)

        cells.arrange(RIGHT, buff=0.08)

        for cell, lbl in zip(cells, labels):
            lbl.move_to(cell)

        row_label = Text(f"{label}:", font_size=16, color=GRAY)
        row_label.next_to(cells, LEFT, buff=0.2)

        return VGroup(cells, labels, row_label)

    def create_array_text(self, texts, label, highlights=None):
        """Create a row of cells with text labels."""
        cells = VGroup()
        labels = VGroup()

        for i, t in enumerate(texts):
            cell = Square(side_length=0.6)  # Smaller cells
            color = GREEN if highlights and highlights[i] else RED
            cell.set_stroke(color, width=2)
            if highlights and highlights[i]:
                cell.set_fill(GREEN, opacity=0.2)
            cells.add(cell)

            text = Text(t, font_size=14, color=color)
            labels.add(text)

        cells.arrange(RIGHT, buff=0.08)

        for cell, lbl in zip(cells, labels):
            lbl.move_to(cell)

        row_label = Text(f"{label}:", font_size=16, color=GRAY)
        row_label.next_to(cells, LEFT, buff=0.2)

        return VGroup(cells, labels, row_label)


class ParallelScanExplanation(Scene):
    """Explains why this is hard to parallelize."""

    def construct(self):
        title = Text("The Parallelization Challenge", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Problem statement
        problem = VGroup(
            Text("Sequential dependency:", font_size=24, color=YELLOW),
            Text("Each position needs to know the count of ALL previous positives",
                 font_size=20),
            Text("→ Classic prefix sum / scan problem", font_size=20, color=BLUE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        problem.next_to(title, DOWN, buff=0.5)
        self.play(Write(problem))
        self.wait(1)

        # Parallel approach
        approach = VGroup(
            Text("Parallel Scan Solution:", font_size=24, color=GREEN),
            Text("1. Divide input into blocks", font_size=20),
            Text("2. Each block computes LOCAL prefix in parallel", font_size=20),
            Text("3. Combine block results (small sequential step)", font_size=20),
            Text("4. Add block offsets to get GLOBAL prefix", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        approach.next_to(problem, DOWN, buff=0.5)
        self.play(Write(approach))
        self.wait(1)

        # Complexity
        complexity = VGroup(
            Text("Complexity: O(n) work, O(log n) depth", font_size=24, color=ORANGE),
            Text("On GPU: Thousands of threads work simultaneously!", font_size=20)
        ).arrange(DOWN, buff=0.2)
        complexity.next_to(approach, DOWN, buff=0.5)
        self.play(Write(complexity))
        self.wait(2)


class TritonKernelVisualization(Scene):
    """Visualizes how the Triton kernel processes data."""

    def construct(self):
        title = Text("Triton Kernel Execution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Create a larger array split into blocks
        n = 16
        values = [3, -1, 2, 5, -3, 1, -2, 4, -1, 2, -3, 1, 5, -2, 3, -1]
        block_size = 4
        n_blocks = n // block_size

        # Visualize blocks
        blocks = VGroup()
        for b in range(n_blocks):
            block_vals = values[b*block_size:(b+1)*block_size]
            block_group = VGroup()

            for v in block_vals:
                cell = Square(side_length=0.5)
                cell.set_stroke(WHITE, width=1)
                text = Text(str(v), font_size=14)
                cell.add(text)
                block_group.add(cell)

            block_group.arrange(RIGHT, buff=0.05)

            # Block border
            border = SurroundingRectangle(block_group, color=BLUE, buff=0.1)
            block_label = Text(f"Block {b}", font_size=12, color=BLUE)
            block_label.next_to(border, UP, buff=0.1)

            blocks.add(VGroup(block_group, border, block_label))

        blocks.arrange(RIGHT, buff=0.3)
        blocks.next_to(title, DOWN, buff=0.8)

        self.play(FadeIn(blocks))
        self.wait(0.5)

        # Phase 1: Local computation
        phase1 = Text("Phase 1: Each block computes locally (parallel)",
                      font_size=20, color=YELLOW)
        phase1.next_to(blocks, DOWN, buff=0.5)
        self.play(Write(phase1))

        # Highlight all blocks simultaneously
        self.play(*[block[1].animate.set_color(GREEN) for block in blocks])
        self.wait(0.5)

        # Phase 2: Aggregate
        self.play(FadeOut(phase1))
        phase2 = Text("Phase 2: Combine block aggregates",
                      font_size=20, color=YELLOW)
        phase2.next_to(blocks, DOWN, buff=0.5)
        self.play(Write(phase2))

        # Show arrows between blocks
        arrows = VGroup()
        for i in range(n_blocks - 1):
            arrow = Arrow(
                blocks[i].get_right(),
                blocks[i+1].get_left(),
                buff=0.1,
                color=ORANGE
            )
            arrows.add(arrow)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3))
        self.wait(0.5)

        # Phase 3: Final adjustment
        self.play(FadeOut(phase2), FadeOut(arrows))
        phase3 = Text("Phase 3: Add global offsets to each block (parallel)",
                      font_size=20, color=YELLOW)
        phase3.next_to(blocks, DOWN, buff=0.5)
        self.play(Write(phase3))

        self.play(*[block[1].animate.set_color(RED) for block in blocks])
        self.wait(1)

        # Result
        result = Text("Result: Full prefix sum computed in O(log n) parallel steps!",
                      font_size=24, color=GREEN)
        result.next_to(phase3, DOWN, buff=0.5)
        self.play(Write(result))
        self.wait(2)
