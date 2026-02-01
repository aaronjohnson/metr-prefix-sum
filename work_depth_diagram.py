"""
Work-Depth Diagram for Prefix Sum with Odd-Positive Masking

Visualizes the parallelism structure of the algorithm:
- Work W(n): Total operations
- Depth D(n): Critical path length (span)
- Parallelism: W(n) / D(n)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# =============================================================================
# Work-Depth Analysis
# =============================================================================

def analyze_work_depth():
    """
    Analyze work and depth for each implementation variant.

    Algorithm has 3 main phases:
    1. Compute is_positive and prefix count
    2. Apply odd mask
    3. Compute final prefix sum

    Each prefix sum (scan) can be:
    - Sequential: W=O(n), D=O(n)
    - Parallel (Blelloch): W=O(n), D=O(log n)
    """

    analyses = {
        "Reference (Sequential)": {
            "work": "O(n)",
            "depth": "O(n)",
            "parallelism": "O(1)",
            "work_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "description": "Single loop, each element depends on previous",
            "color": "#e74c3c",
        },
        "Single-Block Triton": {
            "work": "O(n)",
            "depth": "O(log n)",
            "parallelism": "O(n/log n)",
            "work_coefficient": 2.0,  # Two scans
            "depth_coefficient": 2.0,  # log n for each scan
            "description": "Two parallel scans: pos_count + final sum",
            "color": "#3498db",
        },
        "Multi-Block (CPU Phase 2)": {
            "work": "O(n)",
            "depth": "O(n)",  # CPU phase is sequential!
            "parallelism": "O(1)",
            "work_coefficient": 2.0,
            "depth_coefficient": 1.0,  # Dominated by CPU loop
            "description": "GPU phases parallel, but CPU Phase 2 is O(n)",
            "color": "#f39c12",
        },
        "Multi-Block (All GPU)": {
            "work": "O(n)",
            "depth": "O(log n)",
            "parallelism": "O(n/log n)",
            "work_coefficient": 2.5,  # 3 phases with some overhead
            "depth_coefficient": 3.0,  # 3 sequential phases, each O(log n)
            "description": "3 phases, each O(log n) depth",
            "color": "#2ecc71",
        },
    }

    return analyses


# =============================================================================
# DAG Visualization for Parallel Scan
# =============================================================================

def create_scan_dag(ax, n=8, title="Parallel Prefix Sum (Blelloch)", y_offset=0):
    """
    Draw the DAG for a parallel prefix sum using Blelloch's algorithm.

    Up-sweep (reduce) + Down-sweep phases.
    """
    ax.set_xlim(-1, n + 1)

    # Node positions
    levels = int(np.log2(n)) + 1

    # Colors
    input_color = "#3498db"
    reduce_color = "#e74c3c"
    downsweep_color = "#2ecc71"
    output_color = "#9b59b6"

    # Draw input level
    for i in range(n):
        circle = plt.Circle((i + 0.5, y_offset), 0.3, color=input_color,
                            ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(i + 0.5, y_offset, f'x[{i}]', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    # Up-sweep (reduce) phase
    y = y_offset + 1
    for level in range(levels - 1):
        stride = 2 ** (level + 1)
        for i in range(stride - 1, n, stride):
            # Draw node
            circle = plt.Circle((i + 0.5, y), 0.25, color=reduce_color,
                                ec='black', linewidth=1, zorder=3)
            ax.add_patch(circle)
            ax.text(i + 0.5, y, '+', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

            # Draw arrows from children
            left_child = i - 2**level
            # Arrow from left child
            ax.annotate('', xy=(i + 0.5, y - 0.25),
                       xytext=(left_child + 0.5, y - 0.75),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            # Arrow from right child (previous level or input)
            ax.annotate('', xy=(i + 0.5, y - 0.25),
                       xytext=(i + 0.5, y - 0.75),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        y += 1

    # Mark the depth
    total_depth = 2 * (levels - 1)

    # Down-sweep phase
    y_down = y
    for level in range(levels - 2, -1, -1):
        stride = 2 ** (level + 1)
        for i in range(stride - 1, n, stride):
            if level > 0:
                right_child = i
                left_child = i - 2**level

                # Node at left child position
                circle = plt.Circle((left_child + 0.5, y_down), 0.25,
                                   color=downsweep_color, ec='black',
                                   linewidth=1, zorder=3)
                ax.add_patch(circle)
                ax.text(left_child + 0.5, y_down, '+', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')

        y_down += 1

    # Output level
    y_out = y_down
    for i in range(n):
        circle = plt.Circle((i + 0.5, y_out), 0.3, color=output_color,
                            ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(i + 0.5, y_out, f'Σ', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Annotations
    ax.text(-0.8, y_offset, 'Input', ha='right', va='center', fontsize=9)
    ax.text(-0.8, y_offset + (levels-1)/2, 'Up-sweep\n(Reduce)', ha='right',
            va='center', fontsize=9, color=reduce_color)
    ax.text(-0.8, y_down - 0.5, 'Down-sweep\n(Scan)', ha='right',
            va='center', fontsize=9, color=downsweep_color)
    ax.text(-0.8, y_out, 'Output', ha='right', va='center', fontsize=9)

    # Depth indicator
    ax.annotate('', xy=(n + 0.7, y_offset), xytext=(n + 0.7, y_out),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(n + 0.9, (y_offset + y_out) / 2,
            f'Depth\nO(log n)\n= {total_depth}',
            ha='left', va='center', fontsize=9, fontweight='bold')

    return y_out + 1


def create_sequential_dag(ax, n=8, title="Sequential Scan", y_offset=0):
    """Draw the DAG for sequential prefix sum."""

    input_color = "#3498db"
    op_color = "#e74c3c"
    output_color = "#9b59b6"

    # Input nodes
    for i in range(n):
        circle = plt.Circle((i + 0.5, y_offset), 0.3, color=input_color,
                            ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(i + 0.5, y_offset, f'x[{i}]', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    # Sequential chain
    y = y_offset + 1
    for i in range(n):
        # Operation node
        circle = plt.Circle((i + 0.5, y), 0.25, color=op_color,
                            ec='black', linewidth=1, zorder=3)
        ax.add_patch(circle)
        ax.text(i + 0.5, y, '+', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Arrow from input
        ax.annotate('', xy=(i + 0.5, y - 0.25),
                   xytext=(i + 0.5, y_offset + 0.3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        # Arrow from previous (except first)
        if i > 0:
            ax.annotate('', xy=(i + 0.5 - 0.2, y),
                       xytext=(i - 0.5 + 0.25, y),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Output nodes
    y_out = y + 1
    for i in range(n):
        circle = plt.Circle((i + 0.5, y_out), 0.3, color=output_color,
                            ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(i + 0.5, y_out, f'Σ', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        ax.annotate('', xy=(i + 0.5, y_out - 0.3),
                   xytext=(i + 0.5, y + 0.25),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Critical path highlight
    ax.plot([0.5, n - 0.5], [y, y], 'r-', linewidth=3, alpha=0.3, zorder=1)

    # Depth indicator
    ax.annotate('', xy=(n + 0.7, y_offset), xytext=(n + 0.7, y_out),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(n + 0.9, (y_offset + y_out) / 2,
            f'Depth\nO(n)\n= {n}',
            ha='left', va='center', fontsize=9, fontweight='bold')

    # Annotations
    ax.text(-0.8, y_offset, 'Input', ha='right', va='center', fontsize=9)
    ax.text(-0.8, y, 'Sequential\nChain', ha='right', va='center',
            fontsize=9, color=op_color)
    ax.text(-0.8, y_out, 'Output', ha='right', va='center', fontsize=9)

    return y_out + 1


# =============================================================================
# Full Algorithm DAG (Odd-Positive Prefix Sum)
# =============================================================================

def create_full_algorithm_dag(ax, n=8):
    """
    Draw the complete algorithm DAG showing:
    1. is_positive computation (parallel)
    2. prefix count of positives (scan)
    3. odd mask computation (parallel)
    4. masked value selection (parallel)
    5. final prefix sum (scan)
    """

    colors = {
        'input': '#3498db',
        'compare': '#e67e22',
        'scan1': '#e74c3c',
        'mask': '#9b59b6',
        'select': '#1abc9c',
        'scan2': '#2ecc71',
        'output': '#34495e',
    }

    y = 0
    box_height = 0.8
    spacing = 1.2

    # Helper to draw a processing stage
    def draw_stage(y, label, color, width_ratio=1.0, depth_label=None):
        box = FancyBboxPatch((0.5, y), n * width_ratio, box_height,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(n * width_ratio / 2 + 0.5, y + box_height/2, label,
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

        if depth_label:
            ax.text(n * width_ratio + 1, y + box_height/2, depth_label,
                    ha='left', va='center', fontsize=9,
                    family='monospace', color='gray')

        return y + box_height + spacing * 0.3

    # Draw stages
    y = draw_stage(y, f'Input x[0:{n}]', colors['input'], depth_label='')

    # Arrows between stages
    def draw_arrow(y_from, y_to):
        ax.annotate('', xy=(n/2 + 0.5, y_to),
                   xytext=(n/2 + 0.5, y_from + box_height),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, 'is_positive = (x > 0)', colors['compare'],
                   depth_label='D = O(1)')

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, 'pos_count = cumsum(is_positive)', colors['scan1'],
                   depth_label='D = O(log n)')

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, 'exclusive = inclusive - is_positive', colors['compare'],
                   depth_label='D = O(1)')

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, 'mask = (exclusive & 1) == 1', colors['mask'],
                   depth_label='D = O(1)')

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, 'masked_x = where(mask, x, 0)', colors['select'],
                   depth_label='D = O(1)')

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, 'result = cumsum(masked_x)', colors['scan2'],
                   depth_label='D = O(log n)')

    y_prev = y - spacing * 0.3
    y = y_prev + spacing
    draw_arrow(y_prev, y)
    y = draw_stage(y, f'Output result[0:{n}]', colors['output'], depth_label='')

    # Total depth annotation
    ax.annotate('', xy=(n + 2.5, 0), xytext=(n + 2.5, y - spacing * 0.3),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(n + 2.8, y / 2, 'Total Depth\nD = O(log n)',
            ha='left', va='center', fontsize=10, fontweight='bold')

    # Work annotation
    ax.text(n + 2.8, y / 2 - 2, 'Total Work\nW = O(n)',
            ha='left', va='center', fontsize=10, fontweight='bold',
            color='#2c3e50')

    ax.set_xlim(-0.5, n + 5)
    ax.set_ylim(-0.5, y + 0.5)

    return y


# =============================================================================
# Comparison Chart
# =============================================================================

def create_comparison_chart(ax):
    """Create a bar chart comparing work and depth across implementations."""

    analyses = analyze_work_depth()
    names = list(analyses.keys())
    n = 1024  # Example size

    # Calculate actual values for n=1024
    log_n = np.log2(n)

    data = {
        "Reference (Sequential)": {"work": n, "depth": n},
        "Single-Block Triton": {"work": 2 * n, "depth": 2 * log_n},
        "Multi-Block (CPU Phase 2)": {"work": 2 * n, "depth": n},  # CPU dominates
        "Multi-Block (All GPU)": {"work": 2.5 * n, "depth": 3 * log_n},
    }

    x = np.arange(len(names))
    width = 0.35

    # Normalize for visualization (log scale would be better but let's use ratio)
    works = [data[name]["work"] / n for name in names]
    depths = [data[name]["depth"] / n for name in names]
    parallelisms = [data[name]["work"] / data[name]["depth"] for name in names]

    colors = [analyses[name]["color"] for name in names]

    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, works, width, label='Work (normalized)',
                   color=colors, alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, depths, width, label='Depth (normalized)',
                   color=colors, alpha=0.4, edgecolor='black', hatch='//')

    # Add parallelism labels
    for i, (w, d, p) in enumerate(zip(works, depths, parallelisms)):
        ax.annotate(f'P={p:.0f}', xy=(i, max(w, d) + 0.1),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Normalized Value (÷n)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    ax.legend(loc='upper right')
    ax.set_title(f'Work vs Depth Comparison (n={n})', fontsize=11, fontweight='bold')

    # Add horizontal line at 1 for reference
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(names) - 0.5, 1.05, 'n', ha='center', fontsize=9, color='gray')


def create_parallelism_scaling(ax):
    """Show how parallelism scales with input size."""

    n_values = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    # Parallelism = Work / Depth
    sequential_p = np.ones_like(n_values, dtype=float)  # Always 1
    parallel_p = n_values / (2 * np.log2(n_values))  # n / (2 log n)
    cpu_phase2_p = np.ones_like(n_values, dtype=float)  # Bottlenecked to 1
    all_gpu_p = n_values / (3 * np.log2(n_values))  # n / (3 log n)

    ax.semilogx(n_values, sequential_p, 'o-', label='Reference (Sequential)',
                color='#e74c3c', linewidth=2, markersize=6)
    ax.semilogx(n_values, parallel_p, 's-', label='Single-Block Triton',
                color='#3498db', linewidth=2, markersize=6)
    ax.semilogx(n_values, cpu_phase2_p, '^-', label='Multi-Block (CPU Phase 2)',
                color='#f39c12', linewidth=2, markersize=6)
    ax.semilogx(n_values, all_gpu_p, 'D-', label='Multi-Block (All GPU)',
                color='#2ecc71', linewidth=2, markersize=6)

    ax.set_xlabel('Input Size (n)', fontsize=10)
    ax.set_ylabel('Parallelism (W/D)', fontsize=10)
    ax.set_title('Parallelism Scaling with Input Size', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate key insight
    ax.annotate('CPU Phase 2\nbottleneck!',
                xy=(4096, 1), xytext=(8192, 100),
                fontsize=9, color='#f39c12',
                arrowprops=dict(arrowstyle='->', color='#f39c12'))


# =============================================================================
# Main Figure
# =============================================================================

def create_work_depth_figure(save_path="work_depth_diagram.png"):
    """Create the complete work-depth diagram figure."""

    fig = plt.figure(figsize=(16, 14))

    # Layout: 2x2 grid with bottom row being the full algorithm
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

    # Top left: Sequential DAG
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_aspect('equal')
    create_sequential_dag(ax1, n=8)
    ax1.set_title('Sequential Prefix Sum\nW=O(n), D=O(n), Parallelism=1',
                  fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Top right: Parallel DAG (simplified)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_aspect('equal')
    create_scan_dag(ax2, n=8)
    ax2.set_title('Parallel Prefix Sum (Blelloch)\nW=O(n), D=O(log n), Parallelism=O(n/log n)',
                  fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Middle left: Comparison chart
    ax3 = fig.add_subplot(gs[1, 0])
    create_comparison_chart(ax3)

    # Middle right: Parallelism scaling
    ax4 = fig.add_subplot(gs[1, 1])
    create_parallelism_scaling(ax4)

    # Bottom: Full algorithm DAG
    ax5 = fig.add_subplot(gs[2, :])
    create_full_algorithm_dag(ax5, n=8)
    ax5.set_title('Complete Algorithm: Prefix Sum with Odd-Positive Masking\n'
                  '(Single-Block Triton Implementation)',
                  fontsize=12, fontweight='bold', pad=10)
    ax5.axis('off')

    # Main title
    fig.suptitle('Work-Depth Analysis: Prefix Sum with Odd-Positive Masking',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved work-depth diagram to: {save_path}")

    return fig


# =============================================================================
# Text Summary
# =============================================================================

def print_work_depth_summary():
    """Print a text summary of the work-depth analysis."""

    print("\n" + "=" * 80)
    print("WORK-DEPTH ANALYSIS: Prefix Sum with Odd-Positive Masking")
    print("=" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM STRUCTURE                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. is_positive = (x > 0)           →  W=O(n), D=O(1)    [Parallel map]     │
│  2. pos_count = cumsum(is_positive) →  W=O(n), D=O(log n) [Parallel scan]   │
│  3. exclusive = inclusive - is_pos  →  W=O(n), D=O(1)    [Parallel map]     │
│  4. mask = (exclusive & 1) == 1     →  W=O(n), D=O(1)    [Parallel map]     │
│  5. masked_x = where(mask, x, 0)    →  W=O(n), D=O(1)    [Parallel map]     │
│  6. result = cumsum(masked_x)       →  W=O(n), D=O(log n) [Parallel scan]   │
│                                                                              │
│  TOTAL: W = O(n), D = O(log n)                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ IMPLEMENTATION COMPARISON                                                    │
├──────────────────────────┬──────────┬──────────┬─────────────┬──────────────┤
│ Implementation           │ Work     │ Depth    │ Parallelism │ A10G (84 SM) │
├──────────────────────────┼──────────┼──────────┼─────────────┼──────────────┤
│ Reference (Sequential)   │ O(n)     │ O(n)     │ 1           │ Uses 1 core  │
│ Single-Block Triton      │ O(n)     │ O(log n) │ O(n/log n)  │ 1 SM         │
│ Multi-Block (CPU Phase2) │ O(n)     │ O(n)     │ 1           │ 84 SM idle!  │
│ Multi-Block (All GPU)    │ O(n)     │ O(log n) │ O(n/log n)  │ All 84 SMs   │
└──────────────────────────┴──────────┴──────────┴─────────────┴──────────────┘
""")

    print("""
CONCRETE EXAMPLE (n = 16384):
─────────────────────────────────────────────────────────────────────────────

  Implementation              Work        Depth       Parallelism   Speedup
  ─────────────────────────────────────────────────────────────────────────
  Reference (Sequential)      16384       16384       1             1x
  Single-Block Triton         32768       28          1170          1170x
  Multi-Block (CPU Phase 2)   32768       16384       2             2x
  Multi-Block (All GPU)       40960       42          975           975x

KEY INSIGHT:
  The CPU Phase 2 bottleneck reduces parallelism from O(n/log n) back to O(1),
  wasting 83 of 84 available SMs on the A10G!
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print_work_depth_summary()
    create_work_depth_figure()

    print("\nTo view the diagram:")
    print("  Open: work_depth_diagram.png")
