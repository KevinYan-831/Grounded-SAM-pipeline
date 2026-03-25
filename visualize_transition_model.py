#!/usr/bin/env python3
"""
Visualize the transition probability model as a state diagram and heatmap.
"""

import json
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_heatmap(ax, prob_matrix, labels_short):
    """Draw transition probability heatmap."""
    im = ax.imshow(prob_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_short)))
    ax.set_yticks(range(len(labels_short)))
    ax.set_xticklabels(labels_short, fontsize=10, rotation=30, ha="right")
    ax.set_yticklabels(labels_short, fontsize=10)
    ax.set_xlabel("To State", fontsize=12, fontweight="bold")
    ax.set_ylabel("From State", fontsize=12, fontweight="bold")
    ax.set_title("Transition Probability Matrix", fontsize=14, fontweight="bold", pad=12)

    # Annotate cells
    for i in range(len(labels_short)):
        for j in range(len(labels_short)):
            val = prob_matrix[i][j]
            color = "white" if val > 0.5 else "black"
            text = f"{val:.3f}" if val > 0 else "0"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold" if val > 0.1 else "normal")

    return im


def draw_state_diagram(ax, prob_matrix, count_matrix, labels_short, colors):
    """Draw state transition diagram with arrows."""
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("State Transition Diagram", fontsize=14, fontweight="bold", pad=12)

    n = len(labels_short)
    # Position states in a circle
    angles = [np.pi/2, np.pi, -np.pi/2, 0]  # top, left, bottom, right
    positions = [(1.4 * np.cos(a), 1.4 * np.sin(a)) for a in angles]
    radius = 0.45

    # Draw states as circles
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), radius, facecolor=colors[i],
                           edgecolor="black", linewidth=2, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        # State name (wrap text)
        name = labels_short[i]
        ax.text(x, y + 0.08, name, ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=4)
        # Total count from this state
        total = sum(count_matrix[i])
        ax.text(x, y - 0.18, f"n={total}", ha="center", va="center",
                fontsize=7.5, color="gray", zorder=4)

    # Draw transitions (arrows)
    for i in range(n):
        for j in range(n):
            p = prob_matrix[i][j]
            if p < 0.005 or i == j:
                continue
            x1, y1 = positions[i]
            x2, y2 = positions[j]

            # Shorten arrow to not overlap circles
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist == 0:
                continue
            ux, uy = dx / dist, dy / dist

            # Offset start/end by radius
            sx = x1 + ux * (radius + 0.05)
            sy = y1 + uy * (radius + 0.05)
            ex = x2 - ux * (radius + 0.05)
            ey = y2 - uy * (radius + 0.05)

            # Slight curve for bidirectional arrows
            curve = 0.15 if prob_matrix[j][i] > 0.005 else 0
            style = f"arc3,rad={curve}"

            arrow_width = max(0.5, p * 4)
            alpha = max(0.3, min(1.0, p * 2))

            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                       arrowprops=dict(arrowstyle="-|>",
                                      connectionstyle=style,
                                      lw=arrow_width,
                                      color="darkslategray",
                                      alpha=alpha,
                                      mutation_scale=15),
                       zorder=2)

            # Label on arrow
            mx = (sx + ex) / 2 + curve * (-(ey - sy)) * 0.6
            my = (sy + ey) / 2 + curve * (ex - sx) * 0.6
            ax.text(mx, my, f"{p:.2%}", ha="center", va="center",
                   fontsize=7, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="gray", alpha=0.9),
                   zorder=5)

    # Self-loops
    loop_offsets = [(0, 0.55), (-0.55, 0), (0, -0.55), (0.55, 0)]
    for i in range(n):
        p = prob_matrix[i][i]
        if p < 0.005:
            continue
        x, y = positions[i]
        ox, oy = loop_offsets[i]
        lx, ly = x + ox, y + oy

        arc = mpatches.FancyArrowPatch(
            (x + ox * 0.7, y + oy * 0.7),
            (x + ox * 0.65 + 0.1, y + oy * 0.65),
            connectionstyle="arc3,rad=0.8",
            arrowstyle="-|>",
            lw=max(0.5, p * 3),
            color="darkslategray",
            alpha=max(0.3, min(1.0, p)),
            mutation_scale=12,
            zorder=2,
        )
        ax.add_patch(arc)
        ax.text(lx, ly, f"{p:.1%}", ha="center", va="center",
               fontsize=7, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.15", facecolor="lightyellow",
                        edgecolor="gray", alpha=0.9),
               zorder=5)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/jixin/labeling_trajectories_manual/transition_model.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/home/jixin/labeling_trajectories_manual/transition_model_visualization.png"

    with open(model_path) as f:
        model = json.load(f)

    prob_matrix = np.array(model["transition_probability_matrix"])
    count_matrix = np.array(model["transition_count_matrix"])

    labels_short = ["Normal", "Unstable", "Permanent", "Transient"]
    colors = ["#4CAF50", "#FFC107", "#F44336", "#2196F3"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle(
        f"Welding Process Transition Model — {model['num_trajectories']} Trajectories, "
        f"{int(count_matrix.sum())} Transitions",
        fontsize=15, fontweight="bold", y=0.98
    )

    # Left: state diagram
    draw_state_diagram(axes[0], prob_matrix.tolist(), count_matrix.tolist(),
                       labels_short, colors)

    # Right: heatmap
    im = draw_heatmap(axes[1], prob_matrix.tolist(), labels_short)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="Probability")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
