"""Comparative Metrics Visualization and Charting Module for Q-Nav.

Generates Matplotlib and Plotly comparison charts for:
- Success rate
- Path length
- Collision count
- Step count
- Navigation efficiency
- Multi-dimensional performance summary
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px



def plot_metric_bar_comparison(
    comparison_data: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Generate a multi-panel comparison chart of key navigation metrics.

    Panels:
    1. Mean Path Length & Steps
    2. Collisions & Safety Interventions
    3. Success Rate & Navigation Efficiency
    4. Per-Scenario Path Length Breakdown

    Args:
        comparison_data: Dictionary produced by ModelComparator.run_benchmark().
        save_path: Optional path to save high-resolution figure.

    Returns:
        Matplotlib Figure object.
    """
    summary = comparison_data.get("summary", {})
    c_sum = summary.get("classical", {})
    q_sum = summary.get("quantum", {})
    scenarios = comparison_data.get("scenario_comparisons", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    labels = ["Classical APF", "Quantum-Assisted"]

    # -------------------------------------------------------------
    # 1. Success Rate & Efficiency
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#FFFFFF")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax1.spines.values():
        sp.set_color("#E2E8F0")

    x = np.arange(2)
    width = 0.35

    sr = [c_sum.get("success_rate", 0.0) * 100, q_sum.get("success_rate", 0.0) * 100]
    eff = [
        c_sum.get("mean_navigation_efficiency", 0.0) * 100,
        q_sum.get("mean_navigation_efficiency", 0.0) * 100,
    ]

    rects1 = ax1.bar(
        x - width / 2, sr, width, label="Success Rate (%)", color="#16A34A", alpha=0.9
    )
    rects2 = ax1.bar(
        x + width / 2, eff, width, label="Efficiency (%)", color="#172033", alpha=0.9
    )

    ax1.set_ylabel("Percentage (%)", color="#64748B", fontsize=10)
    ax1.set_title("Mission Success & Navigation Efficiency", color="#172033", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="#172033", fontsize=10, fontweight="600")
    ax1.set_ylim(0, 115)
    ax1.tick_params(colors="#64748B")
    ax1.legend(facecolor="#FFFFFF", edgecolor="#E2E8F0", labelcolor="#172033", fontsize=9)

    for bar in list(rects1) + list(rects2):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            color="#172033",
            fontsize=8.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # 2. Mean Path Length & Steps
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#FFFFFF")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax2.spines.values():
        sp.set_color("#E2E8F0")

    path_lens = [c_sum.get("mean_path_length", 0.0), q_sum.get("mean_path_length", 0.0)]
    steps = [c_sum.get("mean_steps", 0.0), q_sum.get("mean_steps", 0.0)]

    rects3 = ax2.bar(
        x - width / 2, path_lens, width, label="Mean Path Length", color="#EA580C", alpha=0.9
    )
    rects4 = ax2.bar(
        x + width / 2, steps, width, label="Mean Steps", color="#64748B", alpha=0.9
    )

    ax2.set_ylabel("Value", color="#64748B", fontsize=10)
    ax2.set_title("Trajectory Distance & Execution Steps", color="#172033", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="#172033", fontsize=10, fontweight="600")
    ax2.tick_params(colors="#64748B")
    ax2.legend(facecolor="#FFFFFF", edgecolor="#E2E8F0", labelcolor="#172033", fontsize=9)

    for bar in list(rects3) + list(rects4):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            color="#172033",
            fontsize=8.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # 3. Collision Count & Safety Interventions
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#FFFFFF")
    ax3.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax3.spines.values():
        sp.set_color("#E2E8F0")

    cols = [c_sum.get("mean_collisions", 0.0), q_sum.get("mean_collisions", 0.0)]
    interventions = [
        c_sum.get("mean_safety_violations", 0.0),
        q_sum.get("mean_safety_violations", 0.0),
    ]

    rects5 = ax3.bar(
        x - width / 2, cols, width, label="Mean Collisions", color="#DC2626", alpha=0.9
    )
    rects6 = ax3.bar(
        x + width / 2,
        interventions,
        width,
        label="Safety Interventions",
        color="#F59E0B",
        alpha=0.9,
    )

    ax3.set_ylabel("Count", color="#64748B", fontsize=10)
    ax3.set_title("Safety Metrics & Obstacle Incidents", color="#172033", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color="#172033", fontsize=10, fontweight="600")
    ax3.tick_params(colors="#64748B")
    ax3.legend(facecolor="#FFFFFF", edgecolor="#E2E8F0", labelcolor="#172033", fontsize=9)

    for bar in list(rects5) + list(rects6):
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.1,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            color="#172033",
            fontsize=8.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # 4. Per-Scenario Path Length Comparison
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#FFFFFF")
    ax4.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax4.spines.values():
        sp.set_color("#E2E8F0")

    if scenarios:
        scn_names = [s.get("scenario_id", f"S{i}") for i, s in enumerate(scenarios)]
        c_paths = [s["classical"]["metrics"]["path_length"] for s in scenarios]
        q_paths = [s["quantum"]["metrics"]["path_length"] for s in scenarios]

        xs = np.arange(len(scn_names))
        w = 0.35

        ax4.bar(xs - w / 2, c_paths, w, label="Classical APF", color="#EA580C", alpha=0.9)
        ax4.bar(xs + w / 2, q_paths, w, label="Quantum-Assisted", color="#B8860B", alpha=0.9)

        ax4.set_ylabel("Path Length (units)", color="#64748B", fontsize=10)
        ax4.set_title("Path Length by Scenario", color="#172033", fontsize=11, fontweight="bold")
        ax4.set_xticks(xs)
        ax4.set_xticklabels(scn_names, color="#64748B", fontsize=8, rotation=20)
        ax4.tick_params(colors="#64748B")
        ax4.legend(facecolor="#FFFFFF", edgecolor="#E2E8F0", labelcolor="#172033", fontsize=9)
    else:
        ax4.text(
            0.5,
            0.5,
            "No scenario data",
            color="#64748B",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    fig.suptitle(
        "Q-Nav: Classical Baseline vs Quantum-Assisted Navigation Benchmarking",
        color="#172033",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_single_metric_comparison(
    classical_val: float,
    quantum_val: float,
    metric_name: str,
    ylabel: str,
    save_path: Optional[Union[str, Path]] = None,
    is_percentage: bool = False,
) -> plt.Figure:
    """Generate a clean, focused comparative bar chart for a single metric."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax.spines.values():
        sp.set_color("#E2E8F0")

    labels = ["Classical APF", "Quantum-Assisted"]
    vals = [classical_val, quantum_val]
    colors = ["#EA580C", "#B8860B"]

    bars = ax.bar(labels, vals, color=colors, width=0.45, alpha=0.9, edgecolor="#E2E8F0")
    ax.set_ylabel(ylabel, color="#64748B", fontsize=11)
    ax.set_title(f"Model Comparison: {metric_name}", color="#172033", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#64748B", labelsize=10)

    for bar, val in zip(bars, vals):
        height = bar.get_height()
        fmt = f"{val:.1%}" if is_percentage else (f"{val:.2f}" if isinstance(val, float) else f"{val}")
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.02 * max(1.0, max(vals))),
            fmt,
            ha="center",
            va="bottom",
            color="#172033",
            fontsize=10,
            fontweight="bold",
        )

    top_lim = max(vals) * 1.18 if max(vals) > 0 else 1.0
    if is_percentage:
        top_lim = min(1.15, max(1.0, top_lim))
    ax.set_ylim(0, top_lim)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_all_benchmark_figures(
    comparison_data: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Generate and save all required comparison figures in output_dir.

    Outputs:
    - benchmark_summary.png
    - success_rate_comparison.png
    - path_length_comparison.png
    - collision_comparison.png
    - efficiency_comparison.png
    - safety_intervention_comparison.png
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = comparison_data.get("summary", {})
    c_sum = summary.get("classical", {})
    q_sum = summary.get("quantum", {})

    saved = {}

    # 1. Benchmark multi-panel summary
    p1 = out / "benchmark_summary.png"
    plot_metric_bar_comparison(comparison_data, save_path=p1)
    saved["benchmark_summary"] = p1

    # 2. Success rate
    p2 = out / "success_rate_comparison.png"
    plot_single_metric_comparison(
        c_sum.get("success_rate", 0.0),
        q_sum.get("success_rate", 0.0),
        metric_name="Mission Success Rate",
        ylabel="Success Rate",
        save_path=p2,
        is_percentage=True,
    )
    saved["success_rate"] = p2

    # 3. Path length
    p3 = out / "path_length_comparison.png"
    plot_single_metric_comparison(
        c_sum.get("mean_path_length", 0.0),
        q_sum.get("mean_path_length", 0.0),
        metric_name="Mean Path Length",
        ylabel="Distance (units)",
        save_path=p3,
    )
    saved["path_length"] = p3

    # 4. Collisions
    p4 = out / "collision_comparison.png"
    plot_single_metric_comparison(
        c_sum.get("mean_collisions", 0.0),
        q_sum.get("mean_collisions", 0.0),
        metric_name="Mean Collisions",
        ylabel="Collision Count",
        save_path=p4,
    )
    saved["collision"] = p4

    # 5. Efficiency
    p5 = out / "efficiency_comparison.png"
    plot_single_metric_comparison(
        c_sum.get("mean_navigation_efficiency", c_sum.get("mean_efficiency", 0.0)),
        q_sum.get("mean_navigation_efficiency", q_sum.get("mean_efficiency", 0.0)),
        metric_name="Navigation Efficiency",
        ylabel="Efficiency Ratio",
        save_path=p5,
        is_percentage=True,
    )
    saved["efficiency"] = p5

    # 6. Safety interventions
    p6 = out / "safety_intervention_comparison.png"
    plot_single_metric_comparison(
        c_sum.get("mean_safety_interventions", c_sum.get("mean_safety_violations", 0.0)),
        q_sum.get("mean_safety_interventions", q_sum.get("mean_safety_violations", 0.0)),
        metric_name="Safety Interventions",
        ylabel="Intervention Count",
        save_path=p6,
    )
    saved["safety_interventions"] = p6

    return saved


