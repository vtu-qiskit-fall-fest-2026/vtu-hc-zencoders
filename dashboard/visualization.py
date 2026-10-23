"""Trajectory and Environment Visualization Module for Q-Nav.

Provides 2D rendering of obstacle maps, target waypoints, and comparative flight trajectories
using Matplotlib and Plotly.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts and headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_map_base(
    map_data: Dict[str, Any],
    start: Optional[Tuple[float, float]] = None,
    target: Optional[Tuple[float, float]] = None,
    target_radius: float = 2.0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Render the base 2D airspace map with static obstacles and mission waypoints.

    Args:
        map_data: Dictionary defining map dimensions and obstacles.
        start: (x, y) coordinates of drone start point.
        target: (x, y) coordinates of destination target.
        target_radius: Detection threshold circle around target.
        ax: Existing Matplotlib Axes to draw on, or None to create a new one.
        figsize: Figure dimensions if creating a new figure.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
    else:
        fig = ax.get_figure()

    dims = map_data.get("dimensions", {"width": 50.0, "height": 50.0})
    width = float(dims.get("width", 50.0))
    height = float(dims.get("height", 50.0))

    # Background styling — Clean Light Workspace
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-1, width + 1)
    ax.set_ylim(-1, height + 1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax.spines.values():
        sp.set_color("#E2E8F0")

    # Map boundaries
    boundary_rect = patches.Rectangle(
        (0, 0),
        width,
        height,
        linewidth=2,
        edgecolor="#94A3B8",
        facecolor="none",
        linestyle="-",
        label="Flight Perimeter",
    )
    ax.add_patch(boundary_rect)

    # Render obstacles
    obstacles = map_data.get("obstacles", [])
    obs_labeled = False
    for obs in obstacles:
        shape_type = obs.get("type", "circle").lower()
        obs_label = "Obstacle" if not obs_labeled else None
        if shape_type == "circle":
            cx, cy = float(obs["x"]), float(obs["y"])
            r = float(obs["radius"])
            circle = patches.Circle(
                (cx, cy),
                r,
                facecolor="#475569",
                alpha=0.6,
                edgecolor="#334155",
                linewidth=1.5,
                label=obs_label,
            )
            ax.add_patch(circle)
            obs_labeled = True
        elif shape_type in ("rectangle", "box"):
            x_min = float(obs["x_min"])
            y_min = float(obs["y_min"])
            bw = float(obs["x_max"]) - x_min
            bh = float(obs["y_max"]) - y_min
            box = patches.Rectangle(
                (x_min, y_min),
                bw,
                bh,
                facecolor="#475569",
                alpha=0.6,
                edgecolor="#334155",
                linewidth=1.5,
                hatch="//",
                label=obs_label,
            )
            ax.add_patch(box)
            obs_labeled = True

    # Render Start Position (UAV Origin)
    if start is not None:
        ax.scatter(
            [start[0]],
            [start[1]],
            c="#EA580C",
            s=180,
            marker="o",
            edgecolors="#FFFFFF",
            linewidth=2,
            zorder=10,
            label="Start Waypoint",
        )
        ax.annotate(
            "START",
            (start[0], start[1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#EA580C",
            fontweight="bold",
            fontsize=9,
        )

    # Render Target Position & Acceptance Zone
    if target is not None:
        ax.scatter(
            [target[0]],
            [target[1]],
            c="#16A34A",
            s=250,
            marker="*",
            edgecolors="#FFFFFF",
            linewidth=1.5,
            zorder=10,
            label="Target Goal",
        )
        target_zone = patches.Circle(
            target,
            target_radius,
            facecolor="#16A34A",
            alpha=0.15,
            edgecolor="#16A34A",
            linestyle="--",
            linewidth=1.2,
        )
        ax.add_patch(target_zone)
        ax.annotate(
            "TARGET",
            (target[0], target[1]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            color="#16A34A",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xlabel("X Distance (units)", color="#64748B", fontsize=10)
    ax.set_ylabel("Y Distance (units)", color="#64748B", fontsize=10)
    ax.tick_params(colors="#64748B")

    return fig, ax


def plot_trajectory_comparison(
    scenario: Dict[str, Any],
    map_data: Dict[str, Any],
    classical_trajectory: List[Tuple[float, float]],
    quantum_trajectory: List[Tuple[float, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side or overlaid classical and quantum UAV flight paths.

    Args:
        scenario: Scenario dict with start and target positions.
        map_data: Map dict with obstacles.
        classical_trajectory: List of (x, y) coordinates for classical path.
        quantum_trajectory: List of (x, y) coordinates for quantum path.
        save_path: Optional path to save the output figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    start = tuple(scenario.get("start", (5.0, 5.0)))
    target = tuple(scenario.get("target", (45.0, 45.0)))
    scenario_id = scenario.get("id", "Scenario")

    fig, ax = plot_map_base(map_data, start=start, target=target)

    # Plot Classical Trajectory
    if classical_trajectory and len(classical_trajectory) > 1:
        c_pts = np.array(classical_trajectory)
        ax.plot(
            c_pts[:, 0],
            c_pts[:, 1],
            color="#EA580C",  # Primary Orange: Classical APF
            linewidth=2.5,
            linestyle="-",
            alpha=0.9,
            label="Classical APF Path",
            zorder=6,
        )
        ax.scatter(
            c_pts[::3, 0],
            c_pts[::3, 1],
            color="#EA580C",
            s=20,
            marker="o",
            alpha=0.7,
            zorder=7,
        )

    # Plot Quantum Trajectory
    if quantum_trajectory and len(quantum_trajectory) > 1:
        q_pts = np.array(quantum_trajectory)
        ax.plot(
            q_pts[:, 0],
            q_pts[:, 1],
            color="#B8860B",  # Quantum Gold: Quantum-Assisted
            linewidth=2.5,
            linestyle="-.",
            alpha=0.9,
            label="Quantum-Assisted Path",
            zorder=6,
        )
        ax.scatter(
            q_pts[::3, 0],
            q_pts[::3, 1],
            color="#B8860B",
            s=24,
            marker="^",
            alpha=0.7,
            zorder=7,
        )

    plot_title = title or f"Q-Nav Flight Trajectory: {scenario_id}"
    ax.set_title(plot_title, color="#172033", fontsize=13, fontweight="bold", pad=12)
    ax.legend(
        loc="upper left",
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        labelcolor="#172033",
        fontsize=9,
    )

    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_dynamic_trajectory_comparison(
    scenario: Dict[str, Any],
    classical_trajectory: Optional[List[Tuple[float, float]]] = None,
    quantum_trajectory: Optional[List[Tuple[float, float]]] = None,
    hybrid_trajectory: Optional[List[Tuple[float, float]]] = None,
    dynamic_telemetry: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 9),
) -> plt.Figure:
    """Plot multi-model trajectories in dynamic obstacle environments.

    Visualizes static obstacles, moving obstacles with velocity vectors and orbit paths,
    and trajectories for Classical APF, Pure Quantum, and Hybrid models.

    Args:
        scenario: Scenario dict with bounds, start, target, and obstacles.
        classical_trajectory: Coordinates list for classical path.
        quantum_trajectory: Coordinates list for pure quantum path.
        hybrid_trajectory: Optional coordinates list for hybrid path.
        dynamic_telemetry: Optional step telemetry rows for dynamic obstacle positions.
        save_path: Optional file path to save the rendered figure.
        title: Optional plot title.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    bounds = scenario.get("bounds", [50.0, 50.0])
    width, height = float(bounds[0]), float(bounds[1])

    # Background styling — Clean Light Workspace
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-1, width + 1)
    ax.set_ylim(-1, height + 1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for sp in ax.spines.values():
        sp.set_color("#E2E8F0")

    # Flight boundary perimeter
    boundary_rect = patches.Rectangle(
        (0, 0), width, height,
        linewidth=2, edgecolor="#94A3B8", facecolor="none",
        linestyle="-", label="Flight Perimeter",
    )
    ax.add_patch(boundary_rect)

    # Parse obstacles from scenario
    raw_obstacles = scenario.get("obstacles", [])
    static_labeled = False
    moving_labeled = False

    for obs in raw_obstacles:
        obs_type = obs.get("type", "static")
        pos = obs.get("position", [0.0, 0.0])
        r = float(obs.get("radius", 2.0))

        if obs_type == "static":
            label = "Static Obstacle" if not static_labeled else None
            circle = patches.Circle(
                (pos[0], pos[1]), r,
                facecolor="#475569", alpha=0.6,
                edgecolor="#334155", linewidth=1.5,
                label=label,
            )
            ax.add_patch(circle)
            static_labeled = True

        elif obs_type == "moving":
            pattern = obs.get("pattern", "linear")
            label = f"Moving Obstacle ({pattern})" if not moving_labeled else None
            circle = patches.Circle(
                (pos[0], pos[1]), r,
                facecolor="#F59E0B", alpha=0.5,
                edgecolor="#D97706", linewidth=1.8,
                linestyle="--",
                label=label,
            )
            ax.add_patch(circle)
            moving_labeled = True

            # If circular pattern, draw orbit trajectory
            if pattern == "circular" and "center" in obs:
                c_pt = obs["center"]
                orb_r = float(obs.get("orbit_radius", 5.0))
                orbit_circle = patches.Circle(
                    (c_pt[0], c_pt[1]), orb_r,
                    facecolor="none", edgecolor="#F59E0B",
                    linestyle=":", linewidth=1.2, alpha=0.6,
                    label="Orbit Path" if not moving_labeled else None,
                )
                ax.add_patch(orbit_circle)

            # Draw initial velocity vector
            vel = obs.get("velocity", [0.0, 0.0])
            if pattern == "circular" and "angular_speed" in obs:
                # Tangential velocity at phase
                phase = obs.get("phase", 0.0)
                omega = obs.get("angular_speed", 0.2)
                orb_r = obs.get("orbit_radius", 5.0)
                vel = [-orb_r * omega * np.sin(phase), orb_r * omega * np.cos(phase)]

            vx, vy = float(vel[0]), float(vel[1])
            v_norm = np.hypot(vx, vy)
            if v_norm > 1e-4:
                arrow_scale = 3.0
                ax.annotate(
                    "",
                    xy=(pos[0] + vx * arrow_scale, pos[1] + vy * arrow_scale),
                    xytext=(pos[0], pos[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#DC2626",
                        lw=2.0,
                        mutation_scale=14,
                    ),
                    zorder=8,
                )

    # Draw moving obstacle history if dynamic telemetry rows are supplied
    if dynamic_telemetry:
        obs_histories: Dict[int, List[Tuple[float, float]]] = {}
        for row in dynamic_telemetry:
            oid = row.get("obstacle_id", 0)
            ox, oy = row.get("obstacle_x"), row.get("obstacle_y")
            if ox is not None and oy is not None:
                obs_histories.setdefault(oid, []).append((float(ox), float(oy)))

        for oid, path in obs_histories.items():
            if len(path) > 1:
                p_arr = np.array(path)
                ax.plot(
                    p_arr[:, 0], p_arr[:, 1],
                    color="#F59E0B", linestyle=":", linewidth=1.5,
                    alpha=0.65, label="Obstacle Swept Path" if oid == 0 else None,
                    zorder=4,
                )

    # Render Start & Target
    start = tuple(scenario.get("start", (5.0, 5.0)))
    target = tuple(scenario.get("target", (45.0, 45.0)))

    ax.scatter(
        [start[0]], [start[1]],
        c="#EA580C", s=180, marker="o",
        edgecolors="#FFFFFF", linewidth=2,
        zorder=10, label="Start Waypoint",
    )
    ax.annotate(
        "START", (start[0], start[1]),
        textcoords="offset points", xytext=(0, 10),
        ha="center", color="#EA580C", fontweight="bold", fontsize=9,
    )

    ax.scatter(
        [target[0]], [target[1]],
        c="#16A34A", s=250, marker="*",
        edgecolors="#FFFFFF", linewidth=1.5,
        zorder=10, label="Target Goal",
    )
    target_zone = patches.Circle(
        target, 2.0,
        facecolor="#16A34A", alpha=0.15,
        edgecolor="#16A34A", linestyle="--", linewidth=1.2,
    )
    ax.add_patch(target_zone)
    ax.annotate(
        "TARGET", (target[0], target[1]),
        textcoords="offset points", xytext=(0, 12),
        ha="center", color="#16A34A", fontweight="bold", fontsize=9,
    )

    # Plot Classical Trajectory
    if classical_trajectory and len(classical_trajectory) > 1:
        c_pts = np.array(classical_trajectory)
        ax.plot(
            c_pts[:, 0], c_pts[:, 1],
            color="#EA580C", linewidth=2.5, linestyle="-",
            alpha=0.9, label="Classical APF Path", zorder=6,
        )
        ax.scatter(
            c_pts[::3, 0], c_pts[::3, 1],
            color="#EA580C", s=20, marker="o", alpha=0.7, zorder=7,
        )

    # Plot Pure Quantum Trajectory
    if quantum_trajectory and len(quantum_trajectory) > 1:
        q_pts = np.array(quantum_trajectory)
        ax.plot(
            q_pts[:, 0], q_pts[:, 1],
            color="#B8860B", linewidth=2.5, linestyle="-.",
            alpha=0.9, label="Pure Quantum Path", zorder=6,
        )
        ax.scatter(
            q_pts[::3, 0], q_pts[::3, 1],
            color="#B8860B", s=24, marker="^", alpha=0.7, zorder=7,
        )

    # Plot Hybrid Trajectory
    if hybrid_trajectory and len(hybrid_trajectory) > 1:
        h_pts = np.array(hybrid_trajectory)
        ax.plot(
            h_pts[:, 0], h_pts[:, 1],
            color="#172033", linewidth=2.8, linestyle="--",
            alpha=0.95, label="Hybrid Quantum-Assisted Path", zorder=7,
        )
        ax.scatter(
            h_pts[::3, 0], h_pts[::3, 1],
            color="#172033", s=24, marker="D", alpha=0.8, zorder=8,
        )

    plot_title = title or f"Dynamic UAV Navigation: {scenario.get('name', 'Scenario')}"
    ax.set_title(plot_title, color="#172033", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Distance (units)", color="#64748B", fontsize=10)
    ax.set_ylabel("Y Distance (units)", color="#64748B", fontsize=10)
    ax.tick_params(colors="#64748B")
    ax.legend(
        loc="upper left",
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        labelcolor="#172033",
        fontsize=8.5,
    )

    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    return fig


def create_interactive_trajectory_plotly(
    scenario: Dict[str, Any],
    map_data: Optional[Dict[str, Any]] = None,
    classical_trajectory: Optional[List[Any]] = None,
    quantum_trajectory: Optional[List[Any]] = None,
    hybrid_trajectory: Optional[List[Any]] = None,
    step_highlight: Optional[int] = None,
    title: Optional[str] = None,
):
    """Render an interactive 2D airspace visualization using Plotly."""
    import plotly.graph_objects as go

    bounds = scenario.get("bounds", [50.0, 50.0])
    width, height = float(bounds[0]), float(bounds[1])

    start = tuple(scenario.get("start", (5.0, 5.0)))
    target = tuple(scenario.get("target", (45.0, 45.0)))
    target_radius = float(scenario.get("target_radius", 2.0))

    fig = go.Figure()

    # 1. Target acceptance zone
    t_theta = np.linspace(0, 2 * np.pi, 50)
    tx_circle = target[0] + target_radius * np.cos(t_theta)
    ty_circle = target[1] + target_radius * np.sin(t_theta)
    fig.add_trace(go.Scatter(
        x=tx_circle,
        y=ty_circle,
        mode="lines",
        line=dict(color="#16A34A", dash="dash", width=1.5),
        fill="toself",
        fillcolor="rgba(22, 163, 74, 0.12)",
        name="Target Acceptance Zone",
        hoverinfo="skip",
    ))

    # 2. Static Obstacles
    raw_obstacles = scenario.get("obstacles", [])
    if not raw_obstacles and map_data:
        raw_obstacles = map_data.get("obstacles", [])

    for idx, obs in enumerate(raw_obstacles):
        obs_type = obs.get("type", "static").lower()
        if obs_type in ("static", "circle"):
            pos = obs.get("position", [obs.get("x", 0.0), obs.get("y", 0.0)])
            r = float(obs.get("radius", 2.0))
            cx, cy = float(pos[0]), float(pos[1])
            ox = cx + r * np.cos(t_theta)
            oy = cy + r * np.sin(t_theta)
            fig.add_trace(go.Scatter(
                x=ox,
                y=oy,
                mode="lines",
                line=dict(color="#475569", width=1.5),
                fill="toself",
                fillcolor="rgba(71, 85, 105, 0.35)",
                name="Static Obstacle" if idx == 0 else None,
                showlegend=bool(idx == 0),
                hovertemplate=f"<b>Static Obstacle</b><br>Pos: ({cx:.1f}, {cy:.1f})<br>Radius: {r:.1f}<extra></extra>",
            ))
        elif obs_type in ("rectangle", "box"):
            x_min = float(obs["x_min"])
            y_min = float(obs["y_min"])
            x_max = float(obs["x_max"])
            y_max = float(obs["y_max"])
            fig.add_trace(go.Scatter(
                x=[x_min, x_max, x_max, x_min, x_min],
                y=[y_min, y_min, y_max, y_max, y_min],
                mode="lines",
                line=dict(color="#475569", width=1.5),
                fill="toself",
                fillcolor="rgba(71, 85, 105, 0.35)",
                name="Static Barrier",
                showlegend=False,
                hovertemplate=f"<b>Static Barrier</b><br>Bounds: [{x_min}, {x_max}] x [{y_min}, {y_max}]<extra></extra>",
            ))
        elif obs_type == "moving":
            pos = obs.get("position", [0.0, 0.0])
            r = float(obs.get("radius", 2.0))
            pattern = obs.get("pattern", "linear")
            cx, cy = float(pos[0]), float(pos[1])
            ox = cx + r * np.cos(t_theta)
            oy = cy + r * np.sin(t_theta)
            fig.add_trace(go.Scatter(
                x=ox,
                y=oy,
                mode="lines",
                line=dict(color="#F59E0B", width=2, dash="dash"),
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.25)",
                name=f"Moving Hazard ({pattern})",
                hovertemplate=f"<b>Moving Hazard ({pattern})</b><br>Pos: ({cx:.1f}, {cy:.1f})<extra></extra>",
            ))
            # Orbit path
            if pattern == "circular" and "center" in obs:
                c_pt = obs["center"]
                orb_r = float(obs.get("orbit_radius", 5.0))
                orb_x = c_pt[0] + orb_r * np.cos(t_theta)
                orb_y = c_pt[1] + orb_r * np.sin(t_theta)
                fig.add_trace(go.Scatter(
                    x=orb_x,
                    y=orb_y,
                    mode="lines",
                    line=dict(color="#F59E0B", width=1, dash="dot"),
                    name="Hazard Orbit",
                    hoverinfo="skip",
                ))

    # 3. Flight Trajectories
    def _add_traj(pts, name, color, dash="solid", width=3):
        if pts and len(pts) > 1:
            arr = np.array(pts)
            fig.add_trace(go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                mode="lines+markers",
                line=dict(color=color, width=width, dash=dash),
                marker=dict(size=4, color=color),
                name=name,
                hovertemplate=f"<b>{name}</b><br>Step %{{pointNumber}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
            ))

    _add_traj(classical_trajectory, "Classical APF Path", "#EA580C", dash="solid", width=2.5)
    _add_traj(quantum_trajectory, "Pure Quantum Path", "#B8860B", dash="dashdot", width=2.5)
    _add_traj(hybrid_trajectory, "Hybrid Quantum-Assisted Path", "#172033", dash="dash", width=3)

    # 4. Step Highlight (Active UAV Scrubbing)
    if step_highlight is not None:
        active_pts = hybrid_trajectory or quantum_trajectory or classical_trajectory
        if active_pts and len(active_pts) > step_highlight:
            uav_pos = active_pts[step_highlight]
            fig.add_trace(go.Scatter(
                x=[uav_pos[0]],
                y=[uav_pos[1]],
                mode="markers",
                marker=dict(size=14, color="#EA580C", symbol="cross", line=dict(color="#FFFFFF", width=2)),
                name=f"UAV at Step {step_highlight}",
                hovertemplate=f"<b>Active UAV Position</b><br>Step: {step_highlight}<br>Pos: ({uav_pos[0]:.2f}, {uav_pos[1]:.2f})<extra></extra>",
            ))

    # 5. Start and Target markers
    fig.add_trace(go.Scatter(
        x=[start[0]],
        y=[start[1]],
        mode="markers+text",
        marker=dict(size=14, color="#EA580C", symbol="circle", line=dict(color="#FFFFFF", width=2)),
        text=["START"],
        textposition="top center",
        textfont=dict(color="#EA580C", size=11, family="Inter"),
        name="Start Waypoint",
        hovertemplate=f"<b>Start</b><br>({start[0]:.1f}, {start[1]:.1f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[target[0]],
        y=[target[1]],
        mode="markers+text",
        marker=dict(size=18, color="#16A34A", symbol="star", line=dict(color="#FFFFFF", width=1.5)),
        text=["TARGET"],
        textposition="top center",
        textfont=dict(color="#16A34A", size=11, family="Inter"),
        name="Target Goal",
        hovertemplate=f"<b>Target Goal</b><br>({target[0]:.1f}, {target[1]:.1f})<extra></extra>",
    ))

    # Layout styling — Clean Professional Light Workspace
    chart_title = title or f"Airspace Flight Trajectory: {scenario.get('name', scenario.get('id', 'Mission'))}"
    fig.update_layout(
        title=dict(text=chart_title, font=dict(color="#172033", size=14, family="Outfit")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            range=[-1, width + 1],
            zeroline=False,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#64748B"),
            title=dict(text="X Distance (units)", font=dict(color="#64748B")),
            constrain="domain",
        ),
        yaxis=dict(
            range=[-1, height + 1],
            zeroline=False,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#64748B"),
            title=dict(text="Y Distance (units)", font=dict(color="#64748B")),
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#172033", size=11),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        height=580,
        margin=dict(l=20, r=20, t=60, b=30),
    )

    return fig


def render_quantum_circuit_wire_svg(n_qubits: int = 6, n_layers: int = 3) -> str:
    """Generate clean SVG code representing the parameterized quantum circuit wires."""
    height = 50 + n_qubits * 50
    width = 720
    wires_svg = []

    # Wires & labels
    for q in range(n_qubits):
        y = 50 + q * 50
        wires_svg.append(f'<text x="20" y="{y + 5}" fill="#172033" font-family="JetBrains Mono, monospace" font-size="13" font-weight="600">q{q}</text>')
        wires_svg.append(f'<text x="50" y="{y + 5}" fill="#64748B" font-family="Inter, sans-serif" font-size="12">|0⟩</text>')
        wires_svg.append(f'<line x1="75" y1="{y}" x2="680" y2="{y}" stroke="#CBD5E1" stroke-width="1.8"/>')

        # Feature Encoding box (RY, RZ) — Orange Accent
        wires_svg.append(f'<rect x="100" y="{y - 16}" width="70" height="32" rx="6" fill="#FFF7ED" stroke="#EA580C" stroke-width="1.2"/>')
        wires_svg.append(f'<text x="135" y="{y + 4}" fill="#C2410C" font-family="Inter, sans-serif" font-size="11" font-weight="600" text-anchor="middle">RY/RZ(x)</text>')

    # Variational layers — Quantum Gold Accent
    layer_start_x = 200
    layer_width = 130
    for l in range(min(n_layers, 3)):
        lx = layer_start_x + l * layer_width
        wires_svg.append(f'<text x="{lx + 55}" y="24" fill="#B8860B" font-family="Inter, sans-serif" font-size="11" font-weight="600" text-anchor="middle">Layer {l+1}</text>')
        wires_svg.append(f'<line x1="{lx}" y1="35" x2="{lx}" y2="{height - 20}" stroke="#E2E8F0" stroke-dasharray="3,3" stroke-width="1"/>')

        for q in range(n_qubits):
            y = 50 + q * 50
            # Parameterized Rotation Gate — Gold Accent
            wires_svg.append(f'<rect x="{lx + 15}" y="{y - 16}" width="42" height="32" rx="6" fill="#FEF3C7" stroke="#B8860B" stroke-width="1.2"/>')
            wires_svg.append(f'<text x="{lx + 36}" y="{y + 4}" fill="#92400E" font-family="JetBrains Mono, monospace" font-size="10" font-weight="600" text-anchor="middle">R(θ)</text>')

            # CNOT ladder
            next_q = (q + 1) % n_qubits
            next_y = 50 + next_q * 50
            if q < n_qubits - 1 or l == 0:
                cx = lx + 85
                wires_svg.append(f'<line x1="{cx}" y1="{y}" x2="{cx}" y2="{next_y}" stroke="#B8860B" stroke-width="1.5"/>')
                wires_svg.append(f'<circle cx="{cx}" cy="{y}" r="4" fill="#B8860B"/>')
                wires_svg.append(f'<circle cx="{cx}" cy="{next_y}" r="6" fill="none" stroke="#B8860B" stroke-width="1.5"/>')
                wires_svg.append(f'<line x1="{cx - 5}" y1="{next_y}" x2="{cx + 5}" y2="{next_y}" stroke="#B8860B" stroke-width="1.5"/>')
                wires_svg.append(f'<line x1="{cx}" y1="{next_y - 5}" x2="{cx}" y2="{next_y + 5}" stroke="#B8860B" stroke-width="1.5"/>')

    # Measurement meters — Navy Accent
    mx = layer_start_x + min(n_layers, 3) * layer_width + 30
    wires_svg.append(f'<text x="{mx + 20}" y="24" fill="#172033" font-family="Inter, sans-serif" font-size="11" font-weight="600" text-anchor="middle">Measure</text>')
    for q in range(n_qubits):
        y = 50 + q * 50
        wires_svg.append(f'<rect x="{mx}" y="{y - 16}" width="38" height="32" rx="6" fill="#F8FAFC" stroke="#172033" stroke-width="1.5"/>')
        wires_svg.append(f'<text x="{mx + 19}" y="{y + 4}" fill="#172033" font-family="JetBrains Mono, monospace" font-size="11" font-weight="bold" text-anchor="middle">⟨Z⟩</text>')

    svg_content = f"""
    <svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background: #FFFFFF; border-radius: 10px; border: 1px solid #E2E8F0; padding: 10px;">
        {"".join(wires_svg)}
    </svg>
    """
    return svg_content


