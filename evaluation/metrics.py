"""Navigation Performance Metrics Module for Q-Nav.

Computes standardized quantitative metrics for evaluating single episodes
and multi-scenario comparative benchmarks, including statistical aggregations
(mean, std, median, min, max, 95% CI).
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def compute_episode_metrics(
    episode_result: Dict[str, Any],
    start_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    duration_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics for a completed navigation episode.

    Args:
        episode_result: Result dictionary returned by NavigationController.run_episode().
        start_pos: (x, y) start coordinates.
        target_pos: (x, y) target coordinates.
        duration_seconds: Real elapsed wall-clock execution time.

    Returns:
        Dictionary of computed metric values.
    """
    success = bool(episode_result.get("success", False))
    steps = int(episode_result.get("total_steps", 0))
    path_len = float(episode_result.get("path_length", 0.0))
    collisions = int(episode_result.get("collision_count", 0))
    final_dist = float(episode_result.get("final_distance", 0.0))
    reward = float(episode_result.get("total_reward", 0.0))
    safety_interventions = int(episode_result.get("safety_interventions", 0))
    unsafe_attempts = int(episode_result.get("unsafe_action_attempts", 0))
    boundary_violations = int(episode_result.get("boundary_violations", 0))
    min_obs_clearance = float(episode_result.get("min_obstacle_clearance", 15.0))
    min_bound_clearance = float(episode_result.get("min_boundary_clearance", 25.0))
    action_switches = int(episode_result.get("action_switch_count", 0))
    oscillations = int(episode_result.get("oscillation_count", 0))
    mean_dec_time = float(episode_result.get("mean_decision_time", 0.0))
    qc_time = float(episode_result.get("quantum_circuit_execution_time", 0.0))

    # Straight line theoretical optimum
    p_start = np.array(start_pos, dtype=float)
    p_target = np.array(target_pos, dtype=float)
    straight_line_dist = float(np.linalg.norm(p_target - p_start))

    # Navigation efficiency: theoretical straight-line distance divided by actual path length
    if path_len > 1e-6 and success:
        efficiency = float(min(1.0, straight_line_dist / path_len))
    elif success:
        efficiency = 1.0
    else:
        # Penalize incomplete missions proportionally
        efficiency = float(max(0.0, 1.0 - (final_dist / max(1.0, straight_line_dist)))) * 0.5

    collision_rate = float(collisions / max(1, steps))

    # Progress ratio: (initial_distance - final_distance) / initial_distance
    initial_dist = max(1e-6, straight_line_dist)
    progress_ratio = float(np.clip((initial_dist - final_dist) / initial_dist, -1.0, 1.0))

    # Path smoothness: sum of absolute angle deviations between successive steps
    coords = episode_result.get("trajectory", [])
    smoothness = 0.0
    energy_proxy = 0.0
    if len(coords) >= 3:
        pts = np.array(coords)
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:] - pts[1:-1]
        headings = np.arctan2(v1[:, 1], v1[:, 0])
        next_headings = np.arctan2(v2[:, 1], v2[:, 0])
        diffs = np.abs((next_headings - headings + np.pi) % (2.0 * np.pi) - np.pi)
        smoothness = float(np.sum(diffs))
        # Energy proxy: translation distance + angular change penalty
        energy_proxy = float(path_len + 0.3 * smoothness)
    else:
        energy_proxy = float(path_len)

    # Dynamic obstacle metrics
    pred_collisions = int(episode_result.get("predicted_collision_count", 0))
    min_m_clearance = float(episode_result.get("min_moving_obstacle_clearance", 50.0))
    mean_m_clearance = float(episode_result.get("mean_moving_obstacle_clearance", 50.0))

    return {
        "model_type": episode_result.get("model_type", "unknown"),
        "success": success,
        "success_rate": 1.0 if success else 0.0,
        "path_length": round(path_len, 3),
        "steps": steps,
        "final_distance": round(final_dist, 3),
        "final_distance_to_target": round(final_dist, 3),
        "collision_count": collisions,
        "collision_rate": round(collision_rate, 4),
        "safety_interventions": safety_interventions,
        "safety_violations": safety_interventions,  # Backward compatibility alias
        "unsafe_action_attempts": unsafe_attempts,
        "boundary_violations": boundary_violations,
        "min_obstacle_clearance": round(min_obs_clearance, 3),
        "min_boundary_clearance": round(min_bound_clearance, 3),
        "predicted_collision_count": pred_collisions,
        "min_moving_obstacle_clearance": round(min_m_clearance, 3),
        "mean_moving_obstacle_clearance": round(mean_m_clearance, 3),
        "straight_line_distance": round(straight_line_dist, 3),
        "navigation_efficiency": round(efficiency, 4),
        "progress_ratio": round(progress_ratio, 4),
        "heading_smoothness": round(smoothness, 3),
        "path_smoothness": round(smoothness, 3),  # Backward compatibility alias
        "action_switch_count": action_switches,
        "oscillation_count": oscillations,
        "mean_decision_time": round(mean_dec_time, 6),
        "quantum_circuit_execution_time": round(qc_time, 6),
        "total_episode_time": round(duration_seconds, 4),
        "duration_seconds": round(duration_seconds, 4),  # Backward compatibility alias
        "reward": round(reward, 2),
        "total_reward": round(reward, 2),
        "energy_proxy": round(energy_proxy, 3),
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, median, min, max, and 95% confidence interval for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "ci_95": 0.0}
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    # 95% confidence interval: 1.96 * std / sqrt(n)
    ci_95 = float(1.96 * std_val / np.sqrt(n)) if n > 1 else 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "median": round(median_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "ci_95": round(ci_95, 4),
    }


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of episode metrics into summary statistics and distributions.

    Args:
        metrics_list: List of metric dicts from multiple scenarios or repeated seed trials.

    Returns:
        Summary dictionary with detailed statistical distributions and top-level means.
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)
    successes = sum(1 for m in metrics_list if m.get("success", False))
    success_rate = float(successes / n)

    # Metric keys to aggregate statistically
    numeric_keys = [
        "success_rate",
        "path_length",
        "steps",
        "final_distance",
        "collision_count",
        "collision_rate",
        "safety_interventions",
        "unsafe_action_attempts",
        "boundary_violations",
        "min_obstacle_clearance",
        "min_boundary_clearance",
        "predicted_collision_count",
        "min_moving_obstacle_clearance",
        "mean_moving_obstacle_clearance",
        "navigation_efficiency",
        "progress_ratio",
        "heading_smoothness",
        "action_switch_count",
        "oscillation_count",
        "mean_decision_time",
        "quantum_circuit_execution_time",
        "total_episode_time",
        "total_reward",
        "energy_proxy",
    ]

    stats: Dict[str, Dict[str, float]] = {}
    for key in numeric_keys:
        vals = [float(m.get(key, 0.0)) for m in metrics_list]
        stats[key] = compute_statistics(vals)

    mean_path = stats["path_length"]["mean"]
    mean_steps = stats["steps"]["mean"]
    mean_collisions = stats["collision_count"]["mean"]
    mean_collision_rate = stats["collision_rate"]["mean"]
    mean_efficiency = stats["navigation_efficiency"]["mean"]
    mean_final_dist = stats["final_distance"]["mean"]
    mean_interventions = stats["safety_interventions"]["mean"]
    mean_reward = stats["total_reward"]["mean"]
    mean_duration = stats["total_episode_time"]["mean"]

    return {
        "total_episodes": n,
        "success_count": successes,
        "success_rate": round(success_rate, 4),
        "mean_path_length": round(mean_path, 3),
        "mean_steps": round(mean_steps, 2),
        "mean_collisions": round(mean_collisions, 2),
        "mean_collision_count": round(mean_collisions, 2),
        "mean_collision_rate": round(mean_collision_rate, 4),
        "mean_navigation_efficiency": round(mean_efficiency, 4),
        "mean_efficiency": round(mean_efficiency, 4),
        "mean_final_distance": round(mean_final_dist, 3),
        "mean_safety_violations": round(mean_interventions, 2),
        "mean_safety_interventions": round(mean_interventions, 2),
        "mean_unsafe_action_attempts": stats["unsafe_action_attempts"]["mean"],
        "mean_boundary_violations": stats["boundary_violations"]["mean"],
        "mean_min_obstacle_clearance": stats["min_obstacle_clearance"]["mean"],
        "mean_min_boundary_clearance": stats["min_boundary_clearance"]["mean"],
        "mean_predicted_collision_count": stats["predicted_collision_count"]["mean"],
        "mean_min_moving_obstacle_clearance": stats["min_moving_obstacle_clearance"]["mean"],
        "mean_moving_obstacle_clearance": stats["mean_moving_obstacle_clearance"]["mean"],
        "mean_progress_ratio": stats["progress_ratio"]["mean"],
        "mean_heading_smoothness": stats["heading_smoothness"]["mean"],
        "mean_action_switch_count": stats["action_switch_count"]["mean"],
        "mean_oscillation_count": stats["oscillation_count"]["mean"],
        "mean_decision_time": stats["mean_decision_time"]["mean"],
        "mean_quantum_circuit_execution_time": stats["quantum_circuit_execution_time"]["mean"],
        "mean_reward": round(mean_reward, 2),
        "mean_duration_seconds": round(mean_duration, 4),
        "mean_episode_time": round(mean_duration, 4),
        "mean_energy_proxy": stats["energy_proxy"]["mean"],
        "distributions": stats,
    }
