import sys
import time
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from classical.feature_extractor import NavigationFeatureExtractor
from simulation.obstacles import ObstacleField, CircularObstacle, RectangularObstacle, MovingObstacle
from simulation.collision import check_boundary_violation
from config import DEFAULT_CONFIG


def enhanced_teacher_policy(
    drone_pos: np.ndarray,
    target_pos: np.ndarray,
    obstacles: ObstacleField,
    action_names: List[str],
    action_vectors: Dict[str, Tuple[float, float]],
    current_heading: Optional[float] = None,
    current_speed: float = 1.5,
    map_size: float = 50.0,
    lookahead_steps: int = 3,
    goal_threshold: float = 2.0,
) -> Tuple[str, int, float, List[str]]:
    """Enhanced multi-step lookahead teacher policy with APF force guidance,

    collision horizon checking, clearance rewards, and heading continuity.

    Returns:
        (best_action, best_idx, confidence, safe_actions)
    """
    curr_dist = float(np.linalg.norm(target_pos - drone_pos))
    stay_idx = action_names.index("STAY") if "STAY" in action_names else 8

    # 1. Goal arrival
    if curr_dist <= goal_threshold:
        return "STAY", stay_idx, 1.0, ["STAY"]

    # Target unit vector
    t_diff = target_pos - drone_pos
    t_dir = t_diff / (curr_dist + 1e-8)

    best_score = -float("inf")
    best_action = None
    best_idx = stay_idx
    safe_actions = []
    action_scores = {}

    for idx, name in enumerate(action_names):
        vec = np.array(action_vectors[name], dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue  # evaluate STAY only if no motion possible

        unit_v = vec / norm
        action_safe = True
        min_clearance = float("inf")
        p_prev = drone_pos.copy()

        # Multi-step forward projection (K steps)
        for k in range(1, lookahead_steps + 1):
            p_k = drone_pos + unit_v * (current_speed * k)

            if check_boundary_violation(p_k, radius=0.8, map_width=map_size, map_height=map_size):
                action_safe = False
                break

            if obstacles is not None:
                collided, _ = obstacles.check_collision(p_k, radius=1.0)
                if collided:
                    action_safe = False
                    break

                if obstacles.check_swept_collision(p_prev, p_k, radius=0.9):
                    action_safe = False
                    break

                _, clr_k, _ = obstacles.get_nearest_obstacle(p_k, radius=0.8)
                if clr_k < 1.1:
                    action_safe = False
                    break

                if clr_k < min_clearance:
                    min_clearance = clr_k

            p_prev = p_k

        if not action_safe:
            continue

        safe_actions.append(name)

        # 1. Target progress over K steps
        p_final = drone_pos + unit_v * (current_speed * lookahead_steps)
        dist_final = float(np.linalg.norm(target_pos - p_final))
        progress = (curr_dist - dist_final) / (current_speed * lookahead_steps)

        # 2. Heading alignment with goal vector
        align = float(np.dot(unit_v, t_dir))

        # 3. Clearance reward (saturating at 6.0m)
        clearance_rew = min(min_clearance, 6.0) / 6.0

        # 4. Heading smoothness (penalize abrupt 180 deg reversals and sharp turns)
        smoothness = 0.0
        if current_heading is not None:
            act_angle = float(np.arctan2(unit_v[1], unit_v[0]))
            ang_diff = abs((act_angle - current_heading + np.pi) % (2.0 * np.pi) - np.pi)
            smoothness = -1.5 * (ang_diff / np.pi) ** 2

        # Composite score
        score = (progress * 3.5) + (align * 2.5) + (clearance_rew * 1.5) + smoothness
        action_scores[name] = score

        if score > best_score:
            best_score = score
            best_action = name
            best_idx = idx

    if best_action is None:
        if len(safe_actions) == 0:
            return "STAY", stay_idx, 0.5, ["STAY"]
        best_action = safe_actions[0]
        best_idx = action_names.index(best_action)
        best_score = 0.0

    # Compute softmax confidence among evaluated safe actions
    if len(action_scores) > 1:
        vals = np.array(list(action_scores.values()))
        exp_v = np.exp(vals - np.max(vals))
        conf = float(exp_v[list(action_scores.keys()).index(best_action)] / np.sum(exp_v))
    else:
        conf = 0.9

    return best_action, best_idx, conf, safe_actions
