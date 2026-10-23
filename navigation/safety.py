"""Safety Layer and Collision Prevention for Q-Nav.

Validates proposed candidate actions against boundary constraints, static obstacles,
and dynamic moving obstacle clearances, providing emergency evasive maneuvers when necessary.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from config import NavigationConfig, SimulationConfig, DEFAULT_CONFIG
from simulation.obstacles import ObstacleField, MovingObstacle
from simulation.collision import (
    check_boundary_violation,
    distance_to_boundary,
    check_swept_circle_line,
    predict_collision,
)


class SafetyLayer:
    """Monitors and enforces spatial safety boundaries, static clearance, and dynamic obstacle avoidance."""

    def __init__(
        self,
        nav_config: Optional[NavigationConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
        config: Optional[Any] = None,
        obstacles: Optional[ObstacleField] = None,
    ):
        """Initialize safety layer with buffer thresholds and dynamic prediction parameters."""
        if config is not None:
            if hasattr(config, "navigation"):
                nav_config = config.navigation
            if hasattr(config, "sim"):
                sim_config = config.sim

        self.nav_config = nav_config or DEFAULT_CONFIG.navigation
        self.sim_config = sim_config or DEFAULT_CONFIG.sim
        self.obstacle_buffer = self.nav_config.safety_obstacle_buffer
        self.drone_radius = self.sim_config.drone_radius
        self.collision_radius = self.sim_config.collision_radius
        self.lookahead_steps = self.nav_config.safety_lookahead_steps

        # Dynamic obstacle prediction parameters
        self.prediction_horizon: float = float(
            getattr(self.nav_config, "prediction_horizon", 1.0)
        )
        self.prediction_dt: float = float(
            getattr(self.nav_config, "prediction_dt", 0.1)
        )
        self.dynamic_safety_margin: float = float(
            getattr(self.nav_config, "dynamic_safety_margin", 0.5)
        )

        # Interventions counter for metrics
        self.intervention_count = 0
        self.unsafe_action_attempts = 0
        self.boundary_violation_attempts = 0
        self.total_checks = 0
        self.predicted_collision_count = 0
        self.min_clearance_observed = float("inf")
        self.min_boundary_clearance_observed = float("inf")
        self.obstacles: Optional[ObstacleField] = obstacles
        self.map_width: float = self.sim_config.map_width
        self.map_height: float = self.sim_config.map_height

        # Proactive safety telemetry records
        self.last_proactive_event: Optional[Dict[str, Any]] = None
        self.proactive_events_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset safety intervention counters and telemetry logs."""
        self.intervention_count = 0
        self.unsafe_action_attempts = 0
        self.boundary_violation_attempts = 0
        self.total_checks = 0
        self.predicted_collision_count = 0
        self.min_clearance_observed = float("inf")
        self.min_boundary_clearance_observed = float("inf")
        self.last_proactive_event = None
        self.proactive_events_log.clear()

    def get_safe_actions(
        self,
        state: Dict[str, Any],
        speed: Optional[float] = None,
        obstacles: Optional[ObstacleField] = None,
        map_width: Optional[float] = None,
        map_height: Optional[float] = None,
    ) -> List[int]:
        """Return list of discrete action indices (0-8) that satisfy lookahead safety constraints.

        Args:
            state: Current environment state dictionary.
            speed: Translation speed per step.
            obstacles: ObstacleField in environment.
            map_width: Width of map.
            map_height: Height of map.

        Returns:
            List of integer action indices that are safe to execute.
        """
        current_pos = np.asarray(state.get("drone_position", state.get("pos", [0.0, 0.0])), dtype=float)
        spd = speed if speed is not None else float(state.get("drone_speed", self.sim_config.drone_speed))
        obs = obstacles or state.get("obstacles") or self.obstacles or ObstacleField()
        w = map_width if map_width is not None else float(state.get("map_width", self.map_width))
        h = map_height if map_height is not None else float(state.get("map_height", self.map_height))

        safe_actions: List[int] = []
        for idx, name in enumerate(self.nav_config.action_names):
            vec = np.array(self.nav_config.action_vectors[name], dtype=float)
            is_safe, _, _ = self.evaluate_action_safety(
                current_pos=current_pos,
                direction=vec,
                speed=spd,
                obstacles=obs,
                map_width=w,
                map_height=h,
                record_telemetry=False,
            )
            if is_safe:
                safe_actions.append(idx)

        return safe_actions

    def evaluate_action_safety(
        self,
        current_pos: np.ndarray,
        direction: np.ndarray,
        speed: float,
        obstacles: ObstacleField,
        map_width: float = 50.0,
        map_height: float = 50.0,
        record_telemetry: bool = True,
    ) -> Tuple[bool, float, str]:
        """Check if executing a step along direction is safe against static and moving obstacles.

        Args:
            current_pos: [x, y] coordinates of drone.
            direction: 2D proposed motion direction vector.
            speed: Translation speed per step.
            obstacles: ObstacleField in the environment.
            map_width: Width of the map.
            map_height: Height of the map.
            record_telemetry: Whether to increment intervention and clearance telemetry counters.

        Returns:
            Tuple of (is_safe, min_clearance, reason)
        """
        if record_telemetry:
            self.total_checks += 1
        dir_vec = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(dir_vec)

        # Track current boundary clearance
        cur_b_dist = max(0.0, distance_to_boundary(current_pos, map_width, map_height) - self.drone_radius)
        if record_telemetry and cur_b_dist < self.min_boundary_clearance_observed:
            self.min_boundary_clearance_observed = float(cur_b_dist)

        moving_obstacles = getattr(obstacles, "moving_obstacles", [])

        if norm < 1e-6:
            # STAY action: check current position boundary and obstacle distance
            is_oob = check_boundary_violation(
                current_pos, self.drone_radius, map_width, map_height
            )
            if is_oob:
                if record_telemetry:
                    self.unsafe_action_attempts += 1
                    self.boundary_violation_attempts += 1
                return False, 0.0, "out_of_bounds"

            has_collided, _ = obstacles.check_collision(
                current_pos, self.collision_radius
            )
            if has_collided:
                if record_telemetry:
                    self.unsafe_action_attempts += 1
                return False, 0.0, "in_obstacle"

            # Check moving obstacles forward in time even while staying in place
            if moving_obstacles:
                n_intervals = max(1, int(np.ceil(self.prediction_horizon / self.prediction_dt)))
                for k in range(1, n_intervals + 1):
                    t_eval = k * self.prediction_dt
                    for m_obs in moving_obstacles:
                        p_pred = m_obs.predict_position(t_eval)
                        pred_dist = float(np.linalg.norm(current_pos - p_pred)) - (self.drone_radius + m_obs.radius)
                        if pred_dist < (self.dynamic_safety_margin):
                            if record_telemetry:
                                self.unsafe_action_attempts += 1
                                self.predicted_collision_count += 1
                                self.last_proactive_event = {
                                    "predicted_collision": True,
                                    "obstacle_id": m_obs.obs_id,
                                    "predicted_min_clearance": float(pred_dist),
                                }
                            return False, pred_dist, f"moving_obstacle_hazard_obs_{m_obs.obs_id}"

            _, min_dist, _ = obstacles.get_nearest_obstacle(
                current_pos, self.drone_radius
            )
            if record_telemetry and min_dist < self.min_clearance_observed:
                self.min_clearance_observed = float(min_dist)
            return True, min_dist, "safe"

        unit_dir = dir_vec / norm

        # 1. Project lookahead positions for static obstacles and boundary
        for step in range(1, self.lookahead_steps + 1):
            projected_pos = current_pos + unit_dir * (speed * step)

            # Boundary check
            if check_boundary_violation(
                projected_pos, self.drone_radius, map_width, map_height
            ):
                if record_telemetry:
                    self.unsafe_action_attempts += 1
                    self.boundary_violation_attempts += 1
                return False, 0.0, f"boundary_violation_step_{step}"

            # Obstacle static collision check
            has_collided, _ = obstacles.check_collision(
                projected_pos, self.collision_radius
            )
            if has_collided:
                if record_telemetry:
                    self.unsafe_action_attempts += 1
                return False, 0.0, f"obstacle_collision_step_{step}"

            # Swept-path continuous collision check
            prev_proj = current_pos + unit_dir * (speed * (step - 1))
            if obstacles.check_swept_collision(
                prev_proj, projected_pos, self.collision_radius
            ):
                if record_telemetry:
                    self.unsafe_action_attempts += 1
                return False, 0.0, f"swept_collision_step_{step}"

        # 2. Predictive swept collision checking against moving obstacles
        if moving_obstacles:
            n_intervals = max(1, int(np.ceil(self.prediction_horizon / self.prediction_dt)))
            for k in range(1, n_intervals + 1):
                t_curr = k * self.prediction_dt
                t_prev = (k - 1) * self.prediction_dt
                p_uav_prev = current_pos + unit_dir * (speed * t_prev)
                p_uav_curr = current_pos + unit_dir * (speed * t_curr)

                for m_obs in moving_obstacles:
                    p_obs_prev = m_obs.predict_position(t_prev)
                    p_obs_curr = m_obs.predict_position(t_curr)

                    collided, min_c_dist, _ = predict_collision(
                        p_uav_prev,
                        p_uav_curr,
                        self.drone_radius + self.dynamic_safety_margin,
                        p_obs_prev,
                        p_obs_curr,
                        m_obs.radius,
                    )
                    clearance = min_c_dist - (self.drone_radius + m_obs.radius)

                    if record_telemetry and clearance < self.min_clearance_observed:
                        self.min_clearance_observed = float(clearance)

                    if collided or clearance < self.dynamic_safety_margin:
                        if record_telemetry:
                            self.unsafe_action_attempts += 1
                            self.predicted_collision_count += 1
                            self.last_proactive_event = {
                                "predicted_collision": True,
                                "obstacle_id": m_obs.obs_id,
                                "predicted_min_clearance": float(clearance),
                            }
                        return False, float(clearance), f"moving_obstacle_hazard_obs_{m_obs.obs_id}"

        # Calculate clearance to nearest obstacle along trajectory
        next_pos = current_pos + unit_dir * speed
        _, min_clearance, _ = obstacles.get_nearest_obstacle(next_pos, self.drone_radius)
        if min_clearance < self.min_clearance_observed and record_telemetry:
            self.min_clearance_observed = float(min_clearance)

        next_b_dist = max(0.0, distance_to_boundary(next_pos, map_width, map_height) - self.drone_radius)
        if next_b_dist < self.min_boundary_clearance_observed and record_telemetry:
            self.min_boundary_clearance_observed = float(next_b_dist)

        if min_clearance < self.obstacle_buffer:
            if record_telemetry:
                self.unsafe_action_attempts += 1
            return False, min_clearance, "clearance_below_safety_buffer"

        return True, min_clearance, "safe"

    def select_safe_action(
        self,
        current_pos: Optional[np.ndarray] = None,
        candidate_actions: Optional[List[str]] = None,
        action_scores: Optional[Dict[str, float]] = None,
        action_vectors: Optional[Dict[str, Tuple[float, float]]] = None,
        speed: float = 1.5,
        obstacles: Optional[ObstacleField] = None,
        map_width: float = 50.0,
        map_height: float = 50.0,
        *,
        proposed_action: Optional[Union[int, str]] = None,
        state: Optional[Dict[str, Any]] = None,
        safe_actions: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[Tuple[str, np.ndarray, bool], int]:
        """Find the highest-scoring candidate action that satisfies safety requirements.

        Supports both string-named candidate action tuples and integer action selection.
        If all actions are unsafe, returns integer 0 / STAY (never None).

        Returns:
            Tuple of (safe_action_name, safe_direction_vector, was_intervened) or integer action.
        """
        # Handle integer proposed_action interface
        if proposed_action is not None:
            if safe_actions is None and state is not None:
                safe_actions = self.get_safe_actions(state)

            if isinstance(proposed_action, (int, np.integer)):
                p_act = int(proposed_action)
                if safe_actions is not None and p_act in safe_actions:
                    return p_act
                # Intervention: pick alternative safe action or STAY (0)
                self.intervention_count += 1
                if safe_actions and len(safe_actions) > 0:
                    return int(safe_actions[0])
                return 0  # STAY, never None

        obs_field = obstacles if obstacles is not None else self.obstacles
        if obs_field is None:
            obs_field = ObstacleField()

        if candidate_actions is None or action_scores is None or action_vectors is None or current_pos is None:
            return "STAY", np.zeros(2, dtype=float), True

        ranked_actions = sorted(
            candidate_actions, key=lambda a: action_scores.get(a, -float("inf")), reverse=True
        )

        primary_action = ranked_actions[0]
        primary_vec = np.array(action_vectors[primary_action], dtype=float)

        is_safe, _, _ = self.evaluate_action_safety(
            current_pos, primary_vec, speed, obs_field, map_width, map_height
        )

        if is_safe:
            return primary_action, primary_vec, False

        # Safety intervention triggered!
        self.intervention_count += 1
        selected_action: Optional[str] = None
        selected_vec: Optional[np.ndarray] = None

        # Search remaining ranked actions for a safe alternative
        for alt_action in ranked_actions[1:]:
            alt_vec = np.array(action_vectors[alt_action], dtype=float)
            alt_safe, _, _ = self.evaluate_action_safety(
                current_pos, alt_vec, speed, obstacles, map_width, map_height
            )
            if alt_safe:
                selected_action = alt_action
                selected_vec = alt_vec
                break

        # Emergency Fallback: move directly away from nearest obstacle
        if selected_action is None:
            _, _, away_dir = obstacles.get_nearest_obstacle(current_pos, self.drone_radius)
            away_norm = np.linalg.norm(away_dir)
            if away_norm > 1e-6:
                emergency_vec = away_dir / away_norm
                em_safe, _, _ = self.evaluate_action_safety(
                    current_pos, emergency_vec, speed, obstacles, map_width, map_height
                )
                if em_safe:
                    selected_action = "EMERGENCY_EVASION"
                    selected_vec = emergency_vec

        # If completely trapped, halt with STAY (do NOT return None)
        if selected_action is None:
            selected_action = "STAY"
            selected_vec = np.array(action_vectors.get("STAY", (0.0, 0.0)), dtype=float)

        # Record proactive intervention log
        event_entry = {
            "original_action": primary_action,
            "replacement_action": selected_action,
            "intervention": True,
            "predicted_collision": bool(
                self.last_proactive_event.get("predicted_collision", False)
            ) if self.last_proactive_event else False,
            "obstacle_id": self.last_proactive_event.get("obstacle_id") if self.last_proactive_event else None,
            "predicted_min_clearance": float(
                self.last_proactive_event.get("predicted_min_clearance", 0.0)
            ) if self.last_proactive_event else 0.0,
        }
        self.proactive_events_log.append(event_entry)

        return selected_action, selected_vec, True

