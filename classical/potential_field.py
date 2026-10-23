"""Artificial Potential Field (APF) Navigation Module for Q-Nav.

Implements attractive potential toward the target, repulsive potentials from obstacles
and map boundaries, tangential obstacle circulation forces, stagnation detection,
oscillation detection, and intelligent dual-direction bypass recovery.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from config import ClassicalConfig, DEFAULT_CONFIG
from simulation.obstacles import ObstacleField


class ArtificialPotentialField:
    """Computes virtual attractive, repulsive, and tangential forces guiding the UAV."""

    def __init__(
        self,
        config: Optional[ClassicalConfig] = None,
        k_att: Optional[float] = None,
        k_rep: Optional[float] = None,
        k_tangential: Optional[float] = None,
        stagnation_window: Optional[int] = None,
        stagnation_distance_threshold: Optional[float] = None,
        progress_threshold: Optional[float] = None,
        oscillation_window: Optional[int] = None,
        oscillation_switch_threshold: Optional[int] = None,
    ):
        """Initialize APF with gain coefficients and influence distance."""
        self.config = config or DEFAULT_CONFIG.classical
        self.k_att = k_att if k_att is not None else self.config.attractive_coefficient
        self.k_rep = k_rep if k_rep is not None else self.config.repulsive_coefficient
        self.k_tan = k_tangential if k_tangential is not None else self.config.tangential_coefficient
        self.d_0 = self.config.obstacle_influence_distance
        self.minima_threshold = self.config.local_minima_threshold
        self.perturbation_mag = self.config.perturbation_magnitude
        self.enable_tangential = self.config.enable_tangential_force

        # Stagnation & Oscillation parameters
        self.stagnation_window = stagnation_window if stagnation_window is not None else self.config.stagnation_window
        self.stagnation_distance_thresh = (
            stagnation_distance_threshold
            if stagnation_distance_threshold is not None
            else self.config.stagnation_distance_threshold
        )
        self.progress_thresh = progress_threshold if progress_threshold is not None else self.config.progress_threshold
        self.oscillation_window = oscillation_window if oscillation_window is not None else self.config.oscillation_window
        self.oscillation_switch_thresh = (
            oscillation_switch_threshold
            if oscillation_switch_threshold is not None
            else min(self.config.oscillation_switch_threshold, max(2, self.oscillation_window - 1))
        )
        self.recovery_duration = self.config.recovery_duration_steps
        self.fallback_thresh = self.config.fallback_stagnation_steps

        # State tracking
        self.recent_positions: List[np.ndarray] = []
        self.recent_target_distances: List[float] = []
        self.recent_actions: List[np.ndarray] = []

        self.stagnation_counter = 0
        self.recovery_steps_remaining = 0
        self.active_bypass_direction: Optional[np.ndarray] = None
        self.oscillation_detected = False

    def reset(self) -> None:
        """Reset internal history and counters."""
        self.recent_positions.clear()
        self.recent_target_distances.clear()
        self.recent_actions.clear()
        self.stagnation_counter = 0
        self.recovery_steps_remaining = 0
        self.active_bypass_direction = None
        self.oscillation_detected = False

    def compute_attractive_force(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """Calculate attractive force vector pointing directly toward target.

        F_att = k_att * (p_target - p_drone)
        """
        diff = np.asarray(target_pos, dtype=float) - np.asarray(drone_pos, dtype=float)
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2, dtype=float)
        unit_vec = diff / dist
        magnitude = min(dist, 10.0) * self.k_att
        return unit_vec * magnitude

    def compute_repulsive_force(
        self,
        drone_pos: np.ndarray,
        obstacles: ObstacleField,
        drone_radius: float = 0.8,
    ) -> np.ndarray:
        """Calculate aggregate repulsive force from all nearby obstacles.

        For each obstacle within influence distance d_0:
            F_rep = k_rep * (1/d - 1/d_0) * (1 / d^2) * unit_away
        """
        total_rep = np.zeros(2, dtype=float)
        nearby = obstacles.get_nearby_obstacles(
            drone_pos, search_radius=self.d_0, drone_radius=drone_radius
        )

        for obs, dist in nearby:
            effective_dist = max(0.1, dist)
            if effective_dist < self.d_0:
                rep_magnitude = (
                    self.k_rep
                    * (1.0 / effective_dist - 1.0 / self.d_0)
                    * (1.0 / (effective_dist**2))
                )
                obs_center = obs.get_center()
                diff = np.asarray(drone_pos, dtype=float) - obs_center
                norm = np.linalg.norm(diff)
                unit_away = diff / norm if norm > 1e-6 else np.array([1.0, 0.0])
                total_rep += unit_away * rep_magnitude

        return total_rep

    def compute_tangential_force(
        self,
        drone_pos: np.ndarray,
        obstacles: ObstacleField,
        f_att: np.ndarray,
        drone_radius: float = 0.8,
    ) -> np.ndarray:
        """Calculate tangential circulation force allowing the drone to curve around obstacles.

        The circulation direction (clockwise vs counter-clockwise) is chosen
        to align with the target attractive force direction.
        """
        if not self.enable_tangential:
            return np.zeros(2, dtype=float)

        total_tan = np.zeros(2, dtype=float)
        nearby = obstacles.get_nearby_obstacles(
            drone_pos, search_radius=self.d_0 * 0.8, drone_radius=drone_radius
        )

        for obs, dist in nearby:
            effective_dist = max(0.1, dist)
            if effective_dist < self.d_0:
                obs_center = obs.get_center()
                diff = np.asarray(drone_pos, dtype=float) - obs_center
                norm = np.linalg.norm(diff)
                if norm < 1e-6:
                    continue
                unit_away = diff / norm

                # Candidate tangent vector 1: rotate +90 degrees (-y, x)
                tan_candidate = np.array([-unit_away[1], unit_away[0]], dtype=float)

                # Choose tangent orientation that aligns best with attractive target vector
                if np.dot(tan_candidate, f_att) < 0.0:
                    tan_candidate = -tan_candidate

                tan_mag = (
                    self.k_tan
                    * (1.0 / effective_dist - 1.0 / self.d_0)
                    * (1.0 / effective_dist)
                )
                total_tan += tan_candidate * tan_mag

        return total_tan

    def compute_boundary_repulsive_force(
        self,
        drone_pos: np.ndarray,
        map_width: float,
        map_height: float,
        margin: float = 5.0,
    ) -> np.ndarray:
        """Calculate repulsive force from map boundaries to keep drone within limits."""
        rep = np.zeros(2, dtype=float)
        x, y = float(drone_pos[0]), float(drone_pos[1])

        if x < margin:
            rep[0] += self.k_rep * (1.0 / max(0.1, x) - 1.0 / margin)
        if (map_width - x) < margin:
            rep[0] -= self.k_rep * (1.0 / max(0.1, map_width - x) - 1.0 / margin)
        if y < margin:
            rep[1] += self.k_rep * (1.0 / max(0.1, y) - 1.0 / margin)
        if (map_height - y) < margin:
            rep[1] -= self.k_rep * (1.0 / max(0.1, map_height - y) - 1.0 / margin)

        return rep

    def detect_stagnation(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> bool:
        """Detect whether the UAV is making insufficient progress toward the target.

        Uses stagnation_window, stagnation_distance_threshold, and progress_threshold.
        """
        dist_to_target = float(np.linalg.norm(target_pos - drone_pos))
        self.recent_positions.append(drone_pos.copy())
        self.recent_target_distances.append(dist_to_target)

        if len(self.recent_positions) > self.stagnation_window:
            self.recent_positions.pop(0)
            self.recent_target_distances.pop(0)

        if len(self.recent_positions) < self.stagnation_window or dist_to_target <= 2.5:
            return False

        # Displacement over window
        displacement = float(np.linalg.norm(self.recent_positions[-1] - self.recent_positions[0]))
        # Progress toward target (positive if getting closer)
        target_progress = self.recent_target_distances[0] - self.recent_target_distances[-1]

        is_stagnant = (
            displacement < self.stagnation_distance_thresh
            or target_progress < self.progress_thresh
        )
        return is_stagnant

    def detect_oscillation(self, action_vector: Union[np.ndarray, str, List[float]]) -> bool:
        """Detect repeated alternating motion (e.g. LEFT, RIGHT, LEFT, RIGHT)."""
        if isinstance(action_vector, str):
            vec_map = {
                "EAST": (1.0, 0.0), "WEST": (-1.0, 0.0), "NORTH": (0.0, 1.0), "SOUTH": (0.0, -1.0),
                "RIGHT": (1.0, 0.0), "LEFT": (-1.0, 0.0), "UP": (0.0, 1.0), "DOWN": (0.0, -1.0),
                "STAY": (0.0, 0.0)
            }
            vec = np.array(vec_map.get(action_vector.upper(), (0.0, 0.0)), dtype=float)
        else:
            vec = np.asarray(action_vector, dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            self.recent_actions.append(vec / norm)
        else:
            self.recent_actions.append(np.zeros(2))

        if len(self.recent_actions) > self.oscillation_window:
            self.recent_actions.pop(0)

        if len(self.recent_actions) < self.oscillation_window:
            return False

        # Count directional reversals (negative dot products between consecutive steps)
        reversals = 0
        for i in range(len(self.recent_actions) - 1):
            dot = np.dot(self.recent_actions[i], self.recent_actions[i + 1])
            if dot < -0.5:
                reversals += 1

        self.oscillation_detected = reversals >= self.oscillation_switch_thresh
        return self.oscillation_detected

    def select_safer_bypass_direction(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacles: ObstacleField,
        lookahead_distance: float = 3.5,
    ) -> np.ndarray:
        """Evaluate left and right bypass directions and select the direction with higher obstacle clearance.

        1. Generate left (+90 deg) and right (-90 deg) bypass directions relative to target vector.
        2. Estimate clearance for both candidate positions.
        3. Return the safer direction vector.
        """
        diff = target_pos - drone_pos
        dist = np.linalg.norm(diff)
        if dist > 1e-6:
            fwd = diff / dist
        else:
            fwd = np.array([1.0, 0.0])

        # Left (+90 deg) and Right (-90 deg) unit vectors
        left_dir = np.array([-fwd[1], fwd[0]], dtype=float)
        right_dir = np.array([fwd[1], -fwd[0]], dtype=float)

        # Lookahead candidate positions
        pos_left = drone_pos + left_dir * lookahead_distance
        pos_right = drone_pos + right_dir * lookahead_distance

        # Estimate clearance to nearest obstacle
        _, clear_left, _ = obstacles.get_nearest_obstacle(pos_left, radius=0.8)
        _, clear_right, _ = obstacles.get_nearest_obstacle(pos_right, radius=0.8)

        # Select direction with larger clearance
        if clear_left >= clear_right:
            return left_dir
        else:
            return right_dir

    def compute_total_force(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacles: ObstacleField,
        map_width: float = 50.0,
        map_height: float = 50.0,
        drone_radius: float = 0.8,
        last_action_vector: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute the combined navigation force with tangential smoothing, stagnation bypass, and recovery.

        Returns:
            Tuple of (total_force_vector, force_breakdown_dict)
        """
        f_att = self.compute_attractive_force(drone_pos, target_pos)
        f_rep_obs = self.compute_repulsive_force(drone_pos, obstacles, drone_radius)
        f_tan_obs = self.compute_tangential_force(drone_pos, obstacles, f_att, drone_radius)
        f_rep_bound = self.compute_boundary_repulsive_force(
            drone_pos, map_width, map_height
        )

        f_base = f_att + f_rep_obs + f_tan_obs + f_rep_bound
        force_norm = float(np.linalg.norm(f_base))

        dist_to_target = float(np.linalg.norm(target_pos - drone_pos))

        # Check conditions
        is_stagnant = self.detect_stagnation(drone_pos, target_pos)
        if last_action_vector is not None:
            self.detect_oscillation(last_action_vector)

        is_equilibrium = (force_norm < self.minima_threshold and dist_to_target > 2.0)
        needs_recovery = (is_stagnant or self.oscillation_detected or is_equilibrium)

        if needs_recovery and self.recovery_steps_remaining <= 0:
            # Trigger fresh recovery mode: choose safer bypass direction
            self.stagnation_counter += 1
            self.recovery_steps_remaining = self.recovery_duration
            self.active_bypass_direction = self.select_safer_bypass_direction(
                drone_pos, target_pos, obstacles
            )

        f_recovery = np.zeros(2, dtype=float)
        in_recovery_mode = False

        if self.recovery_steps_remaining > 0 and self.active_bypass_direction is not None:
            in_recovery_mode = True
            self.recovery_steps_remaining -= 1
            # Recovery force: strong lateral bypass + forward bias
            f_recovery = (
                self.active_bypass_direction * (self.perturbation_mag * 3.5)
                + (f_att / (np.linalg.norm(f_att) + 1e-6)) * (self.perturbation_mag * 1.5)
            )
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        f_total = f_base + f_recovery

        breakdown = {
            "f_att": f_att,
            "f_rep_obs": f_rep_obs,
            "f_tan_obs": f_tan_obs,
            "f_rep_bound": f_rep_bound,
            "f_recovery": f_recovery,
            "f_total": f_total,
            "is_stagnant": is_stagnant,
            "is_equilibrium": is_equilibrium,
            "oscillation_detected": self.oscillation_detected,
            "in_recovery_mode": in_recovery_mode,
            "stagnation_counter": self.stagnation_counter,
            "recovery_steps_remaining": self.recovery_steps_remaining,
        }

        return f_total, breakdown
