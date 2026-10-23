"""Navigation Feature Extraction Module for Q-Nav.

Computes and normalizes spatial, kinematic, and environmental features
used by both Classical APF and Quantum Feature Encoding pipelines.

The 12-dimensional feature vector consists of:
1. Normalized distance to target
2. Target direction X
3. Target direction Y
4. Target angle sine
5. Target angle cosine
6. Nearest obstacle distance
7. Front obstacle distance
8. Left obstacle distance
9. Right obstacle distance
10. Obstacle density
11. Normalized speed
12. Boundary proximity
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class NavigationFeatureExtractor:
    """Extracts a standardized 12-dimensional normalized feature vector from the environment state."""

    FEATURE_NAMES: List[str] = [
        "norm_target_dist",
        "target_dir_x",
        "target_dir_y",
        "target_angle_sin",
        "target_angle_cos",
        "norm_nearest_obs_dist",
        "front_obstacle_dist",
        "left_obstacle_dist",
        "right_obstacle_dist",
        "obstacle_density",
        "norm_speed",
        "boundary_proximity",
    ]

    def __init__(
        self,
        max_map_distance: float = 70.71,  # sqrt(50^2 + 50^2)
        max_obstacle_distance: float = 15.0,
        max_speed: float = 3.0,
        boundary_margin: float = 10.0,
    ):
        """Initialize the feature extractor with normalization bounds.

        Args:
            max_map_distance: Diagonal length of map for distance normalization.
            max_obstacle_distance: Influence threshold for obstacle proximity.
            max_speed: Maximum possible drone translation speed.
            boundary_margin: Distance threshold for boundary proximity warning.
        """
        self.max_map_distance = max_map_distance
        self.max_obstacle_distance = max_obstacle_distance
        self.max_speed = max_speed
        self.boundary_margin = boundary_margin

    def compute_sector_obstacle_distance(
        self,
        drone_pos: np.ndarray,
        heading: float,
        sector_center_angle: float,
        sector_half_width: float,
        obstacles: Any,
    ) -> float:
        """Compute the minimum clearance to any obstacle within an angular sector relative to heading.

        Args:
            drone_pos: [x, y] drone position.
            heading: Current drone heading angle in radians.
            sector_center_angle: Relative center angle of sector (e.g. 0 for front, +pi/2 for left, -pi/2 for right).
            sector_half_width: Angular half-width in radians.
            obstacles: ObstacleField or None.

        Returns:
            Minimum distance to an obstacle in that sector, capped at max_obstacle_distance.
        """
        if obstacles is None or not hasattr(obstacles, "obstacles") or not obstacles.obstacles:
            return self.max_obstacle_distance

        min_dist = self.max_obstacle_distance
        target_sector_angle = heading + sector_center_angle

        for obs in obstacles.obstacles:
            obs_center = obs.get_center()
            diff = obs_center - drone_pos
            center_dist = float(np.linalg.norm(diff))

            if center_dist > (self.max_obstacle_distance + getattr(obs, "radius", 3.0)):
                continue

            angle_to_obs = np.arctan2(diff[1], diff[0])
            # Smallest angular difference wrapped to [-pi, pi]
            angle_diff = (angle_to_obs - target_sector_angle + np.pi) % (2.0 * np.pi) - np.pi

            if abs(angle_diff) <= sector_half_width:
                surf_dist = obs.distance_to_surface(drone_pos, radius=0.8)
                if surf_dist < min_dist:
                    min_dist = max(0.0, surf_dist)

        return min_dist

    def extract_features(self, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Extract the exact 12-dimensional normalized feature vector.

        Args:
            state: Dictionary produced by UAVEnvironment.get_state().

        Returns:
            Tuple of (feature_vector_1d_array, feature_dict)
        """
        pos = np.asarray(state.get("drone_position", np.zeros(2)), dtype=float)
        heading = float(state.get("drone_heading", 0.0))
        obstacles = state.get("obstacles", None)

        # 1. Normalized distance to target
        target_dist = float(state.get("target_distance", 0.0))
        norm_target_dist = float(np.clip(target_dist / self.max_map_distance, 0.0, 1.0))

        # 2 & 3. Target direction unit vector
        target_dir = state.get("target_direction", np.zeros(2, dtype=float))
        target_dir_x = float(target_dir[0])
        target_dir_y = float(target_dir[1])

        # 4 & 5. Target angle sine and cosine relative to current heading
        target_angle = float(np.arctan2(target_dir_y, target_dir_x))
        rel_target_angle = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
        target_angle_sin = float(np.sin(rel_target_angle))
        target_angle_cos = float(np.cos(rel_target_angle))

        # 6. Nearest obstacle distance
        nearest_dist_raw = float(state.get("nearest_obstacle_distance", self.max_obstacle_distance))
        if nearest_dist_raw == float("inf"):
            nearest_dist_raw = self.max_obstacle_distance
        nearest_obstacle_dist = float(np.clip(nearest_dist_raw / self.max_obstacle_distance, 0.0, 1.0))

        # 7. Front obstacle distance (sector: heading +/- 35 deg = 0.61 rad)
        front_raw = self.compute_sector_obstacle_distance(
            drone_pos=pos,
            heading=heading,
            sector_center_angle=0.0,
            sector_half_width=0.65,
            obstacles=obstacles,
        )
        front_obstacle_dist = float(np.clip(front_raw / self.max_obstacle_distance, 0.0, 1.0))

        # 8. Left obstacle distance (sector: heading + 90 deg = +pi/2, half_width 0.65 rad)
        left_raw = self.compute_sector_obstacle_distance(
            drone_pos=pos,
            heading=heading,
            sector_center_angle=np.pi / 2.0,
            sector_half_width=0.65,
            obstacles=obstacles,
        )
        left_obstacle_dist = float(np.clip(left_raw / self.max_obstacle_distance, 0.0, 1.0))

        # 9. Right obstacle distance (sector: heading - 90 deg = -pi/2, half_width 0.65 rad)
        right_raw = self.compute_sector_obstacle_distance(
            drone_pos=pos,
            heading=heading,
            sector_center_angle=-np.pi / 2.0,
            sector_half_width=0.65,
            obstacles=obstacles,
        )
        right_obstacle_dist = float(np.clip(right_raw / self.max_obstacle_distance, 0.0, 1.0))

        # 10. Obstacle density
        density = float(state.get("obstacle_density", 0.0))
        obstacle_density = float(np.clip(density, 0.0, 1.0))

        # 11. Normalized speed
        speed = float(state.get("drone_speed", 0.0))
        norm_speed = float(np.clip(speed / self.max_speed, 0.0, 1.0))

        # 12. Boundary proximity
        b_dist = float(state.get("boundary_distance", self.boundary_margin))
        boundary_proximity = float(np.clip(b_dist / self.boundary_margin, 0.0, 1.0))

        features_dict = {
            "norm_target_dist": norm_target_dist,
            "target_dir_x": target_dir_x,
            "target_dir_y": target_dir_y,
            "target_angle_sin": target_angle_sin,
            "target_angle_cos": target_angle_cos,
            "norm_nearest_obs_dist": nearest_obstacle_dist,
            "front_obstacle_dist": front_obstacle_dist,
            "left_obstacle_dist": left_obstacle_dist,
            "right_obstacle_dist": right_obstacle_dist,
            "obstacle_density": obstacle_density,
            "norm_speed": norm_speed,
            "boundary_proximity": boundary_proximity,
        }

        feature_vector = np.array(
            [features_dict[name] for name in self.FEATURE_NAMES], dtype=float
        )

        return feature_vector, features_dict

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return ordered list of the 12 feature names."""
        return cls.FEATURE_NAMES.copy()

    def extract(
        self,
        drone_position: Any,
        target_position: Optional[Any] = None,
        obstacles: Optional[Any] = None,
    ) -> np.ndarray:
        """Extract 12D feature vector from positions or state dictionary.

        Args:
            drone_position: Drone [x, y] coordinates or full state dict.
            target_position: Target [x, y] coordinates.
            obstacles: ObstacleField or None.

        Returns:
            1D numpy array of 12 normalized features.
        """
        if isinstance(drone_position, dict):
            vec, _ = self.extract_features(drone_position)
            return vec

        pos = np.asarray(drone_position, dtype=float)
        t_pos = np.asarray(target_position if target_position is not None else pos, dtype=float)
        diff = t_pos - pos
        dist = float(np.linalg.norm(diff))
        dir_vec = diff / (dist + 1e-8) if dist > 1e-8 else np.zeros(2, dtype=float)

        state: Dict[str, Any] = {
            "position": pos,
            "drone_position": pos,
            "target_position": t_pos,
            "target_distance": dist,
            "target_direction": dir_vec,
            "obstacles": obstacles,
            "drone_speed": 1.5,
            "drone_heading": float(np.arctan2(dir_vec[1], dir_vec[0])) if dist > 1e-8 else 0.0,
        }
        if obstacles is not None and hasattr(obstacles, "get_nearest_obstacle"):
            _, min_d, obs_u = obstacles.get_nearest_obstacle(pos)
            state["nearest_obstacle_distance"] = min_d
            state["nearest_obstacle_direction"] = obs_u
            if hasattr(obstacles, "calculate_density"):
                try:
                    state["obstacle_density"] = obstacles.calculate_density(pos, radius=8.0)
                except TypeError:
                    try:
                        state["obstacle_density"] = obstacles.calculate_density(pos)
                    except Exception:
                        state["obstacle_density"] = 0.0

        from simulation.collision import distance_to_boundary
        map_dim = self.max_map_distance / np.sqrt(2.0)
        state["boundary_distance"] = distance_to_boundary(pos, map_dim, map_dim)

        vec, _ = self.extract_features(state)
        return vec


FeatureExtractor = NavigationFeatureExtractor
