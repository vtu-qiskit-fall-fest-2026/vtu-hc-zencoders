"""Obstacle Representation and Spatial Queries for Q-Nav.

Supports circular, rectangular (AABB), and dynamic moving obstacles, distance queries,
nearest-obstacle detection, obstacle density calculation, and predictive collision checking.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from simulation.collision import (
    distance_point_to_point,
    check_circle_circle,
    distance_circle_circle,
    check_circle_box,
    distance_circle_box,
    check_swept_circle_line,
    predict_collision,
)


class BaseObstacle(ABC):
    """Abstract base class for all obstacle geometries."""

    def __init__(self, obs_id: int, shape_type: str):
        self.obs_id = obs_id
        self.shape_type = shape_type

    @abstractmethod
    def check_collision(self, center: np.ndarray, radius: float) -> bool:
        """Check collision with a circle of given center and radius."""
        pass

    @abstractmethod
    def distance_to_surface(self, center: np.ndarray, radius: float = 0.0) -> float:
        """Calculate clearance from obstacle surface to circle surface."""
        pass

    @abstractmethod
    def get_center(self) -> np.ndarray:
        """Return the geometric center of the obstacle."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize obstacle to a dictionary."""
        pass


class CircularObstacle(BaseObstacle):
    """Cylindrical/circular obstacle defined by center (x, y) and radius."""

    def __init__(self, obs_id: int, x: float, y: float, radius: float):
        super().__init__(obs_id, "circle")
        self.center = np.array([x, y], dtype=float)
        self.radius = float(radius)

    def check_collision(self, center: np.ndarray, radius: float) -> bool:
        return check_circle_circle(self.center, self.radius, center, radius)

    def distance_to_surface(self, center: np.ndarray, radius: float = 0.0) -> float:
        return distance_circle_circle(self.center, self.radius, center, radius)

    def get_center(self) -> np.ndarray:
        return self.center.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obs_id,
            "type": "circle",
            "x": float(self.center[0]),
            "y": float(self.center[1]),
            "radius": float(self.radius),
        }


class RectangularObstacle(BaseObstacle):
    """Axis-aligned rectangular box defined by (x_min, y_min, x_max, y_max)."""

    def __init__(
        self,
        obs_id: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ):
        super().__init__(obs_id, "rectangle")
        self.bounds = (float(x_min), float(y_min), float(x_max), float(y_max))
        self.center = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=float
        )

    def check_collision(self, center: np.ndarray, radius: float) -> bool:
        return check_circle_box(center, radius, self.bounds)

    def distance_to_surface(self, center: np.ndarray, radius: float = 0.0) -> float:
        return distance_circle_box(center, radius, self.bounds)

    def get_center(self) -> np.ndarray:
        return self.center.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obs_id,
            "type": "rectangle",
            "x_min": self.bounds[0],
            "y_min": self.bounds[1],
            "x_max": self.bounds[2],
            "y_max": self.bounds[3],
        }


class MovingObstacle(BaseObstacle):
    """Dynamic moving obstacle supporting linear, circular, and deterministic random walk motion.

    Attributes:
        obs_id: Unique obstacle identifier.
        position: Current 2D position [x, y].
        initial_position: Original initial position [x, y].
        radius: Physical collision radius of the obstacle.
        pattern: Motion pattern ('linear', 'circular', 'random_walk').
        velocity: Current 2D velocity vector [vx, vy].
        initial_velocity: Original initial velocity vector [vx, vy].
        center: Orbit center [cx, cy] for circular motion.
        orbit_radius: Radius of circular orbit (separate from physical collision radius).
        angular_speed: Angular speed (radians/sec) for circular motion.
        phase: Current angular phase for circular motion.
        initial_phase: Initial angular phase for circular motion.
        random_seed: Random seed for deterministic random walk.
        bounds: Map coordinate boundaries (min_x, min_y, max_x, max_y).
        t: Simulation elapsed time.
    """

    def __init__(
        self,
        obs_id: Optional[int] = None,
        position: Optional[Union[List[float], Tuple[float, float], np.ndarray]] = None,
        radius: float = 1.0,
        pattern: str = "linear",
        velocity: Optional[Union[List[float], Tuple[float, float], np.ndarray]] = None,
        center: Optional[Union[List[float], Tuple[float, float], np.ndarray]] = None,
        orbit_radius: Optional[float] = None,
        angular_speed: float = 0.5,
        phase: float = 0.0,
        random_seed: int = 42,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, 50.0, 50.0),
        max_speed: Optional[float] = None,
        obstacle_id: Optional[int] = None,
        **kwargs,
    ):
        # Handle cases where position is passed as first positional argument instead of obs_id
        if obs_id is not None and not isinstance(obs_id, (int, np.integer)):
            # First arg was actually position!
            actual_pos = obs_id
            actual_id = int(obstacle_id or 0)
        else:
            actual_id = int(obs_id if obs_id is not None else (obstacle_id or 0))
            actual_pos = position if position is not None else kwargs.get("initial_position", [0.0, 0.0])

        super().__init__(actual_id, "moving")
        self.initial_position = np.array(actual_pos, dtype=float)
        self.position = self.initial_position.copy()
        self.radius = float(radius)
        self.pattern = str(pattern).lower()

        # Velocity handling
        if velocity is not None:
            self.initial_velocity = np.array(velocity, dtype=float)
        else:
            self.initial_velocity = np.array([0.5, 0.0], dtype=float)
        self.velocity = self.initial_velocity.copy()

        # Circular motion parameters
        if center is not None:
            self.center = np.array(center, dtype=float)
        else:
            self.center = self.initial_position.copy()

        if orbit_radius is not None:
            self.orbit_radius = float(orbit_radius)
        else:
            diff = self.initial_position - self.center
            dist = float(np.linalg.norm(diff))
            self.orbit_radius = dist if dist > 1e-6 else 3.0

        self.angular_speed = float(angular_speed)
        self.phase = float(phase)
        self.initial_phase = float(phase)

        # Random walk state
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.bounds = tuple(float(b) for b in bounds)
        v_mag = float(np.linalg.norm(self.initial_velocity))
        self.max_speed = float(max_speed) if max_speed is not None else max(1.0, v_mag * 1.5)


        # Simulation clock and state history
        self.t: float = 0.0
        self.history: List[Tuple[float, float]] = [(float(self.position[0]), float(self.position[1]))]

    def reset(self) -> None:
        """Reset obstacle to initial position, velocity, phase, clock, and deterministic RNG state."""
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()
        self.t = 0.0
        self.phase = self.initial_phase
        self.rng = np.random.default_rng(self.random_seed)
        self.history = [(float(self.position[0]), float(self.position[1]))]

    def _reflect_bounds(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply coordinate-wise boundary reflection handling X and Y coordinates independently."""
        min_x, min_y, max_x, max_y = self.bounds
        eff_min_x, eff_max_x = min_x + self.radius, max_x - self.radius
        eff_min_y, eff_max_y = min_y + self.radius, max_y - self.radius

        if eff_min_x >= eff_max_x:
            eff_min_x, eff_max_x = min_x, max_x
        if eff_min_y >= eff_max_y:
            eff_min_y, eff_max_y = min_y, max_y

        x = float(pos[0])
        y = float(pos[1])
        vx = float(vel[0])
        vy = float(vel[1])

        # Explicit coordinate checks
        if x < eff_min_x:
            x = eff_min_x
            vx = abs(vx) if vx != 0.0 else 0.5
        elif x > eff_max_x:
            x = eff_max_x
            vx = -abs(vx) if vx != 0.0 else -0.5

        if y < eff_min_y:
            y = eff_min_y
            vy = abs(vy) if vy != 0.0 else 0.5
        elif y > eff_max_y:
            y = eff_max_y
            vy = -abs(vy) if vy != 0.0 else -0.5

        return np.array([x, y], dtype=float), np.array([vx, vy], dtype=float)

    def step(self, dt: float = 1.0) -> np.ndarray:
        """Advance obstacle state forward by dt.

        Args:
            dt: Simulation timestep delta.

        Returns:
            New 2D position [x, y].
        """
        self.t += dt

        if self.pattern == "linear":
            proposed_pos = self.position + self.velocity * dt
            self.position, self.velocity = self._reflect_bounds(proposed_pos, self.velocity)

        elif self.pattern == "circular":
            theta = self.phase + self.angular_speed * self.t
            x = self.center[0] + self.orbit_radius * np.cos(theta)
            y = self.center[1] + self.orbit_radius * np.sin(theta)
            self.position = np.array([x, y], dtype=float)
            self.velocity = np.array([
                -self.orbit_radius * self.angular_speed * np.sin(theta),
                self.orbit_radius * self.angular_speed * np.cos(theta),
            ], dtype=float)

        elif self.pattern == "random_walk":
            dv = self.rng.normal(0.0, 0.2, size=2)
            new_vel = self.velocity + dv
            spd = float(np.linalg.norm(new_vel))
            if spd > self.max_speed and spd > 1e-6:
                new_vel = (new_vel / spd) * self.max_speed
            elif spd < 0.2:
                new_vel = (new_vel / max(1e-6, spd)) * 0.5

            proposed_pos = self.position + new_vel * dt
            self.position, self.velocity = self._reflect_bounds(proposed_pos, new_vel)

        self.history.append((float(self.position[0]), float(self.position[1])))
        return self.position.copy()

    def predict_position(self, horizon: float, dt: Optional[float] = None) -> np.ndarray:
        """Predict obstacle position at future time (t + horizon) without mutating internal state.

        Args:
            horizon: Lookahead duration in seconds.
            dt: Optional simulation delta.

        Returns:
            Predicted 2D position [x, y].
        """
        if horizon <= 0.0:
            return self.position.copy()

        if self.pattern == "circular":
            future_t = self.t + horizon
            theta = self.phase + self.angular_speed * future_t
            x = self.center[0] + self.orbit_radius * np.cos(theta)
            y = self.center[1] + self.orbit_radius * np.sin(theta)
            return np.array([x, y], dtype=float)

        elif self.pattern in ("linear", "random_walk"):
            step_dt = dt if dt is not None and dt > 0 else min(horizon, 0.2)
            n_steps = max(1, int(np.ceil(horizon / step_dt)))
            eff_dt = horizon / n_steps

            sim_pos = self.position.copy()
            sim_vel = self.velocity.copy()

            for _ in range(n_steps):
                prop_pos = sim_pos + sim_vel * eff_dt
                sim_pos, sim_vel = self._reflect_bounds(prop_pos, sim_vel)

            return sim_pos

        return self.position.copy()

    def check_collision(self, center: np.ndarray, radius: float) -> bool:
        return check_circle_circle(self.position, self.radius, center, radius)

    def distance_to_surface(self, center: np.ndarray, radius: float = 0.0) -> float:
        return distance_circle_circle(self.position, self.radius, center, radius)

    def get_center(self) -> np.ndarray:
        return self.position.copy()

    @property
    def id(self) -> int:
        """Obstacle identifier."""
        return self.obs_id

    @id.setter
    def id(self, val: int) -> None:
        self.obs_id = int(val)

    def update(self, dt: float = 1.0) -> np.ndarray:
        """Advance obstacle state forward by dt (alias for step)."""
        return self.step(dt)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obs_id,
            "type": "moving",
            "pattern": self.pattern,
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "radius": float(self.radius),
            "velocity": [float(self.velocity[0]), float(self.velocity[1])],
            "center": [float(self.center[0]), float(self.center[1])],
            "angular_speed": float(self.angular_speed),
            "phase": float(self.phase),
            "random_seed": int(self.random_seed),
            "orbit_radius": float(self.orbit_radius),
        }


class ObstacleField:
    """Manages an aggregate collection of static and moving obstacles in the environment."""

    def __init__(
        self,
        obstacles: Optional[List[BaseObstacle]] = None,
        static_obstacles: Optional[List[BaseObstacle]] = None,
        moving_obstacles: Optional[List[MovingObstacle]] = None,
    ):
        all_obs = list(obstacles or [])
        if static_obstacles:
            all_obs.extend(static_obstacles)
        if moving_obstacles:
            all_obs.extend(moving_obstacles)
        self.obstacles: List[BaseObstacle] = all_obs
        self.moving_obstacles: List[MovingObstacle] = [
            o for o in self.obstacles if isinstance(o, MovingObstacle)
        ]

    def add_obstacle(self, obstacle: BaseObstacle) -> None:
        self.obstacles.append(obstacle)
        if isinstance(obstacle, MovingObstacle) and obstacle not in self.moving_obstacles:
            self.moving_obstacles.append(obstacle)

    def add_moving_obstacle(self, obstacle: MovingObstacle) -> None:
        self.add_obstacle(obstacle)

    def clear(self) -> None:
        self.obstacles.clear()
        self.moving_obstacles.clear()

    def step_moving_obstacles(self, dt: float = 1.0) -> None:
        """Advance all moving obstacles by timestep dt."""
        for obs in self.moving_obstacles:
            obs.step(dt)

    def reset_moving_obstacles(self) -> None:
        """Reset all moving obstacles to their initial conditions."""
        for obs in self.moving_obstacles:
            obs.reset()

    def get_moving_obstacles(self) -> List[MovingObstacle]:
        """Return list of moving obstacles in field."""
        return self.moving_obstacles.copy()

    def check_collision(
        self,
        position: np.ndarray,
        radius: float,
    ) -> Tuple[bool, List[BaseObstacle]]:
        """Check if position with given radius intersects any obstacle.

        Returns:
            Tuple of (has_collided, list_of_collided_obstacles)
        """
        collided = []
        for obs in self.obstacles:
            if obs.check_collision(position, radius):
                collided.append(obs)
        return (len(collided) > 0, collided)

    def check_swept_collision(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        radius: float,
    ) -> bool:
        """Check continuous swept path collision between p_start and p_end against all obstacles."""
        for obs in self.obstacles:
            if isinstance(obs, MovingObstacle):
                # For moving obstacles, if prev position known check swept circle
                prev_obs_pos = obs.history[-2] if len(obs.history) >= 2 else (obs.position[0], obs.position[1])
                collided, _, _ = predict_collision(
                    p_start,
                    p_end,
                    radius,
                    np.array(prev_obs_pos, dtype=float),
                    obs.position,
                    obs.radius,
                )
                if collided:
                    return True
            elif isinstance(obs, CircularObstacle):
                if check_swept_circle_line(
                    p_start, p_end, obs.center, obs.radius + radius
                ):
                    return True
            else:
                # Discretize path for rectangular obstacles
                steps = 5
                for t in np.linspace(0.0, 1.0, steps):
                    pt = p_start + t * (p_end - p_start)
                    if obs.check_collision(pt, radius):
                        return True
        return False

    def get_nearest_obstacle(
        self,
        position: np.ndarray,
        radius: float = 0.0,
    ) -> Tuple[Optional[BaseObstacle], float, np.ndarray]:
        """Find the nearest obstacle, clearance distance, and unit direction vector (obstacle -> drone).

        Returns:
            (nearest_obstacle, min_distance, unit_direction_from_obstacle)
        """
        if not self.obstacles:
            return None, float("inf"), np.zeros(2, dtype=float)

        min_dist = float("inf")
        nearest_obs: Optional[BaseObstacle] = None

        for obs in self.obstacles:
            dist = obs.distance_to_surface(position, radius)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs

        if nearest_obs is not None:
            diff = np.asarray(position, dtype=float) - nearest_obs.get_center()
            norm = np.linalg.norm(diff)
            unit_dir = diff / norm if norm > 1e-6 else np.array([1.0, 0.0])
        else:
            unit_dir = np.zeros(2, dtype=float)

        return nearest_obs, min_dist, unit_dir

    def get_nearest_moving_obstacle(
        self,
        position: np.ndarray,
        radius: float = 0.0,
    ) -> Tuple[Optional[MovingObstacle], float, np.ndarray]:
        """Find the nearest moving obstacle specifically."""
        if not self.moving_obstacles:
            return None, float("inf"), np.zeros(2, dtype=float)

        min_dist = float("inf")
        nearest_obs: Optional[MovingObstacle] = None

        for obs in self.moving_obstacles:
            dist = obs.distance_to_surface(position, radius)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs

        if nearest_obs is not None:
            diff = np.asarray(position, dtype=float) - nearest_obs.get_center()
            norm = np.linalg.norm(diff)
            unit_dir = diff / norm if norm > 1e-6 else np.array([1.0, 0.0])
        else:
            unit_dir = np.zeros(2, dtype=float)

        return nearest_obs, min_dist, unit_dir

    def get_nearby_obstacles(
        self,
        position: np.ndarray,
        search_radius: float,
        drone_radius: float = 0.0,
    ) -> List[Tuple[BaseObstacle, float]]:
        """Return list of (obstacle, distance) within search_radius."""
        nearby = []
        for obs in self.obstacles:
            dist = obs.distance_to_surface(position, drone_radius)
            if dist <= search_radius:
                nearby.append((obs, dist))
        nearby.sort(key=lambda x: x[1])
        return nearby

    def calculate_density(
        self,
        position: np.ndarray,
        radius: float,
    ) -> float:
        """Calculate local obstacle density in [0, 1] within a local radius around position."""
        if radius <= 0 or not self.obstacles:
            return 0.0

        circle_area = np.pi * (radius**2)
        total_obs_area = 0.0

        for obs in self.obstacles:
            center = obs.get_center()
            if distance_point_to_point(position, center) <= (
                radius + getattr(obs, "radius", 2.0)
            ):
                if isinstance(obs, (CircularObstacle, MovingObstacle)):
                    total_obs_area += np.pi * (obs.radius**2)
                elif isinstance(obs, RectangularObstacle):
                    dx = obs.bounds[2] - obs.bounds[0]
                    dy = obs.bounds[3] - obs.bounds[1]
                    total_obs_area += dx * dy

        return float(min(1.0, total_obs_area / circle_area))

    @property
    def static_obstacles(self) -> List[BaseObstacle]:
        """Return list of all static obstacles in the field."""
        return [obs for obs in self.obstacles if not isinstance(obs, MovingObstacle)]

    @classmethod
    def from_list(cls, obs_data: List[Dict[str, Any]]) -> "ObstacleField":
        """Factory method to construct ObstacleField from serialized dicts."""
        obstacles: List[BaseObstacle] = []
        for item in obs_data:
            shape_type = item.get("type", "circle").lower()
            obs_id = item.get("id", len(obstacles))
            if shape_type == "circle":
                obstacles.append(
                    CircularObstacle(
                        obs_id=obs_id,
                        x=float(item["x"]),
                        y=float(item["y"]),
                        radius=float(item["radius"]),
                    )
                )
            elif shape_type in ("rectangle", "box"):
                obstacles.append(
                    RectangularObstacle(
                        obs_id=obs_id,
                        x_min=float(item["x_min"]),
                        y_min=float(item["y_min"]),
                        x_max=float(item["x_max"]),
                        y_max=float(item["y_max"]),
                    )
                )
            elif shape_type == "moving":
                pos = item.get("position", [item.get("x", 10.0), item.get("y", 10.0)])
                obstacles.append(
                    MovingObstacle(
                        obs_id=obs_id,
                        position=pos,
                        radius=float(item.get("radius", 1.0)),
                        pattern=item.get("pattern", "linear"),
                        velocity=item.get("velocity", [0.5, 0.0]),
                        center=item.get("center", None),
                        orbit_radius=item.get("orbit_radius", None),
                        angular_speed=float(item.get("angular_speed", 0.5)),
                        phase=float(item.get("phase", 0.0)),
                        random_seed=int(item.get("random_seed", 42)),
                        bounds=tuple(item.get("bounds", (0.0, 0.0, 50.0, 50.0))),
                    )
                )
        return cls(obstacles)

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all obstacles to a list of dicts."""
        return [obs.to_dict() for obs in self.obstacles]

