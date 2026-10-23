"""Simulation Environment for Q-Nav.

Provides the complete 2D UAV simulation loop, static and moving obstacle management,
boundary enforcement, reward calculation, and trajectory capture.
"""

from typing import Dict, Any, Tuple, Optional, List
from collections import deque
import numpy as np
from config import QNavConfig, DEFAULT_CONFIG
from simulation.drone import Drone
from simulation.obstacles import (
    ObstacleField,
    CircularObstacle,
    RectangularObstacle,
    MovingObstacle,
)
from simulation.target import Target
from simulation.collision import (
    check_boundary_violation,
    distance_to_boundary,
    predict_collision,
)
from utils.random_utils import set_seed


def verify_solvability(scenario_data: Dict[str, Any], grid_res: float = 1.0) -> Tuple[bool, str]:
    """Verify that a scenario target is geometrically reachable under obstacle constraints.

    Performs a 2D connectivity search (BFS) on a discretized clearance grid from start to target.

    Args:
        scenario_data: Dictionary containing start, target, bounds, and obstacles.
        grid_res: Resolution of grid discretization in coordinate units.

    Returns:
        Tuple of (is_solvable, message). If unsolvable, returns
        (False, "Scenario rejected: not verified solvable.").
    """
    bounds = scenario_data.get("bounds", [50.0, 50.0])
    w, h = float(bounds[0]), float(bounds[1])
    start = np.array(scenario_data.get("start", [2.0, 2.0]), dtype=float)
    target = np.array(scenario_data.get("target", [w - 2.0, h - 2.0]), dtype=float)

    # 1. Bounds check
    if not (0.0 <= start[0] <= w and 0.0 <= start[1] <= h):
        return False, "Scenario rejected: not verified solvable."
    if not (0.0 <= target[0] <= w and 0.0 <= target[1] <= h):
        return False, "Scenario rejected: not verified solvable."

    # 2. Check obstacles collision at start or target
    obstacles = scenario_data.get("obstacles", [])
    drone_r = 0.8
    for obs in obstacles:
        pos = obs.get("position", [obs.get("x", 0.0), obs.get("y", 0.0)])
        r = float(obs.get("radius", 1.0))
        d_start = np.linalg.norm(start - np.array(pos, dtype=float))
        d_target = np.linalg.norm(target - np.array(pos, dtype=float))
        if d_start <= (drone_r + r) or d_target <= (drone_r + r):
            return False, "Scenario rejected: not verified solvable."

    # 3. Discretize grid and test path connectivity with BFS
    cols = max(2, int(w / grid_res) + 1)
    rows = max(2, int(h / grid_res) + 1)

    start_cell = (
        int(np.clip(round(start[0] / grid_res), 0, cols - 1)),
        int(np.clip(round(start[1] / grid_res), 0, rows - 1)),
    )
    target_cell = (
        int(np.clip(round(target[0] / grid_res), 0, cols - 1)),
        int(np.clip(round(target[1] / grid_res), 0, rows - 1)),
    )

    # Mark blocked cells by obstacles (using static obstacles + initial dynamic positions)
    blocked = set()
    for obs in obstacles:
        pos = np.array(obs.get("position", [obs.get("x", 0.0), obs.get("y", 0.0)]), dtype=float)
        r = float(obs.get("radius", 1.0)) + drone_r
        min_c = max(0, int((pos[0] - r) / grid_res))
        max_c = min(cols - 1, int((pos[0] + r) / grid_res))
        min_r = max(0, int((pos[1] - r) / grid_res))
        max_r = min(rows - 1, int((pos[1] + r) / grid_res))
        for c in range(min_c, max_c + 1):
            for r_idx in range(min_r, max_r + 1):
                pt = np.array([c * grid_res, r_idx * grid_res])
                if np.linalg.norm(pt - pos) <= r:
                    blocked.add((c, r_idx))

    if start_cell in blocked or target_cell in blocked:
        return False, "Scenario rejected: not verified solvable."

    # 4. BFS graph traversal
    queue = deque([start_cell])
    visited = {start_cell}
    found = False

    while queue:
        curr = queue.popleft()
        if curr == target_cell:
            found = True
            break

        c, r_idx = curr
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            neighbor = (c + dc, r_idx + dr)
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                if neighbor not in visited and neighbor not in blocked:
                    visited.add(neighbor)
                    queue.append(neighbor)

    if not found:
        return False, "Scenario rejected: not verified solvable."

    return True, "Scenario verified solvable."


class UAVEnvironment:
    """The 2D UAV navigation environment simulating drone dynamics, static obstacles, and moving obstacles."""

    def __init__(
        self,
        config: Optional[QNavConfig] = None,
        map_data: Optional[Dict[str, Any]] = None,
        scenario_data: Optional[Dict[str, Any]] = None,
        dynamic_obstacles: bool = False,
    ):
        """Initialize the UAV simulation environment.

        Args:
            config: Master configuration. Defaults to DEFAULT_CONFIG.
            map_data: Optional dictionary describing obstacles and map dimensions.
            scenario_data: Optional dictionary with start, target, max_steps, and dynamic obstacles.
            dynamic_obstacles: Whether to activate dynamic moving obstacles. Defaults to False.
        """
        self.config = config or DEFAULT_CONFIG
        self.map_width = self.config.sim.map_width
        self.map_height = self.config.sim.map_height
        self.max_steps = self.config.sim.max_steps

        # Dynamic obstacles state
        self.dynamic_obstacles: bool = dynamic_obstacles
        self.moving_obstacles: List[MovingObstacle] = []
        self.sim_time: float = 0.0
        self.dynamic_telemetry: List[Dict[str, Any]] = []

        # Internal episode state
        self.current_step = 0
        self.done = False
        self.success = False
        self.collided = False
        self.out_of_bounds = False
        self.timed_out = False
        self.collision_count = 0
        self.total_reward = 0.0

        # Subsystems
        start_pos = (5.0, 5.0)
        target_pos = (45.0, 45.0)

        # Apply map data if provided
        if map_data:
            self._load_map_data(map_data)
        else:
            self.obstacles = ObstacleField()

        # Apply scenario data if provided
        if scenario_data:
            start_pos = tuple(scenario_data.get("start", start_pos))
            target_pos = tuple(scenario_data.get("target", target_pos))
            if "max_steps" in scenario_data:
                self.max_steps = int(scenario_data["max_steps"])
            if "bounds" in scenario_data:
                b = scenario_data["bounds"]
                self.map_width = float(b[0])
                self.map_height = float(b[1])
            if scenario_data.get("type") == "dynamic" or "obstacles" in scenario_data:
                self._load_scenario_obstacles(scenario_data)

        self.start_position = start_pos
        self.drone = Drone(
            start_position=start_pos,
            speed=self.config.sim.drone_speed,
            radius=self.config.sim.drone_radius,
        )
        self.target = Target(
            position=target_pos,
            detection_radius=self.config.sim.target_detection_radius,
        )

        set_seed(self.config.sim.random_seed)

    def _load_map_data(self, map_data: Dict[str, Any]) -> None:
        """Parse dimensions and obstacles from a map dictionary."""
        dims = map_data.get("dimensions", {})
        self.map_width = float(dims.get("width", self.config.sim.map_width))
        self.map_height = float(dims.get("height", self.config.sim.map_height))
        obs_list = map_data.get("obstacles", [])
        self.obstacles = ObstacleField.from_list(obs_list)
        self.moving_obstacles = self.obstacles.get_moving_obstacles()
        if self.moving_obstacles:
            self.dynamic_obstacles = True

    def _load_scenario_obstacles(self, scenario_data: Dict[str, Any]) -> None:
        """Parse static and moving obstacles defined in a scenario configuration."""
        obs_items = scenario_data.get("obstacles", [])
        for item in obs_items:
            item_type = item.get("type", "static").lower()
            obs_id = len(self.obstacles.obstacles)
            pos = item.get("position", [item.get("x", 10.0), item.get("y", 10.0)])
            radius = float(item.get("radius", 1.0))

            if item_type in ("static", "circle"):
                obs = CircularObstacle(obs_id=obs_id, x=pos[0], y=pos[1], radius=radius)
                self.obstacles.add_obstacle(obs)
            elif item_type in ("moving", "dynamic"):
                bounds = tuple(item.get("bounds", (0.0, 0.0, self.map_width, self.map_height)))
                m_obs = MovingObstacle(
                    obs_id=obs_id,
                    position=pos,
                    radius=radius,
                    pattern=item.get("pattern", "linear"),
                    velocity=item.get("velocity", [0.4, 0.0]),
                    center=item.get("center", None),
                    orbit_radius=item.get("orbit_radius", None),
                    angular_speed=float(item.get("angular_speed", 0.5)),
                    phase=float(item.get("phase", 0.0)),
                    random_seed=int(item.get("random_seed", 42)),
                    bounds=bounds,
                )
                self.obstacles.add_obstacle(m_obs)
                self.moving_obstacles.append(m_obs)
                self.dynamic_obstacles = True

    def reset(
        self,
        start_position: Optional[Tuple[float, float]] = None,
        target_position: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Resets UAV, static and moving obstacles, simulation time, and dynamic telemetry.
        The exact same scenario and seed reproduce the identical obstacle trajectories.

        Args:
            start_position: Optional new start coordinates.
            target_position: Optional new target coordinates.

        Returns:
            Initial state dictionary.
        """
        self.current_step = 0
        self.sim_time = 0.0
        self.done = False
        self.success = False
        self.collided = False
        self.out_of_bounds = False
        self.timed_out = False
        self.collision_count = 0
        self.total_reward = 0.0
        self.dynamic_telemetry.clear()

        # Reset moving obstacles
        for m_obs in self.moving_obstacles:
            m_obs.reset()

        if start_position is not None:
            self.start_position = start_position

        self.drone.reset(start_position=self.start_position)

        if target_position is not None:
            self.target.reset(position=target_position)
        else:
            self.target.reset()

        return self.get_state()

    def step(
        self,
        direction: Optional[Any] = None,
        speed: Optional[float] = None,
        action_idx: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance the simulation by one step along desired direction vector or action index.

        Follows the 11-step lifecycle:
        1. Obtain current UAV state
        2. Obtain current obstacle states
        3. Calculate navigation features
        4. Generate classical/quantum action (upstream controller)
        5. Apply safety filtering (upstream controller)
        6. Move UAV
        7. Advance moving obstacles
        8. Perform collision detection
        9. Update simulation time
        10. Record telemetry
        11. Check target reached / episode termination

        Args:
            direction: 2D direction vector or integer action index.
            speed: Optional speed override.
            action_idx: Optional explicit integer action index (0 to 8).

        Returns:
            Tuple of (state_dict, reward, done, info_dict)
        """
        if self.done:
            return self.get_state(), 0.0, True, self.get_info()

        # Support action_idx or integer direction
        if isinstance(direction, (int, np.integer)):
            action_idx = int(direction)
            direction = None

        if action_idx is not None:
            # Map action_idx to unit vector according to action_names or fallback dir_map
            if 0 <= action_idx < len(self.config.navigation.action_names):
                a_name = self.config.navigation.action_names[action_idx]
                direction = np.array(self.config.navigation.action_vectors[a_name], dtype=float)
            else:
                inv_sqrt2 = 1.0 / np.sqrt(2.0)
                dir_map = {
                    0: (0.0, 0.0),
                    1: (0.0, 1.0),
                    2: (inv_sqrt2, inv_sqrt2),
                    3: (1.0, 0.0),
                    4: (inv_sqrt2, -inv_sqrt2),
                    5: (0.0, -1.0),
                    6: (-inv_sqrt2, -inv_sqrt2),
                    7: (-1.0, 0.0),
                    8: (-inv_sqrt2, inv_sqrt2),
                }
                dir_vec = dir_map.get(action_idx, (0.0, 0.0))
                direction = np.array(dir_vec, dtype=float)
        elif direction is None:
            direction = np.zeros(2, dtype=float)
        else:
            direction = np.asarray(direction, dtype=float)

        self.current_step += 1
        dt = self.config.sim.dt
        prev_pos = self.drone.position.copy()
        prev_dist_target = self.target.distance_to(prev_pos)

        # 6. Move UAV
        new_pos = self.drone.move(direction=direction, speed=speed, dt=dt)

        # 7. Advance moving obstacles
        obs_prev_positions: Dict[int, np.ndarray] = {}
        for m_obs in self.moving_obstacles:
            obs_prev_positions[m_obs.obs_id] = m_obs.position.copy()
            m_obs.step(dt)

        # 8. Perform collision detection
        # A. Static obstacle collisions (current + swept)
        has_collided, _ = self.obstacles.check_collision(
            new_pos, self.config.sim.collision_radius
        )
        swept_collided = self.obstacles.check_swept_collision(
            prev_pos, new_pos, self.config.sim.collision_radius
        )

        # B. Moving obstacle collisions (endpoint + swept relative motion)
        moving_collided = False
        nearest_m_dist = float("inf")
        nearest_m_obs: Optional[MovingObstacle] = None

        for m_obs in self.moving_obstacles:
            p_obs_prev = obs_prev_positions.get(m_obs.obs_id, m_obs.position)
            p_obs_curr = m_obs.position

            # Continuous relative swept check
            c_collided, min_c_dist, _ = predict_collision(
                prev_pos,
                new_pos,
                self.drone.radius,
                p_obs_prev,
                p_obs_curr,
                m_obs.radius,
            )
            surf_dist = m_obs.distance_to_surface(new_pos, self.drone.radius)
            if surf_dist < nearest_m_dist:
                nearest_m_dist = surf_dist
                nearest_m_obs = m_obs

            if c_collided or m_obs.check_collision(new_pos, self.config.sim.collision_radius):
                moving_collided = True

        step_collided = has_collided or swept_collided or moving_collided
        if step_collided:
            self.collided = True
            self.collision_count += 1

        # C. Boundary collision
        is_oob = check_boundary_violation(
            new_pos, self.drone.radius, self.map_width, self.map_height
        )
        if is_oob:
            self.out_of_bounds = True

        # 9. Update simulation time
        self.sim_time += dt

        # 11. Check target reached and step limit
        is_target_reached = self.target.is_reached(new_pos)
        if is_target_reached:
            self.success = True

        if self.current_step >= self.max_steps:
            self.timed_out = True

        # Calculate reward
        curr_dist_target = self.target.distance_to(new_pos)
        progress_reward = (prev_dist_target - curr_dist_target) * 2.0
        step_penalty = -0.1

        reward = progress_reward + step_penalty

        if step_collided:
            reward -= 5.0

        if is_oob:
            reward -= 10.0

        if is_target_reached:
            reward += 100.0
            self.done = True
        elif is_oob or self.timed_out:
            self.done = True

        self.total_reward += reward

        info = self.get_info()
        return self.get_state(), reward, self.done, info

    def get_state(self) -> Dict[str, Any]:
        """Aggregate current environment observations into a dictionary."""
        pos = self.drone.position.copy()
        target_pos = self.target.position.copy()
        target_unit, target_dist = self.target.direction_from(pos)
        nearest_obs, min_obs_dist, obs_unit = self.obstacles.get_nearest_obstacle(
            pos, radius=self.drone.radius
        )
        b_dist = distance_to_boundary(pos, self.map_width, self.map_height)
        density = self.obstacles.calculate_density(
            pos, radius=self.config.classical.obstacle_influence_distance
        )

        nearest_m_obs, min_m_dist, m_unit = self.obstacles.get_nearest_moving_obstacle(
            pos, radius=self.drone.radius
        )

        return {
            "step": self.current_step,
            "sim_time": self.sim_time,
            "position": pos,
            "drone_position": pos,
            "drone_velocity": self.drone.velocity.copy(),
            "drone_heading": self.drone.heading,
            "drone_speed": self.drone.speed,
            "target_position": target_pos,
            "target_distance": target_dist,
            "target_direction": target_unit,
            "nearest_obstacle": nearest_obs,
            "nearest_obstacle_distance": min_obs_dist,
            "nearest_obstacle_direction": obs_unit,
            "nearest_moving_obstacle": nearest_m_obs,
            "nearest_moving_obstacle_distance": min_m_dist,
            "nearest_moving_obstacle_direction": m_unit,
            "obstacle_density": density,
            "boundary_distance": b_dist,
            "map_width": self.map_width,
            "map_height": self.map_height,
            "obstacles": self.obstacles,
            "dynamic_obstacles": self.dynamic_obstacles,
            "moving_obstacles": self.moving_obstacles,
        }

    def get_info(self) -> Dict[str, Any]:
        """Return episode metadata and outcome metrics."""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "sim_time": self.sim_time,
            "success": self.success,
            "collided": self.collided,
            "collision_count": self.collision_count,
            "out_of_bounds": self.out_of_bounds,
            "timed_out": self.timed_out,
            "total_reward": self.total_reward,
            "path_length": self.get_path_length(),
            "final_distance_to_target": float(
                self.target.distance_to(self.drone.position)
            ),
            "dynamic_obstacles": self.dynamic_obstacles,
            "moving_obstacles_count": len(self.moving_obstacles),
        }

    def _get_observation(self) -> Dict[str, Any]:
        """Alias for get_state() returning current environment observation dictionary."""
        return self.get_state()

    def get_path_length(self) -> float:
        """Calculate the total Euclidean distance traveled by the drone so far."""
        coords = self.drone.get_trajectory_coords()
        if len(coords) < 2:
            return 0.0
        pts = np.array(coords)
        diffs = pts[1:] - pts[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(segment_lengths))


Environment = UAVEnvironment


def generate_random_obstacles(
    n_obstacles: int = 20,
    map_width: float = 50.0,
    map_height: float = 50.0,
    min_radius: float = 1.0,
    max_radius: float = 2.5,
    seed: Optional[int] = None,
) -> ObstacleField:
    """Generate an ObstacleField populated with random circular obstacles.

    Args:
        n_obstacles: Number of obstacles to generate.
        map_width: Environment width in meters.
        map_height: Environment height in meters.
        min_radius: Minimum obstacle radius.
        max_radius: Maximum obstacle radius.
        seed: Optional RNG seed.

    Returns:
        ObstacleField with generated obstacles.
    """
    rng = np.random.default_rng(seed)
    field = ObstacleField()
    for i in range(n_obstacles):
        x = float(rng.uniform(6.0, map_width - 6.0))
        y = float(rng.uniform(6.0, map_height - 6.0))
        r = float(rng.uniform(min_radius, max_radius))
        field.add_obstacle(CircularObstacle(obs_id=i, x=x, y=y, radius=r))
    return field

