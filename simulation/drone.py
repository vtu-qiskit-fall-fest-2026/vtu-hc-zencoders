"""Drone Kinematics and State Representation for Q-Nav.

Maintains continuous 2D position, velocity vector, heading angle,
and full historical flight trajectory.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class Drone:
    """Represents the autonomous UAV in the 2D simulation environment."""

    def __init__(
        self,
        start_position: Tuple[float, float] = (5.0, 5.0),
        speed: float = 1.5,
        radius: float = 0.8,
    ):
        """Initialize drone at start position.

        Args:
            start_position: (x, y) initial coordinates.
            speed: Base translation speed per simulation step.
            radius: Geometric radius for collision detection.
        """
        self.radius = float(radius)
        self.base_speed = float(speed)
        self.start_position = np.array(start_position, dtype=float)

        # Dynamic state
        self.position = np.array(start_position, dtype=float)
        self.previous_position = np.array(start_position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.speed = 0.0
        self.heading = 0.0  # Radians, 0 points in +X direction
        self.step_count = 0

        # Trajectory history: list of snapshot dicts
        self.trajectory: List[Dict[str, Any]] = []
        self._record_trajectory_point()

    def reset(self, start_position: Optional[Tuple[float, float]] = None) -> None:
        """Reset drone to initial or newly specified start position.

        Args:
            start_position: Optional new start coordinates.
        """
        if start_position is not None:
            self.start_position = np.array(start_position, dtype=float)

        self.position = self.start_position.copy()
        self.previous_position = self.start_position.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.speed = 0.0
        self.heading = 0.0
        self.step_count = 0
        self.trajectory = []
        self._record_trajectory_point()

    def move(
        self,
        direction: np.ndarray,
        speed: Optional[float] = None,
        dt: float = 1.0,
    ) -> np.ndarray:
        """Apply a movement step along a given direction vector.

        Args:
            direction: 2D vector indicating desired motion direction.
            speed: Optional speed override. If None, uses base_speed.
            dt: Time step duration.

        Returns:
            The new 2D position [x, y].
        """
        dir_vec = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(dir_vec)

        self.previous_position = self.position.copy()

        if norm > 1e-6:
            unit_dir = dir_vec / norm
            effective_speed = self.base_speed if speed is None else float(speed)
            self.velocity = unit_dir * effective_speed
            self.speed = effective_speed
            self.heading = float(np.arctan2(unit_dir[1], unit_dir[0]))
            self.position = self.position + self.velocity * dt
        else:
            self.velocity = np.zeros(2, dtype=float)
            self.speed = 0.0

        self.step_count += 1
        self._record_trajectory_point()
        return self.position.copy()

    def _record_trajectory_point(self) -> None:
        """Record the current flight state into trajectory history."""
        self.trajectory.append(
            {
                "step": self.step_count,
                "x": float(self.position[0]),
                "y": float(self.position[1]),
                "vx": float(self.velocity[0]),
                "vy": float(self.velocity[1]),
                "speed": float(self.speed),
                "heading": float(self.heading),
            }
        )

    def get_trajectory_coords(self) -> List[Tuple[float, float]]:
        """Return list of (x, y) coordinates visited so far."""
        return [(point["x"], point["y"]) for point in self.trajectory]

    def get_state(self) -> Dict[str, Any]:
        """Return a complete dictionary summary of current drone state."""
        return {
            "position": self.position.copy(),
            "previous_position": self.previous_position.copy(),
            "velocity": self.velocity.copy(),
            "speed": self.speed,
            "heading": self.heading,
            "step_count": self.step_count,
            "radius": self.radius,
        }
