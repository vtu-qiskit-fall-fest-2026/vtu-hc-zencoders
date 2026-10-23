"""Target/Waypoint Representation for Q-Nav.

Defines the destination position, detection threshold, and vector geometry.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from simulation.collision import distance_point_to_point


class Target:
    """Represents the navigation objective / destination for the UAV."""

    def __init__(
        self,
        position: Tuple[float, float] = (45.0, 45.0),
        detection_radius: float = 2.0,
    ):
        """Initialize target.

        Args:
            position: (x, y) target coordinates.
            detection_radius: Acceptance radius within which target is reached.
        """
        self.position = np.array(position, dtype=float)
        self.detection_radius = float(detection_radius)
        self.reached = False

    def reset(self, position: Optional[Tuple[float, float]] = None) -> None:
        """Reset target state or move to new coordinates."""
        if position is not None:
            self.position = np.array(position, dtype=float)
        self.reached = False

    def distance_to(self, position: np.ndarray) -> float:
        """Euclidean distance from a given position to the target center."""
        return distance_point_to_point(position, self.position)

    def direction_from(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute the unit direction vector and distance from position to target.

        Returns:
            Tuple of (unit_vector, distance)
        """
        diff = self.position - np.asarray(position, dtype=float)
        dist = float(np.linalg.norm(diff))
        if dist > 1e-6:
            unit_vec = diff / dist
        else:
            unit_vec = np.zeros(2, dtype=float)
        return unit_vec, dist

    def is_reached(self, position: np.ndarray) -> bool:
        """Determine whether the given position is within detection radius."""
        dist = self.distance_to(position)
        if dist <= self.detection_radius:
            self.reached = True
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize target state."""
        return {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "detection_radius": self.detection_radius,
            "reached": self.reached,
        }
