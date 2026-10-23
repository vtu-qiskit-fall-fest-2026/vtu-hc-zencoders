"""Simulation module for 2D autonomous UAV navigation."""

from simulation.drone import Drone
from simulation.obstacles import (
    BaseObstacle,
    CircularObstacle,
    RectangularObstacle,
    ObstacleField,
)
from simulation.target import Target
from simulation.collision import (
    check_circle_circle,
    distance_circle_circle,
    check_circle_box,
    distance_circle_box,
    check_swept_circle_line,
    check_boundary_violation,
    distance_to_boundary,
)
from simulation.environment import UAVEnvironment

__all__ = [
    "Drone",
    "BaseObstacle",
    "CircularObstacle",
    "RectangularObstacle",
    "ObstacleField",
    "Target",
    "check_circle_circle",
    "distance_circle_circle",
    "check_circle_box",
    "distance_circle_box",
    "check_swept_circle_line",
    "check_boundary_violation",
    "distance_to_boundary",
    "UAVEnvironment",
]
