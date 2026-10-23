"""Unit tests for UAV simulation environment, kinematics, obstacles, and collisions."""

import pytest
import numpy as np
from simulation.drone import Drone
from simulation.target import Target
from simulation.obstacles import (
    CircularObstacle,
    RectangularObstacle,
    ObstacleField,
)
from simulation.collision import (
    check_circle_circle,
    distance_circle_circle,
    check_circle_box,
    distance_circle_box,
    check_boundary_violation,
    distance_to_boundary,
    check_swept_circle_line,
)
from simulation.environment import UAVEnvironment


class TestDroneKinematics:
    """Tests for Drone kinematic state and trajectory tracking."""

    def test_drone_initialization(self):
        drone = Drone(start_position=(10.0, 15.0), speed=2.0, radius=0.8)
        assert np.allclose(drone.position, [10.0, 15.0])
        assert drone.speed == 0.0
        assert drone.radius == 0.8
        assert len(drone.trajectory) == 1

    def test_drone_movement(self):
        drone = Drone(start_position=(0.0, 0.0), speed=2.0)
        # Move East (1, 0)
        new_pos = drone.move(direction=np.array([1.0, 0.0]), speed=2.0)
        assert np.allclose(new_pos, [2.0, 0.0])
        assert np.allclose(drone.velocity, [2.0, 0.0])
        assert drone.step_count == 1
        assert len(drone.trajectory) == 2

    def test_drone_reset(self):
        drone = Drone(start_position=(5.0, 5.0), speed=1.5)
        drone.move(direction=np.array([0.0, 1.0]))
        assert drone.step_count == 1
        drone.reset(start_position=(2.0, 3.0))
        assert np.allclose(drone.position, [2.0, 3.0])
        assert drone.step_count == 0
        assert len(drone.trajectory) == 1


class TestObstaclesAndCollisions:
    """Tests for obstacle geometries and collision checking algorithms."""

    def test_circle_circle_collision(self):
        # Overlapping circles
        assert check_circle_circle(np.array([0.0, 0.0]), 2.0, np.array([3.0, 0.0]), 2.0) is True
        # Separated circles
        assert check_circle_circle(np.array([0.0, 0.0]), 1.0, np.array([5.0, 0.0]), 1.0) is False
        # Distance calculation
        dist = distance_circle_circle(np.array([0.0, 0.0]), 1.0, np.array([4.0, 0.0]), 1.0)
        assert np.isclose(dist, 2.0)

    def test_circle_box_collision(self):
        box_bounds = (10.0, 10.0, 20.0, 20.0)
        # Inside box
        assert check_circle_box(np.array([15.0, 15.0]), 1.0, box_bounds) is True
        # Touching boundary
        assert check_circle_box(np.array([9.5, 15.0]), 1.0, box_bounds) is True
        # Far outside
        assert check_circle_box(np.array([5.0, 5.0]), 1.0, box_bounds) is False

    def test_swept_circle_collision(self):
        circle_center = np.array([10.0, 10.0])
        radius = 2.0
        # Line passing directly through circle
        assert check_swept_circle_line(
            p_start=np.array([5.0, 10.0]),
            p_end=np.array([15.0, 10.0]),
            circle_center=circle_center,
            radius=radius,
        ) is True
        # Line passing far away
        assert check_swept_circle_line(
            p_start=np.array([5.0, 20.0]),
            p_end=np.array([15.0, 20.0]),
            circle_center=circle_center,
            radius=radius,
        ) is False

    def test_boundary_collision(self):
        # Inside
        assert check_boundary_violation(np.array([25.0, 25.0]), 1.0, 50.0, 50.0) is False
        # Outside left
        assert check_boundary_violation(np.array([0.5, 25.0]), 1.0, 50.0, 50.0) is True
        # Outside top
        assert check_boundary_violation(np.array([25.0, 49.5]), 1.0, 50.0, 50.0) is True
        # Distance to boundary
        assert np.isclose(distance_to_boundary(np.array([10.0, 20.0]), 50.0, 50.0), 10.0)

    def test_circular_obstacle_class(self):
        obs = CircularObstacle(obs_id=1, x=20.0, y=20.0, radius=4.0)
        assert np.allclose(obs.get_center(), [20.0, 20.0])
        assert obs.check_collision(np.array([20.0, 23.0]), radius=1.0) is True
        assert obs.check_collision(np.array([20.0, 30.0]), radius=1.0) is False
        clearance = obs.distance_to_surface(np.array([20.0, 30.0]), radius=1.0)
        assert np.isclose(clearance, 5.0)

    def test_rectangular_obstacle_class(self):
        obs = RectangularObstacle(obs_id=2, x_min=10.0, y_min=10.0, x_max=20.0, y_max=30.0)
        assert np.allclose(obs.get_center(), [15.0, 20.0])
        assert obs.check_collision(np.array([15.0, 20.0]), radius=0.8) is True
        assert obs.check_collision(np.array([5.0, 5.0]), radius=0.8) is False
        clearance = obs.distance_to_surface(np.array([5.0, 20.0]), radius=0.8)
        assert clearance > 0.0

    def test_boundary_margin(self):
        map_w, map_h = 50.0, 50.0
        drone_r = 0.8
        margin = 10.0
        # Position close to left boundary
        pos_edge = np.array([2.0, 25.0])
        dist = distance_to_boundary(pos_edge, map_w, map_h) - drone_r
        proximity = np.clip(dist / margin, 0.0, 1.0)
        assert 0.0 <= proximity < 0.2



class TestTargetAndEnvironment:
    """Tests for destination waypoint target detection and simulation step loop."""

    def test_target_detection(self):
        target = Target(position=(30.0, 40.0), detection_radius=2.0)
        assert target.is_reached(np.array([30.0, 41.0])) is True
        assert target.is_reached(np.array([10.0, 10.0])) is False
        unit_vec, dist = target.direction_from(np.array([30.0, 30.0]))
        assert np.isclose(dist, 10.0)
        assert np.allclose(unit_vec, [0.0, 1.0])

    def test_environment_step_and_reset(self):
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [
                {"id": 1, "type": "circle", "x": 20.0, "y": 20.0, "radius": 3.0}
            ],
        }
        scenario = {"start": [5.0, 5.0], "target": [45.0, 45.0], "max_steps": 50}

        env = UAVEnvironment(map_data=map_data, scenario_data=scenario)
        init_state = env.reset()
        assert np.allclose(init_state["drone_position"], [5.0, 5.0])
        assert env.current_step == 0

        # Step toward target
        next_state, reward, done, info = env.step(direction=np.array([1.0, 1.0]))
        assert env.current_step == 1
        assert done is False
        assert next_state["drone_position"][0] > 5.0
        assert next_state["drone_position"][1] > 5.0
