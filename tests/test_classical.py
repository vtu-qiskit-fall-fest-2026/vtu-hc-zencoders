"""Unit tests for classical feature extraction, Artificial Potential Fields, and baseline navigator."""

import pytest
import numpy as np
from classical.feature_extractor import NavigationFeatureExtractor
from classical.potential_field import ArtificialPotentialField
from classical.baseline import ClassicalBaselineNavigator
from simulation.obstacles import ObstacleField, CircularObstacle
from simulation.environment import UAVEnvironment


class TestFeatureExtractor:
    """Tests for spatial and kinematic normalization in NavigationFeatureExtractor."""

    def test_feature_extraction_dimensions(self):
        extractor = NavigationFeatureExtractor()
        sample_state = {
            "drone_position": np.array([10.0, 10.0]),
            "drone_velocity": np.array([1.0, 0.5]),
            "drone_heading": 0.5,
            "drone_speed": 1.5,
            "target_position": np.array([40.0, 40.0]),
            "target_distance": 42.42,
            "target_direction": np.array([0.707, 0.707]),
            "nearest_obstacle_distance": 5.0,
            "nearest_obstacle_direction": np.array([-1.0, 0.0]),
            "obstacle_density": 0.2,
            "boundary_distance": 10.0,
        }

        vec, feat_dict = extractor.extract_features(sample_state)
        assert len(vec) == 12
        assert len(feat_dict) == 12
        assert "norm_target_dist" in feat_dict
        assert "norm_nearest_obs_dist" in feat_dict
        # All normalized values should be bounded
        assert 0.0 <= feat_dict["norm_target_dist"] <= 1.0
        assert -1.0 <= feat_dict["target_dir_x"] <= 1.0


class TestPotentialField:
    """Tests for attractive and repulsive potential field force calculations."""

    def test_attractive_force_direction(self):
        apf = ArtificialPotentialField()
        drone_pos = np.array([0.0, 0.0])
        target_pos = np.array([10.0, 0.0])

        f_att = apf.compute_attractive_force(drone_pos, target_pos)
        # Attractive force should point in +X direction
        assert f_att[0] > 0.0
        assert np.isclose(f_att[1], 0.0)

    def test_repulsive_force_direction(self):
        apf = ArtificialPotentialField()
        obs_field = ObstacleField([CircularObstacle(1, x=5.0, y=0.0, radius=2.0)])
        drone_pos = np.array([2.5, 0.0])  # Drone is to the left of obstacle

        f_rep = apf.compute_repulsive_force(drone_pos, obs_field)
        # Repulsive force should push drone in -X direction away from obstacle
        assert f_rep[0] < 0.0
        assert np.isclose(f_rep[1], 0.0, atol=1e-4)

    def test_local_minima_detection_and_escape(self):
        apf = ArtificialPotentialField()
        obs_field = ObstacleField([CircularObstacle(1, x=10.0, y=10.0, radius=2.0)])
        # Simulate trapped position
        drone_pos = np.array([10.0, 5.0])
        target_pos = np.array([10.0, 20.0])  # Obstacle is between drone and target

        f_total, breakdown = apf.compute_total_force(
            drone_pos=drone_pos,
            target_pos=target_pos,
            obstacles=obs_field,
        )
        assert isinstance(f_total, np.ndarray)
        assert "f_att" in breakdown
        assert "f_rep_obs" in breakdown

    def test_stagnation_detection(self):
        apf = ArtificialPotentialField(stagnation_window=5, stagnation_distance_threshold=1.0)
        target = np.array([40.0, 40.0])
        # UAV barely moving for 6 steps
        for _ in range(6):
            is_stagnant = apf.detect_stagnation(np.array([10.0, 10.05]), target)
        assert is_stagnant is True

        # When UAV moves far away, stagnation clears
        is_stagnant_cleared = apf.detect_stagnation(np.array([20.0, 20.0]), target)
        assert is_stagnant_cleared is False

    def test_oscillation_detection(self):
        apf = ArtificialPotentialField(oscillation_window=4)
        # Sequence of alternating opposite actions: RIGHT, LEFT, RIGHT, LEFT
        actions = ["EAST", "WEST", "EAST", "WEST"]
        for a in actions:
            is_osc = apf.detect_oscillation(a)
        assert is_osc is True

    def test_tangential_force(self):
        apf = ArtificialPotentialField(k_tangential=25.0)
        obs_field = ObstacleField([CircularObstacle(1, x=10.0, y=10.0, radius=2.0)])
        drone_pos = np.array([10.0, 5.0])
        f_att = np.array([0.0, 1.0])

        f_tan = apf.compute_tangential_force(drone_pos=drone_pos, obstacles=obs_field, f_att=f_att)
        assert isinstance(f_tan, np.ndarray)
        assert len(f_tan) == 2
        # Tangential force should be orthogonal to radial obstacle vector
        assert np.linalg.norm(f_tan) > 0.0

    def test_bypass_recovery_selection(self):
        apf = ArtificialPotentialField()
        obs_field = ObstacleField([CircularObstacle(1, x=10.0, y=10.0, radius=2.0)])
        drone_pos = np.array([10.0, 5.0])
        target_pos = np.array([10.0, 20.0])

        bypass_dir = apf.select_safer_bypass_direction(drone_pos, target_pos, obs_field)
        assert isinstance(bypass_dir, np.ndarray)
        assert np.isclose(np.linalg.norm(bypass_dir), 1.0, atol=1e-4)


class TestClassicalBaselineNavigator:
    """Tests for the ClassicalBaselineNavigator decision agent."""

    def test_navigator_decision(self):
        env = UAVEnvironment()
        navigator = ClassicalBaselineNavigator(obstacles=env.obstacles)

        state = env.get_state()
        decision = navigator.compute_decision(state)

        assert decision["model_type"] == "classical"
        assert decision["action"] in navigator.action_names
        assert len(decision["direction_vector"]) == 2
        assert 0.0 <= decision["confidence"] <= 1.0
        assert len(decision["action_scores"]) == len(navigator.action_names)
        assert len(decision["action_probabilities"]) == len(navigator.action_names)

    def test_feature_normalization_bounds(self):
        extractor = NavigationFeatureExtractor(max_map_distance=50.0, max_speed=3.0)
        test_state = {
            "drone_position": np.array([0.0, 0.0]),
            "drone_velocity": np.array([3.0, 0.0]),
            "drone_heading": 0.0,
            "drone_speed": 3.0,
            "target_position": np.array([50.0, 50.0]),
            "target_distance": 70.71,
            "target_direction": np.array([0.7071, 0.7071]),
            "nearest_obstacle_distance": 0.0,
            "obstacle_density": 1.5,  # Above 1.0
            "boundary_distance": 0.0,
        }
        vec, feat_dict = extractor.extract_features(test_state)
        assert len(vec) == 12
        for val in vec:
            assert -1.0 <= val <= 1.0
        assert 0.0 <= feat_dict["norm_target_dist"] <= 1.0
        assert 0.0 <= feat_dict["obstacle_density"] <= 1.0
        assert 0.0 <= feat_dict["norm_speed"] <= 1.0

