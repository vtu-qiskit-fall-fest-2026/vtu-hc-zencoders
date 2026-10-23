"""Comprehensive Unit Tests for Dynamic / Moving Obstacle Navigation.

Validates:
1. MovingObstacle patterns (linear, circular, random_walk)
2. Boundary reflection and deterministic reproducibility
3. State mutation-free trajectory prediction
4. Swept-circle predictive collision detection
5. Dynamic UAVEnvironment integration and solvability verification
6. Proactive dynamic safety layer and emergency STAY fallback
7. 12D feature vector invariance
8. Multi-model comparative benchmarking and dynamic telemetry generation
"""

import pytest
import numpy as np
from pathlib import Path

from simulation.obstacles import MovingObstacle, CircularObstacle, ObstacleField
from simulation.collision import (
    check_circle_circle,
    distance_circle_circle,
    predict_collision,
    check_boundary_violation,
    check_swept_circle_circle,
)
from simulation.environment import UAVEnvironment, verify_solvability
from navigation.safety import SafetyLayer
from classical.feature_extractor import NavigationFeatureExtractor
from config import QNavConfig, DYNAMIC_TELEMETRY_PATH
from evaluation.compare_models import ModelComparator
from utils.random_utils import set_seed


class TestMovingObstacle:
    """Tests for the MovingObstacle class and motion patterns."""

    def test_initialization(self):
        obs = MovingObstacle(
            position=[10.0, 15.0],
            radius=1.5,
            pattern="linear",
            velocity=[0.5, -0.2],
            center=[10.0, 15.0],
            orbit_radius=4.0,
            angular_speed=0.3,
            phase=0.5,
            random_seed=42,
            bounds=[0.0, 0.0, 50.0, 50.0],
            obstacle_id=7,
        )
        assert obs.id == 7
        assert np.allclose(obs.position, [10.0, 15.0])
        assert np.allclose(obs.initial_position, [10.0, 15.0])
        assert obs.radius == 1.5
        assert obs.pattern == "linear"
        assert np.allclose(obs.velocity, [0.5, -0.2])
        assert np.allclose(obs.initial_velocity, [0.5, -0.2])
        assert obs.t == 0.0
        assert np.allclose(obs.center, [10.0, 15.0])
        assert obs.orbit_radius == 4.0
        assert obs.angular_speed == 0.3
        assert obs.phase == 0.5
        assert obs.random_seed == 42

    def test_reset_restores_initial_state(self):
        obs = MovingObstacle(
            position=[5.0, 5.0],
            radius=1.0,
            pattern="linear",
            velocity=[1.0, 0.0],
            random_seed=123,
        )
        # Advance obstacle several steps
        for _ in range(10):
            obs.update(dt=0.5)

        assert not np.allclose(obs.position, [5.0, 5.0])
        assert obs.t > 0.0

        # Reset
        obs.reset()
        assert np.allclose(obs.position, [5.0, 5.0])
        assert np.allclose(obs.velocity, [1.0, 0.0])
        assert obs.t == 0.0

    def test_linear_movement_step(self):
        obs = MovingObstacle(
            position=[10.0, 10.0],
            radius=1.0,
            pattern="linear",
            velocity=[2.0, -1.0],
            bounds=[0.0, 0.0, 50.0, 50.0],
        )
        new_pos = obs.update(dt=0.5)
        # 10 + 2*0.5 = 11.0; 10 - 1*0.5 = 9.5
        assert np.allclose(new_pos, [11.0, 9.5])
        assert np.allclose(obs.position, [11.0, 9.5])
        assert obs.t == pytest.approx(0.5)

    def test_linear_boundary_reflection(self):
        # Obstacle placed close to upper X boundary
        obs = MovingObstacle(
            position=[49.0, 20.0],
            radius=1.0,
            pattern="linear",
            velocity=[2.0, 0.0],
            bounds=[0.0, 0.0, 50.0, 50.0],
        )
        # Update by 1.0s: proposed pos = [51.0, 20.0] -> crosses max_x (50 - 1 = 49)
        obs.update(dt=1.0)
        # Velocity in X must reflect to negative
        assert obs.velocity[0] < 0.0
        # Position clamped within bounds
        assert obs.position[0] <= 49.0

    def test_circular_movement(self):
        center = [25.0, 25.0]
        orbit_r = 5.0
        obs = MovingObstacle(
            position=[30.0, 25.0],
            radius=1.5,
            pattern="circular",
            center=center,
            orbit_radius=orbit_r,
            angular_speed=np.pi / 2.0,  # 90 degrees per second
            phase=0.0,
        )
        # At t=0: x = 25 + 5*cos(0) = 30.0, y = 25 + 5*sin(0) = 25.0
        assert np.allclose(obs.position, [30.0, 25.0])

        # Step by 1.0 second: angle theta = pi/2
        obs.update(dt=1.0)
        # At theta = pi/2: x = 25 + 5*0 = 25.0, y = 25 + 5*1 = 30.0
        assert np.allclose(obs.position, [25.0, 30.0], atol=1e-5)
        # Verify physical radius is distinct from orbit radius
        assert obs.radius == 1.5
        assert obs.orbit_radius == 5.0

    def test_deterministic_random_walk(self):
        # Two independent instances with identical random_seed must produce identical paths
        obs1 = MovingObstacle(
            position=[20.0, 20.0],
            radius=1.0,
            pattern="random_walk",
            velocity=[0.5, 0.0],
            random_seed=777,
        )
        obs2 = MovingObstacle(
            position=[20.0, 20.0],
            radius=1.0,
            pattern="random_walk",
            velocity=[0.5, 0.0],
            random_seed=777,
        )

        for _ in range(15):
            p1 = obs1.update(dt=0.2)
            p2 = obs2.update(dt=0.2)
            assert np.allclose(p1, p2)
            assert np.allclose(obs1.velocity, obs2.velocity)

    def test_prediction_without_state_mutation(self):
        obs = MovingObstacle(
            position=[10.0, 10.0],
            radius=1.0,
            pattern="linear",
            velocity=[1.0, 2.0],
            bounds=[0.0, 0.0, 50.0, 50.0],
        )
        initial_pos = obs.position.copy()
        initial_t = obs.t
        initial_vel = obs.velocity.copy()

        # Predict future position at horizon 1.5s
        predicted = obs.predict_position(horizon=1.5, dt=0.1)
        expected = initial_pos + initial_vel * 1.5
        assert np.allclose(predicted, expected)

        # Verify internal state was NOT mutated
        assert np.allclose(obs.position, initial_pos)
        assert obs.t == initial_t
        assert np.allclose(obs.velocity, initial_vel)


class TestCollisionDetection:
    """Tests for static, moving, swept, and predictive collision detection."""

    def test_predict_collision_approaching(self):
        # UAV at (0, 0) moving East towards (10, 0)
        uav_start = (0.0, 0.0)
        uav_end = (6.0, 0.0)
        uav_radius = 1.0

        # Obstacle at (10, 0) moving West towards (4, 0)
        obs_start = (10.0, 0.0)
        obs_vel = (-6.0, 0.0)
        obs_radius = 1.0

        # Head-on collision during the timestep
        collided, min_dist, _ = predict_collision(
            uav_pos=uav_start,
            uav_next_pos=uav_end,
            uav_radius=uav_radius,
            obs_pos=obs_start,
            obs_velocity=obs_vel,
            obs_radius=obs_radius,
            dt=1.0,
            safety_margin=0.2,
        )
        assert collided is True
        assert min_dist <= (uav_radius + obs_radius + 0.2)

    def test_predict_collision_parallel_safe(self):
        # Parallel motion with large vertical clearance
        uav_start = (0.0, 0.0)
        uav_end = (5.0, 0.0)
        obs_start = (0.0, 15.0)
        obs_vel = (5.0, 0.0)

        collided, min_dist, _ = predict_collision(
            uav_pos=uav_start,
            uav_next_pos=uav_end,
            uav_radius=1.0,
            obs_pos=obs_start,
            obs_velocity=obs_vel,
            obs_radius=1.0,
            dt=1.0,
            safety_margin=0.5,
        )
        assert collided is False
        assert min_dist >= 12.0

    def test_swept_circle_circle(self):
        # Test swept collision where path intersects moving obstacle
        collided = check_swept_circle_circle(
            p1_start=np.array([0.0, 0.0]),
            p1_end=np.array([10.0, 0.0]),
            r1=1.0,
            p2_start=np.array([5.0, 5.0]),
            p2_end=np.array([5.0, -5.0]),
            r2=1.0,
        )
        # Paths cross perpendicularly at (5, 0) at tau = 0.5
        assert collided is True


class TestEnvironmentDynamicIntegration:
    """Tests for UAVEnvironment with dynamic obstacles and solvability verification."""

    @pytest.fixture
    def dynamic_scenario_data(self):
        return {
            "name": "test_dynamic",
            "type": "dynamic",
            "bounds": [50.0, 50.0],
            "start": [5.0, 5.0],
            "target": [45.0, 45.0],
            "max_steps": 100,
            "obstacles": [
                {
                    "id": 0,
                    "type": "static",
                    "position": [15.0, 15.0],
                    "radius": 2.0,
                },
                {
                    "id": 1,
                    "type": "moving",
                    "pattern": "linear",
                    "position": [25.0, 10.0],
                    "radius": 1.5,
                    "velocity": [0.0, 1.0],
                },
            ],
        }

    def test_environment_loads_dynamic_obstacles(self, dynamic_scenario_data):
        env = UAVEnvironment(scenario_data=dynamic_scenario_data, dynamic_obstacles=True)
        assert env.dynamic_obstacles is True
        assert len(env.moving_obstacles) == 1
        assert len(env.obstacles.moving_obstacles) == 1
        assert len(env.obstacles.static_obstacles) == 1

        # Check moving obstacle properties
        m_obs = env.moving_obstacles[0]
        assert m_obs.pattern == "linear"
        assert np.allclose(m_obs.position, [25.0, 10.0])

    def test_environment_step_advances_moving_obstacles(self, dynamic_scenario_data):
        env = UAVEnvironment(scenario_data=dynamic_scenario_data, dynamic_obstacles=True)
        env.reset()

        initial_obs_pos = env.moving_obstacles[0].position.copy()
        # Step environment
        env.step(action_idx=1)  # Move North

        new_obs_pos = env.moving_obstacles[0].position
        # Velocity is [0, 1], so Y should have increased
        assert new_obs_pos[1] > initial_obs_pos[1]
        assert env.sim_time > 0.0

    def test_environment_reset_restores_moving_obstacles(self, dynamic_scenario_data):
        env = UAVEnvironment(scenario_data=dynamic_scenario_data, dynamic_obstacles=True)
        env.reset()
        for _ in range(5):
            env.step(action_idx=1)

        assert env.sim_time > 0.0
        assert not np.allclose(env.moving_obstacles[0].position, [25.0, 10.0])

        env.reset()
        assert env.sim_time == 0.0
        assert np.allclose(env.moving_obstacles[0].position, [25.0, 10.0])

    def test_solvability_verification(self, dynamic_scenario_data):
        # Open scenario should be verifiable
        is_solvable, msg = verify_solvability(dynamic_scenario_data)
        assert is_solvable is True

        # Scenario completely walled off from start to target
        blocked_scenario = {
            "bounds": [20.0, 20.0],
            "start": [2.0, 2.0],
            "target": [18.0, 18.0],
            "obstacles": [
                # Dense solid wall of obstacles across x=10
                {"type": "static", "position": [10.0, y], "radius": 2.0}
                for y in np.linspace(0, 20, 15)
            ],
        }
        is_solv, b_msg = verify_solvability(blocked_scenario)
        assert is_solv is False
        assert "not verified solvable" in b_msg.lower() or "blocked" in b_msg.lower()

    def test_static_compatibility_remains_intact(self):
        # Standard static environment with dynamic_obstacles=False must work identically
        static_scenario = {
            "name": "static_test",
            "start": [5.0, 5.0],
            "target": [40.0, 40.0],
            "obstacles": [{"type": "static", "position": [20.0, 20.0], "radius": 3.0}],
        }
        env = UAVEnvironment(scenario_data=static_scenario, dynamic_obstacles=False)
        assert env.dynamic_obstacles is False
        assert len(env.moving_obstacles) == 0
        state = env.reset()
        assert "position" in state
        assert "nearest_obstacle_distance" in state


class TestProactiveSafetyLayer:
    """Tests for predictive collision checking and action filtering."""

    def test_rejects_action_into_moving_obstacle(self):
        config = QNavConfig()
        config.navigation.prediction_horizon = 1.0
        config.navigation.prediction_dt = 0.1
        config.navigation.dynamic_safety_margin = 0.5

        # Drone at (10, 10), Moving obstacle approaching at (12, 10) with velocity [-1.5, 0]
        moving_obs = MovingObstacle(
            position=[12.0, 10.0],
            radius=1.0,
            pattern="linear",
            velocity=[-1.5, 0.0],
        )
        field = ObstacleField(static_obstacles=[], moving_obstacles=[moving_obs])
        safety = SafetyLayer(config=config, obstacles=field)

        state = {
            "position": np.array([10.0, 10.0]),
            "velocity": np.array([0.0, 0.0]),
            "speed": 1.5,
            "heading": 0.0,
            "moving_obstacles": [moving_obs],
        }

        # Action 3 (East): moves drone towards (11.5, 10.0), directly colliding with incoming obstacle
        safe_actions = safety.get_safe_actions(state)
        # Action 3 (East: [1, 0]) should be rejected due to predictive collision
        assert 3 not in safe_actions

        # Action 1 (North) moves out of the obstacle path and is safe
        assert 1 in safe_actions

    def test_emergency_stay_when_constrained(self):
        config = QNavConfig()
        # Create an obstacle field with a safety layer
        safety = SafetyLayer(config=config)
        state = {
            "position": np.array([10.0, 10.0]),
            "velocity": np.array([0.0, 0.0]),
            "speed": 1.5,
            "heading": 0.0,
        }
        # Validate that select_safe_action returns an integer STAY (0) and never None
        filtered = safety.select_safe_action(proposed_action=3, state=state, safe_actions=[0])
        assert isinstance(filtered, int)
        assert filtered == 0


class Test12DFeatureVectorInvariance:
    """Verifies that the existing 12D quantum feature vector is strictly preserved."""

    def test_12d_feature_vector_structure(self):
        extractor = NavigationFeatureExtractor()
        # Verify exactly 12 standard feature names in order
        assert len(extractor.FEATURE_NAMES) == 12
        assert extractor.FEATURE_NAMES == [
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

        # Extract features with mock state
        state = {
            "drone_position": np.array([10.0, 10.0]),
            "drone_heading": 0.0,
            "target_position": np.array([45.0, 45.0]),
            "velocity": np.array([1.0, 0.0]),
            "distance_to_target": 49.5,
            "nearest_obstacle_distance": 5.0,
            "obstacle_density": 0.2,
            "boundary_distance": 10.0,
        }
        feat_vec, feat_dict = extractor.extract_features(state)
        assert isinstance(feat_vec, np.ndarray)
        assert feat_vec.shape == (12,)
        assert np.all(feat_vec >= 0.0) and np.all(feat_vec <= 1.0)
        assert len(feat_dict) == 12


class TestDynamicBenchmarking:
    """Tests 3-model dynamic scenario execution and dynamic telemetry output."""

    def test_run_dynamic_scenario_benchmark(self):
        config = QNavConfig()
        comparator = ModelComparator(config=config)

        scenario = {
            "id": "test_dyn_scn",
            "name": "Test Dynamic Corridor",
            "type": "dynamic",
            "bounds": [50.0, 50.0],
            "start": [5.0, 5.0],
            "target": [40.0, 40.0],
            "max_steps": 30,
            "obstacles": [
                {"id": 0, "type": "static", "position": [15.0, 15.0], "radius": 2.0},
                {
                    "id": 1,
                    "type": "moving",
                    "pattern": "linear",
                    "position": [20.0, 20.0],
                    "radius": 1.5,
                    "velocity": [0.5, -0.5],
                },
            ],
        }

        # Run benchmark with include_hybrid=True on matching seed
        res = comparator.run_benchmark(
            scenarios=[scenario],
            maps_by_id={},
            seeds=[42],
            save_outputs=True,
            is_dynamic=True,
            include_hybrid=True,
        )

        assert "classical" in res["summary"]
        assert "quantum" in res["summary"]
        assert "hybrid" in res["summary"]

        # Verify dynamic telemetry file was created and is non-empty
        assert DYNAMIC_TELEMETRY_PATH.exists()
        assert DYNAMIC_TELEMETRY_PATH.stat().st_size > 0

    def test_all_five_dynamic_scenarios_exist_and_solvable(self):
        """Verify that all 5 dynamic benchmark scenarios exist and pass reachability checks."""
        from config import SCENARIOS_DIR
        from utils.io_utils import load_json

        expected_ids = [
            "scenario_dynamic_01",
            "scenario_dynamic_02",
            "scenario_dynamic_03",
            "scenario_dynamic_04",
            "scenario_dynamic_05",
        ]
        found_scenarios = []
        for sid in expected_ids:
            scn_path = SCENARIOS_DIR / f"{sid}.json"
            assert scn_path.exists(), f"Dynamic scenario file missing: {scn_path}"
            data = load_json(scn_path)
            assert data.get("id") == sid or data.get("type") == "dynamic"
            # Verify obstacle definitions
            obs_list = data.get("obstacles", [])
            assert len(obs_list) > 0, f"Scenario {sid} has no obstacles defined"
            has_moving = any(o.get("type") == "moving" for o in obs_list)
            assert has_moving, f"Scenario {sid} lacks moving obstacles"

            # Verify solvability
            solv_ok, solv_msg = verify_solvability(data)
            assert solv_ok, f"Scenario {sid} failed solvability: {solv_msg}"
            found_scenarios.append(data)

        assert len(found_scenarios) == 5

    def test_telemetry_fields_and_action_divergence(self):
        """Verify that telemetry records contain required action fields and show action divergence."""
        from config import SCENARIOS_DIR
        from utils.io_utils import load_json
        from simulation.environment import UAVEnvironment
        from quantum.quantum_decision import QuantumDecisionMaker
        from navigation.controller import NavigationController

        config = QNavConfig()
        scn_path = SCENARIOS_DIR / "scenario_dynamic_02.json"
        scn_data = load_json(scn_path)

        bounds = scn_data.get("bounds", [50.0, 50.0])
        map_data = {
            "map_id": "test_map",
            "dimensions": {"width": float(bounds[0]), "height": float(bounds[1])},
            "obstacles": [],
        }

        # 1. Run Pure Quantum
        env_q = UAVEnvironment(config=config, map_data=map_data, scenario_data=scn_data, dynamic_obstacles=True)
        nav_q = QuantumDecisionMaker(config=config)
        ctrl_q = NavigationController(environment=env_q, navigator=nav_q, config=config, is_hybrid=False)
        ctrl_q.reset(start_position=tuple(scn_data["start"]), target_position=tuple(scn_data["target"]))
        res_q = ctrl_q.run_episode(max_steps=25)

        # 2. Run Hybrid
        env_h = UAVEnvironment(config=config, map_data=map_data, scenario_data=scn_data, dynamic_obstacles=True)
        nav_h = QuantumDecisionMaker(config=config)
        ctrl_h = NavigationController(environment=env_h, navigator=nav_h, config=config, is_hybrid=True)
        ctrl_h.reset(start_position=tuple(scn_data["start"]), target_position=tuple(scn_data["target"]))
        res_h = ctrl_h.run_episode(max_steps=25)

        # 3. Verify telemetry fields exist in all step telemetry rows
        required_fields = [
            "classical_action",
            "quantum_action",
            "hybrid_action",
            "final_action",
            "confidence",
            "hybrid_weight",
            "safety_override",
        ]
        for row in res_q["telemetry_history"][:5]:
            for f in required_fields:
                assert f in row, f"Field '{f}' missing from pure quantum step telemetry row"

        for row in res_h["telemetry_history"][:5]:
            for f in required_fields:
                assert f in row, f"Field '{f}' missing from hybrid step telemetry row"

        # 4. Verify action divergence occurs
        q_actions = [s["quantum_action"] for s in res_h["telemetry_history"]]
        h_actions = [s["hybrid_action"] for s in res_h["telemetry_history"]]
        divergent_count = sum(1 for q, h in zip(q_actions, h_actions) if q != h)
        assert divergent_count > 0, "Expected at least one divergent action between pure quantum and hybrid"
