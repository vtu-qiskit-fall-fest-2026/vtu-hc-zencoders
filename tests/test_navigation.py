"""Unit tests for Action Selection, Safety Layer mediation, and Navigation Controller."""

import pytest
import numpy as np
from navigation.safety import SafetyLayer
from navigation.action_selector import ActionSelector
from navigation.controller import NavigationController
from classical.baseline import ClassicalBaselineNavigator
from quantum.quantum_decision import QuantumDecisionMaker
from simulation.obstacles import ObstacleField, CircularObstacle
from simulation.environment import UAVEnvironment


class TestSafetyLayer:
    """Tests for obstacle clearance checks, boundary enforcement, and evasive maneuvers."""

    def test_safe_action_allowed(self):
        safety = SafetyLayer()
        obs_field = ObstacleField([CircularObstacle(1, x=25.0, y=25.0, radius=2.0)])
        current_pos = np.array([5.0, 5.0])

        # Move away from obstacle toward (4, 4)
        is_safe, clearance, reason = safety.evaluate_action_safety(
            current_pos=current_pos,
            direction=np.array([1.0, 0.0]),
            speed=1.5,
            obstacles=obs_field,
            map_width=50.0,
            map_height=50.0,
        )
        assert is_safe is True
        assert reason == "safe"

    def test_boundary_unsafe_action(self):
        safety = SafetyLayer()
        obs_field = ObstacleField()
        current_pos = np.array([1.0, 25.0])  # Near left border

        # Attempt to step left out of bounds
        is_safe, clearance, reason = safety.evaluate_action_safety(
            current_pos=current_pos,
            direction=np.array([-1.0, 0.0]),
            speed=1.5,
            obstacles=obs_field,
            map_width=50.0,
            map_height=50.0,
        )
        assert is_safe is False
        assert "boundary" in reason

    def test_obstacle_collision_override(self):
        safety = SafetyLayer()
        obs_field = ObstacleField([CircularObstacle(1, x=10.0, y=10.0, radius=3.0)])
        current_pos = np.array([8.0, 10.0])  # Right next to obstacle

        candidate_actions = ["RIGHT", "UP", "DOWN", "LEFT", "STAY"]
        action_scores = {"RIGHT": 10.0, "UP": 5.0, "DOWN": 4.0, "LEFT": 2.0, "STAY": 0.0}
        action_vectors = {
            "RIGHT": (1.0, 0.0),
            "UP": (0.0, 1.0),
            "DOWN": (0.0, -1.0),
            "LEFT": (-1.0, 0.0),
            "STAY": (0.0, 0.0),
        }

        # "RIGHT" would plunge straight into obstacle at (10, 10)
        safe_action, safe_vec, was_intervened = safety.select_safe_action(
            current_pos=current_pos,
            candidate_actions=candidate_actions,
            action_scores=action_scores,
            action_vectors=action_vectors,
            speed=1.5,
            obstacles=obs_field,
        )

        assert was_intervened is True
        assert safe_action != "RIGHT"  # Must have been overridden

    def test_safety_counters_differentiation(self):
        safety = SafetyLayer()
        obs_field = ObstacleField([CircularObstacle(1, x=10.0, y=10.0, radius=3.0)])
        current_pos = np.array([8.0, 10.0])

        candidate_actions = ["RIGHT", "UP", "STAY"]
        action_scores = {"RIGHT": 10.0, "UP": 5.0, "STAY": 0.0}
        action_vectors = {"RIGHT": (1.0, 0.0), "UP": (0.0, 1.0), "STAY": (0.0, 0.0)}

        assert safety.intervention_count == 0
        assert safety.unsafe_action_attempts == 0

        # RIGHT is unsafe -> triggers 1 intervention, and at least 1 unsafe attempt
        safe_action, _, was_intervened = safety.select_safe_action(
            current_pos=current_pos,
            candidate_actions=candidate_actions,
            action_scores=action_scores,
            action_vectors=action_vectors,
            speed=1.5,
            obstacles=obs_field,
        )
        assert was_intervened is True
        assert safety.intervention_count == 1
        assert safety.unsafe_action_attempts >= 1


class TestNavigationController:
    """Tests for the master closed-loop NavigationController with both Classical and Quantum models."""

    def test_classical_controller_run(self):
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [{"id": 1, "type": "circle", "x": 20.0, "y": 20.0, "radius": 3.0}],
        }
        scenario = {"start": [5.0, 5.0], "target": [15.0, 15.0], "max_steps": 20}
        env = UAVEnvironment(map_data=map_data, scenario_data=scenario)
        navigator = ClassicalBaselineNavigator(obstacles=env.obstacles)
        controller = NavigationController(environment=env, navigator=navigator)

        ep_result = controller.run_episode(max_steps=10)
        assert ep_result["total_steps"] > 0
        assert len(ep_result["trajectory"]) >= 2
        assert len(ep_result["telemetry_history"]) > 0

    def test_quantum_controller_run(self):
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [{"id": 1, "type": "circle", "x": 20.0, "y": 20.0, "radius": 3.0}],
        }
        scenario = {"start": [5.0, 5.0], "target": [15.0, 15.0], "max_steps": 20}
        env = UAVEnvironment(map_data=map_data, scenario_data=scenario)
        navigator = QuantumDecisionMaker()
        controller = NavigationController(environment=env, navigator=navigator)

        ep_result = controller.run_episode(max_steps=5)
        assert ep_result["total_steps"] == 5
        assert len(ep_result["trajectory"]) == 6
        assert len(ep_result["telemetry_history"]) == 5

    def test_shared_safety_behavior(self):
        """Verify identical safety layer is applied equally to both Classical and Quantum outputs."""
        safety = SafetyLayer()
        obs_field = ObstacleField([CircularObstacle(1, x=20.0, y=20.0, radius=3.0)])
        # Both proposed identical direction vector toward obstacle
        is_safe, _, reason = safety.evaluate_action_safety(
            current_pos=np.array([18.0, 20.0]),
            direction=np.array([1.0, 0.0]),
            speed=1.5,
            obstacles=obs_field,
        )
        assert is_safe is False
        assert "obstacle" in reason or "clearance" in reason

    def test_reproducible_benchmark_execution(self):
        """Running the same model and scenario with identical seed must yield identical trajectories."""
        from evaluation.compare_models import ModelComparator
        from config import QNavConfig

        cfg = QNavConfig()
        comparator = ModelComparator(config=cfg)
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [{"id": 1, "type": "circle", "x": 25.0, "y": 25.0, "radius": 4.0}],
        }
        scenario = {"id": "repro_test", "start": [5.0, 5.0], "target": [45.0, 45.0], "max_steps": 10}

        r1, _ = comparator.run_single_model_scenario("classical", scenario, map_data, seed=123)
        r2, _ = comparator.run_single_model_scenario("classical", scenario, map_data, seed=123)

        assert np.allclose(r1["trajectory"], r2["trajectory"])

    def test_dashboard_dataframe_construction(self):
        """Verify comparison DataFrame has consistent numeric float columns and converts to Arrow."""
        import pandas as pd
        import pyarrow as pa

        comparison_df = pd.DataFrame({
            "Metric": [
                "Success Rate",
                "Mean Path Length",
                "Mean Steps",
                "Mean Collisions",
                "Collision Rate",
                "Navigation Efficiency",
                "Safety Interventions",
                "Mean Episode Time",
            ],
            "Classical Baseline": [
                float(0.40),
                float(145.80),
                float(97.2),
                float(0.0),
                float(0.0),
                float(0.4587),
                float(13.0),
                float(0.0641),
            ],
            "Quantum-Assisted": [
                float(0.20),
                float(175.80),
                float(117.2),
                float(0.0),
                float(0.0),
                float(0.2254),
                float(61.2),
                float(0.2347),
            ],
        })

        assert comparison_df["Classical Baseline"].dtype == np.float64
        assert comparison_df["Quantum-Assisted"].dtype == np.float64
        # Converting to PyArrow table must succeed without ArrowTypeError
        arrow_table = pa.Table.from_pandas(comparison_df)
        assert arrow_table.num_rows == 8

    def test_numpy_json_serialization(self, tmp_path):
        """Verify save_json handles complex nested NumPy types, arrays, booleans, and Paths without error."""
        from utils.io_utils import save_json, load_json
        from pathlib import Path

        test_data = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14159),
            "bool_val": np.bool_(True),
            "array_val": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "path_val": Path("outputs/test.json"),
        }

        out_file = tmp_path / "test_serialize.json"
        save_json(test_data, out_file)
        loaded = load_json(out_file)

        assert loaded["int_val"] == 42
        assert np.isclose(loaded["float_val"], 3.14159)
        assert loaded["bool_val"] is True
        assert loaded["array_val"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_get_safe_actions(self):
        """Verify safety layer filters out actions heading into obstacles or boundary walls."""
        safety = SafetyLayer()
        obs_field = ObstacleField([CircularObstacle(1, x=20.0, y=20.0, radius=3.0)])
        state = {
            "drone_position": np.array([14.5, 20.0]),
            "drone_speed": 1.5,
            "obstacles": obs_field,
            "map_width": 50.0,
            "map_height": 50.0,
        }
        safe_actions = safety.get_safe_actions(state)
        # RIGHT (idx 2) leads directly into obstacle at (20, 20), so it must be unsafe
        assert 2 not in safe_actions
        # LEFT (idx 6) and STAY (idx 8) should be safe
        assert 8 in safe_actions
        assert 6 in safe_actions
        assert len(safe_actions) > 0

    def test_hybrid_select_action(self):
        """Verify hybrid decision fusion combines quantum probabilities and classical APF safely."""
        env = UAVEnvironment()
        controller = NavigationController(environment=env)
        state = env.get_state()

        # Classical action = 0 (RIGHT), quantum strongly prefers 1 (UP)
        classical_action_idx = 0
        quantum_probs = np.zeros(9)
        quantum_probs[1] = 0.9
        quantum_probs[0] = 0.1

        selected = controller.select_action(state, classical_action_idx, quantum_probs)
        assert 0 <= selected < 9
        assert len(controller.fusion_log) == 1
        entry = controller.fusion_log[-1]
        assert entry["classical_action"] == 0
        assert entry["quantum_top_action"] == 1
        assert "fusion_confidence" in entry

    def test_hybrid_controller_run(self):
        """Verify controller runs successfully in hybrid mode with both classical and quantum inputs."""
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [{"id": 1, "type": "circle", "x": 20.0, "y": 20.0, "radius": 3.0}],
        }
        scenario = {"start": [5.0, 5.0], "target": [15.0, 15.0], "max_steps": 20}
        env = UAVEnvironment(map_data=map_data, scenario_data=scenario)
        controller = NavigationController(environment=env, is_hybrid=True)

        ep_result = controller.run_episode(max_steps=5, is_hybrid=True)
        assert ep_result["is_hybrid"] is True
        assert len(ep_result["fusion_log"]) == 5
        assert ep_result["total_steps"] == 5
        assert len(ep_result["trajectory"]) == 6


