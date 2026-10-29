"""Phase 3 Regression Tests: Safety-Aware Action Masking for Pure Quantum and Hybrid Navigation.

Validates all 12 Phase 3 requirements:
1. Unsafe actions cannot be selected when safe actions exist.
2. Safe-action probabilities sum to 1 within 1e-9.
3. All-zero probabilities are handled deterministically (uniform over safe set).
4. NaN and infinite inputs are handled safely.
5. Negative or malformed probability inputs cannot enable unsafe actions.
6. No-safe-action cases use the existing emergency behavior.
7. Recovery state and recovery side remain stable during masking.
8. Classical APF behavior is completely unchanged.
9. Pure Quantum and Hybrid use identical SafetyLayer constraints.
10. Telemetry records the expected masking information.
11. Masking cannot introduce a collision in deterministic regression scenarios.
12. Invalid probability vector shapes are handled explicitly.
"""

import pytest
import numpy as np
from pathlib import Path

from navigation.safety import SafetyLayer, compute_safe_action_probabilities
from navigation.controller import NavigationController
from evaluation.compare_models import ModelComparator
from classical.baseline import ClassicalBaselineNavigator
from quantum.quantum_decision import QuantumDecisionMaker
from simulation.environment import UAVEnvironment
from simulation.obstacles import ObstacleField, CircularObstacle
from config import DEFAULT_CONFIG
from run import load_scenarios_and_maps


class TestPhase3SafetyMasking:
    """Complete test suite for Phase 3 Safety-Aware Action Masking."""

    def test_unsafe_actions_cannot_be_selected_when_safe_actions_exist(self):
        """1. Verify that unsafe actions receive probability 0.0 and cannot be selected."""
        quantum_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01])
        safe_actions = [4]

        res = compute_safe_action_probabilities(quantum_probs, safe_actions)
        norm_probs = res["normalized_probabilities"]

        assert norm_probs[4] == pytest.approx(1.0, abs=1e-9)
        for a in range(9):
            if a != 4:
                assert norm_probs[a] == 0.0

        assert res["masked_actions_count"] == 8
        assert res["probability_mass_removed"] == pytest.approx(0.95, abs=1e-6)
        assert res["emergency_fallback"] is False

    def test_safe_action_probabilities_sum_to_one(self):
        """2. Verify that normalized safe-action probabilities sum to 1.0 within 1e-9."""
        rng = np.random.RandomState(42)
        test_subsets = [
            [0],
            [2, 5],
            [1, 3, 4, 7],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
        for safe_set in test_subsets:
            raw_probs = rng.dirichlet(np.ones(9))
            res = compute_safe_action_probabilities(raw_probs, safe_set)
            norm_probs = res["normalized_probabilities"]

            assert np.sum(norm_probs) == pytest.approx(1.0, abs=1e-9)
            for a in range(9):
                if a not in safe_set:
                    assert norm_probs[a] == 0.0
                else:
                    assert norm_probs[a] >= 0.0

    def test_all_zero_probabilities_handled_deterministically(self):
        """3. Verify all-zero probabilities fallback to deterministic uniform over safe set."""
        all_zeros = np.zeros(9, dtype=float)
        safe_actions = [1, 3, 6]

        res = compute_safe_action_probabilities(all_zeros, safe_actions)
        norm_probs = res["normalized_probabilities"]

        expected_val = 1.0 / len(safe_actions)
        assert np.sum(norm_probs) == pytest.approx(1.0, abs=1e-9)
        for a in safe_actions:
            assert norm_probs[a] == pytest.approx(expected_val, abs=1e-9)
        for a in range(9):
            if a not in safe_actions:
                assert norm_probs[a] == 0.0

        assert res["emergency_fallback"] is False

    def test_nan_and_infinite_inputs_handled_safely(self):
        """4. Verify NaN and infinite inputs are sanitized without crashing or propagating NaNs."""
        malformed = np.array([np.nan, np.inf, -np.inf, 0.5, 0.5, np.nan, 0.0, 0.0, 0.0])
        safe_actions = [3, 4]

        res = compute_safe_action_probabilities(malformed, safe_actions)
        norm_probs = res["normalized_probabilities"]

        assert not np.any(np.isnan(norm_probs))
        assert not np.any(np.isinf(norm_probs))
        assert np.sum(norm_probs) == pytest.approx(1.0, abs=1e-9)
        assert norm_probs[3] == pytest.approx(0.5, abs=1e-9)
        assert norm_probs[4] == pytest.approx(0.5, abs=1e-9)

        malformed_safe = np.array([np.nan, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res2 = compute_safe_action_probabilities(malformed_safe, [0, 1])
        norm2 = res2["normalized_probabilities"]
        assert not np.any(np.isnan(norm2))
        assert not np.any(np.isinf(norm2))
        assert np.sum(norm2) == pytest.approx(1.0, abs=1e-9)
        assert norm2[0] == pytest.approx(0.5, abs=1e-9)
        assert norm2[1] == pytest.approx(0.5, abs=1e-9)

    def test_negative_or_malformed_probabilities_cannot_enable_unsafe(self):
        """5. Verify negative values are clipped to 0.0 and cannot enable unsafe actions."""
        negative_probs = np.array([-1.5, 0.8, -0.2, 0.0, 0.0, -0.9, 0.0, 0.0, 0.0])
        safe_actions = [1, 2]

        res = compute_safe_action_probabilities(negative_probs, safe_actions)
        norm_probs = res["normalized_probabilities"]

        assert np.all(norm_probs >= 0.0)
        assert norm_probs[2] == 0.0
        assert norm_probs[1] == pytest.approx(1.0, abs=1e-9)
        for a in range(9):
            if a not in safe_actions:
                assert norm_probs[a] == 0.0

    def test_no_safe_actions_invokes_emergency_behavior(self):
        """6. Verify that an empty safe set triggers emergency fallback."""
        quantum_probs = np.ones(9) / 9.0
        safe_actions = []

        res = compute_safe_action_probabilities(quantum_probs, safe_actions)
        assert res["emergency_fallback"] is True
        assert np.sum(res["normalized_probabilities"]) == 0.0
        assert res["masked_actions_count"] == 9
        assert res["probability_mass_removed"] == pytest.approx(1.0, abs=1e-9)

        # Test controller behavior when no safe actions exist
        env = UAVEnvironment()
        tight_obs = [
            CircularObstacle(1, x=20.0, y=21.0, radius=0.8),
            CircularObstacle(2, x=20.0, y=19.0, radius=0.8),
            CircularObstacle(3, x=21.0, y=20.0, radius=0.8),
            CircularObstacle(4, x=19.0, y=20.0, radius=0.8),
        ]
        env.obstacles = ObstacleField(tight_obs)
        q_nav = QuantumDecisionMaker(config=DEFAULT_CONFIG, load_checkpoint=False)
        controller = NavigationController(environment=env, navigator=q_nav)
        controller.reset(start_position=(20.0, 20.0), target_position=(20.0, 40.0))

        _, _, _, telem = controller.step()
        assert telem.get("is_safe") is False or telem.get("emergency_fallback") is True
        assert telem.get("action") in ("EMERGENCY_EVASION", "STAY")

    def test_recovery_state_and_side_stable_during_masking(self):
        """7. Verify recovery state machine and recovery side remain stable during masking."""
        env = UAVEnvironment()
        env.obstacles = ObstacleField([CircularObstacle(1, x=25.0, y=25.0, radius=4.0)])
        q_nav = QuantumDecisionMaker(config=DEFAULT_CONFIG, load_checkpoint=False)
        controller = NavigationController(environment=env, navigator=q_nav)
        controller.reset(start_position=(25.0, 20.0), target_position=(25.0, 35.0))

        controller.recovery_mode = True
        controller.recovery_side = "left"
        controller.recovery_steps_left = 10
        controller.recovery_vector = np.array([-1.0, 0.0])
        controller.recovery_start_dist = 15.0

        initial_side = controller.recovery_side
        for step_idx in range(5):
            _, _, _, telem = controller.step()
            assert controller.recovery_side == initial_side
            assert telem.get("recovery_active") is True
            assert telem.get("recovery_side") == initial_side

    def test_classical_apf_behavior_unchanged(self):
        """8. Verify Classical APF behavior and outputs are completely unchanged by masking."""
        scenarios_list, maps_by_id = load_scenarios_and_maps()
        scn = [s for s in scenarios_list if s["id"] == "scenario_01"][0]
        map_data = maps_by_id[scn["map_id"]]

        comparator = ModelComparator(config=DEFAULT_CONFIG)
        c_res, c_metrics = comparator.run_single_model_scenario(
            model_type="classical",
            scenario=scn,
            map_data=map_data,
            seed=42,
        )

        assert c_res["success"] is True
        assert c_res["collision_count"] == 0
        assert c_res["telemetry_history"][0]["unmasked_quantum_action"] == "N/A"
        assert c_res["telemetry_history"][0]["masked_actions_count"] == 0
        assert c_res["telemetry_history"][0]["probability_mass_removed"] == 0.0

    def test_pure_quantum_and_hybrid_use_identical_safetylayer_constraints(self):
        """9. Verify Pure Quantum and Hybrid use identical SafetyLayer constraints."""
        env = UAVEnvironment()
        env.obstacles = ObstacleField([CircularObstacle(1, x=15.0, y=15.0, radius=2.5)])
        q_nav = QuantumDecisionMaker(config=DEFAULT_CONFIG, load_checkpoint=False)

        ctrl_q = NavigationController(environment=env, navigator=q_nav, is_hybrid=False)
        ctrl_q.reset(start_position=(15.0, 11.5), target_position=(15.0, 25.0))
        _, _, _, telem_q = ctrl_q.step()

        ctrl_h = NavigationController(environment=env, navigator=q_nav, is_hybrid=True)
        ctrl_h.reset(start_position=(15.0, 11.5), target_position=(15.0, 25.0))
        _, _, _, telem_h = ctrl_h.step()

        assert telem_q.get("safe_action_mask") == telem_h.get("safe_action_mask")
        assert telem_q.get("masked_actions_count") == telem_h.get("masked_actions_count")

    def test_telemetry_records_expected_masking_fields(self):
        """10. Verify that step telemetry records all 12 required masking fields."""
        scenarios_list, maps_by_id = load_scenarios_and_maps()
        scn = [s for s in scenarios_list if s["id"] == "scenario_01"][0]
        map_data = maps_by_id[scn["map_id"]]

        comparator = ModelComparator(config=DEFAULT_CONFIG)
        q_res, q_metrics = comparator.run_single_model_scenario(
            model_type="quantum",
            scenario=scn,
            map_data=map_data,
            seed=42,
        )

        assert "safety_masking_log" in q_res
        assert len(q_res["telemetry_history"]) > 0

        first_step = q_res["telemetry_history"][0]
        required_fields = [
            "step",
            "original_quantum_probabilities",
            "safe_action_mask",
            "masked_actions_count",
            "probability_mass_removed",
            "normalized_probabilities",
            "unmasked_quantum_action",
            "masking_changed_action",
            "emergency_fallback",
            "is_safe",
            "recovery_active",
            "recovery_side",
            "safety_interventions",
        ]
        for f in required_fields:
            assert f in first_step, f"Field '{f}' missing from telemetry step."

    def test_masking_cannot_introduce_collision_in_deterministic_scenarios(self):
        """11. Verify that masking does not introduce collisions in deterministic benchmark scenarios."""
        scenarios_list, maps_by_id = load_scenarios_and_maps()
        test_scenarios = [s for s in scenarios_list if s["id"] in ("scenario_01", "scenario_03")]
        comparator = ModelComparator(config=DEFAULT_CONFIG)

        for scn in test_scenarios:
            map_data = maps_by_id[scn["map_id"]]
            for m_type in ("quantum", "hybrid"):
                res, metrics = comparator.run_single_model_scenario(
                    model_type=m_type,
                    scenario=scn,
                    map_data=map_data,
                    seed=42,
                )
                assert res["collided"] is False, f"Collision observed in {scn['id']} for {m_type}!"
                assert res["collision_count"] == 0

    def test_invalid_probability_vector_shapes_handled_explicitly(self):
        """12. Verify invalid probability vector shapes raise ValueError."""
        safe_actions = [0, 1, 2]

        # 2D array
        with pytest.raises(ValueError, match="1D"):
            compute_safe_action_probabilities(np.zeros((3, 3)), safe_actions)

        # Length != 9 (e.g. 5)
        with pytest.raises(ValueError, match="length 9"):
            compute_safe_action_probabilities(np.zeros(5), safe_actions)

        # Empty array
        with pytest.raises(ValueError, match="length 9"):
            compute_safe_action_probabilities(np.array([]), safe_actions)
