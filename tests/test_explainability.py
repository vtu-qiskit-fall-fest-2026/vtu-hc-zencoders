"""Unit and Integration Tests for Quantum Decision Explainability / XAI.

Validates:
- Saliency and gradient sensitivity calculation via finite differences
- 12D and 4D feature input handling
- Accurate mapping from 4D quantum inputs to canonical 12D feature names
- Normalization summing to ~1.0
- Graceful zero-gradient handling (no NaN values)
- Signed gradient reporting
- Top-3 feature extraction and ordering
- Deterministic reproducibility
- Controller explanation logging
- NumPy JSON serialization
- Distinction between quantum recommendation and final hybrid action
- Explanations disabled by default
- CLI flag parsing
- Dashboard DataFrame formatting compatibility
"""

import json
import pytest
import numpy as np
import pandas as pd

from config import (
    DEFAULT_CONFIG,
    FEATURE_NAMES,
    QUANTUM_SELECTED_FEATURE_INDICES,
    QUANTUM_SELECTED_FEATURE_NAMES,
    QNavConfig,
)
from quantum.quantum_decision import QuantumDecisionMaker
from simulation.environment import UAVEnvironment
from navigation.controller import NavigationController
from utils.io_utils import NumpyJSONEncoder


class TestQuantumExplainability:
    """Test suite covering quantum decision sensitivity and XAI functionality."""

    @pytest.fixture
    def decision_maker(self):
        """Instantiate a QuantumDecisionMaker instance for testing."""
        return QuantumDecisionMaker(config=DEFAULT_CONFIG)

    @pytest.fixture
    def sample_12d_features(self):
        """Standard 12D feature vector matching simulation sensor output."""
        return np.array([
            0.45,   # normalized_distance_to_target
            0.60,   # target_direction_x
            0.80,   # target_direction_y
            0.80,   # target_angle_sin
            0.60,   # target_angle_cos
            0.35,   # nearest_obstacle_distance
            0.50,   # front_obstacle_distance
            0.40,   # left_obstacle_distance
            0.60,   # right_obstacle_distance
            0.25,   # obstacle_density
            0.75,   # normalized_speed
            0.30,   # boundary_proximity
        ], dtype=float)

    @pytest.fixture
    def sample_4d_features(self):
        """4D feature vector matching key selected features [1, 2, 5, 9]."""
        return np.array([
            0.60,   # target_direction_x
            0.80,   # target_direction_y
            0.35,   # nearest_obstacle_distance
            0.25,   # obstacle_density
        ], dtype=float)

    def test_saliency_explanation_shape(self, decision_maker, sample_4d_features):
        """Verify saliency explanation contains all expected keys, correct dimensions and valid values."""
        res = decision_maker.explain_decision(sample_4d_features, action=0, method="saliency")

        assert isinstance(res, dict)
        for expected_key in [
            "method",
            "action",
            "action_name",
            "action_probability",
            "features_used",
            "feature_importance",
            "signed_gradients",
            "feature_details",
            "top_features",
            "all_feature_importance",
            "all_signed_gradients",
        ]:
            assert expected_key in res, f"Missing key: {expected_key}"

        assert res["method"] == "saliency"
        assert len(res["features_used"]) == 4
        assert len(res["feature_importance"]) == 4
        assert len(res["signed_gradients"]) == 4

    def test_gradient_explanation_shape(self, decision_maker, sample_4d_features):
        """Verify gradient (log-prob sensitivity) explanation contains expected keys and non-NaN values."""
        res = decision_maker.explain_decision(sample_4d_features, action=2, method="gradient")

        assert res["method"] == "gradient"
        assert res["action"] == 2
        assert len(res["features_used"]) == 4
        for val in res["feature_importance"].values():
            assert not np.isnan(val)
            assert not np.isinf(val)
            assert val >= 0.0

    def test_12d_input(self, decision_maker, sample_12d_features):
        """Verify 12D input features are accepted and mapped properly."""
        res = decision_maker.explain_decision(sample_12d_features, action=1, method="saliency")

        assert res["features_used"] == list(QUANTUM_SELECTED_FEATURE_NAMES)
        assert len(res["feature_importance"]) == 4
        # Verify 12D dictionary has all 12 canonical names
        assert len(res["all_feature_importance"]) == 12
        for name in FEATURE_NAMES:
            assert name in res["all_feature_importance"]

    def test_4d_input(self, decision_maker, sample_4d_features):
        """Verify 4D input features operate directly without re-filtering errors."""
        res = decision_maker.explain_decision(sample_4d_features, action="FORWARD", method="saliency")

        assert res["features_used"] == list(QUANTUM_SELECTED_FEATURE_NAMES)
        assert res["action_name"] == "FORWARD"

    def test_correct_12d_to_4d_feature_name_mapping(self, decision_maker, sample_12d_features):
        """Verify explanations map [1, 2, 5, 9] to original canonical feature names, NOT feature_0/1."""
        res = decision_maker.explain_decision(sample_12d_features, action=0)

        expected_names = [
            "target_direction_x",
            "target_direction_y",
            "nearest_obstacle_distance",
            "obstacle_density",
        ]
        assert res["features_used"] == expected_names
        for name in expected_names:
            assert name in res["feature_importance"]
            assert name in res["signed_gradients"]
            assert not any(f.startswith("feature_") for f in res["features_used"])

    def test_normalization_sums_approximately_to_1(self, decision_maker, sample_4d_features):
        """Verify normalized feature importance values sum approximately to 1.0."""
        res_sal = decision_maker.explain_decision(sample_4d_features, action=3, method="saliency")
        total_sal = sum(res_sal["feature_importance"].values())
        assert total_sal == pytest.approx(1.0, abs=1e-3)

        res_grad = decision_maker.explain_decision(sample_4d_features, action=3, method="gradient")
        total_grad = sum(res_grad["feature_importance"].values())
        assert total_grad == pytest.approx(1.0, abs=1e-3)

    def test_zero_gradient_handling(self, sample_4d_features):
        """Verify zero or near-zero sensitivity returns all zeros rather than producing NaN."""
        dm = QuantumDecisionMaker(config=DEFAULT_CONFIG)
        # Monkey-patch _get_action_probabilities to return constant probability
        dm._get_action_probabilities = lambda x: np.full(9, 1.0 / 9.0)

        res = dm.explain_decision(sample_4d_features, action=0, method="saliency")
        for val in res["feature_importance"].values():
            assert not np.isnan(val)
            assert val == 0.0

        for grad in res["signed_gradients"].values():
            assert not np.isnan(grad)
            assert grad == 0.0

    def test_signed_gradient_output(self, decision_maker, sample_4d_features):
        """Verify signed gradients retain directionality and match feature_details."""
        res = decision_maker.explain_decision(sample_4d_features, action=0, method="saliency")
        for name, details in res["feature_details"].items():
            assert "importance" in details
            assert "signed_gradient" in details
            assert details["importance"] >= 0.0
            assert res["signed_gradients"][name] == details["signed_gradient"]

    def test_top_3_feature_ordering(self, decision_maker, sample_12d_features):
        """Verify top_features list has length <= 3 and is strictly ordered descending by importance."""
        res = decision_maker.explain_decision(sample_12d_features, action=4, method="saliency")
        top_3 = res["top_features"]

        assert len(top_3) <= 3
        assert len(top_3) == min(3, len(res["features_used"]))

        for i in range(len(top_3) - 1):
            assert top_3[i]["importance"] >= top_3[i + 1]["importance"]

    def test_deterministic_explanation(self, decision_maker, sample_4d_features):
        """Verify explain_decision produces identical numeric results given identical inputs."""
        res1 = decision_maker.explain_decision(sample_4d_features, action=2, method="saliency", eps=1e-4)
        res2 = decision_maker.explain_decision(sample_4d_features, action=2, method="saliency", eps=1e-4)

        for name in res1["feature_importance"]:
            assert res1["feature_importance"][name] == pytest.approx(res2["feature_importance"][name], abs=1e-8)
            assert res1["signed_gradients"][name] == pytest.approx(res2["signed_gradients"][name], abs=1e-8)

    def test_explanation_logging(self, decision_maker):
        """Verify NavigationController logs explanations when explain_decisions is True."""
        env = UAVEnvironment(config=DEFAULT_CONFIG)
        controller = NavigationController(
            environment=env,
            navigator=decision_maker,
            config=DEFAULT_CONFIG,
            is_hybrid=False,
            explain_decisions=True,
        )
        controller.reset()

        assert controller.explain_decisions is True
        assert len(controller.explanation_log) == 0

        # Run 2 steps
        controller.step()
        controller.step()

        assert len(controller.explanation_log) == 2
        assert len(controller.explanation_times) == 2
        first_entry = controller.explanation_log[0]

        assert "step" in first_entry
        assert "quantum_top_action" in first_entry
        assert "selected_action" in first_entry
        assert "explanation" in first_entry
        assert "top_3_features" in first_entry
        assert "explanation_time" in first_entry
        assert first_entry["explanation_time"] > 0.0

    def test_json_serialization(self, decision_maker, sample_12d_features):
        """Verify explanation structure is fully JSON serializable with standard and NumpyJSONEncoder."""
        res = decision_maker.explain_decision(sample_12d_features, action=0)
        # Should serialize with standard json
        json_str = json.dumps(res, cls=NumpyJSONEncoder)
        decoded = json.loads(json_str)

        assert decoded["action"] == res["action"]
        assert len(decoded["top_features"]) == len(res["top_features"])

    def test_hybrid_quantum_vs_final_action_distinction(self, decision_maker):
        """Verify hybrid mode distinguishes quantum recommendation from final fused/APF action."""
        env = UAVEnvironment(config=DEFAULT_CONFIG)
        from classical.baseline import ClassicalBaselineNavigator
        classical_nav = ClassicalBaselineNavigator(config=DEFAULT_CONFIG, obstacles=env.obstacles)

        controller = NavigationController(
            environment=env,
            navigator=decision_maker,
            classical_navigator=classical_nav,
            config=DEFAULT_CONFIG,
            is_hybrid=True,
            explain_decisions=True,
        )
        controller.reset()

        # Step controller
        controller.step()

        assert len(controller.explanation_log) >= 1
        entry = controller.explanation_log[0]

        assert "classical_action" in entry
        assert "quantum_top_action" in entry
        assert "selected_action" in entry
        assert "quantum_action_used" in entry
        assert "classical_action_used" in entry
        assert entry["quantum_action == final_action"] == entry["quantum_action_used"]

    def test_explanation_disabled_by_default(self):
        """Verify explain_decisions is False by default and doesn't incur explanation overhead."""
        env = UAVEnvironment(config=DEFAULT_CONFIG)
        dm = QuantumDecisionMaker(config=DEFAULT_CONFIG)
        controller = NavigationController(
            environment=env,
            navigator=dm,
            config=DEFAULT_CONFIG,
        )
        controller.reset()

        assert controller.explain_decisions is False
        controller.step()
        assert len(controller.explanation_log) == 0
        assert len(controller.explanation_times) == 0

    def test_cli_flag(self):
        """Verify CLI parser recognizes --explain-decisions."""
        from run import parse_arguments
        import sys

        orig_argv = sys.argv
        try:
            sys.argv = ["run.py", "--explain-decisions"]
            args = parse_arguments()
            assert args.explain_decisions is True

            sys.argv = ["run.py"]
            args_default = parse_arguments()
            assert args_default.explain_decisions is False
        finally:
            sys.argv = orig_argv

    def test_dashboard_explanation_dataframe(self, decision_maker, sample_12d_features):
        """Verify explanation data converts to Pandas DataFrame cleanly and supports mean/median/std."""
        res1 = decision_maker.explain_decision(sample_12d_features, action=0)
        res2 = decision_maker.explain_decision(sample_12d_features, action=1)

        records = []
        for step, res in enumerate([res1, res2]):
            for feat, imp in res["feature_importance"].items():
                records.append({
                    "step": step,
                    "feature": feat,
                    "importance": float(imp),
                })

        df = pd.DataFrame(records)
        assert len(df) == 8

        stats = []
        for feat, group in df.groupby("feature"):
            vals = group["importance"].values
            stats.append({
                "Feature": str(feat),
                "Mean Importance": float(np.mean(vals)),
                "Median Importance": float(np.median(vals)),
                "Std Importance": float(np.std(vals)),
            })

        df_stats = pd.DataFrame(stats)
        assert "Mean Importance" in df_stats.columns
        assert "Std Importance" in df_stats.columns
        assert not df_stats.isna().any().any()
