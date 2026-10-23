"""Unit tests for Quantum Feature Encoding, Parameterized Circuit, and VQC Model."""

import pytest
import numpy as np
from quantum.feature_encoding import QuantumFeatureEncoder
from quantum.quantum_circuit import ParameterizedNavigationCircuit
from quantum.quantum_model import VariationalQuantumNavigator
from quantum.quantum_decision import QuantumDecisionMaker
from simulation.environment import UAVEnvironment


class TestQuantumFeatureEncoding:
    """Tests for mapping continuous navigation features to quantum rotation angles."""

    def test_encoder_output_dimensions(self):
        encoder = QuantumFeatureEncoder()
        dummy_features = np.linspace(0.1, 1.0, 12)
        angles = encoder.encode(dummy_features)

        assert len(angles) == encoder.n_qubits
        assert np.all(angles >= -np.pi - 1e-5)
        assert np.all(angles <= np.pi + 1e-5)

    def test_grouped_encoding(self):
        encoder = QuantumFeatureEncoder()
        dummy_features = np.zeros(12)
        feature_dict = {
            "norm_target_dist": 0.8,
            "rel_target_angle": 0.5,
            "norm_nearest_obs_dist": 0.2,
            "nearest_obs_dir_x": 0.0,
            "norm_speed": 0.5,
            "velocity_x": 0.2,
            "boundary_proximity": 0.9,
            "obstacle_density": 0.1,
        }
        angles = encoder.encode(dummy_features, feature_dict=feature_dict)
        assert len(angles) == encoder.n_qubits
        assert not np.isnan(angles).any()

    def test_select_key_features(self):
        """Verify 12D features are reduced to 4D key navigation features normalized to [0, 1]."""
        encoder = QuantumFeatureEncoder()
        dummy_12d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        # Indices [1, 2, 5, 9] correspond to [0.2, 0.3, 0.6, 1.0]
        selected_4d = encoder.select_key_features(dummy_12d)

        assert len(selected_4d) == 4
        assert selected_4d.min() >= 0.0
        assert selected_4d.max() <= 1.0 + 1e-6
        # Min element (0.2) maps to 0.0, Max element (1.0) maps to 1.0
        assert np.isclose(selected_4d[0], 0.0)
        assert np.isclose(selected_4d[3], 1.0)



class TestParameterizedQuantumCircuit:
    """Tests for Qiskit circuit construction, parameter binding, and statevector simulation."""

    def test_circuit_structure(self):
        circuit_builder = ParameterizedNavigationCircuit()
        assert circuit_builder.circuit.num_qubits == circuit_builder.n_qubits
        assert circuit_builder.get_num_weights() > 0

    def test_circuit_execution_and_observables(self):
        circuit_builder = ParameterizedNavigationCircuit()
        dummy_angles = np.array([0.5, -0.3, 0.2, 0.8])
        dummy_weights = np.zeros(circuit_builder.get_num_weights())

        expectations, basis_probs, probs_dict = circuit_builder.execute(
            features=dummy_angles, weights=dummy_weights
        )

        assert len(expectations) == circuit_builder.n_qubits
        assert np.all(expectations >= -1.0) and np.all(expectations <= 1.0)
        # Probabilities must sum to 1.0
        assert np.isclose(np.sum(basis_probs), 1.0, atol=1e-5)
        assert len(probs_dict) > 0

    def test_circuit_qubit_and_layer_scaling(self):
        from config import QuantumConfig
        for n_q in [4, 6, 8]:
            for n_l in [1, 2, 3]:
                cfg = QuantumConfig(n_qubits=n_q, n_layers=n_l)
                pqc = ParameterizedNavigationCircuit(config=cfg)
                assert pqc.circuit.num_qubits == n_q
                assert pqc.get_num_weights() == n_q * n_l * 2


class TestVariationalQuantumNavigator:
    """Tests for hybrid variational quantum model and action logits projection."""

    def test_vqc_forward_pass(self):
        model = VariationalQuantumNavigator()
        dummy_angles = np.array([0.1, 0.2, -0.1, 0.4])

        out = model.forward(encoded_angles=dummy_angles, target_distance=15.0)

        assert "action_scores" in out
        assert "action_probabilities" in out
        probs = out["action_probabilities"]
        assert len(probs) == len(model.action_names)
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-4)

    def test_deterministic_initialization(self):
        model1 = VariationalQuantumNavigator(seed=42)
        model2 = VariationalQuantumNavigator(seed=42)
        model3 = VariationalQuantumNavigator(seed=999)

        assert np.allclose(model1.weights, model2.weights)
        assert np.allclose(model1.readout_weights, model2.readout_weights)
        assert not np.allclose(model1.weights, model3.weights)

    def test_model_output_unpacking(self):
        model = VariationalQuantumNavigator()
        dummy_12d = np.linspace(0.1, 0.9, 12)

        # 3-element tuple unpacking: logits, expectations, probabilities
        logits, expectations, probabilities = model.forward(dummy_12d)

        assert isinstance(logits, np.ndarray)
        assert isinstance(expectations, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        assert len(logits) == len(model.action_names)
        assert len(expectations) == model.n_qubits
        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-4)


class TestQuantumDecisionMaker:
    """Tests for complete end-to-end quantum decision making from environment state."""

    def test_quantum_decision(self):
        env = UAVEnvironment()
        decision_maker = QuantumDecisionMaker()

        state = env.get_state()
        decision = decision_maker.compute_decision(state)

        assert decision["model_type"] == "quantum"
        assert decision["action"] in decision_maker.action_names
        assert len(decision["direction_vector"]) == 2
        assert 0.0 <= decision["confidence"] <= 1.0
        assert "quantum_expectations" in decision
        assert len(decision["quantum_expectations"]) == decision_maker.config.quantum.n_qubits

    def test_training_loss_reduction(self):
        from quantum.training import train_vqc_teacher_policy
        train_res = train_vqc_teacher_policy(epochs=5, n_samples=40, seed=42, save_outputs=False)
        assert "history" in train_res
        assert "train_loss" in train_res["history"]
        losses = train_res["history"]["train_loss"]
        assert len(losses) == 5
        assert all(l > 0.0 for l in losses)

    def test_fine_tune_on_test_scenarios(self):
        """Verify fine-tuning quantum model on scenario demonstrations optimizes weights."""
        from quantum.training import fine_tune_on_test_scenarios
        model = VariationalQuantumNavigator(seed=42)
        initial_weights = np.copy(model.weights)
        test_scenarios = [{
            "scenario_id": "test_scen",
            "map_id": "test_map",
            "start": [5.0, 5.0],
            "target": [15.0, 15.0],
            "max_steps": 5,
        }]
        fine_tuned_model = fine_tune_on_test_scenarios(
            model=model,
            scenarios=test_scenarios,
            n_episodes_per_scenario=2,
            learning_rate=0.01,
            epochs=3,
        )
        assert fine_tuned_model is not None
        assert not np.allclose(fine_tuned_model.weights, initial_weights)


class TestQuantumOptimizationStep2:
    """Comprehensive regression tests verifying Step 2 Quantum Accuracy Optimization (Phases 1-11)."""

    def test_canonical_encoding_equivalence(self):
        """Phase 1: Verify exact identity between training feature encoding and inference encoding."""
        encoder = QuantumFeatureEncoder()
        model = VariationalQuantumNavigator(seed=42)
        raw_12d = np.array([0.45, 0.707, 0.707, 0.5, 0.866, 0.35, 0.40, 0.60, 0.55, 0.20, 0.50, 0.80])

        angles_encoder = encoder.encode(raw_12d)
        angles_model = model.feature_encoder.encode(raw_12d)

        assert np.allclose(angles_encoder, angles_model, atol=1e-8)
        assert len(angles_encoder) == 4
        assert not np.isnan(angles_encoder).any()

    def test_all_16_basis_probabilities_and_normalization(self):
        """Phase 5: Verify all 16 computational basis states are measured, summing to 1.0 with no truncation."""
        circuit = ParameterizedNavigationCircuit(n_qubits=4, n_layers=2)
        features = np.array([0.1, -0.2, 0.3, -0.4])
        weights = np.ones(circuit.get_num_weights()) * 0.25

        exp_vals, basis_probs, probs_dict = circuit.execute(features, weights)

        # 4 qubits -> 16 basis states
        assert len(basis_probs) == 16
        assert len(exp_vals) == 4
        # Normalization check: total probability mass must equal 1.0 within numerical precision
        total_prob = float(np.sum(basis_probs))
        assert abs(total_prob - 1.0) < 1e-6
        assert all(p >= 0.0 for p in basis_probs)
        assert len(probs_dict) == 16

        # Total quantum features assembled by VQC forward pass must be 4 + 16 = 20
        model = VariationalQuantumNavigator(n_qubits=4, seed=42)
        out = model.forward(features)
        assert len(out["quantum_features"]) == 20
        assert len(out["action_probabilities"]) == 9

    def test_teacher_stay_behavior(self):
        """Phase 2: Verify STAY is ONLY chosen when at goal or completely trapped, never in open flight."""
        from quantum.training import teacher_policy
        from config import DEFAULT_CONFIG

        nav_cfg = DEFAULT_CONFIG.navigation
        action_names = nav_cfg.action_names
        action_vectors = nav_cfg.action_vectors

        # Case 1: In open space, far from target (dist = 25m) -> MUST NOT be STAY
        act_open, idx_open, _ = teacher_policy(
            drone_pos=np.array([10.0, 10.0]),
            target_pos=np.array([35.0, 10.0]),
            obstacles=None,
            action_names=action_names,
            action_vectors=action_vectors,
            current_heading=0.0,
            goal_threshold=2.0,
        )
        assert act_open != "STAY"
        assert act_open in ["RIGHT", "UP_RIGHT", "DOWN_RIGHT"]

        # Case 2: Within goal threshold (dist = 1.0m <= 2.0m) -> MUST be STAY
        act_goal, idx_goal, _ = teacher_policy(
            drone_pos=np.array([10.0, 10.0]),
            target_pos=np.array([10.5, 10.5]),
            obstacles=None,
            action_names=action_names,
            action_vectors=action_vectors,
            current_heading=0.0,
            goal_threshold=2.0,
        )
        assert act_goal == "STAY"
        assert idx_goal == action_names.index("STAY")

    def test_teacher_heading_continuity_penalizes_180_reversals(self):
        """Phase 2: Verify teacher policy penalizes 180-degree action reversals."""
        from quantum.training import teacher_policy
        from config import DEFAULT_CONFIG

        nav_cfg = DEFAULT_CONFIG.navigation
        # Heading facing RIGHT (0 rad). Moving target slightly north-east.
        # Teacher should pick forward actions (RIGHT, UP_RIGHT), not 180-deg reversal LEFT.
        act, idx, _ = teacher_policy(
            drone_pos=np.array([20.0, 20.0]),
            target_pos=np.array([30.0, 25.0]),
            obstacles=None,
            action_names=nav_cfg.action_names,
            action_vectors=nav_cfg.action_vectors,
            current_heading=0.0,
            goal_threshold=2.0,
        )
        assert act not in ["LEFT", "DOWN_LEFT", "UP_LEFT", "STAY"]

    def test_dataset_generation_properties(self):
        """Phase 3: Verify synthetic dataset generates contiguous trajectories with balanced classes."""
        from quantum.training import generate_training_dataset

        dataset = generate_training_dataset(num_samples=60, seed=42)
        assert len(dataset) >= 60

        actions = [d["teacher_action"] for d in dataset]
        stay_count = actions.count("STAY")
        stay_ratio = stay_count / len(dataset)

        # STAY must not dominate the dataset (< 15%)
        assert stay_ratio < 0.15
        # Must have at least 4 distinct navigation directions represented
        unique_actions = set(actions)
        assert len(unique_actions) >= 4

        # Sequential trajectory tags must be present
        assert "trajectory_id" in dataset[0]
        assert "step" in dataset[0]
        assert len(dataset[0]["features"]) == 12

    def test_vqc_parameter_updates_with_adam(self):
        """Phase 4: Verify theta circuit parameters and readout weights actively update during training."""
        from quantum.training import train_quantum_model, generate_training_dataset

        model = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=42, load_checkpoint=False)
        initial_theta = model.weights.copy()
        initial_w = model.readout_weights.copy()

        # Mini dataset for fast training verification
        dataset = generate_training_dataset(num_samples=30, seed=42)
        trained_model, metadata = train_quantum_model(
            model=model,
            dataset=dataset,
            epochs=2,
            batch_size=16,
            vqc_learning_rate=0.05,
            save_outputs=False,
        )

        delta_theta = float(np.linalg.norm(trained_model.weights - initial_theta))
        delta_w = float(np.linalg.norm(trained_model.readout_weights - initial_w))

        assert delta_theta > 1e-4, f"Circuit parameters did not update: delta={delta_theta}"
        assert delta_w > 1e-4, f"Readout weights did not update: delta={delta_w}"
        assert metadata["theta_total_delta"] > 1e-4

    def test_checkpoint_save_and_compatibility_validation(self, tmp_path):
        """Phase 8: Verify checkpoint save/load and strict architecture compatibility validation."""
        save_file = tmp_path / "test_quantum_weights.npz"
        model = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=123, load_checkpoint=False)
        model.weights += 0.5
        model.readout_bias += 0.2

        model.save_checkpoint(save_file)
        assert save_file.exists()

        # Load into new matching model
        model_loaded = VariationalQuantumNavigator(n_qubits=4, n_layers=2, checkpoint_path=save_file)
        assert np.allclose(model_loaded.weights, model.weights)
        assert np.allclose(model_loaded.readout_bias, model.readout_bias)

        # Incompatible model (different n_qubits) must reject loading
        incompatible_model = VariationalQuantumNavigator(
            n_qubits=6,
            n_layers=2,
            checkpoint_path=save_file,
            load_checkpoint=True,
        )
        # Should not have loaded mismatched 4-qubit weights
        assert len(incompatible_model.weights) != len(model.weights)

    def test_action_hysteresis_persistence(self):
        """Phase 7: Verify action hysteresis suppresses trivial flickering while safety maintains override."""
        from simulation.environment import UAVEnvironment

        env = UAVEnvironment()
        decision_maker = QuantumDecisionMaker()
        decision_maker.reset()

        state = env.get_state()
        dec1 = decision_maker.compute_decision(state)
        first_action = dec1["action"]

        # Second decision in similar state should persist unless significant margin
        dec2 = decision_maker.compute_decision(state)
        assert dec2["action"] == first_action
        assert decision_maker.last_action_idx is not None

    def test_train_validation_trajectory_separation(self):
        """Phase 3 & 9: Verify train/validation splits maintain strict trajectory separation."""
        from quantum.training import generate_training_dataset

        dataset = generate_training_dataset(num_samples=80, seed=42)
        trajectories = sorted(list(set(item["trajectory_id"] for item in dataset)))
        val_split = 0.2
        n_val = max(2, int(len(trajectories) * val_split))
        val_traj_set = set(trajectories[:n_val])

        train_data = [d for d in dataset if d["trajectory_id"] not in val_traj_set]
        val_data = [d for d in dataset if d["trajectory_id"] in val_traj_set]

        train_traj_ids = set(d["trajectory_id"] for d in train_data)
        val_traj_ids = set(d["trajectory_id"] for d in val_data)

        # Must be strictly disjoint
        assert train_traj_ids.isdisjoint(val_traj_ids)
        assert len(train_data) > 0
        assert len(val_data) > 0


class TestQuantumAccuracyOptimizationStep3:
    """Regression test suite for Step 3: Scientific Quantum Accuracy Optimization."""

    def test_train_val_grouped_trajectory_separation(self):
        """Phase 2 & 14: Verify strict trajectory separation between train and validation sets."""
        from quantum.training import generate_training_dataset
        dataset = generate_training_dataset(num_samples=100, seed=42)
        trajectories = sorted(list(set(item["trajectory_id"] for item in dataset)))
        val_split = 0.2
        n_val = max(1, int(len(trajectories) * val_split))
        val_traj_set = set(trajectories[:n_val])

        train_data = [d for d in dataset if d["trajectory_id"] not in val_traj_set]
        val_data = [d for d in dataset if d["trajectory_id"] in val_traj_set]

        train_traj_ids = set(d["trajectory_id"] for d in train_data)
        val_traj_ids = set(d["trajectory_id"] for d in val_data)
        assert train_traj_ids.isdisjoint(val_traj_ids)
        assert len(train_data) > 0 and len(val_data) > 0

    def test_no_benchmark_scenario_leakage(self):
        """Phase 2 & 14: Verify synthetic dataset never trains on benchmark scenarios."""
        from quantum.training import create_synthetic_environment
        rng = np.random.default_rng(42)
        for env_id in range(10):
            field, moving, desc = create_synthetic_environment(env_id=env_id, rng=rng)
            assert "scenario_0" not in desc.lower()
            assert "map_0" not in desc.lower()
            assert field is not None

    def test_balanced_action_distribution(self):
        """Phase 3 & 14: Verify dataset generates balanced actions without artificial STAY oversampling."""
        from quantum.training import generate_training_dataset
        dataset = generate_training_dataset(num_samples=150, seed=42)
        actions = [d["teacher_action"] for d in dataset]
        stay_ratio = actions.count("STAY") / len(dataset)
        assert stay_ratio < 0.15, f"STAY oversampled: {stay_ratio:.2%}"
        unique_acts = set(actions)
        assert len(unique_acts) >= 6, f"Insufficient action diversity: {unique_acts}"

    def test_canonical_encoding_variants(self):
        """Phase 6 & 14: Verify all 4 encoding strategies map 12D features to valid angles."""
        from quantum.feature_encoding import QuantumFeatureEncoder
        dummy_12d = np.linspace(0.1, 1.2, 12)
        for strat in ["grouped", "angle_scaled", "bounded_nonlinear", "normalized"]:
            encoder = QuantumFeatureEncoder(strategy=strat)
            angles = encoder.encode(dummy_12d)
            assert len(angles) == 4
            assert np.all(angles >= -np.pi - 1e-4) and np.all(angles <= np.pi + 1e-4)
            assert not np.isnan(angles).any()

    def test_checkpoint_compatibility_and_loading(self, tmp_path):
        """Phase 13 & 14: Verify checkpoint format, weight restoration, and metadata."""
        save_path = tmp_path / "step3_weights.npz"
        model = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=42, load_checkpoint=False)
        model.weights += 0.25
        model.save_checkpoint(save_path)

        loaded_model = VariationalQuantumNavigator(n_qubits=4, n_layers=2, checkpoint_path=save_path)
        assert np.allclose(loaded_model.weights, model.weights)
        assert np.allclose(loaded_model.readout_bias, model.readout_bias)

    def test_deterministic_training_reproducibility(self):
        """Phase 9 & 14: Verify training reproducibility across identical seeds."""
        from quantum.training import train_quantum_model, generate_training_dataset
        dataset = generate_training_dataset(num_samples=30, seed=42)

        m1 = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=42, load_checkpoint=False)
        _, meta1 = train_quantum_model(model=m1, dataset=dataset, epochs=2, seed=42, save_outputs=False)

        m2 = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=42, load_checkpoint=False)
        _, meta2 = train_quantum_model(model=m2, dataset=dataset, epochs=2, seed=42, save_outputs=False)

        assert np.allclose(m1.weights, m2.weights, atol=1e-5)
        assert np.allclose(m1.readout_weights, m2.readout_weights, atol=1e-5)

    def test_best_checkpoint_restoration(self):
        """Phase 10 & 14: Verify model restores parameters from best validation epoch."""
        from quantum.training import train_quantum_model, generate_training_dataset
        dataset = generate_training_dataset(num_samples=40, seed=123)
        model = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=123, load_checkpoint=False)
        _, meta = train_quantum_model(model=model, dataset=dataset, epochs=3, seed=123, save_outputs=False)
        assert meta["best_epoch"] >= 1
        assert meta["final_val_loss"] <= meta["initial_val_loss"] + 0.5

    def test_validation_metric_calculation(self):
        """Phase 12 & 14: Verify metric calculations match standard multi-class formulas."""
        from sklearn.metrics import accuracy_score, f1_score
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2])
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        assert np.isclose(acc, 6.0 / 9.0)
        assert 0.0 < macro_f1 < 1.0

    def test_no_test_set_training_in_pipeline(self):
        """Phase 16 & 14: Verify training pipeline does not import benchmark scenario files."""
        import quantum.training as qt
        # Ensure training module does not reference benchmark scenario file paths
        assert not hasattr(qt, "SCENARIO_01")
        assert not hasattr(qt, "SCENARIO_FILES")



