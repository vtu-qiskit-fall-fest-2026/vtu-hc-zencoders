"""Quantum Navigation Decision Module for Q-Nav.

Bridges raw environmental observations to the quantum execution pipeline,
selecting the highest-probability quantum-assisted navigation action.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from config import QNavConfig, DEFAULT_CONFIG, METRICS_DIR
from classical.feature_extractor import NavigationFeatureExtractor
from quantum.feature_encoding import QuantumFeatureEncoder
from quantum.quantum_model import VariationalQuantumNavigator, inspect_checkpoint_architecture
from utils.logging_utils import get_logger

logger = get_logger("quantum_decision")


class QuantumDecisionMaker:
    """Autonomous navigation decision maker powered by a Variational Quantum Circuit."""

    def __init__(
        self,
        config: Optional[QNavConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_checkpoint: Optional[bool] = None,
        strict_load: bool = True,
    ):
        """Initialize feature extractor, quantum encoder, and quantum model.

        Args:
            config: Master configuration.
            checkpoint_path: Optional explicit checkpoint path.
            load_checkpoint: Whether to load checkpoint weights if available. If None, defaults to True when checkpoint_path is given, else False.
            strict_load: Whether to enforce strict architecture compatibility on load.
        """
        import copy
        self.config = copy.deepcopy(config) if config is not None else copy.deepcopy(DEFAULT_CONFIG)
        self.feature_extractor = NavigationFeatureExtractor(
            max_map_distance=np.sqrt(
                self.config.sim.map_width**2 + self.config.sim.map_height**2
            ),
            max_obstacle_distance=self.config.classical.obstacle_influence_distance,
            max_speed=self.config.sim.drone_speed * 1.5,
        )

        should_load = load_checkpoint if load_checkpoint is not None else (checkpoint_path is not None)
        target_ckpt = Path(checkpoint_path) if checkpoint_path is not None else METRICS_DIR / "quantum_weights.npz"
        self.checkpoint_path = target_ckpt if (should_load and target_ckpt.exists()) else None

        # Synchronize architecture with checkpoint if checkpoint exists and loading requested
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            try:
                ckpt_info = inspect_checkpoint_architecture(self.checkpoint_path)
                if (
                    self.config.quantum.n_qubits != ckpt_info["n_qubits"]
                    or self.config.quantum.n_layers != ckpt_info["n_layers"]
                ):
                    logger.info(
                        f"Synchronizing QuantumDecisionMaker with checkpoint architecture: "
                        f"{self.config.quantum.n_qubits}Q{self.config.quantum.n_layers}L -> {ckpt_info['arch_str']}"
                    )
                    self.config.quantum.n_qubits = ckpt_info["n_qubits"]
                    self.config.quantum.n_layers = ckpt_info["n_layers"]
            except Exception as e:
                if strict_load:
                    raise
                logger.warning(f"Could not inspect checkpoint architecture: {e}")

        self.encoder = QuantumFeatureEncoder(config=self.config.quantum)
        self.model = VariationalQuantumNavigator(
            config=self.config.quantum,
            nav_config=self.config.navigation,
            checkpoint_path=self.checkpoint_path,
            load_checkpoint=False,
        )

        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self.model.load_weights(self.checkpoint_path, strict=strict_load)

        self.checkpoint_loaded = getattr(self.model, "checkpoint_loaded", False)

        if hasattr(self.model, "n_qubits") and self.encoder.n_qubits != self.model.n_qubits:
            self.encoder.n_qubits = self.model.n_qubits

        self.action_vectors = self.config.navigation.action_vectors
        self.action_names = self.config.navigation.action_names
        self.last_action_idx: Optional[int] = None
        self.hysteresis_factor: float = getattr(self.config.quantum, "hysteresis_factor", 0.25)

    def reset(self) -> None:
        """Reset internal state including action hysteresis memory."""
        self.last_action_idx = None

    def compute_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute navigation action from current environment state using quantum circuit simulation.

        Args:
            state: Dictionary output from UAVEnvironment.get_state().

        Returns:
            Standardized decision dictionary compatible with ClassicalBaselineNavigator.
        """
        # 1. Feature extraction
        feature_vec, feature_dict = self.feature_extractor.extract_features(state)

        # 2. Quantum feature angle encoding (canonical grouped strategy)
        rotation_angles = self.encoder.encode(
            features=feature_vec, feature_dict=feature_dict
        )

        # 3. Execute Quantum Model (Hardware or Simulator)
        target_dist = float(state.get("target_distance", 10.0))
        import time
        t_qc_start = time.perf_counter()

        is_hw = getattr(self.config.quantum, "real_hardware", False)
        hw_info = None

        if is_hw:
            from quantum.ibm_hardware import execute_on_real_hardware
            hw_res = execute_on_real_hardware(
                features=rotation_angles,
                weights=self.model.weights,
                backend_name=getattr(self.config.quantum, "hardware_backend", None),
                shots=getattr(self.config.quantum, "hardware_shots", 1024),
                timeout=getattr(self.config.quantum, "hardware_timeout", 600),
                retries=getattr(self.config.quantum, "hardware_retries", 0),
                allow_fallback=getattr(self.config.quantum, "hardware_fallback_simulator", False),
                config=self.config,
            )
            hw_info = hw_res
            circuit_exec_time = float(hw_res.get("execution_duration", time.perf_counter() - t_qc_start))
            action_probs = {
                name: float(hw_res["action_probabilities"][i])
                for i, name in enumerate(self.action_names)
            }
            action_scores = dict(action_probs)
            prob_values = np.array([action_probs[k] for k in self.action_names], dtype=float)
            best_idx = int(hw_res["action"])
            self.last_action_idx = best_idx
            chosen_action = hw_res["action_name"]
            chosen_vector = np.array(self.action_vectors[chosen_action], dtype=float)
            confidence = float(hw_res["confidence"])
            basis_probabilities = hw_res.get("raw_basis_probabilities", [])
            probs_dict = hw_res.get("counts", {})
            expectations = np.zeros(self.config.quantum.n_qubits, dtype=float)
        else:
            model_out = self.model.forward(
                features=feature_vec, encoded_angles=rotation_angles, target_distance=target_dist
            )
            circuit_exec_time = time.perf_counter() - t_qc_start
            raw_scores = model_out["action_scores"]
            raw_probs = model_out["action_probabilities"]

            # Controlled action hysteresis/persistence to prevent high-frequency jitter
            fused_scores = np.array([raw_scores[k] for k in self.action_names], dtype=float)
            stay_idx = self.action_names.index("STAY") if "STAY" in self.action_names else 8

            if self.last_action_idx is not None and self.last_action_idx != stay_idx:
                if target_dist > 2.0:
                    # Apply configurable persistence bonus
                    fused_scores[self.last_action_idx] += self.hysteresis_factor
                    # Mild penalty for 180-degree exact reversal
                    prev_name = self.action_names[self.last_action_idx]
                    prev_vec = np.array(self.action_vectors[prev_name], dtype=float)
                    for a_idx, a_name in enumerate(self.action_names):
                        if a_idx != stay_idx:
                            a_vec = np.array(self.action_vectors[a_name], dtype=float)
                            if np.dot(prev_vec, a_vec) < -0.9:
                                fused_scores[a_idx] -= (self.hysteresis_factor * 0.5)

            exp_f = np.exp(fused_scores - np.max(fused_scores))
            prob_values = exp_f / np.sum(exp_f)
            best_idx = int(np.argmax(fused_scores))
            self.last_action_idx = best_idx

            action_scores = {name: float(fused_scores[i]) for i, name in enumerate(self.action_names)}
            action_probs = {name: float(prob_values[i]) for i, name in enumerate(self.action_names)}
            chosen_action = self.action_names[best_idx]
            chosen_vector = np.array(self.action_vectors[chosen_action], dtype=float)
            confidence = float(prob_values[best_idx])
            basis_probabilities = model_out["basis_probabilities"]
            probs_dict = model_out["probs_dict"]
            expectations = model_out["expectations"]

        return {
            "model_type": "quantum_hardware" if is_hw else "quantum",
            "action": chosen_action,
            "direction_vector": chosen_vector,
            "action_scores": action_scores,
            "action_probabilities": action_probs,
            "confidence": confidence,
            "circuit_execution_time": float(circuit_exec_time),
            "feature_vector": feature_vec,
            "feature_dict": feature_dict,
            "rotation_angles": rotation_angles,
            "quantum_expectations": expectations,
            "quantum_probs_dict": probs_dict,
            "basis_probabilities": basis_probabilities,
            "is_real_hardware": is_hw,
            "hardware_metadata": hw_info,
        }

    def _get_action_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Helper returning action probability vector (length 9) for given feature vector.

        Handles both dictionary-like QuantumModelOutput and tuple unpacking.
        """
        feat_arr = np.asarray(features, dtype=float)
        out = self.model.forward(features=feat_arr)
        if isinstance(out, dict) and "probabilities" in out:
            return np.asarray(out["probabilities"], dtype=float)
        try:
            _, _, probs = out
            return np.asarray(probs, dtype=float)
        except Exception:
            pass
        if isinstance(out, dict) and "action_probabilities" in out:
            ap = out["action_probabilities"]
            return np.array([ap.get(k, 0.0) for k in self.action_names], dtype=float)
        raise ValueError("Could not extract action probabilities from model.forward output")

    def _compute_saliency_importance(
        self, features: np.ndarray, action_idx: int, eps: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute saliency-based estimated feature influence using central finite differences.

        dP(action)/dx_i ≈ (P(action | x_i + eps) - P(action | x_i - eps)) / (2 * eps)

        Args:
            features: 1D input feature vector (4D or 12D).
            action_idx: Target action index to explain (0-8).
            eps: Finite difference step size.

        Returns:
            Tuple of (normalized_importance, signed_gradients).
        """
        x = np.asarray(features, dtype=float).copy()
        n_dim = len(x)
        signed_grads = np.zeros(n_dim, dtype=float)

        for i in range(n_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            p_plus = self._get_action_probabilities(x_plus)[action_idx]
            p_minus = self._get_action_probabilities(x_minus)[action_idx]
            signed_grads[i] = (p_plus - p_minus) / (2.0 * eps)

        raw_importance = np.abs(signed_grads)
        total_imp = float(np.sum(raw_importance))

        if total_imp < 1e-12:
            normalized_importance = np.zeros(n_dim, dtype=float)
        else:
            normalized_importance = raw_importance / (total_imp + 1e-8)

        return normalized_importance, signed_grads

    def _compute_gradient_importance(
        self, features: np.ndarray, action_idx: int, eps: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient-based estimated feature sensitivity using central finite differences on log probabilities.

        d(log P(action))/dx_i ≈ (log(P_plus + eps_log) - log(P_minus + eps_log)) / (2 * eps)

        Note: This is a numerical sensitivity estimate, not a causal explanation.

        Args:
            features: 1D input feature vector (4D or 12D).
            action_idx: Target action index to explain (0-8).
            eps: Finite difference step size.

        Returns:
            Tuple of (normalized_importance, signed_gradients).
        """
        x = np.asarray(features, dtype=float).copy()
        n_dim = len(x)
        signed_grads = np.zeros(n_dim, dtype=float)
        eps_log = 1e-8

        for i in range(n_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            p_plus = self._get_action_probabilities(x_plus)[action_idx]
            p_minus = self._get_action_probabilities(x_minus)[action_idx]
            signed_grads[i] = (np.log(p_plus + eps_log) - np.log(p_minus + eps_log)) / (2.0 * eps)

        raw_importance = np.abs(signed_grads)
        total_imp = float(np.sum(raw_importance))

        if total_imp < 1e-12:
            normalized_importance = np.zeros(n_dim, dtype=float)
        else:
            normalized_importance = raw_importance / (total_imp + 1e-8)

        return normalized_importance, signed_grads

    def explain_decision(
        self,
        features: np.ndarray,
        action: Any,
        method: str = "saliency",
        eps: float = 1e-4,
    ) -> Dict[str, Any]:
        """Explain the estimated feature influence on the selected quantum navigation action.

        Supports 'saliency' (direct probability perturbation) and 'gradient' (log-probability sensitivity).
        Correctly maps 4D internal quantum model inputs back to canonical 12D feature names.

        Args:
            features: Input feature vector (12D original or 4D pre-selected).
            action: Action index (0-8) or action name string.
            method: 'saliency' or 'gradient'.
            eps: Finite-difference step size.

        Returns:
            Structured dictionary with normalized importance, signed gradients, and top 3 features.
        """
        from config import (
            FEATURE_NAMES,
            QUANTUM_SELECTED_FEATURE_INDICES,
            QUANTUM_SELECTED_FEATURE_NAMES,
        )

        # Resolve action index and name
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            action_name = (
                self.action_names[action_idx]
                if 0 <= action_idx < len(self.action_names)
                else f"ACTION_{action_idx}"
            )
        else:
            action_name = str(action)
            action_idx = (
                self.action_names.index(action_name)
                if action_name in self.action_names
                else 0
            )

        feat_arr = np.asarray(features, dtype=float)
        use_4d_model = getattr(self.model.config, "use_feature_selection", True)

        # Determine feature mapping and evaluation vector
        if len(feat_arr) == 4:
            x_eval = feat_arr
            features_used = list(QUANTUM_SELECTED_FEATURE_NAMES)
        elif len(feat_arr) == 6:
            x_eval = feat_arr
            features_used = ["target_dir_x", "target_dir_y", "norm_nearest_obs_dist", "front_obstacle_dist", "obstacle_density", "boundary_proximity"]
        elif len(feat_arr) == 12:
            n_q = getattr(self.model, "n_qubits", 4)
            if use_4d_model:
                if n_q == 6:
                    x_eval = self.encoder.select_key_features(feat_arr, n_features=6)
                    features_used = ["target_dir_x", "target_dir_y", "norm_nearest_obs_dist", "front_obstacle_dist", "obstacle_density", "boundary_proximity"]
                else:
                    x_eval = self.encoder.select_key_features(feat_arr, n_features=4)
                    features_used = list(QUANTUM_SELECTED_FEATURE_NAMES)
            else:
                x_eval = feat_arr
                features_used = list(FEATURE_NAMES)
        else:
            x_eval = feat_arr
            features_used = [f"feature_{i}" for i in range(len(feat_arr))]

        # Compute sensitivities
        method_lower = method.lower()
        if method_lower == "gradient":
            importance, signed_grads = self._compute_gradient_importance(
                x_eval, action_idx, eps=eps
            )
        else:
            importance, signed_grads = self._compute_saliency_importance(
                x_eval, action_idx, eps=eps
            )

        # Baseline action probability
        probs = self._get_action_probabilities(x_eval)
        action_prob = float(probs[action_idx]) if 0 <= action_idx < len(probs) else 0.0

        feature_importance: Dict[str, float] = {}
        signed_gradients: Dict[str, float] = {}
        feature_details: Dict[str, Dict[str, float]] = {}

        for i, name in enumerate(features_used):
            imp_val = float(importance[i])
            grad_val = float(signed_grads[i])
            feature_importance[name] = imp_val
            signed_gradients[name] = grad_val
            feature_details[name] = {
                "importance": imp_val,
                "signed_gradient": grad_val,
            }

        # Full 12D mapping (for transparency across all 12 potential features)
        all_feature_importance: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
        all_signed_gradients: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
        for name in features_used:
            if name in all_feature_importance:
                all_feature_importance[name] = feature_importance[name]
                all_signed_gradients[name] = signed_gradients[name]

        # Top 3 features sorted by absolute importance descending
        sorted_indices = np.argsort(-importance)
        top_features: List[Dict[str, Any]] = []
        for rank_idx in sorted_indices[: min(3, len(features_used))]:
            name = features_used[rank_idx]
            top_features.append({
                "feature": name,
                "importance": float(importance[rank_idx]),
                "signed_gradient": float(signed_grads[rank_idx]),
            })

        return {
            "method": method_lower,
            "action": int(action_idx),
            "action_name": str(action_name),
            "action_probability": float(action_prob),
            "features_used": features_used,
            "feature_importance": feature_importance,
            "signed_gradients": signed_gradients,
            "feature_details": feature_details,
            "top_features": top_features,
            "all_feature_importance": all_feature_importance,
            "all_signed_gradients": all_signed_gradients,
        }

