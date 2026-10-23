"""Quantum Feature Encoding Module for Q-Nav.

Translates high-dimensional continuous navigation features into quantum state rotation angles
suitable for parameterized single-qubit rotations (Ry, Rz).

Supported Encoding Strategies:
- "grouped": Semantic domain-informed grouping of the 12 features into functional navigation qubits:
    * q0: Target Guidance (norm_target_dist, target_angle_sin, target_angle_cos)
    * q1: Frontal Obstacle Proximity (nearest_obstacle_dist, front_obstacle_dist)
    * q2: Lateral Clearance Differential (left_obstacle_dist vs right_obstacle_dist)
    * q3: Environmental Margins (boundary_proximity, obstacle_density, norm_speed)
    * q4..q7: Higher-order target vectors & kinematics if N > 4
- "pooling": Average and max pooling of feature segments into N qubit angles.
- "repeated": Cyclic angle encoding with alternating phase shifts.
- "linear": Normalized random projection matrix mapping R^12 -> [-pi, pi]^N.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from config import QuantumConfig, DEFAULT_CONFIG


class QuantumFeatureEncoder:
    """Encodes continuous navigation features into quantum rotation angles."""

    def __init__(self, config: Optional[QuantumConfig] = None, strategy: Optional[str] = None):
        """Initialize encoder with target qubit count and angle scaling.

        Args:
            config: Quantum configuration settings.
            strategy: Optional encoding strategy override ("grouped", "angle_scaled", "bounded_nonlinear", "normalized").
        """
        self.config = config or DEFAULT_CONFIG.quantum
        self.n_qubits = self.config.n_qubits
        self.angle_scale = self.config.angle_scale  # Default pi
        self.strategy = strategy or getattr(self.config, "encoding_strategy", self.config.feature_mapping)

        # Projection matrix for linear reduction strategy
        rng = np.random.default_rng(self.config.circuit_seed)
        self.projection_matrix = rng.standard_normal((self.n_qubits, 12))
        norms = np.linalg.norm(self.projection_matrix, axis=1, keepdims=True)
        self.projection_matrix = self.projection_matrix / (norms + 1e-8)

    def select_key_features(self, features_12d: np.ndarray, n_features: Optional[int] = None) -> np.ndarray:
        """Select key navigation features (4D or 6D) for quantum encoding.

        Based on domain knowledge and feature importance analysis:
        1. target_dir_x (index 1) - critical for navigation
        2. target_dir_y (index 2) - critical for navigation
        3. norm_nearest_obs_dist (index 5) - safety
        4. front_obstacle_dist (index 6) - immediate collision avoidance
        5. obstacle_density (index 9) - safety
        6. boundary_proximity (index 11) - boundary safety

        Args:
            features_12d: 12-dimensional feature vector.
            n_features: Number of features to select (4 or 6). If None, defaults to self.n_qubits.

        Returns:
            Selected feature vector normalized to [0, 1].
        """
        features_arr = np.asarray(features_12d, dtype=float)
        k = n_features if n_features is not None else getattr(self, "n_qubits", 4)
        if k == 6:
            # 6 most important features: [1, 2, 5, 6, 9, 11]
            key_indices = [1, 2, 5, 6, 9, 11]
        else:
            key_indices = [1, 2, 5, 9]

        selected = features_arr[key_indices].copy()

        # Normalize to [0, 1] for angle encoding
        denom = float(selected.max() - selected.min())
        if denom < 1e-8:
            selected = np.full_like(selected, 0.5)
        else:
            selected = (selected - selected.min()) / (denom + 1e-8)

        return selected

    def select_key_features_6d(self, features_12d: np.ndarray) -> np.ndarray:
        """Explicitly select 6 most important navigation features [1, 2, 5, 6, 9, 11]."""
        return self.select_key_features(features_12d, n_features=6)

    def select_key_features_4d(self, features_12d: np.ndarray) -> np.ndarray:
        """Explicitly select 4 key features [1, 2, 5, 9]."""
        return self.select_key_features(features_12d, n_features=4)

    def encode(
        self,
        features: np.ndarray,
        feature_dict: Optional[Dict[str, float]] = None,
        strategy: Optional[str] = None,
        n_qubits: Optional[int] = None,
    ) -> np.ndarray:
        """Transform a feature vector (12D, 6D, or 4D) into n_qubits rotation angles in [-pi, pi].

        Args:
            features: 1D array of features (length 12, 6, or 4).
            feature_dict: Optional dictionary with named features.
            strategy: Optional override for encoding strategy.
            n_qubits: Optional override for target qubit count.

        Returns:
            1D array of rotation angles in range [-pi, pi] of length n_qubits.
        """
        target_qubits = n_qubits or self.n_qubits
        feat_arr = np.asarray(features, dtype=float)

        # If already matching target_qubits (e.g. 4 or 6)
        if len(feat_arr) == target_qubits:
            # Map [0, 1] to [-pi, pi]
            return (feat_arr * 2.0 - 1.0) * self.angle_scale

        active_strategy = strategy or self.strategy

        if active_strategy == "key_features" and len(feat_arr) == 12:
            feat_kd = self.select_key_features(feat_arr, n_features=target_qubits)
            return (feat_kd * 2.0 - 1.0) * self.angle_scale

        if active_strategy in ("bounded_nonlinear", "bounded", "nonlinear"):
            return self._encode_bounded_nonlinear(features, feature_dict)
        elif active_strategy in ("angle_scaled", "scaled"):
            return self._encode_angle_scaled(features, feature_dict)
        elif active_strategy in ("grouped_normalized", "normalized"):
            return self._encode_grouped_normalized(features, feature_dict)
        elif active_strategy in ("grouped", "canonical") and self.n_qubits >= 4:
            return self._encode_grouped(features, feature_dict)
        elif active_strategy == "pooling":
            return self._encode_pooling(features)
        elif active_strategy == "repeated":
            return self._encode_repeated(features)
        elif active_strategy in ("linear", "pca"):
            return self._encode_linear(features)
        else:
            return self._encode_grouped(features, feature_dict)

    def _encode_grouped(
        self,
        features: np.ndarray,
        fd: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Domain-informed semantic grouping mapping 12 navigation features to qubits.

        Feature index definitions:
        0: norm_target_dist
        1: target_dir_x
        2: target_dir_y
        3: target_angle_sin
        4: target_angle_cos
        5: nearest_obstacle_dist
        6: front_obstacle_dist
        7: left_obstacle_dist
        8: right_obstacle_dist
        9: obstacle_density
        10: norm_speed
        11: boundary_proximity
        """
        angles = np.zeros(self.n_qubits, dtype=float)

        # Extract values either from dict or array indices
        if fd is not None:
            t_dist = fd.get("norm_target_dist", 0.5)
            sin_t = fd.get("target_angle_sin", 0.0)
            cos_t = fd.get("target_angle_cos", 1.0)
            near_obs = fd.get("norm_nearest_obs_dist", fd.get("nearest_obstacle_dist", 1.0))
            front_obs = fd.get("front_obstacle_dist", 1.0)
            left_obs = fd.get("left_obstacle_dist", 1.0)
            right_obs = fd.get("right_obstacle_dist", 1.0)
            density = fd.get("obstacle_density", 0.0)
            speed = fd.get("norm_speed", 0.5)
            bound_prox = fd.get("boundary_proximity", 1.0)
            t_dir_x = fd.get("target_dir_x", 0.707)
            t_dir_y = fd.get("target_dir_y", 0.707)
        else:
            feat_arr = np.asarray(features, dtype=float)
            if len(feat_arr) < 12:
                feat_arr = np.pad(feat_arr, (0, 12 - len(feat_arr)), constant_values=0.5)
            t_dist, t_dir_x, t_dir_y, sin_t, cos_t = feat_arr[0], feat_arr[1], feat_arr[2], feat_arr[3], feat_arr[4]
            near_obs, front_obs, left_obs, right_obs = feat_arr[5], feat_arr[6], feat_arr[7], feat_arr[8]
            density, speed, bound_prox = feat_arr[9], feat_arr[10], feat_arr[11]

        # Qubit 0: Target Guidance (bearing angle and distance)
        # If target direction vector is provided, encodes world target bearing angle
        if fd is not None and "target_dir_x" in fd and "target_dir_y" in fd:
            angles[0] = float(np.arctan2(t_dir_y, t_dir_x))
        elif fd is not None and "rel_target_angle" in fd:
            # Preserves compatibility with tests using relative angle mock dictionaries
            angles[0] = float(fd["rel_target_angle"])
        elif abs(t_dir_x) > 1e-6 or abs(t_dir_y) > 1e-6:
            # Full 12D feature array provided with valid target_dir_x, target_dir_y
            angles[0] = float(np.arctan2(t_dir_y, t_dir_x))
        else:
            rel_angle = np.arctan2(sin_t, cos_t)
            angles[0] = (0.7 * (rel_angle / np.pi) + 0.3 * (t_dist - 0.5)) * np.pi

        # Qubit 1: Frontal & Nearest Obstacle Warning
        # High value when obstacle directly blocks forward path
        front_danger = 1.0 - front_obs
        near_danger = 1.0 - near_obs
        angles[1] = ((0.6 * front_danger + 0.4 * near_danger) * 2.0 - 1.0) * np.pi

        # Qubit 2: Lateral Clearance Differential (Steering bias: left vs right)
        # Positive if right is blocked (steer left), negative if left is blocked (steer right)
        lateral_diff = left_obs - right_obs
        angles[2] = np.clip(lateral_diff * 1.5, -1.0, 1.0) * np.pi

        # Qubit 3: Environmental Boundary & Cluster Pressure
        env_pressure = (1.0 - bound_prox) * 0.7 + density * 0.5
        angles[3] = (env_pressure * 2.0 - 1.0) * np.pi

        # Higher-order qubits if N > 4
        if self.n_qubits > 4:
            # Qubit 4: Direct target vector Cartesian X/Y
            angles[4] = np.arctan2(t_dir_y, t_dir_x)
        if self.n_qubits > 5:
            # Qubit 5: Kinematic momentum & velocity
            angles[5] = (speed * 2.0 - 1.0) * np.pi
        if self.n_qubits > 6:
            # Qubit 6: Obstacle density modulation
            angles[6] = (density * 2.0 - 1.0) * np.pi
        if self.n_qubits > 7:
            # Qubit 7: Composite safety index
            composite = (bound_prox + near_obs + front_obs) / 3.0
            angles[7] = (composite * 2.0 - 1.0) * np.pi

        return np.clip(angles, -np.pi, np.pi)

    def _encode_angle_scaled(
        self,
        features: np.ndarray,
        fd: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Strategy B: Angle-scaled per-feature linear encoding."""
        feat_12 = np.asarray(features[:12], dtype=float)
        if len(feat_12) < 12:
            feat_12 = np.pad(feat_12, (0, 12 - len(feat_12)), constant_values=0.5)

        angles = np.zeros(self.n_qubits, dtype=float)
        if self.n_qubits >= 4:
            angles[0] = float(feat_12[1] * np.pi)  # t_dir_x in [-1, 1]
            angles[1] = float(feat_12[2] * np.pi)  # t_dir_y in [-1, 1]
            angles[2] = float((feat_12[5] * 2.0 - 1.0) * np.pi)  # nearest obs
            angles[3] = float((feat_12[9] * 2.0 - 1.0) * np.pi)  # density
        if self.n_qubits > 4:
            angles[4] = float((feat_12[6] * 2.0 - 1.0) * np.pi)  # front obs
        if self.n_qubits > 5:
            angles[5] = float((feat_12[7] - feat_12[8]) * np.pi)  # lateral diff
        if self.n_qubits > 6:
            angles[6] = float((feat_12[10] * 2.0 - 1.0) * np.pi)  # speed
        if self.n_qubits > 7:
            angles[7] = float((feat_12[11] * 2.0 - 1.0) * np.pi)  # boundary prox
        return np.clip(angles, -np.pi, np.pi)

    def _encode_bounded_nonlinear(
        self,
        features: np.ndarray,
        fd: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Strategy C: Bounded nonlinear tanh encoding to avoid boundary saturation."""
        feat_12 = np.asarray(features[:12], dtype=float)
        if len(feat_12) < 12:
            feat_12 = np.pad(feat_12, (0, 12 - len(feat_12)), constant_values=0.5)

        angles = np.zeros(self.n_qubits, dtype=float)
        if self.n_qubits >= 4:
            angles[0] = float(np.arctan2(feat_12[2], feat_12[1]))
            angles[1] = float(np.tanh(2.5 * (0.5 - feat_12[6])) * np.pi)
            angles[2] = float(np.tanh(2.0 * (feat_12[7] - feat_12[8])) * np.pi)
            angles[3] = float(np.tanh(2.0 * (feat_12[9] - 0.3)) * np.pi)
        if self.n_qubits > 4:
            angles[4] = float(np.arctan2(feat_12[3], feat_12[4]))
        if self.n_qubits > 5:
            angles[5] = float(np.tanh(2.0 * (feat_12[10] - 0.5)) * np.pi)
        if self.n_qubits > 6:
            angles[6] = float(np.tanh(2.0 * (0.5 - feat_12[11])) * np.pi)
        if self.n_qubits > 7:
            composite = (feat_12[11] + feat_12[5] + feat_12[6]) / 3.0
            angles[7] = float(np.tanh(2.0 * (composite - 0.5)) * np.pi)
        return np.clip(angles, -np.pi, np.pi)

    def _encode_grouped_normalized(
        self,
        features: np.ndarray,
        fd: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Strategy D: Grouped semantic mapping followed by stable L2 normalization."""
        g = self._encode_grouped(features, fd)
        norm = float(np.linalg.norm(g))
        if norm > 1e-6:
            return np.clip(g / norm * np.pi, -np.pi, np.pi)
        return g

    def _encode_pooling(self, features: np.ndarray) -> np.ndarray:
        """Divide the 12 features into N segments and pool each segment into an angle."""
        feat_12 = np.asarray(features[:12], dtype=float)
        if len(feat_12) < 12:
            feat_12 = np.pad(feat_12, (0, 12 - len(feat_12)), constant_values=0.0)

        angles = np.zeros(self.n_qubits, dtype=float)
        segment_size = len(feat_12) / self.n_qubits

        for q in range(self.n_qubits):
            start_idx = int(q * segment_size)
            end_idx = int((q + 1) * segment_size) if q < self.n_qubits - 1 else len(feat_12)
            segment = feat_12[start_idx:end_idx]
            if len(segment) > 0:
                mean_val = float(np.mean(segment))
                # Map [0, 1] or [-1, 1] to [-pi, pi]
                angle = (mean_val if abs(mean_val) <= 1.0 else np.tanh(mean_val)) * np.pi
                angles[q] = angle

        return np.clip(angles, -np.pi, np.pi)

    def _encode_repeated(self, features: np.ndarray) -> np.ndarray:
        """Cyclic angle encoding with alternating Ry and phase offsets."""
        feat_12 = np.asarray(features[:12], dtype=float)
        if len(feat_12) < 12:
            feat_12 = np.pad(feat_12, (0, 12 - len(feat_12)), constant_values=0.0)

        angles = np.zeros(self.n_qubits, dtype=float)
        for q in range(self.n_qubits):
            val = feat_12[q % 12]
            phase_shift = (q // 12) * (np.pi / 4.0)
            angles[q] = np.clip((val * np.pi + phase_shift), -np.pi, np.pi)

        return angles

    def _encode_linear(self, features: np.ndarray) -> np.ndarray:
        """Linear projection from 12D feature space to N qubit angles."""
        feat_12 = np.asarray(features[:12], dtype=float)
        if len(feat_12) < 12:
            feat_12 = np.pad(feat_12, (0, 12 - len(feat_12)), constant_values=0.0)
        reduced = self.projection_matrix @ feat_12
        return np.clip(np.tanh(reduced) * np.pi, -np.pi, np.pi)
