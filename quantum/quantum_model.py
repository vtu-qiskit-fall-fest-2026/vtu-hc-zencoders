"""Variational Quantum Navigation Model for Q-Nav.

Combines a parameterized quantum circuit (VQC) using Qiskit 2.x with a classical readout layer
to evaluate discrete navigation action probabilities from quantum state observables.

Disclaimer:
The action logit projection and softmax layers are classical operations.
Experimental results obtained via quantum simulation do not imply quantum computational advantage.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from config import QuantumConfig, NavigationConfig, DEFAULT_CONFIG
from quantum.feature_encoding import QuantumFeatureEncoder
from quantum.quantum_circuit import ParameterizedNavigationCircuit


class QuantumModelOutput(dict):
    """Dictionary output container that supports 3-element tuple unpacking:
    logits, expectations, probabilities = model.forward(...)
    and array-like indexing for action probabilities.
    """

    def __iter__(self):
        return iter((self["logits"], self["expectations"], self["probabilities"]))

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer, slice)):
            return self["probabilities"][key]
        return super().__getitem__(key)

    def __array__(self, dtype=None):
        return np.asarray(self["probabilities"], dtype=dtype)


def inspect_checkpoint_architecture(
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Inspect checkpoint file and extract architecture parameters and training metadata.

    Args:
        checkpoint_path: Path to the .npz checkpoint file. Defaults to METRICS_DIR / 'quantum_weights.npz'.

    Returns:
        Dict containing n_qubits, n_layers, arch_str, learning_rate, l2_strength, etc.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If required architecture metadata ('n_qubits', 'n_layers') is missing.
    """
    import json
    from config import METRICS_DIR
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else METRICS_DIR / "quantum_weights.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path.resolve()}")

    try:
        data = np.load(ckpt_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to read checkpoint file {ckpt_path.resolve()}: {e}") from e

    if "n_qubits" not in data or "n_layers" not in data:
        raise ValueError(
            f"Checkpoint {ckpt_path.resolve()} is missing required architecture metadata ('n_qubits', 'n_layers')."
        )

    def _extract_scalar(val, target_type):
        if hasattr(val, "item"):
            val = val.item()
        return target_type(val)

    try:
        n_qubits = _extract_scalar(data["n_qubits"], int)
        n_layers = _extract_scalar(data["n_layers"], int)
    except Exception as e:
        raise ValueError(f"Invalid architecture metadata format in {ckpt_path.resolve()}: {e}") from e

    arch_info: Dict[str, Any] = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "arch_str": f"{n_qubits}Q{n_layers}L",
        "checkpoint_path": str(ckpt_path.resolve()),
    }

    # Extract optional metadata fields
    for key, typ in [
        ("learning_rate", float),
        ("l2_strength", float),
        ("training_samples", int),
        ("validation_samples", int),
        ("timestamp", str),
        ("head_type", str),
        ("encoding_strategy", str),
    ]:
        if key in data:
            try:
                arch_info[key] = _extract_scalar(data[key], typ)
            except Exception:
                arch_info[key] = None
        else:
            arch_info[key] = None

    if "weights" in data:
        arch_info["weights_shape"] = tuple(data["weights"].shape)
    else:
        arch_info["weights_shape"] = None

    if "metadata" in data:
        meta_raw = data["metadata"]
        if hasattr(meta_raw, "item"):
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, str):
            try:
                arch_info["metadata"] = json.loads(meta_raw)
            except Exception:
                arch_info["metadata"] = {}
        elif isinstance(meta_raw, dict):
            arch_info["metadata"] = meta_raw
    else:
        arch_info["metadata"] = {}

    return arch_info


class VariationalQuantumNavigator:
    """Hybrid Classical-Quantum Navigation Model using a Variational Quantum Circuit."""

    def __init__(
        self,
        config: Optional[QuantumConfig] = None,
        nav_config: Optional[NavigationConfig] = None,
        n_qubits: Optional[int] = None,
        n_layers: Optional[int] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        head_type: Optional[str] = None,
        encoding_strategy: Optional[str] = None,
        load_checkpoint: Optional[bool] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the variational quantum model.

        Args:
            config: Quantum configuration parameters (qubits, layers, backend, seeds).
            nav_config: Navigation action set configuration.
            n_qubits: Optional qubit count override.
            n_layers: Optional variational layer count override.
            shots: Optional measurement shot count override.
            seed: Optional random seed override.
            head_type: Classification head type: "linear", "mlp", or "hybrid".
            encoding_strategy: Optional feature encoding strategy override.
            load_checkpoint: Whether to load fine-tuned checkpoint if available.
            checkpoint_path: Optional explicit file path to checkpoint.
        """
        base_cfg = config or DEFAULT_CONFIG.quantum
        # Clone or create config with any specified overrides
        from dataclasses import replace
        overrides = {}
        if n_qubits is not None:
            overrides["n_qubits"] = n_qubits
        if n_layers is not None:
            overrides["n_layers"] = n_layers
        if shots is not None:
            overrides["shots"] = shots
        if seed is not None:
            overrides["weight_seed"] = seed

        self.config = replace(base_cfg, **overrides) if overrides else base_cfg
        self.nav_config = nav_config or DEFAULT_CONFIG.navigation
        self.circuit = ParameterizedNavigationCircuit(config=self.config)
        self.encoding_strategy = encoding_strategy or getattr(self.config, "encoding_strategy", "bounded_nonlinear")
        self.encoder = QuantumFeatureEncoder(config=self.config, strategy=self.encoding_strategy)
        self.feature_encoder = self.encoder

        self.n_qubits = self.config.n_qubits
        self.num_weights = self.circuit.get_num_weights()
        self.action_names = self.nav_config.action_names
        self.action_vectors = self.nav_config.action_vectors
        self.checkpoint_path = checkpoint_path
        self.head_type = head_type or getattr(self.config, "head_type", "linear")

        # Default to not loading checkpoint if an explicit test seed was provided and no path given
        if load_checkpoint is None:
            load_checkpoint = (checkpoint_path is not None) or (seed is None)
        self.load_checkpoint_flag = load_checkpoint
        self.checkpoint_loaded = False

        # Initialize variational circuit weights and classical readout projection
        self._init_parameters()
        if self.checkpoint_path is not None:
            self.checkpoint_loaded = self.load_weights(self.checkpoint_path, strict=False)
        elif self.load_checkpoint_flag:
            self.checkpoint_loaded = self.load_weights(strict=False)

    def _init_parameters(self) -> None:
        """Initialize variational weights and readout transformation matrix deterministically."""
        rng = np.random.default_rng(self.config.weight_seed)

        # Variational angles in [0, 2*pi]
        self.weights = rng.uniform(0.0, 2.0 * np.pi, size=self.num_weights)

        # Readout matrix mapping quantum observables to discrete actions
        self.num_basis_features = 1 << self.n_qubits
        total_quantum_features = self.n_qubits + self.num_basis_features

        # Projection matrix with normalized Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (total_quantum_features + len(self.action_names)))
        self.readout_weights = rng.normal(
            loc=0.0, scale=scale, size=(len(self.action_names), total_quantum_features)
        )
        self.readout_bias = np.zeros(len(self.action_names), dtype=float)

        # Initialize MLP head parameters (DQ -> 32 -> 9)
        mlp_h = 32
        scale_mlp1 = np.sqrt(2.0 / (total_quantum_features + mlp_h))
        self.mlp_w1 = rng.normal(loc=0.0, scale=scale_mlp1, size=(mlp_h, total_quantum_features))
        self.mlp_b1 = np.zeros(mlp_h, dtype=float)
        scale_mlp2 = np.sqrt(2.0 / (mlp_h + len(self.action_names)))
        self.mlp_w2 = rng.normal(loc=0.0, scale=scale_mlp2, size=(len(self.action_names), mlp_h))
        self.mlp_b2 = np.zeros(len(self.action_names), dtype=float)

        # Initialize Hybrid head parameters ((DQ + 12) -> 32 -> 9)
        hybrid_in = total_quantum_features + 12
        scale_hyb1 = np.sqrt(2.0 / (hybrid_in + mlp_h))
        self.hybrid_w1 = rng.normal(loc=0.0, scale=scale_hyb1, size=(mlp_h, hybrid_in))
        self.hybrid_b1 = np.zeros(mlp_h, dtype=float)
        self.hybrid_w2 = rng.normal(loc=0.0, scale=scale_mlp2, size=(len(self.action_names), mlp_h))
        self.hybrid_b2 = np.zeros(len(self.action_names), dtype=float)

    def save_checkpoint(
        self,
        filepath: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save complete VQC model checkpoint with architecture parameters and metadata."""
        import json
        from datetime import datetime, timezone
        from config import METRICS_DIR
        out_path = Path(filepath) if filepath is not None else METRICS_DIR / "quantum_weights.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = dict(metadata or {})
        now_ts = datetime.now(timezone.utc).isoformat()
        if "timestamp" not in meta:
            meta["timestamp"] = now_ts

        save_dict = {
            "weights": self.weights,
            "readout_weights": self.readout_weights,
            "readout_bias": self.readout_bias,
            "n_qubits": np.int64(self.n_qubits),
            "n_layers": np.int64(self.circuit.n_layers),
            "head_type": str(self.head_type),
            "encoding_strategy": str(getattr(self.config, "encoding_strategy", "grouped")),
            "action_names": np.array(self.action_names),
            "learning_rate": np.float64(meta.get("learning_rate", getattr(self.config, "learning_rate", 0.005))),
            "l2_strength": np.float64(meta.get("l2_strength", getattr(self.config, "l2_strength", 0.1))),
            "training_samples": np.int64(meta.get("training_samples", getattr(self.config, "training_samples", 0))),
            "validation_samples": np.int64(meta.get("validation_samples", getattr(self.config, "validation_samples", 0))),
            "timestamp": str(meta["timestamp"]),
            "metadata": json.dumps(meta),
        }
        if hasattr(self, "mlp_w1"):
            save_dict["mlp_w1"] = self.mlp_w1
            save_dict["mlp_b1"] = self.mlp_b1
            save_dict["mlp_w2"] = self.mlp_w2
            save_dict["mlp_b2"] = self.mlp_b2
        if hasattr(self, "hybrid_w1"):
            save_dict["hybrid_w1"] = self.hybrid_w1
            save_dict["hybrid_b1"] = self.hybrid_b1
            save_dict["hybrid_w2"] = self.hybrid_w2
            save_dict["hybrid_b2"] = self.hybrid_b2

        np.savez_compressed(out_path, **save_dict)
        return out_path

    def save_weights(
        self,
        filepath: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Alias for save_checkpoint."""
        return self.save_checkpoint(filepath, metadata=metadata)

    def load_weights(
        self,
        filepath: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> bool:
        """Load and strictly validate checkpoint weights into the model.

        Args:
            filepath: Optional path to the .npz checkpoint.
            strict: If True, raises ValueError on architecture or shape mismatch.
                    If False, logs a warning and returns False.

        Returns:
            True if loaded successfully, False if file doesn't exist (when strict=False) or incompatible.

        Raises:
            FileNotFoundError: If filepath does not exist and strict=True.
            ValueError: If architecture (n_qubits, n_layers) or weights shape mismatches and strict=True.
        """
        from config import METRICS_DIR
        from utils.logging_utils import get_logger
        logger = get_logger("quantum_model")
        ckpt_path = Path(filepath) if filepath is not None else METRICS_DIR / "quantum_weights.npz"

        if not ckpt_path.exists():
            if strict:
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path.resolve()}")
            return False

        try:
            saved = np.load(ckpt_path, allow_pickle=True)

            def _extract_scalar(val, target_type):
                if hasattr(val, "item"):
                    val = val.item()
                return target_type(val)

            ckpt_qubits = _extract_scalar(saved["n_qubits"], int) if "n_qubits" in saved else self.n_qubits
            ckpt_layers = _extract_scalar(saved["n_layers"], int) if "n_layers" in saved else self.circuit.n_layers

            if ckpt_qubits != self.n_qubits or ckpt_layers != self.circuit.n_layers:
                msg = (
                    f"Checkpoint architecture mismatch: "
                    f"model={self.n_qubits}Q{self.circuit.n_layers}L, "
                    f"checkpoint={ckpt_qubits}Q{ckpt_layers}L. "
                    f"Cannot load incompatible checkpoint."
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
                return False

            if saved["weights"].shape != self.weights.shape:
                msg = (
                    f"Checkpoint weights shape mismatch: "
                    f"model={self.weights.shape}, checkpoint={saved['weights'].shape}."
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
                return False

            self.weights = np.asarray(saved["weights"], dtype=float)
            if "readout_weights" in saved:
                self.readout_weights = np.asarray(saved["readout_weights"], dtype=float)
            if "readout_bias" in saved:
                self.readout_bias = np.asarray(saved["readout_bias"], dtype=float)

            if "head_type" in saved:
                h_val = saved["head_type"]
                self.head_type = str(h_val.item() if hasattr(h_val, "item") else h_val)
            if "mlp_w1" in saved:
                self.mlp_w1 = np.asarray(saved["mlp_w1"], dtype=float)
                self.mlp_b1 = np.asarray(saved["mlp_b1"], dtype=float)
                self.mlp_w2 = np.asarray(saved["mlp_w2"], dtype=float)
                self.mlp_b2 = np.asarray(saved["mlp_b2"], dtype=float)
            if "hybrid_w1" in saved:
                self.hybrid_w1 = np.asarray(saved["hybrid_w1"], dtype=float)
                self.hybrid_b1 = np.asarray(saved["hybrid_b1"], dtype=float)
                self.hybrid_w2 = np.asarray(saved["hybrid_w2"], dtype=float)
                self.hybrid_b2 = np.asarray(saved["hybrid_b2"], dtype=float)

            self.checkpoint_loaded = True
            return True

        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            if strict:
                raise ValueError(f"Failed to load checkpoint from {ckpt_path}: {e}") from e
            logger.warning(f"Failed to load checkpoint from {ckpt_path}: {e}")
            return False

    def load_checkpoint(
        self,
        filepath: Optional[Union[str, Path]] = None,
        strict: bool = False,
    ) -> bool:
        """Load and validate checkpoint into the current model.

        Args:
            filepath: Optional path to the .npz checkpoint.
            strict: Whether to raise ValueError on mismatch (default: False).
        """
        return self.load_weights(filepath, strict=strict)

    def forward(
        self,
        features: Optional[np.ndarray] = None,
        encoded_angles: Optional[np.ndarray] = None,
        target_distance: float = 10.0,
    ) -> QuantumModelOutput:
        """Execute quantum circuit simulation and produce action logits and probabilities.

        Can be unpacked directly:
            logits, expectations, probabilities = model.forward(features)
        Or accessed as a dictionary:
            output["action_scores"], output["action_probabilities"]

        Args:
            features: 1D array of rotation angles (length == n_qubits) OR raw 12D features.
            encoded_angles: Alias for features.
            target_distance: Distance to target.

        Returns:
            QuantumModelOutput with logits, expectations, probabilities, and telemetry.
        """
        feat_arr = None
        if features is not None:
            feat_arr = np.asarray(features, dtype=float)
            if encoded_angles is None:
                if len(feat_arr) == 12:
                    # Canonical grouped encoding across all 12 navigation features
                    encoded_angles = self.feature_encoder.encode(feat_arr, n_qubits=self.n_qubits)
                elif len(feat_arr) == self.n_qubits:
                    encoded_angles = np.clip(feat_arr, -np.pi, np.pi)
                else:
                    encoded_angles = self.feature_encoder.encode(feat_arr, n_qubits=self.n_qubits)
        elif encoded_angles is not None:
            encoded_angles = np.asarray(encoded_angles, dtype=float)
        else:
            raise ValueError("Must provide features or encoded_angles to model.forward()")

        if len(encoded_angles) != self.circuit.n_qubits:
            if len(encoded_angles) > self.circuit.n_qubits:
                encoded_angles = encoded_angles[:self.circuit.n_qubits]
            else:
                encoded_angles = np.pad(
                    encoded_angles,
                    (0, self.circuit.n_qubits - len(encoded_angles)),
                    mode='constant',
                    constant_values=0.0
                )

        # 1. Execute actual quantum circuit simulation on Qiskit backend
        expectations, basis_probs, probs_dict = self.circuit.execute(
            features=encoded_angles,
            weights=self.weights,
        )

        # 2. Assemble complete quantum feature vector: expectations + all 2^N basis probabilities
        quantum_features = np.concatenate([expectations, basis_probs])

        # 3. Project quantum features into action logits according to configured head
        if self.head_type == "mlp" and hasattr(self, "mlp_w1"):
            h = np.maximum(0.0, self.mlp_w1 @ quantum_features + self.mlp_b1)
            logits = self.mlp_w2 @ h + self.mlp_b2
        elif self.head_type == "hybrid" and hasattr(self, "hybrid_w1"):
            c_feats = feat_arr[:12] if (feat_arr is not None and len(feat_arr) >= 12) else np.zeros(12)
            comb = np.concatenate([quantum_features, c_feats])
            h = np.maximum(0.0, self.hybrid_w1 @ comb + self.hybrid_b1)
            logits = self.hybrid_w2 @ h + self.hybrid_b2
        else:
            logits = self.readout_weights @ quantum_features + self.readout_bias

        # 4. Compute softmax action probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        action_scores = {
            name: float(logits[i]) for i, name in enumerate(self.action_names)
        }
        action_probs = {
            name: float(probabilities[i]) for i, name in enumerate(self.action_names)
        }

        output = QuantumModelOutput({
            "logits": logits,
            "expectations": expectations,
            "probabilities": probabilities,
            "action_scores": action_scores,
            "action_probabilities": action_probs,
            "basis_probabilities": basis_probs,
            "probs_dict": probs_dict,
            "quantum_features": quantum_features,
            "encoded_angles": encoded_angles,
        })
        return output

    def train_with_regularization(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_epochs=100,
        learning_rate=0.01,
        l2_strength=0.1,
        early_stopping=True,
    ):
        """Train VQC with L2 regularization and early stopping

        Args:
            X_train: Training features (N_train, 4 or 6)
            y_train: Training labels (N_train,)
            X_val: Validation features (N_val, 4 or 6)
            y_val: Validation labels (N_val,)
            n_epochs: Maximum training epochs
            learning_rate: Optimizer learning rate
            l2_strength: L2 regularization coefficient
            early_stopping: Whether to use early stopping

        Returns:
            Training history dictionary
        """
        import numpy as np
        from scipy.optimize import minimize
        from sklearn.linear_model import LogisticRegression

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'weights': []
        }

        best_val_loss = float('inf')
        best_weights = self.weights.copy()
        patience = 10
        patience_counter = 0

        def fit_readout(w_flat):
            """Align readout projection matrix with L2 regularization on quantum features."""
            try:
                self.weights = w_flat.reshape(self.weights.shape)
                q_tr = np.array([self.forward(x)["quantum_features"] for x in X_train])
                C_val = 1.0 / max(1e-4, float(l2_strength))
                clf = LogisticRegression(C=C_val, max_iter=200, solver='lbfgs')
                clf.fit(q_tr, y_train)
                n_classes = len(self.action_names)
                new_W = np.zeros((n_classes, q_tr.shape[1]))
                new_b = np.zeros(n_classes)
                if len(clf.classes_) == 2:
                    c0, c1 = clf.classes_[0], clf.classes_[1]
                    if c1 < n_classes:
                        new_W[c1] = clf.coef_[0]
                        new_b[c1] = clf.intercept_[0]
                    if c0 < n_classes:
                        new_W[c0] = -clf.coef_[0]
                        new_b[c0] = -clf.intercept_[0]
                else:
                    for idx, c in enumerate(clf.classes_):
                        if c < n_classes:
                            new_W[c] = clf.coef_[idx]
                            new_b[c] = clf.intercept_[idx]
                self.readout_weights = new_W
                self.readout_bias = new_b
            except Exception:
                pass

        # Align readout weights initially
        fit_readout(self.weights.flatten())

        def evaluate_loss_acc(weights_flat, X, y, max_eval=60):
            """Evaluate regularized cross-entropy loss and accuracy in a single pass."""
            self.weights = weights_flat.reshape(self.weights.shape)
            n_eval = min(len(X), max_eval) if max_eval > 0 else len(X)
            sub_X, sub_y = X[:n_eval], y[:n_eval]
            total_loss = 0.0
            correct = 0
            for features, label in zip(sub_X, sub_y):
                probs = self.forward(features)
                ce_loss = -np.log(probs[label] + 1e-8)
                total_loss += ce_loss
                if np.argmax(probs) == label:
                    correct += 1
            l2_reg = 0.5 * l2_strength * np.sum(self.weights ** 2)
            loss_val = float(total_loss / max(1, n_eval) + l2_reg)
            acc_val = float(correct / max(1, n_eval))
            return loss_val, acc_val

        # Training loop
        weights_flat = self.weights.flatten()
        batch_sz = min(5, len(X_train))
        rng_opt = np.random.default_rng(42)

        for epoch in range(n_epochs):
            # Select batch for fast finite-difference step
            if len(X_train) > batch_sz:
                b_idx = rng_opt.choice(len(X_train), size=batch_sz, replace=False)
                X_batch, y_batch = X_train[b_idx], y_train[b_idx]
            else:
                X_batch, y_batch = X_train, y_train

            def batch_loss_fn(w_flat):
                self.weights = w_flat.reshape(self.weights.shape)
                b_tot = 0.0
                for feats, lbl in zip(X_batch, y_batch):
                    probs = self.forward(feats)
                    b_tot += -np.log(probs[lbl] + 1e-8)
                l2 = 0.5 * l2_strength * np.sum(self.weights ** 2)
                return float(b_tot / max(1, len(X_batch)) + l2)

            # Optimize loss using L-BFGS-B
            result = minimize(
                batch_loss_fn,
                x0=weights_flat,
                method='L-BFGS-B',
                options={'maxiter': 1}
            )

            weights_flat = result.x
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                fit_readout(weights_flat)

            # Compute metrics in a single pass
            is_last = (epoch == n_epochs - 1)
            eval_limit_tr = 0 if is_last else 60
            eval_limit_v = 0 if is_last else 40
            train_loss, train_acc = evaluate_loss_acc(weights_flat, X_train, y_train, max_eval=eval_limit_tr)
            val_loss, val_acc = evaluate_loss_acc(weights_flat, X_val, y_val, max_eval=eval_limit_v)

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['train_acc'].append(float(train_acc))
            history['val_acc'].append(float(val_acc))
            history['weights'].append(weights_flat.copy())

            # Print progress every 10 epochs or first/last
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch+1}/{n_epochs}:", flush=True)
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%", flush=True)
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", flush=True)
                print(f"  L2 Reg: {0.5 * l2_strength * np.sum(self.weights ** 2):.4f}", flush=True)

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = weights_flat.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
                        print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
                        break

        # Restore best weights
        self.weights = best_weights.reshape(self.weights.shape)
        fit_readout(self.weights.flatten())

        return history


class EnsembleQuantumNavigator:
    """Ensemble of multiple quantum models for better generalization.

    Args:
        n_models: Number of models in ensemble
        model_config: Configuration dictionary for each model
    """

    def __init__(self, n_models=5, model_config=None):
        self.n_models = n_models
        self.models = []
        cfg = model_config or {}
        self.config = cfg

        # Create ensemble with diverse random seeds
        for i in range(n_models):
            m_cfg = dict(cfg)
            if "seed" not in m_cfg:
                m_cfg["seed"] = 42 + i * 17
            self.models.append(VariationalQuantumNavigator(**m_cfg))

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train all models with different random seeds and noisy feature perturbations."""
        import numpy as np
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.n_models}...")
            # Add noise to training data for diversity
            X_train_noisy = X_train + np.random.randn(*X_train.shape) * 0.02

            model.train_with_regularization(
                X_train_noisy, y_train, X_val, y_val,
                n_epochs=50,
                learning_rate=0.01,
                l2_strength=0.1,
                early_stopping=True
            )

    def forward(self, features):
        """Average predictions from all models."""
        import numpy as np
        all_probs = []

        for model in self.models:
            probs = model.forward(features)
            all_probs.append(np.asarray(probs))

        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        avg_probs /= avg_probs.sum()  # Normalize

        return avg_probs

    def compute_decision(self, state):
        """Compute navigation decision compatible with NavigationController."""
        import numpy as np
        m0 = self.models[0]
        if "feature_vector" in state:
            f_vec = state["feature_vector"]
        else:
            f_vec, _ = m0.feature_encoder.extract_features(state) if hasattr(m0.feature_encoder, "extract_features") else (np.zeros(m0.n_qubits), {})
        probs = self.forward(f_vec)
        best_idx = int(np.argmax(probs))
        action_name = m0.action_names[best_idx]
        chosen_vec = np.array(m0.action_vectors[action_name], dtype=float)
        return {
            "model_type": "quantum_ensemble",
            "action": action_name,
            "direction_vector": chosen_vec,
            "action_probabilities": {name: float(probs[i]) for i, name in enumerate(m0.action_names)},
            "confidence": float(probs[best_idx]),
        }
