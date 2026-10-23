import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import DEFAULT_CONFIG, QNavConfig
from quantum.quantum_model import VariationalQuantumNavigator
from quantum.feature_encoding import QuantumFeatureEncoder
from quantum.training import generate_training_dataset, create_synthetic_environment, teacher_policy
from classical.feature_extractor import NavigationFeatureExtractor

def build_controlled_dataset(num_samples=1800, seed=42):
    """Generate trajectory-isolated dataset across synthetic environments with zero leakage."""
    print(f"Generating controlled dataset (target: {num_samples} samples, seed: {seed})...", flush=True)
    raw_data = generate_training_dataset(num_samples=num_samples, seed=seed)
    
    # Trajectory-level split
    trajs = sorted(list(set(item["trajectory_id"] for item in raw_data)))
    rng = np.random.default_rng(seed)
    shuffled_trajs = rng.permutation(trajs)
    n_val_trajs = max(2, int(len(trajs) * 0.20))
    val_traj_set = set(shuffled_trajs[:n_val_trajs])
    
    train_items = [d for d in raw_data if d["trajectory_id"] not in val_traj_set]
    val_items = [d for d in raw_data if d["trajectory_id"] in val_traj_set]
    
    print(f"Dataset split: {len(train_items)} train samples ({len(trajs)-n_val_trajs} trajs) | {len(val_items)} val samples ({n_val_trajs} trajs)", flush=True)
    
    # Class distribution check
    y_tr = np.array([d["teacher_index"] for d in train_items])
    y_val = np.array([d["teacher_index"] for d in val_items])
    print(f"Train action counts: {dict(zip(*np.unique(y_tr, return_counts=True)))}", flush=True)
    print(f"Val action counts:   {dict(zip(*np.unique(y_val, return_counts=True)))}", flush=True)
    
    return train_items, val_items

def extract_quantum_representations(train_items, val_items, n_qubits=6, n_layers=3, encoding_strategy="informative", seed=42):
    """Extract quantum circuit observables (expectations + basis probabilities) using VQC."""
    model = VariationalQuantumNavigator(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
        load_checkpoint=False,
    )
    
    def encode_vector(raw_12d):
        f = np.asarray(raw_12d, dtype=float)
        angles = np.zeros(n_qubits, dtype=float)
        # 1. Target bearing angle (-pi to pi)
        angles[0] = float(np.arctan2(f[2], f[1]))
        # 2. Normalized target distance mapped to [-pi, pi]
        angles[1] = float((np.clip(f[0], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
        # 3. Front obstacle danger (-pi to pi, high if close)
        angles[2] = float((1.0 - np.clip(f[6], 0.0, 1.0) * 2.0) * np.pi)
        # 4. Nearest obstacle distance in [-pi, pi]
        angles[3] = float((np.clip(f[5], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
        # 5. Lateral steering clearance differential (left vs right)
        angles[4] = float(np.clip(f[7] - f[8], -1.0, 1.0) * np.pi)
        # 6. Environmental cluster density & boundary in [-pi, pi]
        angles[5] = float((np.clip(f[9], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
        
        if n_qubits > 6:
            # 7. Speed / kinetic momentum
            angles[6] = float((np.clip(f[10], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
        if n_qubits > 7:
            # 8. Boundary proximity
            angles[7] = float((np.clip(f[11], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
            
        return angles

    print(f"Extracting quantum observables for {n_qubits}Q{n_layers}L across dataset...", flush=True)
    t0 = time.time()
    
    X_q_train = []
    for d in train_items:
        ang = encode_vector(d["features"])
        exp_vals, basis_probs, _ = model.circuit.execute(ang, model.weights)
        X_q_train.append(np.concatenate([exp_vals, basis_probs]))
    X_q_train = np.array(X_q_train, dtype=np.float32)
    
    X_q_val = []
    for d in val_items:
        ang = encode_vector(d["features"])
        exp_vals, basis_probs, _ = model.circuit.execute(ang, model.weights)
        X_q_val.append(np.concatenate([exp_vals, basis_probs]))
    X_q_val = np.array(X_q_val, dtype=np.float32)
    
    X_c_train = np.array([d["features"] for d in train_items], dtype=np.float32)
    X_c_val = np.array([d["features"] for d in val_items], dtype=np.float32)
    
    y_train = np.array([d["teacher_index"] for d in train_items], dtype=np.int64)
    y_val = np.array([d["teacher_index"] for d in val_items], dtype=np.int64)
    
    elapsed = time.time() - t0
    print(f"Extracted quantum features (Dim: {X_q_train.shape[1]}) in {elapsed:.2f}s", flush=True)
    return model, X_q_train, X_q_val, X_c_train, X_c_val, y_train, y_val

if __name__ == "__main__":
    train_items, val_items = build_controlled_dataset(num_samples=1800, seed=42)
    model, X_q_tr, X_q_v, X_c_tr, X_c_v, y_tr, y_v = extract_quantum_representations(
        train_items, val_items, n_qubits=6, n_layers=3, seed=42
    )
    
    # Test Linear Head with Class Weighting
    print("\n--- Model 1: Quantum Linear Head (Balanced) ---")
    clf_lin = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500)
    clf_lin.fit(X_q_tr, y_tr)
    pred_v = clf_lin.predict(X_q_v)
    acc_v = accuracy_score(y_v, pred_v) * 100
    f1_m = f1_score(y_v, pred_v, average='macro')
    print(f"Linear Head Val Acc: {acc_v:.2f}%, Macro F1: {f1_m:.4f}")
    
    # Test MLP Head on Quantum Features
    print("\n--- Model 2: Quantum MLP Head ---")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    mlp.fit(X_q_tr, y_tr)
    pred_mlp = mlp.predict(X_q_v)
    acc_mlp = accuracy_score(y_v, pred_mlp) * 100
    f1_mlp = f1_score(y_v, pred_mlp, average='macro')
    print(f"Quantum MLP Head Val Acc: {acc_mlp:.2f}%, Macro F1: {f1_mlp:.4f}")
    
    # Test Hybrid Residual Head (Quantum + Classical Features)
    print("\n--- Model 3: Residual Hybrid Head (Quantum 70D + Classical 12D) ---")
    X_hyb_tr = np.concatenate([X_q_tr, X_c_tr], axis=1)
    X_hyb_v = np.concatenate([X_q_v, X_c_v], axis=1)
    mlp_hyb = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    mlp_hyb.fit(X_hyb_tr, y_tr)
    pred_hyb = mlp_hyb.predict(X_hyb_v)
    acc_hyb = accuracy_score(y_v, pred_hyb) * 100
    f1_hyb = f1_score(y_v, pred_hyb, average='macro')
    print(f"Residual Hybrid Head Val Acc: {acc_hyb:.2f}%, Macro F1: {f1_hyb:.4f}")
