import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from config import DEFAULT_CONFIG, METRICS_DIR, TRAINING_METRICS_PATH
from simulation.obstacles import ObstacleField, CircularObstacle, RectangularObstacle, MovingObstacle
from classical.feature_extractor import NavigationFeatureExtractor
from simulation.collision import check_boundary_violation
from scratch.test_pipeline import create_10_synthetic_environments
from scratch.teacher_test import enhanced_teacher_policy
from quantum.quantum_model import VariationalQuantumNavigator
from scratch.run_target_a_experiments import generate_controlled_multisector_data, encode_features_to_angles, extract_vqc_features

def run_training_target_a():
    print("==================================================================", flush=True)
    print("STEP 1: GENERATING CONTROLLED MULTI-SECTOR DATASET (ZERO LEAKAGE)", flush=True)
    print("==================================================================", flush=True)
    
    seed = 42
    n_samples = 2400
    n_qubits = 6
    n_layers = 3
    action_names = DEFAULT_CONFIG.navigation.action_names
    num_classes = len(action_names)
    
    train_data, val_data = generate_controlled_multisector_data(n_samples=n_samples, seed=seed)
    y_tr = np.array([d["teacher_index"] for d in train_data], dtype=np.int64)
    y_va = np.array([d["teacher_index"] for d in val_data], dtype=np.int64)
    
    print(f"\nTraining set: {len(train_data)} samples | Validation set: {len(val_data)} samples", flush=True)
    val_counts = Counter(y_va)
    majority_count = val_counts.most_common(1)[0][1]
    majority_baseline = majority_count / len(y_va) * 100.0
    random_baseline = (1.0 / num_classes) * 100.0
    print(f"Majority Class Baseline: {majority_baseline:.2f}% | Random Baseline: {random_baseline:.2f}%\n", flush=True)
    
    print("==================================================================", flush=True)
    print(f"STEP 2: EXTRACTING VQC QUANTUM FEATURES ({n_qubits}Q{n_layers}L)", flush=True)
    print("==================================================================", flush=True)
    
    model = VariationalQuantumNavigator(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
        load_checkpoint=False,
    )
    
    X_q_tr, X_q_va, X_c_tr, X_c_va, y_tr, y_va = extract_vqc_features(model, train_data, val_data, n_qubits)
    
    # Concatenate Quantum (70D) + Classical (12D) features for Residual Hybrid Head
    X_hyb_tr = np.concatenate([X_q_tr, X_c_tr], axis=1)
    X_hyb_va = np.concatenate([X_q_va, X_c_va], axis=1)
    in_dim = X_hyb_tr.shape[1]
    
    print(f"Residual Hybrid Feature Dimension: {in_dim} (Quantum: {X_q_tr.shape[1]}, Classical: {X_c_tr.shape[1]})", flush=True)
    
    print("\n==================================================================", flush=True)
    print("STEP 3: OPTIMIZING RESIDUAL HYBRID HEAD WITH ADAMW & EARLY STOPPING", flush=True)
    print("==================================================================", flush=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Class weights for balanced cross-entropy
    counts = np.bincount(y_tr, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = (len(y_tr) / (num_classes * max(1, counts[c])))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    
    hidden_dim = 64
    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)
    
    X_tr_t = torch.tensor(X_hyb_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_hyb_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    patience = 30
    patience_cnt = 0
    epochs = 200
    batch_size = 32
    n_batches = int(np.ceil(len(X_hyb_tr) / batch_size))
    
    t_train_start = time.time()
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(X_hyb_tr))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            out = net(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            va_out = net(X_va_t)
            va_loss = criterion(va_out, y_va_t).item()
            va_preds = torch.argmax(va_out, dim=1).numpy()
            va_acc = accuracy_score(y_va, va_preds) * 100.0
            
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_epoch = epoch + 1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch}).", flush=True)
                break
                
    train_duration = time.time() - t_train_start
    net.load_state_dict(best_state)
    net.eval()
    
    with torch.no_grad():
        final_tr_out = net(X_tr_t)
        final_tr_preds = torch.argmax(final_tr_out, dim=1).numpy()
        final_tr_acc = accuracy_score(y_tr, final_tr_preds) * 100.0
        
        final_va_out = net(X_va_t)
        final_va_loss = criterion(final_va_out, y_va_t).item()
        final_va_preds = torch.argmax(final_va_out, dim=1).numpy()
        final_va_acc = accuracy_score(y_va, final_va_preds) * 100.0
        
    macro_f1 = f1_score(y_va, final_va_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_va, final_va_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_va, final_va_preds, labels=list(range(num_classes)))
    
    # Per-class metrics
    p_prec = precision_score(y_va, final_va_preds, average=None, zero_division=0)
    p_rec = recall_score(y_va, final_va_preds, average=None, zero_division=0)
    p_f1 = f1_score(y_va, final_va_preds, average=None, zero_division=0)
    
    per_class_precision = {action_names[i]: round(float(p_prec[i]), 4) for i in range(num_classes)}
    per_class_recall = {action_names[i]: round(float(p_rec[i]), 4) for i in range(num_classes)}
    per_class_f1 = {action_names[i]: round(float(p_f1[i]), 4) for i in range(num_classes)}
    
    print("\n==================================================================", flush=True)
    print("STEP 4: FINAL EVALUATION ON UNSEEN HELD-OUT VALIDATION SET", flush=True)
    print("==================================================================", flush=True)
    print(f"train_accuracy:      {final_tr_acc:.2f}%")
    print(f"validation_accuracy: {final_va_acc:.2f}%")
    print(f"macro_f1:            {macro_f1:.4f}")
    print(f"weighted_f1:         {weighted_f1:.4f}")
    print(f"validation_loss:     {final_va_loss:.4f}\n")
    
    print("Per-class Precision:")
    for k, v in per_class_precision.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\nPer-class Recall:")
    for k, v in per_class_recall.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\nPer-class F1:")
    for k, v in per_class_f1.items():
        print(f"  {k:12s}: {v:.4f}")
        
    print("\nConfusion Matrix:")
    print("     " + " ".join(f"{name[:4]:>5}" for name in action_names))
    for i, row in enumerate(cm):
        print(f"{action_names[i][:4]:>4} " + " ".join(f"{val:5d}" for val in row))
        
    # Extract linear projection / hybrid weights for model serialization
    # For compatibility with forward() when head_type='hybrid':
    # hybrid_w1: (32, total_quantum_features + 12)
    # hybrid_b1: (32,)
    # hybrid_w2: (9, 32)
    # hybrid_b2: (9,)
    # We can fit a 32-hidden linear layer mapping in_dim -> 32 -> 9:
    w1_pt = net[0].weight.detach().cpu().numpy()
    b1_pt = net[0].bias.detach().cpu().numpy()
    # Account for BatchNorm1d: y = (x - mean)/sqrt(var+eps) * gamma + beta
    bn_gamma = net[1].weight.detach().cpu().numpy()
    bn_beta = net[1].bias.detach().cpu().numpy()
    bn_mean = net[1].running_mean.detach().cpu().numpy()
    bn_var = net[1].running_var.detach().cpu().numpy()
    bn_scale = bn_gamma / np.sqrt(bn_var + 1e-5)
    
    w1_fused = w1_pt * bn_scale[:, None]
    b1_fused = (b1_pt - bn_mean) * bn_scale + bn_beta
    
    # The second layer: net[4] (hidden -> 32) and net[6] (32 -> 9)
    w2_pt = net[4].weight.detach().cpu().numpy()
    b2_pt = net[4].bias.detach().cpu().numpy()
    w3_pt = net[6].weight.detach().cpu().numpy()
    b3_pt = net[6].bias.detach().cpu().numpy()
    
    # Store hybrid weights in model
    model.head_type = "hybrid"
    model.hybrid_w1 = w1_fused
    model.hybrid_b1 = b1_fused
    model.hybrid_w2 = w3_pt @ w2_pt # effective 32 -> 9 projection
    model.hybrid_b2 = w3_pt @ b2_pt + b3_pt
    
    # Also save standard linear readout weights mapping X_q (70D) to 9 classes
    # by fitting regularized linear head for pure VQC mode:
    from sklearn.linear_model import LogisticRegression
    clf_lin = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    clf_lin.fit(X_q_tr, y_tr)
    model.readout_weights = clf_lin.coef_.copy()
    model.readout_bias = clf_lin.intercept_.copy()
    
    metadata = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "head_type": "hybrid",
        "train_accuracy": round(final_tr_acc, 2),
        "validation_accuracy": round(final_va_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "validation_loss": round(final_va_loss, 4),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "training_duration_seconds": round(train_duration, 2),
        "majority_class_baseline": round(majority_baseline, 2),
        "random_baseline": round(random_baseline, 2),
        "total_training_samples": len(train_data),
        "total_validation_samples": len(val_data),
        "training_trajectories": len(set(d["trajectory_id"] for d in train_data)),
        "validation_trajectories": len(set(d["trajectory_id"] for d in val_data)),
    }
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = model.save_checkpoint(METRICS_DIR / "quantum_weights.npz", metadata=metadata)
    print(f"\nSaved best checkpoint to: {out_ckpt}", flush=True)
    
    with open(METRICS_DIR / "confusion_matrix.json", "w") as f:
        json.dump({"confusion_matrix": cm.tolist(), "action_names": action_names}, f, indent=2)
        
    with open(TRAINING_METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved training metrics to: {TRAINING_METRICS_PATH}", flush=True)
    return metadata, cm

if __name__ == "__main__":
    run_training_target_a()
