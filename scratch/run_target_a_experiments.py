import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from config import DEFAULT_CONFIG
from simulation.obstacles import ObstacleField, CircularObstacle, RectangularObstacle, MovingObstacle
from classical.feature_extractor import NavigationFeatureExtractor
from simulation.collision import check_boundary_violation
from scratch.test_pipeline import create_10_synthetic_environments
from scratch.teacher_test import enhanced_teacher_policy
from quantum.quantum_model import VariationalQuantumNavigator

def generate_controlled_multisector_data(n_samples=2400, seed=42, lookahead=2):
    """Generate multi-sector flight data across procedural synthetic topologies."""
    print(f"Generating {n_samples} state-action pairs (seed={seed}, lookahead={lookahead})...", flush=True)
    rng = np.random.default_rng(seed)
    extractor = NavigationFeatureExtractor()
    nav_config = DEFAULT_CONFIG.navigation
    action_names = nav_config.action_names
    action_vectors = nav_config.action_vectors

    dataset = []
    traj_count = 0
    angles = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]

    env_id = 0
    while len(dataset) < n_samples:
        field, moving, desc = create_10_synthetic_environments(env_id, rng)
        env_id += 1

        for angle in angles:
            if len(dataset) >= n_samples:
                break

            center = np.array([25.0, 25.0])
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            p_start = np.clip(center - 17.0 * dir_vec + rng.uniform(-3.0, 3.0, 2), 6.0, 44.0)
            p_target = np.clip(center + 17.0 * dir_vec + rng.uniform(-3.0, 3.0, 2), 6.0, 44.0)

            hit_s, _ = field.check_collision(p_start, radius=2.0)
            hit_t, _ = field.check_collision(p_target, radius=2.0)
            if hit_s or hit_t or np.linalg.norm(p_target - p_start) < 14.0:
                continue

            traj_count += 1
            drone_pos = p_start.copy()
            heading = float(np.arctan2(p_target[1] - drone_pos[1], p_target[0] - drone_pos[0]))
            speed = 1.5

            for step in range(40):
                t_diff = p_target - drone_pos
                t_dist = float(np.linalg.norm(t_diff))
                t_unit = t_diff / (t_dist + 1e-8)

                _, near_obs_dist, near_obs_dir = field.get_nearest_obstacle(drone_pos, radius=0.8)
                density = field.calculate_density(drone_pos, radius=8.0)
                b_dist = float(min(drone_pos[0], 50.0 - drone_pos[0], drone_pos[1], 50.0 - drone_pos[1]))

                mock_state = {
                    "drone_position": drone_pos.copy(),
                    "drone_velocity": np.array([np.cos(heading), np.sin(heading)]) * speed,
                    "drone_heading": heading,
                    "drone_speed": speed,
                    "target_position": p_target.copy(),
                    "target_distance": t_dist,
                    "target_direction": t_unit,
                    "nearest_obstacle_distance": near_obs_dist,
                    "nearest_obstacle_direction": near_obs_dir,
                    "obstacle_density": density,
                    "boundary_distance": b_dist,
                    "obstacles": field,
                }

                feat_vec, feat_dict = extractor.extract_features(mock_state)
                teacher_act, teacher_idx, conf, safe_acts = enhanced_teacher_policy(
                    drone_pos=drone_pos,
                    target_pos=p_target,
                    obstacles=field,
                    action_names=action_names,
                    action_vectors=action_vectors,
                    current_heading=heading,
                    current_speed=speed,
                    lookahead_steps=lookahead,
                    goal_threshold=2.0,
                )

                dataset.append({
                    "trajectory_id": traj_count,
                    "features": feat_vec,
                    "teacher_index": teacher_idx,
                    "teacher_action": teacher_act,
                })

                if t_dist <= 2.0 or teacher_act == "STAY":
                    break

                vec = np.array(action_vectors[teacher_act])
                nv = np.linalg.norm(vec)
                if nv > 1e-5:
                    u = vec / nv
                    drone_pos += u * speed
                    heading = float(np.arctan2(u[1], u[0]))

                if check_boundary_violation(drone_pos, radius=0.8, map_width=50.0, map_height=50.0):
                    break
                hit, _ = field.check_collision(drone_pos, radius=0.9)
                if hit:
                    break

    # Strictly isolated trajectory-level train/validation split
    trajs = sorted(list(set(d["trajectory_id"] for d in dataset)))
    shuffled = rng.permutation(trajs)
    n_val_trajs = max(3, int(len(trajs) * 0.20))
    val_set = set(shuffled[:n_val_trajs])

    train_data = [d for d in dataset if d["trajectory_id"] not in val_set]
    val_data = [d for d in dataset if d["trajectory_id"] in val_set]

    print(f"Generated {len(dataset)} samples across {traj_count} trajectories.", flush=True)
    print(f"Train samples: {len(train_data)} ({len(trajs) - n_val_trajs} trajs) | Val samples: {len(val_data)} ({n_val_trajs} trajs)", flush=True)
    return train_data, val_data

def encode_features_to_angles(feat_12d, n_qubits):
    f = np.asarray(feat_12d, dtype=float)
    angles = np.zeros(n_qubits, dtype=float)
    # q0: World target bearing angle in [-pi, pi]
    angles[0] = float(np.arctan2(f[2], f[1]))
    # q1: Normalized target distance in [-pi, pi]
    angles[1] = float((np.clip(f[0], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
    # q2: Frontal obstacle danger in [-pi, pi]
    angles[2] = float(np.tanh(2.0 * (0.5 - f[6])) * np.pi)
    # q3: Nearest obstacle distance in [-pi, pi]
    angles[3] = float((np.clip(f[5], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
    if n_qubits > 4:
        # q4: Lateral clearance differential (left vs right)
        angles[4] = float(np.tanh(2.0 * (f[7] - f[8])) * np.pi)
    if n_qubits > 5:
        # q5: Environmental obstacle density & boundary in [-pi, pi]
        angles[5] = float((np.clip(f[9], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
    if n_qubits > 6:
        # q6: Relative target angle sin/cos
        angles[6] = float(np.arctan2(f[3], f[4]))
    if n_qubits > 7:
        # q7: Boundary proximity in [-pi, pi]
        angles[7] = float((np.clip(f[11], 0.0, 1.0) * 2.0 - 1.0) * np.pi)
    return angles

def extract_vqc_features(model, train_data, val_data, n_qubits):
    print(f"Executing quantum circuits for {n_qubits}Q{model.circuit.n_layers}L...", flush=True)
    t0 = time.time()
    X_q_tr = []
    for d in train_data:
        ang = encode_features_to_angles(d["features"], n_qubits)
        ev, bp, _ = model.circuit.execute(ang, model.weights)
        X_q_tr.append(np.concatenate([ev, bp]))
    X_q_tr = np.array(X_q_tr, dtype=np.float32)

    X_q_va = []
    for d in val_data:
        ang = encode_features_to_angles(d["features"], n_qubits)
        ev, bp, _ = model.circuit.execute(ang, model.weights)
        X_q_va.append(np.concatenate([ev, bp]))
    X_q_va = np.array(X_q_va, dtype=np.float32)

    X_c_tr = np.array([d["features"] for d in train_data], dtype=np.float32)
    X_c_va = np.array([d["features"] for d in val_data], dtype=np.float32)

    y_tr = np.array([d["teacher_index"] for d in train_data], dtype=np.int64)
    y_va = np.array([d["teacher_index"] for d in val_data], dtype=np.int64)

    dt = time.time() - t0
    print(f"Extraction complete in {dt:.2f}s. Quantum feature dim = {X_q_tr.shape[1]}", flush=True)
    return X_q_tr, X_q_va, X_c_tr, X_c_va, y_tr, y_va

def train_pytorch_head(head_type, in_dim, num_classes, X_tr, y_tr, X_va, y_va, lr=0.005, weight_decay=1e-4, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Class weights for balanced cross-entropy
    counts = np.bincount(y_tr, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = (len(y_tr) / (num_classes * max(1, counts[c])))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    if head_type == "linear":
        net = nn.Sequential(nn.Linear(in_dim, num_classes))
    elif head_type == "mlp":
        net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    elif head_type == "residual_hybrid":
        # Quantum features (in_dim - 12) + Classical features (12)
        net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    else:
        raise ValueError(head_type)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_preds = None
    best_model_state = None
    patience = 25
    patience_cnt = 0

    batch_size = 32
    n_batches = int(np.ceil(len(X_tr) / batch_size))

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(X_tr))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            out = net(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            tr_out = net(X_tr_t)
            tr_preds = torch.argmax(tr_out, dim=1).numpy()
            tr_acc = accuracy_score(y_tr, tr_preds) * 100

            va_out = net(X_va_t)
            va_loss = criterion(va_out, y_va_t).item()
            va_preds = torch.argmax(va_out, dim=1).numpy()
            va_acc = accuracy_score(y_va, va_preds) * 100

        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_preds = va_preds.copy()
            best_model_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    net.load_state_dict(best_model_state)
    net.eval()
    with torch.no_grad():
        final_out = net(X_va_t)
        final_probs = torch.softmax(final_out, dim=1).numpy()
        final_preds = np.argmax(final_probs, axis=1)
        final_loss = criterion(final_out, y_va_t).item()

    acc = accuracy_score(y_va, final_preds) * 100
    m_f1 = f1_score(y_va, final_preds, average='macro', zero_division=0)
    w_f1 = f1_score(y_va, final_preds, average='weighted', zero_division=0)

    return {
        "net": net,
        "train_accuracy": tr_acc,
        "val_accuracy": acc,
        "val_loss": final_loss,
        "macro_f1": m_f1,
        "weighted_f1": w_f1,
        "preds": final_preds,
    }

if __name__ == "__main__":
    train_data, val_data = generate_controlled_multisector_data(n_samples=2400, seed=42)
    y_val = np.array([d["teacher_index"] for d in val_data])
    action_names = DEFAULT_CONFIG.navigation.action_names

    architectures = [
        ("6Q3L", 6, 3),
        ("6Q4L", 6, 4),
        ("6Q5L", 6, 5),
        ("8Q3L", 8, 3),
    ]

    results = {}
    for arch_name, n_q, n_l in architectures:
        print(f"\n=======================================================")
        print(f"Testing Architecture: {arch_name} ({n_q} Qubits, {n_l} Layers)")
        print(f"=======================================================")
        model = VariationalQuantumNavigator(n_qubits=n_q, n_layers=n_l, seed=42, load_checkpoint=False)
        X_q_tr, X_q_va, X_c_tr, X_c_va, y_tr, y_va = extract_vqc_features(model, train_data, val_data, n_q)

        # 1. Linear Head (Pure VQC Readout)
        res_lin = train_pytorch_head("linear", X_q_tr.shape[1], 9, X_q_tr, y_tr, X_q_va, y_va, lr=0.01)
        print(f"  {arch_name} Linear Head      -> Val Acc: {res_lin['val_accuracy']:5.2f}% | Macro F1: {res_lin['macro_f1']:.4f} | Val Loss: {res_lin['val_loss']:.4f}")

        # 2. MLP Head (Nonlinear Quantum Readout)
        res_mlp = train_pytorch_head("mlp", X_q_tr.shape[1], 9, X_q_tr, y_tr, X_q_va, y_va, lr=0.005)
        print(f"  {arch_name} MLP Head         -> Val Acc: {res_mlp['val_accuracy']:5.2f}% | Macro F1: {res_mlp['macro_f1']:.4f} | Val Loss: {res_mlp['val_loss']:.4f}")

        # 3. Residual Hybrid Head (Quantum + Classical Features)
        X_hyb_tr = np.concatenate([X_q_tr, X_c_tr], axis=1)
        X_hyb_va = np.concatenate([X_q_va, X_c_va], axis=1)
        res_hyb = train_pytorch_head("residual_hybrid", X_hyb_tr.shape[1], 9, X_hyb_tr, y_tr, X_hyb_va, y_va, lr=0.005)
        print(f"  {arch_name} Residual Hybrid -> Val Acc: {res_hyb['val_accuracy']:5.2f}% | Macro F1: {res_hyb['macro_f1']:.4f} | Val Loss: {res_hyb['val_loss']:.4f}")

        results[arch_name] = {
            "linear": res_lin,
            "mlp": res_mlp,
            "hybrid": res_hyb,
            "model": model,
        }
