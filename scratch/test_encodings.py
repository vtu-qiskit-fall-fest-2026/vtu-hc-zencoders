import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scratch.test_separability import generate_multi_sector_dataset

def encode_strategy_a_grouped(feat_12d, n_qubits=4):
    """Strategy A: Semantic grouped encoding with world target bearing."""
    t_dist, t_dir_x, t_dir_y, sin_t, cos_t = feat_12d[0], feat_12d[1], feat_12d[2], feat_12d[3], feat_12d[4]
    near_obs, front_obs, left_obs, right_obs = feat_12d[5], feat_12d[6], feat_12d[7], feat_12d[8]
    density, speed, bound_prox = feat_12d[9], feat_12d[10], feat_12d[11]

    angles = np.zeros(n_qubits, dtype=float)
    # q0: World target angle (key for world directional actions)
    world_angle = np.arctan2(t_dir_y, t_dir_x)
    angles[0] = world_angle

    # q1: Frontal & nearest obstacle warning
    front_danger = 1.0 - front_obs
    near_danger = 1.0 - near_obs
    angles[1] = ((0.6 * front_danger + 0.4 * near_danger) * 2.0 - 1.0) * np.pi

    # q2: Lateral clearance differential
    lateral_diff = left_obs - right_obs
    angles[2] = np.clip(lateral_diff * 1.5, -1.0, 1.0) * np.pi

    # q3: Environmental margin
    env_pressure = (1.0 - bound_prox) * 0.7 + density * 0.5
    angles[3] = (env_pressure * 2.0 - 1.0) * np.pi

    if n_qubits > 4:
        # q4: Relative bearing
        rel_angle = np.arctan2(sin_t, cos_t)
        angles[4] = rel_angle
    if n_qubits > 5:
        # q5: Kinematics
        angles[5] = (speed * 2.0 - 1.0) * np.pi

    return np.clip(angles, -np.pi, np.pi)

def encode_strategy_b_anglescaled(feat_12d, n_qubits=4):
    """Strategy B: Angle-scaled encoding."""
    # Key indices: target_dir_x (1), target_dir_y (2), nearest_obs (5), density (9)
    # Or linear reduction with stable fixed projection
    angles = np.zeros(n_qubits, dtype=float)
    if n_qubits == 4:
        angles[0] = feat_12d[1] * np.pi # t_dir_x in [-1, 1]
        angles[1] = feat_12d[2] * np.pi # t_dir_y in [-1, 1]
        angles[2] = (feat_12d[5] * 2.0 - 1.0) * np.pi # near_obs
        angles[3] = (feat_12d[9] * 2.0 - 1.0) * np.pi # density
    elif n_qubits == 6:
        angles[0] = feat_12d[1] * np.pi
        angles[1] = feat_12d[2] * np.pi
        angles[2] = (feat_12d[5] * 2.0 - 1.0) * np.pi
        angles[3] = (feat_12d[6] * 2.0 - 1.0) * np.pi
        angles[4] = (feat_12d[7] - feat_12d[8]) * np.pi
        angles[5] = (feat_12d[9] * 2.0 - 1.0) * np.pi
    return np.clip(angles, -np.pi, np.pi)

def encode_strategy_c_bounded_nonlinear(feat_12d, n_qubits=4):
    """Strategy C: Bounded nonlinear tanh encoding."""
    # Compute centered values and tanh
    angles = np.zeros(n_qubits, dtype=float)
    world_angle = np.arctan2(feat_12d[2], feat_12d[1])
    angles[0] = world_angle
    angles[1] = np.tanh(2.0 * (0.5 - feat_12d[6])) * np.pi # front danger
    angles[2] = np.tanh(2.0 * (feat_12d[7] - feat_12d[8])) * np.pi # lateral diff
    angles[3] = np.tanh(2.0 * (feat_12d[9] - 0.3)) * np.pi # density
    if n_qubits > 4:
        angles[4] = np.arctan2(feat_12d[3], feat_12d[4]) # rel angle
    if n_qubits > 5:
        angles[5] = np.tanh(2.0 * (feat_12d[10] - 0.5)) * np.pi
    return np.clip(angles, -np.pi, np.pi)

def encode_strategy_d_grouped_normalized(feat_12d, n_qubits=4):
    """Strategy D: Grouped + L2 normalized encoding."""
    g = encode_strategy_a_grouped(feat_12d, n_qubits)
    norm = np.linalg.norm(g)
    if norm > 1e-6:
        # Scale unit vector by pi
        return np.clip(g / norm * np.pi, -np.pi, np.pi)
    return g

def evaluate_encodings():
    data, _, _ = generate_multi_sector_dataset(3000, seed=42)
    trajs = sorted(list(set(d["trajectory_id"] for d in data)))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(trajs)
    n_val = int(len(trajs) * 0.2)
    val_set = set(shuffled[:n_val])

    train_data = [d for d in data if d["trajectory_id"] not in val_set]
    val_data = [d for d in data if d["trajectory_id"] in val_set]
    y_train = np.array([d["teacher_index"] for d in train_data])
    y_val = np.array([d["teacher_index"] for d in val_data])

    strats = {
        "A: Grouped": encode_strategy_a_grouped,
        "B: Angle-Scaled": encode_strategy_b_anglescaled,
        "C: Bounded Nonlinear": encode_strategy_c_bounded_nonlinear,
        "D: Grouped Normalized": encode_strategy_d_grouped_normalized,
    }

    for n_q in [4, 6]:
        print(f"\n=== Evaluating Encoding Variants on Unseen Validation Set (Qubits={n_q}) ===", flush=True)
        for name, fn in strats.items():
            X_tr = np.array([fn(d["features"], n_q) for d in train_data])
            X_va = np.array([fn(d["features"], n_q) for d in val_data])
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_tr, y_train)
            pred = rf.predict(X_va)
            acc = accuracy_score(y_val, pred) * 100
            f1 = f1_score(y_val, pred, average='macro')
            print(f"  {name:25s} -> Val Acc: {acc:5.2f}%, Macro F1: {f1:.4f}", flush=True)

if __name__ == "__main__":
    evaluate_encodings()
