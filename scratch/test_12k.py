import time
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from scratch.test_separability import generate_multi_sector_dataset

def augment_features(feat_12d):
    """Derived features scientifically computed from 12D input without leakage."""
    # feat_12d: [norm_t_dist, t_dir_x, t_dir_y, sin_t, cos_t, near_obs, front_obs, left_obs, right_obs, density, speed, bound_prox]
    t_angle = np.arctan2(feat_12d[2], feat_12d[1])
    lat_diff = feat_12d[7] - feat_12d[8] # left - right
    min_obs = min(feat_12d[5], feat_12d[6]) # min(near, front)
    threat = 1.0 - min_obs
    bound_threat = 1.0 - feat_12d[11]
    rel_angle = np.arctan2(feat_12d[3], feat_12d[4])

    derived = np.array([
        t_angle,
        lat_diff,
        min_obs,
        threat,
        bound_threat,
        rel_angle,
    ], dtype=float)
    return np.concatenate([feat_12d, derived])

def test_12k_scale():
    print("Generating 12,000 samples across 10 topologies...", flush=True)
    t0 = time.time()
    data, counter, n_trajs = generate_multi_sector_dataset(12000, seed=42)
    print(f"Generated {len(data)} samples in {time.time()-t0:.2f}s", flush=True)

    trajs = sorted(list(set(d["trajectory_id"] for d in data)))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(trajs)
    n_val = int(len(trajs) * 0.2)
    val_set = set(shuffled[:n_val])

    train_data = [d for d in data if d["trajectory_id"] not in val_set]
    val_data = [d for d in data if d["trajectory_id"] in val_set]

    y_train = np.array([d["teacher_index"] for d in train_data])
    y_val = np.array([d["teacher_index"] for d in val_data])

    X_train_raw = np.array([d["features"] for d in train_data])
    X_val_raw = np.array([d["features"] for d in val_data])

    X_train_aug = np.array([augment_features(d["features"]) for d in train_data])
    X_val_aug = np.array([augment_features(d["features"]) for d in val_data])

    print(f"Train size: {len(X_train_raw)}, Val size: {len(X_val_raw)}", flush=True)

    # Raw 12D
    hgb_raw = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    hgb_raw.fit(X_train_raw, y_train)
    pred_raw = hgb_raw.predict(X_val_raw)
    print(f"HistGB (12D raw): Val Acc = {accuracy_score(y_val, pred_raw)*100:.2f}%, Macro F1 = {f1_score(y_val, pred_raw, average='macro'):.4f}", flush=True)

    # Augmented
    hgb_aug = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    hgb_aug.fit(X_train_aug, y_train)
    pred_aug = hgb_aug.predict(X_val_aug)
    print(f"HistGB (Augmented 18D): Val Acc = {accuracy_score(y_val, pred_aug)*100:.2f}%, Macro F1 = {f1_score(y_val, pred_aug, average='macro'):.4f}", flush=True)

    rf_aug = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    rf_aug.fit(X_train_aug, y_train)
    pred_rf = rf_aug.predict(X_val_aug)
    print(f"Random Forest (Augmented 18D): Val Acc = {accuracy_score(y_val, pred_rf)*100:.2f}%, Macro F1 = {f1_score(y_val, pred_rf, average='macro'):.4f}", flush=True)

if __name__ == "__main__":
    test_12k_scale()
