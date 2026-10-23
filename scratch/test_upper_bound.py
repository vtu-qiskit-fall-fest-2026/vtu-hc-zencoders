import time
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scratch.test_separability import generate_multi_sector_dataset

def test_dataset_scale_upper_bound():
    print("Generating 8,000 samples across 10 procedural topologies...", flush=True)
    t0 = time.time()
    data, counter, n_trajs = generate_multi_sector_dataset(8000, seed=42)
    dt = time.time() - t0
    print(f"Generated {len(data)} samples across {n_trajs} trajectories in {dt:.2f}s", flush=True)

    trajs = sorted(list(set(d["trajectory_id"] for d in data)))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(trajs)
    n_val = int(len(trajs) * 0.2)
    val_set = set(shuffled[:n_val])

    train_data = [d for d in data if d["trajectory_id"] not in val_set]
    val_data = [d for d in data if d["trajectory_id"] in val_set]

    X_train = np.array([d["features"] for d in train_data])
    y_train = np.array([d["teacher_index"] for d in train_data])
    X_val = np.array([d["features"] for d in val_data])
    y_val = np.array([d["teacher_index"] for d in val_data])

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}", flush=True)
    val_counts = Counter(y_val)
    majority_acc = val_counts.most_common(1)[0][1] / len(y_val)
    print(f"Random chance: 11.11%, Majority baseline: {majority_acc*100:.2f}%", flush=True)

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred) * 100
    rf_f1 = f1_score(y_val, rf_pred, average='macro')
    print(f"Random Forest (150 trees):    Val Acc = {rf_acc:.2f}%, Macro F1 = {rf_f1:.4f}", flush=True)

    # 2. Hist Gradient Boosting
    hgb = HistGradientBoostingClassifier(max_iter=150, random_state=42)
    hgb.fit(X_train, y_train)
    hgb_pred = hgb.predict(X_val)
    hgb_acc = accuracy_score(y_val, hgb_pred) * 100
    hgb_f1 = f1_score(y_val, hgb_pred, average='macro')
    print(f"HistGradientBoosting:         Val Acc = {hgb_acc:.2f}%, Macro F1 = {hgb_f1:.4f}", flush=True)

    # 3. MLP Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_val)
    mlp_acc = accuracy_score(y_val, mlp_pred) * 100
    mlp_f1 = f1_score(y_val, mlp_pred, average='macro')
    print(f"MLP (64, 32):                 Val Acc = {mlp_acc:.2f}%, Macro F1 = {mlp_f1:.4f}", flush=True)

if __name__ == "__main__":
    test_dataset_scale_upper_bound()
