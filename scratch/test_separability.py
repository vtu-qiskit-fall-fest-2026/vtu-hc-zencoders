import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from classical.feature_extractor import NavigationFeatureExtractor
from simulation.obstacles import ObstacleField, CircularObstacle, RectangularObstacle, MovingObstacle
from simulation.collision import check_boundary_violation
from config import DEFAULT_CONFIG
from scratch.test_pipeline import create_10_synthetic_environments
from scratch.teacher_test import enhanced_teacher_policy

def generate_multi_sector_dataset(n_target_samples=3000, seed=42, lookahead=2):
    rng = np.random.default_rng(seed)
    extractor = NavigationFeatureExtractor()
    nav_config = DEFAULT_CONFIG.navigation
    action_names = nav_config.action_names
    action_vectors = nav_config.action_vectors

    dataset = []
    traj_count = 0
    action_counter = Counter()
    angles = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]

    env_id = 0
    while len(dataset) < n_target_samples:
        field, moving, desc = create_10_synthetic_environments(env_id, rng)
        env_id += 1

        for angle in angles:
            if len(dataset) >= n_target_samples:
                break

            center = np.array([25.0, 25.0])
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            # start and target along this compass angle
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

            for step in range(45):
                t_diff = p_target - drone_pos
                t_dist = float(np.linalg.norm(t_diff))
                t_unit = t_diff / (t_dist + 1e-8)

                _, near_obs_dist, near_obs_dir = field.get_nearest_obstacle(drone_pos, radius=0.8)
                density = field.calculate_density(drone_pos, radius=8.0)
                b_dist = float(min(drone_pos[0], 50.0 - drone_pos[0], drone_pos[1], 50.0 - drone_pos[1]))

                for m in moving:
                    m.step(dt=1.0)

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
                    "sample_id": len(dataset),
                    "trajectory_id": traj_count,
                    "step": step,
                    "features": feat_vec,
                    "feature_dict": feat_dict,
                    "teacher_action": teacher_act,
                    "teacher_index": teacher_idx,
                    "confidence": conf,
                    "safe_actions": safe_acts,
                })
                action_counter[teacher_act] += 1

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

    return dataset, action_counter, traj_count

if __name__ == "__main__":
    t0 = time.time()
    data, counter, n_trajs = generate_multi_sector_dataset(3000, seed=42)
    dt = time.time() - t0
    print(f"Generated {len(data)} samples across {n_trajs} trajectories in {dt:.2f}s")
    print("Class distribution:")
    for k, v in counter.most_common():
        print(f"  {k:12s}: {v:4d} ({v/len(data)*100:.2f}%)")

    # Grouped split by trajectory
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

    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}")
    # Majority baseline
    val_counts = Counter(y_val)
    majority_acc = val_counts.most_common(1)[0][1] / len(y_val)
    print(f"Random baseline: {1/9*100:.2f}%")
    print(f"Majority class baseline: {majority_acc*100:.2f}%")

    # Classical baselines
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    print(f"Random Forest 12D Val Acc: {accuracy_score(y_val, rf_pred)*100:.2f}%, Macro F1: {f1_score(y_val, rf_pred, average='macro'):.4f}")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    print(f"Logistic Regression 12D Val Acc: {accuracy_score(y_val, lr_pred)*100:.2f}%, Macro F1: {f1_score(y_val, lr_pred, average='macro'):.4f}")
