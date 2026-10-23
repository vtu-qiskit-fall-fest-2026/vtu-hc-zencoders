import sys
import numpy as np
from collections import Counter
from scratch.test_pipeline import create_10_synthetic_environments
from scratch.teacher_test import enhanced_teacher_policy
from config import DEFAULT_CONFIG
from simulation.collision import check_boundary_violation

def test_teacher_k_values():
    nav_config = DEFAULT_CONFIG.navigation
    action_names = nav_config.action_names
    action_vectors = nav_config.action_vectors
    angles = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]

    for k in [2, 3, 4, 5]:
        rng = np.random.default_rng(42)
        total_trajs = 0
        successes = 0
        collisions = 0
        action_counts = Counter()
        reversals = 0
        total_steps = 0

        for env_id in [0, 2, 4, 8]: # 4 diverse topologies
            field, moving, desc = create_10_synthetic_environments(env_id, rng)
            for a_idx, angle in enumerate(angles):
                center = np.array([25.0, 25.0])
                dir_vec = np.array([np.cos(angle), np.sin(angle)])
                p_start = np.clip(center - 16.0 * dir_vec + rng.uniform(-2.0, 2.0, 2), 6.0, 44.0)
                p_target = np.clip(center + 16.0 * dir_vec + rng.uniform(-2.0, 2.0, 2), 6.0, 44.0)

                hit_s, _ = field.check_collision(p_start, radius=2.0)
                hit_t, _ = field.check_collision(p_target, radius=2.0)
                if hit_s or hit_t:
                    continue

                total_trajs += 1
                drone_pos = p_start.copy()
                heading = float(np.arctan2(p_target[1] - drone_pos[1], p_target[0] - drone_pos[0]))
                speed = 1.5

                for step in range(40):
                    total_steps += 1
                    t_dist = float(np.linalg.norm(p_target - drone_pos))
                    act, idx, conf, safe = enhanced_teacher_policy(
                        drone_pos=drone_pos,
                        target_pos=p_target,
                        obstacles=field,
                        action_names=action_names,
                        action_vectors=action_vectors,
                        current_heading=heading,
                        current_speed=speed,
                        lookahead_steps=k,
                        goal_threshold=2.0,
                    )
                    action_counts[act] += 1

                    if t_dist <= 2.0 or act == "STAY":
                        successes += 1
                        break

                    vec = np.array(action_vectors[act])
                    nv = np.linalg.norm(vec)
                    if nv > 1e-5:
                        u = vec / nv
                        new_heading = float(np.arctan2(u[1], u[0]))
                        diff = abs((new_heading - heading + np.pi) % (2*np.pi) - np.pi)
                        if diff > 2.5:
                            reversals += 1
                        drone_pos += u * speed
                        heading = new_heading

                    if check_boundary_violation(drone_pos, radius=0.8, map_width=50.0, map_height=50.0):
                        break
                    hit, _ = field.check_collision(drone_pos, radius=0.9)
                    if hit:
                        collisions += 1
                        break

        succ_rate = (successes / total_trajs) * 100 if total_trajs else 0
        col_rate = (collisions / total_trajs) * 100 if total_trajs else 0
        rev_rate = (reversals / total_steps) * 100 if total_steps else 0
        print(f"--- Lookahead K={k} ---", flush=True)
        print(f"Trajs: {total_trajs}, Success: {succ_rate:.1f}%, Collision: {col_rate:.1f}%, Reversals: {rev_rate:.2f}%, Total Steps: {total_steps}", flush=True)
        print("Class distribution:", flush=True)
        for a in action_names:
            pct = (action_counts[a] / total_steps) * 100 if total_steps else 0
            print(f"  {a:11s}: {action_counts[a]:4d} ({pct:5.1f}%)", flush=True)
        print(flush=True)

if __name__ == "__main__":
    test_teacher_k_values()
