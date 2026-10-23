import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from config import DEFAULT_CONFIG, SCENARIOS_DIR, MAPS_DIR
from utils.io_utils import load_json
from simulation.environment import UAVEnvironment
from classical.baseline import ClassicalBaselineNavigator
from quantum.quantum_decision import QuantumDecisionMaker
from navigation.safety import SafetyLayer
from navigation.action_selector import ActionSelector
from evaluation.metrics import compute_episode_metrics

class EnhancedNavigationController:
    """Navigation controller with stagnation, oscillation detection, and tangential recovery."""
    def __init__(self, env, navigator, config=None, is_hybrid=False, classical_navigator=None):
        self.env = env
        self.config = config or DEFAULT_CONFIG
        self.navigator = navigator
        self.is_hybrid = is_hybrid
        self.classical_navigator = classical_navigator
        
        self.safety_layer = SafetyLayer(nav_config=self.config.navigation, sim_config=self.config.sim)
        self.safety_layer.obstacles = self.env.obstacles
        self.safety_layer.map_width = self.env.map_width
        self.safety_layer.map_height = self.env.map_height
        
        self.action_selector = ActionSelector(safety_layer=self.safety_layer, config=self.config.navigation)
        
        # History & tracking
        self.positions = []
        self.target_distances = []
        self.actions = []
        self.action_vectors = self.config.navigation.action_vectors
        self.action_names = self.config.navigation.action_names
        
        # Recovery state machine
        self.recovery_mode = False
        self.recovery_steps_left = 0
        self.recovery_vector = None
        self.recovery_tangent_sign = 1.0  # +1 for left tangent, -1 for right tangent
        self.last_recovery_step = -100

    def reset(self, start_pos, target_pos):
        self.positions.clear()
        self.target_distances.clear()
        self.actions.clear()
        self.recovery_mode = False
        self.recovery_steps_left = 0
        self.recovery_vector = None
        self.recovery_tangent_sign = 1.0
        self.last_recovery_step = -100
        self.safety_layer.reset()
        if hasattr(self.navigator, "reset"):
            self.navigator.reset()
        if self.classical_navigator and hasattr(self.classical_navigator, "reset"):
            self.classical_navigator.reset()
        return self.env.reset(start_position=start_pos, target_position=target_pos)

    def detect_oscillation_and_stagnation(self, state):
        pos = np.asarray(state["drone_position"], dtype=float)
        dist = float(state["target_distance"])
        step = self.env.current_step
        
        self.positions.append(pos)
        self.target_distances.append(dist)
        
        if len(self.positions) > 10:
            self.positions.pop(0)
            self.target_distances.pop(0)
            
        if dist <= 2.5:
            return False, "near_goal"
            
        # 1. Oscillation check: alternating actions in last 6 steps
        if len(self.actions) >= 4:
            # Check 180-degree reversals
            rev_count = 0
            for i in range(len(self.actions) - 1, max(0, len(self.actions) - 6), -1):
                v1 = np.array(self.action_vectors[self.actions[i]], dtype=float)
                v0 = np.array(self.action_vectors[self.actions[i-1]], dtype=float)
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v0) > 0:
                    if np.dot(v1, v0) / (np.linalg.norm(v1) * np.linalg.norm(v0)) < -0.7:
                        rev_count += 1
            if rev_count >= 2:
                return True, "oscillation_reversal"
                
            # Check 2-cycle repetition A-B-A-B
            if len(self.actions) >= 4 and self.actions[-1] == self.actions[-3] and self.actions[-2] == self.actions[-4] and self.actions[-1] != self.actions[-2]:
                return True, "oscillation_cycle"
                
        # 2. Stagnation check: net displacement over 8 steps < 1.2
        if len(self.positions) >= 8:
            disp = np.linalg.norm(self.positions[-1] - self.positions[0])
            prog = self.target_distances[0] - self.target_distances[-1]
            if disp < 1.5 or prog < 0.3:
                return True, "stagnation_local_min"
                
        return False, "none"

    def compute_tangential_escape(self, state):
        drone_pos = np.asarray(state["drone_position"], dtype=float)
        target_pos = np.asarray(state["target_position"], dtype=float)
        t_vec = target_pos - drone_pos
        t_dist = np.linalg.norm(t_vec)
        u_target = t_vec / t_dist if t_dist > 1e-6 else np.array([1.0, 0.0])
        
        # Find nearest obstacle
        obs, min_dist, away_dir = self.env.obstacles.get_nearest_obstacle(drone_pos, self.env.drone.radius)
        if obs is not None:
            c = obs.get_center()
            diff = drone_pos - c
            d_norm = np.linalg.norm(diff)
            u_away = diff / d_norm if d_norm > 1e-6 else away_dir
        else:
            u_away = -u_target
            
        # Two orthogonal tangents
        t_left = np.array([-u_away[1], u_away[0]], dtype=float)
        t_right = np.array([u_away[1], -u_away[0]], dtype=float)
        
        # Test which tangent direction aligns better or is safer
        align_l = np.dot(t_left, u_target)
        align_r = np.dot(t_right, u_target)
        
        # Alternate sign if recently used the other
        if align_l >= align_r:
            preferred_tan = t_left if self.recovery_tangent_sign > 0 else t_right
        else:
            preferred_tan = t_right if self.recovery_tangent_sign > 0 else t_left
            
        # Blend tangent (70%) with target pull (30%)
        rec_dir = 0.70 * preferred_tan + 0.30 * u_target
        r_norm = np.linalg.norm(rec_dir)
        return rec_dir / r_norm if r_norm > 1e-6 else preferred_tan

    def step(self):
        state = self.env.get_state()
        safe_actions = self.safety_layer.get_safe_actions(state)
        stay_idx = self.action_names.index("STAY") if "STAY" in self.action_names else 8
        
        if len(safe_actions) == 0:
            safe_actions = [stay_idx]
            
        # Check oscillation / stagnation
        triggered, reason = self.detect_oscillation_and_stagnation(state)
        step = self.env.current_step
        
        if triggered and not self.recovery_mode and (step - self.last_recovery_step > 4):
            self.recovery_mode = True
            self.recovery_steps_left = 8
            self.last_recovery_step = step
            self.recovery_vector = self.compute_tangential_escape(state)
            # Flip sign for next time
            self.recovery_tangent_sign *= -1.0
            
        if self.recovery_mode:
            self.recovery_steps_left -= 1
            if self.recovery_steps_left <= 0:
                self.recovery_mode = False
                
        # Query navigator(s)
        c_action_idx = 0
        if self.is_hybrid or self.classical_navigator:
            c_dec = self.classical_navigator.compute_decision(state)
            c_act_name = c_dec.get("action", "STAY")
            c_action_idx = self.action_names.index(c_act_name) if c_act_name in self.action_names else 0
            
        q_dec = self.navigator.compute_decision(state)
        q_probs_raw = q_dec.get("action_probabilities", {})
        if isinstance(q_probs_raw, dict):
            q_arr = np.array([q_probs_raw.get(k, 0.0) for k in self.action_names], dtype=float)
        else:
            q_arr = np.asarray(q_probs_raw, dtype=float)
            
        # Compute action scores across all candidate actions
        scores = np.full(len(self.action_names), -100.0)
        
        if self.recovery_mode and self.recovery_vector is not None:
            # Deterministic tangential escape guidance
            for a_idx in safe_actions:
                vec = np.array(self.action_vectors[self.action_names[a_idx]], dtype=float)
                v_norm = np.linalg.norm(vec)
                if v_norm > 1e-6:
                    u_v = vec / v_norm
                    scores[a_idx] = float(np.dot(u_v, self.recovery_vector))
                else:
                    scores[a_idx] = -1.0
        else:
            # Normal navigation
            # Target unit vector
            drone_pos = np.asarray(state["drone_position"], dtype=float)
            target_pos = np.asarray(state["target_position"], dtype=float)
            t_diff = target_pos - drone_pos
            t_dist = np.linalg.norm(t_diff)
            u_t = t_diff / t_dist if t_dist > 1e-6 else np.array([0.0, 0.0])
            
            for a_idx in safe_actions:
                a_name = self.action_names[a_idx]
                vec = np.array(self.action_vectors[a_name], dtype=float)
                v_norm = np.linalg.norm(vec)
                
                if v_norm < 1e-6:
                    scores[a_idx] = 2.0 if t_dist <= 2.0 else -2.0
                    continue
                    
                u_v = vec / v_norm
                prog = float(np.dot(u_v, u_t))
                
                if self.is_hybrid:
                    # Confidence-aware hybrid fusion
                    # If quantum top action opposes progress, downweight quantum
                    q_val = q_arr[a_idx]
                    c_val = 1.0 if a_idx == c_action_idx else 0.0
                    # Progress alignment
                    scores[a_idx] = 0.5 * q_val + 0.3 * c_val + 0.2 * max(0.0, prog)
                else:
                    # Pure quantum
                    scores[a_idx] = q_arr[a_idx] + 0.15 * prog
                    
                # Action hysteresis: bonus to last action, penalty to 180 deg reversal
                if self.actions:
                    last_vec = np.array(self.action_vectors[self.actions[-1]], dtype=float)
                    if np.linalg.norm(last_vec) > 1e-6:
                        u_last = last_vec / np.linalg.norm(last_vec)
                        cos_sim = float(np.dot(u_v, u_last))
                        if cos_sim > 0.9:
                            scores[a_idx] += 0.15  # persistence
                        elif cos_sim < -0.7:
                            scores[a_idx] -= 0.30  # reversal penalty
                            
        best_a_idx = int(np.argmax(scores))
        chosen_name = self.action_names[best_a_idx]
        chosen_vec = np.array(self.action_vectors[chosen_name], dtype=float)
        
        self.actions.append(chosen_name)
        
        # Step environment
        next_state, reward, done, info = self.env.step(direction=chosen_vec, speed=self.env.drone.base_speed)
        return next_state, reward, done, info

    def run_episode(self, max_steps=100):
        done = False
        while not done and self.env.current_step < max_steps:
            _, _, done, _ = self.step()
            
        switches = sum(1 for i in range(1, len(self.actions)) if self.actions[i] != self.actions[i-1])
        oscillations = sum(1 for i in range(2, len(self.actions)) if self.actions[i] == self.actions[i-2] and self.actions[i] != self.actions[i-1])
        
        return {
            "success": bool(self.env.success),
            "collided": bool(self.env.collided),
            "total_steps": int(self.env.current_step),
            "final_distance": float(self.env.target.distance_to(self.env.drone.position)),
            "path_length": float(self.env.get_path_length()),
            "collision_count": int(self.env.collision_count),
            "action_switch_count": switches,
            "oscillation_count": oscillations,
        }

if __name__ == "__main__":
    ckpt_path = Path("outputs/metrics/quantum_weights.npz")
    scenarios = ["scenario_01", "scenario_02", "scenario_03", "scenario_04", "scenario_05"]
    seeds = [42, 143, 244]
    
    for mode in ["quantum", "hybrid"]:
        total_success = 0
        total_episodes = len(scenarios) * len(seeds)
        print(f"\n=======================================================")
        print(f"Testing Enhanced Navigation: Mode = {mode.upper()}")
        print(f"=======================================================")
        for sc_id in scenarios:
            sc_data = load_json(SCENARIOS_DIR / f"{sc_id}.json")
            map_data = load_json(MAPS_DIR / f"{sc_data['map_id']}.json")
            sc_succ = 0
            for seed in seeds:
                env = UAVEnvironment(config=DEFAULT_CONFIG, map_data=map_data, scenario_data=sc_data)
                q_nav = QuantumDecisionMaker(config=DEFAULT_CONFIG, checkpoint_path=ckpt_path, load_checkpoint=True)
                c_nav = ClassicalBaselineNavigator(config=DEFAULT_CONFIG, obstacles=env.obstacles) if mode == "hybrid" else None
                
                ctrl = EnhancedNavigationController(
                    env=env,
                    navigator=q_nav,
                    config=DEFAULT_CONFIG,
                    is_hybrid=(mode == "hybrid"),
                    classical_navigator=c_nav,
                )
                start_p = tuple(sc_data.get("start", (5.0, 5.0)))
                target_p = tuple(sc_data.get("target", (45.0, 45.0)))
                ctrl.reset(start_p, target_p)
                res = ctrl.run_episode(max_steps=int(sc_data.get("max_steps", 100)))
                if res["success"]:
                    total_success += 1
                    sc_succ += 1
                print(f"  {sc_id} (seed {seed}): Success={res['success']}, Steps={res['total_steps']}, FinalDist={res['final_distance']:.2f}, Collisions={res['collision_count']}")
            print(f"--> {sc_id} overall: {sc_succ}/{len(seeds)} successful")
        print(f"\n*** Total {mode.upper()} Success: {total_success}/{total_episodes} ({total_success/total_episodes*100:.1f}%) ***")
