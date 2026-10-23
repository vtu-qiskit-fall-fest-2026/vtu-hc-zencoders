"""Navigation Controller Module for Q-Nav.

Coordinates the main closed-loop UAV navigation cycle: environment state sensing,
model decision query, safety mediation, kinetic step execution, and telemetry logging.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from config import QNavConfig, DEFAULT_CONFIG
from simulation.environment import UAVEnvironment
from navigation.action_selector import ActionSelector
from navigation.safety import SafetyLayer


class NavigationController:
    """Master controller executing closed-loop autonomous UAV navigation."""

    def __init__(
        self,
        environment: UAVEnvironment,
        navigator: Optional[Any] = None,
        config: Optional[QNavConfig] = None,
        is_hybrid: bool = False,
        classical_navigator: Optional[Any] = None,
        explain_decisions: bool = False,
    ):
        """Initialize controller with environment and active navigator.

        Args:
            environment: UAVEnvironment instance.
            navigator: ClassicalBaselineNavigator, QuantumDecisionMaker, or None.
            config: Master configuration.
            is_hybrid: Whether to run in hybrid quantum-classical decision mode.
            classical_navigator: Optional classical navigator for hybrid fusion.
            explain_decisions: Whether to generate XAI feature importance explanations.
        """
        self.env = environment
        self.config = config or DEFAULT_CONFIG
        self.is_hybrid = is_hybrid or getattr(self.config.navigation, "use_hybrid", False)
        self.explain_decisions = explain_decisions or getattr(
            self.config.navigation, "explain_decisions", False
        )
        self.explanation_log: List[Dict[str, Any]] = []
        self.explanation_times: List[float] = []

        if navigator is None:
            if self.is_hybrid:
                from quantum.quantum_decision import QuantumDecisionMaker
                self.navigator = QuantumDecisionMaker(config=self.config)
            else:
                from classical.baseline import ClassicalBaselineNavigator
                self.navigator = ClassicalBaselineNavigator(
                    config=self.config, obstacles=self.env.obstacles
                )
        else:
            self.navigator = navigator

        self.safety_layer = SafetyLayer(
            nav_config=self.config.navigation,
            sim_config=self.config.sim,
        )
        self.safety_layer.obstacles = self.env.obstacles
        self.safety_layer.map_width = self.env.map_width
        self.safety_layer.map_height = self.env.map_height

        self.action_selector = ActionSelector(
            safety_layer=self.safety_layer,
            config=self.config.navigation,
        )

        if classical_navigator is not None:
            self.classical_navigator = classical_navigator
        else:
            from classical.baseline import ClassicalBaselineNavigator
            if isinstance(self.navigator, ClassicalBaselineNavigator):
                self.classical_navigator = self.navigator
            else:
                self.classical_navigator = ClassicalBaselineNavigator(
                    config=self.config, obstacles=self.env.obstacles
                )


        self.history: List[Dict[str, Any]] = []
        self.dynamic_telemetry_history: List[Dict[str, Any]] = []
        self.fusion_log: List[Dict[str, Any]] = []
        self.decision_times: List[float] = []
        self.circuit_times: List[float] = []
        self.action_history: List[str] = []

        # Local-minima & oscillation tracking
        self.positions: List[np.ndarray] = []
        self.target_distances: List[float] = []
        self.action_vectors = self.config.navigation.action_vectors
        self.action_names = self.config.navigation.action_names

        # Deterministic tangential recovery state machine
        self.recovery_mode = False
        self.recovery_steps_left = 0
        self.recovery_vector: Optional[np.ndarray] = None
        self.recovery_tangent_sign = 1.0
        self.last_recovery_step = -100

    def reset(
        self,
        start_position: Optional[Tuple[float, float]] = None,
        target_position: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Reset environment, safety layer, navigator, and controller history."""
        self.history.clear()
        self.dynamic_telemetry_history.clear()
        self.fusion_log.clear()
        self.explanation_log.clear()
        self.decision_times.clear()
        self.circuit_times.clear()
        self.explanation_times.clear()
        self.action_history.clear()
        self.positions.clear()
        self.target_distances.clear()
        self.recovery_mode = False
        self.recovery_steps_left = 0
        self.recovery_vector = None
        self.recovery_tangent_sign = 1.0
        self.last_recovery_step = -100
        self.safety_layer.reset()
        self.safety_layer.obstacles = self.env.obstacles
        self.safety_layer.map_width = self.env.map_width
        self.safety_layer.map_height = self.env.map_height

        if hasattr(self.navigator, "reset"):
            self.navigator.reset()
        if hasattr(self.navigator, "set_obstacles"):
            self.navigator.set_obstacles(self.env.obstacles)
        if self.classical_navigator is not None:
            if hasattr(self.classical_navigator, "reset"):
                self.classical_navigator.reset()
            if hasattr(self.classical_navigator, "set_obstacles"):
                self.classical_navigator.set_obstacles(self.env.obstacles)

        return self.env.reset(
            start_position=start_position, target_position=target_position
        )

    def detect_oscillation_and_stagnation(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Detect local-minima entrapment and oscillatory limit cycles."""
        pos = np.asarray(state["drone_position"], dtype=float)
        dist = float(state["target_distance"])

        self.positions.append(pos)
        self.target_distances.append(dist)

        if len(self.positions) > 10:
            self.positions.pop(0)
            self.target_distances.pop(0)

        if dist <= 2.5:
            return False, "near_goal"

        # 1. Oscillation check: alternating actions in recent steps
        if len(self.action_history) >= 4:
            # Check 180-degree reversals
            rev_count = 0
            for i in range(len(self.action_history) - 1, max(0, len(self.action_history) - 6), -1):
                v1 = np.array(self.action_vectors[self.action_history[i]], dtype=float)
                v0 = np.array(self.action_vectors[self.action_history[i - 1]], dtype=float)
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v0) > 0:
                    if np.dot(v1, v0) / (np.linalg.norm(v1) * np.linalg.norm(v0)) < -0.7:
                        rev_count += 1
            if rev_count >= 2:
                return True, "oscillation_reversal"

            # Check 2-cycle repetition A-B-A-B
            if (
                len(self.action_history) >= 4
                and self.action_history[-1] == self.action_history[-3]
                and self.action_history[-2] == self.action_history[-4]
                and self.action_history[-1] != self.action_history[-2]
            ):
                return True, "oscillation_cycle"

        # 2. Stagnation check: net displacement over 8 steps < 1.5 or progress < 0.3
        if len(self.positions) >= 8:
            disp = np.linalg.norm(self.positions[-1] - self.positions[0])
            prog = self.target_distances[0] - self.target_distances[-1]
            if disp < 1.5 or prog < 0.3:
                return True, "stagnation_local_min"

        return False, "none"

    def compute_tangential_escape(self, state: Dict[str, Any]) -> np.ndarray:
        """Compute safe orthogonal tangential bypass around nearest obstacle."""
        drone_pos = np.asarray(state["drone_position"], dtype=float)
        target_pos = np.asarray(state["target_position"], dtype=float)
        t_vec = target_pos - drone_pos
        t_dist = np.linalg.norm(t_vec)
        u_target = t_vec / t_dist if t_dist > 1e-6 else np.array([1.0, 0.0])

        obs, min_dist, away_dir = self.env.obstacles.get_nearest_obstacle(drone_pos, self.env.drone.radius)
        if obs is not None:
            c = obs.get_center()
            diff = drone_pos - c
            d_norm = np.linalg.norm(diff)
            u_away = diff / d_norm if d_norm > 1e-6 else away_dir
        else:
            u_away = -u_target

        t_left = np.array([-u_away[1], u_away[0]], dtype=float)
        t_right = np.array([u_away[1], -u_away[0]], dtype=float)

        align_l = float(np.dot(t_left, u_target))
        align_r = float(np.dot(t_right, u_target))

        if align_l >= align_r:
            preferred_tan = t_left if self.recovery_tangent_sign > 0 else t_right
        else:
            preferred_tan = t_right if self.recovery_tangent_sign > 0 else t_left

        rec_dir = 0.70 * preferred_tan + 0.30 * u_target
        r_norm = np.linalg.norm(rec_dir)
        return rec_dir / r_norm if r_norm > 1e-6 else preferred_tan

    def _compute_pure_quantum_action(
        self,
        state: Dict[str, Any],
        quantum_probs: Any,
        safe_actions: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Compute pure quantum action from quantum probability distribution and progress scoring."""
        if safe_actions is None:
            safe_actions = self.safety_layer.get_safe_actions(state)

        stay_idx = (
            self.config.navigation.action_names.index("STAY")
            if "STAY" in self.config.navigation.action_names
            else 8
        )
        if len(safe_actions) == 0:
            return stay_idx, 0.0

        if isinstance(quantum_probs, dict):
            q_arr = np.array([quantum_probs.get(k, 0.0) for k in self.action_names], dtype=float)
        else:
            q_arr = np.asarray(quantum_probs, dtype=float)

        if len(safe_actions) == 1:
            return safe_actions[0], 1.0

        scores = np.full(len(self.action_names), -100.0)
        if self.recovery_mode and self.recovery_vector is not None:
            for a_idx in safe_actions:
                vec = np.array(self.action_vectors[self.action_names[a_idx]], dtype=float)
                v_norm = np.linalg.norm(vec)
                if v_norm > 1e-6:
                    u_v = vec / v_norm
                    scores[a_idx] = float(np.dot(u_v, self.recovery_vector))
                else:
                    scores[a_idx] = -1.0
        else:
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
                scores[a_idx] = q_arr[a_idx] + 0.15 * prog

                if self.action_history:
                    last_vec = np.array(self.action_vectors[self.action_history[-1]], dtype=float)
                    if np.linalg.norm(last_vec) > 1e-6:
                        u_last = last_vec / np.linalg.norm(last_vec)
                        cos_sim = float(np.dot(u_v, u_last))
                        if cos_sim > 0.9:
                            scores[a_idx] += 0.15
                        elif cos_sim < -0.7:
                            scores[a_idx] -= 0.30

        best_a_idx = int(np.argmax(scores))
        return best_a_idx, float(scores[best_a_idx])

    def select_action(self, state: Dict[str, Any], classical_action_idx: int, quantum_probs: Any) -> int:
        """Hybrid decision: classical for safety, quantum for high-level preference,
        with progress-aware scoring, action hysteresis, and deterministic tangential recovery.

        Args:
            state: Current environment state.
            classical_action_idx: Classical APF selected action (0-8).
            quantum_probs: Quantum model action probabilities (length 9 array or dict).

        Returns:
            Selected action index (0-8).
        """
        # Get safe actions from safety layer
        safe_actions = self.safety_layer.get_safe_actions(state)

        stay_idx = (
            self.config.navigation.action_names.index("STAY")
            if "STAY" in self.config.navigation.action_names
            else 8
        )
        if len(safe_actions) == 0:
            return stay_idx  # STAY (emergency)

        # Parse quantum probabilities
        if isinstance(quantum_probs, dict):
            q_arr = np.array(
                [quantum_probs.get(k, 0.0) for k in self.config.navigation.action_names],
                dtype=float,
            )
        else:
            q_arr = np.asarray(quantum_probs, dtype=float)

        q_weight = float(getattr(self.config.navigation, "hybrid_quantum_weight", 0.7))

        if len(safe_actions) == 1:
            selected_action = safe_actions[0]
            self.fusion_log.append({
                "step": state.get("step", self.env.current_step),
                "classical_action": int(classical_action_idx),
                "quantum_top_action": int(np.argmax(q_arr)),
                "fused_action": int(selected_action),
                "classical_action_name": self.action_names[classical_action_idx],
                "quantum_top_action_name": self.action_names[int(np.argmax(q_arr))],
                "fused_action_name": self.action_names[selected_action],
                "fusion_confidence": 1.0,
                "hybrid_weight": q_weight,
            })
            return selected_action

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
            # Progress-aware hybrid fusion with action hysteresis
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
                q_val = float(q_arr[a_idx])
                c_val = 1.0 if a_idx == classical_action_idx else 0.0

                scores[a_idx] = 0.5 * q_val + 0.3 * c_val + 0.2 * max(0.0, prog)

                # Action hysteresis
                if self.action_history:
                    last_vec = np.array(self.action_vectors[self.action_history[-1]], dtype=float)
                    if np.linalg.norm(last_vec) > 1e-6:
                        u_last = last_vec / np.linalg.norm(last_vec)
                        cos_sim = float(np.dot(u_v, u_last))
                        if cos_sim > 0.9:
                            scores[a_idx] += 0.15
                        elif cos_sim < -0.7:
                            scores[a_idx] -= 0.30

        selected_action = int(np.argmax(scores))

        self.fusion_log.append({
            "step": state.get("step", self.env.current_step),
            "classical_action": int(classical_action_idx),
            "quantum_top_action": int(np.argmax(q_arr)),
            "fused_action": int(selected_action),
            "classical_action_name": self.action_names[classical_action_idx],
            "quantum_top_action_name": self.action_names[int(np.argmax(q_arr))],
            "fused_action_name": self.action_names[selected_action],
            "fusion_confidence": float(scores[selected_action]),
            "hybrid_weight": q_weight,
        })

        return selected_action

    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute a single closed-loop navigation cycle.

        Returns:
            Tuple of (next_state, reward, done, step_telemetry)
        """
        import time

        # 1. Sense environment
        state = self.env.get_state()

        # Check oscillation / stagnation once per simulation step
        triggered, reason = self.detect_oscillation_and_stagnation(state)
        step_num = self.env.current_step
        if triggered and not self.recovery_mode and (step_num - self.last_recovery_step > 4):
            self.recovery_mode = True
            self.recovery_steps_left = 8
            self.last_recovery_step = step_num
            self.recovery_vector = self.compute_tangential_escape(state)
            self.recovery_tangent_sign *= -1.0
        if self.recovery_mode:
            self.recovery_steps_left -= 1
            if self.recovery_steps_left <= 0:
                self.recovery_mode = False

        safe_actions = self.safety_layer.get_safe_actions(state)

        # 2. Query classical APF recommendation
        c_dec = self.classical_navigator.compute_decision(state)
        c_action_name = str(c_dec.get("action", "STAY"))
        c_idx = (
            self.config.navigation.action_names.index(c_action_name)
            if c_action_name in self.config.navigation.action_names
            else 0
        )

        is_quantum = "Quantum" in type(self.navigator).__name__

        if is_quantum:
            t_dec_start = time.perf_counter()
            q_dec = self.navigator.compute_decision(state)
            dec_time = time.perf_counter() - t_dec_start
            self.decision_times.append(dec_time)

            qc_time = float(q_dec.get("circuit_execution_time", 0.0))
            self.circuit_times.append(qc_time)

            q_probs = q_dec.get("action_probabilities", {})
            q_top_name = str(q_dec.get("action", "STAY"))
            q_features = q_dec.get("feature_vector", None)

            # Evaluate pure quantum action choice
            q_idx, q_conf = self._compute_pure_quantum_action(state, q_probs, safe_actions=safe_actions)
            q_action_name = self.action_names[q_idx]

            # Evaluate hybrid quantum-classical fusion action choice
            fused_idx = self.select_action(state, c_idx, q_probs)
            h_action_name = self.action_names[fused_idx]
            h_conf = float(self.fusion_log[-1]["fusion_confidence"]) if self.fusion_log else 1.0

            if self.is_hybrid:
                chosen_name = h_action_name
                conf = h_conf
                hybrid_weight = float(getattr(self.config.navigation, "hybrid_quantum_weight", 0.7))
                m_type = "quantum_hybrid"
            else:
                chosen_name = q_action_name
                conf = q_conf
                hybrid_weight = 1.0
                m_type = "quantum"

            chosen_vec = np.array(self.action_vectors[chosen_name], dtype=float)
            decision = {
                "model_type": m_type,
                "action": chosen_name,
                "direction_vector": chosen_vec,
                "action_scores": {
                    name: (2.0 if name == chosen_name else 0.0)
                    for name in self.action_names
                },
                "action_probabilities": q_probs if isinstance(q_probs, dict) else {
                    name: float(q_probs[i])
                    for i, name in enumerate(self.action_names)
                },
                "confidence": conf,
                "circuit_execution_time": qc_time,
            }
        else:
            # Pure Classical APF navigator
            t_dec_start = time.perf_counter()
            decision = self.navigator.compute_decision(state)
            dec_time = time.perf_counter() - t_dec_start
            self.decision_times.append(dec_time)
            qc_time = 0.0
            self.circuit_times.append(qc_time)

            q_top_name = "N/A"
            q_features = None
            q_action_name = "N/A"
            h_action_name = "N/A"
            conf = float(decision.get("confidence", 1.0))
            hybrid_weight = 0.0

        # 3. Arbitrate and enforce safety constraints
        final_action = self.action_selector.select_action(
            decision=decision,
            current_pos=state["drone_position"],
            speed=self.env.drone.base_speed,
            obstacles=self.env.obstacles,
            map_width=self.env.map_width,
            map_height=self.env.map_height,
        )
        self.action_history.append(final_action["action"])

        # 4. Optional Explainability (XAI) feature sensitivity logging
        if self.explain_decisions and hasattr(self.navigator, "explain_decision"):
            t_exp_start = time.perf_counter()
            features = q_features if q_features is not None else (
                self.navigator.feature_extractor.extract_features(state)[0]
                if hasattr(self.navigator, "feature_extractor") else None
            )
            if features is not None:
                method = getattr(self.config.navigation, "explanation_method", "saliency")
                eps = getattr(self.config.navigation, "explanation_eps", 1e-4)
                exp = self.navigator.explain_decision(
                    features=features,
                    action=q_top_name,
                    method=method,
                    eps=eps,
                )
                final_exp = None
                if final_action["action"] != q_top_name:
                    final_exp = self.navigator.explain_decision(
                        features=features,
                        action=final_action["action"],
                        method=method,
                        eps=eps,
                    )
                exp_time = time.perf_counter() - t_exp_start
                self.explanation_times.append(exp_time)

                q_used = bool(q_top_name == final_action["action"])
                c_used = bool(c_action_name == final_action["action"]) if (self.is_hybrid and c_action_name) else False

                exp_entry = {
                    "step": int(self.env.current_step),
                    "classical_action": c_action_name if (self.is_hybrid and c_action_name) else "N/A",
                    "quantum_top_action": q_top_name,
                    "selected_action": final_action["action"],
                    "final_action": final_action["action"],
                    "quantum_action_explained": q_top_name,
                    "quantum_action_used": q_used,
                    "classical_action_used": c_used,
                    "quantum_action == final_action": q_used,
                    "classical_action == final_action": c_used,
                    "explanation": exp,
                    "final_action_explanation": final_exp or exp,
                    "top_3_features": exp.get("top_features", []),
                    "explanation_time": float(exp_time),
                }
                self.explanation_log.append(exp_entry)

        # 5. Actuate movement in simulation environment
        next_state, reward, done, info = self.env.step(
            direction=final_action["direction_vector"],
            speed=self.env.drone.base_speed,
        )

        # 6. Capture comprehensive step telemetry with distinct action breakdowns
        safety_override = bool(final_action["was_overridden"])
        telemetry = {
            "step": self.env.current_step,
            "drone_x": float(state["drone_position"][0]),
            "drone_y": float(state["drone_position"][1]),
            "target_x": float(state["target_position"][0]),
            "target_y": float(state["target_position"][1]),
            "distance_to_target": float(state["target_distance"]),
            "target_angle": float(
                np.arctan2(state["target_direction"][1], state["target_direction"][0])
            ),
            "nearest_obstacle_distance": float(state["nearest_obstacle_distance"]),
            "obstacle_density": float(state["obstacle_density"]),
            "heading": float(state["drone_heading"]),
            "velocity_x": float(state["drone_velocity"][0]),
            "velocity_y": float(state["drone_velocity"][1]),
            "boundary_distance": float(state["boundary_distance"]),
            "action": final_action["action"],
            "raw_action": final_action["raw_action"],
            "classical_action": str(c_action_name),
            "quantum_action": str(q_action_name),
            "hybrid_action": str(h_action_name),
            "final_action": str(final_action["action"]),
            "confidence": round(float(final_action.get("confidence", conf)), 4),
            "hybrid_weight": round(float(hybrid_weight), 4),
            "safety_override": bool(safety_override),
            "was_overridden": bool(safety_override),
            "decision_time": float(dec_time),
            "circuit_time": float(qc_time),
            "reward": float(reward),
            "done": bool(done),
            "success": bool(self.env.success),
            "collided": bool(self.env.collided),
            "model_type": final_action["model_type"],
        }
        self.history.append(telemetry)

        # Capture dynamic obstacles telemetry if active
        if getattr(self.env, "moving_obstacles", []):
            nearest_m_obs, nearest_m_dist, _ = self.env.obstacles.get_nearest_moving_obstacle(
                state["drone_position"], self.env.drone.radius
            )
            pred_min_clr = (
                float(self.safety_layer.last_proactive_event.get("predicted_min_clearance", nearest_m_dist))
                if self.safety_layer.last_proactive_event
                else float(nearest_m_dist)
            )
            pred_col = (
                bool(self.safety_layer.last_proactive_event.get("predicted_collision", False))
                if self.safety_layer.last_proactive_event
                else False
            )

            for m_obs in self.env.moving_obstacles:
                dyn_row = {
                    "time": round(float(self.env.sim_time), 3),
                    "step": int(self.env.current_step),
                    "UAV_x": round(float(state["drone_position"][0]), 3),
                    "UAV_y": round(float(state["drone_position"][1]), 3),
                    "target_x": round(float(state["target_position"][0]), 3),
                    "target_y": round(float(state["target_position"][1]), 3),
                    "obstacle_id": int(m_obs.obs_id),
                    "obstacle_x": round(float(m_obs.position[0]), 3),
                    "obstacle_y": round(float(m_obs.position[1]), 3),
                    "obstacle_velocity_x": round(float(m_obs.velocity[0]), 3),
                    "obstacle_velocity_y": round(float(m_obs.velocity[1]), 3),
                    "nearest_moving_obstacle_distance": round(float(nearest_m_dist), 3) if nearest_m_dist != float("inf") else 50.0,
                    "predicted_min_clearance": round(float(pred_min_clr), 3),
                    "predicted_collision": bool(pred_col),
                    "safety_intervention": bool(safety_override),
                    "safety_override": bool(safety_override),
                    "original_action": str(final_action["raw_action"]),
                    "classical_action": str(c_action_name),
                    "quantum_action": str(q_action_name),
                    "hybrid_action": str(h_action_name),
                    "final_action": str(final_action["action"]),
                    "confidence": round(float(final_action.get("confidence", conf)), 4),
                    "hybrid_weight": round(float(hybrid_weight), 4),
                }
                self.dynamic_telemetry_history.append(dyn_row)

        return next_state, reward, done, telemetry

    def run_episode(
        self,
        max_steps: Optional[int] = None,
        is_hybrid: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute full navigation episode until completion or step limit.

        Args:
            max_steps: Optional override for maximum simulation steps.
            is_hybrid: Optional toggle for hybrid execution mode.

        Returns:
            Dictionary containing episode trajectory, metrics summary, and step history.
        """
        if is_hybrid is not None:
            self.is_hybrid = is_hybrid
        limit = max_steps or self.env.max_steps
        done = False

        while not done and self.env.current_step < limit:
            _, _, done, _ = self.step()

        trajectory = self.env.drone.get_trajectory_coords()
        info = self.env.get_info()

        # Action switch count
        switches = 0
        for i in range(1, len(self.action_history)):
            if self.action_history[i] != self.action_history[i - 1]:
                switches += 1

        # Oscillation count (A -> B -> A pattern)
        oscillations = 0
        for i in range(2, len(self.action_history)):
            if self.action_history[i] == self.action_history[i - 2] and self.action_history[i] != self.action_history[i - 1]:
                oscillations += 1

        # Model type resolution
        nav_type = getattr(self.navigator, "model_type", None)
        if not nav_type:
            nav_type = "quantum" if "Quantum" in type(self.navigator).__name__ else "classical"
        if self.is_hybrid:
            nav_type = "quantum_assisted"

        # Dynamic clearance stats
        m_clearances = [
            float(r["nearest_moving_obstacle_distance"])
            for r in self.dynamic_telemetry_history
            if r.get("nearest_moving_obstacle_distance") is not None
        ]
        min_m_clr = float(min(m_clearances)) if m_clearances else 50.0
        mean_m_clr = float(np.mean(m_clearances)) if m_clearances else 50.0

        return {
            "model_type": nav_type,
            "is_hybrid": self.is_hybrid,
            "success": bool(self.env.success),
            "collided": bool(self.env.collided),
            "collision_count": int(self.env.collision_count),
            "total_steps": int(self.env.current_step),
            "path_length": float(self.env.get_path_length()),
            "final_distance": float(self.env.target.distance_to(self.env.drone.position)),
            "total_reward": float(self.env.total_reward),
            "safety_interventions": int(self.safety_layer.intervention_count),
            "unsafe_action_attempts": int(self.safety_layer.unsafe_action_attempts),
            "boundary_violations": int(self.safety_layer.boundary_violation_attempts),
            "min_obstacle_clearance": float(self.safety_layer.min_clearance_observed) if self.safety_layer.min_clearance_observed != float("inf") else 15.0,
            "min_boundary_clearance": float(self.safety_layer.min_boundary_clearance_observed) if self.safety_layer.min_boundary_clearance_observed != float("inf") else 25.0,
            "predicted_collision_count": int(self.safety_layer.predicted_collision_count),
            "min_moving_obstacle_clearance": round(min_m_clr, 3),
            "mean_moving_obstacle_clearance": round(mean_m_clr, 3),
            "mean_decision_time": float(np.mean(self.decision_times)) if self.decision_times else 0.0,
            "quantum_circuit_execution_time": float(sum(self.circuit_times)) if self.circuit_times else 0.0,
            "action_switch_count": int(switches),
            "oscillation_count": int(oscillations),
            "trajectory": trajectory,
            "action_history": self.action_history.copy(),
            "telemetry_history": self.history.copy(),
            "dynamic_telemetry_history": self.dynamic_telemetry_history.copy(),
            "fusion_log": self.fusion_log.copy(),
            "explanation_log": self.explanation_log.copy(),
            "total_explanation_time": float(sum(self.explanation_times)),
            "mean_explanation_time": float(np.mean(self.explanation_times)) if self.explanation_times else 0.0,
            "info": info,
        }

