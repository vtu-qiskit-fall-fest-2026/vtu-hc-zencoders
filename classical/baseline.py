"""Classical Baseline Navigator for Q-Nav.

Coordinates feature extraction, Artificial Potential Field gradient evaluation,
discrete action scoring, and candidate selection.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from config import QNavConfig, DEFAULT_CONFIG
from classical.feature_extractor import NavigationFeatureExtractor
from classical.potential_field import ArtificialPotentialField
from simulation.obstacles import ObstacleField


class ClassicalBaselineNavigator:
    """Autonomous navigation agent using classical features and Artificial Potential Fields."""

    def __init__(
        self,
        config: Optional[Any] = None,
        obstacles: Optional[ObstacleField] = None,
        env: Optional[Any] = None,
    ):
        """Initialize the baseline navigator.

        Args:
            config: Master configuration or Environment instance.
            obstacles: Reference to ObstacleField in the environment.
            env: Optional environment reference.
        """
        self.env = env
        if config is not None and (hasattr(config, "get_state") or hasattr(config, "obstacles")):
            self.env = config
            real_config = getattr(config, "config", DEFAULT_CONFIG)
            real_obs = getattr(config, "obstacles", obstacles)
            config = real_config
            obstacles = real_obs
        elif env is not None:
            if obstacles is None:
                obstacles = getattr(env, "obstacles", None)
            if config is None:
                config = getattr(env, "config", DEFAULT_CONFIG)

        self.config = config or DEFAULT_CONFIG
        self.feature_extractor = NavigationFeatureExtractor(
            max_map_distance=np.sqrt(
                self.config.sim.map_width**2 + self.config.sim.map_height**2
            ),
            max_obstacle_distance=self.config.classical.obstacle_influence_distance,
            max_speed=self.config.sim.drone_speed * 1.5,
        )
        self.apf = ArtificialPotentialField(config=self.config.classical)
        self.obstacles = obstacles or ObstacleField()

        self.action_vectors = self.config.navigation.action_vectors
        self.action_names = self.config.navigation.action_names
        self.last_action_vector: Optional[np.ndarray] = None

    def set_obstacles(self, obstacles: ObstacleField) -> None:
        """Update obstacle field reference."""
        self.obstacles = obstacles

    def reset(self) -> None:
        """Reset internal APF state and local minima history."""
        self.apf.reset()
        self.last_action_vector = None

    def compute_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the navigation action decision from the current environment state.

        Args:
            state: Dictionary output from UAVEnvironment.get_state().

        Returns:
            Dictionary containing chosen action, direction vector, action scores,
            probability distribution, confidence, and APF metadata.
        """
        # Ensure state contains obstacle reference if available
        if "obstacles" not in state or state["obstacles"] is None:
            state["obstacles"] = self.obstacles

        # 1. Feature extraction (12 normalized features)
        feature_vec, feature_dict = self.feature_extractor.extract_features(state)

        drone_pos = state["drone_position"]
        target_pos = state["target_position"]

        # 2. Potential Field Force calculation with stagnation & recovery
        f_total, breakdown = self.apf.compute_total_force(
            drone_pos=drone_pos,
            target_pos=target_pos,
            obstacles=self.obstacles,
            map_width=state.get("map_width", self.config.sim.map_width),
            map_height=state.get("map_height", self.config.sim.map_height),
            drone_radius=self.config.sim.drone_radius,
            last_action_vector=self.last_action_vector,
        )

        f_norm = np.linalg.norm(f_total)
        desired_dir = f_total / f_norm if f_norm > 1e-6 else np.zeros(2, dtype=float)

        # 3. Score candidate actions against desired force vector
        action_scores: Dict[str, float] = {}

        for name in self.action_names:
            vec = np.array(self.action_vectors[name], dtype=float)
            vec_norm = np.linalg.norm(vec)

            if vec_norm < 1e-6:
                # STAY action: preferred only if already inside target
                if state["target_distance"] <= self.config.sim.target_detection_radius:
                    action_scores[name] = 2.0
                else:
                    action_scores[name] = -1.0
            else:
                unit_v = vec / vec_norm
                alignment = float(np.dot(unit_v, desired_dir))
                action_scores[name] = alignment

        # 4. Softmax over action scores to produce probabilities
        score_values = np.array([action_scores[k] for k in self.action_names], dtype=float)
        exp_scores = np.exp(score_values - np.max(score_values))
        probs = exp_scores / np.sum(exp_scores)
        action_probs = {name: float(probs[i]) for i, name in enumerate(self.action_names)}

        # 5. Greedy selection
        best_action_idx = int(np.argmax(score_values))
        chosen_action = self.action_names[best_action_idx]
        chosen_vector = np.array(self.action_vectors[chosen_action], dtype=float)
        confidence = float(probs[best_action_idx])

        self.last_action_vector = chosen_vector.copy()

        return {
            "model_type": "classical",
            "action": chosen_action,
            "direction_vector": chosen_vector,
            "action_scores": action_scores,
            "action_probabilities": action_probs,
            "confidence": confidence,
            "feature_vector": feature_vec,
            "feature_dict": feature_dict,
            "desired_direction": desired_dir,
            "apf_breakdown": breakdown,
        }

    def select_action(self, features_or_state: Optional[Any] = None) -> int:
        """Select discrete navigation action index (0-8) using APF or environment state.

        Args:
            features_or_state: 1D features array, state dict, or None (queries self.env).

        Returns:
            Integer action index in range [0, 8].
        """
        if isinstance(features_or_state, dict):
            state = features_or_state
        elif self.env is not None and hasattr(self.env, "get_state"):
            state = self.env.get_state()
        elif isinstance(features_or_state, (list, np.ndarray)) and len(features_or_state) >= 12:
            feat_arr = np.asarray(features_or_state, dtype=float)
            t_x, t_y = float(feat_arr[1]), float(feat_arr[2])
            best_idx = 0
            best_dot = -float("inf")
            for idx, name in enumerate(self.action_names):
                vec = np.array(self.action_vectors[name], dtype=float)
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    continue
                dot = (vec[0] * t_x + vec[1] * t_y) / norm
                if dot > best_dot:
                    best_dot = dot
                    best_idx = idx
            return best_idx
        else:
            state = {}

        dec = self.compute_decision(state)
        chosen_action = dec.get("action", "STAY")
        if chosen_action in self.action_names:
            return self.action_names.index(chosen_action)
        return int(np.argmax(dec.get("action_scores", [0.0] * len(self.action_names))))


ClassicalBaseline = ClassicalBaselineNavigator
