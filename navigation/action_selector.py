"""Action Selector Module for Q-Nav.

Ranks and arbitrates candidate actions from Classical or Quantum decision makers,
enforcing safety constraints and producing the final execution command.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from config import NavigationConfig, DEFAULT_CONFIG
from navigation.safety import SafetyLayer
from simulation.obstacles import ObstacleField


class ActionSelector:
    """Selects and filters navigation actions based on model scores and safety layer validation."""

    def __init__(
        self,
        safety_layer: Optional[SafetyLayer] = None,
        config: Optional[NavigationConfig] = None,
    ):
        """Initialize action selector with safety layer reference.

        Args:
            safety_layer: Reference to SafetyLayer instance.
            config: Navigation configuration.
        """
        self.config = config or DEFAULT_CONFIG.navigation
        self.safety_layer = safety_layer or SafetyLayer(nav_config=self.config)
        self.action_names = self.config.action_names
        self.action_vectors = self.config.action_vectors

    def select_action(
        self,
        decision: Dict[str, Any],
        current_pos: np.ndarray,
        speed: float,
        obstacles: ObstacleField,
        map_width: float = 50.0,
        map_height: float = 50.0,
    ) -> Dict[str, Any]:
        """Filter decision through safety layer and produce final motion command.

        Args:
            decision: Raw decision dict from Classical or Quantum model.
            current_pos: Drone position.
            speed: Current speed.
            obstacles: Obstacles in environment.
            map_width: Map width.
            map_height: Map height.

        Returns:
            Dictionary with finalized action, direction vector, and safety intervention flags.
        """
        raw_action = decision.get("action", "STAY")
        action_scores = decision.get("action_scores", {})

        # Safety-filtered action selection
        final_action, final_vector, was_intervened = self.safety_layer.select_safe_action(
            current_pos=current_pos,
            candidate_actions=self.action_names,
            action_scores=action_scores,
            action_vectors=self.action_vectors,
            speed=speed,
            obstacles=obstacles,
            map_width=map_width,
            map_height=map_height,
        )

        return {
            "action": final_action,
            "direction_vector": final_vector,
            "raw_action": raw_action,
            "was_overridden": was_intervened,
            "confidence": decision.get("confidence", 0.0),
            "model_type": decision.get("model_type", "unknown"),
            "action_scores": action_scores,
            "action_probabilities": decision.get("action_probabilities", {}),
        }
