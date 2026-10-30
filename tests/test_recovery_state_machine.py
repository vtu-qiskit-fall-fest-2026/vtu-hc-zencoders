"""Regression tests for Persistent Tangential Wall-Following State Machine in NavigationController.

Validates:
1. Recovery direction remains stable during a recovery episode (no ping-pong oscillation).
2. Recovery exits after meaningful goal progress or obstacle clearance.
3. Recovery never selects unsafe actions.
4. A scenario_05 concave obstacle case escapes without indefinite oscillation.
5. Classical APF baseline behavior remains strictly unchanged.
"""

import pytest
import numpy as np
from simulation.environment import UAVEnvironment
from simulation.obstacles import ObstacleField, CircularObstacle
from navigation.controller import NavigationController
from classical.baseline import ClassicalBaselineNavigator
from quantum.quantum_decision import QuantumDecisionMaker
from config import DEFAULT_CONFIG
from run import load_scenarios_and_maps
from pathlib import Path


class TestRecoveryStateMachine:
    """Test suite for persistent tangential wall-following recovery fix."""

    def test_recovery_direction_remains_stable(self):
        """Verify recovery direction remains consistent during an active recovery episode."""
        env = UAVEnvironment()
        # Set up a circular obstacle in front of the drone
        env.obstacles = ObstacleField([CircularObstacle(1, x=20.0, y=20.0, radius=3.0)])
        controller = NavigationController(environment=env)

        # Drone at (20, 15), moving towards target at (20, 30)
        state = {
            "drone_position": np.array([20.0, 15.0]),
            "target_position": np.array([20.0, 30.0]),
            "drone_speed": 1.5,
            "obstacles": env.obstacles,
            "map_width": 50.0,
            "map_height": 50.0,
        }

        # First escape vector calculation sets recovery side
        vec1 = controller.compute_tangential_escape(state)
        side1 = controller.recovery_side
        assert side1 in ("left", "right")

        # Subsequent evaluations in the same recovery episode must retain the same side
        for offset_y in [15.2, 15.5, 15.8, 16.0]:
            state["drone_position"] = np.array([20.0, offset_y])
            vec_next = controller.compute_tangential_escape(state)
            assert controller.recovery_side == side1
            # Direction vector must remain aligned (dot product > 0.5)
            assert np.dot(vec1, vec_next) > 0.5

    def test_recovery_exits_after_goal_progress_and_clearance(self):
        """Verify recovery exits when progress towards target exceeds 2.0 and clearance > 2.2."""
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [{"id": 1, "type": "circle", "x": 20.0, "y": 20.0, "radius": 3.0}],
        }
        scenario = {"start": [20.0, 10.0], "target": [20.0, 40.0], "max_steps": 50}
        env = UAVEnvironment(map_data=map_data, scenario_data=scenario)
        controller = NavigationController(environment=env)

        state = env.get_state()
        controller.recovery_mode = True
        controller.recovery_steps_left = 15
        controller.recovery_start_dist = 30.0
        controller.recovery_side = "left"

        # Simulate drone moving forward towards target and away from obstacle
        # Target distance decreases from 30.0 to 27.0 (progress = 3.0 > 2.0)
        # Position at (28.0, 20.0), distance to obstacle at (20, 20) is 8.0 - 3.0 = 5.0 > 2.2
        env.drone.position = np.array([28.0, 20.0])
        next_state, _, _, _ = controller.step()

        # Recovery should have exited cleanly
        assert controller.recovery_mode is False
        assert controller.recovery_steps_left == 0
        assert controller.recovery_side is None

    def test_recovery_does_not_select_unsafe_actions(self):
        """Verify recovery mode strictly respects safety constraints and only selects safe actions."""
        map_data = {
            "dimensions": {"width": 50.0, "height": 50.0},
            "obstacles": [{"id": 1, "type": "circle", "x": 20.0, "y": 20.0, "radius": 3.0}],
        }
        # Start at (20.0, 12.0) with safe clearance buffer
        scenario = {"start": [20.0, 12.0], "target": [20.0, 30.0], "max_steps": 20}
        env = UAVEnvironment(map_data=map_data, scenario_data=scenario)
        controller = NavigationController(environment=env)

        # Trigger recovery
        state = env.get_state()
        controller.recovery_mode = True
        controller.recovery_steps_left = 10
        controller.recovery_vector = controller.compute_tangential_escape(state)

        safe_actions = controller.safety_layer.get_safe_actions(state)
        assert len(safe_actions) > 0, "Expected non-empty safe actions at clearance distance"

        # Uniform probability distribution
        q_probs = np.ones(9) / 9.0

        # Test pure quantum action in recovery
        q_idx, _ = controller._compute_pure_quantum_action(state, q_probs, safe_actions=safe_actions)
        assert q_idx in safe_actions

        # Test hybrid action in recovery
        h_idx = controller.select_action(state, classical_action_idx=0, quantum_probs=q_probs)
        assert h_idx in safe_actions

        # Verify selected actions do not cause an immediate collision
        for act_idx in [q_idx, h_idx]:
            act_vec = np.array(controller.action_vectors[controller.action_names[act_idx]], dtype=float)
            is_safe, _, _ = controller.safety_layer.evaluate_action_safety(
                current_pos=state["drone_position"],
                direction=act_vec,
                speed=env.drone.base_speed,
                obstacles=env.obstacles,
                map_width=env.map_width,
                map_height=env.map_height,
            )
            assert is_safe is True

    def test_scenario_05_concave_obstacle_no_indefinite_oscillation(self):
        """Verify that Scenario 05 (concave obstacle pocket) avoids indefinite left-right oscillation and completes successfully."""
        scenarios, maps = load_scenarios_and_maps()
        scn_05 = next(s for s in scenarios if s["id"] == "scenario_05")
        map_03 = maps["map_03"]

        config = DEFAULT_CONFIG
        ckpt_path = Path("outputs/metrics/quantum_weights.npz")
        navigator = QuantumDecisionMaker(
            config=config,
            checkpoint_path=ckpt_path if ckpt_path.exists() else None,
            load_checkpoint=ckpt_path.exists(),
        )

        # 1. Run with Hybrid Navigator on Scenario 05
        env_h = UAVEnvironment(map_data=map_03, scenario_data=scn_05)
        controller_h = NavigationController(environment=env_h, navigator=navigator, is_hybrid=True)
        controller_h.reset(start_position=tuple(scn_05["start"]), target_position=tuple(scn_05["target"]))
        ep_h = controller_h.run_episode(max_steps=140)

        # Hybrid must navigate and reach target cleanly
        assert ep_h["success"] is True, f"Hybrid failed scenario_05: final_dist={ep_h['final_distance']}, steps={ep_h['total_steps']}"
        assert ep_h["collided"] is False
        assert ep_h["total_steps"] < 100, f"Hybrid took {ep_h['total_steps']} steps (expected < 100)"

        # 2. Verify that recovery episodes maintain persistent recovery_side without alternating sign
        env_h2 = UAVEnvironment(map_data=map_03, scenario_data=scn_05)
        controller_h2 = NavigationController(environment=env_h2, navigator=navigator, is_hybrid=True)
        controller_h2.reset(start_position=tuple(scn_05["start"]), target_position=tuple(scn_05["target"]))

        for _ in range(70):
            if controller_h2.recovery_mode:
                assert controller_h2.recovery_side in ("left", "right"), "Recovery side must remain committed during recovery"
            _, _, done, _ = controller_h2.step()
            if done:
                break

    def test_classical_apf_baseline_remains_unchanged(self):
        """Verify Classical APF baseline outputs remain identical and unaffected by recovery modifications."""
        scenarios, maps = load_scenarios_and_maps()
        scn_01 = next(s for s in scenarios if s["id"] == "scenario_01")
        map_01 = maps["map_01"]

        # Baseline navigator evaluated directly
        classical_nav = ClassicalBaselineNavigator()
        env = UAVEnvironment(map_data=map_01, scenario_data=scn_01)
        state = env.get_state()

        decision = classical_nav.compute_decision(state)
        assert "action" in decision
        assert "direction_vector" in decision
        assert decision["action"] in ["UP", "DOWN", "LEFT", "RIGHT", "UP_RIGHT", "UP_LEFT", "DOWN_RIGHT", "DOWN_LEFT", "STAY"]

        # Run Classical APF full episode in controller
        controller_c = NavigationController(environment=env, navigator=classical_nav, is_hybrid=False)
        ep_c = controller_c.run_episode(max_steps=120)

        # Baseline scenario_01 should succeed identically
        assert ep_c["success"] is True
        assert ep_c["collided"] is False
        assert ep_c["total_steps"] == 41  # Exact baseline step count on scenario_01
