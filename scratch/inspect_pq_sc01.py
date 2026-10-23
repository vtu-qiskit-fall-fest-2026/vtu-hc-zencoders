import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratch.test_enhanced_controller import EnhancedNavigationController
from config import DEFAULT_CONFIG, SCENARIOS_DIR, MAPS_DIR
from utils.io_utils import load_json
from simulation.environment import UAVEnvironment
from quantum.quantum_decision import QuantumDecisionMaker

sc_data = load_json(SCENARIOS_DIR / "scenario_01.json")
map_data = load_json(MAPS_DIR / f"{sc_data['map_id']}.json")
env = UAVEnvironment(config=DEFAULT_CONFIG, map_data=map_data, scenario_data=sc_data)
q_nav = QuantumDecisionMaker(config=DEFAULT_CONFIG, checkpoint_path=Path("outputs/metrics/quantum_weights.npz"), load_checkpoint=True)

ctrl = EnhancedNavigationController(env=env, navigator=q_nav, config=DEFAULT_CONFIG, is_hybrid=False)
ctrl.reset(tuple(sc_data["start"]), tuple(sc_data["target"]))

for i in range(40):
    ctrl.step()
    print(f"Step {i+1}: Pos={ctrl.env.drone.position}, Action={ctrl.actions[-1]}, Rec={ctrl.recovery_mode}, RecSteps={ctrl.recovery_steps_left}")
