import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, SCENARIOS_DIR, MAPS_DIR
from utils.io_utils import load_json
from evaluation.compare_models import ModelComparator

comparator = ModelComparator(config=DEFAULT_CONFIG)

scenarios = ["scenario_01", "scenario_02", "scenario_03", "scenario_04", "scenario_05"]

for sc_id in scenarios:
    sc_data = load_json(SCENARIOS_DIR / f"{sc_id}.json")
    map_data = load_json(MAPS_DIR / f"{sc_data['map_id']}.json")
    print(f"\n=================== {sc_id} (Seed 42) ===================")
    for mode in ["classical", "quantum", "hybrid"]:
        ep_res, metrics = comparator.run_single_model_scenario(
            model_type=mode,
            scenario=sc_data,
            map_data=map_data,
            seed=42,
        )
        succ = metrics["success"]
        steps = metrics["steps"]
        f_dist = metrics["final_distance_to_target"]
        coll = metrics["collision_count"]
        sw = ep_res.get("action_switch_count", 0)
        osc = ep_res.get("oscillation_count", 0)
        print(f"  {mode:10s} -> Success: {succ!s:5s} | Steps: {steps:3d} | FinalDist: {f_dist:5.2f} | Collisions: {coll} | Switches: {sw} | Osc: {osc}")
