import json
from pathlib import Path

data = json.load(open("outputs/metrics/comparison.json"))
print(f"Total trials: {data['total_trials']}")
print(f"Seeds: {data['seeds']}")

print("\n--- EPISODE RESULTS ---")
for item in data["scenario_comparisons"]:
    sc = item["scenario_id"]
    sd = item["seed"]
    c_s = item["classical"]["result"]["success"]
    q_s = item["quantum"]["result"]["success"]
    h_s = item["hybrid"]["result"]["success"]
    c_steps = item["classical"]["result"]["total_steps"]
    q_steps = item["quantum"]["result"]["total_steps"]
    h_steps = item["hybrid"]["result"]["total_steps"]
    c_col = item["classical"]["result"]["collision_count"]
    q_col = item["quantum"]["result"]["collision_count"]
    h_col = item["hybrid"]["result"]["collision_count"]
    print(f"{sc} [seed {sd}]: Classical={c_s} ({c_steps} steps, col={c_col}) | Pure Quantum={q_s} ({q_steps} steps, col={q_col}) | Hybrid={h_s} ({h_steps} steps, col={h_col})")

print("\n--- SUMMARY ---")
for m in ["classical", "quantum", "hybrid"]:
    s = data["summary"][m]
    print(f"{m.upper()}: Success Rate = {s['success_rate']*100:.1f}% ({s['success_count']}/{s['total_episodes']}), Collisions = {s['mean_collision_count']}, Collision Rate = {s['mean_collision_rate']}")
