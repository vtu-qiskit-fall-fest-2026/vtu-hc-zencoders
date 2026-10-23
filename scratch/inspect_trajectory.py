import json

data = json.load(open('outputs/metrics/comparison.json'))
for sc in data.get('scenario_comparisons', []):
    if sc['scenario_id'] == 'scenario_01' and sc['seed'] == 42:
        traj_q = sc['quantum']['trajectory']
        print(f"Scenario 01 Quantum trajectory (first 25 points):")
        for i, pt in enumerate(traj_q[:25]):
            print(f"  step {i}: pos=({pt[0]:.2f}, {pt[1]:.2f})")
        print(f"Scenario 01 Quantum trajectory (points 25 to 50):")
        for i, pt in enumerate(traj_q[25:50], start=25):
            print(f"  step {i}: pos=({pt[0]:.2f}, {pt[1]:.2f})")
        break
