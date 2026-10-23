import json

data = json.load(open('outputs/metrics/comparison.json'))
for sc in data.get('scenario_comparisons', []):
    c = sc['classical']['metrics']
    q = sc['quantum']['metrics']
    h = sc.get('hybrid', {}).get('metrics', {})
    print(f"{sc['scenario_id']} seed={sc['seed']}: Classical={c.get('success')} ({c.get('steps_taken')} steps, dist={c.get('final_distance_to_target', 0):.1f}), Quantum={q.get('success')} ({q.get('steps_taken')} steps, dist={q.get('final_distance_to_target', 0):.1f}), Hybrid={h.get('success')} ({h.get('steps_taken')} steps, dist={h.get('final_distance_to_target', 0):.1f})")

summary = data.get('summary', {})
print("\nSUMMARY:")
for m in ['classical', 'quantum', 'hybrid']:
    if m in summary:
        print(f"{m}: success_rate={summary[m].get('success_rate'):.1%}, mean_steps={summary[m].get('mean_steps')}")
