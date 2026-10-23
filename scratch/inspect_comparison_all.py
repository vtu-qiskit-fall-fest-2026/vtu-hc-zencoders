import json

data = json.load(open('outputs/metrics/comparison.json'))
for s in data.get('scenario_comparisons', []):
    sc_id = s['scenario_id']
    seed = s['seed']
    c_s = s['classical']['metrics']['success']
    q_s = s['quantum']['metrics']['success']
    h_s = s['hybrid']['metrics']['success']
    print(f"{sc_id} (seed {seed}): Classical={c_s}, Quantum={q_s}, Hybrid={h_s}")
