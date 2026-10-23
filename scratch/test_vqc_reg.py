import sys
sys.path.insert(0, '.')
import time
import numpy as np
from quantum.training import hyperparameter_search

# Test hyperparameter search on a quick grid
grid = {
    'n_qubits': [4, 6],
    'n_layers': [2],
    'learning_rate': [0.005],
    'l2_strength': [0.1],
}

X_tr = np.random.uniform(0, 1, size=(20, 6))
y_tr = np.random.randint(0, 9, size=20)
X_v = np.random.uniform(0, 1, size=(10, 6))
y_v = np.random.randint(0, 9, size=10)

t0 = time.time()
best = hyperparameter_search(X_tr, y_tr, X_v, y_v, param_grid=grid)
dt = time.time() - t0
print(f"Hyperparameter search took {dt:.2f}s, best: {best}")
