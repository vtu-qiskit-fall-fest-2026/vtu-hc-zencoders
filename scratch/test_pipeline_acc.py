import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.training import generate_augmented_training_data
from quantum.quantum_model import VariationalQuantumNavigator
import numpy as np

X_tr, y_tr, X_v, y_v = generate_augmented_training_data(
    n_episodes=16,
    augmentation_factor=2,
    n_features=6,
    seed=42,
)

print(f"X_tr shape: {X_tr.shape}, X_v shape: {X_v.shape}")
print(f"y_tr classes: {np.unique(y_tr, return_counts=True)}")
print(f"y_v classes: {np.unique(y_v, return_counts=True)}")

model = VariationalQuantumNavigator(n_qubits=6, n_layers=3, seed=42, load_checkpoint=False)
history = model.train_with_regularization(
    X_tr, y_tr, X_v, y_v,
    n_epochs=10,
    learning_rate=0.005,
    l2_strength=0.1,
    early_stopping=True
)

print("History val_acc:", history['val_acc'])
print("History train_acc:", history['train_acc'])
