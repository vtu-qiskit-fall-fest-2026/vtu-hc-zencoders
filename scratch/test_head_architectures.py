import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from quantum.quantum_model import VariationalQuantumNavigator
from quantum.feature_encoding import QuantumFeatureEncoder
from scratch.test_separability import generate_multi_sector_dataset
from scratch.test_encodings import encode_strategy_a_grouped, encode_strategy_c_bounded_nonlinear

# Let's test on PyTorch or NumPy for maximum speed and scientific rigor
class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=9):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.fc(x)

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, n_classes=9, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class HybridResidualHead(nn.Module):
    def __init__(self, q_dim, c_dim=12, hidden_dim=32, n_classes=9, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim + c_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x_q, x_c):
        x = torch.cat([x_q, x_c], dim=1)
        return self.net(x)

def run_head_comparison():
    print("Generating dataset...", flush=True)
    data, _, _ = generate_multi_sector_dataset(3000, seed=42)
    trajs = sorted(list(set(d["trajectory_id"] for d in data)))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(trajs)
    n_val = int(len(trajs) * 0.2)
    val_set = set(shuffled[:n_val])

    train_data = [d for d in data if d["trajectory_id"] not in val_set]
    val_data = [d for d in data if d["trajectory_id"] in val_set]

    # Test with 4-qubit and 6-qubit VQC models
    for n_q in [4, 6]:
        print(f"\n==========================================", flush=True)
        print(f"Testing Quantum Observables with n_qubits={n_q}", flush=True)
        print(f"==========================================", flush=True)
        model = VariationalQuantumNavigator(n_qubits=n_q, n_layers=2, seed=42, load_checkpoint=False)

        # Extract quantum features
        print(f"Extracting quantum features for {len(train_data)} train and {len(val_data)} val samples...", flush=True)
        t0 = time.time()
        X_q_train = []
        for d in train_data:
            ang = encode_strategy_c_bounded_nonlinear(d["features"], n_q)
            exp_vals, basis_probs, _ = model.circuit.execute(ang, model.weights)
            X_q_train.append(np.concatenate([exp_vals, basis_probs]))
        X_q_train = np.array(X_q_train, dtype=np.float32)

        X_q_val = []
        for d in val_data:
            ang = encode_strategy_c_bounded_nonlinear(d["features"], n_q)
            exp_vals, basis_probs, _ = model.circuit.execute(ang, model.weights)
            X_q_val.append(np.concatenate([exp_vals, basis_probs]))
        X_q_val = np.array(X_q_val, dtype=np.float32)

        X_c_train = np.array([d["features"] for d in train_data], dtype=np.float32)
        X_c_val = np.array([d["features"] for d in val_data], dtype=np.float32)

        y_train = np.array([d["teacher_index"] for d in train_data], dtype=np.int64)
        y_val = np.array([d["teacher_index"] for d in val_data], dtype=np.int64)
        print(f"Quantum feature extraction completed in {time.time()-t0:.2f}s. Dim: {X_q_train.shape[1]}", flush=True)

        q_dim = X_q_train.shape[1]

        # 1. Linear Head
        train_ds = TensorDataset(torch.from_numpy(X_q_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        lin_head = LinearHead(q_dim)
        optimizer = torch.optim.AdamW(lin_head.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_lin_acc = 0.0
        best_lin_f1 = 0.0
        for epoch in range(35):
            lin_head.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = lin_head(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            lin_head.eval()
            with torch.no_grad():
                preds = lin_head(torch.from_numpy(X_q_val)).argmax(dim=1).numpy()
                acc = accuracy_score(y_val, preds) * 100
                f1 = f1_score(y_val, preds, average='macro')
                if acc > best_lin_acc:
                    best_lin_acc = acc
                    best_lin_f1 = f1
        print(f"Head 1 (Linear):            Val Acc = {best_lin_acc:.2f}%, Macro F1 = {best_lin_f1:.4f}", flush=True)

        # 2. MLP Head
        mlp_head = MLPHead(q_dim, hidden_dim=32, dropout=0.05)
        optimizer = torch.optim.AdamW(mlp_head.parameters(), lr=0.005, weight_decay=1e-4)
        best_mlp_acc = 0.0
        best_mlp_f1 = 0.0
        for epoch in range(45):
            mlp_head.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = mlp_head(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            mlp_head.eval()
            with torch.no_grad():
                preds = mlp_head(torch.from_numpy(X_q_val)).argmax(dim=1).numpy()
                acc = accuracy_score(y_val, preds) * 100
                f1 = f1_score(y_val, preds, average='macro')
                if acc > best_mlp_acc:
                    best_mlp_acc = acc
                    best_mlp_f1 = f1
        print(f"Head 2 (MLP 32):            Val Acc = {best_mlp_acc:.2f}%, Macro F1 = {best_mlp_f1:.4f}", flush=True)

        # 3. Hybrid Quantum-Classical Residual Head
        hybrid_train_ds = TensorDataset(torch.from_numpy(X_q_train), torch.from_numpy(X_c_train), torch.from_numpy(y_train))
        hybrid_loader = DataLoader(hybrid_train_ds, batch_size=32, shuffle=True)
        hyb_head = HybridResidualHead(q_dim, c_dim=12, hidden_dim=32, dropout=0.05)
        optimizer = torch.optim.AdamW(hyb_head.parameters(), lr=0.005, weight_decay=1e-4)
        best_hyb_acc = 0.0
        best_hyb_f1 = 0.0
        for epoch in range(45):
            hyb_head.train()
            for bx_q, bx_c, by in hybrid_loader:
                optimizer.zero_grad()
                out = hyb_head(bx_q, bx_c)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            hyb_head.eval()
            with torch.no_grad():
                preds = hyb_head(torch.from_numpy(X_q_val), torch.from_numpy(X_c_val)).argmax(dim=1).numpy()
                acc = accuracy_score(y_val, preds) * 100
                f1 = f1_score(y_val, preds, average='macro')
                if acc > best_hyb_acc:
                    best_hyb_acc = acc
                    best_hyb_f1 = f1
        print(f"Head 3 (Hybrid Residual):   Val Acc = {best_hyb_acc:.2f}%, Macro F1 = {best_hyb_f1:.4f}", flush=True)

if __name__ == "__main__":
    run_head_comparison()
