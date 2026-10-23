import time
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from quantum.quantum_model import VariationalQuantumNavigator
from quantum.quantum_circuit import ParameterizedNavigationCircuit
from scratch.test_separability import generate_multi_sector_dataset
from scratch.test_encodings import encode_strategy_c_bounded_nonlinear, encode_strategy_a_grouped

def train_vqc_experiment(
    n_qubits=4,
    n_layers=2,
    head_type="linear",  # "linear", "mlp", "hybrid"
    n_samples=3000,
    epochs=15,
    lr=0.01,
    vqc_lr=0.008,
    batch_size=32,
    seed=42,
):
    """Controlled scientific VQC training experiment with joint theta & head optimization."""
    rng = np.random.default_rng(seed)
    print(f"\n--- Training Experiment: {n_qubits}Q/{n_layers}L, Head={head_type}, Seed={seed} ---", flush=True)

    # 1. Dataset
    t0 = time.time()
    data, counter, n_trajs = generate_multi_sector_dataset(n_samples, seed=seed)
    trajs = sorted(list(set(d["trajectory_id"] for d in data)))
    shuffled_trajs = rng.permutation(trajs)
    n_val = int(len(trajs) * 0.2)
    val_set = set(shuffled_trajs[:n_val])

    train_data = [d for d in data if d["trajectory_id"] not in val_set]
    val_data = [d for d in data if d["trajectory_id"] in val_set]

    y_train = np.array([d["teacher_index"] for d in train_data], dtype=int)
    y_val = np.array([d["teacher_index"] for d in val_data], dtype=int)

    n_train = len(train_data)
    n_val = len(val_data)
    num_classes = 9

    # Encode angles
    train_angles = [encode_strategy_c_bounded_nonlinear(d["features"], n_qubits) for d in train_data]
    val_angles = [encode_strategy_c_bounded_nonlinear(d["features"], n_qubits) for d in val_data]

    train_12d = np.array([d["features"] for d in train_data], dtype=float)
    val_12d = np.array([d["features"] for d in val_data], dtype=float)

    # Initialize circuit and weights
    circuit = ParameterizedNavigationCircuit(n_qubits=n_qubits, n_layers=n_layers)
    n_weights = circuit.get_num_weights()
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_weights)

    q_feat_dim = n_qubits + (1 << n_qubits)
    in_dim = q_feat_dim if head_type != "hybrid" else (q_feat_dim + 12)

    # Head parameters
    if head_type == "linear":
        scale = np.sqrt(2.0 / (in_dim + num_classes))
        W = rng.normal(0.0, scale, size=(num_classes, in_dim))
        b = np.zeros(num_classes)
        params_head = [W, b]
    elif head_type in ("mlp", "hybrid"):
        h_dim = 32
        scale1 = np.sqrt(2.0 / (in_dim + h_dim))
        W1 = rng.normal(0.0, scale1, size=(h_dim, in_dim))
        b1 = np.zeros(h_dim)
        scale2 = np.sqrt(2.0 / (h_dim + num_classes))
        W2 = rng.normal(0.0, scale2, size=(num_classes, h_dim))
        b2 = np.zeros(num_classes)
        params_head = [W1, b1, W2, b2]

    # Adam moments
    m_head = [np.zeros_like(p) for p in params_head]
    v_head = [np.zeros_like(p) for p in params_head]
    m_th = np.zeros_like(theta)
    v_th = np.zeros_like(theta)
    adam_step = 0

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint = None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    print(f"Data ready: {n_train} train, {n_val} val ({len(trajs)} trajs). Optimizing {n_weights} VQC weights...", flush=True)

    for epoch in range(epochs):
        epoch_t0 = time.time()
        # 1. Compute quantum features for current theta on train data
        X_q_train = []
        for i in range(n_train):
            exp_v, basis_p, _ = circuit.execute(train_angles[i], theta)
            X_q_train.append(np.concatenate([exp_v, basis_p]))
        X_q_train = np.array(X_q_train, dtype=float)

        if head_type == "hybrid":
            X_in_train = np.concatenate([X_q_train, train_12d], axis=1)
        else:
            X_in_train = X_q_train

        perm = rng.permutation(n_train)
        epoch_losses = []
        epoch_correct = 0

        # Mini-batch gradient descent
        for b_start in range(0, n_train, batch_size):
            adam_step += 1
            b_end = min(b_start + batch_size, n_train)
            batch_idx = perm[b_start:b_end]
            bx = X_in_train[batch_idx]
            by = y_train[batch_idx]
            bs = len(batch_idx)

            if head_type == "linear":
                logits = bx @ W.T + b
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                loss = float(np.mean(-np.log(probs[np.arange(bs), by] + 1e-7)))
                epoch_losses.append(loss)
                epoch_correct += np.sum(np.argmax(probs, axis=1) == by)

                grad_logits = (probs.copy())
                grad_logits[np.arange(bs), by] -= 1.0
                grad_logits /= bs

                grad_W = grad_logits.T @ bx + (1e-4 * W)
                grad_b = np.sum(grad_logits, axis=0)
                grads_head = [grad_W, grad_b]

                # Backprop into quantum inputs: dLoss / d(X_q)
                grad_X = grad_logits @ W
                grad_Xq = grad_X[:, :q_feat_dim]

            elif head_type in ("mlp", "hybrid"):
                h = np.maximum(0.0, bx @ W1.T + b1) # ReLU
                logits = h @ W2.T + b2
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                loss = float(np.mean(-np.log(probs[np.arange(bs), by] + 1e-7)))
                epoch_losses.append(loss)
                epoch_correct += np.sum(np.argmax(probs, axis=1) == by)

                grad_logits = (probs.copy())
                grad_logits[np.arange(bs), by] -= 1.0
                grad_logits /= bs

                grad_W2 = grad_logits.T @ h + (1e-4 * W2)
                grad_b2 = np.sum(grad_logits, axis=0)

                grad_h = grad_logits @ W2
                grad_h[h <= 0] = 0.0 # ReLU deriv

                grad_W1 = grad_h.T @ bx + (1e-4 * W1)
                grad_b1 = np.sum(grad_h, axis=0)
                grads_head = [grad_W1, grad_b1, grad_W2, grad_b2]

                grad_X = grad_h @ W1
                grad_Xq = grad_X[:, :q_feat_dim]

            # Adam step for head
            for p_i in range(len(params_head)):
                m_head[p_i] = 0.9 * m_head[p_i] + 0.1 * grads_head[p_i]
                v_head[p_i] = 0.999 * v_head[p_i] + 0.001 * (grads_head[p_i] ** 2)
                m_hat = m_head[p_i] / (1.0 - 0.9 ** adam_step)
                v_hat = v_head[p_i] / (1.0 - 0.999 ** adam_step)
                params_head[p_i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            # Update theta via periodic finite difference (every 4 batches on 2 samples)
            if b_start % (batch_size * 4) == 0:
                sub_indices = batch_idx[:2]
                grad_th = np.zeros_like(theta)
                fd_eps = 0.08
                for th_i in range(n_weights):
                    th_pert = theta.copy()
                    th_pert[th_i] += fd_eps
                    d_sum = 0.0
                    for s_k, s_idx in enumerate(sub_indices):
                        exp_p, bas_p, _ = circuit.execute(train_angles[s_idx], th_pert)
                        dq = (np.concatenate([exp_p, bas_p]) - X_q_train[s_idx]) / fd_eps
                        d_sum += float(np.dot(grad_Xq[s_k], dq))
                    grad_th[th_i] = d_sum / len(sub_indices)

                m_th = 0.9 * m_th + 0.1 * grad_th
                v_th = 0.999 * v_th + 0.001 * (grad_th ** 2)
                m_th_hat = m_th / (1.0 - 0.9 ** adam_step)
                v_th_hat = v_th / (1.0 - 0.999 ** adam_step)
                theta -= vqc_lr * m_th_hat / (np.sqrt(v_th_hat) + 1e-8)

        # Epoch Validation
        X_q_val = []
        for i in range(n_val):
            exp_v, basis_p, _ = circuit.execute(val_angles[i], theta)
            X_q_val.append(np.concatenate([exp_v, basis_p]))
        X_q_val = np.array(X_q_val, dtype=float)

        if head_type == "hybrid":
            X_in_val = np.concatenate([X_q_val, val_12d], axis=1)
        else:
            X_in_val = X_q_val

        if head_type == "linear":
            val_logits = X_in_val @ W.T + b
        else:
            val_h = np.maximum(0.0, X_in_val @ W1.T + b1)
            val_logits = val_h @ W2.T + b2

        exp_vl = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
        val_probs = exp_vl / np.sum(exp_vl, axis=1, keepdims=True)
        val_loss = float(np.mean(-np.log(val_probs[np.arange(n_val), y_val] + 1e-7)))
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy_score(y_val, val_preds) * 100
        val_f1 = f1_score(y_val, val_preds, average="macro")

        tr_loss = float(np.mean(epoch_losses))
        tr_acc = (epoch_correct / n_train) * 100

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        improved = ""
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_checkpoint = {
                "theta": theta.copy(),
                "head_params": [p.copy() for p in params_head],
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "val_f1": best_val_f1,
                "val_loss": best_val_loss,
            }
            improved = " * BEST"

        ep_dt = time.time() - epoch_t0
        print(f"Epoch {epoch+1:2d}/{epochs:2d} ({ep_dt:4.1f}s): Tr Loss={tr_loss:.4f}, Tr Acc={tr_acc:5.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc:5.2f}%, Macro F1={val_f1:.4f}{improved}", flush=True)

    print(f"\n--> Summary for {n_qubits}Q/{n_layers}L ({head_type}): Best Val Acc = {best_val_acc:.2f}%, Best Val Loss = {best_val_loss:.4f}, Best Macro F1 = {best_val_f1:.4f} at Epoch {best_epoch}", flush=True)
    return best_checkpoint, history

if __name__ == "__main__":
    # Test 4Q/2L Linear vs Hybrid
    c_lin, _ = train_vqc_experiment(n_qubits=4, n_layers=2, head_type="linear", n_samples=3000, epochs=8)
    c_hyb, _ = train_vqc_experiment(n_qubits=4, n_layers=2, head_type="hybrid", n_samples=3000, epochs=8)
