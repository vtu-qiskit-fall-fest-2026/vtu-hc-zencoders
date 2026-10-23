import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from quantum.quantum_circuit import ParameterizedNavigationCircuit
from scratch.test_separability import generate_multi_sector_dataset
from scratch.test_encodings import encode_strategy_c_bounded_nonlinear

def sweep_lr_and_seeds():
    lrs = [0.003, 0.008, 0.015, 0.025]
    seeds = [42, 123, 456]
    n_samples = 1500
    epochs = 5
    batch_size = 32

    lr_results = {}

    for lr in lrs:
        seed_accs = []
        seed_f1s = []
        seed_losses = []
        print(f"\n--- Testing Learning Rate = {lr} across seeds {seeds} ---", flush=True)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            data, _, _ = generate_multi_sector_dataset(n_samples, seed=seed)

            trajs = sorted(list(set(d["trajectory_id"] for d in data)))
            shuffled = rng.permutation(trajs)
            n_val = int(len(trajs) * 0.2)
            val_set = set(shuffled[:n_val])

            train_data = [d for d in data if d["trajectory_id"] not in val_set]
            val_data = [d for d in data if d["trajectory_id"] in val_set]

            y_train = np.array([d["teacher_index"] for d in train_data], dtype=int)
            y_val = np.array([d["teacher_index"] for d in val_data], dtype=int)
            n_train = len(train_data)
            n_val = len(val_data)

            # Model A (4Q, 2L)
            circuit = ParameterizedNavigationCircuit(n_qubits=4, n_layers=2)
            n_weights = circuit.get_num_weights()
            theta = rng.uniform(0.0, 2.0 * np.pi, size=n_weights)
            train_angles = [encode_strategy_c_bounded_nonlinear(d["features"], 4) for d in train_data]
            val_angles = [encode_strategy_c_bounded_nonlinear(d["features"], 4) for d in val_data]

            q_feat_dim = 4 + 16
            scale = np.sqrt(2.0 / (q_feat_dim + 9))
            W = rng.normal(0.0, scale, size=(9, q_feat_dim))
            b = np.zeros(9)

            m_W, v_W = np.zeros_like(W), np.zeros_like(W)
            m_b, v_b = np.zeros_like(b), np.zeros_like(b)
            m_th, v_th = np.zeros_like(theta), np.zeros_like(theta)
            adam_step = 0
            vqc_lr = lr * 0.6

            best_acc = 0.0
            best_f1 = 0.0
            best_loss = float("inf")

            for ep in range(epochs):
                # Quantum features on train
                X_q_train = []
                for i in range(n_train):
                    e_v, b_p, _ = circuit.execute(train_angles[i], theta)
                    X_q_train.append(np.concatenate([e_v, b_p]))
                X_q_train = np.array(X_q_train, dtype=float)

                perm = rng.permutation(n_train)
                for b_start in range(0, n_train, batch_size):
                    adam_step += 1
                    b_end = min(b_start + batch_size, n_train)
                    b_idx = perm[b_start:b_end]
                    bx = X_q_train[b_idx]
                    by = y_train[b_idx]
                    bs = len(b_idx)

                    logits = bx @ W.T + b
                    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                    grad_logits = probs.copy()
                    grad_logits[np.arange(bs), by] -= 1.0
                    grad_logits /= bs

                    grad_W = grad_logits.T @ bx + (1e-4 * W)
                    grad_b = np.sum(grad_logits, axis=0)

                    # AdamW
                    m_W = 0.9 * m_W + 0.1 * grad_W
                    v_W = 0.999 * v_W + 0.001 * (grad_W ** 2)
                    W -= lr * (m_W / (1.0 - 0.9 ** adam_step)) / (np.sqrt(v_W / (1.0 - 0.999 ** adam_step)) + 1e-8)

                    m_b = 0.9 * m_b + 0.1 * grad_b
                    v_b = 0.999 * v_b + 0.001 * (grad_b ** 2)
                    b -= lr * (m_b / (1.0 - 0.9 ** adam_step)) / (np.sqrt(v_b / (1.0 - 0.999 ** adam_step)) + 1e-8)

                    if b_start % (batch_size * 4) == 0:
                        grad_Xq = grad_logits @ W
                        sub = b_idx[:2]
                        grad_th = np.zeros_like(theta)
                        fd_eps = 0.08
                        for th_i in range(n_weights):
                            th_p = theta.copy()
                            th_p[th_i] += fd_eps
                            d_sum = 0.0
                            for s_k, s_idx in enumerate(sub):
                                e_p, b_p, _ = circuit.execute(train_angles[s_idx], th_p)
                                dq = (np.concatenate([e_p, b_p]) - X_q_train[s_idx]) / fd_eps
                                d_sum += float(np.dot(grad_Xq[s_k], dq))
                            grad_th[th_i] = d_sum / len(sub)
                        m_th = 0.9 * m_th + 0.1 * grad_th
                        v_th = 0.999 * v_th + 0.001 * (grad_th ** 2)
                        theta -= vqc_lr * (m_th / (1.0 - 0.9 ** adam_step)) / (np.sqrt(v_th / (1.0 - 0.999 ** adam_step)) + 1e-8)

                # Validation
                X_q_val = []
                for i in range(n_val):
                    e_v, b_p, _ = circuit.execute(val_angles[i], theta)
                    X_q_val.append(np.concatenate([e_v, b_p]))
                X_q_val = np.array(X_q_val, dtype=float)

                val_logits = X_q_val @ W.T + b
                exp_vl = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
                val_probs = exp_vl / np.sum(exp_vl, axis=1, keepdims=True)
                loss = float(np.mean(-np.log(val_probs[np.arange(n_val), y_val] + 1e-7)))
                preds = np.argmax(val_probs, axis=1)
                acc = accuracy_score(y_val, preds) * 100
                f1 = f1_score(y_val, preds, average="macro")

                if acc > best_acc or (acc == best_acc and loss < best_loss):
                    best_acc = acc
                    best_f1 = f1
                    best_loss = loss

            print(f"  LR {lr}, Seed {seed}: Val Acc={best_acc:.2f}%, Val Loss={best_loss:.4f}, Macro F1={best_f1:.4f}", flush=True)
            seed_accs.append(best_acc)
            seed_f1s.append(best_f1)
            seed_losses.append(best_loss)

        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs))
        mean_f1 = float(np.mean(seed_f1s))
        std_f1 = float(np.std(seed_f1s))
        mean_loss = float(np.mean(seed_losses))
        print(f"-> Summary for LR {lr}: Val Acc = {mean_acc:.2f}% +/- {std_acc:.2f}%, Macro F1 = {mean_f1:.4f} +/- {std_f1:.4f}", flush=True)

        lr_results[str(lr)] = {
            "mean_val_accuracy": round(mean_acc, 2),
            "std_val_accuracy": round(std_acc, 2),
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "mean_val_loss": round(mean_loss, 4),
            "seeds": seeds,
            "seed_accuracies": [round(a, 2) for a in seed_accs],
        }

    with open("scratch/lr_sweep_results.json", "w") as f:
        json.dump(lr_results, f, indent=2)
    print("Saved scratch/lr_sweep_results.json", flush=True)

if __name__ == "__main__":
    sweep_lr_and_seeds()
