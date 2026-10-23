import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from quantum.quantum_circuit import ParameterizedNavigationCircuit
from scratch.test_separability import generate_multi_sector_dataset
from scratch.test_encodings import encode_strategy_c_bounded_nonlinear

def run_architecture_sweep():
    print("Generating standard benchmark dataset for architecture comparison (2500 samples)...", flush=True)
    rng_master = np.random.default_rng(42)
    data, _, _ = generate_multi_sector_dataset(2500, seed=42)

    trajs = sorted(list(set(d["trajectory_id"] for d in data)))
    shuffled_trajs = rng_master.permutation(trajs)
    n_val = int(len(trajs) * 0.2)
    val_set = set(shuffled_trajs[:n_val])

    train_data = [d for d in data if d["trajectory_id"] not in val_set]
    val_data = [d for d in data if d["trajectory_id"] in val_set]

    y_train = np.array([d["teacher_index"] for d in train_data], dtype=int)
    y_val = np.array([d["teacher_index"] for d in val_data], dtype=int)
    n_train = len(train_data)
    n_val = len(val_data)

    models_to_test = [
        {"name": "Model A", "qubits": 4, "layers": 2},
        {"name": "Model B", "qubits": 4, "layers": 3},
        {"name": "Model C", "qubits": 4, "layers": 4},
        {"name": "Model D", "qubits": 6, "layers": 3},
        {"name": "Model E", "qubits": 6, "layers": 4},
    ]

    results = []

    for m_cfg in models_to_test:
        n_q = m_cfg["qubits"]
        n_l = m_cfg["layers"]
        name = m_cfg["name"]
        print(f"\n=======================================================", flush=True)
        print(f"Evaluating {name}: {n_q} qubits, {n_l} variational layers", flush=True)
        print(f"=======================================================", flush=True)

        circuit = ParameterizedNavigationCircuit(n_qubits=n_q, n_layers=n_l)
        n_weights = circuit.get_num_weights()
        q_feat_dim = n_q + (1 << n_q)

        # Measure single-sample inference time
        test_angles = np.zeros(n_q)
        test_theta = np.zeros(n_weights)
        t_inf_start = time.time()
        for _ in range(100):
            circuit.execute(test_angles, test_theta)
        inf_time_ms = (time.time() - t_inf_start) / 100 * 1000

        # Encode input angles
        train_angles = [encode_strategy_c_bounded_nonlinear(d["features"], n_q) for d in train_data]
        val_angles = [encode_strategy_c_bounded_nonlinear(d["features"], n_q) for d in val_data]

        rng = np.random.default_rng(42)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_weights)

        # Classical readout head (linear 9-class projection)
        scale = np.sqrt(2.0 / (q_feat_dim + 9))
        W = rng.normal(0.0, scale, size=(9, q_feat_dim))
        b = np.zeros(9)

        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_b, v_b = np.zeros_like(b), np.zeros_like(b)
        m_th, v_th = np.zeros_like(theta), np.zeros_like(theta)
        adam_step = 0
        batch_size = 32
        epochs = 6
        lr = 0.015
        vqc_lr = 0.008

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        train_start = time.time()

        for epoch in range(epochs):
            # Quantum execution on train
            X_q_train = []
            for i in range(n_train):
                exp_v, bas_p, _ = circuit.execute(train_angles[i], theta)
                X_q_train.append(np.concatenate([exp_v, bas_p]))
            X_q_train = np.array(X_q_train, dtype=float)

            perm = rng.permutation(n_train)
            for b_start in range(0, n_train, batch_size):
                adam_step += 1
                b_end = min(b_start + batch_size, n_train)
                batch_idx = perm[b_start:b_end]
                bx = X_q_train[batch_idx]
                by = y_train[batch_idx]
                bs = len(batch_idx)

                logits = bx @ W.T + b
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                grad_logits = probs.copy()
                grad_logits[np.arange(bs), by] -= 1.0
                grad_logits /= bs

                grad_W = grad_logits.T @ bx + (1e-4 * W)
                grad_b = np.sum(grad_logits, axis=0)

                m_W = 0.9 * m_W + 0.1 * grad_W
                v_W = 0.999 * v_W + 0.001 * (grad_W ** 2)
                W -= lr * (m_W / (1.0 - 0.9 ** adam_step)) / (np.sqrt(v_W / (1.0 - 0.999 ** adam_step)) + 1e-8)

                m_b = 0.9 * m_b + 0.1 * grad_b
                v_b = 0.999 * v_b + 0.001 * (grad_b ** 2)
                b -= lr * (m_b / (1.0 - 0.9 ** adam_step)) / (np.sqrt(v_b / (1.0 - 0.999 ** adam_step)) + 1e-8)

                # Periodic theta update
                if b_start % (batch_size * 4) == 0:
                    grad_Xq = grad_logits @ W
                    sub = batch_idx[:2]
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

            # Val evaluation
            X_q_val = []
            for i in range(n_val):
                exp_v, bas_p, _ = circuit.execute(val_angles[i], theta)
                X_q_val.append(np.concatenate([exp_v, bas_p]))
            X_q_val = np.array(X_q_val, dtype=float)

            val_logits = X_q_val @ W.T + b
            exp_vl = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
            val_probs = exp_vl / np.sum(exp_vl, axis=1, keepdims=True)
            val_loss = float(np.mean(-np.log(val_probs[np.arange(n_val), y_val] + 1e-7)))
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = accuracy_score(y_val, val_preds) * 100
            val_f1 = f1_score(y_val, val_preds, average="macro")

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_f1 = val_f1

        train_time = time.time() - train_start
        total_params = n_weights + (9 * q_feat_dim) + 9

        print(f"Result for {name}:")
        print(f"  Qubits: {n_q}, Layers: {n_l}")
        print(f"  VQC Weights: {n_weights}, Total Params: {total_params}")
        print(f"  Inference Time: {inf_time_ms:.2f} ms/sample")
        print(f"  Training Time: {train_time:.1f} s")
        print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Macro F1: {best_val_f1:.4f}")

        results.append({
            "model_name": name,
            "n_qubits": n_q,
            "n_layers": n_l,
            "n_vqc_weights": n_weights,
            "total_params": total_params,
            "inference_time_ms": round(inf_time_ms, 2),
            "training_time_s": round(train_time, 1),
            "val_accuracy": round(best_val_acc, 2),
            "val_loss": round(best_val_loss, 4),
            "macro_f1": round(best_val_f1, 4),
        })

    with open("scratch/architecture_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved scratch/architecture_comparison.json", flush=True)

if __name__ == "__main__":
    run_architecture_sweep()
