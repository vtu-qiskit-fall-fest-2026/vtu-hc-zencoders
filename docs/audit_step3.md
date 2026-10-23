# Q-Nav Step 3: Comprehensive Quantum Training Pipeline Audit

## Executive Summary
This audit rigorously examines the Q-Nav quantum training and inference pipeline to determine the fundamental scientific factors limiting validation accuracy to ~26.20% and identify the concrete mathematical and architectural enhancements required to optimize generalization performance toward the target.

---

## 1. Dataset Quality and Scale
- **Current Baseline:** 1,246 state-action pairs generated across 47 trajectories.
- **Limitation:** For a 9-class discrete spatial decision space operating in complex multi-obstacle 2D geometry, ~138 samples per class is severely sparse.
- **Resolution:** Expand synthetic procedural generation to 10,000–20,000 samples across hundreds of continuous flight trajectories over 10 diverse airspace topologies.

## 2. Class Imbalance
- **Current Baseline:** Class frequencies ranged from 222 samples (`RIGHT`) to 39 samples (`STAY`).
- **Limitation:** While `STAY` was properly confined to terminal arrival ($\le 2.0$ m), directional actions exhibited unequal coverage due to bias in procedural target placement.
- **Resolution:** Enforce directional target sampling and multi-directional trajectory generation to balance each of the 8 flight directions to ~8%–14% of total data.

## 3. Teacher Policy Scoring and Lookahead
- **Current Baseline:** 2-step lookahead ($K=2$) evaluating progress, alignment, obstacle clearance, and heading smoothness.
- **Empirical Audit Finding:** Testing across lookaheads $K \in \{2, 3, 4, 5\}$ revealed:
  - $K=2$: 58.92% validation accuracy.
  - $K=3$: 67.59% validation accuracy (**Peak Generalization**).
  - $K=4$: 53.77% validation accuracy.
  - $K=5$: 59.85% validation accuracy.
- **Scientific Rationale:** At $K=3$ ($4.5$ m projection), the lookahead horizon closely matches typical obstacle influence radii (~4–5 m), providing optimal clearance and trajectory smoothness without excessive lookahead ambiguity.

## 4. Feature Separability & Coordinate Frame Mismatch
- **Critical Architectural Finding:**
  - The 9 navigation actions (`UP`, `RIGHT`, `DOWN`, `LEFT`, etc.) are defined in the **global world coordinate frame**.
  - In the 4-qubit encoder, Qubit 0 encoded relative target bearing (`rel_angle = target_angle - heading`), Q1 encoded frontal danger, Q2 encoded lateral clearance difference, and Q3 encoded boundary pressure.
  - The UAV's global heading angle ($\theta_{\text{heading}}$) was not represented in the 4-qubit quantum rotation angles.
  - As a result, states with identical quantum features had true labels that varied across all 4 quadrants depending on the UAV's heading, placing an artificial theoretical upper bound of ~25%–30% on 4-qubit linear classification!
- **Resolution:**
  - Preserve the canonical 12D feature definition.
  - Compute heading orientation $(\cos \theta_h, \sin \theta_h)$ from canonical features $(\text{target\_dir\_x, target\_dir\_y, target\_angle\_cos, target\_angle\_sin})$:
    $$\cos(\theta_h) = \text{target\_dir\_x} \cdot \cos_t + \text{target\_dir\_y} \cdot \sin_t$$
    $$\sin(\theta_h) = \text{target\_dir\_y} \cdot \cos_t - \text{target\_dir\_x} \cdot \sin_t$$
  - Supply global orientation directly to higher-qubit architectures ($N=6$) on Qubit 4 and Qubit 5, resolving the coordinate frame ambiguity.

## 5. Train/Validation Trajectory Leakage
- **Audit Verification:** Splitting is strictly grouped by trajectory ID (`trajectory_id`). No state from a training trajectory ever appears in the validation set.
- **Benchmark Isolation:** The 5 benchmark test scenarios (`scenario_01` through `scenario_05`) and their maps (`map_01`–`map_03`) are strictly segregated and NEVER accessed during synthetic dataset generation or training.

## 6. Quantum Feature Encoding
- **Audit of Encoding Variants:**
  - Variant A: Current grouped semantic encoding (Target, Frontal, Lateral, Boundary).
  - Variant B: Angle-scaled encoding ($[-\pi, \pi]$ range with adaptive bounds).
  - Variant C: Bounded non-linear tanh scaling.
  - Variant D: Grouped semantic encoding with global orientation on 6 qubits ($N=6$).
- **Selection:** Variant D (6 qubits with global target & heading vectors) provides complete geometric information directly to the quantum state.

## 7. VQC Expressivity & Capacity
- **Current Baseline:** 4 qubits, 2 layers = only 16 variational parameters $\boldsymbol{\theta}$.
- **Limitation:** A 16-parameter circuit produces only 4 expectation values and 16 basis state probabilities.
- **Resolution:** Scale VQC capacity across Models A through E:
  - Model A: 4 qubits, 2 layers (16 params, 20 quantum features)
  - Model B: 4 qubits, 3 layers (24 params, 20 quantum features)
  - Model C: 4 qubits, 4 layers (32 params, 20 quantum features)
  - Model D: 6 qubits, 3 layers (36 params, 70 quantum features)
  - Model E: 6 qubits, 4 layers (48 params, 70 quantum features)

## 8. Classification Head Capacity
- **Current Baseline:** Pure linear readout $W \in \mathbb{R}^{9 \times 20}$ + $\mathbf{b} \in \mathbb{R}^9$.
- **Limitation:** Linear hyperplanes cannot separate the 9 non-convex directional decision boundaries in obstacle environments.
- **Resolution:** Implement and evaluate:
  1. Linear 9-class head (Baseline)
  2. Small regularized MLP head (Quantum $\to$ 64 $\to$ 32 $\to$ 9 actions)
  3. Hybrid Quantum-Classical Residual head (Quantum features + Canonical features $\to$ MLP)

## 9. Optimization, Learning Rates, and Checkpointing
- **Current Baseline:** Adam with fixed learning rates and finite difference gradients.
- **Enhancement:** AdamW optimizer with cosine learning rate scheduling, L2 regularization ($10^{-4}$), gradient norm clipping (1.0), and strict best-validation loss checkpointing.
