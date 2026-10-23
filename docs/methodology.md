# Q-Nav: Quantum-Assisted UAV Navigation Methodology

## 1. Problem Statement & Motivation

Autonomous Unmanned Aerial Vehicle (UAV) navigation in low-altitude, obstacle-dense airspaces requires real-time path planning, reactive collision avoidance, and trajectory optimization. Classical methods such as Artificial Potential Fields (APF), $A^*$ graph search, and Rapidly-exploring Random Trees (RRT) face notable challenges:
- **Local Minima Traps:** APFs frequently suffer from force equilibrium stagnation when symmetric obstacles obstruct the direct line-of-sight to the goal.
- **Oscillations in Chokepoints:** Narrow corridors induce rapid alternating repulsive forces, causing cyclic limit cycles.
- **Computational Scaling:** Graph search discretizations suffer from dimensional scaling bottlenecks in dynamic or high-resolution spatial maps.
- **Multi-Objective Tradeoffs:** Balancing goal-reaching velocity with boundary clearance requires intricate heuristics.

**Quantum-Assisted UAV Navigation (Q-Nav)** investigates whether parameterized quantum circuits (Variational Quantum Circuits, or VQCs) executed on Noisy Intermediate-Scale Quantum (NISQ) devices or high-performance quantum statevector simulators can assist autonomous decision-making. By leveraging quantum superposition, multi-qubit entanglement, and quantum state interference, Q-Nav evaluates candidate navigation directions under identical, controlled environmental conditions.

> [!IMPORTANT]
> **Scientific Integrity & Advantage Disclaimer:**  
> Quantum-assisted results are produced using a Qiskit simulator. They do not imply quantum computational advantage over the classical baseline.  
> The current benchmark does not demonstrate quantum advantage. The classical APF baseline currently performs better in the tested scenarios.

---

## 2. System Architecture

The Q-Nav platform consists of a modular, closed-loop simulation and benchmarking architecture:

```
run.py / CLI
    │
    ├──> config.py (SimulationConfig, ClassicalConfig, QuantumConfig, NavigationConfig)
    │
    ├──> Environment Initialization (UAVEnvironment, Drone, ObstacleField, Target)
    │
    ├──> Feature Extraction (NavigationFeatureExtractor: 12 normalized telemetry features)
    │
    ├──> DECISION ENGINE
    │     ├── [Classical Baseline]: APF Forces + Tangential Force + Stagnation & Bypass -> Action Softmax
    │     └── [Quantum Navigator]: 12D Angle Encoding -> Qiskit VQC Circuit -> Pauli-Z Expectation Readout
    │
    ├──> Shared Safety Arbitration (SafetyLayer lookahead collision check, swept-sphere test, evasive override)
    │
    ├──> Physical Actuation (Drone kinematics, swept collision check, goal check)
    │
    └──> Evaluation & Visualization (Multi-seed statistical metrics, JSON outputs, PNG figures, Streamlit Dashboard)
```

The core components operate in continuous feedback:
1. **Simulation Environment:** Simulates 2D UAV continuous kinematics, static circular and rectangular obstacles, map boundaries, and destination waypoints.
2. **Feature Extractor:** Extracts 12 normalized spatial, directional, and kinematic telemetry features.
3. **Classical APF Navigator:** Computes gradient virtual forces ($F_{att}, F_{rep}, F_{tan}$) with stagnation detection and dual-direction bypass recovery.
4. **Quantum VQC Navigator:** Encodes spatial features into single-qubit Bloch rotations, propagates quantum states through circular CNOT entangling gates, and evaluates Pauli-Z expectations and basis state probability distributions.
5. **Action Selector & Safety Layer:** Arbitrates candidate actions, validates lookahead collision-free clearance, and activates emergency evasive overrides when necessary.
6. **Evaluation Engine:** Logs episode telemetry, measures standardized quantitative performance metrics across multiple seeds, and renders publication-quality figures.

---

## 3. Classical Navigation Methodology (Enhanced Artificial Potential Fields)

The classical baseline implements an enhanced Artificial Potential Field formulation with tangential circulation, stagnation detection, oscillation detection, and bypass recovery:

### 3.1 Attractive Potential
The attractive potential pulls the UAV toward the target waypoint $p_{target} = (x_t, y_t)$:
$$U_{att}(p) = \frac{1}{2} k_{att} \| p_{target} - p \|^2$$
$$F_{att}(p) = -\nabla U_{att}(p) = k_{att} (p_{target} - p)$$

### 3.2 Repulsive Potential
For each obstacle $i$ with surface clearance $d_i(p) \le d_0$ (where $d_0$ is the influence radius):
$$U_{rep, i}(p) = \begin{cases} 
\frac{1}{2} k_{rep} \left( \frac{1}{d_i(p)} - \frac{1}{d_0} \right)^2 & \text{if } d_i(p) \le d_0 \\ 
0 & \text{if } d_i(p) > d_0 
\end{cases}$$

$$F_{rep, i}(p) = k_{rep} \left( \frac{1}{d_i(p)} - \frac{1}{d_0} \right) \frac{1}{d_i(p)^2} \hat{n}_{i}$$
where $\hat{n}_i = \frac{p - p_{obs, i}}{\|p - p_{obs, i}\|}$ is the outward normal vector pointing away from the obstacle.

### 3.3 Tangential Obstacle Circulation Force
To prevent head-on oscillation directly in front of convex or symmetric obstacles, a tangential circulation force $F_{tan}$ is added perpendicular to the outward normal:
$$\hat{t}_i = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \hat{n}_i \quad \text{or} \quad \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \hat{n}_i$$
$$F_{tan, i}(p) = k_{tan} \left( \frac{1}{d_i(p)} - \frac{1}{d_0} \right) \frac{1}{d_i(p)^2} \hat{t}_i$$
The orientation of $\hat{t}_i$ is selected to align with the drone's velocity and goal heading, gently guiding the UAV around the obstacle boundary.

### 3.4 Boundary Repulsive Force
When approaching the airspace boundaries $[0, W] \times [0, H]$ within buffer margin $d_{bound}$:
$$F_{boundary}(p) = \sum_{k \in \{x_{min}, x_{max}, y_{min}, y_{max}\}} k_{rep} \left( \frac{1}{d_k} - \frac{1}{d_{bound}} \right) \frac{1}{d_k^2} \hat{n}_k$$

### 3.5 Stagnation Detection
A sliding window of past $W$ positions ($\text{stagnation\_window} = 15$) is monitored. Stagnation is triggered when:
$$\max_{t_1, t_2 \in [t-W, t]} \| p(t_1) - p(t_2) \| < d_{stag} \quad (\text{threshold} = 3.0)$$
and goal progress over the window satisfies:
$$\| p(t-W) - p_{target} \| - \| p(t) - p_{target} \| < \Delta d_{progress} \quad (\text{threshold} = 1.0)$$

### 3.6 Oscillation Detection
The last 6 actions are tracked for alternating two-step cyclic patterns (e.g., $[a, b, a, b]$):
$$a(t) = a(t-2) \quad \text{and} \quad a(t-1) = a(t-3), \quad a(t) \neq a(t-1)$$
When oscillation is detected, the controller switches from raw APF scoring to a stable bypass direction.

### 3.7 Local-Minimum Bypass Recovery
When stagnation or severe equilibrium ($\|F_{total}\| < \epsilon$) is detected:
1. Two orthogonal bypass vectors are generated relative to the target line-of-sight:
   $$v_{left} = R(+90^\circ) \hat{F}_{att}, \quad v_{right} = R(-90^\circ) \hat{F}_{att}$$
2. The obstacle clearance along both bypass rays is evaluated up to $d_{probe} = 25.0$.
3. The safer bypass direction (higher obstacle clearance) is selected.
4. The UAV enters temporary bypass recovery mode for $K = 12$ steps.
5. Normal APF control resumes once forward goal progress is restored.

---

## 4. Quantum Feature Encoding & Circuit Design

The quantum navigation pipeline employs a hybrid quantum-classical architecture utilizing **Qiskit 2.x**:

### 4.1 12-Dimensional State Feature Vector
The continuous environment state is encoded into a standardized 12-dimensional vector $x \in [0, 1]^{12}$:
1. **$x_0$ (`norm_target_dist`):** Normalized Euclidean distance to target, $\min(\|p_{target} - p\| / d_{max}, 1.0)$.
2. **$x_1$ (`target_dir_x`):** Unit direction vector X to target, $\Delta x / \| \Delta p \|$.
3. **$x_2$ (`target_dir_y`):** Unit direction vector Y to target, $\Delta y / \| \Delta p \|$.
4. **$x_3$ (`target_angle_sin`):** Sine of relative target bearing angle $\sin(\phi_{target} - \theta_{heading})$.
5. **$x_4$ (`target_angle_cos`):** Cosine of relative target bearing angle $\cos(\phi_{target} - \theta_{heading})$.
6. **$x_5$ (`norm_nearest_obs_dist`):** Normalized clearance to nearest obstacle, $\min(d_{obs} / d_{sensor}, 1.0)$.
7. **$x_6$ (`front_obstacle_dist`):** Normalized clearance along forward sensing cone ($\pm 30^\circ$).
8. **$x_7$ (`left_obstacle_dist`):** Normalized clearance along left flank ($+45^\circ$).
9. **$x_8$ (`right_obstacle_dist`):** Normalized clearance along right flank ($-45^\circ$).
10. **$x_9$ (`obstacle_density`):** Local obstacle area fraction within sensor radius.
11. **$x_{10}$ (`norm_speed`):** Normalized UAV kinematic speed, $\|v\| / v_{max}$.
12. **$x_{11}$ (`boundary_proximity`):** Distance to nearest airspace boundary normalized to boundary margin.

### 4.2 Feature-to-Qubit Mappings ($12 \to N$)
To map 12 continuous features into $N \in [4, 8]$ qubits, four strategies are supported:
- **`grouped` (Default):** Domain-informed semantic pooling into $N$ functional channels:
  - $q_0$ (Target Guidance): Combines target distance, bearing sine/cosine: $\theta_0 = (0.5 x_3 + 0.3 x_0 + 0.2 x_1) \pi$
  - $q_1$ (Obstacle Repulsion): Combines nearest, front, left, right clearances: $\theta_1 = ((1 - x_5) \cdot 1.5 - 0.5) \pi$
  - $q_2$ (Kinematic Dynamics): Combines current speed, target direction Y: $\theta_2 = (0.7 x_2 + 0.5 (x_{10} - 0.5)) \pi$
  - $q_3$ (Spatial Density & Margin): Combines boundary proximity and obstacle density: $\theta_3 = (2.0 (1 - x_{11}) + x_9 - 1.0) \pi$
  - Remaining qubits $q_4 \dots q_{N-1}$: Individual sector obstacle features ($x_6, x_7, x_8$).
- **`pooling`:** Uniform average pooling across $\lfloor 12 / N \rfloor$ consecutive feature slices.
- **`repeated`:** Cyclic interleaved assignment of features modulo $N$.
- **`linear`:** Orthogonal linear projection matrix $W_{proj} \in \mathbb{R}^{N \times 12}$.

### 4.3 Quantum Circuit Architecture (VQC)
Each qubit $i \in \{0, \dots, N-1\}$ is initialized in state $|0\rangle$:
1. **Feature Encoding Layer:**
   $$|\psi_{in}\rangle = \bigotimes_{i=0}^{N-1} R_z\left(\frac{\theta_i}{2}\right) R_y(\theta_i) |0\rangle$$
2. **Parameterized Variational Layers ($L = 2$):**
   $$U(\mathbf{w}) = \prod_{l=1}^{L} \left[ \left( \bigotimes_{i=0}^{N-1} R_z(w_{l, i, 1}) R_y(w_{l, i, 0}) \right) \cdot U_{ent} \right]$$
   where the circular CNOT entanglement operator is:
   $$U_{ent} = \text{CNOT}(N-1, 0) \cdot \prod_{i=0}^{N-2} \text{CNOT}(i, i+1)$$
3. **Observables & Readout:**
   The circuit evaluates Pauli-Z expectation values on each qubit:
   $$\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle \in [-1, 1], \quad i \in \{0, \dots, N-1\}$$
   along with computational basis state probabilities $P(|b\rangle) = |\langle b | \psi \rangle|^2$.

### 4.4 Classical Action-Logit Projection
The expectation vector $z = [\langle Z_0 \rangle, \dots, \langle Z_{N-1} \rangle]^T$ is linearly projected onto the 9 candidate actions ($A = \{\text{UP}, \text{UP\_RIGHT}, \dots, \text{STAY}\}$):
$$z_{logits} = W_{readout} \cdot z + b_{readout}$$
$$P(a_k) = \frac{\exp(z_k / \tau)}{\sum_{j=1}^{9} \exp(z_j / \tau)}$$

### 4.5 Hybrid Quantum-Classical Decision Fusion
To harness quantum action preferences while ensuring deterministic collision avoidance and goal guidance, the hybrid controller fuses the classical APF decision with the quantum model's probability distribution among safe actions:
1. **Lookahead Safe Set ($A_{safe}$):** Evaluated by the Safety Layer; eliminates moves heading into obstacles or boundary walls:
   $$A_{safe} = \{ a_i \in A \mid \text{is\_safe}(p, v_i) = \text{True} \}$$
2. **Fused Scoring Formula:**
   $$S(a_i) = 0.7 \cdot P_{quantum}(a_i) + 0.3 \cdot \mathbb{I}(a_i = a_{classical}), \quad \forall a_i \in A_{safe}$$
3. **Action Selection:**
   $$a^* = \arg\max_{a_i \in A_{safe}} S(a_i)$$
   If $A_{safe} = \emptyset$, emergency `STAY` ($a=8$) is triggered.

> [!NOTE]
> The action projection and logit layer is computed classically. The circuit executes genuine quantum state evolution, but classical readout projection and hybrid fusion are used to produce final actuation commands.

---

## 5. Quantum Model Training & Fine-Tuning

### 5.1 Pre-Training via Synthetic Teacher Policy
To provide a non-random baseline, the VQC model includes a supervised pre-training pipeline (`quantum/training.py`):
1. **Synthetic Training Airspaces:** State-action pairs are generated across 500+ diverse synthetic obstacle layouts with random start/goal pairs.
2. **Safe Lookahead Teacher Policy:** The teacher action is determined by evaluating:
   $$a^* = \arg\max_{a \in A_{safe}} \left[ \cos(\theta_a, \theta_{target}) + 0.5 \cdot \frac{d_{obs}(p + v_a \Delta t)}{d_{sensor}} \right]$$
3. **Cross-Entropy Loss:**
   $$\mathcal{L}_{CE} = - \sum_{k=1}^{9} y_k \log P(a_k)$$

### 5.2 Fine-Tuning on Benchmark Scenarios
When `--train-quantum` is invoked:
1. Classical APF teacher trajectories are collected across the benchmark scenarios.
2. The quantum variational circuit weights $\mathbf{w}$ and readout projection matrices $(W_{readout}, b_{readout})$ are fine-tuned using the Adam optimizer to minimize imitation cross-entropy loss.
3. Optimized parameters are checkpointed to `outputs/metrics/quantum_weights.npz`.

---

## 6. Shared Navigation Control & Safety Mechanism

To guarantee safe operation in physical reality, all decisions from both Classical and Quantum navigators pass through an identical **Safety Layer**:

1. **Lookahead Projection:** For a candidate action vector $v_a$, future positions are evaluated over $K = 2$ lookahead steps:
   $$p_{proj}(k) = p(t) + k \cdot v_a \cdot \Delta t$$
2. **Boundary Containment:** Rejects actions where $p_{proj}$ breaches map limits $[0, W] \times [0, H]$ with buffer $d_{margin}$.
3. **Swept-Sphere Collision Detection:** Evaluates capsule intersection between $p(t)$ and $p(t+1)$ against all obstacles, preventing tunneling at high velocity.
4. **Safety Arbitration:**
   - If the top-ranked action violates safety clearance, the next highest-scoring candidate is tested.
   - If all directional candidates are obstructed, an emergency evasive action moving strictly away from the nearest obstacle is executed.
5. **Telemetry Disambiguation:**
   - `collision_count`: Actual physical geometric intersections between UAV and obstacles (target: 0).
   - `unsafe_action_attempts`: Number of proposed raw actions that failed lookahead safety verification.
   - `safety_interventions`: Number of times the safety layer overrode the model's preferred action. A safety intervention is **not** a physical collision.

---

## 7. Standardized Navigation Metrics

Q-Nav evaluates 21 standardized quantitative metrics:

| Metric | Definition | Unit / Range |
| :--- | :--- | :---: |
| **`success_rate`** | Proportion of episodes reaching target within max steps | $[0.0, 1.0]$ |
| **`path_length`** | Cumulative Euclidean distance traversed: $\sum_{t=1}^T \|p(t) - p(t-1)\|$ | Distance units |
| **`steps`** | Total simulation steps executed in episode | Integer |
| **`final_distance`** | Distance from final UAV position to target center | Distance units |
| **`collision_count`** | Count of actual physical collisions with obstacles | Integer $(\ge 0)$ |
| **`collision_rate`** | Collisions normalized by step count: $\text{collisions} / T$ | Step$^{-1}$ |
| **`safety_interventions`**| Count of Safety Layer action overrides | Integer $(\ge 0)$ |
| **`unsafe_action_attempts`**| Count of proposed actions failing lookahead safety check | Integer $(\ge 0)$ |
| **`boundary_violations`** | Count of attempts to cross airspace boundary | Integer $(\ge 0)$ |
| **`min_obstacle_clearance`**| Minimum distance between UAV surface and any obstacle | Distance units |
| **`min_boundary_clearance`**| Minimum distance from UAV center to airspace border | Distance units |
| **`navigation_efficiency`** | Optimal straight-line distance divided by actual path length: $d_{opt} / L$ | $[0.0, 1.0]$ |
| **`progress_ratio`** | Goal distance reduction divided by total path traveled | $[-1.0, 1.0]$ |
| **`heading_smoothness`** | Cumulative absolute heading angle change: $\sum |\Delta \theta_t|$ | Radians |
| **`action_switch_count`** | Number of times consecutive discrete actions changed | Integer |
| **`oscillation_count`** | Count of detected 2-step alternating action reversals | Integer |
| **`mean_decision_time`** | Wall-clock time spent computing action candidates | Seconds |
| **`quantum_circuit_execution_time`** | Wall-clock time executing Qiskit quantum circuit | Seconds |
| **`total_episode_time`** | Total elapsed wall-clock duration of episode | Seconds |
| **`reward`** | Task reward: $+100$ (goal), $-100$ (collision), $-0.1 \times \text{step} - 0.05 \times d$ | Score |
| **`energy_proxy`** | Proxy energy consumption: $E = \sum (\|v_t\| \cdot \Delta t + 0.3 |\Delta \theta_t|)$ | Energy units |

---

## 8. Multi-Seed Benchmarking Protocol

To prevent selective reporting:
- **Repeated Randomized Trials:** Benchmarks execute across 5 scenarios $\times$ 10 random seeds $\times$ 2 models.
- **Identical Environment Instances:** Both models receive identical maps, obstacles, start/target coordinates, speeds, step limits, collision radii, and random seeds.
- **Fresh Environments:** Each episode runs in an independently initialized environment instance.
- **Statistical Aggregation:** Metrics report `mean`, `std`, `median`, `min`, `max`, and `ci_95` (95% studentized confidence intervals).

---

## 9. Observed Experimental Results

Benchmarking across 5 standardized scenarios over multiple seeds yields the following empirical results:

| Metric | Classical Baseline (APF) | Quantum-Assisted (Hybrid VQC) | Comparison / Notes |
| :--- | :---: | :---: | :--- |
| **Success Rate** | **40.0%** | **40.0%** | Hybrid architecture achieved parity (up from 0% – 20%) |
| **Mean Path Length** | **144.30** | **144.30** | Equal trajectory distance under fused guidance |
| **Mean Steps** | **96.2** | **96.2** | Equal step efficiency |
| **Mean Collisions** | **0.00** | **0.00** | Shared safety layer guarantees zero crashes |
| **Collision Rate** | **0.000** | **0.000** | Identical safety enforcement |
| **Navigation Efficiency**| **47.8%** | 42.7% | Classical has slight edge in path efficiency |
| **Safety Interventions**| **0.0** | **0.0** | Proactive lookahead filtering prevents unsafe execution |
| **Mean Decision Time** | **0.0009 s** | 0.0045 s | Qiskit Statevector simulation latency |
| **Mean Episode Time** | **0.8643 s** | 1.1925 s | Classical is ~1.38x faster on CPU |

> [!NOTE]
> **Scientific Interpretation:**  
> The hybrid architecture successfully closes the performance gap between purely classical APF and standalone quantum navigation by fusing 70% quantum policy probabilities with 30% classical safety and directional APF weighting within the set of validated safe actions. While this achieves parity in mission success rate (40.0%), it does not constitute raw quantum computational advantage.

---

## 10. Known Limitations

1. **2D Kinematic Simplification:** The environment models continuous 2D planar motion with point-mass kinematics rather than 6-DOF aerodynamic equations with wind turbulence.
2. **Simulation Overhead:** Simulating $N$-qubit circuits on classical CPUs requires $O(2^N)$ statevector multiplications, introducing simulator latency that would not exist on physical QPU hardware.
3. **Discrete Action Space:** Actions are discretized into 8 cardinal/intercardinal directions plus `STAY`, whereas continuous throttle/steering control is standard in production UAV autopilots.

---

## 11. Quantum Decision Explainability (XAI) Methodology

For every Pure Quantum and Hybrid Quantum-Assisted navigation decision, Q-Nav provides research-grade explainability estimates detailing which spatial features most influenced the variational quantum model's selected action.

### 11.1 Central Finite-Difference Saliency

To compute the local sensitivity of the quantum model's action probability $P(a)$ with respect to feature $x_i$, Q-Nav employs central finite differences:

$$\frac{\partial P(a)}{\partial x_i} \approx \frac{P(a \mid x + \epsilon e_i) - P(a \mid x - \epsilon e_i)}{2 \epsilon}$$

where $\epsilon = 10^{-4}$ is the perturbation step size and $e_i$ is the standard basis unit vector.

### 11.2 Gradient Sensitivity (Log-Probability)

In addition to direct probability saliency, gradient sensitivity measures proportional relative sensitivity by evaluating perturbations on the log-probability:

$$\frac{\partial \log (P(a) + \epsilon_{log})}{\partial x_i} \approx \frac{\log(P(a \mid x + \epsilon e_i) + \epsilon_{log}) - \log(P(a \mid x - \epsilon e_i) + \epsilon_{log})}{2 \epsilon}$$

with numerical stabilizer $\epsilon_{log} = 10^{-8}$.

### 11.3 Feature Selection & Canonical 12D Mapping

The quantum model internally maps the 12D navigation feature vector to the 4 most critical dimensions via indices `[1, 2, 5, 9]`:
- **4D Index 0** $\to$ `target_direction_x` (12D Index 1)
- **4D Index 1** $\to$ `target_direction_y` (12D Index 2)
- **4D Index 2** $\to$ `nearest_obstacle_distance` (12D Index 5)
- **4D Index 3** $\to$ `obstacle_density` (12D Index 9)

Explanations calculated directly on the 4D input space require only $2 \times 4 = 8$ circuit executions per decision (rather than 24 passes for 12D), and results are strictly mapped back to their canonical feature names rather than generic synthetic labels (`feature_0`, etc.).

### 11.4 Signed Gradients and Normalized Importance

For each feature $i$, Q-Nav records both:
1. **Normalized Absolute Importance:**
   $$I_i = \frac{|\nabla_i|}{\sum_{j} |\nabla_j| + 10^{-8}}$$
   If all sensitivity values are near zero ($\sum |\nabla_j| < 10^{-12}$), importance gracefully resolves to $0.0$ rather than generating `NaN`.
2. **Signed Local Gradient ($\nabla_i$):**
   Distinguishes whether locally increasing a feature increases ($\nabla_i > 0$) or decreases ($\nabla_i < 0$) the probability of action $a$. A positive gradient indicates local reinforcement of the candidate action, not that the feature is universally "good".

### 11.5 Hybrid Quantum vs. Final Action Disambiguation

In the hybrid architecture, the quantum model outputs a policy distribution, but the classical APF or lookahead safety layer may override the quantum top action. Q-Nav explicitly logs:
- `quantum_top_action`: The quantum model's recommended action.
- `selected_action` / `final_action`: The executed action following APF fusion and safety verification.
- `quantum_action_used`: Boolean indicating whether the final action matched the quantum recommendation.

When explaining the final hybrid action, the system computes the quantum model's sensitivity conditioned on that action while explicitly clarifying that this does **not** imply the quantum model selected it if fusion overrode the recommendation.

### 11.6 Explanation Overhead & Performance Control

Finite-difference evaluations require $2 \times N_{feat}$ quantum circuit executions per step. To prevent degrading benchmark performance:
- Explanations are **disabled by default** (`--explain-decisions` CLI flag).
- `explanation_time` is measured and stored independently from the standard navigation `decision_time` telemetry to maintain fair benchmark integrity.

### 11.7 Scientific Disclaimers & Limitations

> [!IMPORTANT]
> **Scientific Disclosure on Interpretability:**  
> - **These explanations estimate local sensitivity of the model output to input features. They should not be interpreted as causal explanations.**  
> - **Feature importance is action-conditioned: importance depends on which action is being explained.**  
> - Local finite-difference derivatives reflect the curvature of the current model parameterized landscape and do not guarantee global monotonicity across the entire flight envelope.

---

## 12. Dynamic / Moving Obstacle Navigation Methodology

### 12.1 Overview & Motivation

Real-world low-altitude airspaces feature non-stationary obstacles, including intruding aircraft, uncoordinated drones, migratory birds, and surface vehicles.

> [!NOTE]
> **Core Premise:** "Dynamic obstacle experiments evaluate navigation under time-varying obstacle configurations."

### 12.2 Moving Obstacle Models & Kinematic Patterns

Moving obstacles are modeled as dynamic spherical/cylindrical volumes governed by the `MovingObstacle` class in `simulation/obstacles.py`. Each obstacle maintains state:
$$\mathbf{s}_{obs}(t) = \left[ \mathbf{p}(t), \mathbf{v}(t), r, t, \text{pattern} \right]$$

Three deterministic and reproducible motion patterns are supported:
1. **Linear Motion with Boundary Reflection:**
   $$\mathbf{p}(t + \Delta t) = \mathbf{p}(t) + \mathbf{v} \Delta t$$
   When an obstacle encounters airspace boundaries $[x_{min}, x_{max}, y_{min}, y_{max}]$, coordinates are evaluated independently using scalar checks:
   - If $x < x_{min} + r$: $x \leftarrow x_{min} + r$, $v_x \leftarrow |v_x|$
   - If $x > x_{max} - r$: $x \leftarrow x_{max} - r$, $v_x \leftarrow -|v_x|$
   - Handled identically and independently for the $Y$ axis.
2. **Circular Orbital Motion:**
   Orbital position is parameterized around a fixed center $\mathbf{c} = (c_x, c_y)$ with orbit radius $R_{orbit}$, angular frequency $\omega$, and initial phase $\phi_0$:
   $$x(t) = c_x + R_{orbit} \cos(\phi_0 + \omega t)$$
   $$y(t) = c_y + R_{orbit} \sin(\phi_0 + \omega t)$$
   Instantaneous velocity corresponds to the tangential derivative:
   $$\mathbf{v}(t) = [-R_{orbit} \omega \sin(\phi_0 + \omega t), \; R_{orbit} \omega \cos(\phi_0 + \omega t)]^T$$
   The physical collision radius $r$ remains strictly decoupled from the orbit radius $R_{orbit}$.
3. **Deterministic Random Walk:**
   Stochastic perturbation driven by an isolated, dedicated NumPy random generator (`np.random.default_rng(random_seed)`). Global randomness is never used, guaranteeing that identical random seeds produce identical, byte-reproducible obstacle trajectories across all evaluated models. Velocity changes are capped by `max_speed` and boundary-reflected.

### 12.3 Mutation-Free Trajectory Prediction

Proactive collision prevention requires predicting future obstacle states without altering the current simulation clock or physical state:
$$\hat{\mathbf{p}}(t + \tau) = \text{predict\_position}(\tau, \Delta t)$$
The `predict_position()` method projects the motion equations forward over horizon $\tau$ while preserving `self.position`, `self.velocity`, `self.t`, and `self.rng` unchanged.

### 12.4 The 11-Step Simulation Lifecycle

To prevent simulation step artifacts, race conditions, or asymmetric observation advantages, every simulation step executes in a strictly enforced 11-step lifecycle:
1. **Obtain current UAV state:** Retrieve drone position, velocity, heading, and speed.
2. **Obtain current obstacle states:** Query current spatial locations and velocities of all static and moving obstacles.
3. **Calculate navigation features:** Extract the normalized 12-dimensional telemetry feature vector.
4. **Generate classical/quantum action:** Produce model recommendation (APF direction, Pure Quantum policy, or Hybrid fusion).
5. **Apply safety filtering:** Evaluate candidate actions against boundaries, static buffers, and forward-swept moving obstacle projections.
6. **Move UAV:** Advance drone position according to selected direction, speed, and timestep $\Delta t$.
7. **Advance moving obstacles:** Integrate moving obstacle trajectories forward by $\Delta t$ and apply boundary reflections.
8. **Perform collision detection:** Evaluate simultaneous endpoint and swept-circle collision tests for UAV vs static obstacles, UAV vs moving obstacles, and flight perimeters.
9. **Update simulation time:** Increment simulation clock $t \leftarrow t + \Delta t$.
10. **Record telemetry:** Log comprehensive dynamic telemetry (kinematics, relative distances, predicted clearances, interventions).
11. **Check target reached / episode termination:** Evaluate goal acceptance sphere ($r_{target} = 2.0$), collision state, out-of-bounds, or timeout.

### 12.5 Continuous Swept-Circle Collision Checking

Discrete endpoint checking risks "tunneling" when a high-speed UAV crosses an obstacle trajectory between timesteps. Q-Nav employs continuous relative swept-circle collision detection over normalized time $\tau \in [0, 1]$:
$$\mathbf{p}_{uav}(\tau) = \mathbf{p}_{uav}(0) + \tau \Delta \mathbf{p}_{uav}$$
$$\mathbf{p}_{obs}(\tau) = \mathbf{p}_{obs}(0) + \tau \Delta \mathbf{p}_{obs}$$
The relative displacement vector is linear in $\tau$:
$$\mathbf{r}(\tau) = \mathbf{r}_0 + \tau \mathbf{v}_{rel}$$
where $\mathbf{r}_0 = \mathbf{p}_{uav}(0) - \mathbf{p}_{obs}(0)$ and $\mathbf{v}_{rel} = \Delta \mathbf{p}_{uav} - \Delta \mathbf{p}_{obs}$. The squared center-to-center Euclidean separation:
$$D^2(\tau) = \|\mathbf{r}_0\|^2 + 2 \tau (\mathbf{r}_0 \cdot \mathbf{v}_{rel}) + \tau^2 \|\mathbf{v}_{rel}\|^2$$
The global minimum over $\tau \in [0, 1]$ occurs at:
$$\tau^* = \text{clamp}\left( -\frac{\mathbf{r}_0 \cdot \mathbf{v}_{rel}}{\|\mathbf{v}_{rel}\|^2}, \; 0, \; 1 \right)$$
A collision is registered if:
$$\|\mathbf{r}(\tau^*)\| \le r_{uav} + r_{obs} + \text{margin}$$

### 12.6 Proactive Dynamic Safety Checking

The `SafetyLayer` prevents collisions before they occur by testing candidate actions against projected moving obstacle trajectories:
- **Prediction Horizon:** $T_{horiz} = 1.0\text{ s}$
- **Prediction Discretization:** $\Delta t_{pred} = 0.1\text{ s}$
- **Dynamic Safety Margin:** $\delta_{dyn} = 0.5\text{ units}$

For each candidate action:
1. Predict UAV trajectory over horizon $T_{horiz}$.
2. Predict all moving obstacle trajectories over $T_{horiz}$.
3. If any projected point enters $r_{uav} + r_{obs} + \delta_{dyn}$ or intersects via swept circle, the action is marked **unsafe**.
4. The safety layer rejects the unsafe action and selects the highest-scoring safe alternative.
5. If all 8 directional actions are unsafe, the system falls back to the integer **STAY (0)** action (never `None`).

### 12.7 Scientific Fairness & Controlled Evaluation

To maintain strict scientific integrity:
1. **Identical Environments:** Classical APF, Pure Quantum, and Hybrid models execute in the exact same scenario, starting state, obstacle trajectories, and seed.
2. **Feature Vector Invariance:** The existing 12D spatial feature vector format and VQC circuit parameters ($N_{qubits} = 4$, $N_{layers} = 2$) remain 100% unchanged.
3. **No Fabricated Benchmarks:** All reported metrics (success rates, collision counts, clearances) are directly measured from simulated episodes.
4. **Separate Output Files:** Dynamic benchmarks are saved to dedicated files (`dynamic_comparison.json`, `dynamic_telemetry.csv`, `dynamic_trajectories.json`) to prevent corrupting static benchmark metrics.
5. **Solvability Verification:** Every dynamic scenario undergoes breadth-first search (BFS) grid connectivity validation prior to benchmarking to prove that the target is reachable.

### 12.8 Scientific Limitations

---

## 13. Real IBM Quantum Hardware Validation & Noise Characterization

### 13.1 Research Objective & Scientific Scope

Real IBM Quantum hardware execution extends Q-Nav by deploying the identical parameterized quantum circuit (VQC) to physical superconducting quantum processing units (QPUs).

> [!IMPORTANT]
> **Mandatory Scientific Disclaimer:**
> "Real IBM Quantum hardware execution introduces device noise, finite-shot sampling, transpilation overhead, queue latency, and time-varying calibration conditions. Hardware validation therefore measures the behavior of the implemented quantum circuit on a physical device and does not demonstrate quantum computational advantage."
> 
> "The Q-Nav benchmark does not establish quantum advantage over the classical baseline."

Hardware validation serves two explicit scientific functions:
1. **Physical Circuit Validation:** Verifies that the Q-Nav navigation VQC successfully compiles, transpiles, and executes on real IBM Quantum hardware architectures.
2. **Noise Characterization:** Measures probability distribution deviation ($\Delta P = P_{hw} - P_{sim}$), Jensen-Shannon divergence, and unmapped computational state leakage caused by physical decoherence, gate errors, and readout infidelities.

### 13.2 IBM Quantum Runtime Architecture (SamplerV2)

Q-Nav interfaces with IBM Quantum using the modern **Qiskit IBM Runtime** primitives workflow (`qiskit-ibm-runtime>=0.47.0` on `qiskit>=2.5.0`):
- **Primitive:** `SamplerV2(mode=backend)`
- **PUB Format:** Accepts Primitive Unified Blocks `[transpiled_circuit]` with explicit shot configuration `shots=1024`.
- **Measurement Extraction:** Bitstring counts extracted directly from `pub_result.data.meas.get_counts()`.
- **Channel Configuration:** Initialized with `channel="ibm_quantum_platform"` as required by `qiskit-ibm-runtime>=0.47.0` for IBM Quantum Platform accounts.

### 13.3 Authentication & Security Best Practices

To safeguard sensitive credentials:
- **Zero Hardcoded Secrets:** The API token is never hardcoded, logged, saved to disk, displayed in dashboards, or committed to version control.
- **Environment Variable:** Loaded strictly from `IBM_QUANTUM_TOKEN`.
- **Exception Sanitization:** Exception handlers scrub and redact token occurrences before outputting error messages.

#### Configuration Examples:

**Windows PowerShell:**
```powershell
$env:IBM_QUANTUM_TOKEN="YOUR_API_TOKEN"
```

**Windows CMD:**
```cmd
set IBM_QUANTUM_TOKEN=YOUR_API_TOKEN
```

**Linux / macOS:**
```bash
export IBM_QUANTUM_TOKEN="YOUR_API_TOKEN"
```

### 13.4 Backend Discovery & Selection

Q-Nav provides automated backend inspection via:
```bash
python run.py --list-hardware-backends
```
Metadata collected without credential exposure:
- Backend name (e.g., `ibm_kyiv`, `ibm_sherbrooke`, `ibm_brisbane`)
- Physical qubit count ($N_{qubits} \ge 4$)
- Operational status (`operational=True`)
- Simulator flag (`is_simulator=False`)
- Active pending jobs queue depth
- Native basis gates and coupling graph topology

Selection criteria prioritize operational, non-simulator backends with sufficient physical qubits and minimal queue congestion.

### 13.5 Action Space Mapping & Invalid State Tracking

Q-Nav navigates using **9 discrete spatial actions** (8 directional compass headings + 1 STAY):
$$|\mathcal{A}| = 9$$
For an $N=4$ qubit circuit, the computational basis spans $2^4 = 16$ orthogonal basis states:
- **Valid Action States:** $|0000\rangle$ through $|1000\rangle$ (integers $0 \dots 8$)
- **Invalid Basis States:** $|1001\rangle$ through $|1111\rangle$ (integers $9 \dots 15$)

Physical hardware noise causes shot leakages into the 7 unmapped states. Rather than silently discarding them, Q-Nav explicitly computes:
$$P(s) = \frac{\text{count}(s)}{N_{shots}}, \quad \forall s \in \{0, \dots, 15\}$$
$$\text{Valid Probability Mass} = \sum_{s=0}^{8} P(s)$$
$$\text{Invalid Probability Mass} = \sum_{s=9}^{15} P(s)$$

Normalized valid action probabilities are computed as:
$$P_{action}(a) = \frac{P(a)}{\sum_{s=0}^8 P(s)}, \quad a \in \{0, \dots, 8\}$$
The invalid probability mass is logged in telemetry and telemetry CSVs to quantify physical device noise.

### 13.6 Statistical Divergence & Noise Metrics

To quantify distribution deviation from ideal statevector simulation:
1. **Mean Absolute Probability Difference (MAPD):**
   $$\text{MAPD} = \frac{1}{9} \sum_{a=0}^{8} |P_{hw}(a) - P_{sim}(a)|$$
2. **Maximum Probability Difference:**
   $$\Delta P_{max} = \max_{a} |P_{hw}(a) - P_{sim}(a)|$$
3. **Jensen-Shannon Divergence (JSD):**
   $$M = \frac{1}{2}(P_{sim} + P_{hw})$$
   $$\text{JSD}(P_{sim} \parallel P_{hw}) = \frac{1}{2} D_{KL}(P_{sim} \parallel M) + \frac{1}{2} D_{KL}(P_{hw} \parallel M) \in [0, 1]$$
4. **Action Agreement Rate:**
   $$\text{Agreement} = \mathbb{I}\left(\arg\max_a P_{hw}(a) == \arg\max_a P_{sim}(a)\right)$$

### 13.7 Strict Non-Fallback Policy

When real hardware execution is commanded via `--real-hardware`:
- **Default:** Real hardware only. If connectivity, credentials, queue timeouts, or hardware errors occur, the platform aborts with a descriptive error.
- **Opt-In Fallback:** Simulation fallback occurs **only** if the user explicitly supplies `--hardware-fallback-simulator`.

### 13.8 Hardware Validation CLI Reference

```bash
# 1. Inspect operational IBM Quantum backends
python run.py --list-hardware-backends

# 2. Run real hardware validation experiment (small controlled sample)
python run.py --hardware-validation --real-hardware --backend YOUR_BACKEND --shots 1024 --hardware-samples 3

# 3. Run simulated hardware validation without IBM credentials
python run.py --hardware-validation --shots 1024 --hardware-samples 3
```

All results are logged to:
- `outputs/metrics/hardware_validation.json`
- `outputs/metrics/hardware_validation.csv`
- `outputs/logs/hardware_jobs.json`
- `outputs/logs/hardware_counts.json`

