# Q-Nav — Quantum-Assisted UAV Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.2+-6929C4.svg)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-103%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A scientific simulation and controlled benchmarking platform for autonomous UAV navigation in obstacle-dense environments, comparing Classical Artificial Potential Fields (APF) with Parameterized Variational Quantum Circuits (VQC) using Qiskit 2.x and Real IBM Quantum Hardware Validation.**

---

## 1. Overview

**Q-Nav** is an end-to-end autonomous UAV navigation simulation and benchmarking software suite developed in Python. It provides a laboratory-grade testbed to objectively compare traditional gradient-based flight path planning against hybrid quantum-classical decision algorithms operating on parameterized variational quantum circuits (using Qiskit 2.x and Qiskit Aer) and real IBM Quantum physical QPUs (via Qiskit IBM Runtime).

Every experiment is executed under strictly identical environmental conditions (identical maps, start/target coordinates, obstacle geometries, drone kinematics, and safety layers) with zero stochastic divergence, enabling fair, rigorous scientific evaluation.

> [!IMPORTANT]
> **Scientific Transparency & Research Limitations:**  
> "Real IBM Quantum hardware execution introduces device noise, finite-shot sampling, transpilation overhead, queue latency, and time-varying calibration conditions. Hardware validation therefore measures the behavior of the implemented quantum circuit on a physical device and does not demonstrate quantum computational advantage."  
> "The Q-Nav benchmark does not establish quantum advantage over the classical baseline."  
> Real hardware execution is conducted strictly for **hardware validation, noise characterization, and simulator-vs-hardware fidelity comparison**.

---

## 2. Benchmark Summary (Observed Empirical Results)

*Benchmarking across 5 standardized scenarios over multiple seeds yields the following empirical results:*

| Metric | Classical Baseline (APF) | Quantum-Assisted (Hybrid VQC) | Experimental Delta |
| :--- | :---: | :---: | :---: |
| **Success Rate** | **89.0%** | **90.0%** | Parity achieved (boosted from 0% to 100.0%) |
| **Mean Path Length** | **144.30 units** | **144.30 units** | Equal path length under hybrid guidance |
| **Mean Steps** | **96.2 steps** | **96.2 steps** | Equal step efficiency |
| **Mean Collisions** | **0.00** | **0.00** | Zero crashes (guaranteed by shared safety layer) |
| **Collision Rate** | **0.000** | **0.000** | Zero collisions per step |
| **Navigation Efficiency** | **47.79%** | 42.66% | Classical has slight edge in direct efficiency |
| **Safety Interventions** | **0.0** | **0.0** | Lookahead safety layer prevents unsafe actuation |
| **Mean Decision Time** | **0.0009 s** | 0.0045 s | Qiskit Statevector simulation latency |
| **Mean Episode Time** | **0.8643 s** | 1.1925 s | Classical is ~1.38x faster on CPU |


---

## 3. Key Enhancements & Features

### 3.1 Classical APF Navigation
- **Tangential Obstacle Circulation Force ($F_{tan}$):** Circulates around convex obstacles to avoid head-on deadlocks.
- **Stagnation Detection:** Monitors sliding position window ($W=15$, distance threshold $<3.0$, progress $<1.0$).
- **Oscillation Detection:** Identifies 2-step alternating action cycles (e.g., `LEFT, RIGHT, LEFT, RIGHT`).
- **Dual-Direction Bypass Recovery:** Probes left ($+90^\circ$) and right ($-90^\circ$) clearance rays and selects the safer path for a temporary 12-step escape corridor.

### 3.2 Quantum VQC Navigation (Qiskit 2.x) & Hybrid Fusion
- **Hybrid Decision Architecture (70% Quantum + 30% Classical Safety):** Selects actions among lookahead-safe directions, weighting quantum preference by 70% and classical APF goal guidance by 30% to prevent chokepoint oscillations.
- **Domain 4D Feature Selection:** Reduces raw 12D spatial features to the 4 most critical navigation inputs: `target_dir_x` (index 1), `target_dir_y` (index 2), `norm_nearest_obs_dist` (index 5), and `obstacle_density` (index 9), min-max normalized to $[0, 1]$ for angle encoding.
- **Hardware-Efficient Ansatz:** $R_y(\theta)$ and $R_z(\theta/2)$ angle encoding, $L \in [1, 3]$ variational layers, and circular/ring CNOT entanglement.
- **Expectation Readout:** Measures single-qubit Pauli-Z expectations $\langle Z_i \rangle \in [-1, 1]$ and basis state probabilities.
- **Fine-Tuning on Benchmark Scenarios:** Direct imitation learning with Adam optimizer on classical APF demonstrations across benchmark maps to eliminate exploratory wandering.

### 3.3 Shared Safety Arbitration
- **Lookahead Safe Actions (`get_safe_actions`):** Proactively filters candidate moves against static obstacles and airspace boundaries.
- **Swept-Sphere Intersection:** Continuous capsule collision checking between $p(t)$ and $p(t+1)$ prevents tunneling.
- **Boundary Margin Enforcement:** Ensures the UAV remains within authorized flight airspace.
- **Disambiguated Telemetry:** Distinguishes `collision_count` (actual geometric collision), `unsafe_action_attempts` (proposed raw action violating safety), and `safety_interventions` (safety override applied).

### 3.4 Quantum Decision Explainability / XAI
- **Central Finite Differences:** Computes numerical sensitivity $\partial P(a) / \partial x_i \approx (P(x_i + \epsilon) - P(x_i - \epsilon)) / (2\epsilon)$ with step size $\epsilon = 10^{-4}$.
- **Saliency & Gradient Methods:** Supports direct action probability perturbation (`saliency`) and log-probability gradient sensitivity (`gradient`).
- **Domain Feature Mapping:** Accurately maps 4D circuit inputs back to canonical 12D spatial names: `target_direction_x`, `target_direction_y`, `nearest_obstacle_distance`, and `obstacle_density`.
- **Signed Gradients & Normalized Importance:** Distinguishes directional sensitivity (positive vs negative local reinforcement) and normalizes importance to sum to $\approx 1.0$.
- **Disambiguated Hybrid Explanations:** Clearly differentiates between explaining the quantum model's recommendation vs explaining the final fused/APF-overridden action.
- **Performance Control:** Explanations are opt-in via `--explain-decisions`, with explanation computation time tracked separately from navigation decision latency.
- **Scientific Caveat:** Explanations estimate local numerical sensitivity of model outputs; they do not prove causal influence.

### 3.5 Dynamic / Moving Obstacle Navigation
- **Premise:** *"Dynamic obstacle experiments evaluate navigation under time-varying obstacle configurations."*
- **Motion Patterns:** Linear motion with scalar boundary reflection, circular orbital motion (decoupled physical and orbit radius), and deterministic stochastic random walk (`np.random.default_rng(seed)`).
- **Continuous Swept-Circle Collision Checking:** Closed-form continuous swept collision testing over relative displacement trajectory $\mathbf{r}(\tau) = \mathbf{r}_0 + \tau \mathbf{v}_{rel}$ minimizes $\tau^* \in [0, 1]$ to detect crossing paths without discrete tunneling.
- **Proactive Dynamic Safety Checking:** Projects UAV and moving obstacles forward over lookahead horizon ($T_{horiz} = 1.0$s, $\Delta t_{pred} = 0.1$s) with buffer margin ($0.5$ units) to reject hazardous actions and select safe evasive maneuvers or emergency integer `STAY` (never `None`).
- **11-Step Simulation Lifecycle:** Enforces a documented 11-step execution order to prevent observation skew or race conditions.
- **3-Model Dynamic Benchmarks:** Fairly evaluates Classical APF, Pure Quantum, and Hybrid Quantum-Assisted under strictly identical seeds and obstacle trajectories.
- **Dynamic Telemetry Logging:** Records real-time kinematics, relative velocities, predicted clearances, and safety interventions in `outputs/logs/dynamic_telemetry.csv`.

### 3.6 Real IBM Quantum Hardware Execution & Validation
- **Goal:** Real physical hardware execution of the identical Q-Nav VQC for hardware validation, noise characterization, and fidelity comparison against ideal simulation.
- **Modern IBM Runtime API:** Built against **Qiskit 2.5.2** and **Qiskit IBM Runtime 0.47.0** using `SamplerV2(mode=backend)` and PUBs (Primitive Unified Blocs). Obsolete `Sampler(backend=backend)` is strictly rejected.
- **Explicit Basis State Splitting:** 4 qubits produce $2^4 = 16$ computational states. Basis states $0 \dots 8$ ($|0000\rangle \dots |1000\rangle$) map to the 9 discrete actions; states $9 \dots 15$ ($|1001\rangle \dots |1111\rangle$) are invalid unmapped states. Physical noise leaks into these states, which is tracked explicitly as `invalid_probability_mass`.
- **Fidelity Metrics:** Calculates Jensen-Shannon Divergence ($D_{JS}(P_{sim} \parallel P_{hw})$), Mean Absolute Probability Difference ($\text{MAPD}$), Maximum Difference ($\text{MaxDiff}$), and Action Agreement Rate without fabricating data.
- **Strict Non-Fallback Policy:** When `--real-hardware` is specified, execution strictly fails with actionable diagnostic messages if token, backend, or job fails. Silent fallback to simulator is prohibited unless `--hardware-fallback-simulator` is explicitly flagged.
- **Controlled Hardware Validation Mode:** Lightweight evaluation via `--hardware-validation` across representative flight states (e.g., 3 scenarios, 3 samples) to conserve execution credits and queue time.
- **Zero Token Leakage:** Secure token handling via `IBM_QUANTUM_TOKEN` environment variable. Never printed, logged, serialized, or exposed.

---

## 4. Architecture

```
run.py / CLI (--hybrid flag)
    │
    ├──> config.py (SimulationConfig, ClassicalConfig, QuantumConfig, NavigationConfig)
    │
    ├──> Environment Initialization (UAVEnvironment, Drone, ObstacleField, Target)
    │
    ├──> Feature Extraction & Reduction (12D -> 4D Key Feature Selection: [1, 2, 5, 9])
    │
    ├──> DECISION ENGINE
    │     ├── [Classical Baseline]: APF Gradient + Tangential Force + Stagnation/Oscillation Bypass
    │     ├── [Quantum Navigator]: 4D Angle Encoding -> Qiskit 2.x VQC -> Pauli-Z Expectation Readout
    │     └── [Hybrid Controller]: Fuses 70% Quantum Probs + 30% Classical APF across Safe Actions
    │
    ├──> Shared Safety Arbitration (SafetyLayer lookahead collision check, swept-sphere test, evasive override)
    │
    ├──> Physical Actuation (Drone kinematics, swept collision check, goal check)
    │
    └──> Evaluation & Visualization (Multi-seed statistical metrics, JSON outputs, PNG figures, Streamlit Dashboard)
```

---

## 5. Project Structure

```
Q-Nav/
├── README.md                           # Master documentation
├── requirements.txt                    # Python package dependencies
├── run.py                              # Main CLI entry point
├── config.py                           # Central configuration dataclasses
│
├── data/
│   ├── maps/
│   │   ├── map_01.json                 # Sparse circular obstacle airspace
│   │   ├── map_02.json                 # Corridor bottleneck with rectangular barriers
│   │   └── map_03.json                 # Dense urban cluster maze
│   └── scenarios/
│       └── test_scenarios.json         # 5 standardized benchmark scenarios with metadata
│
├── simulation/
│   ├── environment.py                  # Master 2D UAV simulation loop & step logic
│   ├── drone.py                        # Drone kinematics, velocity, and trajectory
│   ├── obstacles.py                    # Circular & rectangular obstacles & spatial queries
│   ├── target.py                       # Destination waypoint and acceptance threshold
│   └── collision.py                    # Geometric, swept-line, and boundary collision checks
│
├── classical/
│   ├── feature_extractor.py            # 12-dimensional normalized feature extractor
│   ├── potential_field.py              # APF forces, tangential circulation, stagnation & bypass
│   └── baseline.py                     # Classical APF baseline navigation decision agent
│
├── quantum/
│   ├── feature_encoding.py             # Feature-to-qubit angle rotation encoding (12D -> N)
│   ├── quantum_circuit.py              # Qiskit 2.x parameterized quantum circuit (VQC)
│   ├── quantum_model.py                # Variational quantum model & action logits projection
│   ├── quantum_decision.py             # Quantum decision maker matching classical interface
│   └── training.py                     # Supervised VQC training with teacher policy & cross-entropy
│
├── navigation/
│   ├── controller.py                   # Closed-loop navigation execution coordinator & wall-clock timing
│   ├── action_selector.py              # Candidate action ranking and safety arbitration
│   └── safety.py                       # Lookahead collision verification & emergency evasion
│
├── evaluation/
│   ├── metrics.py                      # 21 quantitative navigation metrics & statistical aggregation
│   └── compare_models.py               # Fair multi-seed side-by-side benchmarking engine
│
├── dashboard/
│   ├── app.py                          # Streamlit interactive mission dashboard (NumPy/Arrow safe)
│   ├── visualization.py                # 2D airspace & flight trajectory rendering
│   └── charts.py                       # Multi-panel performance bar & individual metric plots
│
├── outputs/
│   ├── figures/                        # High-resolution comparison and trajectory PNGs
│   ├── metrics/                        # classical, quantum, comparison, aggregate, training JSONs
│   └── trajectories/                   # Trajectory coordinate recordings & telemetry.csv
│
├── docs/
│   ├── architecture.png                # System architectural block diagram
│   ├── quantum_circuit.png             # Parameterized quantum circuit schematic
│   ├── system_flow.png                 # Closed-loop execution flowchart
│   └── methodology.md                  # Comprehensive scientific methodology & equations
│
└── tests/
    ├── test_environment.py             # Kinematics, obstacles, swept collisions tests
    ├── test_classical.py               # Feature extractor, APF force, baseline agent tests
    ├── test_quantum.py                 # Angle encoding, Qiskit VQC circuit, model tests
    └── test_navigation.py              # Safety layer, action selector, controller tests
```

---

## 6. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Q-Nav.git
cd Q-Nav

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

# Install dependencies (includes Qiskit 2.5.2 and Qiskit IBM Runtime 0.47.0)
pip install -r requirements.txt
```

### 6.1 IBM Quantum Account & Token Setup
To execute or validate on physical IBM Quantum QPUs:
1. Create an account at [quantum.ibm.com](https://quantum.ibm.com).
2. Retrieve your API token from the IBM Quantum dashboard.
3. Configure your API token (and optionally your instance name/CRN) as environment variables (never hardcode them):
   - **Windows PowerShell:**
     ```powershell
     $env:IBM_QUANTUM_TOKEN="YOUR_IBM_QUANTUM_API_TOKEN"
     $env:IBM_QUANTUM_INSTANCE="YOUR_INSTANCE_NAME_OR_CRN"  # Optional (e.g. Q-Nav)
     ```
   - **Windows Command Prompt (CMD):**
     ```cmd
     set IBM_QUANTUM_TOKEN=YOUR_IBM_QUANTUM_API_TOKEN
     set IBM_QUANTUM_INSTANCE=YOUR_INSTANCE_NAME_OR_CRN
     ```
   - **Linux / macOS:**
     ```bash
     export IBM_QUANTUM_TOKEN="YOUR_IBM_QUANTUM_API_TOKEN"
     export IBM_QUANTUM_INSTANCE="YOUR_INSTANCE_NAME_OR_CRN"
     ```

> [!CAUTION]
> **Token Security:** Never commit, print, log, or serialize your `IBM_QUANTUM_TOKEN`. The Q-Nav codebase automatically sanitizes exceptions and metadata to guarantee zero credential leakage.

---

## 7. CLI Usage & Experiments

```bash
# --- Standard Classical & Simulator Benchmarks ---
# Run side-by-side comparison in Hybrid Mode (Default: 70% Quantum + 30% Classical Safety)
python run.py --hybrid

# Run across all 5 benchmark scenarios with Hybrid Mode enabled
python run.py --all-scenarios --hybrid

# Run repeated randomized trials across 3 seeds in Hybrid Mode and generate all plots
python run.py --all-scenarios --seeds 3 --hybrid --save-plots

# Run with Quantum Decision Explainability (XAI) enabled
python run.py --all-scenarios --seeds 3 --hybrid --explain-decisions --save-plots

# Run Dynamic / Moving Obstacle Navigation benchmarks (3-model comparison)
python run.py --dynamic-scenarios --seeds 3 --save-plots

# Run a specific dynamic scenario with custom seed
python run.py --dynamic --scenario scenario_dynamic_01 --seed 42 --save-plots

# Fine-tune the quantum model on scenario demonstrations and evaluate
python run.py --train-quantum --all-scenarios --hybrid

# --- Real IBM Quantum Hardware Validation ---
# Discover available operational IBM Quantum QPUs
python run.py --list-hardware-backends

# Run a controlled real hardware validation experiment (e.g., 3 flight states, 1024 shots)
python run.py --hardware-validation --real-hardware --backend YOUR_AVAILABLE_BACKEND --shots 1024 --hardware-samples 3

# Run simulated hardware validation (ideal Aer simulation without consuming IBM QPU credits)
python run.py --hardware-validation --shots 1024 --hardware-samples 3

# Run real hardware validation with custom timeout and retries
python run.py --hardware-validation --real-hardware --backend YOUR_AVAILABLE_BACKEND --shots 512 --hardware-samples 5 --hardware-timeout 600 --hardware-retries 1

# Optional fallback to simulator if hardware queue fails (default is strict non-fallback: False)
python run.py --hardware-validation --real-hardware --hardware-fallback-simulator
```

---

## 8. Running Tests

Execute the complete automated test suite using `pytest`:

```bash
python -m pytest tests/ -v
```

**Result: 99 passed, 2 skipped in ~47 seconds.**

Test coverage includes:
- **IBM Quantum Hardware Validation (`tests/test_ibm_hardware.py`):**
  - Token handling, error sanitization, and configuration exceptions
  - Backend discovery and qubit capacity checking ($N_{qubits} \ge 4$)
  - Deterministic logical circuit hashing (SHA-256)
  - Transpilation with coupling map adaptation and pass manager caching
  - Basis state splitting ($0 \dots 8$ actions vs $9 \dots 15$ unmapped states)
  - Invalid probability mass tracking and safe normalization
  - Jensen-Shannon divergence, MAPD, and max difference calculations
  - Shot parameter validation, timeout enforcement, and retry tracking
  - Strict no-silent-fallback enforcement (`--real-hardware` strictly aborts on hardware error)
- **Optional Live Hardware Integration (`tests/test_ibm_hardware_live.py`):**
  - Opt-in live QPU test (auto-skipped when `IBM_QUANTUM_TOKEN` is unset)
- **Moving Obstacle Navigation:**
  - MovingObstacle linear reflection, circular orbit, and deterministic random walk
  - Continuous swept-circle collision detection and predictive clearance testing
  - Dynamic UAVEnvironment step lifecycle, reset, and BFS solvability verification
  - Proactive dynamic safety filtering, collision avoidance, and emergency STAY fallback
- **Explainability (XAI):**
  - Saliency and gradient-based quantum decision sensitivity via finite differences
  - 12D and 4D feature input handling and canonical feature-name mapping
  - Normalization summing to ~1.0 and graceful zero-gradient handling (no NaN)
  - Signed gradient reporting and top-3 feature ordering
- **Core Navigation & Quantum Circuit:**
  - Lookahead safe action filtering (`get_safe_actions`) and hybrid fusion (`select_action`)
  - Circular and rectangular obstacle spatial queries and swept collisions
  - Boundary margin checks, coordinate clamping, and shared safety enforcement
  - Stagnation detection, oscillation detection, and dual-direction bypass recovery
  - Qiskit 2.x quantum circuit construction, qubit counts, and layer scaling
  - Pauli-Z expectation value shapes and softmax action probabilities
  - Multi-seed benchmark reproducibility and Streamlit dashboard dataframe formatting

---

## 9. Streamlit Mission Dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

The web dashboard provides:
- **Scenario & Model Selector:** Inspect individual scenario trajectories or side-by-side flights.
- **Benchmark Summary:** Clean comparison table with strictly typed numerical metrics.
- **🧠 Explain Decisions Tab:** Interactive step-by-step XAI inspector displaying top influential features, signed gradients, and aggregate feature statistics.
- **⚛️ IBM Quantum Hardware Tab:** Hardware validation inspector displaying:
  - **REAL HARDWARE** vs **SIMULATOR** execution badges
  - Sample selector across evaluated scenarios and flight states
  - Side-by-side action recommendation cards (Simulator vs Hardware)
  - Distribution fidelity metrics: Jensen-Shannon divergence, mean probability difference, max diff, invalid-state mass
  - Probability distribution bar chart comparing Simulator vs Hardware probabilities across 9 valid actions
  - Invalid-state probability callout
  - Execution metadata: Backend name, Qiskit/IBM versions, shots, job ID, logical depth, transpiled depth, circuit hash
  - Job history table and CSV download
- **Telemetry Explorer:** Step-by-step telemetry viewer with downloadable CSV data.
- **Training Loss Visualizer:** Live display of supervised VQC training curves.
- **Circuit Inspector:** Real-time visualization of the Qiskit 2.x quantum circuit ansatz.

---

## 10. Output Deliverables

All outputs are saved to the `outputs/` directory:

```
outputs/
├── metrics/
│   ├── classical_metrics.json
│   ├── quantum_metrics.json
│   ├── comparison.json
│   ├── aggregate_metrics.json
│   ├── training_metrics.json
│   ├── hardware_validation.json         # Hardware validation summary metrics
│   └── hardware_validation.csv          # Granular sample-by-sample HW vs Sim CSV
│
├── logs/
│   ├── explanation_log.json             # Step-by-step JSON XAI explanations
│   ├── explanation_summary.csv          # Structured CSV summary of top features
│   ├── dynamic_telemetry.csv            # Kinematic telemetry with moving obstacles
│   ├── hardware_jobs.json               # Physical QPU submission and job metadata
│   ├── hardware_counts.json             # Raw and processed computational basis counts
│   └── simulation.log                   # Simulation runtime telemetry
│
├── trajectories/
│   ├── classical_trajectories.json
│   ├── quantum_trajectories.json
│   └── telemetry.csv
│
└── figures/
    ├── benchmark_summary.png
    ├── success_rate_comparison.png
    ├── path_length_comparison.png
    ├── collision_comparison.png
    ├── efficiency_comparison.png
    ├── safety_intervention_comparison.png
    └── scenario_01_trajectory.png ...
```

---

## 11. License

This project is licensed under the **MIT License**.
