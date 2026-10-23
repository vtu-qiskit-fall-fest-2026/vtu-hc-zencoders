"""Q-Nav: Quantum-Assisted UAV Navigation
Configuration Module

Contains all global simulation, navigation, classical algorithm,
quantum circuit, training, and evaluation parameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional


# ==============================================================================
# Base Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MAPS_DIR = DATA_DIR / "maps"
SCENARIOS_DIR = DATA_DIR / "scenarios"
FEATURES_CSV_PATH = DATA_DIR / "navigation_features.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
TRAJECTORIES_DIR = OUTPUT_DIR / "trajectories"
LOGS_DIR = OUTPUT_DIR / "logs"

DOCS_DIR = BASE_DIR / "docs"

# Specific Output Paths
CLASSICAL_METRICS_PATH = METRICS_DIR / "classical_metrics.json"
QUANTUM_METRICS_PATH = METRICS_DIR / "quantum_metrics.json"
COMPARISON_METRICS_PATH = METRICS_DIR / "comparison.json"
AGGREGATE_METRICS_PATH = METRICS_DIR / "aggregate_metrics.json"
TRAINING_METRICS_PATH = METRICS_DIR / "training_metrics.json"

CLASSICAL_TRAJECTORIES_PATH = TRAJECTORIES_DIR / "classical_trajectories.json"
QUANTUM_TRAJECTORIES_PATH = TRAJECTORIES_DIR / "quantum_trajectories.json"
TELEMETRY_CSV_PATH = TRAJECTORIES_DIR / "telemetry.csv"
EXPLANATION_LOG_PATH = LOGS_DIR / "explanation_log.json"
EXPLANATION_SUMMARY_PATH = LOGS_DIR / "explanation_summary.csv"
DYNAMIC_TELEMETRY_PATH = LOGS_DIR / "dynamic_telemetry.csv"
HARDWARE_DIR = OUTPUT_DIR / "hardware"
HARDWARE_METRICS_PATH = METRICS_DIR / "hardware_validation.json"
HARDWARE_CSV_PATH = METRICS_DIR / "hardware_validation.csv"
HARDWARE_JOBS_PATH = LOGS_DIR / "hardware_jobs.json"
HARDWARE_COUNTS_PATH = LOGS_DIR / "hardware_counts.json"


# Ensure all output and data directories exist
for directory in [
    DATA_DIR,
    MAPS_DIR,
    SCENARIOS_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    TRAJECTORIES_DIR,
    LOGS_DIR,
    DOCS_DIR,
    HARDWARE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Simulation & Environment Configuration
# ==============================================================================
@dataclass
class SimulationConfig:
    """Parameters defining the physical simulation environment."""

    # Map dimensions (units)
    map_width: float = 50.0
    map_height: float = 50.0

    # Drone kinematic properties
    drone_speed: float = 1.5
    drone_radius: float = 0.8
    collision_radius: float = 1.0  # Safe buffer around drone + obstacle

    # Target detection
    target_detection_radius: float = 2.0

    # Simulation limits
    max_steps: int = 140
    dt: float = 1.0  # Time step delta

    # Randomness
    random_seed: int = 42

    # Boundary buffer
    boundary_margin: float = 1.0


# ==============================================================================
# Classical Navigation Configuration
# ==============================================================================
@dataclass
class ClassicalConfig:
    """Parameters for the Classical Artificial Potential Field (APF) and baseline navigator."""

    # Attractive potential scaling
    attractive_coefficient: float = 1.2

    # Repulsive potential scaling
    repulsive_coefficient: float = 150.0

    # Distance threshold where obstacles exert repulsive forces
    obstacle_influence_distance: float = 8.0

    # Minimum force threshold below which local minima is suspected
    local_minima_threshold: float = 0.05

    # Perturbation magnitude to escape local minima
    perturbation_magnitude: float = 1.5

    # Direction smoothing factor (0.0 to 1.0)
    momentum_factor: float = 0.2

    # Tangential obstacle avoidance force
    enable_tangential_force: bool = True
    tangential_coefficient: float = 80.0

    # Stagnation detection parameters
    stagnation_window: int = 8
    stagnation_distance_threshold: float = 1.2
    progress_threshold: float = 0.4

    # Oscillation detection parameters
    oscillation_window: int = 6
    oscillation_switch_threshold: int = 4

    # Fallback waypoint / A* navigation threshold
    fallback_stagnation_steps: int = 15
    recovery_duration_steps: int = 5


# ==============================================================================
# Quantum Navigation Configuration
# ==============================================================================
@dataclass
class QuantumConfig:
    """Parameters for Quantum Feature Encoding, Parameterized Circuit, and VQC Model."""

    # Number of qubits in circuit
    n_qubits: int = 4

    # Number of variational/entangling layers
    n_layers: int = 2

    # Backend: "statevector" (exact, fast, reproducible) or "aer_simulator"
    backend_name: str = "statevector"

    # Number of measurement shots (relevant when using shot-based backend)
    shots: int = 1024

    # Feature dimensionality reduction method: "grouped", "pooling", "repeated", or "linear"
    feature_mapping: str = "grouped"
    encoding_strategy: str = "grouped"

    # Feature scaling: maps features into rotation angle range [-pi, pi] or [0, pi]
    angle_scale: float = 3.141592653589793  # pi

    # Reproducibility seed for quantum circuit parameters
    circuit_seed: int = 42

    # Pre-trained/heuristic variational weights seed
    weight_seed: int = 101

    # Training parameters
    learning_rate: float = 0.04
    vqc_learning_rate: float = 0.02
    epochs: int = 25
    batch_size: int = 16
    training_samples: int = 1200
    early_stopping_patience: int = 5
    # Feature selection / encoding: canonical grouped mapping
    use_canonical_encoding: bool = True
    use_feature_selection: bool = True
    # Action stability / hysteresis
    hysteresis_factor: float = 0.25

    # Real Hardware Validation Settings
    real_hardware: bool = False
    hardware_backend: Optional[str] = None
    hardware_instance: Optional[str] = None
    hardware_shots: int = 1024
    hardware_timeout: int = 600
    hardware_retries: int = 0
    hardware_samples: int = 3
    hardware_fallback_simulator: bool = False


# ==============================================================================
# Navigation Control & Safety Configuration
# ==============================================================================
@dataclass
class NavigationConfig:
    """Parameters for action selection and safety layer enforcement."""

    # Lookahead steps for collision checking in safety layer
    safety_lookahead_steps: int = 2

    # Safety buffer distance to obstacles
    safety_obstacle_buffer: float = 1.2

    # Allow diagonal movement (8 directions) vs cardinal only (4 directions)
    allow_diagonal: bool = True

    # Action set names
    action_names: List[str] = field(
        default_factory=lambda: [
            "UP",
            "UP_RIGHT",
            "RIGHT",
            "DOWN_RIGHT",
            "DOWN",
            "DOWN_LEFT",
            "LEFT",
            "UP_LEFT",
            "STAY",
        ]
    )

    # 2D unit vector mappings for discrete actions
    action_vectors: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "UP": (0.0, 1.0),
            "UP_RIGHT": (0.70710678, 0.70710678),
            "RIGHT": (1.0, 0.0),
            "DOWN_RIGHT": (0.70710678, -0.70710678),
            "DOWN": (0.0, -1.0),
            "DOWN_LEFT": (-0.70710678, -0.70710678),
            "LEFT": (-1.0, 0.0),
            "UP_LEFT": (-0.70710678, 0.70710678),
            "STAY": (0.0, 0.0),
        }
    )
    # Hybrid quantum-classical action fusion
    use_hybrid: bool = False
    hybrid_quantum_weight: float = 0.7
    hybrid_classical_weight: float = 0.3
    # Quantum decision explainability (XAI)
    explain_decisions: bool = False
    explanation_method: str = "saliency"
    explanation_eps: float = 1e-4
    # Dynamic obstacles and predictive safety
    prediction_horizon: float = 1.0
    prediction_dt: float = 0.1
    dynamic_safety_margin: float = 0.5
    # Action stability / hysteresis
    action_hysteresis: float = 0.25

# Module-level alias for discrete action names
ACTIONS: List[str] = [
    "UP",
    "UP_RIGHT",
    "RIGHT",
    "DOWN_RIGHT",
    "DOWN",
    "DOWN_LEFT",
    "LEFT",
    "UP_LEFT",
    "STAY",
]

# Canonical 12D environmental telemetry feature names
FEATURE_NAMES: List[str] = [
    "normalized_distance_to_target",
    "target_direction_x",
    "target_direction_y",
    "target_angle_sin",
    "target_angle_cos",
    "nearest_obstacle_distance",
    "front_obstacle_distance",
    "left_obstacle_distance",
    "right_obstacle_distance",
    "obstacle_density",
    "normalized_speed",
    "boundary_proximity",
]

# Selected key features for quantum state encoding: [1, 2, 5, 9]
QUANTUM_SELECTED_FEATURE_INDICES: List[int] = [1, 2, 5, 9]

QUANTUM_SELECTED_FEATURE_NAMES: List[str] = [
    FEATURE_NAMES[idx] for idx in QUANTUM_SELECTED_FEATURE_INDICES
]


@dataclass
class LoggingConfig:
    """Logging settings for console and file output."""

    log_level: str = "INFO"
    log_file: Path = LOGS_DIR / "simulation.log"
    experiment_log_file: Path = LOGS_DIR / "experiments.log"
    log_to_console: bool = True
    log_to_file: bool = True


# ==============================================================================
# Global Default Config Bundle
# ==============================================================================
@dataclass
class QNavConfig:
    """Master configuration aggregating all sub-configurations."""

    sim: SimulationConfig = field(default_factory=SimulationConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Default global instance for easy import
DEFAULT_CONFIG = QNavConfig()
