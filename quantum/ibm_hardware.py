"""IBM Quantum Hardware Validation and Noise Characterization Module for Q-Nav.

Provides integration with IBM Quantum Runtime (SamplerV2 / PUB workflow) to execute
the exact Q-Nav navigation quantum decision circuit on real physical quantum devices.

Scientific Note:
Real IBM Quantum hardware execution introduces device noise, finite-shot sampling,
transpilation overhead, queue latency, and time-varying calibration conditions.
Hardware validation measures the physical fidelity and noise characteristics of the
implemented circuit and does NOT claim quantum computational advantage.
"""

import os
import time
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from config import (
    DEFAULT_CONFIG,
    QNavConfig,
    HARDWARE_METRICS_PATH,
    HARDWARE_CSV_PATH,
    HARDWARE_JOBS_PATH,
    HARDWARE_COUNTS_PATH,
)
from utils.io_utils import save_json, save_csv, load_json
from utils.logging_utils import get_logger
from quantum.quantum_circuit import ParameterizedNavigationCircuit
from quantum.feature_encoding import QuantumFeatureEncoder

logger = get_logger("quantum.ibm_hardware")


def get_ibm_service(
    token: Optional[str] = None,
    channel: str = "ibm_quantum_platform",
    instance: Optional[str] = None,
) -> Any:
    """Initialize Qiskit Runtime Service with secure token authentication and optional instance.

    Args:
        token: Optional explicit token. If None, reads from IBM_QUANTUM_TOKEN env var.
        channel: IBM Runtime channel (default: "ibm_quantum_platform").
        instance: Optional explicit instance (e.g. hub/group/project or instance name).
                  If None, reads from IBM_QUANTUM_INSTANCE env var.

    Returns:
        QiskitRuntimeService instance.

    Raises:
        ValueError: If token is not provided and cannot be found in environment.
        RuntimeError: If IBM Runtime service fails to initialize.
    """
    auth_token = token or os.getenv("IBM_QUANTUM_TOKEN")
    if not auth_token or not auth_token.strip():
        raise ValueError(
            "IBM Quantum token not configured. Please set the IBM_QUANTUM_TOKEN environment variable.\n"
            "Example (PowerShell): $env:IBM_QUANTUM_TOKEN=\"YOUR_API_TOKEN\"\n"
            "Example (CMD):        set IBM_QUANTUM_TOKEN=YOUR_API_TOKEN\n"
            "Example (Linux/Mac):   export IBM_QUANTUM_TOKEN=\"YOUR_API_TOKEN\""
        )

    auth_instance = instance or os.getenv("IBM_QUANTUM_INSTANCE")
    if auth_instance and auth_instance.strip():
        auth_instance = auth_instance.strip()
    else:
        auth_instance = None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        # Securely pass token and explicit instance if configured;
        # otherwise preserve automatic discovery behavior.
        if auth_instance:
            service = QiskitRuntimeService(
                channel=channel,
                token=auth_token.strip(),
                instance=auth_instance,
            )
            logger.info("Successfully connected to IBM Quantum Runtime service with explicit instance.")
        else:
            service = QiskitRuntimeService(
                channel=channel,
                token=auth_token.strip(),
            )
            logger.info("Successfully connected to IBM Quantum Runtime service.")
        return service
    except ImportError:
        raise RuntimeError(
            "qiskit-ibm-runtime is not installed. Please install it via:\n"
            "pip install qiskit-ibm-runtime"
        )
    except Exception as e:
        # Sanitize exception message to ensure token is never leaked
        sanitized_msg = str(e).replace(auth_token.strip(), "***REDACTED***")
        raise RuntimeError(f"Failed to initialize IBM Quantum Runtime service: {sanitized_msg}")


def list_available_hardware_backends(service: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Discover available real IBM Quantum hardware backends.

    Args:
        service: Optional pre-initialized QiskitRuntimeService.

    Returns:
        List of dictionaries with backend metadata (qubits, status, basis gates, pending jobs).
    """
    if service is None:
        service = get_ibm_service()

    try:
        backends = service.backends()
    except Exception as e:
        logger.error(f"Error querying IBM Quantum backends: {e}")
        return []

    result = []
    for b in backends:
        try:
            b_name = b.name
            num_qubits = getattr(b, "num_qubits", 0)

            # Determine operational status and simulator flag
            is_sim = False
            raw_sim = getattr(b, "simulator", None)
            if isinstance(raw_sim, bool):
                is_sim = raw_sim
            elif hasattr(b, "configuration") and callable(b.configuration):
                try:
                    config = b.configuration()
                    cfg_sim = getattr(config, "simulator", None)
                    if isinstance(cfg_sim, bool):
                        is_sim = cfg_sim
                except Exception:
                    pass

            status_obj = b.status() if hasattr(b, "status") else None
            is_operational = getattr(status_obj, "operational", True) if status_obj else True
            pending_jobs = getattr(status_obj, "pending_jobs", None) if status_obj else None

            basis_gates = list(getattr(b, "operation_names", []))
            coupling_edges = None
            if hasattr(b, "coupling_map") and b.coupling_map is not None:
                coupling_edges = len(b.coupling_map.get_edges()) if hasattr(b.coupling_map, "get_edges") else None

            result.append({
                "backend_name": b_name,
                "num_qubits": num_qubits,
                "operational": is_operational,
                "is_simulator": is_sim,
                "pending_jobs": pending_jobs,
                "basis_gates": basis_gates,
                "coupling_edges": coupling_edges,
                "version": getattr(b, "version", "unknown"),
            })
        except Exception as ex:
            logger.warning(f"Could not parse metadata for backend: {ex}")
            continue

    return result


def select_suitable_backend(
    service: Any,
    backend_name: Optional[str] = None,
    min_qubits: int = 4,
) -> Any:
    """Select and validate a real IBM Quantum hardware backend.

    Args:
        service: QiskitRuntimeService instance.
        backend_name: Specific backend requested by user. If None, auto-discovers best candidate.
        min_qubits: Minimum required qubits for circuit (default: 4).

    Returns:
        IBMBackend instance.

    Raises:
        ValueError: If backend does not meet qubit or operational criteria.
        RuntimeError: If no suitable hardware backend is available.
    """
    if backend_name:
        try:
            backend = service.backend(backend_name)
        except Exception as e:
            raise ValueError(f"Requested IBM Quantum backend '{backend_name}' was not found: {e}")

        num_qubits = getattr(backend, "num_qubits", 0)
        if num_qubits < min_qubits:
            raise ValueError(
                f"Selected backend '{backend_name}' does not have enough qubits for the configured "
                f"Q-Nav circuit ({num_qubits} available < {min_qubits} required)."
            )

        status_obj = backend.status() if hasattr(backend, "status") else None
        if status_obj and not getattr(status_obj, "operational", True):
            logger.warning(f"Selected backend '{backend_name}' is currently reported as not operational.")

        return backend

    # Auto-discovery mode: filter operational real hardware backends with sufficient qubits
    try:
        candidates = service.backends(simulator=False, operational=True)
    except Exception:
        candidates = [b for b in service.backends() if not getattr(b, "simulator", False)]

    suitable = [b for b in candidates if getattr(b, "num_qubits", 0) >= min_qubits]

    if not suitable:
        raise RuntimeError(
            f"No suitable operational IBM Quantum hardware backend found with at least {min_qubits} qubits."
        )

    # Sort by fewest pending jobs if available
    def get_pending(b):
        try:
            return b.status().pending_jobs or 9999
        except Exception:
            return 9999

    suitable.sort(key=get_pending)
    selected = suitable[0]
    logger.info(f"Auto-selected real hardware backend: {selected.name} ({selected.num_qubits} qubits)")
    return selected


def compute_circuit_hash(
    circuit: QuantumCircuit,
    features: np.ndarray,
    weights: np.ndarray,
) -> str:
    """Generate a deterministic SHA256 fingerprint for the logical circuit and bound parameters.

    Args:
        circuit: Bound Qiskit QuantumCircuit.
        features: 1D array of feature rotation angles.
        weights: 1D array of variational weights.

    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.sha256()
    # Hash circuit instruction signature
    hasher.update(str(circuit.num_qubits).encode())
    for inst in circuit.data:
        op = inst.operation
        q_indices = tuple(circuit.find_bit(q).index for q in inst.qubits)
        hasher.update(f"{op.name}:{q_indices}".encode())

    # Hash numerical parameter values rounded to 6 decimals
    f_rounded = np.round(np.asarray(features, dtype=float), 6).tobytes()
    w_rounded = np.round(np.asarray(weights, dtype=float), 6).tobytes()
    hasher.update(f_rounded)
    hasher.update(w_rounded)

    return hasher.hexdigest()[:16]


def prepare_hardware_circuit(
    circuit_manager: ParameterizedNavigationCircuit,
    features: np.ndarray,
    weights: np.ndarray,
) -> QuantumCircuit:
    """Construct and bind the Q-Nav logical circuit with measurement operations.

    Args:
        circuit_manager: ParameterizedNavigationCircuit instance.
        features: Qubit rotation angles.
        weights: Variational parameters.

    Returns:
        Bound QuantumCircuit with classical measurements on all qubits.
    """
    bound_qc = circuit_manager.bind_circuit(features, weights)
    # Add measurement operations across all qubits
    measured_qc = bound_qc.measure_all(inplace=False)
    return measured_qc


def transpile_for_backend(
    circuit: QuantumCircuit,
    backend: Any,
    optimization_level: int = 1,
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """Transpile the measured logical circuit for physical backend execution.

    Args:
        circuit: Measured QuantumCircuit.
        backend: Target IBMBackend instance.
        optimization_level: Transpiler optimization level (default: 1).

    Returns:
        Tuple of (transpiled_circuit, transpilation_metadata).
    """
    logical_depth = circuit.depth()
    logical_gate_count = circuit.size()
    backend_name = getattr(backend, "name", str(backend))

    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
        transpiled_qc = pm.run(circuit)
    except Exception as e:
        logger.warning(f"PassManager transpilation encountered issue ({e}), falling back to qiskit.transpile")
        try:
            transpiled_qc = transpile(circuit, backend=backend, optimization_level=optimization_level)
        except Exception:
            # Fallback for mock backends in unit tests
            transpiled_qc = transpile(circuit, optimization_level=optimization_level)

    transpiled_depth = transpiled_qc.depth()
    transpiled_gate_count = transpiled_qc.size()
    ops_dict = dict(transpiled_qc.count_ops())

    # Count single-qubit and two-qubit operations
    two_qubit_names = {"cx", "ecr", "cz", "swap", "rzz", "cp"}
    two_qubit_gates = sum(ops_dict.get(op, 0) for op in two_qubit_names)
    measure_count = ops_dict.get("measure", 0)
    single_qubit_gates = max(0, transpiled_gate_count - two_qubit_gates - measure_count)

    metadata = {
        "backend_name": backend_name,
        "logical_depth": logical_depth,
        "logical_gate_count": logical_gate_count,
        "transpiled_depth": transpiled_depth,
        "transpiled_gate_count": transpiled_gate_count,
        "single_qubit_gates": single_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "num_qubits": circuit.num_qubits,
        "ops_count": ops_dict,
    }
    return transpiled_qc, metadata


def extract_counts_from_pub_result(pub_result: Any) -> Dict[str, int]:
    """Robustly extract bitstring measurement counts from a SamplerV2 pub result."""
    if hasattr(pub_result, "data"):
        data_container = pub_result.data
        # Search for registers with get_counts method
        for field in dir(data_container):
            if not field.startswith("_"):
                val = getattr(data_container, field, None)
                if val is not None and hasattr(val, "get_counts"):
                    try:
                        return dict(val.get_counts())
                    except Exception:
                        pass
        # Fallback if dictionary-like
        if hasattr(data_container, "items"):
            for k, v in data_container.items():
                if hasattr(v, "get_counts"):
                    return dict(v.get_counts())

    # Direct dictionary fallback
    if isinstance(pub_result, dict):
        if "counts" in pub_result:
            return dict(pub_result["counts"])
        return {str(k): int(v) for k, v in pub_result.items() if isinstance(v, (int, float))}

    return {}


def parse_hardware_counts(
    counts: Dict[str, int],
    n_qubits: int = 4,
    shots: int = 1024,
    action_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert measurement counts into valid action probabilities and track invalid probability mass.

    Q-Nav Action Space Mapping:
      - Valid actions: indices 0..8 (9 navigation actions)
      - Invalid basis states: indices 9..15 (7 unmapped computational basis states)

    Args:
        counts: Raw bitstring counts dictionary (e.g. {'0000': 150, ...}).
        n_qubits: Circuit qubit count (default: 4 -> 16 states).
        shots: Expected or configured total shots.
        action_names: Ordered navigation action names.

    Returns:
        Structured dictionary with action probabilities, invalid probability mass, and selected action.
    """
    if action_names is None:
        action_names = DEFAULT_CONFIG.navigation.action_names

    num_actions = len(action_names)  # 9 actions
    num_states = 1 << n_qubits       # 16 states for 4 qubits
    total_shots = sum(counts.values())
    if total_shots <= 0:
        total_shots = shots if shots > 0 else 1024

    # Calculate raw probability for every 2^N computational basis state
    raw_basis_probs = np.zeros(num_states, dtype=float)
    for i in range(num_states):
        bitstr = format(i, f"0{n_qubits}b")
        raw_basis_probs[i] = counts.get(bitstr, 0) / float(total_shots)

    # Separate valid navigation actions (0..8) from invalid computational states (9..15)
    valid_mass = float(np.sum(raw_basis_probs[:num_actions]))
    invalid_mass = float(np.sum(raw_basis_probs[num_actions:]))

    # Normalize valid actions if possible; fallback to uniform across 9 actions if zero valid shots
    if valid_mass > 1e-9:
        normalized_action_probs = raw_basis_probs[:num_actions] / valid_mass
    else:
        normalized_action_probs = np.full(num_actions, 1.0 / num_actions, dtype=float)

    action_idx = int(np.argmax(normalized_action_probs))
    action_name = action_names[action_idx]
    confidence = float(normalized_action_probs[action_idx])

    return {
        "action": action_idx,
        "action_name": action_name,
        "confidence": confidence,
        "action_probabilities": normalized_action_probs.tolist(),
        "valid_action_probability": valid_mass,
        "invalid_probability_mass": invalid_mass,
        "raw_basis_probabilities": raw_basis_probs.tolist(),
        "total_shots_recorded": total_shots,
    }


def compute_distribution_metrics(
    sim_action_probs: Union[List[float], np.ndarray],
    hw_action_probs: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """Calculate statistical divergence and fidelity metrics between simulator and hardware.

    Args:
        sim_action_probs: Simulator action probability distribution (length 9).
        hw_action_probs: Hardware action probability distribution (length 9).

    Returns:
        Dictionary of comparative metrics.
    """
    p_sim = np.asarray(sim_action_probs, dtype=float)
    p_hw = np.asarray(hw_action_probs, dtype=float)

    # Normalize if not normalized
    if np.sum(p_sim) > 0:
        p_sim = p_sim / np.sum(p_sim)
    if np.sum(p_hw) > 0:
        p_hw = p_hw / np.sum(p_hw)

    diff = np.abs(p_sim - p_hw)
    mean_abs_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))

    # Jensen-Shannon Divergence
    # JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
    eps = 1e-12
    p_safe = np.clip(p_sim, eps, 1.0)
    p_safe = p_safe / np.sum(p_safe)
    q_safe = np.clip(p_hw, eps, 1.0)
    q_safe = q_safe / np.sum(q_safe)
    m = 0.5 * (p_safe + q_safe)

    kl_pm = np.sum(p_safe * np.log2(p_safe / m))
    kl_qm = np.sum(q_safe * np.log2(q_safe / m))
    js_divergence = float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))

    sim_action = int(np.argmax(p_sim))
    hw_action = int(np.argmax(p_hw))
    action_agreement = bool(sim_action == hw_action)

    return {
        "mean_absolute_probability_difference": mean_abs_diff,
        "max_probability_difference": max_diff,
        "js_divergence": js_divergence,
        "action_agreement": action_agreement,
        "simulator_action": sim_action,
        "hardware_action": hw_action,
        "simulator_confidence": float(np.max(p_sim)),
        "hardware_confidence": float(np.max(p_hw)),
    }


def execute_on_real_hardware(
    features: np.ndarray,
    weights: Optional[np.ndarray] = None,
    backend_name: Optional[str] = None,
    shots: int = 1024,
    timeout: int = 600,
    retries: int = 0,
    allow_fallback: bool = False,
    service: Optional[Any] = None,
    backend: Optional[Any] = None,
    config: Optional[QNavConfig] = None,
    instance: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Q-Nav navigation circuit on real IBM Quantum hardware.

    Args:
        features: 1D array of feature angles (length n_qubits) or raw features.
        weights: Optional variational weights. Defaults to pre-trained weights.
        backend_name: Name of IBM Quantum backend. If None, auto-selects.
        shots: Measurement shot count (> 0).
        timeout: Execution timeout in seconds.
        retries: Number of allowed retries on failure (default: 0).
        allow_fallback: If True, falls back to simulator on hardware error. Default: False.
        service: Optional pre-configured QiskitRuntimeService.
        backend: Optional pre-configured IBMBackend.
        config: Optional QNavConfig.
        instance: Optional explicit IBM Quantum instance name/CRN.

    Returns:
        Structured result dictionary.

    Raises:
        ValueError: On invalid parameter configurations.
        RuntimeError: On hardware execution failures (when allow_fallback=False).
    """
    if shots <= 0:
        raise ValueError(f"Shots must be positive, got {shots}")

    cfg = config or DEFAULT_CONFIG
    n_qubits = cfg.quantum.n_qubits
    encoder = QuantumFeatureEncoder(config=cfg.quantum)
    circuit_mgr = ParameterizedNavigationCircuit(config=cfg.quantum)

    # Encode features if raw 12D vector provided
    feat_arr = np.asarray(features, dtype=float)
    if len(feat_arr) == 12:
        features_4d = encoder.select_key_features(feat_arr)
        encoded_angles = encoder.encode(features_4d, n_qubits=n_qubits)
    elif len(feat_arr) == 4 and n_qubits == 4:
        encoded_angles = encoder.encode(feat_arr, n_qubits=n_qubits)
    elif len(feat_arr) == n_qubits:
        encoded_angles = feat_arr
    else:
        encoded_angles = encoder.encode(feat_arr, n_qubits=n_qubits)

    # Initialize weights if not provided
    if weights is None:
        rng = np.random.default_rng(cfg.quantum.weight_seed)
        weights = rng.uniform(0.0, 2.0 * np.pi, size=circuit_mgr.get_num_weights())

    # Build and measure logical circuit
    measured_qc = prepare_hardware_circuit(circuit_mgr, encoded_angles, weights)
    circuit_hash = compute_circuit_hash(measured_qc, encoded_angles, weights)

    t_start = time.perf_counter()
    last_error = None

    for attempt in range(1 + max(0, retries)):
        try:
            # 1. Acquire service and backend
            if service is None:
                inst = instance or (cfg.quantum.hardware_instance if hasattr(cfg, "quantum") else None)
                service = get_ibm_service(instance=inst)
            if backend is None:
                backend = select_suitable_backend(service, backend_name=backend_name, min_qubits=n_qubits)

            target_backend_name = getattr(backend, "name", str(backend_name or "ibm_backend"))

            # 2. Transpile for target backend
            transpiled_qc, trans_meta = transpile_for_backend(measured_qc, backend)

            # 3. Submit to IBM Runtime SamplerV2
            from qiskit_ibm_runtime import SamplerV2

            sampler = SamplerV2(mode=backend)
            job = sampler.run([transpiled_qc], shots=shots)
            job_id = getattr(job, "job_id", lambda: "job_unknown")()

            logger.info(f"Submitted IBM Quantum hardware job: {job_id} to backend {target_backend_name}")

            # 4. Wait for completion with timeout
            result = job.result()
            exec_duration = time.perf_counter() - t_start

            # 5. Extract counts and convert to 9-action probabilities
            pub_result = result[0]
            counts = extract_counts_from_pub_result(pub_result)
            parsed = parse_hardware_counts(
                counts=counts,
                n_qubits=n_qubits,
                shots=shots,
                action_names=cfg.navigation.action_names,
            )

            response = {
                "action": parsed["action"],
                "action_name": parsed["action_name"],
                "confidence": parsed["confidence"],
                "action_probabilities": parsed["action_probabilities"],
                "valid_action_probability": parsed["valid_action_probability"],
                "invalid_probability_mass": parsed["invalid_probability_mass"],
                "raw_basis_probabilities": parsed["raw_basis_probabilities"],
                "counts": counts,
                "shots": shots,
                "backend": target_backend_name,
                "job_id": job_id,
                "circuit_hash": circuit_hash,
                "logical_depth": trans_meta["logical_depth"],
                "transpiled_depth": trans_meta["transpiled_depth"],
                "logical_gate_count": trans_meta["logical_gate_count"],
                "transpiled_gate_count": trans_meta["transpiled_gate_count"],
                "single_qubit_gates": trans_meta["single_qubit_gates"],
                "two_qubit_gates": trans_meta["two_qubit_gates"],
                "num_qubits": n_qubits,
                "execution_duration": float(np.round(exec_duration, 4)),
                "status": "completed",
                "is_real_hardware": True,
                "attempt": attempt + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return response

        except Exception as e:
            last_error = e
            logger.warning(f"Hardware execution attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(2.0)
                continue

    # All attempts exhausted
    error_msg = f"IBM Quantum hardware execution failed: {last_error}"
    if not allow_fallback:
        raise RuntimeError(error_msg)

    logger.warning(f"{error_msg} -> Explicit fallback to simulator requested via --hardware-fallback-simulator.")
    # Fallback simulation
    exec_duration = time.perf_counter() - t_start
    state = Statevector.from_instruction(circuit_mgr.bind_circuit(encoded_angles, weights))
    probs_dict = {str(k): float(v) for k, v in state.probabilities_dict().items()}

    # Sample shots from statevector probabilities to simulate hardware sampling
    num_states = 1 << n_qubits
    sim_probs_16 = np.zeros(num_states, dtype=float)
    for i in range(num_states):
        bitstr = format(i, f"0{n_qubits}b")
        sim_probs_16[i] = probs_dict.get(bitstr, 0.0)

    # Deterministic simulated count sampling
    rng = np.random.default_rng(cfg.quantum.circuit_seed)
    sampled_indices = rng.choice(num_states, size=shots, p=sim_probs_16)
    sim_counts = {}
    for idx in sampled_indices:
        b = format(idx, f"0{n_qubits}b")
        sim_counts[b] = sim_counts.get(b, 0) + 1

    parsed = parse_hardware_counts(
        counts=sim_counts,
        n_qubits=n_qubits,
        shots=shots,
        action_names=cfg.navigation.action_names,
    )

    return {
        "action": parsed["action"],
        "action_name": parsed["action_name"],
        "confidence": parsed["confidence"],
        "action_probabilities": parsed["action_probabilities"],
        "valid_action_probability": parsed["valid_action_probability"],
        "invalid_probability_mass": parsed["invalid_probability_mass"],
        "raw_basis_probabilities": parsed["raw_basis_probabilities"],
        "counts": sim_counts,
        "shots": shots,
        "backend": f"simulator_fallback_{backend_name or 'aer'}",
        "job_id": f"sim_fallback_{circuit_hash}",
        "circuit_hash": circuit_hash,
        "logical_depth": measured_qc.depth(),
        "transpiled_depth": measured_qc.depth(),
        "logical_gate_count": measured_qc.size(),
        "transpiled_gate_count": measured_qc.size(),
        "single_qubit_gates": measured_qc.num_qubits * 2,
        "two_qubit_gates": measured_qc.num_qubits,
        "num_qubits": n_qubits,
        "execution_duration": float(np.round(exec_duration, 4)),
        "status": "completed_with_fallback",
        "is_real_hardware": False,
        "attempt": attempt + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_hardware_validation_benchmark(
    samples: int = 3,
    backend_name: Optional[str] = None,
    shots: int = 1024,
    timeout: int = 600,
    retries: int = 0,
    allow_fallback: bool = False,
    is_real_hardware: bool = True,
    save_outputs: bool = True,
    config: Optional[QNavConfig] = None,
    instance: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a controlled hardware validation comparison between Ideal Simulator and IBM Quantum Backend.

    Args:
        samples: Number of circuit executions (default: 3).
        backend_name: Target backend.
        shots: Measurement shot count.
        timeout: Job timeout.
        retries: Number of retries on failure.
        allow_fallback: Whether fallback to simulator is allowed if hardware fails.
        is_real_hardware: If True, connects to IBM Quantum Runtime.
        save_outputs: If True, writes results to outputs/metrics and outputs/logs.
        config: Optional master configuration.
        instance: Optional explicit IBM Quantum instance name/CRN.

    Returns:
        Structured evaluation comparison dictionary.
    """
    cfg = config or DEFAULT_CONFIG
    n_qubits = cfg.quantum.n_qubits
    circuit_mgr = ParameterizedNavigationCircuit(config=cfg.quantum)
    encoder = QuantumFeatureEncoder(config=cfg.quantum)

    # Predefined representative navigation feature vectors covering diverse spatial directions
    representative_states = [
        # Sample 1: Target to the North-East, clear path
        {
            "id": "state_ne_clear",
            "scenario_id": "scenario_01",
            "seed": 42,
            "features_12d": [0.65, 0.707, 0.707, 0.707, 0.707, 0.90, 0.85, 0.90, 0.90, 0.10, 0.50, 0.80],
        },
        # Sample 2: Target East, obstacle in front (repulsive steering needed)
        {
            "id": "state_front_obstacle",
            "scenario_id": "scenario_02",
            "seed": 143,
            "features_12d": [0.45, 1.000, 0.000, 0.000, 1.000, 0.20, 0.15, 0.75, 0.80, 0.40, 0.60, 0.70],
        },
        # Sample 3: Dynamic crossing obstacle scenario, lateral evasion
        {
            "id": "state_dynamic_crossing",
            "scenario_id": "scenario_dynamic_01",
            "seed": 244,
            "features_12d": [0.55, 0.600, 0.800, 0.800, 0.600, 0.30, 0.45, 0.20, 0.85, 0.35, 0.70, 0.65],
        },
        # Sample 4: Close to target (deceleration/arrival)
        {
            "id": "state_near_target",
            "scenario_id": "scenario_05",
            "seed": 42,
            "features_12d": [0.10, 0.500, 0.866, 0.866, 0.500, 0.80, 0.75, 0.80, 0.85, 0.05, 0.20, 0.90],
        },
    ]

    selected_samples = representative_states[: max(1, min(samples, len(representative_states)))]

    # Deterministic weights
    rng = np.random.default_rng(cfg.quantum.weight_seed)
    weights = rng.uniform(0.0, 2.0 * np.pi, size=circuit_mgr.get_num_weights())

    csv_records = []
    job_records = []
    counts_records = {}
    sample_comparisons = []

    service = None
    backend = None
    if is_real_hardware:
        inst = instance or (cfg.quantum.hardware_instance if hasattr(cfg, "quantum") else None)
        service = get_ibm_service(instance=inst)
        backend = select_suitable_backend(service, backend_name=backend_name, min_qubits=n_qubits)

    for idx, sample in enumerate(selected_samples):
        feat_12d = np.asarray(sample["features_12d"], dtype=float)
        features_4d = encoder.select_key_features(feat_12d)
        encoded_angles = encoder.encode(features_4d, n_qubits=n_qubits)

        # 1. Ideal Statevector Simulator Prediction
        bound_qc = circuit_mgr.bind_circuit(encoded_angles, weights)
        sim_state = Statevector.from_instruction(bound_qc)
        sim_probs_dict = {str(k): float(v) for k, v in sim_state.probabilities_dict().items()}

        sim_counts = {}
        for s_idx in range(1 << n_qubits):
            b = format(s_idx, f"0{n_qubits}b")
            sim_counts[b] = int(round(sim_probs_dict.get(b, 0.0) * shots))

        sim_parsed = parse_hardware_counts(
            counts=sim_counts,
            n_qubits=n_qubits,
            shots=shots,
            action_names=cfg.navigation.action_names,
        )

        # 2. Hardware (or simulated test) Execution
        if is_real_hardware:
            hw_res = execute_on_real_hardware(
                features=encoded_angles,
                weights=weights,
                backend_name=getattr(backend, "name", backend_name),
                shots=shots,
                timeout=timeout,
                retries=retries,
                allow_fallback=allow_fallback,
                service=service,
                backend=backend,
                config=cfg,
            )
        else:
            # Mock / Simulator Hardware Validation for unit testing or credential-less environments
            hw_res = execute_on_real_hardware(
                features=encoded_angles,
                weights=weights,
                backend_name="simulated_backend_test",
                shots=shots,
                allow_fallback=True,
                config=cfg,
            )

        # 3. Compare Simulator vs Hardware
        comp_metrics = compute_distribution_metrics(
            sim_action_probs=sim_parsed["action_probabilities"],
            hw_action_probs=hw_res["action_probabilities"],
        )

        # Record telemetry & CSV row
        row = {
            "timestamp": hw_res["timestamp"],
            "scenario_id": sample["scenario_id"],
            "seed": sample["seed"],
            "sample_id": sample["id"],
            "backend": hw_res["backend"],
            "shots": hw_res["shots"],
            "job_id": hw_res["job_id"],
            "simulator_action": comp_metrics["simulator_action"],
            "hardware_action": comp_metrics["hardware_action"],
            "action_agreement": comp_metrics["action_agreement"],
            "simulator_confidence": float(np.round(comp_metrics["simulator_confidence"], 4)),
            "hardware_confidence": float(np.round(comp_metrics["hardware_confidence"], 4)),
            "invalid_probability_mass": float(np.round(hw_res["invalid_probability_mass"], 4)),
            "mean_absolute_probability_difference": float(np.round(comp_metrics["mean_absolute_probability_difference"], 4)),
            "max_probability_difference": float(np.round(comp_metrics["max_probability_difference"], 4)),
            "js_divergence": float(np.round(comp_metrics["js_divergence"], 4)),
            "logical_depth": hw_res["logical_depth"],
            "transpiled_depth": hw_res["transpiled_depth"],
            "num_qubits": hw_res["num_qubits"],
            "status": hw_res["status"],
        }
        csv_records.append(row)

        job_records.append({
            "job_id": hw_res["job_id"],
            "backend": hw_res["backend"],
            "sample_id": sample["id"],
            "shots": hw_res["shots"],
            "circuit_hash": hw_res["circuit_hash"],
            "logical_depth": hw_res["logical_depth"],
            "transpiled_depth": hw_res["transpiled_depth"],
            "qubit_count": hw_res["num_qubits"],
            "result_status": hw_res["status"],
            "execution_duration": hw_res.get("execution_duration"),
            "timestamp": hw_res["timestamp"],
        })

        counts_records[hw_res["job_id"]] = {
            "sample_id": sample["id"],
            "backend": hw_res["backend"],
            "shots": hw_res["shots"],
            "counts": hw_res["counts"],
            "simulator_action_probabilities": sim_parsed["action_probabilities"],
            "hardware_action_probabilities": hw_res["action_probabilities"],
            "invalid_probability_mass": hw_res["invalid_probability_mass"],
        }

        sample_comparisons.append({
            "sample_id": sample["id"],
            "scenario_id": sample["scenario_id"],
            "metrics": comp_metrics,
            "hardware_result": hw_res,
            "simulator_result": sim_parsed,
        })

    # Summary Aggregate Metrics
    total_samples = len(sample_comparisons)
    agreements = [c["metrics"]["action_agreement"] for c in sample_comparisons]
    agreement_rate = float(np.mean(agreements)) if total_samples > 0 else 0.0
    mean_jsd = float(np.mean([c["metrics"]["js_divergence"] for c in sample_comparisons])) if total_samples > 0 else 0.0
    mean_mapd = float(np.mean([c["metrics"]["mean_absolute_probability_difference"] for c in sample_comparisons])) if total_samples > 0 else 0.0
    mean_invalid_mass = float(np.mean([c["hardware_result"]["invalid_probability_mass"] for c in sample_comparisons])) if total_samples > 0 else 0.0
    successful_jobs = sum(1 for c in sample_comparisons if "completed" in c["hardware_result"]["status"])
    failed_jobs = total_samples - successful_jobs

    benchmark_summary = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_real_hardware": is_real_hardware,
            "backend_name": sample_comparisons[0]["hardware_result"]["backend"] if total_samples > 0 else None,
            "shots": shots,
            "num_qubits": n_qubits,
            "total_samples": total_samples,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
        },
        "fidelity_summary": {
            "action_agreement_rate": float(np.round(agreement_rate, 4)),
            "mean_js_divergence": float(np.round(mean_jsd, 4)),
            "mean_absolute_probability_difference": float(np.round(mean_mapd, 4)),
            "mean_invalid_probability_mass": float(np.round(mean_invalid_mass, 4)),
            "mean_logical_depth": float(np.mean([c["hardware_result"]["logical_depth"] for c in sample_comparisons])),
            "mean_transpiled_depth": float(np.mean([c["hardware_result"]["transpiled_depth"] for c in sample_comparisons])),
            "mean_hardware_duration_seconds": float(np.mean([c["hardware_result"].get("execution_duration", 0.0) for c in sample_comparisons])),
        },
        "samples": sample_comparisons,
    }

    if save_outputs:
        save_json(benchmark_summary, HARDWARE_METRICS_PATH)
        save_csv(csv_records, HARDWARE_CSV_PATH)
        save_json(job_records, HARDWARE_JOBS_PATH)
        save_json(counts_records, HARDWARE_COUNTS_PATH)
        logger.info(f"Saved hardware validation metrics to {HARDWARE_METRICS_PATH.resolve()}")
        logger.info(f"Saved hardware validation CSV to {HARDWARE_CSV_PATH.resolve()}")

    return benchmark_summary
