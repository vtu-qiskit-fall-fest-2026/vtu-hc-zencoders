"""Parameterized Quantum Circuit Module for Q-Nav.

Builds and executes parameterized quantum circuits using Qiskit 2.x,
featuring angle encoding, circular entanglement, and variational rotation layers.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from config import QuantumConfig, DEFAULT_CONFIG


class ParameterizedNavigationCircuit:
    """Manages the Qiskit QuantumCircuit architecture for UAV navigation."""

    def __init__(
        self,
        config: Optional[QuantumConfig] = None,
        n_qubits: Optional[int] = None,
        n_layers: Optional[int] = None,
    ):
        """Initialize quantum circuit parameters.

        Args:
            config: Quantum configuration parameters.
            n_qubits: Optional override for number of qubits.
            n_layers: Optional override for number of variational layers.
        """
        self.config = config or DEFAULT_CONFIG.quantum
        self.n_qubits = n_qubits if n_qubits is not None else self.config.n_qubits
        self.n_layers = n_layers if n_layers is not None else self.config.n_layers
        self.backend_name = self.config.backend_name
        self.shots = self.config.shots

        # Build parameterized template circuit
        self._build_template()

        # Cache Pauli-Z observables for expectation value calculation
        self._build_observables()

    def _build_template(self) -> None:
        """Construct the symbolic parameterized quantum circuit template."""
        self.circuit = QuantumCircuit(self.n_qubits)

        # Feature parameter vector (one rotation angle per qubit)
        self.feature_params = ParameterVector("x", self.n_qubits)

        # Layer 1: Feature encoding via Ry and Rz gates
        for i in range(self.n_qubits):
            self.circuit.ry(self.feature_params[i], i)
            self.circuit.rz(self.feature_params[i] * 0.5, i)

        # Variational parameters: 2 angles (Ry, Rz) per qubit per layer
        total_weights = self.n_layers * self.n_qubits * 2
        self.weight_params = ParameterVector("w", total_weights)

        # Variational layers with circular CNOT entanglement
        param_idx = 0
        for layer in range(self.n_layers):
            # Circular entanglement ladder
            for i in range(self.n_qubits):
                target_qubit = (i + 1) % self.n_qubits
                self.circuit.cx(i, target_qubit)

            # Parameterized rotations
            for i in range(self.n_qubits):
                self.circuit.ry(self.weight_params[param_idx], i)
                param_idx += 1
                self.circuit.rz(self.weight_params[param_idx], i)
                param_idx += 1

    def _build_observables(self) -> None:
        """Construct Pauli-Z observables for single-qubit expectation values."""
        self.z_observables = []
        for i in range(self.n_qubits):
            # Qiskit qubit ordering: rightmost character is qubit 0
            pauli_str = ["I"] * self.n_qubits
            pauli_str[self.n_qubits - 1 - i] = "Z"
            self.z_observables.append(SparsePauliOp("".join(pauli_str)))

    def get_num_weights(self) -> int:
        """Return the number of variational weight parameters needed."""
        return len(self.weight_params)

    def bind_circuit(
        self,
        features: np.ndarray,
        weights: np.ndarray,
    ) -> QuantumCircuit:
        """Bind numerical values for feature angles and variational weights.

        Args:
            features: 1D array of rotation angles (length == n_qubits).
            weights: 1D array of variational weights (length == get_num_weights()).

        Returns:
            Bound Qiskit QuantumCircuit ready for execution.
        """
        if len(features) != self.n_qubits:
            raise ValueError(
                f"Feature vector length {len(features)} does not match n_qubits {self.n_qubits}"
            )
        expected_weights = self.get_num_weights()
        if len(weights) != expected_weights:
            raise ValueError(
                f"Weights length {len(weights)} does not match expected {expected_weights}"
            )

        param_dict = {}
        for i in range(self.n_qubits):
            param_dict[self.feature_params[i]] = float(features[i])
        for j in range(expected_weights):
            param_dict[self.weight_params[j]] = float(weights[j])

        return self.circuit.assign_parameters(param_dict)

    def execute(
        self,
        features: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Simulate the bound quantum circuit and return quantum observables.

        Args:
            features: Qubit rotation angles.
            weights: Variational parameters.

        Returns:
            Tuple of:
            - expectation_values: 1D array of <Z_i> for each qubit (range [-1, 1])
            - basis_probabilities: 1D array of 2^n probabilities
            - probs_dict: Dict mapping bitstring (e.g. '0010') to probability
        """
        bound_qc = self.bind_circuit(features, weights)

        # Exact statevector simulation for high speed and reproducibility
        state = Statevector.from_instruction(bound_qc)

        # Compute Pauli-Z expectation values <psi|Z_i|psi>
        expectations = np.array(
            [float(state.expectation_value(op).real) for op in self.z_observables],
            dtype=float,
        )

        # Compute computational basis state probabilities
        probs_dict = {str(k): float(v) for k, v in state.probabilities_dict().items()}

        # 1D array of all 2^N state probabilities ordered numerically (0 to 2^N - 1)
        num_states = 1 << self.n_qubits
        basis_probs = np.zeros(num_states, dtype=float)
        for i in range(num_states):
            bitstr = format(i, f"0{self.n_qubits}b")
            basis_probs[i] = probs_dict.get(bitstr, 0.0)

        return expectations, basis_probs, probs_dict


VariationalQuantumCircuit = ParameterizedNavigationCircuit
