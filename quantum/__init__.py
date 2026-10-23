"""Quantum navigation modules, circuit definitions, and variational models for Q-Nav."""

from quantum.feature_encoding import QuantumFeatureEncoder
from quantum.quantum_circuit import ParameterizedNavigationCircuit
from quantum.quantum_model import (
    VariationalQuantumNavigator,
    inspect_checkpoint_architecture,
)
from quantum.quantum_decision import QuantumDecisionMaker
from quantum.ibm_hardware import (
    get_ibm_service,
    list_available_hardware_backends,
    select_suitable_backend,
    execute_on_real_hardware,
    run_hardware_validation_benchmark,
)

__all__ = [
    "QuantumFeatureEncoder",
    "ParameterizedNavigationCircuit",
    "VariationalQuantumNavigator",
    "QuantumDecisionMaker",
    "inspect_checkpoint_architecture",
    "get_ibm_service",
    "list_available_hardware_backends",
    "select_suitable_backend",
    "execute_on_real_hardware",
    "run_hardware_validation_benchmark",
]
