"""Unit tests for IBM Quantum Hardware integration in Q-Nav.

Tests verify IBM service initialization, backend selection, circuit transpilation,
count-to-probability parsing, invalid basis state separation, divergence metrics,
and error handling without requiring live IBM Quantum credentials (using mocks).
"""

import os
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from qiskit import QuantumCircuit

from config import DEFAULT_CONFIG, QNavConfig
from quantum.ibm_hardware import (
    get_ibm_service,
    list_available_hardware_backends,
    select_suitable_backend,
    compute_circuit_hash,
    prepare_hardware_circuit,
    transpile_for_backend,
    parse_hardware_counts,
    compute_distribution_metrics,
    execute_on_real_hardware,
    run_hardware_validation_benchmark,
)
from quantum.quantum_circuit import ParameterizedNavigationCircuit


class TestIBMServiceAndAuth:
    """Test IBM Quantum service initialization and authentication rules."""

    def test_token_missing_raises_error(self, monkeypatch):
        """Verify that missing token raises clear configuration error."""
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        with pytest.raises(ValueError, match="IBM Quantum token not configured"):
            get_ibm_service(token=None)

    def test_empty_token_raises_error(self, monkeypatch):
        """Verify that whitespace/empty token raises clear configuration error."""
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "   ")
        with pytest.raises(ValueError, match="IBM Quantum token not configured"):
            get_ibm_service(token=None)

    def test_no_token_leakage_in_error_message(self, monkeypatch):
        """Verify secret token is never printed in error messages."""
        secret_token = "secret_ibm_api_token_12345_do_not_leak"
        with patch("qiskit_ibm_runtime.QiskitRuntimeService", side_effect=Exception(f"Connection error with {secret_token}")):
            with pytest.raises(RuntimeError) as exc_info:
                get_ibm_service(token=secret_token)
            assert secret_token not in str(exc_info.value)
            assert "***REDACTED***" in str(exc_info.value)

    def test_service_initialization_channel(self, monkeypatch):
        """Verify that QiskitRuntimeService is initialized with channel='ibm_quantum_platform'."""
        monkeypatch.delenv("IBM_QUANTUM_INSTANCE", raising=False)
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as mock_service_cls:
            get_ibm_service(token="mock_valid_token_xyz")
            mock_service_cls.assert_called_once_with(
                channel="ibm_quantum_platform",
                token="mock_valid_token_xyz",
            )

    def test_service_initialization_with_explicit_instance(self):
        """Verify that QiskitRuntimeService receives explicit instance when provided."""
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as mock_service_cls:
            get_ibm_service(token="mock_valid_token_xyz", instance="Q-Nav")
            mock_service_cls.assert_called_once_with(
                channel="ibm_quantum_platform",
                token="mock_valid_token_xyz",
                instance="Q-Nav",
            )

    def test_service_initialization_with_env_instance(self, monkeypatch):
        """Verify that QiskitRuntimeService reads instance from IBM_QUANTUM_INSTANCE env var."""
        monkeypatch.setenv("IBM_QUANTUM_INSTANCE", "hub/group/project")
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as mock_service_cls:
            get_ibm_service(token="mock_valid_token_xyz")
            mock_service_cls.assert_called_once_with(
                channel="ibm_quantum_platform",
                token="mock_valid_token_xyz",
                instance="hub/group/project",
            )

    def test_service_initialization_without_instance_preserves_auto_discovery(self, monkeypatch):
        """Verify that QiskitRuntimeService preserves auto-discovery when no instance is configured."""
        monkeypatch.delenv("IBM_QUANTUM_INSTANCE", raising=False)
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as mock_service_cls:
            get_ibm_service(token="mock_valid_token_xyz", instance=None)
            mock_service_cls.assert_called_once_with(
                channel="ibm_quantum_platform",
                token="mock_valid_token_xyz",
            )


class TestBackendDiscoveryAndValidation:
    """Test backend querying, filtering, and validation."""

    def test_list_available_backends(self):
        """Verify backend metadata collection without credentials leakage."""
        mock_backend_1 = MagicMock()
        mock_backend_1.name = "ibm_fez"
        mock_backend_1.num_qubits = 156
        mock_backend_1.simulator = False
        mock_backend_1.status().operational = True
        mock_backend_1.status().pending_jobs = 12
        mock_backend_1.operation_names = ["cx", "rz", "sx", "x", "measure"]
        mock_backend_1.coupling_map.get_edges.return_value = [(0, 1), (1, 2)]
        mock_backend_1.version = "1.0.0"

        mock_backend_sim = MagicMock()
        mock_backend_sim.name = "simulator_aer"
        mock_backend_sim.num_qubits = 32
        mock_backend_sim.simulator = True
        mock_backend_sim.status().operational = True
        mock_backend_sim.status().pending_jobs = 0
        mock_backend_sim.operation_names = ["all"]
        mock_backend_sim.coupling_map = None
        mock_backend_sim.version = "0.2.0"

        mock_service = MagicMock()
        mock_service.backends.return_value = [mock_backend_1, mock_backend_sim]

        backends = list_available_hardware_backends(service=mock_service)
        assert len(backends) == 2
        assert backends[0]["backend_name"] == "ibm_fez"
        assert backends[0]["num_qubits"] == 156
        assert backends[0]["operational"] is True
        assert backends[0]["is_simulator"] is False
        assert backends[0]["pending_jobs"] == 12

        assert backends[1]["backend_name"] == "simulator_aer"
        assert backends[1]["is_simulator"] is True

    def test_insufficient_qubits_rejection(self):
        """Verify error when selected backend has fewer qubits than required by circuit."""
        mock_backend = MagicMock()
        mock_backend.name = "ibm_tiny"
        mock_backend.num_qubits = 2  # Circuit requires 4 qubits

        mock_service = MagicMock()
        mock_service.backend.return_value = mock_backend

        with pytest.raises(ValueError, match="does not have enough qubits"):
            select_suitable_backend(service=mock_service, backend_name="ibm_tiny", min_qubits=4)

    def test_missing_backend_raises_error(self):
        """Verify error when requested backend does not exist."""
        mock_service = MagicMock()
        mock_service.backend.side_effect = Exception("Backend not found")

        with pytest.raises(ValueError, match="was not found"):
            select_suitable_backend(service=mock_service, backend_name="nonexistent_backend", min_qubits=4)


class TestCircuitHashAndPreparation:
    """Test deterministic circuit hashing and transpilation metadata."""

    def test_circuit_hash_determinism(self):
        """Verify identical circuit parameters produce identical hash."""
        circuit_mgr = ParameterizedNavigationCircuit(config=DEFAULT_CONFIG.quantum)
        angles = np.array([0.5, -1.2, 0.8, -0.3])
        rng = np.random.default_rng(42)
        weights = rng.uniform(0.0, 2.0 * np.pi, size=circuit_mgr.get_num_weights())

        qc1 = prepare_hardware_circuit(circuit_mgr, angles, weights)
        qc2 = prepare_hardware_circuit(circuit_mgr, angles, weights)

        hash1 = compute_circuit_hash(qc1, angles, weights)
        hash2 = compute_circuit_hash(qc2, angles, weights)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_circuit_hash_sensitivity(self):
        """Verify changing angle parameters changes the circuit hash."""
        circuit_mgr = ParameterizedNavigationCircuit(config=DEFAULT_CONFIG.quantum)
        angles1 = np.array([0.5, -1.2, 0.8, -0.3])
        angles2 = np.array([0.5, -1.2, 0.8, 0.3])  # altered sign
        rng = np.random.default_rng(42)
        weights = rng.uniform(0.0, 2.0 * np.pi, size=circuit_mgr.get_num_weights())

        qc1 = prepare_hardware_circuit(circuit_mgr, angles1, weights)
        qc2 = prepare_hardware_circuit(circuit_mgr, angles2, weights)

        hash1 = compute_circuit_hash(qc1, angles1, weights)
        hash2 = compute_circuit_hash(qc2, angles2, weights)
        assert hash1 != hash2


class TestCountParsingAndInvalidStateSeparation:
    """Test mapping 16 computational basis states to 9 valid actions and tracking invalid mass."""

    def test_valid_and_invalid_state_separation(self):
        """Verify basis states 0..8 map to valid actions and 9..15 map to invalid probability mass."""
        # 4 qubits -> 16 basis states ('0000' to '1111')
        counts = {
            "0000": 100,  # action 0 (UP)
            "0001": 100,  # action 1 (UP_RIGHT)
            "0010": 100,  # action 2 (RIGHT)
            "0011": 100,  # action 3 (DOWN_RIGHT)
            "0100": 100,  # action 4 (DOWN)
            "0101": 100,  # action 5 (DOWN_LEFT)
            "0110": 100,  # action 6 (LEFT)
            "0111": 100,  # action 7 (UP_LEFT)
            "1000": 200,  # action 8 (STAY) -> highest count
            # Invalid states (9 through 15)
            "1001": 200,  # state 9
            "1010": 100,  # state 10
            "1111": 200,  # state 15
        }
        # Total valid shots = 1000, Total invalid shots = 500, Total = 1500
        parsed = parse_hardware_counts(counts, n_qubits=4, shots=1500)

        assert parsed["total_shots_recorded"] == 1500
        assert parsed["action"] == 8  # STAY had 200 counts (highest among 0..8)
        assert parsed["action_name"] == "STAY"
        assert len(parsed["action_probabilities"]) == 9
        # Valid mass = 1000/1500 = 2/3
        assert np.isclose(parsed["valid_action_probability"], 1000.0 / 1500.0)
        # Invalid mass = 500/1500 = 1/3
        assert np.isclose(parsed["invalid_probability_mass"], 500.0 / 1500.0)
        # Sum of normalized valid action probabilities must be 1.0
        assert np.isclose(sum(parsed["action_probabilities"]), 1.0)
        # Normalized STAY probability: 200 / 1000 = 0.20
        assert np.isclose(parsed["action_probabilities"][8], 0.20)

    def test_extreme_all_invalid_fallback(self):
        """Verify robust uniform fallback if noise places 100% shots into invalid basis states."""
        counts = {
            "1111": 1024,  # state 15 (invalid)
        }
        parsed = parse_hardware_counts(counts, n_qubits=4, shots=1024)
        assert np.isclose(parsed["invalid_probability_mass"], 1.0)
        assert np.isclose(parsed["valid_action_probability"], 0.0)
        # Must return valid action within 0..8 (never None or > 8)
        assert 0 <= parsed["action"] <= 8
        assert np.isclose(sum(parsed["action_probabilities"]), 1.0)
        assert np.isclose(parsed["action_probabilities"][0], 1.0 / 9.0)


class TestDivergenceAndNoiseMetrics:
    """Test Jensen-Shannon divergence, mean absolute error, and agreement calculations."""

    def test_identical_distributions_yield_zero_divergence(self):
        """Verify identical simulator and hardware distributions yield 0 JS divergence."""
        probs = [0.1, 0.05, 0.15, 0.2, 0.1, 0.05, 0.1, 0.05, 0.2]
        metrics = compute_distribution_metrics(probs, probs)
        assert np.isclose(metrics["mean_absolute_probability_difference"], 0.0)
        assert np.isclose(metrics["max_probability_difference"], 0.0)
        assert np.isclose(metrics["js_divergence"], 0.0)
        assert metrics["action_agreement"] is True

    def test_orthogonal_distributions_yield_positive_divergence(self):
        """Verify distinct distributions yield non-zero JS divergence and detect disagreement."""
        p_sim = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p_hw = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        metrics = compute_distribution_metrics(p_sim, p_hw)
        assert metrics["action_agreement"] is False
        assert metrics["js_divergence"] > 0.0
        assert metrics["simulator_action"] == 0
        assert metrics["hardware_action"] == 2


class TestHardwareExecutionAndFallbackRules:
    """Test hardware execution flow, mock runtime jobs, and strict fallback constraints."""

    def test_shot_validation_rejects_non_positive_shots(self):
        """Verify negative or zero shot counts are rejected."""
        with pytest.raises(ValueError, match="Shots must be positive"):
            execute_on_real_hardware(
                features=np.array([0.1, 0.2, 0.3, 0.4]),
                shots=0,
            )

    def test_no_silent_fallback_by_default(self):
        """Verify that when real hardware fails and allow_fallback=False, it raises RuntimeError."""
        with patch("quantum.ibm_hardware.get_ibm_service", side_effect=Exception("IBM Cloud down")):
            with pytest.raises(RuntimeError, match="IBM Quantum hardware execution failed"):
                execute_on_real_hardware(
                    features=np.array([0.1, 0.2, 0.3, 0.4]),
                    shots=1024,
                    allow_fallback=False,  # Strict: no silent fallback
                )

    def test_explicit_fallback_permitted_when_flagged(self):
        """Verify fallback to simulation only occurs when allow_fallback=True is explicitly set."""
        with patch("quantum.ibm_hardware.get_ibm_service", side_effect=Exception("Backend queue full")):
            res = execute_on_real_hardware(
                features=np.array([0.1, 0.2, 0.3, 0.4]),
                shots=512,
                allow_fallback=True,  # Explicit opt-in fallback
            )
            assert res["status"] == "completed_with_fallback"
            assert res["is_real_hardware"] is False
            assert "simulator_fallback" in res["backend"]
            assert len(res["action_probabilities"]) == 9
            assert 0 <= res["action"] <= 8

    def test_successful_mocked_hardware_execution(self):
        """Verify full execution pipeline with mocked Qiskit Runtime SamplerV2."""
        mock_backend = MagicMock()
        mock_backend.name = "ibm_mock_device"
        mock_backend.num_qubits = 127
        mock_backend.status().operational = True

        mock_service = MagicMock()
        mock_service.backend.return_value = mock_backend

        # Mock SamplerV2 and PubResult
        mock_pub_res = MagicMock()
        mock_data = MagicMock()
        mock_reg = MagicMock()
        mock_reg.get_counts.return_value = {"0000": 300, "0010": 500, "1111": 224}
        mock_data.meas = mock_reg
        mock_pub_res.data = mock_data

        mock_job = MagicMock()
        mock_job.job_id.return_value = "job_mock_12345"
        mock_job.result.return_value = [mock_pub_res]

        with patch("qiskit_ibm_runtime.SamplerV2") as MockSampler:
            sampler_instance = MockSampler.return_value
            sampler_instance.run.return_value = mock_job

            res = execute_on_real_hardware(
                features=np.array([0.1, 0.2, 0.3, 0.4]),
                backend_name="ibm_mock_device",
                shots=1024,
                service=mock_service,
                backend=mock_backend,
                allow_fallback=False,
            )

            assert res["status"] == "completed"
            assert res["is_real_hardware"] is True
            assert res["backend"] == "ibm_mock_device"
            assert res["job_id"] == "job_mock_12345"
            assert res["action"] == 2  # '0010' had 500 counts (highest valid)
            assert res["action_name"] == "RIGHT"
            assert np.isclose(res["invalid_probability_mass"], 224.0 / 1024.0)


class TestHardwareValidationBenchmark:
    """Test full multi-sample validation benchmark execution."""

    def test_simulated_hardware_validation_benchmark_output_structure(self):
        """Verify benchmark generates expected summary structure and output metrics."""
        summary = run_hardware_validation_benchmark(
            samples=2,
            shots=256,
            is_real_hardware=False,
            allow_fallback=True,
            save_outputs=False,
        )
        assert "metadata" in summary
        assert "fidelity_summary" in summary
        assert "samples" in summary
        assert summary["metadata"]["total_samples"] == 2
        assert 0.0 <= summary["fidelity_summary"]["action_agreement_rate"] <= 1.0
        assert summary["fidelity_summary"]["mean_js_divergence"] >= 0.0
        assert summary["fidelity_summary"]["mean_invalid_probability_mass"] >= 0.0
