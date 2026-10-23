"""Opt-in live IBM Quantum hardware validation tests.

This test requires active network connectivity and valid IBM Quantum credentials
configured in the environment variable IBM_QUANTUM_TOKEN.

Run explicitly with:
    python -m pytest tests/test_ibm_hardware_live.py -v

This test is automatically SKIPPED if IBM_QUANTUM_TOKEN is not configured.
"""

import os
import pytest
import numpy as np

from quantum.ibm_hardware import (
    get_ibm_service,
    list_available_hardware_backends,
    select_suitable_backend,
    execute_on_real_hardware,
)

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
SKIP_REASON = "Live IBM Quantum credentials not configured in IBM_QUANTUM_TOKEN."


@pytest.mark.skipif(not TOKEN or not TOKEN.strip(), reason=SKIP_REASON)
class TestLiveIBMHardwareExecution:
    """Live IBM Quantum hardware execution tests (opt-in only)."""

    def test_live_service_connection(self):
        """Verify real connection to IBM Quantum Runtime service."""
        service = get_ibm_service()
        assert service is not None
        backends = list_available_hardware_backends(service=service)
        assert isinstance(backends, list)

    def test_live_small_circuit_execution(self):
        """Execute a minimal single-decision test on an available real hardware device."""
        service = get_ibm_service()
        backend = select_suitable_backend(service, min_qubits=4)
        assert backend is not None

        # Execute single 4-qubit decision with small shot count
        features_4d = np.array([0.5, 0.2, -0.3, 0.8])
        result = execute_on_real_hardware(
            features=features_4d,
            shots=256,
            timeout=300,
            backend=backend,
            service=service,
            allow_fallback=False,
        )

        assert result["status"] == "completed"
        assert result["is_real_hardware"] is True
        assert 0 <= result["action"] <= 8
        assert len(result["action_probabilities"]) == 9
        assert 0.0 <= result["invalid_probability_mass"] <= 1.0
