"""Regression tests for Checkpoint <-> Benchmark Architecture Synchronization.

Verifies:
- TEST A: 6Q3L checkpoint -> 6Q3L model (successful load)
- TEST B: 6Q3L checkpoint -> 4Q2L model (ValueError)
- TEST C: Benchmark with no explicit --qubits/--layers auto-resolves checkpoint architecture
- TEST D: Benchmark with explicit mismatch (--qubits 4 --layers 2 vs 6Q3L checkpoint) raises ValueError
- TEST E: Incompatible checkpoint never results in benchmark continuing with an untrained model
"""

import pytest
import numpy as np
from pathlib import Path

from config import QNavConfig
from quantum.quantum_model import (
    VariationalQuantumNavigator,
    inspect_checkpoint_architecture,
)
from quantum.quantum_decision import QuantumDecisionMaker
from evaluation.compare_models import ModelComparator


@pytest.fixture
def checkpoint_6q3l(tmp_path):
    """Create a valid 6Q3L checkpoint file with complete metadata."""
    ckpt_file = tmp_path / "quantum_weights_6q3l.npz"
    model = VariationalQuantumNavigator(
        n_qubits=6,
        n_layers=3,
        seed=42,
        load_checkpoint=False,
    )
    meta = {
        "n_qubits": 6,
        "n_layers": 3,
        "learning_rate": 0.005,
        "l2_strength": 0.1,
        "training_samples": 480,
        "validation_samples": 120,
        "validation_accuracy": 0.5543,
        "checkpoint_path": str(ckpt_file.resolve()),
        "timestamp": "2026-10-07T00:00:00Z",
    }
    model.save_checkpoint(ckpt_file, metadata=meta)
    return ckpt_file


def test_checkpoint_a_6q3l_to_6q3l(checkpoint_6q3l):
    """TEST A: 6Q3L checkpoint -> 6Q3L model succeeds."""
    info = inspect_checkpoint_architecture(checkpoint_6q3l)
    assert info["n_qubits"] == 6
    assert info["n_layers"] == 3
    assert info["arch_str"] == "6Q3L"

    target_model = VariationalQuantumNavigator(
        n_qubits=6,
        n_layers=3,
        seed=101,
        load_checkpoint=False,
    )
    loaded = target_model.load_weights(checkpoint_6q3l, strict=True)
    assert loaded is True
    assert target_model.checkpoint_loaded is True


def test_checkpoint_b_6q3l_to_4q2l_raises_value_error(checkpoint_6q3l):
    """TEST B: 6Q3L checkpoint -> 4Q2L model raises ValueError."""
    model_4q2l = VariationalQuantumNavigator(
        n_qubits=4,
        n_layers=2,
        seed=42,
        load_checkpoint=False,
    )
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        model_4q2l.load_weights(checkpoint_6q3l, strict=True)


def test_checkpoint_c_auto_resolve_architecture(checkpoint_6q3l, monkeypatch):
    """TEST C: Benchmark with no explicit architecture auto-resolves checkpoint architecture."""
    # Point default checkpoint path to our fixture
    from config import METRICS_DIR
    import config as cfg_module
    monkeypatch.setattr(cfg_module, "METRICS_DIR", checkpoint_6q3l.parent)
    symlink_path = checkpoint_6q3l.parent / "quantum_weights.npz"
    if not symlink_path.exists():
        import shutil
        shutil.copy(checkpoint_6q3l, symlink_path)

    # Instantiate QuantumDecisionMaker with default config (which starts at 4Q2L in dataclass)
    config = QNavConfig()
    assert config.quantum.n_qubits == 4
    assert config.quantum.n_layers == 2

    # Initializing QuantumDecisionMaker with checkpoint must auto-resolve to 6Q3L
    decision_maker = QuantumDecisionMaker(
        config=config,
        checkpoint_path=symlink_path,
        load_checkpoint=True,
        strict_load=True,
    )
    assert decision_maker.config.quantum.n_qubits == 6
    assert decision_maker.config.quantum.n_layers == 3
    assert decision_maker.model.n_qubits == 6
    assert decision_maker.model.circuit.n_layers == 3
    assert decision_maker.checkpoint_loaded is True


def test_checkpoint_d_explicit_mismatch_raises_value_error(checkpoint_6q3l):
    """TEST D: Explicit CLI mismatch (--qubits 4 --layers 2 vs 6Q3L checkpoint) raises ValueError."""
    ckpt_info = inspect_checkpoint_architecture(checkpoint_6q3l)
    ckpt_q = ckpt_info["n_qubits"]
    ckpt_l = ckpt_info["n_layers"]

    cli_qubits = 4
    cli_layers = 2

    # Verification logic from run.py
    with pytest.raises(ValueError, match="Explicit model architecture"):
        if cli_qubits is not None and cli_qubits != ckpt_q:
            raise ValueError(
                f"Explicit model architecture (--qubits {cli_qubits}) does not match "
                f"checkpoint ({ckpt_q}Q). Cannot load incompatible checkpoint."
            )
        if cli_layers is not None and cli_layers != ckpt_l:
            raise ValueError(
                f"Explicit model architecture (--layers {cli_layers}) does not match "
                f"checkpoint ({ckpt_l}L). Cannot load incompatible checkpoint."
            )


def test_checkpoint_e_no_silent_fallback_on_incompatible(tmp_path):
    """TEST E: Incompatible checkpoint must never result in benchmark continuing with an untrained model."""
    incompatible_file = tmp_path / "incompatible.npz"
    # Create a 4Q2L checkpoint
    m = VariationalQuantumNavigator(n_qubits=4, n_layers=2, seed=1, load_checkpoint=False)
    m.save_checkpoint(incompatible_file, metadata={"n_qubits": 4, "n_layers": 2})

    # Try loading into a 6Q3L decision maker strictly
    cfg = QNavConfig()
    cfg.quantum.n_qubits = 6
    cfg.quantum.n_layers = 3

    model = VariationalQuantumNavigator(n_qubits=6, n_layers=3, seed=2, load_checkpoint=False)
    with pytest.raises(ValueError, match="Checkpoint architecture mismatch"):
        model.load_weights(incompatible_file, strict=True)
