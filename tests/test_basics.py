"""Sanity checks. Run with: python -m pytest tests"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.circuits import brickwork_circuit, haar_u_angles
from src.stats import collision_ratio, haar_collision_mean, haar_null, ks_to_exponential


def test_single_qubit_gate_is_haar():
    # For a Haar-random U, |<0|U|0>|^2 = cos^2(theta/2) is Uniform(0, 1).
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(20000):
        qc = QuantumCircuit(1)
        qc.u(*haar_u_angles(rng), 0)
        vals.append(Statevector(qc).probabilities()[0])
    vals = np.array(vals)
    assert abs(vals.mean() - 0.5) < 0.01
    assert abs(vals.var() - 1 / 12) < 0.003


def test_depth_zero_is_product_state():
    # Collision ratio of a product state factorises over qubits.
    rng = np.random.default_rng(1)
    qc = brickwork_circuit(4, 0, rng)
    probs = Statevector(qc).probabilities()
    per_qubit = []
    for q in range(4):
        p1 = sum(p for i, p in enumerate(probs) if (i >> q) & 1)
        per_qubit.append(2 * (p1**2 + (1 - p1) ** 2))
    assert np.isclose(collision_ratio(probs), np.prod(per_qubit))


def test_haar_null_matches_theory():
    null = haar_null(6, 2000, seed=2)
    assert abs(null[:, 1].mean() - haar_collision_mean(64)) < 0.02


def test_ks_is_zero_for_exact_quantiles():
    D = 1000
    x = -np.log(1 - (np.arange(D) + 0.5) / D)  # Exp(1) quantiles
    assert ks_to_exponential(x / x.sum()) < 0.01
