"""Random brickwork circuits on a line of qubits.

One layer = a Haar-random single-qubit gate on every qubit, then CZ gates on
neighbouring pairs. The pairs alternate like bricks in a wall:

    even layers: (0,1), (2,3), (4,5), ...
    odd layers:  (1,2), (3,4), (5,6), ...

`depth` counts the entangling (CZ) layers. A final layer of single-qubit gates
is always added, so depth 0 is a random product state with no entanglement.
"""

import numpy as np
from qiskit import QuantumCircuit


def haar_u_angles(rng):
    """Euler angles (theta, phi, lam) of a Haar-random single-qubit gate.

    Qiskit's U(theta, phi, lam) equals Rz(phi) Ry(theta) Rz(lam) up to a global
    phase. In these coordinates the Haar measure on SU(2) has density
    proportional to sin(theta), so theta = arccos(1 - 2u) with u ~ Uniform(0, 1).
    phi and lam are uniform on [0, 2*pi).
    """
    theta = np.arccos(1 - 2 * rng.random())
    phi, lam = 2 * np.pi * rng.random(2)
    return theta, phi, lam


def single_qubit_layer(qc, rng):
    for q in range(qc.num_qubits):
        qc.u(*haar_u_angles(rng), q)


def entangling_layer(qc, layer_index):
    start = layer_index % 2
    for q in range(start, qc.num_qubits - 1, 2):
        qc.cz(q, q + 1)


def brickwork_circuit(n_qubits, depth, rng):
    """Random brickwork circuit with `depth` CZ layers (no measurements)."""
    qc = QuantumCircuit(n_qubits)
    for layer in range(depth):
        single_qubit_layer(qc, rng)
        entangling_layer(qc, layer)
    single_qubit_layer(qc, rng)
    return qc
