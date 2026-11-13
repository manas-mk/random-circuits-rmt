"""Statistics of output distributions.

For a Haar-random state in dimension D, the scaled probabilities D * p(x)
follow the Porter-Thomas distribution, which is Exp(1) for complex amplitudes.
Three ways to compare a circuit's output distribution with that prediction:

* ks_to_exponential: Kolmogorov-Smirnov distance between D*p and Exp(1).
  The D values are not independent (they sum to 1), so the usual KS p-value
  does not apply. We compare against a null built from actual Haar states.
* collision_ratio: D * sum(p^2). For Haar states its mean is 2D / (D + 1),
  which tends to 2. A product state on n qubits gives (4/3)^n on average,
  which grows with n: probabilities are too uneven.
* heavy_output_probability: total probability of outputs whose ideal
  probability is above the median. Porter-Thomas gives (1 + ln 2) / 2 = 0.847.
  This is the quantity IBM's Quantum Volume test is built on.
"""

import numpy as np
from qiskit.quantum_info import random_statevector

HOP_PORTER_THOMAS = (1 + np.log(2)) / 2


def ks_to_exponential(probs):
    D = len(probs)
    x = np.sort(D * np.asarray(probs))
    cdf = 1.0 - np.exp(-x)
    above = np.arange(1, D + 1) / D - cdf
    below = cdf - np.arange(0, D) / D
    return float(max(above.max(), below.max()))


def collision_ratio(probs):
    probs = np.asarray(probs)
    return float(len(probs) * np.sum(probs**2))


def heavy_output_probability(probs):
    probs = np.asarray(probs)
    return float(probs[probs > np.median(probs)].sum())


def haar_collision_mean(D):
    return 2 * D / (D + 1)


def haar_null(n_qubits, n_states, seed):
    """KS distance, collision ratio and heavy-output probability for Haar-random states."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_states):
        probs = random_statevector(2**n_qubits, seed=rng).probabilities()
        rows.append((ks_to_exponential(probs), collision_ratio(probs), heavy_output_probability(probs)))
    return np.array(rows)
