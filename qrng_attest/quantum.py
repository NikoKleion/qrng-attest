# qrng_attest.quantum: quantum-device min-entropy from Bloch geometry, readout deconvolution, and a QuantumSource.
# Assumes a trusted measurement and a characterized readout.
import math
import numpy as np

from .sources import Source


# Bloch-sphere geometry
def bloch_z(p0):
    # Bloch z-component from P(0): r_z = 2 P0 - 1
    return 2.0 * float(p0) - 1.0


def geometric_min_entropy(p0):
    # min-entropy of a Z-basis outcome: -log2(max(P0, P1)).
    p0 = min(max(float(p0), 0.0), 1.0)
    return float(-math.log2(max(p0, 1.0 - p0, 1e-12)) + 0.0)


def state_min_entropy(theta):
    # min-entropy for a state at polar angle theta: P0 = cos^2(theta/2).
    return geometric_min_entropy(math.cos(theta / 2.0) ** 2)


# readout channel
def assignment_matrix(F0, F1):
    # assignment matrix M[a, b] = P(read a | true b).
    return np.array([[F0, 1.0 - F1], [1.0 - F0, F1]], float)


def readout_deconvolve(p_obs, M):
    # recover the true distribution from the observed: p_true = M^-1 p_obs, projected onto the simplex.
    p_obs = np.asarray(p_obs, float)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-9:
        return p_obs
    p_true = np.linalg.solve(M, p_obs)
    p_true = np.clip(p_true, 0.0, 1.0)
    s = p_true.sum()
    return p_true / s if s > 0 else np.array([0.5, 0.5])


def calibrate_readout(prep0_outcomes, prep1_outcomes):
    # measure the assignment matrix from known preparations instead of trusting a vendor number.
    a0 = np.asarray(prep0_outcomes, int).ravel()
    a1 = np.asarray(prep1_outcomes, int).ravel()
    n0 = max(len(a0), 1); n1 = max(len(a1), 1)
    F0 = float(np.count_nonzero(a0 == 0)) / n0
    F1 = float(np.count_nonzero(a1 == 1)) / n1
    return assignment_matrix(F0, F1), int(min(n0, n1))


Z_CONF = 2.5758293035489004


def _conservative_p0(p0, sigma, z=Z_CONF):
    # push bias away from 1/2 by z sigma for a one-sided lower bound.
    if p0 >= 0.5:
        return min(1.0, p0 + z * sigma)
    return max(0.0, p0 - z * sigma)


def _corner_entropy(p0_obs, sigma_obs, F0, F1):
    # deconvolved conservative entropy for one candidate assignment matrix.
    M = assignment_matrix(F0, F1)
    p_true = readout_deconvolve(np.array([p0_obs, 1.0 - p0_obs]), M)
    det = F0 * F1 - (1.0 - F0) * (1.0 - F1)
    sigma_src = sigma_obs / abs(det) if abs(det) > 1e-9 else sigma_obs
    p0_src = float(p_true[0])
    return geometric_min_entropy(_conservative_p0(p0_src, sigma_src)), p0_src


def attested_qubit_entropy(bits, M=None, calib_shots=None, M_margin=None):
    # quantum min-entropy of one qubit's stream, 99% one-sided lower bound after readout deconvolution.
    bits = np.asarray(bits, int)
    n = max(len(bits), 2)
    p0_obs = float(np.mean(bits == 0))
    h_obs = geometric_min_entropy(p0_obs)
    sigma_obs = math.sqrt(max(p0_obs * (1.0 - p0_obs), 1e-12) / (n - 1))
    if M is None:
        p0_cons = _conservative_p0(p0_obs, sigma_obs)
        h_q = geometric_min_entropy(p0_cons)
        return {"p0_obs": p0_obs, "h_observed": h_obs, "p0_source": p0_obs, "h_quantum": h_q,
                "r_z": bloch_z(p0_obs), "readout_dominated": False, "has_readout_model": False}
    F0, F1 = float(M[0, 0]), float(M[1, 1])
    if M_margin is not None:
        d0 = d1 = float(M_margin)
    elif calib_shots is not None:
        nc = max(int(calib_shots), 2)
        d0 = Z_CONF * math.sqrt(max(F0 * (1.0 - F0), 1e-12) / nc)
        d1 = Z_CONF * math.sqrt(max(F1 * (1.0 - F1), 1e-12) / nc)
    else:
        d0 = d1 = 0.0
    h_q, p0_src = _corner_entropy(p0_obs, sigma_obs, F0, F1)
    if d0 > 0.0 or d1 > 0.0:
        for f0 in (F0 - d0, F0, F0 + d0):
            for f1 in (F1 - d1, F1, F1 + d1):
                f0c = min(0.9995, max(0.505, f0)); f1c = min(0.9995, max(0.505, f1))
                h_c, _p = _corner_entropy(p0_obs, sigma_obs, f0c, f1c)
                if h_c < h_q:
                    h_q = h_c
    dominated = (h_obs - h_q) > 0.1
    return {"p0_obs": p0_obs, "h_observed": h_obs, "p0_source": p0_src, "h_quantum": h_q,
            "r_z": bloch_z(p0_src), "readout_dominated": dominated, "has_readout_model": True,
            "calib_margin": (d0, d1)}


# QuantumSource
class QuantumSource:
    name = "quantum-source"
    provenance = "unknown"
    expected_state = "plus"
    calibration_shots = None

    def shots(self):
        # (n_shots, n_qubits) array of measurement outcomes in {0, 1}
        raise NotImplementedError

    @property
    def n_qubits(self):
        return np.asarray(self.shots()).shape[1]

    def readout_matrices(self):
        # per-qubit 2x2 assignment matrices, or None.
        return getattr(self, "_readout", None)

    def qubit_stream(self, q):
        # one qubit's bit stream.
        return np.asarray(self.shots())[:, q]

    def describe(self):
        s = np.asarray(self.shots())
        flags = []
        if self.provenance == "synthetic":
            flags.append("synthetic quantum source: this run is simulated, not measured hardware")
        if self.readout_matrices() is None:
            flags.append("no readout model provided; quantum min-entropy falls back to the observed bias")
        elif self.calibration_shots is None:
            flags.append("readout matrix trusted as exact; supply calibration_shots to bound calibration error")
        return {"name": self.name, "n_shots": int(s.shape[0]), "n_qubits": int(s.shape[1]),
                "provenance": self.provenance, "expected_state": self.expected_state, "flags": flags}


class QuantumArraySource(QuantumSource):
    def __init__(self, shots, readout=None, name="qubit-readout", provenance="measured", expected_state="plus",
                 calibration_shots=None):
        self._s = np.ascontiguousarray(np.asarray(shots)).astype(int)
        if self._s.ndim == 1:
            self._s = self._s[:, None]
        self._readout = readout
        self.name = name; self.provenance = provenance; self.expected_state = expected_state
        self.calibration_shots = calibration_shots

    def shots(self):
        return self._s


class QuantumCallableSource(QuantumSource):
    # live device: fn(n_shots) -> (n_shots, n_qubits) outcomes.
    def __init__(self, fn, n_shots, readout=None, name="live-qpu", provenance="measured", expected_state="plus"):
        self.fn = fn; self.n = int(n_shots); self._readout = readout
        self.name = name; self.provenance = provenance; self.expected_state = expected_state
        self._cache = None

    def shots(self):
        if self._cache is None:
            s = np.asarray(self.fn(self.n)).astype(int)
            self._cache = s[:, None] if s.ndim == 1 else s
        return self._cache


def synthetic_qrng(n_shots, n_qubits=4, source_bias=0.0, readout_F=(0.99, 0.98), crosstalk=0.0,
                   drift=0.0, seed=0, model_readout=True):
    # QRNG device model for testing and demos; returns (shots, readout_matrices or None).
    rng = np.random.default_rng(seed)
    F0, F1 = readout_F
    M = assignment_matrix(F0, F1)
    shots = np.empty((n_shots, n_qubits), int)
    ramp = np.linspace(0.0, drift, n_shots)
    for t in range(n_shots):
        b = source_bias + ramp[t]
        p0 = np.clip(0.5 + b, 0.02, 0.98)
        true = (rng.random(n_qubits) > p0).astype(int)
        if crosstalk > 0 and n_qubits > 1:
            copy = rng.random(n_qubits) < crosstalk
            true[1:] = np.where(copy[1:], true[:-1], true[1:])
        obs = np.where(rng.random(n_qubits) < np.where(true == 0, 1 - F0, 1 - F1), 1 - true, true)
        shots[t] = obs
    return shots, ([M] * n_qubits if model_readout else None)
