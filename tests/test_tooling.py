# Tests for self-calibration, qubit-subset selection, and remediation.
import numpy as np

import qrng_attest as es
from qrng_attest.quantum import calibrate_readout, assignment_matrix, attested_qubit_entropy


def test_self_calibration_recovers_known_fidelities():
    rng = np.random.default_rng(0)
    p0 = (rng.random(20000) > 0.95).astype(int)
    p1 = (rng.random(20000) < 0.92).astype(int)
    M, shots = calibrate_readout(p0, p1)
    assert abs(M[0, 0] - 0.95) < 0.01 and abs(M[1, 1] - 0.92) < 0.01
    assert shots == 20000


def test_self_calibration_shot_count_drives_the_margin():
    from qrng_attest import overcredit as oc
    bits = oc.observed_stream(0.98, 0.75, 0.75, 40000, seed=1)
    M = assignment_matrix(0.75, 0.75)
    wide = attested_qubit_entropy(bits, M, calib_shots=500)
    narrow = attested_qubit_entropy(bits, M, calib_shots=20000)
    exact = attested_qubit_entropy(bits, M)
    assert wide["calib_margin"][0] > narrow["calib_margin"][0] > 0.0
    assert wide["h_quantum"] <= narrow["h_quantum"] <= exact["h_quantum"] + 1e-12


def test_source_flags_untrusted_calibration():
    shots, M = es.synthetic_qrng(2000, 2, seed=2)
    plain = es.QuantumArraySource(shots, readout=M).describe()["flags"]
    boxed = es.QuantumArraySource(shots, readout=M, calibration_shots=4000).describe()["flags"]
    assert any("trusted as exact" in f for f in plain)
    assert not any("trusted as exact" in f for f in boxed)


def test_subset_selection_drops_a_bad_qubit_and_raises_throughput():
    shots, M = es.synthetic_qrng(6000, 4, seed=3)
    shots[:, 2] = (np.random.default_rng(1).random(6000) > 0.85).astype(int)
    src = es.QuantumArraySource(shots, readout=M, calibration_shots=4000)
    r = es.qubit_subset(src)
    assert 2 in r["dropped"]
    assert r["worth_it"] and r["total_bits_per_shot"] > r["base_bits_per_shot"]


def test_subset_selection_keeps_everything_on_a_healthy_device():
    shots, M = es.synthetic_qrng(6000, 3, seed=4)
    r = es.qubit_subset(es.QuantumArraySource(shots, readout=M, calibration_shots=4000))
    assert r["dropped"] == []
    assert r["best_qubits"] == [0, 1, 2]


def test_remediation_is_actionable_and_always_present():
    shots, M = es.synthetic_qrng(4000, 3, crosstalk=0.6, seed=5)
    acts = es.remediation(es.attest(es.QuantumArraySource(shots, readout=M), include_predictors=False))
    assert acts and all(isinstance(a, str) and a for a in acts)
    assert any("correlated" in a for a in acts)


def test_remediation_says_nothing_needed_when_healthy():
    shots, M = es.synthetic_qrng(6000, 2, seed=6)
    acts = es.remediation(es.attest(es.QuantumArraySource(shots, readout=M), include_predictors=False))
    assert any("no action required" in a for a in acts)
