# Tests for the quantum-device entropy attestation layer.
import math
import numpy as np

import qrng_attest as es
from qrng_attest.quantum import (geometric_min_entropy, state_min_entropy, bloch_z, assignment_matrix,
                                     readout_deconvolve, attested_qubit_entropy, synthetic_qrng)
from qrng_attest.health import repetition_count_test, adaptive_proportion_test, rct_cutoff
from qrng_attest.attest import mutual_information, independence, drift, attest


# ---------- Bloch geometry ----------
def test_geometric_min_entropy_ideal_and_poles():
    assert abs(geometric_min_entropy(0.5) - 1.0) < 1e-12
    assert geometric_min_entropy(1.0) == 0.0
    assert abs(geometric_min_entropy(0.85) - (-math.log2(0.85))) < 1e-12


def test_state_min_entropy_matches_geometry():
    # P0 = cos^2(theta/2); pi/2 -> 1 bit, 0 -> 0 bits
    assert abs(state_min_entropy(math.pi / 2) - 1.0) < 1e-9
    assert state_min_entropy(0.0) == 0.0


def test_bloch_z_roundtrip():
    for p0 in (0.5, 0.7, 0.9):
        assert abs(bloch_z(p0) - (2 * p0 - 1)) < 1e-12


# ---------- readout channel ----------
def test_readout_deconvolution_recovers_true_bias():
    # observed = M @ true; deconvolve recovers the true distribution
    M = assignment_matrix(0.75, 0.97)
    p_true = np.array([0.68, 0.32])
    p_obs = M @ p_true
    rec = readout_deconvolve(p_obs, M)
    assert np.max(np.abs(rec - p_true)) < 1e-9


def test_readout_dominated_flag():
    # biased source symmetrized by asymmetric readout; deconvolve reveals the bias
    s, M = synthetic_qrng(20000, 1, source_bias=0.18, readout_F=(0.72, 0.98), seed=2)
    q = attested_qubit_entropy(s[:, 0], M[0])
    assert q["readout_dominated"], "should flag readout-manufactured entropy"
    assert q["h_quantum"] < q["h_observed"] - 0.05


# ---------- health tests ----------
def test_rct_flags_stuck_source():
    stuck = np.array([1] * 100 + [0, 1] * 100)
    ok, longest, C = repetition_count_test(stuck, H=1.0)
    assert not ok and longest >= C


def test_rct_passes_random():
    ok, _, _ = repetition_count_test(np.random.default_rng(0).integers(0, 2, 20000), H=1.0)
    assert ok


def test_apt_flags_biased_window():
    biased = (np.random.default_rng(1).random(20000) < 0.95).astype(int)
    ok, worst, C = adaptive_proportion_test(biased, H=1.0)
    assert not ok and worst >= C


def test_rct_cutoff_shrinks_with_entropy():
    assert rct_cutoff(1.0) < rct_cutoff(0.1)


# ---------- independence and drift ----------
def test_mutual_information_bounds():
    x = np.random.default_rng(0).integers(0, 2, 20000)
    assert mutual_information(x, x) > 0.99
    y = np.random.default_rng(1).integers(0, 2, 20000)
    assert mutual_information(x, y) < 0.01


def test_independence_catches_crosstalk():
    s, _ = synthetic_qrng(20000, 4, crosstalk=0.5, seed=3)
    assert not independence(s)["independent"]
    s2, _ = synthetic_qrng(20000, 4, crosstalk=0.0, seed=3)
    assert independence(s2)["independent"]


def test_drift_catches_ramp():
    s, _ = synthetic_qrng(20000, 1, source_bias=-0.15, drift=0.30, seed=4)
    assert not drift(s[:, 0])["stationary"]
    s2, _ = synthetic_qrng(20000, 1, source_bias=0.0, drift=0.0, seed=4)
    assert drift(s2[:, 0])["stationary"]


# ---------- the combined attestation ----------
def test_attestation_passes_healthy_source():
    s, M = synthetic_qrng(20000, 4, source_bias=0.0, readout_F=(0.99, 0.99), seed=1)
    a = attest(es.QuantumArraySource(s, readout=M), include_predictors=False)
    assert a.passed


def test_attestation_flags_and_never_over_credits():
    # true source P(0)=0.68 -> min-entropy 0.556; attested must not exceed it
    s, M = synthetic_qrng(20000, 4, source_bias=0.18, readout_F=(0.72, 0.98), seed=2)
    a = attest(es.QuantumArraySource(s, readout=M), include_predictors=False)
    true_source_h = -math.log2(0.68)
    assert not a.passed
    assert a.attested_min_entropy <= true_source_h + 0.05, "attestation must not over-credit the biased source"


def test_attestation_flags_crosstalk():
    s, M = synthetic_qrng(20000, 4, crosstalk=0.5, seed=3)
    a = attest(es.QuantumArraySource(s, readout=M), include_predictors=False)
    assert not a.passed and not a.independence["independent"]


def test_regression_no_over_credit_in_readout_regime():
    # sweep bias x asymmetric readout; attested must never exceed the true source min-entropy
    for bias in (0.10, 0.20, 0.30):
        for F in [(0.72, 0.98), (0.75, 0.95), (0.70, 0.90)]:
            for seed in (1, 2, 3):
                p0 = min(max(0.5 + bias, 0.02), 0.98)
                true_h = -math.log2(max(p0, 1 - p0))
                s, M = synthetic_qrng(20000, 2, source_bias=bias, readout_F=F, seed=seed)
                a = attest(es.QuantumArraySource(s, readout=M), include_predictors=False)
                assert a.attested_min_entropy <= true_h + 1e-6, \
                    f"over-credit: bias={bias} F={F} seed={seed}: {a.attested_min_entropy} > {true_h}"


def test_regression_stuck_qubit_flagged():
    # a fully-stuck qubit must be flagged
    a = attest(es.QuantumArraySource(np.zeros((5000, 2), int)), include_predictors=False)
    assert not a.passed and a.attested_min_entropy < 0.01
