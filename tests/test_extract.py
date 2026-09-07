# Tests for the leftover-hash-lemma sizing and the Toeplitz extractor.
import math
import numpy as np

import qrng_attest as es
from qrng_attest import extract as ex


def test_lhl_sizing_matches_closed_form():
    n, h, eps = 1000000, 0.68, 2.0 ** -64
    expected = math.floor(n * h - 2 * 64 + 2)
    assert ex.extractable_bits(n, h, eps) == expected


def test_sizing_is_monotone_and_safe():
    assert ex.extractable_bits(1000, 0.5) < ex.extractable_bits(1000, 0.9)
    assert ex.extractable_bits(1000, 0.9, 2.0 ** -128) < ex.extractable_bits(1000, 0.9, 2.0 ** -32)
    assert ex.extractable_bits(10, 0.5) == 0
    assert ex.extractable_bits(1000, 0.0) == 0


def test_output_never_exceeds_available_entropy():
    for n in (500, 5000, 50000):
        for h in (0.1, 0.5, 1.0):
            assert ex.extractable_bits(n, h) <= n * h


def test_input_needed_roundtrips_with_sizing():
    for target in (128, 256, 1024):
        n = ex.input_bits_needed(target, 0.7)
        assert ex.extractable_bits(n, 0.7) >= target


def test_toeplitz_is_linear_over_gf2():
    rng = np.random.default_rng(0)
    n, ell = 500, 100
    seed = ex.random_seed(n, ell, rng=rng)
    a = rng.integers(0, 2, n).astype(np.uint8)
    b = rng.integers(0, 2, n).astype(np.uint8)
    left = ex.toeplitz_extract(a ^ b, seed, ell)
    right = ex.toeplitz_extract(a, seed, ell) ^ ex.toeplitz_extract(b, seed, ell)
    assert np.array_equal(left, right)


def test_toeplitz_output_shape_and_binary():
    rng = np.random.default_rng(1)
    n, ell = 300, 64
    seed = ex.random_seed(n, ell, rng=rng)
    out = ex.toeplitz_extract(rng.integers(0, 2, n), seed, ell)
    assert out.shape == (ell,) and set(np.unique(out)).issubset({0, 1})


def test_toeplitz_rejects_wrong_seed_length():
    try:
        ex.toeplitz_extract(np.zeros(100, np.uint8), np.zeros(10, np.uint8), 20)
    except ValueError:
        return
    raise AssertionError("a wrong-length seed must be rejected")


def test_extraction_is_deterministic_given_the_seed():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    out1, seed = ex.extract(bits, 0.9, epsilon=2.0 ** -32, rng=rng)
    out2 = ex.toeplitz_extract(bits, seed, len(out1))
    assert np.array_equal(out1, out2)


def test_extracted_output_looks_uniform():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 20000).astype(np.uint8)
    out, _ = ex.extract(bits, 0.95, epsilon=2.0 ** -32, rng=rng)
    assert len(out) > 1000
    assert abs(out.mean() - 0.5) < 0.05
    mn, _ = es.min_entropy(out.astype(int))
    assert mn > 0.7


def test_plan_reports_key_costs():
    p = ex.plan(1000000, 0.68)
    assert p.output_bits == ex.extractable_bits(1000000, 0.68)
    assert p.seed_bits == p.n_input + p.output_bits - 1
    assert 0.0 < p.efficiency <= 0.68
    assert "256-bit key" in str(p)


def test_assessment_exposes_an_extraction_plan():
    a = es.assess(es.ArraySource(np.random.default_rng(4).integers(0, 2, 20000), alphabet_size=2))
    p = a.extraction_plan()
    assert p.output_bits > 0
    assert abs(p.h_per_symbol - a.min_entropy) < 1e-12
    assert "extractable" in str(a)


def test_attestation_exposes_an_extraction_plan():
    shots, M = es.synthetic_qrng(4000, 2, seed=5)
    att = es.attest(es.QuantumArraySource(shots, readout=M), include_predictors=False)
    p = att.extraction_plan()
    assert abs(p.h_per_symbol - att.attested_min_entropy) < 1e-12
    assert p.n_input == 4000 * 2
