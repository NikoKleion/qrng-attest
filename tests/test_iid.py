# Tests for the SP 800-90B section 5 IID track.
import math
import numpy as np

from _harness import uniform, biased, sticky, periodic
from qrng_attest import iid


# ---------- the incomplete-gamma helper behind the chi-square p-values ----------
def test_gammaincc_known_values():
    assert abs(iid._gammaincc(1.0, 1.0) - math.exp(-1.0)) < 1e-12
    assert abs(iid._gammaincc(1.0, 2.0) - math.exp(-2.0)) < 1e-12
    assert iid._gammaincc(5.0, 0.0) == 1.0
    assert 0.0 < iid._gammaincc(4.5, 9.0) < 1.0


# ---------- statistics ----------
def test_statistics_shape_and_finiteness():
    S = uniform(2000, seed=1)
    vals, sym = np.unique(S, return_inverse=True)
    t = iid._statistics(sym, S, len(vals), 0.5, float(S.mean()))
    assert len(t) == len(iid.STAT_NAMES) == 19
    assert np.all(np.isfinite(t))


def test_find_collisions_small_example():
    # [0,1,2,0] over alphabet 3: first repeat closes a window of size 4
    assert iid._find_collisions(np.array([0, 1, 2, 0]), 3) == [4]
    assert iid._find_collisions(np.array([0, 0, 0, 0]), 2) == [2, 2]


# ---------- the decision ----------
def test_uniform_is_iid():
    r = iid.iid_decision(uniform(4000, seed=2), max_samples=4000)
    assert r.is_iid and r.permutation_iid and r.chi_square_iid and r.lrs_iid


def test_biased_but_independent_is_iid():
    # a biased coin flipped independently is IID; the MCV estimate carries the bias
    r = iid.iid_decision(biased(4000, 0.8, seed=3), max_samples=4000)
    assert r.is_iid, f"independent biased source should be IID, failed: {r.permutation.failed}"
    assert abs(r.assessed - (-math.log2(0.8))) < 0.05


def test_correlated_source_is_not_iid():
    r = iid.iid_decision(sticky(4000, flip=0.1, seed=4), perms=400, max_samples=4000)
    assert not r.is_iid
    assert not r.permutation_iid


def test_periodic_source_is_not_iid():
    r = iid.iid_decision(periodic(4000, 8, 0.05, seed=5), perms=400, max_samples=4000)
    assert not r.is_iid


def test_chi_square_flags_correlation():
    # a strongly correlated source should fail the independence chi-square
    ok, p_ind, p_gof = iid.chi_square_tests(sticky(4000, flip=0.05, seed=6))
    assert not ok and p_ind < 0.001


def test_iid_assessed_matches_mcv():
    # for an IID source the assessed entropy is exactly the most-common-value estimate
    from qrng_attest.estimators import most_common_value
    S = biased(4000, 0.75, seed=7)
    r = iid.iid_decision(S, perms=200, max_samples=4000)
    assert abs(r.assessed - most_common_value(S, 2)) < 1e-12


def test_iid_decision_is_relabel_invariant():
    # the IID verdict must not depend on symbol labels (regression: median-runs threshold vs raw values)
    S01 = sticky(2000, flip=0.08, seed=6)
    a = iid.iid_decision(S01, perms=400, max_samples=2000).is_iid
    b = iid.iid_decision(np.where(S01 == 0, 2, 7), perms=400, max_samples=2000).is_iid
    assert a == b


def test_iid_assessed_is_relabel_invariant():
    # the IID-credited assessment must not depend on symbol labels (regression: _mcv_assessed word_size)
    base = uniform(4000, k=3, seed=14)
    a = iid.iid_decision(base, perms=200, max_samples=4000).assessed
    b = iid.iid_decision(np.where(base == 2, 9, base), perms=200, max_samples=4000).assessed
    assert abs(a - b) < 1e-12


def test_lrs_test_actually_computes():
    # the LRS test must compute a real probability, not silently return (True, 1.0) for everything (regression)
    import os
    vec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nist_vectors", "rand1_short.bin")
    if not os.path.exists(vec):
        return
    S = np.frombuffer(open(vec, "rb").read(), np.uint8).astype(int)
    ok, prob = iid.lrs_test(S)
    assert ok and 0.0 < prob < 1.0


def test_lrs_length_exact_on_known_strings():
    assert iid._lrs_length(np.array([0, 1, 0, 1, 1, 0, 1])) == 3
    assert iid._lrs_length(np.array([1, 2, 3, 1, 2, 3, 4])) == 3
    assert iid._lrs_length(np.array([0, 0, 0, 0, 0])) == 4
    assert iid._lrs_length(np.array([0, 1])) == 0


def test_lrs_flags_planted_repeat():
    # an improbably long duplicated block in otherwise-random data must fail the LRS test
    rng = np.random.default_rng(1)
    S = rng.integers(0, 2, 6000)
    S[3000:3100] = S[:100]
    ok, prob = iid.lrs_test(S)
    assert not ok and prob < 1e-6


def test_lrs_flags_periodic():
    ok, _prob = iid.lrs_test(np.tile([0, 1, 2, 3], 1000))
    assert not ok


def test_binary_chi_square_uses_tuples():
    # the binary independence test forms m-bit tuples; a balanced-but-sticky source must fail it
    ok, p_ind, _ = iid.chi_square_tests(sticky(6000, flip=0.05, seed=21))
    assert not ok and p_ind < 0.001
    ok2, p2, g2 = iid.chi_square_tests(uniform(6000, seed=22))
    assert ok2 and p2 >= 0.001 and g2 >= 0.001
