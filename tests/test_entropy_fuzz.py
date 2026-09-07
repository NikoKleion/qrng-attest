# Property tests: the assessment never crashes and returns valid values across shapes and alphabets.
import math
import warnings
import numpy as np

from qrng_attest import estimators as E
from qrng_attest import iid

_META = ("H_original", "H_bitstring", "word_size")


def test_min_entropy_valid_across_shapes_and_alphabets():
    for k in (2, 3, 4, 7, 16):
        for n in (2, 3, 5, 17, 50, 300, 1200):
            for seed in range(3):
                S = np.random.default_rng(seed * 991 + k * 17 + n).integers(0, k, n)
                mn, per = E.min_entropy(S)
                cap = math.log2(max(2, len(np.unique(S))))
                assert math.isfinite(mn) and -1e-9 <= mn <= cap + 1e-6, f"k={k} n={n} seed={seed}: {mn}"
                for name, v in per.items():
                    if name in _META:
                        continue
                    assert math.isfinite(v) and v >= -1e-9, f"{name}={v}"


def test_empty_and_degenerate_inputs_are_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mn, per = E.min_entropy(np.array([], int))
        assert math.isnan(mn) and per == {}
        assert E.all_estimators(np.array([], int)) == {}
    assert E.min_entropy(np.array([5]))[0] == 0.0
    assert E.min_entropy(np.zeros(500, int))[0] < 0.05


def test_iid_decision_never_crashes():
    for k in (2, 3, 5, 16):
        for n in (1, 2, 20, 200, 800):
            S = np.random.default_rng(k * 13 + n).integers(0, k, n)
            r = iid.iid_decision(S, perms=50, max_samples=800)
            assert isinstance(r.is_iid, bool)
            assert math.isfinite(r.assessed) and r.assessed >= -1e-9


def test_compression_block_boundary_no_crash():
    # exercise the v = num_blocks - 1000 boundary (blocks 1000..1003) and beyond
    for nbits in (0, 1, 6, 6000, 6006, 6011, 6012, 6018, 12000):
        v = E.compression(np.random.default_rng(nbits + 1).integers(0, 2, nbits), k=2)
        assert v is None or (0.0 <= v <= 1.0 + 1e-9)
