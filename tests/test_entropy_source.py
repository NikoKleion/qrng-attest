# Tests for qrng_attest: the SP 800-90B non-IID suite.
import math
import numpy as np

from _harness import uniform, biased, sticky, periodic
from qrng_attest import estimators as E
from qrng_attest import min_entropy, all_estimators, assess, ArraySource, BytesFileSource, CallableSource

N = 40000


# ---------- core correctness ----------
def test_never_over_estimate():
    # reported min-entropy must not exceed the true min-entropy
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
        S = biased(N, p, seed=1)
        mn, _ = min_entropy(S, k=2)
        true = -math.log2(max(p, 1 - p))
        assert mn <= true + 0.03, f"p={p}: reported {mn} exceeds true {true}"


def test_correlated_source_is_caught():
    # a sticky source looks balanced marginally but is predictable
    S = sticky(N, flip=0.1, seed=2)
    mn, est = min_entropy(S, k=2)
    assert est["most_common_value"] > 0.9, "marginal should look near-full-entropy"
    assert mn < 0.3, f"non-IID min should catch the correlation, got {mn}"


def test_biased_matches_theory():
    S = biased(N, 0.7, seed=3)
    mn, est = min_entropy(S, k=2)
    assert abs(est["most_common_value"] - (-math.log2(0.7))) < 0.05
    assert mn <= -math.log2(0.7) + 0.03


def test_uniform_is_high():
    mn, est = min_entropy(uniform(N, seed=4), k=2)
    assert est["most_common_value"] > 0.95
    assert mn > 0.6


def test_min_equals_min_of_all():
    S = biased(N, 0.65, seed=5)
    mn, est = min_entropy(S, k=2)
    assert mn == min(est.values())


def test_collision_inversion_recovers_p():
    # E = 2 + 2pq for binary inverts to recover p
    for p in (0.7, 0.9):
        E_time = 2.0 + 2.0 * p * (1 - p)
        rec = 0.5 + 0.5 * math.sqrt(5.0 - 2.0 * E_time)
        assert abs(rec - p) < 1e-9


def test_collision_full_entropy_when_anticorrelated():
    # a mean collision time at or above the balanced-binary value (2.5) must report full entropy, not 0:
    assert abs(E.collision(np.tile([0, 1], 20000), k=2) - 1.0) < 1e-9
    assert E.min_entropy(np.tile([0, 1], 20000))[0] < 0.05


def test_collision_discards_trailing_nonrepeat_segment():
    # a final segment that reaches end-of-data without a repeat is discarded, not counted (regression)
    assert E.collision(np.array([0, 0, 0, 1]), k=2) == 1.0


def test_compression_short_stream_returns_none_not_crash():
    # v = num_blocks - d must be >= 2; a stream giving v == 1 returns None instead of dividing by zero (regression)
    assert E.compression(np.zeros(6006, int), k=2) is None
    assert E.compression(np.zeros(6011, int), k=2) is None
    assert E.compression(np.zeros(6012, int), k=2) is not None


def test_multibit_assessment_is_relabel_invariant():
    # relabeling symbols must not change the assessment: word_size follows the alphabet, not raw values (regression)
    base = uniform(20000, k=4, seed=12)
    a, _ = min_entropy(base)
    b, _ = min_entropy(np.where(base == 3, 4, base))
    assert abs(a - b) < 1e-9


def test_predictor_zero_run_not_capped_at_one_for_k3():
    # a never-correct predictor on a non-binary source must exceed 1 bit, not report exactly 1.0 (regression)
    assert E._predictor_entropy(np.zeros(9, bool), k=3) > 1.05


# ---------- regression tests ----------
def test_regression_no_overflow_on_deterministic():
    # _no_run_prob must not overflow on deterministic streams at large N
    for S in (np.zeros(100000, int), np.ones(100000, int), np.tile([0, 1], 50000),
              np.tile(np.array([0, 0, 1, 0, 1, 1, 1, 0]), 12500)):
        mn, est = min_entropy(S, k=2)
        assert mn < 0.05, f"deterministic stream should be ~0 bits, got {mn}"
        assert all(v <= 1.0001 for v in est.values()), "no estimator may exceed log2(k)=1 bit"


def test_regression_no_run_prob_bounded_no_overflow():
    for args in [(0.5, 2, 100000), (0.99, 3, 100000), (0.5, 50000, 50000), (0.5, 17, 1000000)]:
        v = E._no_run_prob(*args)
        assert 0.0 <= v <= 1.0


def test_regression_p_local_no_crash_across_run_lengths():
    for r in (1, 6, 7, 100, 1000, 3000):
        p = E._p_local(r, 100000)
        assert 0.0 <= p <= 1.0


def test_regression_out_of_range_value_no_indexerror():
    # a source whose values exceed the declared k expands to a multi-bit assessment, not an IndexError
    S = np.array([0, 1, 2, 0, 1, 0, 1, 1, 0, 2] * 3000)
    mn, per = min_entropy(S, k=2)
    assert 0.0 <= mn <= math.log2(3) + 1e-9
    assert not any(k.startswith("literal:") and k.split(":")[1] in ("collision", "markov", "compression") for k in per)
    assert "bitstring:collision" in per and "literal:most_common_value" in per


def test_regression_lrs_not_maximal_on_biased():
    # lrs must not return 1.0 on highly biased sources
    assert E.lrs(biased(N, 0.9, seed=6), k=2) < 0.9
    assert E.lrs(biased(N, 0.99, seed=7), k=2) < 0.5


def test_regression_ternary_not_capped_at_one():
    # the literal (per-symbol) track must cap at log2(k), not 1.0, so its estimators can exceed a bit
    mn, per = min_entropy(uniform(N, k=3, seed=8))
    assert per["H_original"] > 1.2, f"ternary literal track should exceed 1 bit, got {per['H_original']}"
    assert per["H_original"] <= math.log2(3) + 0.03
    assert mn <= math.log2(3) + 0.03


def test_predictors_never_exceed_one_bit():
    for fn in (E.multimcw, E.lag, E.multimmc, E.lz78y):
        v = fn(periodic(N, 8, 0.05, seed=9), k=2)
        if v is not None:
            assert v <= 1.0001, f"{fn.__name__} returned {v} > 1 bit"


# ---------- alphabet handling ----------
def test_binary_only_estimators_excluded_for_k3():
    est = all_estimators(uniform(N, k=3, seed=10), k=3)
    for name in ("collision", "compression", "markov"):
        assert name not in est, f"{name} is binary only and should be excluded for k=3"
    for name in ("most_common_value", "t_tuple", "lrs", "multimcw", "lag", "multimmc", "lz78y"):
        assert name in est, f"{name} should run on a non-binary alphabet"


# ---------- the hardware-bridge Source layer ----------
def test_bridge_adapters_agree(tmp_path=None):
    import tempfile, os
    bits = biased(8 * 5000, 0.6, seed=11).astype(np.uint8)
    packed = np.packbits(bits)
    path = os.path.join(tempfile.gettempdir(), "qrng_attest_test_capture.bin")
    with open(path, "wb") as f:
        f.write(packed.tobytes())
    a = assess(ArraySource(np.unpackbits(packed).astype(int), alphabet_size=2))
    b = assess(BytesFileSource(path, mode="bits"))
    c = assess(CallableSource(lambda n: np.unpackbits(packed).astype(int)[:n], len(np.unpackbits(packed))))
    assert abs(a.min_entropy - b.min_entropy) < 1e-9
    assert abs(a.min_entropy - c.min_entropy) < 1e-9


def test_synthetic_flagged_as_simulation():
    from qrng_attest import SyntheticSource
    a = assess(SyntheticSource(lambda rng, n: rng.integers(0, 2, n), N, name="syn"))
    assert any("synthetic" in f for f in a.describe["flags"])
