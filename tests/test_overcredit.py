# Tests pinning the over-credit experiment: the black-box track credits detector noise, the attestation
# track does not, and the two agree when the readout is clean.
import numpy as np

from qrng_attest import overcredit as oc


def test_observed_stream_matches_model():
    # the observed bias equals F0 p0 + (1-F1)(1-p0) within sampling error
    bits = oc.observed_stream(0.98, 0.6, 0.6, 60000, seed=0)
    p0_obs = float(np.mean(bits == 0))
    assert abs(p0_obs - (0.6 * 0.98 + 0.4 * 0.02)) < 0.01


def test_blackbox_overcredits_noisy_readout():
    # the central point: the stream passes the IID screen, the 800-90B credit is far above the true
    p = oc.run_point(p0_src=0.98, F=0.6, n_shots=60000, seed=2)
    assert p["iid_ok"], "the observed stream is Bernoulli-IID and must pass the IID screen"
    assert p["credited"] > 0.6, f"black-box credit should be large, got {p['credited']}"
    assert p["overcredit"] > 0.5, f"over-credit should exceed 0.5 bits/bit, got {p['overcredit']}"
    assert p["attested"] <= p["h_true"] + 0.02, "the attested bound must not exceed the true source entropy"
    assert p["readout_dominated"], "the attestation should flag the stream as readout-dominated"
    assert p["non_iid_min"] > p["h_true"] + 0.3


def test_attested_never_overcredits_across_sweep():
    for p in oc.sweep(n_shots=30000, seed=3):
        assert p["attested"] <= p["h_true"] + 0.02, f"over-credit at F={p['F']}: {p['attested']} > {p['h_true']}"


def test_tracks_agree_when_readout_clean():
    # with a near-perfect readout there is nothing to debit: the two tracks agree
    p = oc.run_point(p0_src=0.98, F=0.999, n_shots=60000, seed=5)
    assert abs(p["overcredit"]) < 0.05
    assert not p["readout_dominated"]


def test_recovered_bias_matches_source():
    p = oc.run_point(p0_src=0.98, F=0.7, n_shots=60000, seed=7)
    assert abs(p["p0_recovered"] - 0.98) < 0.02


# ---------- experiment 2: crosstalk ----------
def test_crosstalk_marginals_look_random_but_joint_is_not():
    p = oc.crosstalk_point(c=0.5, n_shots=60000, seed=30)
    assert all(p["marginals_iid"]), "both marginal streams are uniform-IID and must pass the screen"
    assert p["credited_rate"] > 0.9, "per-stream assessment credits close to full entropy"
    assert p["overcredit_rate"] > 0.2, "the per-stream credit exceeds the true joint rate"


def test_mi_debit_is_unsound_joint_bound_is_not():
    # the reason the attestation debits via the worst-pair joint bound instead of 1 - MI
    for c, seed in ((0.2, 40), (0.5, 41)):
        p = oc.crosstalk_point(c=c, n_shots=60000, seed=seed)
        assert p["mi_debit_rate"] > p["truth_rate"], f"1-MI should over-credit at c={c}"
        assert p["joint_bound_rate"] <= p["truth_rate"] + 1e-3, f"joint bound must not over-credit at c={c}"


def test_joint_bound_no_spurious_debit_when_independent():
    from qrng_attest.attest import pairwise_joint_bound
    rng = np.random.default_rng(50)
    shots = rng.integers(0, 2, (60000, 2))
    rate, _ = pairwise_joint_bound(shots)
    assert rate > 0.98


def test_attest_joint_debit_end_to_end():
    # full attestation on a crosstalked pair must flag it and not exceed the true joint rate
    import qrng_attest as es
    from qrng_attest.quantum import assignment_matrix
    a, b = oc.crosstalk_stream(0.5, 60000, seed=60)
    src = es.QuantumArraySource(np.stack([a, b], 1), readout=[assignment_matrix(0.999, 0.999)] * 2,
                                provenance="synthetic")
    att = es.attest(src, include_predictors=False)
    assert not att.passed and not att.independence["independent"]
    assert att.attested_min_entropy <= (1.0 - np.log2(1.5) / 2.0) + 1e-3


# ---------- experiment 3: mis-calibration ----------
def test_miscalibration_direction():
    # understating readout fidelity is safe; overstating it over-credits
    safe = oc.miscalibration_point(dF=-0.05, n_shots=60000, seed=70)
    assert safe["attested"] <= safe["h_true"] + 1e-9
    bad = oc.miscalibration_point(dF=+0.05, n_shots=60000, seed=70)
    assert bad["overcredit"] > 0.05, "an overstated fidelity must produce a visible over-credit"


def test_boxed_certificate_covers_margin_sized_errors():
    # the epsilon-bounded certificate stays safe for any fidelity error inside the propagated
    p = oc.miscalibration_point(dF=+0.02, n_shots=60000, seed=71, calib_shots=2000)
    assert p["calib_margin"] > 0.02, "the 2000-shot margin should cover a +0.02 error"
    assert p["overcredit"] > 0.02, "the point estimate should over-credit inside the margin"
    assert p["overcredit_boxed"] <= 1e-9, "the boxed bound must not over-credit inside the margin"


def test_boxed_certificate_never_looser_than_point():
    # propagating calibration uncertainty can only lower the bound
    for dF in (-0.03, 0.0, 0.03):
        p = oc.miscalibration_point(dF=dF, n_shots=40000, seed=72)
        assert p["attested_boxed"] <= p["attested"] + 1e-12


def test_boxed_certificate_matches_point_when_margin_zero():
    from qrng_attest.quantum import attested_qubit_entropy, assignment_matrix
    bits = oc.observed_stream(0.98, 0.8, 0.8, 40000, seed=73)
    M = assignment_matrix(0.8, 0.8)
    a = attested_qubit_entropy(bits, M)
    b = attested_qubit_entropy(bits, M, M_margin=0.0)
    assert abs(a["h_quantum"] - b["h_quantum"]) < 1e-12
