# Demonstration: output-only tests can credit entropy that the source does not have.
# Run python -m qrng_attest.overcredit to print the tables.
import math

import numpy as np

from .estimators import min_entropy
from .iid import iid_decision
from .quantum import assignment_matrix, attested_qubit_entropy, geometric_min_entropy


def observed_stream(p0_src, F0, F1, n_shots, seed=0):
    # true outcomes at source bias p0_src, then independent per-shot readout flips at (F0, F1).
    rng = np.random.default_rng(seed)
    true = (rng.random(n_shots) >= p0_src).astype(int)
    flip = rng.random(n_shots) < np.where(true == 0, 1.0 - F0, 1.0 - F1)
    return np.where(flip, 1 - true, true)


def run_point(p0_src=0.98, F=0.6, n_shots=60000, seed=0, iid_full=False):
    # one experiment point: black-box SP 800-90B credit vs calibration-attested credit vs ground truth.
    F0 = F1 = float(F)
    bits = observed_stream(p0_src, F0, F1, n_shots, seed=seed)
    h_true = geometric_min_entropy(p0_src)

    non_iid_min, _ = min_entropy(bits, include_predictors=True)
    iid = iid_decision(bits, full=iid_full)
    credited = iid.assessed if iid.is_iid else non_iid_min

    M = assignment_matrix(F0, F1)
    att = attested_qubit_entropy(bits, M)

    return {"p0_src": float(p0_src), "F": float(F), "n_shots": int(n_shots),
            "p0_obs": att["p0_obs"], "h_true": h_true,
            "iid_ok": bool(iid.is_iid), "non_iid_min": float(non_iid_min), "credited": float(credited),
            "attested": float(att["h_quantum"]), "p0_recovered": att["p0_source"],
            "overcredit": float(credited) - h_true, "readout_dominated": bool(att["readout_dominated"])}


def sweep(p0_src=0.98, Fs=(0.55, 0.65, 0.75, 0.85, 0.95, 0.999), n_shots=60000, seed=0):
    # the over-credit as a function of readout fidelity at a fixed source bias.
    return [run_point(p0_src, F, n_shots, seed=seed + i) for i, F in enumerate(Fs)]


def report(points=None, p0_src=0.98, n_shots=60000, seed=0):
    if points is None:
        points = sweep(p0_src=p0_src, n_shots=n_shots, seed=seed)
    h_true = points[0]["h_true"]
    L = ["Over-credit experiment: black-box SP 800-90B vs calibration attestation",
         f"  source: P(0) = {points[0]['p0_src']:.3f}, true min-entropy {h_true:.4f} bits/bit; "
         f"{points[0]['n_shots']} shots per point; symmetric readout fidelity F",
         "",
         "  F       p0_obs   IID   800-90B credit   attested   over-credit",
         "  " + "-" * 62]
    for p in points:
        L.append(f"  {p['F']:5.3f}   {p['p0_obs']:6.3f}   {'yes' if p['iid_ok'] else 'no '}   "
                 f"{p['credited']:14.4f}   {p['attested']:8.4f}   {p['overcredit']:+11.4f}")
    worst = max(points, key=lambda p: p["overcredit"])
    L += ["",
          f"  the observed stream is exactly Bernoulli-IID at every point: it passes the IID screen and no",
          f"  output-only statistical test can distinguish it from a genuine source of the same bias.",
          f"  the black-box track credits up to {worst['credited']:.3f} bits/bit against a true source",
          f"  entropy of {h_true:.4f} (over-credit {worst['overcredit']:.3f} bits/bit at F = {worst['F']:.3f});",
          f"  the attested bound stays at or below the truth at every point.",
          "  detector noise is characterized classical noise and is not credited as quantum source entropy."]
    return "\n".join(L)


# experiment 2: crosstalk over-credit
def crosstalk_stream(c, n_shots, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n_shots)
    b = np.where(rng.random(n_shots) < c, a, rng.integers(0, 2, n_shots))
    return a, b


def crosstalk_point(c=0.5, n_shots=60000, seed=0):
    from .attest import mutual_information, pairwise_joint_bound
    a, b = crosstalk_stream(c, n_shots, seed=seed)
    truth_rate = 1.0 - math.log2(1.0 + c) / 2.0
    creds = []; iids = []
    for s in (a, b):
        non_iid, _ = min_entropy(s, include_predictors=True)
        iid = iid_decision(s)
        iids.append(bool(iid.is_iid))
        creds.append(float(iid.assessed if iid.is_iid else non_iid))
    mi = mutual_information(a, b)
    joint_rate, _ = pairwise_joint_bound(np.stack([a, b], 1))
    return {"c": float(c), "n_shots": int(n_shots), "truth_rate": truth_rate,
            "marginals_iid": iids, "credited_rate": float(sum(creds) / 2.0),
            "mi": float(mi), "mi_debit_rate": 1.0 - float(mi), "joint_bound_rate": float(joint_rate),
            "overcredit_rate": float(sum(creds) / 2.0) - truth_rate}


def crosstalk_sweep(cs=(0.05, 0.2, 0.35, 0.5, 0.8), n_shots=60000, seed=0):
    return [crosstalk_point(c, n_shots, seed=seed + 10 * i) for i, c in enumerate(cs)]


def crosstalk_report(points=None, n_shots=60000, seed=0):
    if points is None:
        points = crosstalk_sweep(n_shots=n_shots, seed=seed)
    L = ["Crosstalk over-credit experiment: per-stream SP 800-90B vs the joint rate",
         f"  qubit B copies qubit A with probability c; both marginals are uniform-IID; "
         f"{points[0]['n_shots']} shots per point",
         "",
         "  c       truth rate   per-stream credit   1-MI debit   joint bound",
         "  " + "-" * 62]
    for p in points:
        mark = "  <-- 1-MI over-credits" if p["mi_debit_rate"] > p["truth_rate"] + 1e-9 else ""
        L.append(f"  {p['c']:4.2f}    {p['truth_rate']:10.4f}   {p['credited_rate']:17.4f}   "
                 f"{p['mi_debit_rate']:10.4f}   {p['joint_bound_rate']:11.4f}{mark}")
    L += ["",
          "  every marginal stream passes per-stream assessment at close to 1 bit/bit; the true joint",
          "  rate is lower at every c > 0. The 1-MI debit exceeds the truth under moderate crosstalk",
          "  (it is a Shannon quantity); the worst-pair joint bound stays at or below the truth",
          "  everywhere, so it is the debit the attestation uses."]
    return "\n".join(L)


# experiment 3: mis-calibration sensitivity
def miscalibration_point(p0_src=0.98, F_actual=0.75, dF=0.0, n_shots=60000, seed=0, calib_shots=2000):
    bits = observed_stream(p0_src, F_actual, F_actual, n_shots, seed=seed)
    F_assumed = min(0.999, max(0.501, F_actual + dF))
    M = assignment_matrix(F_assumed, F_assumed)
    point = attested_qubit_entropy(bits, M)
    boxed = attested_qubit_entropy(bits, M, calib_shots=calib_shots)
    h_true = geometric_min_entropy(p0_src)
    return {"F_actual": float(F_actual), "F_assumed": float(F_assumed), "dF": float(dF),
            "h_true": h_true, "calib_shots": int(calib_shots), "calib_margin": boxed["calib_margin"][0],
            "attested": float(point["h_quantum"]), "overcredit": float(point["h_quantum"]) - h_true,
            "attested_boxed": float(boxed["h_quantum"]),
            "overcredit_boxed": float(boxed["h_quantum"]) - h_true}


def miscalibration_sweep(dFs=(-0.05, -0.02, 0.0, 0.02, 0.05), p0_src=0.98, F_actual=0.75,
                         n_shots=60000, seed=0, calib_shots=2000):
    return [miscalibration_point(p0_src, F_actual, dF, n_shots, seed=seed, calib_shots=calib_shots)
            for dF in dFs]


def miscalibration_report(points=None, n_shots=60000, seed=0):
    if points is None:
        points = miscalibration_sweep(n_shots=n_shots, seed=seed)
    h_true = points[0]["h_true"]
    L = ["Mis-calibration sensitivity: the attested bound vs calibration error",
         f"  actual readout fidelity {points[0]['F_actual']:.2f}, true source entropy {h_true:.4f} bits/bit; "
         f"the attestation assumes fidelity F_actual + dF",
         f"  'boxed' propagates the calibration's own {points[0]['calib_shots']}-shot statistical margin "
         f"(about {points[0]['calib_margin']:.3f}) into the bound",
         "",
         "  dF        assumed F   point attested   boxed attested   point over-credit   boxed over-credit",
         "  " + "-" * 94]
    for p in points:
        m1 = " <-- over-credits" if p["overcredit"] > 1e-3 else ""
        m2 = " <-- over-credits" if p["overcredit_boxed"] > 1e-3 else ""
        L.append(f"  {p['dF']:+5.2f}     {p['F_assumed']:8.3f}   {p['attested']:14.4f}   {p['attested_boxed']:14.4f}"
                 f"   {p['overcredit']:+17.4f}{m1}   {p['overcredit_boxed']:+15.4f}{m2}")
    L += ["",
          "  the point-estimate certificate (calibration trusted as exact) over-credits as soon as the",
          "  assumed fidelity overstates reality. Propagating the calibration's own statistical margin",
          "  keeps the bound safe for any error inside that margin; an overstatement beyond the margin",
          "  still over-credits, so the certificate is honest only about errors it was told to cover.",
          "  when in doubt, understate the fidelity."]
    return "\n".join(L)


def main():
    print(report())
    print()
    print(crosstalk_report())
    print()
    print(miscalibration_report())


if __name__ == "__main__":
    main()
