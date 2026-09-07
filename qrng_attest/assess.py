# Run the SP 800-90B assessment on any Source and report the min-entropy.
import numpy as np

from . import estimators as _E
from .sources import Source, from_any

_META = ("H_original", "H_bitstring", "word_size")


class Assessment:
    def __init__(self, min_h, per_estimator, describe, iid=None):
        self.non_iid_min = min_h
        self.iid = iid
        self.min_entropy = iid.assessed if (iid is not None and iid.is_iid) else min_h
        self.per_estimator = per_estimator
        self.describe = describe
        self.multibit = "H_bitstring" in per_estimator
        self.meta = {k: per_estimator[k] for k in _META if k in per_estimator}
        self.scores = {k: v for k, v in per_estimator.items() if k not in _META}
        self.driver = min(self.scores, key=self.scores.get) if self.scores else None

    def extraction_plan(self, epsilon=2.0 ** -64):
        from .extract import plan
        note = None if self.iid is not None else "non-IID minimum used; run the IID track for a tighter rate"
        return plan(self.describe["length"], self.min_entropy, epsilon=epsilon, note=note)

    @property
    def marginal(self):
        # the most-common-value (per-position) estimate, whichever track carries it
        for key in ("most_common_value", "bitstring:most_common_value"):
            if key in self.per_estimator:
                return self.per_estimator[key]
        return None

    def __str__(self):
        d = self.describe
        L = [f"SP 800-90B assessment: {d['name']}  [{d['provenance']}]",
             f"  samples {d['length']}, alphabet {d['alphabet_size']}"]
        if self.multibit:
            L.append(f"  multi-bit: {int(self.meta['word_size'])} bits/symbol, "
                     f"H_original {self.meta['H_original']:.4f} bits/symbol, "
                     f"H_bitstring {self.meta['H_bitstring']:.4f} bits/bit")
        iid_ok = self.iid is not None and self.iid.is_iid
        if iid_ok:
            L.append(f"  min-entropy = {self.min_entropy:.4f} bits/sample   "
                     f"(IID, most-common-value; non-IID minimum {self.non_iid_min:.4f} is a conservative floor)")
        else:
            L.append(f"  min-entropy = {self.min_entropy:.4f} bits/sample   (driven by {self.driver})")
        L.append("  per-estimator (bits):")
        for k in sorted(self.scores):
            mark = "  <-- min" if k == self.driver else ""
            L.append(f"    {k:>28}: {self.scores[k]:.4f}{mark}")
        for f in d["flags"]:
            L.append(f"  note: {f}")
        if self.iid is not None:
            L.append("  " + str(self.iid).replace("\n", "\n  "))
        pl = self.extraction_plan()
        L.append(f"  extractable: {pl.output_bits} uniform bits from these {pl.n_input} samples "
                 f"(epsilon 2^-64); a 256-bit key needs {pl.bits_for(256)} samples")
        marg = self.marginal
        if not iid_ok and marg is not None and marg - self.min_entropy > 0.2:
            L.append(f"  the marginal (most-common-value) reads {marg:.2f} bits, but the source is only "
                     f"{self.min_entropy:.2f} bits/sample; the extra apparent entropy is predictable structure that "
                     f"an IID/per-position test would have missed.")
        return "\n".join(L)


def assess(source, include_predictors=True, run_iid=False, iid_full=False, **kw):
    src = source if isinstance(source, Source) else from_any(source, **kw)
    S = np.asarray(src.samples()).astype(int)
    min_h, per = _E.min_entropy(S, include_predictors=include_predictors)
    iid = None
    if run_iid or iid_full:
        from .iid import iid_decision
        iid = iid_decision(S, full=iid_full)
    return Assessment(min_h, per, src.describe(), iid=iid)
