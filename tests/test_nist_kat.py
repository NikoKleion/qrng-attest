# Known-answer validation against the NIST SP 800-90B reference vectors and its ea_non_iid output.
import math
import os
import numpy as np

from _harness import skip
from qrng_attest import estimators as E

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nist_vectors")

NIST_RAND1 = {
    "most_common_value": 0.96105882570055079, "collision": 0.69146412099724186,
    "markov": 0.98759610445940904, "compression": 0.61171620479394506,
    "t_tuple": 0.86762443081707374, "lrs": 0.96262580383278706,
    "multimcw": 0.95261806053262654, "lag": 0.94333370657577120,
    "multimmc": 0.96161666782888089, "lz78y": 0.96144624595247796,
}
EXACT = ("most_common_value", "collision", "markov", "compression", "t_tuple", "lrs")
PREDICTORS = ("multimcw", "lag", "multimmc", "lz78y")

NIST_RAND4 = {
    "assessed": 3.2154882678767365, "H_original": 3.5674726723995995, "H_bitstring": 0.80387206696918412,
    "bitstring:most_common_value": 0.97918948296240216, "bitstring:collision": 0.89817944838102126,
    "bitstring:markov": 0.99061680777094752, "bitstring:compression": 0.80387206696918412,
    "bitstring:t_tuple": 0.8987772293903773, "bitstring:lrs": 0.93296931449533627,
    "literal:most_common_value": 3.7900373902139739, "literal:t_tuple": 3.5674726723995995,
    "literal:lrs": 3.8335255222329829,
}

NIST_RAND8 = {
    "assessed": 5.8608937444852494, "H_bitstring": 0.73261171806065617,
    "bitstring:most_common_value": 0.98338678465915019, "bitstring:collision": 0.83205298221524882,
    "bitstring:markov": 0.99772497672796534, "bitstring:compression": 0.73261171806065617,
    "bitstring:t_tuple": 0.91078644573541423, "bitstring:lrs": 0.98193035773637416,
}


def _load(name):
    path = os.path.join(_DIR, name)
    if not os.path.exists(path):
        skip(f"NIST vector {name} not present (download from the NIST repo to enable this KAT)")
    return np.frombuffer(open(path, "rb").read(), np.uint8).astype(int)


def test_nist_mcv_micro_example():
    # MCV is hand-computable: L=100, 60 ones, H = -log2(p_u)
    S = np.array([1] * 60 + [0] * 40)
    assert abs(E.most_common_value(S, k=2) - 0.460167) < 2e-3


def test_nist_rand1_binary_exact():
    # the six non-predictor estimators reproduce the reference tool to < 1e-9 bits
    est = E.all_estimators(_load("rand1_short.bin"), k=2)
    for name in EXACT:
        assert abs(est[name] - NIST_RAND1[name]) < 1e-9, f"{name}: {est[name]!r} vs {NIST_RAND1[name]!r}"
    for name in PREDICTORS:
        assert abs(est[name] - NIST_RAND1[name]) < 0.02, f"{name}: {est[name]!r} vs {NIST_RAND1[name]!r}"


def test_nist_rand1_assessed():
    mn, _ = E.min_entropy(_load("rand1_short.bin"))
    assert abs(mn - 0.61171620479394506) < 1e-9


def test_nist_rand4_multibit():
    # multi-bit assessment: literal (per-symbol) and bitstring (per-bit) tracks combined
    mn, per = E.min_entropy(_load("rand4_short.bin"))
    assert abs(mn - NIST_RAND4["assessed"]) < 1e-9
    assert abs(per["H_original"] - NIST_RAND4["H_original"]) < 1e-9
    assert abs(per["H_bitstring"] - NIST_RAND4["H_bitstring"]) < 1e-9
    for key, ref in NIST_RAND4.items():
        if ":" in key:
            assert abs(per[key] - ref) < 1e-9, f"{key}: {per[key]!r} vs {ref!r}"
    assert abs(mn - 4.0 * per["H_bitstring"]) < 1e-9


def test_nist_rand8_multibit():
    mn, per = E.min_entropy(_load("rand8_short.bin"))
    assert abs(mn - NIST_RAND8["assessed"]) < 1e-9
    assert abs(per["H_bitstring"] - NIST_RAND8["H_bitstring"]) < 1e-9
    for key, ref in NIST_RAND8.items():
        if ":" in key:
            assert abs(per[key] - ref) < 1e-9, f"{key}: {per[key]!r} vs {ref!r}"
    assert abs(mn - 8.0 * per["H_bitstring"]) < 1e-9
