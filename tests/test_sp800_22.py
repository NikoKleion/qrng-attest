# the optional SP 800-22 battery wrapper; tests that need nistrng skip without it
import numpy as np

from _harness import biased, needs, uniform
from qrng_attest import sp800_22


def test_available_is_boolean():
    assert isinstance(sp800_22.available(), bool)


def test_report_lists_every_row():
    rows = [{"test": "monobit", "eligible": True, "passed": True, "score": 0.5},
            {"test": "random_excursion", "eligible": False, "passed": None, "score": None}]
    text = sp800_22.report(rows)
    assert "monobit" in text and "random_excursion" in text


def test_rejects_non_binary_input():
    needs("nistrng")
    try:
        sp800_22.run_battery(np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_monobit_passes_uniform_and_fails_biased():
    needs("nistrng")
    good = {r["test"]: r for r in sp800_22.run_battery(uniform(100000, seed=1))}
    bad = {r["test"]: r for r in sp800_22.run_battery(biased(100000, p=0.6, seed=1))}
    assert good["monobit"]["passed"] is True and bad["monobit"]["passed"] is False


NIST_EPS = "11001001000011111101101010100010001000010110100011" "00001000110100110001001100011001100010100010111000"


def test_nist_sp800_22_worked_examples():
    # NIST SP 800-22 rev 1a sections 2.1.8, 2.3.8 and 2.13.8 on the 100-bit example sequence
    needs("nistrng")
    bits = np.array([int(c) for c in NIST_EPS])
    rows = {r["test"]: r for r in sp800_22.run_battery(bits, names=["monobit", "runs", "cumulative sums"],
                                                        check_eligibility=False)}
    assert abs(rows["monobit"]["score"] - 0.109599) < 1e-6, rows["monobit"]
    assert abs(rows["runs"]["score"] - 0.500798) < 1e-6, rows["runs"]
    # nistrng reports the mean of the forward (0.219194) and reverse (0.114866) P-values
    assert abs(rows["cumulative sums"]["score"] - (0.219194 + 0.114866) / 2) < 1e-6, rows["cumulative sums"]


def test_long_stream_has_no_integer_overflow():
    needs("nistrng")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sp800_22.run_battery(uniform(100000, seed=2), names=["cumulative sums"])
