# qrng_attest.sp800_22: optional NIST SP 800-22 rev 1a battery through the nistrng package (Pasqualini, BSD-3)
import numpy as np


def available():
    try:
        import nistrng  # noqa: F401
        return True
    except Exception:
        return False


def run_battery(bits, names=None, check_eligibility=True):
    # one row per SP 800-22 test, all fifteen unless names is given; eligible is False when the stream is too
    # short for that test. int64 input: nistrng accumulates in the input dtype and int8 overflows
    from nistrng import SP800_22R1A_BATTERY, run_by_name_battery
    bits = np.asarray(bits, dtype=np.int64).ravel()
    if bits.size and not np.isin(bits, (0, 1)).all():
        raise ValueError("bits must contain only 0 and 1")
    rows = []
    for name in (names if names is not None else SP800_22R1A_BATTERY):
        out = run_by_name_battery(name, bits, SP800_22R1A_BATTERY, check_eligibility)
        if out is None:
            rows.append({"test": name, "eligible": False, "passed": None, "score": None})
            continue
        result, _elapsed = out
        rows.append({"test": name, "eligible": True, "passed": bool(result.passed), "score": float(result.score)})
    return rows


def report(rows):
    # plain-text table of run_battery rows
    lines = [f"{'test':36} {'eligible':>8} {'passed':>7} {'score':>9}"]
    for r in rows:
        score = "" if r["score"] is None else f"{r['score']:.4f}"
        passed = "" if r["passed"] is None else ("yes" if r["passed"] else "no")
        lines.append(f"{r['test']:36} {('yes' if r['eligible'] else 'no'):>8} {passed:>7} {score:>9}")
    return "\n".join(lines)
