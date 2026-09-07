# Optional cross-check against NIST's ea_non_iid tool, if it is installed.
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np


def find_reference_tool(path=None):
    cand = path or os.environ.get("QRNG_ATTEST_NIST_EA") or shutil.which("ea_non_iid")
    return cand if cand and os.path.exists(cand) else None


def available(path=None):
    return find_reference_tool(path) is not None


def _parse(out):
    # parse ea_non_iid stdout: per-estimator estimate lines plus the final assessed min.
    per = {}
    for line in out.splitlines():
        m = re.search(r"([A-Za-z0-9 /+-]+?)\s*(?:Estimate|estimate)\s*[:=]\s*([0-9.]+)", line)
        if m:
            per[m.group(1).strip().lower().replace(" ", "_")] = float(m.group(2))
    a = re.search(r"[Aa]ssessed\s+min\s+entropy\s*[:=]?\s*([0-9.]+)", out)
    if not a:
        a = re.search(r"min\(H_original,\s*[^)]*\)\s*=\s*([0-9.]+)", out)
    assessed = float(a.group(1)) if a else (min(per.values()) if per else None)
    return {"per_estimator": per, "assessed": assessed}


def run_reference(samples, tool=None, bits_per_symbol=1, timeout=900):
    # run the NIST tool on the samples (one byte per sample); returns the parsed result or None.
    tool = find_reference_tool(tool)
    if tool is None:
        return None
    arr = np.asarray(samples).ravel().astype(int)
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(int(x) & 0xFF for x in arr))
        proc = subprocess.run([tool, "-i", "-a", path], capture_output=True, text=True, timeout=timeout)
        return _parse(proc.stdout + "\n" + proc.stderr)
    except Exception:
        return None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def cross_check(source, tool=None):
    # run this package and the NIST tool, and report the largest per-estimator difference.
    from .assess import assess
    from .sources import Source, from_any
    src = source if isinstance(source, Source) else from_any(source)
    ours = assess(src)
    ref = run_reference(src.samples(), tool=tool)
    if ref is None:
        return {"qrng_attest": ours.per_estimator, "qrng_attest_min": ours.min_entropy,
                "nist": None, "note": "NIST reference tool not found; qrng-attest assessment is faithful but not certified"}
    gaps = {k: abs(ours.per_estimator[k] - ref["per_estimator"][k])
            for k in ours.per_estimator if k in ref["per_estimator"]}
    return {"qrng_attest": ours.per_estimator, "qrng_attest_min": ours.min_entropy,
            "nist": ref["per_estimator"], "nist_assessed": ref["assessed"],
            "max_gap": max(gaps.values()) if gaps else None,
            "note": "cross-checked against the NIST reference tool"}
