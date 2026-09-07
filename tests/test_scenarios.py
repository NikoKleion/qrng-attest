# End-to-end scenarios: capture files, device certificates, subset selection, extraction, and the CLI.
import os, tempfile
import numpy as np
import qrng_attest as qa


def test_end_to_end_scenarios():
    def check(name, cond):
        assert cond, name
    tmp = tempfile.gettempdir()


    # 1. good uniform bit capture
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 40000).astype(np.uint8)
    p = os.path.join(tmp, "good.bin"); open(p, "wb").write(np.packbits(bits).tobytes())
    r = qa.assess(qa.BytesFileSource(p, mode="bits"))
    check("good capture assessed high", r.min_entropy > 0.6)
    check("good capture has extraction plan", r.extraction_plan().output_bits > 1000)

    # 2. biased but independent capture
    b = (rng.random(40000) < 0.7).astype(np.uint8)
    p = os.path.join(tmp, "biased.bin"); open(p, "wb").write(np.packbits(b).tobytes())
    r = qa.assess(qa.BytesFileSource(p, mode="bits"), run_iid=True)
    check("biased capture below one bit", r.min_entropy < 0.9)
    check("biased-independent flagged IID", r.iid is not None and r.iid.is_iid)

    # 3. correlated capture caught
    s = np.empty(40000, int); s[0] = 0
    flip = rng.random(40000) < 0.1
    for i in range(1, 40000):
        s[i] = s[i-1] if not flip[i] else 1 - s[i-1]
    r = qa.assess(qa.ArraySource(s, alphabet_size=2))
    check("sticky source caught", r.min_entropy < 0.3)

    # 4. multi-bit byte capture
    sym = rng.integers(0, 16, 10000).astype(np.uint8)
    p = os.path.join(tmp, "bytes.bin"); open(p, "wb").write(sym.tobytes())
    r = qa.assess(qa.BytesFileSource(p, mode="bytes"))
    check("multi-bit assessed within range", 0 < r.min_entropy <= 4.0 + 1e-9)

    # 5. healthy synthetic device certificate
    shots, M = qa.synthetic_qrng(8000, 4, seed=1)
    cert = qa.certify_source(qa.QuantumArraySource(shots, readout=M, calibration_shots=4000, name="healthy"))
    check("healthy device passes", cert.passed)
    check("healthy device credits bits", cert.plan.output_bits > 0)

    # 6. bad device flagged with actions
    shots, M = qa.synthetic_qrng(8000, 4, crosstalk=0.5, drift=0.2, seed=2)
    cert = qa.certify_source(qa.QuantumArraySource(shots, readout=M, name="bad"))
    check("bad device flagged", not cert.passed)
    check("bad device gives actions", len(cert.actions) >= 1 and all(isinstance(a, str) for a in cert.actions))

    # 7. one bad qubit, subset selection recovers throughput
    shots, M = qa.synthetic_qrng(6000, 4, seed=3)
    shots[:, 2] = (rng.random(6000) > 0.85).astype(int)
    src = qa.QuantumArraySource(shots, readout=M, calibration_shots=4000)
    sub = qa.qubit_subset(src)
    check("subset drops the bad qubit", 2 in sub["dropped"])
    check("subset raises throughput", sub["total_bits_per_shot"] > sub["base_bits_per_shot"])

    # 8. self-calibration recovers known fidelities
    from qrng_attest.quantum import calibrate_readout
    prep0 = (rng.random(20000) > 0.95).astype(int)
    prep1 = (rng.random(20000) < 0.9).astype(int)
    Mc, nc = calibrate_readout(prep0, prep1)
    check("self-calibration recovers F0", abs(Mc[0, 0] - 0.95) < 0.01)
    check("self-calibration recovers F1", abs(Mc[1, 1] - 0.9) < 0.01)

    # 9. end to end extraction gives near-uniform bits
    shots, M = qa.synthetic_qrng(20000, 2, seed=4)
    out, cert, seed = qa.extract_from_source(qa.QuantumArraySource(shots, readout=M, calibration_shots=8000),
                                             require_pass=False, seed=7)
    check("extraction produced bits", len(out) > 500)
    check("extracted bits near uniform", abs(out.mean() - 0.5) < 0.06)
    _, e = qa.min_entropy(out.astype(int))
    check("extracted stream marginal near uniform", e["most_common_value"] > 0.9)

    # 10. extraction planner sizing
    plan = qa.extraction_plan(n_input=1000000, h_per_symbol=0.68)
    check("planner sizes a 256 bit key", 300 < plan.bits_for(256) < 700)

    # 11. degenerate inputs do not crash
    for bad in (np.array([], int), np.zeros(500, int), np.array([5])):
        m, _ = qa.min_entropy(bad)
        check("degenerate input handled: len %d" % len(bad), (len(bad) == 0) or np.isfinite(m))

    # 12. backend paths fail gracefully without qiskit
    from qrng_attest.qiskit_bridge import available
    check("qiskit availability reported", isinstance(available(), bool))
    if not available():
        try:
            qa.certify_backend(object(), simulate=True); crashed = False
        except RuntimeError:
            crashed = True
        check("certify_backend errors cleanly without qiskit", crashed)

    # 13. CLI: demo and capture
    from qrng_attest.__main__ import main
    check("cli demo runs", main(["demo", "--shots", "2000", "--qubits", "2"]) == 0)
    p = os.path.join(tmp, "good.bin")
    check("cli capture runs", main(["capture", p, "--no-iid"]) == 0)
    check("cli help runs", main([]) == 0)

