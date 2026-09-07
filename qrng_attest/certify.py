# Top-level helpers: from a source or backend to a certificate and extractable bits.
import numpy as np

from .attest import attest, qubit_subset, remediation
from .extract import extract as _extract, plan as _plan
from .qiskit_bridge import available, run_on_backend, quantum_source_from_backend


class Certificate:
    def __init__(self, attestation, subset, plan, epsilon, source_name):
        self.attestation = attestation
        self.subset = subset
        self.plan = plan
        self.epsilon = float(epsilon)
        self.source_name = source_name
        self.rate = attestation.attested_min_entropy
        self.passed = attestation.passed
        self.actions = remediation(attestation)

    def key_cost(self, key_bits=256):
        return self.plan.bits_for(key_bits)

    def __str__(self):
        a = self.attestation
        d = a.describe
        tracks = {q.get("classical_track", "non-IID") for q in a.per_qubit}
        if tracks == {"IID"}:
            track_line = "  classical estimate: IID track (source passed the IID checks)"
        elif "IID" in tracks:
            track_line = "  classical estimate: mixed (some qubits IID, some on the non-IID floor)"
        else:
            track_line = "  classical estimate: non-IID floor (IID checks not run or not passed)"
        L = [f"QRNG certificate: {self.source_name}  [{d['provenance']}]",
             f"  {d['n_shots']} shots x {d['n_qubits']} qubits",
             f"  attested min-entropy = {self.rate:.4f} bits/bit   "
             f"(verdict: {'PASS' if self.passed else 'FLAGGED'})",
             track_line,
             f"  extractable = {self.plan.output_bits} uniform bits at epsilon = 2^"
             f"{np.log2(self.epsilon):.0f}",
             f"  a 256-bit key costs {self.key_cost(256)} raw bits"]
        if self.subset is not None and self.subset["dropped"]:
            L.append(f"  best qubit subset {self.subset['best_qubits']} "
                     f"(drop {self.subset['dropped']}): "
                     f"{self.subset['base_bits_per_shot']:.3f} -> "
                     f"{self.subset['total_bits_per_shot']:.3f} bits/shot")
        for act in self.actions:
            L.append(f"  action: {act}")
        for f in d["flags"]:
            L.append(f"  note: {f}")
        return "\n".join(L)


def certify_source(source, epsilon=2.0 ** -64, include_predictors=False, select_qubits=True, run_iid=True):
    # run_iid uses the SP 800-90B IID assessment when a qubit passes the IID checks, which for a clean
    # near-uniform source lets the device physics set the bound instead of the non-IID sample-size floor.
    att = attest(source, include_predictors=include_predictors, run_iid=run_iid)
    sub = None
    if select_qubits and att.describe["n_qubits"] > 1:
        sub = qubit_subset(source, include_predictors=include_predictors, run_iid=run_iid)
    plan = att.extraction_plan(epsilon=epsilon)
    return Certificate(att, sub, plan, epsilon, att.describe["name"])


def certify_backend(backend, shots=8192, qubits=None, epsilon=2.0 ** -64, simulate=False,
                    calibration_shots=4000, include_predictors=False, run_iid=True):
    if not available():
        raise RuntimeError('qiskit is required for backend certification: pip install "qrng-attest[qiskit]"')
    if simulate:
        src = quantum_source_from_backend(backend, n_shots=shots, qubits=qubits, provenance="synthetic")
    else:
        src = run_on_backend(backend, n_shots=shots, qubits=qubits,
                             calibration_shots=calibration_shots)
    return certify_source(src, epsilon=epsilon, include_predictors=include_predictors, run_iid=run_iid)


def extract_from_source(source, epsilon=2.0 ** -64, require_pass=True, seed=None, include_predictors=False,
                        run_iid=True):
    # certify a source, then extract uniform bits at the attested rate.
    cert = certify_source(source, epsilon=epsilon, include_predictors=include_predictors,
                          select_qubits=False, run_iid=run_iid)
    if require_pass and not cert.passed:
        raise RuntimeError(f"source did not pass attestation; actions: {cert.actions}")
    raw = np.asarray(source.shots(), int).ravel().astype(np.uint8)
    rng = np.random.default_rng(seed)
    bits, hash_seed = _extract(raw, cert.rate, epsilon=epsilon, rng=rng)
    return bits, cert, hash_seed
