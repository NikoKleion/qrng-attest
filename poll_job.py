# Check the status of the job saved by submit_job.py. When it is done, build the certificate.
import sys

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from run_hardware import parse, QUBITS, SHOTS
from qrng_attest.quantum import calibrate_readout, QuantumArraySource
from qrng_attest.certify import certify_source


def main():
    try:
        name, jid = open("last_job.txt", encoding="utf-8").read().split()
    except FileNotFoundError:
        print("no last_job.txt found. run submit_job.py first.")
        return 1

    svc = QiskitRuntimeService()
    job = svc.job(jid)
    status = str(job.status())
    print("backend:", name)
    print("job id:", jid)
    print("status:", status)

    if "DONE" not in status.upper():
        print("not finished yet. run poll_job.py again in a bit.")
        return 0

    res = job.result()
    nq = len(QUBITS)
    prep0, prep1, main_bits = parse(res[0]), parse(res[1]), parse(res[2])
    Ms = []
    n_cal = SHOTS
    for i in range(nq):
        M, n = calibrate_readout(prep0[:, i], prep1[:, i])
        Ms.append(M)
        n_cal = min(n_cal, n)
    src = QuantumArraySource(main_bits, readout=Ms, calibration_shots=n_cal,
                             name=name + " QRNG", provenance="measured")
    print()
    print(certify_source(src))
    try:
        print()
        print("usage after run:", svc.usage())
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
