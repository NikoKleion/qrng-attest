# Run one small QRNG certification job.
# Dry run uses a local fake device and is free. Real run submits to an IBM QPU and asks you to
# confirm first, so it cannot bill you by accident.
#
# Free local dry run:
#   python run_hardware.py --dry-run
# Real hardware (asks for confirmation before submitting):
#   python run_hardware.py
#
# Change SHOTS or QUBITS below; keep them small to stay inside the free Open plan.
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2

from qrng_attest.quantum import calibrate_readout, QuantumArraySource
from qrng_attest.certify import certify_source

SHOTS = 2000
QUBITS = [0, 1]


def build_circuits(nq):
    # three circuits: prepare 0 and measure, prepare 1 and measure, prepare + and measure.
    c0 = QuantumCircuit(nq, nq)
    c0.measure(range(nq), range(nq))
    c1 = QuantumCircuit(nq, nq)
    for i in range(nq):
        c1.x(i)
    c1.measure(range(nq), range(nq))
    cm = QuantumCircuit(nq, nq)
    for i in range(nq):
        cm.h(i)
    cm.measure(range(nq), range(nq))
    return [c0, c1, cm]


def parse(result_item):
    # turn one result into a (shots, qubits) array of 0s and 1s, qubit 0 first.
    ba = next(iter(result_item.data.values()))
    return np.array([[int(x) for x in s.replace(" ", "")[::-1]] for s in ba.get_bitstrings()], int)


def run_on(backend, label):
    nq = len(QUBITS)
    circs = transpile(build_circuits(nq), backend, initial_layout=QUBITS)
    print("submitting to " + label + " ...")
    job = SamplerV2(mode=backend).run(circs, shots=SHOTS)
    try:
        print("job id:", job.job_id())
    except Exception:
        pass
    res = job.result()
    prep0, prep1, main = parse(res[0]), parse(res[1]), parse(res[2])
    Ms = []
    n_cal = SHOTS
    for i in range(nq):
        M, n = calibrate_readout(prep0[:, i], prep1[:, i])
        Ms.append(M)
        n_cal = min(n_cal, n)
    src = QuantumArraySource(main, readout=Ms, calibration_shots=n_cal,
                             name=str(label) + " QRNG", provenance="measured")
    print()
    print(certify_source(src))


def dry_run():
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    be = FakeManilaV2()
    print("DRY RUN on a local fake device. This is free and does not use your account.")
    run_on(be, be.name)
    return 0


def real_run():
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        svc = QiskitRuntimeService()
    except Exception as e:
        print("No saved IBM account found. Save it first, then run again. Details:")
        print(" ", e)
        return 1
    print("account channel:", svc.channel)
    try:
        print("instances:", svc.instances())
    except Exception:
        pass
    try:
        print("usage:", svc.usage())
    except Exception:
        pass
    be = svc.least_busy(operational=True, simulator=False)
    st = be.status()
    print("least busy backend:", be.name, "| qubits:", be.num_qubits, "| pending jobs:", st.pending_jobs)
    print()
    print("About to submit " + str(SHOTS) + " shots on qubits " + str(QUBITS) + " (three small circuits, one job).")
    print("This is free ONLY on the Open plan (10 minutes per 28 days). On a paid instance it is billed.")
    print("Check above that your instance is the Open plan before continuing.")
    ans = input("Type yes to confirm your instance is the free Open plan and submit: ").strip().lower()
    if ans != "yes":
        print("Not confirmed. Nothing was submitted.")
        return 0
    run_on(be, be.name)
    try:
        print("usage after run:", svc.usage())
    except Exception:
        pass
    return 0


def main(argv):
    if "--dry-run" in argv:
        return dry_run()
    return real_run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
