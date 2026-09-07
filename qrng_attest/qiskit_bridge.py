# qrng_attest.qiskit_bridge: build sources, readout matrices, and calibration from a Qiskit backend.
import numpy as np

try:
    import qiskit
    _HAVE = True
except Exception:
    _HAVE = False


def available():
    return _HAVE


def _require():
    if not _HAVE:
        raise RuntimeError("qiskit_bridge needs qiskit (install in the project venv)")


def gate_duration(backend, name, qubits, default=None):
    # gate duration in seconds, from the BackendV2 Target first, then BackendV1 properties.
    try:
        inst = backend.target[name][tuple(qubits)]
        if inst is not None and inst.duration is not None:
            return float(inst.duration)
    except Exception:
        pass
    try:
        return float(backend.properties().gate_length(name, list(qubits)))
    except Exception:
        return default


def readout_matrix_from_backend(backend, qubit):
    # readout matrix from backend calibration. BackendV2 gives only a symmetric error; prefer self_calibrate_backend.
    _require()
    from .quantum import assignment_matrix
    try:
        p = backend.properties()
        F0 = 1.0 - float(p.qubit_property(qubit, "prob_meas1_prep0")[0])
        F1 = 1.0 - float(p.qubit_property(qubit, "prob_meas0_prep1")[0])
        return assignment_matrix(F0, F1)
    except Exception:
        pass
    try:
        err = float(backend.target["measure"][(qubit,)].error)
        return assignment_matrix(1.0 - err, 1.0 - err)
    except Exception:
        raise RuntimeError("could not read readout error for qubit " + str(qubit)
                           + "; use self_calibrate_backend so the certificate is not based on assumed perfect readout")


def self_calibrate_backend(backend, qubits=None, shots=4000):
    # measure the assignment matrix on the device itself from |0> and |1> preparations.
    _require()
    from qiskit import QuantumCircuit, transpile
    from .quantum import calibrate_readout
    qubits = list(range(backend.num_qubits)) if qubits is None else list(qubits)
    nq = len(qubits)
    outcomes = {}
    for prep in (0, 1):
        qc = QuantumCircuit(nq, nq)
        if prep == 1:
            for i in range(nq):
                qc.x(i)
        qc.measure(range(nq), range(nq))
        tqc = transpile(qc, backend, initial_layout=qubits)
        mem = _execute_shots(backend, tqc, shots)
        outcomes[prep] = np.array([[int(x) for x in m.replace(" ", "")[::-1]] for m in mem], int)
    Ms = []
    n_cal = shots
    for i in range(nq):
        M, n = calibrate_readout(outcomes[0][:, i], outcomes[1][:, i])
        Ms.append(M); n_cal = min(n_cal, n)
    return Ms, int(n_cal)


def quantum_source_from_backend(backend, n_shots=20000, qubits=None, seed=0, provenance="measured"):
    # run a |+>-prepared QRNG circuit under the backend's Aer noise model, returning a QuantumSource with readout matrices.
    _require()
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from .quantum import QuantumArraySource

    qubits = list(range(backend.num_qubits)) if qubits is None else list(qubits)
    nq = len(qubits)
    qc = QuantumCircuit(nq, nq)
    for i in range(nq):
        qc.h(i)
    qc.measure(range(nq), range(nq))
    sim = AerSimulator(noise_model=NoiseModel.from_backend(backend), seed_simulator=seed)
    tqc = transpile(qc, sim, initial_layout=qubits)
    mem = sim.run(tqc, shots=n_shots, memory=True).result().get_memory()
    shots = np.array([[int(x) for x in s.replace(" ", "")[::-1]] for s in mem], int)
    Ms = [readout_matrix_from_backend(backend, q) for q in qubits]
    name = getattr(backend, "name", "qiskit-backend")
    name = name() if callable(name) else name
    return QuantumArraySource(shots, readout=Ms, name=f"{name} QRNG", provenance=provenance)


def _execute_shots(backend, tqc, n_shots):
    # get ordered per-shot bitstrings from the backend: try the runtime SamplerV2, then backend.run(memory=True)
    try:
        from qiskit_ibm_runtime import SamplerV2
        res = SamplerV2(mode=backend).run([tqc], shots=n_shots).result()
        bitarray = next(iter(res[0].data.values()))
        return bitarray.get_bitstrings()
    except Exception:
        return backend.run(tqc, shots=n_shots, memory=True).result().get_memory()


def run_on_backend(backend, n_shots=4096, qubits=None, provenance="measured", self_calibrate=True,
                   calibration_shots=4000):
    # run a |+>-prepared QRNG circuit ON the backend itself (real QPU or its runtime), returning a QuantumSource of
    _require()
    from qiskit import QuantumCircuit, transpile
    from .quantum import QuantumArraySource
    qubits = list(range(backend.num_qubits)) if qubits is None else list(qubits)
    nq = len(qubits)
    n_cal = None
    if self_calibrate:
        Ms, n_cal = self_calibrate_backend(backend, qubits=qubits, shots=calibration_shots)
    else:
        Ms = [readout_matrix_from_backend(backend, q) for q in qubits]
    qc = QuantumCircuit(nq, nq)
    for i in range(nq):
        qc.h(i)
    qc.measure(range(nq), range(nq))
    tqc = transpile(qc, backend, initial_layout=qubits)
    mem = _execute_shots(backend, tqc, n_shots)
    shots = np.array([[int(x) for x in s.replace(" ", "")[::-1]] for s in mem], int)
    name = getattr(backend, "name", "qpu")
    name = name() if callable(name) else name
    return QuantumArraySource(shots, readout=Ms, name=f"{name} QRNG (measured)", provenance=provenance,
                              calibration_shots=n_cal)
