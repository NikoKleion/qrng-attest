# qrng_attest CLI: certify a Qiskit backend, a capture file, or the built-in demo source.
import argparse
import sys

import numpy as np


def _demo(args):
    from . import synthetic_qrng, QuantumArraySource
    from .certify import certify_source
    shots, M = synthetic_qrng(args.shots, args.qubits, source_bias=args.bias,
                              crosstalk=args.crosstalk, drift=args.drift, seed=args.seed)
    src = QuantumArraySource(shots, readout=M, provenance="synthetic",
                             calibration_shots=args.calibration_shots, name="demo QRNG")
    print(certify_source(src))


def _backend(args):
    from .qiskit_bridge import available
    from .certify import certify_backend
    if not available():
        print('qiskit is required: pip install "qrng-attest[qiskit]"'); return 1
    backend = _load_backend(args.name)
    if backend is None:
        print(f"backend {args.name!r} not found"); return 1
    print(certify_backend(backend, shots=args.shots, simulate=args.simulate))


def _load_backend(name):
    try:
        import qiskit_ibm_runtime.fake_provider as fp
    except Exception:
        return None
    for cand in (name, "Fake" + name.capitalize() + "V2", "Fake" + name + "V2"):
        cls = getattr(fp, cand, None)
        if cls is not None:
            return cls()
    return None


def _capture(args):
    from . import assess, BytesFileSource
    print(assess(BytesFileSource(args.path, mode="bytes" if args.bytes else "bits"),
                 run_iid=not args.no_iid))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qrng-attest",
                                 description="entropy assessment and device attestation for quantum RNG")
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="certify a built-in synthetic QRNG")
    d.add_argument("--shots", type=int, default=8000)
    d.add_argument("--qubits", type=int, default=4)
    d.add_argument("--bias", type=float, default=0.0)
    d.add_argument("--crosstalk", type=float, default=0.0)
    d.add_argument("--drift", type=float, default=0.0)
    d.add_argument("--calibration-shots", type=int, default=4000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=_demo)

    b = sub.add_parser("backend", help="certify a Qiskit backend by name")
    b.add_argument("name")
    b.add_argument("--shots", type=int, default=8192)
    b.add_argument("--simulate", action="store_true", help="run under the Aer noise model instead of the device")
    b.set_defaults(fn=_backend)

    c = sub.add_parser("capture", help="assess a captured RNG file")
    c.add_argument("path")
    c.add_argument("--bytes", action="store_true")
    c.add_argument("--no-iid", action="store_true")
    c.set_defaults(fn=_capture)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 0
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
