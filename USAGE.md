# Using qrng-attest

## Overview

qrng-attest estimates the min-entropy of a quantum random source and extracts uniform bits at that rate.
The observed output includes measurement noise, cross-qubit correlation and device drift, which the
attestation checks separately.

## Checks

The entropy math is checked against NIST's published test vectors and a test suite over synthetic data
and captured files. Reading a bit stream runs the same path whether the bits come from a file or a device.

The backend path runs on IBM ibm_marrakesh through qiskit-ibm-runtime: job submission, per-qubit bit
order, readout self-calibration, result parsing, and the certificate. It has run on one device with two
qubits, so on a different backend or with more qubits, check the per-qubit bit order and the calibration
fields your backend version exposes.

## Terms

- **Min-entropy.** H_min = -log2 max_x P(x), in bits per sample (NIST SP 800-90B, section 3.1). A fair coin
  gives 1 bit per flip. This is the quantity the package reports.
- **SP 800-90B assessment.** The non-IID estimators of section 6.3 and the IID track of section 5 of NIST
  SP 800-90B. IID means independent and identically distributed samples. The assessed value is the minimum
  over the estimators; a source that passes the IID track is assessed by the most common value estimate
  (section 6.1).
- **Readout error, assignment matrix.** M[a, b] = P(read a | prepared b) for one qubit, a 2x2 matrix
  measured from |0> and |1> preparations. Readout error adds randomness that the quantum state did not
  produce. The same matrix is the calibration matrix of measurement error mitigation (Bravyi et al., Phys.
  Rev. A 103, 042605 (2021)).
- **Attested min-entropy.** The min-entropy of the source distribution obtained by
  inverting the assignment matrix out of the observed bit frequencies, taken at the worst case over a 99
  percent one-sided confidence region for the frequencies and for the calibration, after the cross-qubit
  correlation and drift checks. Under the model it is a lower bound on the source min-entropy at that
  confidence. The
  single-qubit form follows the Bloch-vector bound of Fiorentino et al., Phys. Rev. A 75, 032334 (2007).
- **Extraction.** A randomness extractor maps n input bits with min-entropy k to ell output bits within
  epsilon of uniform. The package uses Toeplitz hashing (Krawczyk, CRYPTO 1994) sized by the leftover hash
  lemma, ell = k - 2 log2(1 / epsilon) + 2 (Impagliazzo, Levin and Luby, STOC 1989).
- **Certificate, PASS, FLAGGED.** The printed report. FLAGGED means one check failed, and
  the report lists an action for it.

## Install

```bash
pip install qrng-attest
```

Requires Python 3.11 or newer and numpy. Qiskit is optional and only needed to certify a backend. Extras
install it:

```bash
pip install "qrng-attest[qiskit]"    # backend bridge
pip install "qrng-attest[runtime]"   # run on IBM hardware through qiskit-ibm-runtime
pip install "qrng-attest[aer]"       # run under a local noise model
pip install "qrng-attest[sp80022]"   # NIST SP 800-22 battery through nistrng
```

Assessing a captured file or an array needs only numpy.

## Demo

This runs a simulated source and prints a certificate. It needs no hardware.

```bash
python -m qrng_attest demo --qubits 4
```

Add flags to inject problems and see them caught:

```bash
python -m qrng_attest demo --qubits 4 --crosstalk 0.5 --drift 0.2
```

## Assess a file of captured bits

If you have raw output saved to a file, assess it directly.

```bash
python -m qrng_attest capture mydata.bin
```

The file is read as a bit stream by default. If each byte is a symbol, add `--bytes`. The report gives the
min-entropy, which estimator set the limit, and how many uniform bits you can extract.

From Python:

```python
from qrng_attest import assess

report = assess("mydata.bin", mode="bits", run_iid=True)
print(report)
print(report.min_entropy)
```

## Certify a Qiskit backend

This prepares |+> on each qubit of the backend, measures the readout error on the same device, attests
the result, and reports how many uniform bits it supports. It runs the IID track by default; on a
near-uniform source that passes it, the readout error sets the attested rate. Pass `run_iid=False` to use
the non-IID estimate.

```python
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qrng_attest import certify_backend

cert = certify_backend(FakeManilaV2(), shots=8192, simulate=True)
print(cert)
print(cert.rate)            # attested bits per bit
print(cert.passed)          # True if the source cleared every check
print(cert.key_cost(256))   # raw bits needed for a 256 bit key
```

Set `simulate=True` to run under a local noise model. Leave it out or set it to False to run on the real
device, which needs the runtime extra and waits in the queue.

## Self-calibration

The package measures the readout error by preparing known 0 and 1 states and counting how often they read
back wrong. The Qiskit backend interface does not expose the readout error in each direction, and vendor
calibration can be older than the run. The calibration shot count sets the uncertainty of the measured
matrix, and the reported entropy is the worst case over that uncertainty. More calibration shots give a
smaller uncertainty.

`certify_backend` performs this step for you. To do it by hand:

```python
from qrng_attest.qiskit_bridge import self_calibrate_backend

matrices, shots = self_calibrate_backend(backend, shots=4000)
```

## Qubit selection

One bad qubit lowers the whole device's rate. The package reports which subset of qubits gives the most
total bits per shot.

```python
from qrng_attest import qubit_subset

result = qubit_subset(source)
print(result["best_qubits"])          # the subset to keep
print(result["dropped"])              # the qubits to leave out
print(result["total_bits_per_shot"])  # throughput with that subset
```

## Extraction

Once a source is certified, extract uniform bits from it.

```python
from qrng_attest import extract_from_source

bits, cert, seed = extract_from_source(source)
```

`bits` is a numpy array of 0s and 1s close to uniform. `seed` is the public seed the extractor used; it is
not secret and can be reused. To size the run before extracting, use the planner:

```python
from qrng_attest import extraction_plan

plan = extraction_plan(n_input=1000000, h_per_symbol=0.68)
print(plan)
print(plan.bits_for(256))   # input samples needed for a 256 bit key
```

## SP 800-22 battery

With the `sp80022` extra installed, the fifteen tests of NIST SP 800-22 rev 1a run on a 0/1 array through
the nistrng package.

```python
from qrng_attest import sp800_22

rows = sp800_22.run_battery(bits)
print(sp800_22.report(rows))
```

A test the stream is too short for is reported as not eligible. `names` runs a subset, for example
`run_battery(bits, names=["monobit", "runs"])`.

## Certificate output

The certificate prints a few lines. The verdict is PASS or FLAGGED. FLAGGED means a check failed, and the
report lists an action for each problem: recalibrate a qubit, drop a correlated qubit, or shorten the time
between calibrations. The extractable line gives how many
uniform bits it supports.

## Security parameter epsilon

Extraction takes a security parameter epsilon, the largest allowed distance between the output and a
uniform stream. The default is 2 to the power of minus 64. A smaller epsilon gives output closer to
uniform and fewer bits. Pass it to the extract and plan functions.

```python
plan = extraction_plan(n_input=1000000, h_per_symbol=0.68, epsilon=2.0 ** -32)
```

## IID permutation budget

The `assess` IID check has two modes. The default is a reduced shuffle budget, which can deny the IID
credit but never grant it. For the 10000-shuffle reference budget, pass `--iid-full` on the command line
or `full=True` in code.

The certificate path (`certify_source`, `certify_backend`) runs the full 10000-shuffle budget for its
per-qubit IID check. The reduced budget uses a fixed pass threshold calibrated for the full budget, so at
a smaller budget it fails clean data and denies the credit. The permutation test stops early on a clean
stream, so the full budget is cheap on good data. A stream that fails the faster chi-square or LRS check
skips the permutation test, so a bad source stays fast.

## Limitations

It assumes a trusted measurement, a characterized readout, and trusted hardware. It reimplements the NIST
SP 800-90B tests and is checked against the official test vectors. It is not device-independent and is not
NIST-certified.
