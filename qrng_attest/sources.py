# Source abstraction: one contract, many adapters (array, file, live callback, synthetic) yielding integer samples.
import os
import numpy as np

NIST_MIN_SAMPLES = 1_000_000


class Source:
    name = "source"
    alphabet_size = 2
    provenance = "unknown"

    def samples(self):
        raise NotImplementedError

    def describe(self):
        s = np.asarray(self.samples())
        L = int(s.size)
        used = int(s.max()) + 1 if L else 0
        flags = []
        if L < NIST_MIN_SAMPLES:
            flags.append(f"below NIST recommended length ({L} < {NIST_MIN_SAMPLES}); estimates are conservative")
        if used > self.alphabet_size:
            flags.append(f"sample value {used - 1} exceeds declared alphabet {self.alphabet_size}")
        if self.provenance == "synthetic":
            flags.append("synthetic source: this run is simulated, not measured hardware")
        return {"name": self.name, "length": L, "alphabet_size": self.alphabet_size,
                "provenance": self.provenance, "flags": flags}


class ArraySource(Source):
    # wrap an in-memory array of integer samples.
    def __init__(self, data, alphabet_size=None, name="array", provenance="measured"):
        self._d = np.ascontiguousarray(np.asarray(data)).astype(int).ravel()
        if self._d.size and (self._d.min() < 0):
            raise ValueError("samples must be non-negative integers")
        self.alphabet_size = int(alphabet_size or (self._d.max() + 1 if self._d.size else 2))
        self.name = name; self.provenance = provenance

    def samples(self):
        return self._d


class BytesFileSource(Source):
    # read a raw binary capture; mode='bits' -> alphabet 2, mode='bytes' -> alphabet 256.
    def __init__(self, path, mode="bits", name=None, provenance="external", max_bytes=None):
        self.path = path; self.mode = mode
        self.name = name or os.path.basename(path); self.provenance = provenance
        self.max_bytes = max_bytes
        self.alphabet_size = 2 if mode == "bits" else 256

    def samples(self):
        with open(self.path, "rb") as f:
            raw = f.read(self.max_bytes) if self.max_bytes else f.read()
        b = np.frombuffer(raw, dtype=np.uint8)
        return np.unpackbits(b).astype(int) if self.mode == "bits" else b.astype(int)


class CallableSource(Source):
    # wrap a live hardware read: fn(n) -> array of n samples, materialized on demand.
    def __init__(self, fn, n, alphabet_size=2, name="live-device", provenance="measured"):
        self.fn = fn; self.n = int(n); self.alphabet_size = int(alphabet_size)
        self.name = name; self.provenance = provenance
        self._cache = None

    def samples(self):
        if self._cache is None:
            self._cache = np.ascontiguousarray(np.asarray(self.fn(self.n))).astype(int).ravel()
        return self._cache


class SyntheticSource(Source):
    # synthetic stream for testing: gen(rng, n) -> array; provenance 'synthetic'.
    def __init__(self, gen, n, alphabet_size=2, seed=0, name="synthetic"):
        self.gen = gen; self.n = int(n); self.alphabet_size = int(alphabet_size)
        self.seed = seed; self.name = name; self.provenance = "synthetic"
        self._cache = None

    def samples(self):
        if self._cache is None:
            self._cache = np.ascontiguousarray(self.gen(np.random.default_rng(self.seed), self.n)).astype(int).ravel()
        return self._cache


def from_any(obj, **kw):
    # build a Source from a path, array, callable, or pass a Source through unchanged.
    if isinstance(obj, Source):
        return obj
    if isinstance(obj, str) and os.path.exists(obj):
        return BytesFileSource(obj, **kw)
    if callable(obj):
        return CallableSource(obj, **kw)
    return ArraySource(obj, **kw)
