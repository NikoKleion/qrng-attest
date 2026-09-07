# Randomness extraction: turn a min-entropy rate into uniform bits. Leftover hash lemma, Toeplitz matrix.
import math

import numpy as np


def extractable_bits(n_input, h_per_symbol, epsilon=2.0 ** -64, symbol_bits=1):
    if n_input <= 0 or h_per_symbol <= 0.0:
        return 0
    k = float(n_input) * float(h_per_symbol)
    ell = math.floor(k - 2.0 * math.log2(1.0 / epsilon) + 2.0)
    return max(0, int(ell))


def input_bits_needed(target_bits, h_per_symbol, epsilon=2.0 ** -64):
    if h_per_symbol <= 0.0:
        return None
    n = (float(target_bits) + 2.0 * math.log2(1.0 / epsilon) - 2.0) / float(h_per_symbol)
    return int(math.ceil(n))


def seed_length(n_input, ell):
    return int(n_input) + int(ell) - 1 if ell > 0 else 0


def toeplitz_extract(bits, seed, ell):
    bits = np.asarray(bits, np.uint8).ravel() & 1
    seed = np.asarray(seed, np.uint8).ravel() & 1
    n = len(bits)
    if ell <= 0:
        return np.zeros(0, np.uint8)
    need = seed_length(n, ell)
    if len(seed) != need:
        raise ValueError(f"seed must be n + ell - 1 = {need} bits, got {len(seed)}")
    rows = np.lib.stride_tricks.sliding_window_view(seed, n)[:ell][:, ::-1]
    return (rows @ bits) & 1


def random_seed(n_input, ell, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return rng.integers(0, 2, seed_length(n_input, ell)).astype(np.uint8)


class ExtractionPlan:
    def __init__(self, n_input, h_per_symbol, epsilon, ell, seed_bits, symbol_bits=1, note=None):
        self.n_input = int(n_input)
        self.h_per_symbol = float(h_per_symbol)
        self.epsilon = float(epsilon)
        self.output_bits = int(ell)
        self.seed_bits = int(seed_bits)
        self.symbol_bits = int(symbol_bits)
        self.note = note
        self.efficiency = (self.output_bits / self.n_input) if self.n_input else 0.0

    def bits_for(self, target_bits):
        return input_bits_needed(target_bits, self.h_per_symbol, self.epsilon)

    def __str__(self):
        L = [f"Extraction plan (leftover hash lemma, Toeplitz):",
             f"  input {self.n_input} samples at {self.h_per_symbol:.4f} bits/sample "
             f"= {self.n_input * self.h_per_symbol:.1f} bits of min-entropy",
             f"  security parameter epsilon = 2^{math.log2(self.epsilon):.0f}",
             f"  extractable output = {self.output_bits} bits "
             f"({self.efficiency:.4f} output bits per input sample)",
             f"  Toeplitz seed = {self.seed_bits} bits (a one-time public seed, reusable)"]
        for target in (128, 256):
            need = self.bits_for(target)
            L.append(f"  for a {target}-bit key: {need} input samples")
        if self.note:
            L.append(f"  note: {self.note}")
        return "\n".join(L)


def plan(n_input, h_per_symbol, epsilon=2.0 ** -64, symbol_bits=1, note=None):
    ell = extractable_bits(n_input, h_per_symbol, epsilon)
    return ExtractionPlan(n_input, h_per_symbol, epsilon, ell, seed_length(n_input, ell),
                          symbol_bits=symbol_bits, note=note)


def extract(bits, h_per_symbol, epsilon=2.0 ** -64, seed=None, rng=None):
    bits = np.asarray(bits, np.uint8).ravel() & 1
    n = len(bits)
    ell = extractable_bits(n, h_per_symbol, epsilon)
    if ell <= 0:
        return np.zeros(0, np.uint8), None
    seed = seed if seed is not None else random_seed(n, ell, rng=rng)
    return toeplitz_extract(bits, seed, ell), seed
