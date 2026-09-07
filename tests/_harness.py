# Shared test helpers, usable under pytest or the zero-dependency runner.
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class Skip(unittest.SkipTest):
    # both pytest and the zero-dependency runner recognize unittest.SkipTest as a skip
    pass


def skip(reason=""):
    raise Skip(reason)


def needs(module):
    if importlib.util.find_spec(module) is None:
        raise Skip(f"requires {module}")


def uniform(n, k=2, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


def biased(n, p=0.7, seed=0):
    return (np.random.default_rng(seed).random(n) < p).astype(int)


def sticky(n, flip=0.1, seed=0):
    rng = np.random.default_rng(seed); s = np.empty(n, int); s[0] = rng.integers(2); r = rng.random(n)
    for i in range(1, n):
        s[i] = s[i - 1] if r[i] > flip else 1 - s[i - 1]
    return s


def periodic(n, period=8, noise=0.05, seed=0):
    rng = np.random.default_rng(seed); base = rng.integers(0, 2, period); s = np.tile(base, n // period + 1)[:n]
    return np.where(rng.random(n) < noise, 1 - s, s)
