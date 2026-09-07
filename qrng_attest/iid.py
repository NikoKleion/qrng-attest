# qrng_attest.iid: SP 800-90B section 5 IID track.
# Decides whether a source can be treated as independent and identically distributed. Three checks:
import bz2
import math
import numpy as np

from .estimators import most_common_value, _to_bits

PERMS = 2000
MAX_SAMPLES = 8000
LAGS = (1, 2, 8, 16, 32)
STAT_NAMES = (["excursion", "num_directional_runs", "len_directional_runs", "num_increases_decreases",
               "num_runs_median", "len_runs_median", "avg_collision", "max_collision"]
              + [f"periodicity_{p}" for p in LAGS] + [f"covariance_{p}" for p in LAGS] + ["compression"])


# binary conversions (section 5.1, applied only when the alphabet is binary)
def _conversion1(d):
    # number of 1 bits in each 8-bit block.
    n = len(d)
    return np.bincount(np.arange(n) // 8, weights=d, minlength=(n + 7) // 8).astype(np.int64)


def _conversion2(d):
    # unsigned value of each non-overlapping 8-bit block.
    n = len(d)
    w = d * (1 << (7 - (np.arange(n) % 8)))
    return np.bincount(np.arange(n) // 8, weights=w, minlength=(n + 7) // 8).astype(np.int64)


# individual statistics
def _excursion(raw, mean):
    cs = np.cumsum(raw.astype(np.float64))
    return float(np.max(np.abs(cs - np.arange(1, len(raw) + 1) * mean))) if len(raw) else 0.0


def _num_runs(s):
    return (1 + int(np.count_nonzero(s[1:] != s[:-1]))) if len(s) else 0


def _len_runs(s):
    if len(s) == 0:
        return 0
    bounds = np.concatenate(([0], np.flatnonzero(s[1:] != s[:-1]) + 1, [len(s)]))
    return int(np.max(np.diff(bounds)))


def _find_collisions(x, k):
    # for each starting point, how many samples until a value repeats.
    n = len(x); ret = []; i = 0; j = 0
    xl = x.tolist()
    while i + j < n:
        seen = set()
        while i + j < n:
            v = xl[i + j]
            if v in seen:
                ret.append(j + 1); i += j; j = 0; break
            seen.add(v); j += 1
        else:
            break
        i += 1
    return ret


def _periodicity(x, p):
    m = len(x)
    return int(np.count_nonzero(x[:m - p] == x[p:])) if m > p else 0


def _covariance(x, p):
    m = len(x)
    return float(np.dot(x[:m - p].astype(np.float64), x[p:].astype(np.float64))) if m > p else 0.0


def _compression_len(raw):
    # bzip2 length of the sample string. One of the 19 permutation statistics.
    s = " ".join(map(str, raw.tolist())).encode("ascii")
    return len(bz2.compress(s, 5))


def _statistics(symbols, raw, alph, median, mean):
    # the 19 permutation statistics as an array, in STAT_NAMES order.
    if alph == 2:
        conv1 = _conversion1(symbols)
        conv2 = _conversion2(symbols)
        dir_src, per_src, cov_src, col_src, col_k = conv1, conv1, conv1, conv2, 256
    else:
        dir_src = per_src = col_src = symbols
        cov_src = raw
        col_k = alph
    s1 = np.where(dir_src[:-1] <= dir_src[1:], 1, -1) if len(dir_src) > 1 else np.array([], int)
    s2 = np.where(symbols >= median, 1, -1)
    col = _find_collisions(col_src, col_k)
    out = [_excursion(raw, mean),
           _num_runs(s1), _len_runs(s1), max(int(np.count_nonzero(s1 == 1)), len(s1) - int(np.count_nonzero(s1 == 1))),
           _num_runs(s2), _len_runs(s2),
           (sum(col) / len(col)) if col else 0.0, (max(col) if col else 0)]
    out += [_periodicity(per_src, p) for p in LAGS]
    out += [_covariance(cov_src, p) for p in LAGS]
    out.append(_compression_len(raw))
    return np.array(out, dtype=np.float64)


# permutation test (5.1)
class PermutationResult:
    def __init__(self, is_iid, counters, statistics, passed):
        self.is_iid = is_iid
        self.counters = counters
        self.statistics = statistics
        self.passed = passed
        self.failed = [STAT_NAMES[j] for j in range(len(STAT_NAMES)) if not passed[j]]


def permutation_test(S, perms=PERMS, seed=0, max_samples=MAX_SAMPLES):
    S = np.asarray(S).astype(int)
    if len(S) > max_samples:
        S = S[:max_samples]
    raw = S
    vals, symbols = np.unique(S, return_inverse=True)
    alph = int(len(vals))
    mean = float(raw.mean()) if len(raw) else 0.0
    median = 0.5 if alph == 2 else (float(np.median(symbols)) if len(symbols) else 0.0)
    t = _statistics(symbols, raw, alph, median, mean)
    m = len(t)
    C = np.zeros((m, 3), np.int64)
    status = np.ones(m, bool)
    rng = np.random.default_rng(seed)
    for _ in range(perms):
        if not status.any():
            break
        perm = rng.permutation(len(S))
        tp = _statistics(symbols[perm], raw[perm], alph, median, mean)
        gt = tp > t; eq = tp == t
        for j in range(m):
            if not status[j]:
                continue
            C[j, 0 if gt[j] else (1 if eq[j] else 2)] += 1
            if (C[j, 0] + C[j, 1] > 5) and (C[j, 1] + C[j, 2] > 5):
                status[j] = False
    passed = (C[:, 0] + C[:, 1] > 5) & (C[:, 1] + C[:, 2] > 5)
    return PermutationResult(bool(passed.all()), C, t, passed)


# chi-square tests (5.2)
def _gammaincc(a, x):
    # upper incomplete gamma function, used for chi-square p-values.
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a; s = 1.0 / a; d = s
        for _ in range(1000):
            ap += 1.0; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a; c = 1e300; d = 1.0 / b; h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _chi_square_pvalue(score, df):
    if df <= 0:
        return 1.0
    return _gammaincc(df / 2.0, score / 2.0)


def _bin_expectations(p, block_scale):
    # merge symbols into bins so each bin expects at least 5 counts.
    order = np.argsort(p)
    bins = []; cur = []; e = 0.0
    for idx in order:
        cur.append(int(idx)); e += p[idx] * block_scale
        if e >= 5.0:
            bins.append((cur, e)); cur = []; e = 0.0
    if cur:
        if bins:
            prev_syms, prev_e = bins[-1]
            bins[-1] = (prev_syms + cur, prev_e + e)
        else:
            bins.append((cur, e))
    return bins


def _binary_chi_square_independence(bits, n):
    # chi-square independence for binary data, using m-bit tuples.
    p1 = float(bits.mean()); p0 = 1.0 - p1
    min_p = min(p0, p1)
    if min_p <= 0.0:
        return 0.0, 0
    m = 11
    while m > 1 and (min_p ** m) * (n // m) < 5:
        m -= 1
    if m < 2:
        return 0.0, 0
    block_count = n // m
    blocks = bits[:block_count * m].reshape(block_count, m)
    idx = blocks @ (1 << np.arange(m - 1, -1, -1))
    occ = np.bincount(idx, minlength=1 << m).astype(float)
    t = np.arange(1 << m)
    w = np.zeros(1 << m, np.int64)
    for shift in range(m):
        w += (t >> shift) & 1
    e = (p1 ** w) * (p0 ** (m - w)) * block_count
    T = float(np.sum((occ - e) ** 2 / e))
    return T, (1 << m) - 2


def _binary_goodness_of_fit(bits, n):
    # chi-square goodness-of-fit for binary data across 10 sub-sequences.
    sub = n // 10
    if sub < 1:
        return 0.0, 0
    p = float(bits.mean())
    e1 = p * sub; e0 = (1.0 - p) * sub
    if min(e0, e1) <= 0.0:
        return 0.0, 0
    T = 0.0
    for i in range(10):
        o1 = float(bits[i * sub:(i + 1) * sub].sum()); o0 = sub - o1
        T += (o0 - e0) ** 2 / e0 + (o1 - e1) ** 2 / e1
    return T, 9


def chi_square_tests(S):
    # run both chi-square tests. Passes only if both p-values are at least 0.001.
    S = np.asarray(S).astype(int)
    n = len(S)
    vals, sym = np.unique(S, return_inverse=True)
    alph = int(len(vals))
    if n < 20 or alph < 2:
        return True, 1.0, 1.0
    if alph == 2:
        T_ind, df_ind = _binary_chi_square_independence(sym, n)
        T_gof, df_gof = _binary_goodness_of_fit(sym, n)
        p_ind = _chi_square_pvalue(T_ind, df_ind)
        p_gof = _chi_square_pvalue(T_gof, df_gof)
        return bool(p_ind >= 0.001 and p_gof >= 0.001), p_ind, p_gof
    p = np.bincount(sym, minlength=alph).astype(float) / n

    pairs = sym[:2 * (n // 2)].reshape(-1, 2)
    pair_idx = pairs[:, 0] * alph + pairs[:, 1]
    pe = np.outer(p, p).ravel() * math.floor(n * 0.5)
    bins = _bin_expectations(pe, 1.0)
    binmap = np.full(alph * alph, -1, int)
    for b, (syms, _e) in enumerate(bins):
        for s in syms:
            binmap[s] = b
    obs = np.bincount(binmap[pair_idx], minlength=len(bins)).astype(float)
    exp = np.array([e for _s, e in bins])
    T_ind = float(np.sum((obs - exp) ** 2 / exp))
    df_ind = max(1, len(bins) - alph)
    p_ind = _chi_square_pvalue(T_ind, df_ind)

    sub = n // 10
    bins2 = _bin_expectations(p * math.floor(n / 10.0), 1.0)
    binmap2 = np.full(alph, -1, int)
    for b, (syms, _e) in enumerate(bins2):
        for s in syms:
            binmap2[s] = b
    exp2 = np.array([e for _s, e in bins2])
    T_gof = 0.0
    for blk in range(10):
        o = np.bincount(binmap2[sym[blk * sub:(blk + 1) * sub]], minlength=len(bins2)).astype(float)
        T_gof += float(np.sum((o - exp2) ** 2 / exp2))
    df_gof = max(1, 9 * (len(bins2) - 1))
    p_gof = _chi_square_pvalue(T_gof, df_gof)
    return bool(p_ind >= 0.001 and p_gof >= 0.001), p_ind, p_gof


# longest-repeated-substring test (5.2.3)
def _lrs_length(sym):
    # length of the longest repeated substring, via a suffix array.
    n = len(sym)
    if n < 2:
        return 0
    rank = np.asarray(sym, np.int64)
    sa = np.argsort(rank, kind="stable")
    k = 1
    while k < n:
        key2 = np.full(n, -1, np.int64); key2[:n - k] = rank[k:]
        order = np.lexsort((key2, rank))
        r1 = rank[order]; r2 = key2[order]
        diff = np.empty(n, bool); diff[0] = True
        diff[1:] = (r1[1:] != r1[:-1]) | (r2[1:] != r2[:-1])
        new_rank = np.empty(n, np.int64)
        new_rank[order] = np.cumsum(diff) - 1
        rank = new_rank; sa = order
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    pos = rank.tolist(); sal = sa.tolist(); Sl = list(map(int, sym))
    best = 0; h = 0
    for i in range(n):
        r = pos[i]
        if r > 0:
            j = sal[r - 1]
            while i + h < n and j + h < n and Sl[i + h] == Sl[j + h]:
                h += 1
            if h > best:
                best = h
            if h:
                h -= 1
        else:
            h = 0
    return best


def lrs_test(S, max_samples=200000):
    # is the longest repeated substring plausible for an independent source?
    S = np.asarray(S).astype(int)
    if len(S) > max_samples:
        S = S[:max_samples]
    L = len(S)
    if L < 4:
        return True, 1.0
    vals, sym = np.unique(S, return_inverse=True)
    p = np.bincount(sym).astype(float) / L
    p_col = float(np.sum(p * p))
    if p_col > 1.0 - 1e-15:
        return True, 1.0
    W = _lrs_length(sym)
    if W < 1:
        return True, 1.0
    N = (L - W + 1) * (L - W) / 2.0
    lx = W * math.log(p_col)
    if lx > -700.0:
        log_no_col = math.log1p(-math.exp(lx))
        prob = -math.expm1(N * log_no_col)
        return bool(math.log(0.999) >= N * log_no_col), float(prob)
    lNx = math.log(N) + lx
    prob = math.exp(lNx) if lNx > -700.0 else 0.0
    return bool(prob >= 1.0 - 0.999), float(prob)


# combined decision
class IIDResult:
    def __init__(self, perm, chi, lrs, mcv_bits, assessed):
        self.permutation = perm
        self.permutation_iid = perm.is_iid
        self.chi_square_iid, self.p_independence, self.p_goodness = chi
        self.lrs_iid, self.lrs_prob = lrs
        self.is_iid = perm.is_iid and self.chi_square_iid and self.lrs_iid
        self.mcv_min_entropy = mcv_bits
        self.assessed = assessed

    def __str__(self):
        v = "IID" if self.is_iid else "non-IID"
        parts = [f"IID track: {v}",
                 f"  permutation {'pass' if self.permutation_iid else 'fail'}"
                 + ("" if self.permutation_iid else f" (extreme: {', '.join(self.permutation.failed)})"),
                 f"  chi-square  {'pass' if self.chi_square_iid else 'fail'}"
                 f" (independence p={self.p_independence:.3g}, goodness p={self.p_goodness:.3g})",
                 f"  LRS         {'pass' if self.lrs_iid else 'fail'} (prob={self.lrs_prob:.3g})"]
        if self.is_iid:
            parts.append(f"  IID assessed min-entropy = {self.assessed:.4f} bits/sample (most-common-value)")
        return "\n".join(parts)


def _mcv_assessed(S):
    # the most-common-value assessment, used when the source passes the IID checks.
    S = np.asarray(S).astype(int)
    vals = np.unique(S)
    alph = int(len(vals))
    if alph <= 2:
        h = most_common_value(S, 2)
        return h, h
    sym = np.searchsorted(vals, S)
    word = max(1, (alph - 1).bit_length())
    h_lit = most_common_value(sym, alph)
    h_bit = most_common_value(_to_bits(sym, word), 2)
    return min(float(word), word * h_bit, h_lit), h_bit


def iid_decision(S, perms=PERMS, seed=0, max_samples=MAX_SAMPLES, full=False):
    # full=True runs the full 10000-shuffle budget and can take minutes. Change perms and max_samples to trade
    if full:
        perms = 10000; max_samples = 1000000
    perm = permutation_test(S, perms=perms, seed=seed, max_samples=max_samples)
    chi = chi_square_tests(S)
    lrs = lrs_test(S)
    assessed, mcv_bits = _mcv_assessed(S)
    return IIDResult(perm, chi, lrs, mcv_bits, assessed)
