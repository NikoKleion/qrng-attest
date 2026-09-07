# SP 800-90B min-entropy estimators (section 6.3) plus the multi-bit combination. Lower value = less entropy.
import math
import numpy as np

Z = 2.5758293035489008


def _hmin(p):
    return float(-math.log2(min(max(p, 1e-12), 1.0)))


def _upper_bound_p(phat, n):
    # upper 99% confidence bound on a probability from n samples
    if n <= 1:
        return min(1.0, phat)
    return float(min(1.0, phat + Z * math.sqrt(phat * (1.0 - phat) / (n - 1))))


# 6.3.1 Most Common Value
def most_common_value(S, k=None):
    L = len(S)
    counts = np.bincount(S, minlength=(k or (int(S.max()) + 1)))
    phat = counts.max() / L
    return _hmin(_upper_bound_p(phat, L))


# 6.3.2 Collision (binary)
def collision(S, k=2):
    # collision estimate, binary only. Uses the mean number of samples until a value repeats.
    if k != 2:
        return None
    times = []
    i = 0; n = len(S)
    while i < n:
        seen = set(); j = i; repeat = False
        while j < n:
            v = int(S[j]); j += 1
            if v in seen:
                repeat = True; break
            seen.add(v)
        if repeat:
            times.append(j - i)
        i = j
    if len(times) < 2:
        return 1.0
    t = np.array(times, float)
    mean = t.mean(); sd = t.std(ddof=1); v = len(t)
    mean_lb = mean - Z * sd / math.sqrt(v)
    disc = 5.0 - 2.0 * mean_lb
    if disc <= 0:
        return 1.0
    p = 0.5 + 0.5 * math.sqrt(disc)
    p = min(max(p, 0.5), 1.0)
    return _hmin(p)


# 6.3.3 Markov
def markov(S, k=None, path_len=128):
    # Markov estimate: entropy of the most likely path. Change path_len to set the path length.
    k = k or (int(S.max()) + 1)
    L = len(S)
    init = np.bincount(S, minlength=k).astype(float); init /= init.sum()
    trans = np.full((k, k), 0.0)
    for a, b in zip(S[:-1], S[1:]):
        trans[a, b] += 1.0
    rows = trans.sum(axis=1, keepdims=True)
    trans = np.divide(trans, rows, out=np.full_like(trans, 1.0 / k), where=rows > 0)
    logp = np.log2(np.clip(init, 1e-300, None))
    lt = np.log2(np.clip(trans, 1e-300, None))
    for _ in range(path_len - 1):
        logp = (logp[:, None] + lt).max(axis=0)
    best = float(logp.max())
    return float(min(math.log2(k), -best / path_len))


# 6.3.5 / 6.3.6 t-Tuple and LRS
def _tuple_profile(S, k):
    # for each tuple length, the most common count and the collision count. Shared by t_tuple and lrs.
    n = len(S)
    tmax = min(n - 1, max(1, int(62.0 / math.log2(k)) if k > 1 else 62))
    prof = []
    keys = S.astype(np.int64)
    for t in range(1, tmax + 1):
        length = n - t + 1
        if length <= 0:
            break
        if t > 1:
            keys = keys[:length] * k + S[t - 1:t - 1 + length]
        _, counts = np.unique(keys, return_counts=True)
        prof.append((t, int(counts.max()), float(np.sum(counts * (counts - 1) / 2.0)), length))
        if counts.max() < 2:
            break
    return prof


def t_tuple(S, k=None, _prof=None):
    L = len(S); k = k or int(S.max()) + 1
    prof = _prof if _prof is not None else _tuple_profile(S, k)
    best_p = 0.0
    for t, maxc, _coll, length in prof:
        if maxc < 35:
            break
        best_p = max(best_p, (maxc / length) ** (1.0 / t))
    if best_p <= 0:
        return 1.0
    return _hmin(_upper_bound_p(best_p, L))


def lrs(S, k=None, _prof=None):
    L = len(S); k = k or int(S.max()) + 1
    prof = _prof if _prof is not None else _tuple_profile(S, k)
    u = next((t for t, maxc, _c, _l in prof if maxc < 35), 1)
    w = max((t for t, maxc, _c, _l in prof if maxc >= 2), default=0)
    if w < u:
        return 1.0
    best_p = 0.0
    for t, _maxc, coll, length in prof:
        if u <= t <= w:
            denom = length * (length - 1) / 2.0
            if denom > 0 and coll > 0:
                best_p = max(best_p, (coll / denom) ** (1.0 / t))
    if best_p <= 0:
        return 1.0
    return _hmin(_upper_bound_p(best_p, L))


# 6.3.4 Compression
_COMP_B = 6
_COMP_ALPH = 1 << _COMP_B
_COMP_D = 1000


def _compression_G(z, d, num_blocks):
    # helper for the compression estimate: expected value for a symbol of probability z.
    v = num_blocks - d
    omz = 1.0 - z
    if omz <= 0.0:
        return 0.0
    i = np.arange(2, num_blocks + 1, dtype=np.float64)
    Bi = np.exp((i - 1.0) * math.log(omz))
    ai = np.log2(i) * Bi
    Ad1 = float(np.sum(ai[:d - 1]))
    Ai_minus_Ad1 = float(np.sum(ai[d - 1:]))
    tail_i = i[d - 1:num_blocks - 2]
    tail_ai = ai[d - 1:num_blocks - 2]
    first_sum = float(np.sum((num_blocks - tail_i) * tail_ai)) + (num_blocks - d) * Ad1
    return (1.0 / v) * z * (z * first_sum + Ai_minus_Ad1)


def _compression_expect(p, num_blocks):
    q = (1.0 - p) / (_COMP_ALPH - 1.0)
    return _compression_G(p, _COMP_D, num_blocks) + (_COMP_ALPH - 1.0) * _compression_G(q, _COMP_D, num_blocks)


def compression(S, k=2):
    # compression estimate, binary only. Returns bits per bit, or None if the stream is too short.
    if k != 2:
        return None
    bits = np.asarray(S, np.int64)
    n = len(bits)
    num_blocks = n // _COMP_B
    if num_blocks - _COMP_D < 2:
        return None
    blocks = np.zeros(num_blocks, np.int64)
    for j in range(_COMP_B):
        blocks = (blocks << 1) | bits[j:num_blocks * _COMP_B:_COMP_B]
    order = np.lexsort((np.arange(num_blocks), blocks))
    sb = blocks[order]
    prev = np.full(num_blocks, -1, np.int64)
    same = sb[1:] == sb[:-1]
    prev[order[1:][same]] = order[:-1][same]
    v = num_blocks - _COMP_D
    i = np.arange(_COMP_D, num_blocks)
    last1 = np.where(prev[_COMP_D:] >= 0, prev[_COMP_D:] + 1, 0)
    logs = np.log2((i + 1) - last1)
    x_bar = logs.mean()
    sigma = 0.5907 * math.sqrt(math.fsum(logs * logs) / (v - 1.0) - x_bar * x_bar)
    x_bar_prime = x_bar - Z * sigma / math.sqrt(v)
    if _compression_expect(1.0 / _COMP_ALPH, num_blocks) <= x_bar_prime:
        return 1.0
    lo, hi = 1.0 / _COMP_ALPH, 1.0
    p = 0.5 * (lo + hi)
    for _ in range(1076):
        val = _compression_expect(p, num_blocks)
        if abs(val - x_bar_prime) <= 4.0 * np.spacing(max(abs(val), abs(x_bar_prime))):
            break
        if x_bar_prime < val:
            lo = p
        else:
            hi = p
        new_p = 0.5 * (lo + hi)
        if new_p == p:
            break
        p = new_p
    if p <= 1.0 / _COMP_ALPH:
        return 1.0
    return float(-math.log2(p) / _COMP_B)


# 6.3.7-6.3.10 predictor framework
def _longest_true_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        if cur > best:
            best = cur
    return best


def _no_run_prob(p, r, n):
    # probability of no run of r correct predictions in n trials at rate p. Computed in log space.
    if r <= 0 or n <= 0:
        return 0.0
    if r > n:
        return 1.0
    q = 1.0 - p
    if q <= 0.0:
        return 0.0
    if p <= 0.0:
        return 1.0
    lrp = r * math.log(p)
    lo, hi = 1.0 + 1e-15, 1.0 / p
    for _ in range(100):
        x = 0.5 * (lo + hi)
        lt = math.log(q) + lrp + (r + 1) * math.log(x)
        term = math.exp(lt) if lt < 700.0 else math.inf
        if x - 1.0 - term > 0.0:
            hi = x
        else:
            lo = x
    x = 0.5 * (lo + hi)
    denom = (r + 1 - r * x) * q
    numer = 1.0 - p * x
    if abs(denom) < 1e-300 or x <= 0.0:
        return 0.0
    logv = math.log(abs(numer)) - math.log(abs(denom)) - (n + 1) * math.log(x)
    if logv < -700.0:
        return 0.0
    val = math.exp(logv)
    if (numer < 0) != (denom < 0):
        val = -val
    return min(max(val, 0.0), 1.0)


def _p_local(r, n):
    # find the rate p where the chance of no run of length r in n trials is 0.99.
    if n <= 0:
        return 0.5
    if r <= 0:
        return 0.0
    if r >= n:
        return 1.0 - 1e-9
    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _no_run_prob(mid, r, n) > 0.99:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _predictor_entropy(correct, k=2):
    correct = np.asarray(correct, bool)
    N = len(correct); C = int(correct.sum())
    if N == 0:
        return math.log2(k)
    if C == N:
        return 0.0
    pg_b = (1.0 - 0.01 ** (1.0 / N)) if C == 0 else _upper_bound_p(C / N, N)
    if pg_b >= 1.0:
        return 0.0
    p = max(pg_b, _p_local(_longest_true_run(correct), N))
    p = min(max(p, 1.0 / k), 1.0)
    return _hmin(p)


def _ensemble_from_hits(hits):
    # combine sub-predictors by following the one with the most correct predictions so far.
    T, M = hits.shape
    scores = np.zeros(M, np.int64)
    correct = np.empty(T, bool)
    for t in range(T):
        correct[t] = hits[t, int(np.argmax(scores))]
        scores += hits[t]
    return correct


_MCW_WINDOWS = [63, 255, 1023, 4095]


def multimcw(S, k=2):
    n = len(S)
    if n < 2:
        return math.log2(k) if k > 1 else 0.0
    if k == 2:
        idx = np.arange(n)
        pref = np.concatenate([[0], np.cumsum(S)])
        hits = np.zeros((n, len(_MCW_WINDOWS)), bool)
        for wi, w in enumerate(_MCW_WINDOWS):
            lo = np.maximum(0, idx - w)
            ones = pref[idx] - pref[lo]
            size = np.minimum(idx, w)
            pred = (2 * ones > size).astype(int)
            hits[1:, wi] = (pred == S)[1:]
        return _predictor_entropy(_ensemble_from_hits(hits[1:]), k=2)
    Sl = S.tolist()
    hits = np.zeros((n, len(_MCW_WINDOWS)), bool)
    for wi, w in enumerate(_MCW_WINDOWS):
        count = np.zeros(k, np.int64)
        for i in range(1, n):
            count[Sl[i - 1]] += 1
            if i - 1 - w >= 0:
                count[Sl[i - 1 - w]] -= 1
            hits[i, wi] = (int(count.argmax()) == Sl[i])
    return _predictor_entropy(_ensemble_from_hits(hits[1:]), k=k)


def lag(S, k=None, D=128):
    k = k or int(S.max()) + 1
    n = len(S)
    if n <= 1:
        return math.log2(k)
    D = min(D, n - 1)
    hits = np.zeros((n, D), bool)
    for d in range(1, D + 1):
        hits[d:, d - 1] = (S[d:] == S[:-d])
    return _predictor_entropy(_ensemble_from_hits(hits[1:]), k=k)


def multimmc(S, k=2, D=16):
    n = len(S)
    if k == 2:
        models = [dict() for _ in range(D)]
        ctx = [0] * D; mask = [(1 << d) - 1 for d in range(1, D + 1)]
        scores = np.zeros(D, np.int64)
        correct = np.zeros(n, bool)
        for i in range(n):
            if i > 0:
                preds = np.full(D, -1)
                for d in range(1, D + 1):
                    if i >= d:
                        c = models[d - 1].get(ctx[d - 1])
                        if c is not None:
                            preds[d - 1] = 0 if c[0] >= c[1] else 1
                correct[i] = (preds[int(np.argmax(scores))] == S[i])
                scores += (preds == S[i]).astype(np.int64)
                for d in range(1, D + 1):
                    if i >= d:
                        c = models[d - 1].get(ctx[d - 1])
                        if c is None:
                            c = [0, 0]; models[d - 1][ctx[d - 1]] = c
                        c[int(S[i])] += 1
            b = int(S[i])
            for d in range(D):
                ctx[d] = ((ctx[d] << 1) | b) & mask[d]
        return _predictor_entropy(correct[1:])
    Sl = S.tolist()
    models = [dict() for _ in range(D)]
    scores = np.zeros(D, np.int64)
    correct = np.zeros(n, bool)
    for i in range(n):
        if i > 0:
            preds = [-1] * D
            for d in range(1, D + 1):
                if i >= d:
                    c = models[d - 1].get(tuple(Sl[i - d:i]))
                    if c:
                        preds[d - 1] = max(sorted(c), key=c.get)
            best = int(np.argmax(scores))
            correct[i] = preds[best] == Sl[i] if preds[best] >= 0 else False
            for d in range(D):
                if preds[d] == Sl[i]:
                    scores[d] += 1
            for d in range(1, D + 1):
                if i >= d:
                    key = tuple(Sl[i - d:i])
                    c = models[d - 1].get(key)
                    if c is None:
                        c = {}; models[d - 1][key] = c
                    c[Sl[i]] = c.get(Sl[i], 0) + 1
    return _predictor_entropy(correct[1:], k=k)


def lz78y(S, k=2, B=16, maxdict=65536):
    n = len(S)
    if k == 2:
        D = [dict() for _ in range(B)]
        ctx = [0] * B; mask = [(1 << (L + 1)) - 1 for L in range(B)]
        correct = np.zeros(n, bool)
        for i in range(n):
            if i > B:
                pred = -1
                for L in range(1, B + 1):
                    c = D[L - 1].get(ctx[L - 1])
                    if c is not None and (c[0] + c[1]) > 0:
                        pred = 0 if c[0] >= c[1] else 1
                correct[i] = (pred == S[i]) if pred >= 0 else False
                for L in range(1, B + 1):
                    c = D[L - 1].get(ctx[L - 1])
                    if c is None:
                        if len(D[L - 1]) >= maxdict:
                            continue
                        c = [0, 0]; D[L - 1][ctx[L - 1]] = c
                    c[int(S[i])] += 1
            b = int(S[i])
            for L in range(B):
                ctx[L] = ((ctx[L] << 1) | b) & mask[L]
        return _predictor_entropy(correct[B + 1:])
    Sl = S.tolist()
    Dd = [dict() for _ in range(B)]
    correct = np.zeros(n, bool)
    for i in range(n):
        if i > B:
            pred = -1
            for L in range(1, B + 1):
                c = Dd[L - 1].get(tuple(Sl[i - L:i]))
                if c:
                    pred = max(sorted(c), key=c.get)
            correct[i] = (pred == Sl[i]) if pred >= 0 else False
            for L in range(1, B + 1):
                key = tuple(Sl[i - L:i])
                c = Dd[L - 1].get(key)
                if c is None:
                    if len(Dd[L - 1]) >= maxdict:
                        continue
                    c = {}; Dd[L - 1][key] = c
                c[Sl[i]] = c.get(Sl[i], 0) + 1
    return _predictor_entropy(correct[B + 1:], k=k)


# estimator suites and the multi-bit assessment
def all_estimators(S, k=None, include_predictors=True):
    # run every estimator that applies to alphabet size k. Collision, Markov, and Compression are binary
    S = np.asarray(S).astype(int)
    if S.size == 0:
        return {}
    if S.min() < 0:
        raise ValueError("samples must be non-negative integers")
    k = max(int(k or 2), int(S.max()) + 1)
    prof = _tuple_profile(S, k)
    est = {"most_common_value": most_common_value(S, k),
           "t_tuple": t_tuple(S, k, _prof=prof),
           "lrs": lrs(S, k, _prof=prof)}
    if k == 2:
        est["collision"] = collision(S, k)
        est["markov"] = markov(S, k)
        est["compression"] = compression(S, k)
    if include_predictors:
        est.update(multimcw=multimcw(S, k), lag=lag(S, k), multimmc=multimmc(S, k), lz78y=lz78y(S, k))
    return {name: v for name, v in est.items() if v is not None}


def _to_bits(sym, word):
    # split each symbol into `word` bits, most significant first, and concatenate.
    sym = np.asarray(sym, np.int64)
    shifts = word - 1 - np.arange(word)
    return ((sym[:, None] >> shifts[None, :]) & 1).reshape(-1).astype(int)


def min_entropy(S, k=None, include_predictors=True):
    # main entry point. Binary sources are assessed directly. Multi-bit sources are assessed as symbols
    S = np.asarray(S).astype(int)
    if S.size and S.min() < 0:
        raise ValueError("samples must be non-negative integers")
    vals = np.unique(S)
    alph = int(len(vals))
    if alph <= 2:
        sym = np.searchsorted(vals, S) if alph else S
        est = all_estimators(sym, 2, include_predictors)
        return (float(min(est.values())) if est else float("nan")), est
    sym = np.searchsorted(vals, S)
    word = max(1, (alph - 1).bit_length())
    lit = all_estimators(sym, alph, include_predictors)
    bits = _to_bits(sym, word)
    bit = all_estimators(bits, 2, include_predictors)
    h_original = float(min(lit.values()))
    h_bitstring = float(min(bit.values()))
    assessed = min(float(word), word * h_bitstring, h_original)
    per = {f"literal:{n}": v for n, v in lit.items()}
    per.update({f"bitstring:{n}": v for n, v in bit.items()})
    per.update(H_original=h_original, H_bitstring=h_bitstring, word_size=float(word))
    return float(assessed), per
