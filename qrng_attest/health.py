# SP 800-90B section 4.4 continuous health tests: repetition count and adaptive proportion.
import math
import numpy as np

ALPHA = 2.0 ** -20


def _binom_sf(c, W, p):
    # probability of at least c successes in W trials at rate p.
    if c <= 0:
        return 1.0
    if c > W:
        return 0.0
    total = 0.0
    logp = math.log(p) if p > 0 else -math.inf
    log1p = math.log(1 - p) if p < 1 else -math.inf
    for k in range(c, W + 1):
        lc = math.lgamma(W + 1) - math.lgamma(k + 1) - math.lgamma(W - k + 1)
        total += math.exp(lc + k * logp + (W - k) * log1p)
    return total


def rct_cutoff(H, alpha=ALPHA):
    # the run length that trips the repetition count test. Change alpha for the false-alarm rate.
    if H <= 0:
        return math.inf
    return 1 + math.ceil(-math.log2(alpha) / H)


def repetition_count_test(S, H, alpha=ALPHA):
    # returns (passed, longest_run, cutoff). Fails if any value repeats cutoff times in a row.
    S = np.asarray(S, int)
    C = rct_cutoff(H, alpha)
    longest = cur = 1
    for i in range(1, len(S)):
        cur = cur + 1 if S[i] == S[i - 1] else 1
        longest = max(longest, cur)
    return (longest < C, int(longest), C)


def apt_cutoff(H, W=1024, alpha=ALPHA):
    # the count that trips the adaptive proportion test. Change W for the window size.
    p = 2.0 ** -H
    for c in range(1, W + 1):
        if _binom_sf(c, W, p) <= alpha:
            return c
    return W + 1


def adaptive_proportion_test(S, H, W=1024, alpha=ALPHA):
    # returns (passed, max_count, cutoff). Fails if one value fills a window up to the cutoff.
    S = np.asarray(S, int)
    C = apt_cutoff(H, W, alpha)
    worst = 0; ok = True
    for start in range(0, len(S) - W + 1, W):
        win = S[start:start + W]
        cnt = int(np.sum(win == win[0]))
        worst = max(worst, cnt)
        if cnt >= C:
            ok = False
    return (ok, worst, C)


def health_report(S, H, alpha=ALPHA):
    # run both health tests at the claimed per-sample min-entropy H.
    rct = repetition_count_test(S, H, alpha)
    apt = adaptive_proportion_test(S, H, alpha=alpha)
    return {"rct_passed": rct[0], "rct_longest_run": rct[1], "rct_cutoff": rct[2],
            "apt_passed": apt[0], "apt_max_count": apt[1], "apt_cutoff": apt[2],
            "passed": rct[0] and apt[0]}
