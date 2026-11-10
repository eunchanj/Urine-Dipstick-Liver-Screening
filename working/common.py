"""Statistics for the evaluation scripts: discrimination, calibration, operating points and net benefit.

Bootstrap: persons are resampled with replacement to the full sample size (numpy RandomState(seed).randint),
resamples that hold one class only are skipped, and the interval runs from the 2.5th to the 97.5th percentile.
The same n and seed give the same resamples in every function. The survey-weighted functions use the weights
only; their intervals ignore strata and clusters. Nothing here reads or writes files or prints, and assertion
messages carry counts only.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as _expit
from sklearn.metrics import roc_auc_score

__all__ = ["logit", "expit", "n_boot", "boot_ci", "auroc_ci", "auprc_ci", "auprc", "paired_boot", "delong_paired",
           "delong_var", "fit_logistic_1d", "fit_offset_intercept", "calib_stats", "calib_stats_ci", "scaled_brier",
           "decile_curve", "wilson_ci", "confusion_at", "operating_point_ci", "rule_point_ci",
           "threshold_for_sensitivity", "youden_threshold", "alert_match_threshold", "net_benefit",
           "net_benefit_ci", "net_benefit_rule", "fixed_alert_nb_ci", "roc_on_grid", "weighted_prevalence",
           "weighted_auroc", "weighted_oe"]

SEED = 42
N_BOOT = 1000
EPS = 1e-9                  # logit clip: p = 0 becomes about -20.7
TIE = 1e-12                 # alert rates (or Youden indices) closer than this count as equal
Z95 = float(stats.norm.ppf(0.975))
CAL_KEYS = ("citl", "slope", "oe", "brier", "scaled_brier")
OP_KEYS = ("alert_rate", "se", "sp", "ppv", "npv", "alerts_per_case")
COUNT_KEYS = ("tp", "fp", "tn", "fn")


# ------------------------------------------------------------------------------------------------ inputs
def _labels(y):
    """Labels as a float array of 0 and 1."""
    y = np.asarray(y, float).ravel()
    bad = int((~np.isin(y, (0.0, 1.0))).sum())
    assert bad == 0, f"labels must be 0 or 1 ({bad:,} other values)"
    return y


def _values(p, n, unit=False):
    """Scores as a float array of length n: finite, and within [0, 1] when unit is True."""
    p = np.asarray(p, float).ravel()
    assert len(p) == n, f"{len(p):,} scores for {n:,} labels"
    bad = int((~np.isfinite(p)).sum())
    if unit:
        bad += int(((p < 0) | (p > 1)).sum())
    assert bad == 0, f"{bad:,} scores missing or out of range"
    return p


def _flag(f, n):
    """Rule flags as a boolean array of length n."""
    f = np.asarray(f)
    assert len(f) == n, f"{len(f):,} flags for {n:,} labels"
    if f.dtype != bool:
        f = np.asarray(f, float)
        bad = int((~np.isin(f, (0.0, 1.0))).sum())
        assert bad == 0, f"rule flags must be 0 or 1 ({bad:,} other values)"
    return f.astype(bool)


def _weights(w, n):
    """Survey weights as a float array of length n: finite, non-negative, positive in total."""
    w = np.asarray(w, float).ravel()
    assert len(w) == n, f"{len(w):,} weights for {n:,} labels"
    bad = int((~np.isfinite(w) | (w < 0)).sum())
    assert bad == 0 and w.sum() > 0, f"{bad:,} weights missing or negative"
    return w


def _both(y):
    return 0 < y.sum() < len(y)


def _ratio(a, b):
    return float(a / b) if b > 0 else np.nan


# --------------------------------------------------------------------------------------------- bootstrap
def logit(p):
    """Log-odds of p, with p clipped to [EPS, 1 - EPS]."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    """Inverse of logit."""
    return _expit(np.asarray(z, float))


def n_boot(synthetic):
    """Number of resamples: N_BOOT, or int(LIV_SYNTH_BOOT) when synthetic is True and the variable is set."""
    v = os.environ.get("LIV_SYNTH_BOOT", "").strip() if synthetic is True else ""
    if not v:
        return N_BOOT
    k = int(v)
    assert k >= 1, "LIV_SYNTH_BOOT must be a positive whole number"
    return k


def _draws(size, reps, seed):
    rng = np.random.RandomState(seed)
    for _ in range(reps):
        yield rng.randint(0, size, size)


def _resamples(y, reps, seed):
    """Multiplicity of each person in every resample that holds both classes."""
    for i in _draws(len(y), reps, seed):
        m = np.bincount(i, minlength=len(y)).astype(float)
        if 0 < m @ y < len(y):
            yield m


def _pct(vals, k=1):
    """2.5th and 97.5th percentiles of each of k statistics over the resamples where it is finite."""
    a = np.asarray(vals, float).reshape(-1, k)
    lo, hi = np.full(k, np.nan), np.full(k, np.nan)
    for j in range(k):
        v = a[np.isfinite(a[:, j]), j]
        if len(v):
            lo[j], hi[j] = np.percentile(v, [2.5, 97.5])
    return (float(lo[0]), float(hi[0])) if k == 1 else (lo, hi)


def _with_ci(keys, est, boots):
    lo, hi = _pct(boots, len(keys))
    out = {}
    for j, key in enumerate(keys):
        out.update({key: est[j], f"{key}_lo": float(lo[j]), f"{key}_hi": float(hi[j])})
    out["n_boot_used"] = len(boots)
    return out


def boot_ci(y, p, fn, n=N_BOOT, seed=SEED):
    """Percentile 95% CI of fn(y, p) over person resamples."""
    y = _labels(y)
    p = np.asarray(p)
    out = [fn(y[i], p[i]) for i in _draws(len(y), n, seed) if _both(y[i])]
    return _pct(out)


# ------------------------------------------------------------------------------------------ discrimination
def _levels(p):
    """Distinct values of p (ascending) and the level of each person."""
    u, lev = np.unique(p, return_inverse=True)
    return u, lev.ravel()


def _auc(lev, k, y, w):
    """AUROC with person weights w; tied scores count one half."""
    pos = np.bincount(lev, weights=w * y, minlength=k)
    neg = np.bincount(lev, weights=w * (1 - y), minlength=k)
    den = pos.sum() * neg.sum()
    return float(pos @ (np.cumsum(neg) - 0.5 * neg) / den) if den > 0 else np.nan


def _ap(lev, k, y, w):
    """Average precision with person weights w (one step per distinct score, as in sklearn)."""
    pos = np.bincount(lev, weights=w * y, minlength=k)[::-1]
    flagged = np.cumsum(np.bincount(lev, weights=w, minlength=k)[::-1])
    tp = np.cumsum(pos)
    hit = pos > 0
    return float(pos[hit] @ (tp[hit] / flagged[hit]) / tp[-1]) if tp[-1] > 0 else np.nan


def auroc_ci(y, p, n=N_BOOT, seed=SEED):
    """AUROC and its bootstrap 95% CI: (auroc, lo, hi)."""
    y = _labels(y)
    u, lev = _levels(_values(p, len(y)))
    est = _auc(lev, len(u), y, np.ones(len(y)))
    return (est, *_pct([_auc(lev, len(u), y, m) for m in _resamples(y, n, seed)]))


def auprc_ci(y, p, n=N_BOOT, seed=SEED):
    """Average precision and its bootstrap 95% CI: (auprc, lo, hi)."""
    y = _labels(y)
    u, lev = _levels(_values(p, len(y)))
    est = _ap(lev, len(u), y, np.ones(len(y)))
    return (est, *_pct([_ap(lev, len(u), y, m) for m in _resamples(y, n, seed)]))


def auprc(y, p):
    """Average precision, the estimate of auprc_ci without resampling."""
    y = _labels(y)
    u, lev = _levels(_values(p, len(y)))
    return _ap(lev, len(u), y, np.ones(len(y)))


def paired_boot(y, p_a, p_b, fn=None, n=N_BOOT, seed=SEED):
    """fn(y, p_a) - fn(y, p_b) with a 95% CI from the same resamples for both (fn None or roc_auc_score: AUROC)."""
    y = _labels(y)
    if fn is None or fn is roc_auc_score:
        ua, la = _levels(_values(p_a, len(y)))
        ub, lb = _levels(_values(p_b, len(y)))

        def diff(w):
            return _auc(la, len(ua), y, w) - _auc(lb, len(ub), y, w)
        return (diff(np.ones(len(y))), *_pct([diff(m) for m in _resamples(y, n, seed)]))
    a, b = np.asarray(p_a), np.asarray(p_b)
    out = [fn(y[i], a[i]) - fn(y[i], b[i]) for i in _draws(len(y), n, seed) if _both(y[i])]
    return (float(fn(y, a) - fn(y, b)), *_pct(out))


def _placements(y, p):
    """DeLong placement values of the positives and of the negatives."""
    pos, neg = p[y == 1], p[y == 0]
    sn, sp = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sn, pos, "left") + np.searchsorted(sn, pos, "right")) / (2.0 * len(neg))
    v01 = 1 - (np.searchsorted(sp, neg, "left") + np.searchsorted(sp, neg, "right")) / (2.0 * len(pos))
    return v10, v01


def delong_var(y, p):
    """AUROC and its DeLong variance: (auroc, var)."""
    y = _labels(y)
    p = _values(p, len(y))
    if not (y.sum() >= 2 and len(y) - y.sum() >= 2):
        return np.nan, np.nan
    v10, v01 = _placements(y, p)
    return float(v10.mean()), float(np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01))


def delong_paired(y, p1, p2):
    """DeLong test of two AUROCs on the same persons: (auc1, auc2, delta, se, p), delta = auc1 - auc2."""
    y = _labels(y)
    p1, p2 = _values(p1, len(y)), _values(p2, len(y))
    if not (y.sum() >= 2 and len(y) - y.sum() >= 2):
        return (np.nan,) * 5
    a10, a01 = _placements(y, p1)
    b10, b01 = _placements(y, p2)
    s10, s01 = np.cov(a10, b10), np.cov(a01, b01)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(a10) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(a01)
    auc1, auc2 = float(a10.mean()), float(b10.mean())
    se = float(np.sqrt(var)) if var > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(auc1 - auc2) / se)) if se > 0 else np.nan
    return auc1, auc2, auc1 - auc2, se, pval


# --------------------------------------------------------------------------------------------- calibration
def _mle(X, y, w, off, b):
    """Weighted logistic maximum likelihood by Newton-Raphson with step halving; nan when it does not converge."""
    def loglik(c):
        eta = X @ c + off
        return float(w @ (y * eta - np.logaddexp(0, eta)))
    b = np.asarray(b, float).ravel()
    cur = loglik(b)
    for _ in range(100):
        pr = _expit(X @ b + off)
        try:
            step = np.linalg.solve(X.T @ (X * (w * pr * (1 - pr))[:, None]), X.T @ (w * (y - pr)))
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        if np.abs(step).max() < 1e-10:
            return b + step
        t, new = 1.0, loglik(b + step)
        while new < cur - 1e-12 * (1 + abs(cur)) and t > 1e-10:
            t /= 2
            new = loglik(b + t * step)
        if t <= 1e-10:
            break
        b, cur = b + t * step, new
    return np.full(len(b), np.nan)


def fit_logistic_1d(x, y, w=None):
    """Maximum-likelihood (intercept, slope) of the logistic regression of y on x."""
    y = _labels(y)
    x = _values(x, len(y))
    w = np.ones(len(y)) if w is None else _weights(w, len(y))
    b = _mle(np.column_stack([np.ones(len(y)), x]), y, w, 0.0, [logit(w @ y / w.sum()), 0.0])
    return float(b[0]), float(b[1])


def fit_offset_intercept(y, lp, w=None):
    """Maximum-likelihood a in logit P(y = 1) = a + lp, with the slope on lp held at 1."""
    y = _labels(y)
    lp = _values(lp, len(y))
    w = np.ones(len(y)) if w is None else _weights(w, len(y))
    a0 = logit(w @ y / w.sum()) - logit(w @ expit(lp) / w.sum())
    return float(_mle(np.ones((len(y), 1)), y, w, lp, [a0])[0])


def _groups(y, p):
    """Group of each person over the distinct (p, y) pairs, the number of groups, and p and y of each group."""
    u, lev = _levels(p)
    return 2 * lev + y.astype(int), 2 * len(u), np.repeat(u, 2), np.tile([0.0, 1.0], len(u))


def _calib(p, y, w):
    """(citl, slope, oe, brier, scaled_brier) for predictions p, labels y and weights w."""
    keep = w > 0
    p, y, w = p[keep], y[keep], w[keep]
    tot, ev, ex = w.sum(), w @ y, w @ p
    brier = float(w @ (y - p) ** 2 / tot)
    oe = _ratio(ev, ex)
    if not 0 < ev < tot:
        return np.nan, np.nan, oe, brier, np.nan
    prev = ev / tot
    lp = logit(p)
    one = np.ones(len(p))
    citl = _mle(one[:, None], y, w, lp, [logit(prev) - logit(ex / tot)])[0]
    slope = _mle(np.column_stack([one, lp]), y, w, 0.0, [logit(prev), 0.0])[1]
    return float(citl), float(slope), oe, brier, float(1 - brier / (prev * (1 - prev)))


def calib_stats(y, p):
    """Calibration-in-the-large (slope held at 1), calibration slope, O:E, Brier and scaled Brier, as a dict."""
    y = _labels(y)
    g, k, gp, gy = _groups(y, _values(p, len(y), unit=True))
    return dict(zip(CAL_KEYS, _calib(gp, gy, np.bincount(g, minlength=k).astype(float))))


def calib_stats_ci(y, p, n=N_BOOT, seed=SEED):
    """calib_stats with bootstrap 95% CIs: keys <stat>, <stat>_lo, <stat>_hi and n_boot_used."""
    y = _labels(y)
    g, k, gp, gy = _groups(y, _values(p, len(y), unit=True))
    est = _calib(gp, gy, np.bincount(g, minlength=k).astype(float))
    boots = [_calib(gp, gy, np.bincount(g, weights=m, minlength=k)) for m in _resamples(y, n, seed)]
    return _with_ci(CAL_KEYS, est, boots)


def scaled_brier(y, p):
    """1 - Brier / (prevalence * (1 - prevalence))."""
    y = _labels(y)
    p = _values(p, len(y), unit=True)
    prev = y.mean()
    return float(1 - np.mean((y - p) ** 2) / (prev * (1 - prev))) if 0 < prev < 1 else np.nan


def wilson_ci(k, n, z=Z95):
    """Wilson score 95% interval for k events out of n (arrays allowed; nan where n is 0)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ph = k / n
        den = 1 + z * z / n
        mid = (ph + z * z / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    lo, hi = np.clip(mid - half, 0, 1), np.clip(mid + half, 0, 1)
    return (float(lo), float(hi)) if lo.ndim == 0 else (lo, hi)


def decile_curve(y, p, bins=10):
    """Mean predicted and observed risk in quantile bins of p with Wilson 95% CIs; tied p can merge bins."""
    y = _labels(y)
    p = _values(p, len(y), unit=True)
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.digitize(p, q[1:-1])
    k = max(len(q) - 1, 1)
    n = np.bincount(b, minlength=k)
    ev = np.bincount(b, weights=y, minlength=k)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_pred, observed = np.bincount(b, weights=p, minlength=k) / n, ev / n
    lo, hi = wilson_ci(ev, n)
    out = pd.DataFrame(dict(bin=np.arange(1, k + 1), n=n, events=np.rint(ev).astype(int), mean_pred=mean_pred,
                            observed=observed, observed_lo=lo, observed_hi=hi))
    return out[n > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------------------- operating points
def _op(tp, fp, tn, fn):
    """Operating characteristics from confusion counts (nan where a denominator is 0)."""
    return dict(alert_rate=_ratio(tp + fp, tp + fp + tn + fn), se=_ratio(tp, tp + fn), sp=_ratio(tn, tn + fp),
                ppv=_ratio(tp, tp + fp), npv=_ratio(tn, tn + fn), alerts_per_case=_ratio(tp + fp, tp))


def _confusion(y, flag):
    pos = y == 1
    c = [int((flag & pos).sum()), int((flag & ~pos).sum()), int((~flag & ~pos).sum()), int((~flag & pos).sum())]
    return {**_op(*c), **dict(zip(COUNT_KEYS, c))}


def _point_ci(y, flag, reps, seed):
    pos = y == 1
    cells = np.column_stack([flag & pos, flag & ~pos, ~flag & ~pos, ~flag & pos]).astype(float)
    est = _confusion(y, flag)
    boots = [list(_op(*(m @ cells)).values()) for m in _resamples(y, reps, seed)]
    out = _with_ci(OP_KEYS, [est[k] for k in OP_KEYS], boots)
    out.update({k: est[k] for k in COUNT_KEYS})
    return out


def confusion_at(y, p, thr):
    """Alert rate, sensitivity, specificity, PPV, NPV, alerts per case and the counts when p >= thr alerts."""
    y = _labels(y)
    return _confusion(y, _values(p, len(y)) >= thr)


def operating_point_ci(y, p, thr, n=N_BOOT, seed=SEED):
    """confusion_at with bootstrap 95% CIs (<stat>_lo, <stat>_hi); the threshold is the same in every resample."""
    y = _labels(y)
    return _point_ci(y, _values(p, len(y)) >= thr, n, seed)


def rule_point_ci(y, flag, n=N_BOOT, seed=SEED):
    """Operating characteristics of a yes/no rule with bootstrap 95% CIs; the rule is the same in every resample."""
    y = _labels(y)
    return _point_ci(y, _flag(flag, len(y)), n, seed)


def threshold_for_sensitivity(y, p, s):
    """Largest threshold t (a predicted value) whose sensitivity P(p >= t | y = 1) is at least s."""
    y = _labels(y)
    p = _values(p, len(y))
    assert 0 < s <= 1, "target sensitivity must lie in (0, 1]"
    pos = p[y == 1]
    if not len(pos):
        return np.nan
    u, c = np.unique(pos, return_counts=True)
    se = np.cumsum(c[::-1]) / len(pos)
    return float(u[::-1][np.argmax(se >= s)])


def youden_threshold(y, p):
    """Threshold t (a predicted value) maximising sensitivity + specificity - 1 for p >= t; ties go to the larger t."""
    y = _labels(y)
    p = _values(p, len(y))
    if not _both(y):
        return np.nan
    u, lev = _levels(p)
    pos = np.bincount(lev, weights=y, minlength=len(u))[::-1]
    neg = np.bincount(lev, weights=1 - y, minlength=len(u))[::-1]
    j = np.cumsum(pos) / pos.sum() - np.cumsum(neg) / neg.sum()
    return float(u[::-1][np.flatnonzero(j >= j.max() - TIE)[0]])


def alert_match_threshold(p, target_rate):
    """Predicted value t whose alert rate mean(p >= t) is closest to target_rate, ties to the larger t: (t, rate)."""
    p = np.asarray(p, float).ravel()
    bad = int((~np.isfinite(p)).sum())
    assert bad == 0 and len(p) > 0, f"{bad:,} scores missing"
    assert 0 <= target_rate <= 1, "target alert rate must lie in [0, 1]"
    u, c = np.unique(p, return_counts=True)
    rate = np.cumsum(c[::-1])[::-1] / len(p)
    d = np.abs(rate - target_rate)
    j = np.flatnonzero(d <= d.min() + TIE)[-1]
    return float(u[j]), float(rate[j])


# ------------------------------------------------------------------------------------------- net benefit
def _pts(thresholds):
    pt = np.atleast_1d(np.asarray(thresholds, float))
    assert ((pt > 0) & (pt < 1)).all(), "threshold probabilities must lie in (0, 1)"
    return pt


def _nb_rows(tp, fp, size, prev, pt, key):
    odds = pt / (1 - pt)
    nb, nb_all = tp / size - fp / size * odds, prev - (1 - prev) * odds
    return [{"threshold": float(t), key: float(a), "nb_all": float(b), "nb_none": 0.0}
            for t, a, b in zip(pt, nb, nb_all)]


def net_benefit(y, p, thresholds):
    """For each threshold probability pt: net benefit of alerting when p >= pt, and of alerting all or none."""
    y = _labels(y)
    p = _values(p, len(y))
    pt = _pts(thresholds)
    sp, sn = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(sp) - np.searchsorted(sp, pt, "left")
    fp = len(sn) - np.searchsorted(sn, pt, "left")
    return _nb_rows(tp, fp, len(y), y.mean(), pt, "nb_model")


def net_benefit_ci(y, p, thresholds, n=N_BOOT, seed=SEED):
    """net_benefit rows with nb_model_lo, nb_model_hi (bootstrap 95% limits; the alert p >= pt is the same in every
    resample) and n_boot_used. One pass per resample: each person's level is the number of pt at or below p."""
    rows = net_benefit(y, p, thresholds)
    y = _labels(y)
    pt = _pts(thresholds)
    s = np.sort(pt)
    at = np.searchsorted(s, pt, "left") + 1            # an alert at pt needs a level of at least this
    grp = 2 * np.searchsorted(s, _values(p, len(y)), "right") + y.astype(int)
    odds = pt / (1 - pt)
    boots = []
    for m in _resamples(y, n, seed):
        c = np.bincount(grp, weights=m, minlength=2 * len(s) + 2).reshape(-1, 2)    # (level, label)
        c = np.cumsum(c[::-1], axis=0)[::-1]           # persons at each level or above
        boots.append(c[at, 1] / len(y) - c[at, 0] / len(y) * odds)
    lo, hi = (np.atleast_1d(v) for v in _pct(boots, len(pt)))
    for r, a, b in zip(rows, lo, hi):
        r.update(nb_model_lo=float(a), nb_model_hi=float(b), n_boot_used=len(boots))
    return rows


def net_benefit_rule(y, flag, thresholds):
    """Net benefit of a yes/no rule (the same alerts at every pt), with test-all and test-none, for each pt."""
    y = _labels(y)
    f = _flag(flag, len(y))
    pt = _pts(thresholds)
    tp = np.full(len(pt), float((f & (y == 1)).sum()))
    fp = np.full(len(pt), float((f & (y == 0)).sum()))
    return _nb_rows(tp, fp, len(y), y.mean(), pt, "nb_rule")


def fixed_alert_nb_ci(y, alert, thresholds, n=N_BOOT, seed=SEED):
    """net_benefit_rule rows of a fixed 0/1 alert (a rule, or p >= t with t held fixed) with nb_rule_lo, nb_rule_hi
    (bootstrap 95% limits over the resamples of the other CIs) and n_boot_used. The alert is the same at every pt and
    in every resample, so a resample needs only its TP and FP, and one product gives its net benefit at every pt."""
    rows = net_benefit_rule(y, alert, thresholds)
    y = _labels(y)
    f = _flag(alert, len(y))
    pt = _pts(thresholds)
    tp, fp = (f & (y == 1)).astype(float), (f & (y == 0)).astype(float)       # the persons counted in TP and in FP
    rates = np.array([(m @ tp, m @ fp) for m in _resamples(y, n, seed)]).reshape(-1, 2) / len(y)     # TP/n, FP/n
    lo, hi = (np.atleast_1d(v) for v in _pct(rates[:, :1] - rates[:, 1:] * (pt / (1 - pt)), len(pt)))
    for r, a, b in zip(rows, lo, hi):
        r.update(nb_rule_lo=float(a), nb_rule_hi=float(b), n_boot_used=len(rates))
    return rows


def roc_on_grid(y, p, step=0.005):
    """Sensitivity on the empirical ROC curve at false-positive rates 0, step, ..., 1. The curve joins its vertices
    (one per distinct score) by straight lines, so tied scores give one segment and the area under the curve is the
    AUROC with ties counted one half; where several vertices share a rate, the largest sensitivity is taken."""
    y = _labels(y)
    p = _values(p, len(y))
    assert _both(y), "both classes are needed"
    k = int(round(1 / step))
    assert abs(k * step - 1) < 1e-9, "1 must be a whole multiple of step"
    u, lev = _levels(p)
    tp = np.concatenate([[0.0], np.cumsum(np.bincount(lev, weights=y, minlength=len(u))[::-1])])
    fp = np.concatenate([[0.0], np.cumsum(np.bincount(lev, weights=1 - y, minlength=len(u))[::-1])])
    g = np.arange(k + 1) / k
    x = g * fp[-1]                                              # false positives at each grid rate
    j = np.searchsorted(fp, x + 1e-6, side="right") - 1         # last vertex at or before x (top of a vertical run)
    nxt = np.minimum(j + 1, len(fp) - 1)
    span = fp[nxt] - fp[j]
    frac = np.divide(np.maximum(x - fp[j], 0.0), span, out=np.zeros_like(x), where=span > 0)
    return pd.DataFrame(dict(fpr=g, tpr=(tp[j] + frac * (tp[nxt] - tp[j])) / tp[-1]))


# ---------------------------------------------------------------------------------------- survey weights
def weighted_prevalence(y, w, n=N_BOOT, seed=SEED):
    """Weighted prevalence and a person-resample 95% CI that ignores strata and clusters: (est, lo, hi)."""
    y = _labels(y)
    w = _weights(w, len(y))
    est = float(w @ y / w.sum())
    return (est, *_pct([_ratio((m * w) @ y, m @ w) for m in _resamples(y, n, seed)]))


def weighted_auroc(y, p, w, n=N_BOOT, seed=SEED):
    """Weighted AUROC and a person-resample 95% CI that ignores strata and clusters: (est, lo, hi)."""
    y = _labels(y)
    w = _weights(w, len(y))
    u, lev = _levels(_values(p, len(y)))
    est = _auc(lev, len(u), y, w)
    return (est, *_pct([_auc(lev, len(u), y, m * w) for m in _resamples(y, n, seed)]))


def weighted_oe(y, p, w, n=N_BOOT, seed=SEED):
    """Observed over expected with survey weights, and a person-resample 95% CI that ignores strata and clusters."""
    y = _labels(y)
    w = _weights(w, len(y))
    p = _values(p, len(y), unit=True)
    est = _ratio(w @ y, w @ p)
    return (est, *_pct([_ratio((m * w) @ y, (m * w) @ p) for m in _resamples(y, n, seed)]))
