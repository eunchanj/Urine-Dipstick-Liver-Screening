"""s04 - logistic-regression comparators of the ensemble and paired AUROC differences (run by the data holder).

Fits two logistic regressions per outcome on the training persons: LR_ASB (age, male, BMI) and LR_ALL (all inputs),
each on standardised inputs with an L2 penalty (C = 1) and no calibration step; age, BMI, urine pH, urine specific
gravity and the graded dipstick results enter as linear terms, without interactions. The training persons are checked
against train_summary.csv, so the regressions see the persons the ensemble was fitted on. Reads the persons of
cache/dev.parquet and cache/knhanes.parquet through cohort.outcome_frame, and the ensemble predictions of s03 (checked
against performance.csv). Writes the regressions to MODEL_DIR (lr_<MODEL>_<OUTCOME>.joblib),
cache/preds_<MODEL>_<OUTCOME>_<DATASET>.parquet for DEV_TEST and KN, and these aggregate tables:

  model_comparison.csv   AUROC with 95% CI and a calibration summary of the ensemble and the two regressions; for each
                         outcome and dataset three AUROC differences (ENS - LR_ALL, LR_ALL - LR_ASB, ENS - LR_ASB)
                         with paired bootstrap 95% CIs and DeLong P values, the 12 formal tests in all
  roc_points_lr.csv      sensitivity of the two regressions at false-positive rates 0 to 1 by 0.005
  lr_coefficients.csv    coefficients on the standardised and on the unit scale, with the scaler means and SDs

Run from the working folder:
    s=s04_comparators; python $s.py > logs/$s.out 2> cache/$s.err
"""
import os
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

LR_MODELS = ("LR_ASB", "LR_ALL")      # inputs in cohort.LR_INPUTS
ALL_MODELS = ("ENS",) + LR_MODELS
PAIRS = [("ENS", "LR_ALL"),           # the learner: same inputs
         ("LR_ALL", "LR_ASB"),        # the dipstick: same learner, nested inputs
         ("ENS", "LR_ASB")]           # the two together
FORMAL_TESTS = 12                     # 3 pairs x 2 datasets x 2 outcomes; no other P values are computed
LR_C = 1.0
LR_TOL = 1e-10              # lbfgs gradient tolerance: coefficients within about 1e-8 of the optimum
MAX_ITER = 1000
GRAD_MAX = 1e-6             # largest gradient of the penalised mean log-loss accepted at the fitted coefficients
ROC_STEP = 0.005
INTERCEPT = "(intercept)"
CMP_COLS = ["row_type", "outcome", "dataset", "model", "model_b", "n", "events", "auroc", "auroc_lo", "auroc_hi",
            "auprc", "brier", "scaled_brier", "oe", "citl", "slope", "delta", "delta_lo", "delta_hi", "delong_delta",
            "delong_se", "delong_p", "n_boot", "preds_sha256"]
ROC_COLS = ["outcome", "dataset", "model", "fpr", "tpr"]
COEF_COLS = ["outcome", "model", "term", "coef_std", "coef_unit", "center", "scale", "n_train", "events_train"]
TABLES = {"model_comparison.csv": (CMP_COLS, ["n", "events"]),
          "roc_points_lr.csv": (ROC_COLS, []),
          "lr_coefficients.csv": (COEF_COLS, ["n_train", "events_train"])}


def make_lr():
    """Standardised inputs, then logistic regression with an L2 penalty (C = 1) fitted by lbfgs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", C=LR_C, solver="lbfgs", tol=LR_TOL, max_iter=MAX_ITER))])


def training_persons(C, outcome, summary):
    """(X, y) of the outcome's training persons, once they agree with the ensemble's in train_summary.csv."""
    row = C.summary_row(summary, outcome)
    X, y, keys = C.checked_frame(outcome, "DEV_TRAIN")
    C.check_training(row, X, y, keys)
    return X, y


def penalised_gradient(lr, X, y):
    """Gradient of mean log-loss + |w|^2 / (2 C n) (intercept not penalised) at the fitted coefficients."""
    sc, est = lr.named_steps["scale"], lr.named_steps["lr"]
    r = lr.predict_proba(X)[:, 1] - np.asarray(y, float)
    return np.r_[r.mean(), sc.transform(X).T @ r / len(r) + est.coef_[0] / (est.C * len(r))]


def fit(C, X, y, model):
    """(regression, lbfgs iterations) of one model on the training persons, checked to be at the optimum."""
    cols = C.LR_INPUTS[model]
    lr = make_lr().fit(X[cols], y.to_numpy(np.int64))
    est = lr.named_steps["lr"]
    it = int(np.max(est.n_iter_))
    assert it < MAX_ITER, f"{model}: lbfgs reached {MAX_ITER:,} iterations without converging"
    assert [int(c) for c in est.classes_] == [0, 1], f"{model}: classes are not 0 and 1"
    worst = float(np.abs(penalised_gradient(lr, X[cols], y)).max())
    assert worst <= GRAD_MAX, f"{model}: the fitted coefficients are not at the optimum of the penalised likelihood"
    return lr, it


def small_binary(x):
    """True for a 0/1 input with 1-9 persons at one of its two values."""
    v = np.asarray(x, float)
    if not np.isin(v, (0.0, 1.0)).all():
        return False
    k = int(v.sum())
    return 0 < min(k, len(v) - k) < runlog.SMALL


def coefficient_rows(C, outcome, model, lr, X, n, events):
    """(lr_coefficients.csv rows, number of inputs left blank) on the standardised and the unit scale."""
    # A 0/1 input with 1-9 persons at one value: its mean, SD and unit-scale values, and the unit intercept, would
    # give that count, so they are left blank.
    cols = C.LR_INPUTS[model]
    sc, est = lr.named_steps["scale"], lr.named_steps["lr"]
    b, b0 = est.coef_[0].astype(float), float(est.intercept_[0])
    mean, sd = sc.mean_.astype(float), sc.scale_.astype(float)
    unit = b / sd
    hide = [small_binary(X[c]) for c in cols]
    base = dict(outcome=outcome, model=model, n_train=n, events_train=events)
    rows = [dict(base, term=INTERCEPT, coef_std=b0, coef_unit=None if any(hide) else b0 - float(unit @ mean),
                 center=None, scale=None)]
    for j, c in enumerate(cols):
        rows.append(dict(base, term=c, coef_std=float(b[j]), coef_unit=None if hide[j] else float(unit[j]),
                         center=None if hide[j] else float(mean[j]), scale=None if hide[j] else float(sd[j])))
    return rows, sum(hide)


def ensemble_predictions(C, perf, outcome, dataset, y, keys):
    """(p, sha) of the ensemble from s03, once its persons, labels and hash agree with the frame and performance.csv."""
    return C.checked_preds("ENS", outcome, dataset, keys, y, C.recorded_sha("ENS", outcome, dataset, perf))


def lr_predictions(C, lr, model, outcome, dataset, X, y, keys):
    """(p, sha) of one regression for one dataset; the prediction file is written and read back."""
    p = np.asarray(lr.predict_proba(X[C.LR_INPUTS[model]])[:, 1], dtype=float)
    where = f"{outcome} {dataset} {model}:"
    bad = int((~np.isfinite(p) | (p < 0) | (p > 1)).sum())
    assert bad == 0, f"{where} {count_text(bad)} predictions missing or outside 0-1"
    sha = C.write_preds(model, outcome, dataset, keys, y, p)
    back = C.read_preds(model, outcome, dataset).attrs["preds_sha256"]
    assert back == sha, f"{where} the prediction file read back differs from what was written"
    return p, sha


def model_row(common, B, seed, key, model, y, p, sha):
    """model_comparison.csv row of one model: AUROC with its CI, average precision and calibration."""
    auc, lo, hi = common.auroc_ci(y, p, B, seed)
    cal = common.calib_stats(y, p)
    return dict(row_type="model", **key, model=model, n=len(y), events=int(y.sum()), auroc=auc, auroc_lo=lo,
                auroc_hi=hi, auprc=common.auprc(y, p), brier=cal["brier"], scaled_brier=cal["scaled_brier"],
                oe=cal["oe"], citl=cal["citl"], slope=cal["slope"], n_boot=B, preds_sha256=sha)


def delta_row(common, B, seed, key, a, b, y, pa, pb):
    """model_comparison.csv row of AUROC(a) - AUROC(b), with its CI from shared resamples and the DeLong P value."""
    d, lo, hi = common.paired_boot(y, pa, pb, None, B, seed)
    _, _, dd, se, pv = common.delong_paired(y, pa, pb)
    return dict(row_type="delta", **key, model=a, model_b=b, n=len(y), events=int(y.sum()), delta=d, delta_lo=lo,
                delta_hi=hi, delong_delta=dd, delong_se=se, delong_p=pv, n_boot=B)


def main():
    t0 = time.time()
    import paths_local as P
    import common
    import cohort as C
    import joblib

    C.remove_files([os.path.join(runlog.HERE, name) for name in TABLES]
                   + [C.model_path(m, o) for o in C.OUTCOMES for m in LR_MODELS]
                   + [C.preds_path(m, o, d) for o in C.OUTCOMES for m in LR_MODELS for d in C.PRED_DATASETS])
    B = common.n_boot(P.SYNTHETIC)
    summary = C.read_table("train_summary.csv")
    perf = C.read_table("performance.csv")
    os.makedirs(P.MODEL_DIR, exist_ok=True)
    ignore = os.path.join(P.MODEL_DIR, ".gitignore")       # as in s02: the model folder stays out of git
    if not os.path.exists(ignore):
        with open(ignore, "w") as fh:
            fh.write("*\n")
    say(f"bootstrap replicates: {B:,}")
    say("logistic regressions: standardised inputs, L2 penalty with C = 1, lbfgs to a gradient below 1e-6, "
        "no calibration step")

    parts = {name: [] for name in TABLES}
    for outcome in C.OUTCOMES:
        t1 = time.time()
        X, y = training_persons(C, outcome, summary)
        n, events = len(y), int(y.sum())
        say(f"{outcome}: training persons {count_text(n)}, events {count_text(events)}, as in train_summary.csv "
            f"({time.time() - t1:.0f} s)")
        fitted = {}
        for model in LR_MODELS:
            lr, it = fit(C, X, y, model)
            joblib.dump(lr, C.model_path(model, outcome))
            rows, hidden = coefficient_rows(C, outcome, model, lr, X, n, events)
            parts["lr_coefficients.csv"].append(pd.DataFrame(rows, columns=COEF_COLS))
            fitted[model] = lr
            say(f"{outcome} {model}: {len(C.LR_INPUTS[model]):,} inputs, lbfgs converged in {it:,} iterations")
            if hidden:
                say(f"{outcome} {model}: unit-scale values left blank for {hidden:,} input(s) with 1-9 persons at "
                    f"one value")
        for dataset in C.PRED_DATASETS:
            t1 = time.time()
            Xd, yd, keys = C.checked_frame(outcome, dataset)
            p, sha = {}, {}
            p["ENS"], sha["ENS"] = ensemble_predictions(C, perf, outcome, dataset, yd, keys)
            for model in LR_MODELS:
                p[model], sha[model] = lr_predictions(C, fitted[model], model, outcome, dataset, Xd, yd, keys)
            yv = yd.to_numpy(float)
            key = dict(outcome=outcome, dataset=dataset)
            rows = [model_row(common, B, C.SEED, key, m, yv, p[m], sha[m]) for m in ALL_MODELS]
            rows += [delta_row(common, B, C.SEED, key, a, b, yv, p[a], p[b]) for a, b in PAIRS]
            parts["model_comparison.csv"].append(pd.DataFrame(rows, columns=CMP_COLS))
            for m in LR_MODELS:
                parts["roc_points_lr.csv"].append(common.roc_on_grid(yv, p[m], ROC_STEP).assign(**key, model=m))
            say(f"{outcome} {dataset}: persons {count_text(len(yv))}, events {count_text(int(yv.sum()))} "
                f"({time.time() - t1:.0f} s)")

    n_tests = sum(int((t["row_type"] == "delta").sum()) for t in parts["model_comparison.csv"])
    assert n_tests == FORMAL_TESTS, f"{n_tests:,} AUROC differences tested, not {FORMAL_TESTS:,}"
    for name, (cols, counts) in TABLES.items():
        runlog.write_table(pd.concat(parts[name], ignore_index=True)[cols], name, counts)
    say("tables written: " + ", ".join(TABLES))
    say(f"s04 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
