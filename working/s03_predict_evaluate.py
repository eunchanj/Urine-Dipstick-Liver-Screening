"""s03 - predictions of the calibrated ensemble, with its discrimination and calibration (run by the data holder).

Reads the ensembles in MODEL_DIR and, through cohort.outcome_frame, the persons of cache/dev.parquet and
cache/knhanes.parquet. Each ensemble and its training persons are checked against train_summary.csv, and the persons
and events of every dataset against cohort_flow.csv. For each outcome and for DEV_TEST and KN it writes
cache/preds_ENS_<OUTCOME>_<DATASET>.parquet (row, id, y, p) and these aggregate tables:

  performance.csv           persons, events, prevalence, mean prediction, AUROC and average precision with 95% CIs
  calibration.csv           calibration-in-the-large (slope held at 1), slope, O:E, Brier and scaled Brier with 95%
                            CIs; the predictions that are exactly 0 or 1, and the slope without them
  calibration_bins.csv      10 quantile bins of the prediction: persons, events, mean prediction, observed risk
  prediction_histogram.csv  persons and events in 50 bins of the prediction of width 0.02
  roc_points.csv            sensitivity on the ROC curve at false-positive rates 0 to 1 by 0.005

Counts of 1-9 are written <10. A row that shows the persons of a group and one part of them (events, or non-events at
p = 1) leaves that part blank, with the observed risks computed from it, when the other part is 1-9 and the shown part
10 or more, for example the events of a bin whose non-events are 1-9, so that no count of 1-9 can be worked out from
the row.

Run from the working folder:
    s=s03_predict_evaluate; python $s.py > logs/$s.out 2> cache/$s.err
"""
import os
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

MODEL = "ENS"
BINS = 10
HIST_EDGES = np.arange(51) / 50         # 50 bins of width 0.02; the last one includes 1
ROC_STEP = 0.005
PERF_COLS = ["outcome", "dataset", "model", "n", "events", "prevalence", "mean_pred", "auroc", "auroc_lo", "auroc_hi",
             "auprc", "auprc_lo", "auprc_hi", "n_boot", "preds_sha256"]
CAL_COLS = ["outcome", "dataset", "model", "n", "events", "prevalence", "mean_pred", "oe", "oe_lo", "oe_hi", "citl",
            "citl_lo", "citl_hi", "slope", "slope_lo", "slope_hi", "brier", "brier_lo", "brier_hi", "scaled_brier",
            "scaled_brier_lo", "scaled_brier_hi", "n_p0", "events_p0", "n_p1", "nonevents_p1", "slope_excl01",
            "slope_excl01_lo", "slope_excl01_hi", "n_boot"]
BIN_COLS = ["outcome", "dataset", "model", "bins_requested", "bin", "n", "events", "mean_pred", "observed",
            "observed_lo", "observed_hi"]
HIST_COLS = ["outcome", "dataset", "model", "bin_lo", "bin_hi", "n", "events"]
ROC_COLS = ["outcome", "dataset", "model", "fpr", "tpr"]
TABLES = {"performance.csv": (PERF_COLS, ["n", "events"]),
          "calibration.csv": (CAL_COLS, ["n", "events", "n_p0", "events_p0", "n_p1", "nonevents_p1"]),
          "calibration_bins.csv": (BIN_COLS, ["n", "events"]),
          "prediction_histogram.csv": (HIST_COLS, ["n", "events"]),
          "roc_points.csv": (ROC_COLS, [])}
REST_PAIRS = (("n_p0", "events_p0"), ("n_p1", "nonevents_p1"))     # calibration.csv: (persons, the part shown)


def small_rest(total, part):
    """True where part is 10 or more and total - part is 1-9, a count that the row would give by difference."""
    total, part = np.asarray(total, float), np.asarray(part, float)
    return (part >= runlog.SMALL) & (total - part >= 1) & (total - part < runlog.SMALL)


def trained_model(C, outcome, summary):
    """The ensemble of one outcome, once it and its training persons agree with train_summary.csv."""
    row = C.summary_row(summary, outcome)
    model = C.load_ensemble(outcome, row)
    C.check_training(row, *C.outcome_frame(outcome, "DEV_TRAIN"))
    return model


def predictions(C, model, outcome, dataset):
    """(y, p, sha) of one outcome and dataset; the prediction file is written and read back."""
    X, y, keys = C.checked_frame(outcome, dataset)
    where = f"{outcome} {dataset}:"
    p = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    bad = int((~np.isfinite(p) | (p < 0) | (p > 1)).sum())
    assert bad == 0, f"{where} {count_text(bad)} predictions missing or outside 0-1"
    sha = C.write_preds(MODEL, outcome, dataset, keys, y, p)
    back = C.read_preds(MODEL, outcome, dataset).attrs["preds_sha256"]
    assert back == sha, f"{where} the prediction file read back differs from what was written"
    return y.to_numpy(float), p, sha


def evaluate(common, seed, outcome, dataset, y, p, B, sha):
    """Rows of the five tables for one outcome and dataset."""
    n, events = len(y), int(y.sum())
    key = dict(outcome=outcome, dataset=dataset, model=MODEL)
    base = dict(**key, n=n, events=events, prevalence=events / n, mean_pred=float(p.mean()))
    auc, auc_lo, auc_hi = common.auroc_ci(y, p, B, seed)
    ap, ap_lo, ap_hi = common.auprc_ci(y, p, B, seed)
    perf = dict(**base, auroc=auc, auroc_lo=auc_lo, auroc_hi=auc_hi, auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi,
                n_boot=B, preds_sha256=sha)
    zero, one = p == 0, p == 1
    # The slope again on the persons with 0 < p < 1: a prediction of exactly 0 or 1 enters logit(p) at -/+20.7.
    inner = common.calib_stats_ci(y[~zero & ~one], p[~zero & ~one], B, seed)
    cal = dict(**base, **common.calib_stats_ci(y, p, B, seed), n_p0=int(zero.sum()), events_p0=int(y[zero].sum()),
               n_p1=int(one.sum()), nonevents_p1=int((1 - y[one]).sum()), slope_excl01=inner["slope"],
               slope_excl01_lo=inner["slope_lo"], slope_excl01_hi=inner["slope_hi"], n_boot=B)
    for total, part in REST_PAIRS:
        if small_rest(cal[total], cal[part]):
            cal[part] = None
    bins = common.decile_curve(y, p, BINS).assign(**key, bins_requested=BINS)
    small_n, small_events = (bins[c].between(1, runlog.SMALL - 1).to_numpy() for c in ("n", "events"))
    small_non = small_rest(bins["n"], bins["events"])                  # non-events (n - events) of 1-9
    bins.loc[small_n, ["mean_pred", "observed", "observed_lo", "observed_hi"]] = np.nan
    bins.loc[small_events | small_non, ["observed", "observed_lo", "observed_hi"]] = np.nan
    bins["events"] = bins["events"].where(~small_non)
    hist = pd.DataFrame(dict(**key, bin_lo=HIST_EDGES[:-1], bin_hi=HIST_EDGES[1:], n=np.histogram(p, HIST_EDGES)[0],
                             events=np.histogram(p[y == 1], HIST_EDGES)[0]))
    hist["events"] = hist["events"].where(~small_rest(hist["n"], hist["events"]))
    roc = common.roc_on_grid(y, p, ROC_STEP).assign(**key)
    return {"performance.csv": pd.DataFrame([perf], columns=PERF_COLS),
            "calibration.csv": pd.DataFrame([cal], columns=CAL_COLS),
            "calibration_bins.csv": bins[BIN_COLS], "prediction_histogram.csv": hist[HIST_COLS],
            "roc_points.csv": roc[ROC_COLS]}


def main():
    t0 = time.time()
    import paths_local as P
    import common
    import cohort as C

    C.remove_files([os.path.join(runlog.HERE, name) for name in TABLES]
                   + [C.preds_path(MODEL, o, d) for o in C.OUTCOMES for d in C.PRED_DATASETS])
    B = common.n_boot(P.SYNTHETIC)
    summary = C.read_table("train_summary.csv")
    say(f"bootstrap replicates: {B:,}")

    parts = {name: [] for name in TABLES}
    for outcome in C.OUTCOMES:
        t1 = time.time()
        model = trained_model(C, outcome, summary)
        say(f"{outcome}: ensemble and training persons agree with train_summary.csv ({time.time() - t1:.0f} s)")
        for dataset in C.PRED_DATASETS:
            t1 = time.time()
            y, p, sha = predictions(C, model, outcome, dataset)
            for name, df in evaluate(common, C.SEED, outcome, dataset, y, p, B, sha).items():
                parts[name].append(df)
            say(f"{outcome} {dataset}: persons {count_text(len(y))}, events {count_text(int(y.sum()))}, "
                f"predictions exactly 0 {count_text(int((p == 0).sum()))}, exactly 1 {count_text(int((p == 1).sum()))} "
                f"({time.time() - t1:.0f} s)")

    for name, (cols, counts) in TABLES.items():
        runlog.write_table(pd.concat(parts[name], ignore_index=True)[cols], name, counts)
    say("tables written: " + ", ".join(TABLES))
    say(f"s03 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
