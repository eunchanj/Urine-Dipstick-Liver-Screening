"""s07 - SHAP summaries of the calibrated ensemble on test persons (run by the data holder).

For each outcome: TreeSHAP values of the 15 base learners of the ensemble (LightGBM, XGBoost and CatBoost in each of
the 5 calibration folds) on a random sample of SHAP_N = 5,000 DEV_TEST persons (seed 42; all of them when fewer).
Each learner is explained on its own log-odds margin by its library's tree-path-dependent TreeSHAP (LightGBM
pred_contrib, XGBoost pred_contribs, CatBoost ShapValues: the routines shap.TreeExplainer calls for these models),
and the values are averaged over the 15 learners with equal weights. The ensemble is checked against
train_summary.csv, the test persons against cohort_flow.csv and the s03 prediction file, and the values of each
learner against its margin. Writes cache/shap_<OUTCOME>.parquet (row, id, the averaged value of each input) and two
aggregate tables:

  shap_summary.csv    mean absolute SHAP value of each input, over the 15 learners and per learner, and its rank
  shap_by_level.csv   persons and mean SHAP value at each level of each input (age in 5-year bands, BMI in five bands
                      that share the 25 kg/m2 boundary of the BMI25 rule)

Both outcomes use the same sampled persons, since their DEV_TEST frames hold the same persons in the same order.

Run from the working folder:
    s=s07_shap; python $s.py > logs/$s.out 2> cache/$s.err
"""
import os
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

MODEL = "ENS"
DATASET = "DEV_TEST"
LEARNERS = ("lgb", "xgb", "cat")
MARGIN_TOL = 1e-3      # SHAP values plus the bias term against the learner's log-odds margin
PROB_TOL = 1e-5        # logistic of the margin against the learner's predicted probability
PRED_TOL = 1e-9        # ensemble prediction against the s03 prediction file
AGE_EDGES = [19] + list(range(25, 90, 5)) + [np.inf]
AGE_LABELS = ["19-24"] + [f"{a}-{a + 4}" for a in range(25, 85, 5)] + ["85+"]
BMI_EDGES = [-np.inf, 18.5, 23.0, 25.0, 30.0, np.inf]        # kg/m2; each band includes its lower edge
BMI_LABELS = ["<18.5", "18.5-22.9", "23.0-24.9", "25.0-29.9", "30+"]
BANDED = {"age": (AGE_EDGES, AGE_LABELS), "bmi": (BMI_EDGES, BMI_LABELS)}      # inputs reported in bands
SUMMARY_COLS = ["outcome", "dataset", "n_records", "n_learners", "input", "label", "mean_abs_shap", "rank",
                "mean_abs_shap_lgb", "mean_abs_shap_xgb", "mean_abs_shap_cat"]
LEVEL_COLS = ["outcome", "dataset", "input", "level", "n", "mean_shap"]
TABLES = {"shap_summary.csv": (SUMMARY_COLS, ["n_records"]),
          "shap_by_level.csv": (LEVEL_COLS, ["n"])}


def shap_path(outcome):
    return os.path.join(runlog.CACHE, f"shap_{outcome}.parquet")


def members(model):
    """(fold, learner name, fitted learner) for every base learner of the calibrated ensemble."""
    return [(k, name, est) for k, cc in enumerate(model.calibrated_classifiers_)
            for name, est in cc.estimator.named_estimators_.items()]


def ensemble(C, outcome, summary):
    """The ensemble of one outcome, once its file agrees with train_summary.csv and every learner uses INPUTS."""
    row = C.summary_row(summary, outcome)
    assert row["learners"] == ",".join(LEARNERS), "the learners differ from train_summary.csv"
    model = C.load_ensemble(outcome, row)
    folds = len(model.calibrated_classifiers_)
    assert str(folds) == row["calibration_folds"], f"{folds:,} calibration folds, not as in train_summary.csv"
    for cc in model.calibrated_classifiers_:
        assert tuple(cc.estimator.named_estimators_) == LEARNERS, "a calibration fold holds other learners"
    for _, name, est in members(model):
        names = est.feature_names_ if name == "cat" else est.feature_names_in_
        assert list(names) == list(C.INPUTS), "a learner's inputs differ from INPUTS"
    return model


def persons(C, perf, outcome):
    """(X, keys, p) of the outcome's DEV_TEST persons, once they agree with cohort_flow.csv and the s03 predictions."""
    X, y, keys = C.checked_frame(outcome, DATASET)
    p, _ = C.checked_preds(MODEL, outcome, DATASET, keys, y, C.recorded_sha(MODEL, outcome, DATASET, perf))
    return X, keys, p


def sample_rows(n, cap, seed):
    """Sorted positions of a random sample of cap of n persons; all n when n <= cap."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, size=cap, replace=False))


def tree_values(name, est, X):
    """(log-odds margin, TreeSHAP values with the bias term in the last column) of one learner on X."""
    # The shap package is not imported: shap 0.48 cannot load XGBoost 3.1 models, and its TreeExplainer runs these
    # same routines for the three learners.
    if name == "lgb":
        return est.predict(X, raw_score=True), est.predict(X, pred_contrib=True)
    if name == "xgb":
        import xgboost as xgb
        booster, dm = est.get_booster(), xgb.DMatrix(X)
        return booster.predict(dm, output_margin=True), booster.predict(dm, pred_contribs=True)
    assert name == "cat", "unknown learner"
    import catboost as cb
    return (est.predict(X, prediction_type="RawFormulaVal"),
            est.get_feature_importance(data=cb.Pool(X), type="ShapValues"))


def learner_shap(model, X):
    """(values, names): TreeSHAP values of every learner on X, learners x persons x inputs, and the learner names."""
    from scipy.special import expit
    values, names = [], []
    for _, name, est in members(model):
        margin, phi = (np.asarray(a, dtype=float) for a in tree_values(name, est, X))
        assert phi.shape == (len(X), X.shape[1] + 1), f"{name}: TreeSHAP values of another shape"
        gap = float(np.abs(phi.sum(axis=1) - margin).max())
        assert gap <= MARGIN_TOL, f"{name}: TreeSHAP values do not add up to the log-odds margin"
        gap = float(np.abs(expit(margin) - est.predict_proba(X)[:, 1]).max())
        assert gap <= PROB_TOL, f"{name}: the margin is not the log-odds of the predicted probability"
        values.append(phi[:, :-1])
        names.append(name)
    return np.stack(values), np.array(names)


def summary_rows(C, outcome, n, values, names):
    """shap_summary.csv rows: mean absolute value of each input over the persons, averaged over the learners."""
    per = np.abs(values).mean(axis=1)
    overall = per.mean(axis=0)
    rank = pd.Series(overall).rank(ascending=False, method="min").astype(int).to_numpy()
    return [dict(outcome=outcome, dataset=DATASET, n_records=n, n_learners=len(names), input=col,
                 label=C.INPUT_LABELS[col], mean_abs_shap=float(overall[j]), rank=int(rank[j]),
                 **{f"mean_abs_shap_{k}": float(per[names == k, j].mean()) for k in LEARNERS})
            for j, col in enumerate(C.INPUTS)]


def levels(C, col, x):
    """(level of each person, levels in order): the bands of BANDED for age and BMI, else the coded values as text."""
    if col in BANDED:
        edges, labels = BANDED[col]
        band = pd.cut(x, edges, labels=labels, right=False)
        out = int(pd.isna(band).sum())
        assert out == 0, f"{count_text(out)} sampled persons outside the {col} bands"
        return np.asarray(band.astype(str), dtype=object), list(labels)
    u, inv = np.unique(x, return_inverse=True)
    text = np.array([C.level_text(v) for v in u], dtype=object)
    return text[np.ravel(inv)], list(dict.fromkeys(text))


def level_rows(C, outcome, X, mean_values):
    """shap_by_level.csv rows: persons and mean value at each level of each input (mean blank under 10 persons)."""
    rows = []
    for j, col in enumerate(C.INPUTS):
        lab, order = levels(C, col, X[col].to_numpy(float))
        g = pd.Series(mean_values[:, j]).groupby(lab).agg(["size", "mean"]).reindex(order)
        n = g["size"].fillna(0).to_numpy(np.int64)
        mean = np.where(n >= runlog.SMALL, g["mean"].to_numpy(float), np.nan)
        rows += [dict(outcome=outcome, dataset=DATASET, input=col, level=lev, n=int(k), mean_shap=v)
                 for lev, k, v in zip(order, n, mean)]
    return rows


def outcome_shap(C, model, outcome, X, keys, p, cap):
    """(summary rows, level rows, cache frame) of one outcome on a random sample of at most cap test persons."""
    rows = sample_rows(len(X), cap, C.SEED)
    Xs = X.iloc[rows].reset_index(drop=True)
    gap = float(np.abs(model.predict_proba(Xs)[:, 1] - p[rows]).max())
    assert gap <= PRED_TOL, f"{outcome} {DATASET}: the ensemble's predictions differ from the s03 prediction file"
    values, names = learner_shap(model, Xs)
    mean_values = values.mean(axis=0)
    cache = pd.DataFrame({"row": keys["row"].to_numpy(np.int64)[rows], "id": keys["id"].to_numpy()[rows]})
    for j, col in enumerate(C.INPUTS):
        cache[col] = mean_values[:, j]
    return summary_rows(C, outcome, len(rows), values, names), level_rows(C, outcome, Xs, mean_values), cache


def main():
    t0 = time.time()
    import sklearn.calibration  # noqa: F401  (scikit-learn before the boosting libraries, as in recipe.py)
    import cohort as C

    C.remove_files([os.path.join(runlog.HERE, name) for name in TABLES] + [shap_path(o) for o in C.OUTCOMES])
    assert AGE_EDGES[0] == C.ADULT_AGE, "the first age band does not start at the adult age"
    summary = C.read_table("train_summary.csv")
    perf = C.read_table("performance.csv")
    say(f"SHAP sample: up to {C.SHAP_N:,} {DATASET} persons per outcome (seed {C.SEED})")

    parts = {name: [] for name in TABLES}
    for outcome in C.OUTCOMES:
        t1 = time.time()
        model = ensemble(C, outcome, summary)
        X, keys, p = persons(C, perf, outcome)
        say(f"{outcome}: ensemble agrees with train_summary.csv; {DATASET} persons {count_text(len(X))} agree with "
            f"cohort_flow.csv and the s03 predictions ({time.time() - t1:.0f} s)")
        t1 = time.time()
        rows, by_level, cache = outcome_shap(C, model, outcome, X, keys, p, C.SHAP_N)
        cache.to_parquet(shap_path(outcome), index=False)
        parts["shap_summary.csv"].append(pd.DataFrame(rows, columns=SUMMARY_COLS))
        parts["shap_by_level.csv"].append(pd.DataFrame(by_level, columns=LEVEL_COLS))
        say(f"{outcome}: TreeSHAP of {rows[0]['n_learners']:,} learners on {count_text(len(cache))} persons; "
            f"each learner's values add up to its log-odds margin ({time.time() - t1:.0f} s)")

    for name, (cols, counts) in TABLES.items():
        runlog.write_table(pd.concat(parts[name], ignore_index=True)[cols], name, counts)
    say("tables written: " + ", ".join(TABLES))
    say(f"s07 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
