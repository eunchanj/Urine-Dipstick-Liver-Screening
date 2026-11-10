"""s02 - tune and fit the calibrated ensemble of each outcome on the training persons (run by the data holder).

Reads the training persons of cache/dev.parquet, through cohort.train_xy, and cohort_flow.csv. The KNHANES cache and
the test persons are not read, and any further data read stops the run. The two outcomes are fitted separately on the
same training persons in the same order: their rows of train_summary.csv share the SHA-256 of the inputs and of the
ids and differ in the labels (both asserted before each fit). Writes to MODEL_DIR:

  ens_<OUTCOME>.joblib            the fitted model of recipe.py (isotonic calibration of a soft vote of 3 learners)
  optuna_<OUTCOME>_<learner>.csv  Optuna trials: number, value, parameters
  ens_config.json                 inputs, recipe, counts, SHA-256 of inputs, labels, ids and code, package versions

and two aggregate tables to the working folder:

  hyperparameters.csv   tuned and set parameters of each learner, with the searched ranges
  train_summary.csv     recipe, training persons and events, SHA-256 values and package versions

No prediction or performance measure is computed here. Run from the working folder:
    python s02_train.py > logs/s02_train.out 2> cache/s02_train.err
"""
import ast
import importlib.metadata as md
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

N_TRIALS = 25
OUTPUTS = ("hyperparameters.csv", "train_summary.csv")
CONFIG = "ens_config.json"
HYPER_COLS = ["outcome", "learner", "param", "value", "tuned", "search"]
SUMMARY_COLS = ["outcome", "n_train", "events_train", "prevalence_train", "learners", "trials_per_learner",
                "tuning_folds", "seed", "sampler", "voting", "calibration", "calibration_folds", "class_weighting",
                "cv_auroc_lgb", "cv_auroc_xgb", "cv_auroc_cat", "calibration_fold_events_min", "fit_seconds",
                "model_file", "sha256_model", "sha256_inputs", "sha256_labels", "sha256_ids", "sha256_cohort_py",
                "sha256_recipe_py", "sha256_s02", "versions", "synthetic"]
SUMMARY_COUNTS = ["n_train", "events_train", "calibration_fold_events_min"]
SAME_PERSONS = ("sha256_inputs", "sha256_ids")       # equal in the train_summary.csv rows of the two outcomes
# names that reach the survey data: the code of this file uses none of them (survey_names_used)
SURVEY_NAMES = ("KN_CACHE", "KN_FILES", "KN_CODE_TABLE", "KN_YEARS", "kn_cohort", "kn_names", "read_knhanes_raw")
CODE = {"sha256_cohort_py": "cohort.py", "sha256_recipe_py": "recipe.py", "sha256_s02": "s02_train.py"}
PACKAGES = ("lightgbm", "xgboost", "catboost", "optuna", "scikit-learn")
MORE_PACKAGES = ("numpy", "pandas", "pyarrow", "joblib")
WEIGHTS = ("class_weight", "class_weights", "auto_class_weights", "scale_pos_weight")
UNKEPT = {("cat", "thread_count"): -1}


def plain(v):
    """A JSON value for numpy scalars."""
    return v.item() if isinstance(v, np.generic) else str(v)


def value_text(v):
    """A parameter value as text: '200', '0.1', 'False', 'logloss'."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def search_text(dist):
    """Searched range of a tuned parameter: 'int 100-300 step 50', 'float 0.02-0.1 log'."""
    import optuna
    if isinstance(dist, optuna.distributions.IntDistribution):
        text = f"int {dist.low}-{dist.high}" + (f" step {dist.step}" if dist.step != 1 else "")
    else:
        text = f"float {dist.low!r}-{dist.high!r}" + (f" step {dist.step!r}" if dist.step else "")
    return text + (" log" if dist.log else "")


def survey_names_used(source):
    """The names of SURVEY_NAMES that this Python source uses in its code (comments and text do not count)."""
    nodes = list(ast.walk(ast.parse(source)))
    used = {n.id for n in nodes if isinstance(n, ast.Name)} | {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    return sorted(used & set(SURVEY_NAMES))


def check_same_persons(rows, hashes):
    """Stop unless these training hashes equal those of the outcomes already fitted: the two outcomes have the same
    persons and inputs and differ only in their labels."""
    off = sum(hashes[k] != r[k] for r in rows for k in SAME_PERSONS)
    assert off == 0, f"{off:,} training input or id hash(es) differ between the outcomes"


def training_cache(C):
    """Hand cohort.py the DEV_TRAIN rows of cache/dev.parquet; any later data read raises."""
    dev = pd.read_parquet(C.DEV_CACHE, filters=[("role", "==", "DEV_TRAIN")])
    other = int((dev["role"].astype(str) != "DEV_TRAIN").sum())
    assert other == 0, f"{count_text(other)} cache rows outside the training persons"
    C.read_cache = lambda: (dev, None)

    def refused(*args, **kwargs):
        raise AssertionError("s02 reads no data beyond the training persons")

    import pyarrow.parquet as pq
    pd.read_parquet = refused
    pq.read_table = refused
    sys.modules["pyreadstat"] = None
    return dev


def differs(params, learner, name, value):
    """A learner's parameter differs from the value given; CatBoost does not keep thread_count -1 (its default)."""
    if name not in params:
        return UNKEPT.get((learner, name), object()) != value
    return params[name] != value


def check_model(R, model, info, studies, n_trials):
    """The fitted object is the recipe's: isotonic calibration over 5 folds of the soft-voting ensemble, the learners
    given every recipe parameter, each fold fitted with the tuned values and no class weights; every Optuna study ran
    all its trials."""
    assert type(model).__name__ == "CalibratedClassifierCV" and model.method == R.CALIBRATION, \
        "calibration differs from the recipe"
    assert model.cv == R.CALIBRATION_FOLDS == len(model.calibrated_classifiers_), \
        "calibration folds differ from the recipe"
    assert model.estimator.voting == R.VOTING, "voting differs from the recipe"
    template = dict(model.estimator.estimators)
    assert tuple(template) == R.LEARNERS, "learners differ from the recipe"
    for k in R.LEARNERS:
        off = sum(differs(template[k].get_params(), k, p, v) for p, v in info[k].items())
        assert off == 0, f"{off:,} parameter(s) of a learner differ from the recipe values"
    for cc in model.calibrated_classifiers_:
        fitted = cc.estimator.named_estimators_
        assert tuple(fitted) == R.LEARNERS, "a calibration fold holds other learners"
        for k in R.LEARNERS:
            params = fitted[k].get_params()
            off = sum(differs(params, k, p, v) for p, v in info[k].items())
            assert off == 0, f"{off:,} parameter(s) of a fitted learner differ from the tuned values"
            weighted = sum(params.get(w) is not None for w in WEIGHTS) + bool(params.get("is_unbalance"))
            assert weighted == 0, "a fitted learner has class weights"
    for k in R.LEARNERS:
        st = studies[k]
        done = sum(t.state.name == "COMPLETE" for t in st.trials)
        assert len(st.trials) == done == n_trials, "an Optuna study did not complete its trials"
        assert st.best_value == info["cv_aucs"][k], "best value differs from its study"
        assert type(st.sampler).__name__ == R.SAMPLER + "Sampler", "sampler differs from the recipe"


def hyper_rows(R, outcome, info, studies):
    """hyperparameters.csv rows: every parameter given to a learner, tuned ones with their searched range."""
    rows = []
    for k in R.LEARNERS:
        best = studies[k].best_trial
        for param, value in info[k].items():
            tuned = param in best.params
            rows.append(dict(outcome=outcome, learner=k, param=param, value=value_text(value), tuned=tuned,
                             search=search_text(best.distributions[param]) if tuned else None))
    return rows


def main():
    t0 = time.time()
    import paths_local as P
    import cohort as C
    import recipe as R
    import joblib

    C.check_paths()             # the model folder and the code table of this study, before anything is deleted
    with open(os.path.join(runlog.HERE, CODE["sha256_s02"]), encoding="utf-8") as fh:
        used = survey_names_used(fh.read())
    assert not used, f"{len(used):,} name(s) of the survey data used in s02"
    present = sum(os.path.exists(C.model_path("ENS", o)) for o in C.OUTCOMES)
    assert present == 0, (f"{present:,} fitted ensemble file(s) already in MODEL_DIR; to fit again, first move them "
                          "into MODEL_DIR/old/")
    old = [os.path.join(runlog.HERE, f) for f in OUTPUTS] + [os.path.join(P.MODEL_DIR, f) for f in [CONFIG] + [
        f"optuna_{o}_{k}.csv" for o in C.OUTCOMES for k in R.LEARNERS]]
    for p in old:
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(P.MODEL_DIR, exist_ok=True)
    ignore = os.path.join(P.MODEL_DIR, ".gitignore")
    if not os.path.exists(ignore):
        with open(ignore, "w") as fh:
            fh.write("*\n")

    n_trials = C.n_trials()
    assert P.SYNTHETIC or n_trials == N_TRIALS, "Optuna trials per learner differ from the recipe"
    dev = training_cache(C)
    say(f"training persons read from the development cache: {count_text(len(dev))} ({time.time() - t0:.0f} s)")

    code = {col: C.file_sha256(os.path.join(runlog.HERE, f)) for col, f in CODE.items()}
    versions = {p: md.version(p) for p in PACKAGES}
    config = dict(
        synthetic=bool(P.SYNTHETIC), inputs=list(C.INPUTS),
        recipe=dict(learners=list(R.LEARNERS), trials_per_learner=n_trials, sampler=R.SAMPLER, seed=R.SEED,
                    tuning=f"mean AUROC over StratifiedKFold({R.TUNING_FOLDS}, shuffle=True, random_state={R.SEED})",
                    voting=R.VOTING, calibration=R.CALIBRATION, calibration_folds=R.CALIBRATION_FOLDS,
                    class_weighting=R.CLASS_WEIGHTING,
                    prediction="predict_proba: mean of the calibrated fold ensembles"),
        outcomes={}, code_sha256={f: code[col] for col, f in CODE.items()},
        versions={"python": sys.version.split()[0], **{p: md.version(p) for p in PACKAGES + MORE_PACKAGES}})
    hyper, summary = [], []
    for outcome, o in C.OUTCOMES.items():
        t1 = time.time()
        X, y = C.train_xy(outcome)
        X_frame, y_frame, keys = C.outcome_frame(outcome, "DEV_TRAIN")
        assert X.equals(X_frame) and y.equals(y_frame), "train_xy differs from the training outcome frame"
        n, events = len(y), int(y.sum())
        assert list(X.columns) == list(C.INPUTS), "input columns differ from INPUTS"
        missing = int(X.isna().sum().sum())
        assert missing == 0, f"{count_text(missing)} missing inputs among the training persons"
        bad = int((~y.isin([0, 1])).sum())
        assert bad == 0, f"{count_text(bad)} training labels are not 0 or 1"
        other = int((keys["role"].astype(str) != "DEV_TRAIN").sum())
        assert other == 0, f"{count_text(other)} persons outside the training persons"
        assert 0 < events < n, "training labels of one class only"
        C.check_counts(outcome, "DEV_TRAIN", n, events)
        train = C.train_hashes(X, y, keys)
        check_same_persons(summary, train)
        assert all(train["sha256_labels"] != r["sha256_labels"] for r in summary), \
            "two outcomes have identical training labels"
        say(f"{outcome}: training persons {count_text(n)}, events {count_text(events)}")

        model, info, studies = R.build_tuned_voting_ensemble(X, y, n_trials)
        fit_seconds = round(time.time() - t1)
        check_model(R, model, info, studies, n_trials)
        fold_events = [int(y.iloc[cal].sum()) for _, cal in C.cal_folds(model, X, y)]
        path = C.model_path("ENS", outcome)
        joblib.dump(model, path)
        for k, st in studies.items():
            st.trials_dataframe(attrs=("number", "value", "params")).to_csv(
                os.path.join(P.MODEL_DIR, f"optuna_{outcome}_{k}.csv"), index=False)

        hashes = dict(sha256_model=C.file_sha256(path), **train)
        hyper += hyper_rows(R, outcome, info, studies)
        small = any(0 < v < runlog.SMALL for v in (n, events))
        summary.append(dict(
            outcome=outcome, n_train=n, events_train=events, prevalence_train=None if small else events / n,
            learners=",".join(R.LEARNERS), trials_per_learner=n_trials, tuning_folds=R.TUNING_FOLDS, seed=R.SEED,
            sampler=R.SAMPLER, voting=R.VOTING, calibration=R.CALIBRATION, calibration_folds=R.CALIBRATION_FOLDS,
            class_weighting=R.CLASS_WEIGHTING,
            **{f"cv_auroc_{k}": info["cv_aucs"][k] for k in R.LEARNERS},
            calibration_fold_events_min=min(fold_events), fit_seconds=fit_seconds,
            model_file=os.path.basename(path), **hashes, **code,
            versions=";".join(f"{p}={v}" for p, v in versions.items()), synthetic=bool(P.SYNTHETIC)))
        config["outcomes"][outcome] = dict(
            threshold=o["threshold"], unit=o["unit"], n_train=n, events_train=events,
            model_file=os.path.basename(path), parameters={k: info[k] for k in R.LEARNERS},
            search={k: {p: search_text(d) for p, d in studies[k].best_trial.distributions.items()}
                    for k in R.LEARNERS},
            cv_auroc=info["cv_aucs"], calibration_fold_events=fold_events, fit_seconds=fit_seconds, **hashes)
        with open(os.path.join(P.MODEL_DIR, CONFIG), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=1, default=plain)
        say(f"{outcome}: ensemble saved ({time.time() - t1:.0f} s)")

    runlog.write_table(pd.DataFrame(hyper, columns=HYPER_COLS), "hyperparameters.csv")
    runlog.write_table(pd.DataFrame(summary, columns=SUMMARY_COLS), "train_summary.csv", SUMMARY_COUNTS)
    say("tables written: hyperparameters.csv, train_summary.csv")
    say(f"s02 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
