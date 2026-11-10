"""s05 - alert thresholds set on the development test persons and applied unchanged to KNHANES (run by the data holder).

Every threshold comes from the DEV_TEST predictions written by s03 (ENS) and s04 (LR_ALL, LR_ASB), separately for each
outcome: for ENS the Youden threshold and the largest thresholds with 80 % and 90 % sensitivity; for ENS, LR_ALL and
LR_ASB the threshold whose DEV_TEST alert rate is nearest to that of the rule (BMI >= 25 kg/m2), the candidates being
the distinct DEV_TEST predictions and ties going to the larger threshold. thresholds.json keeps the thresholds with
the SHA-256 of the prediction files they come from; the files are hashed again, and must agree, before the thresholds
are used on DEV_TEST and KN. Writes these aggregate tables:

  thresholds.json         thresholds, DEV_TEST alert rates (target, achieved, difference) and the prediction hashes
  operating_points.csv    alert rate, sensitivity, specificity, PPV, NPV and alerts per case with 95% CIs, the
                          threshold the same in every resample; DEV_TEST rows marked as the derivation set
  comparators_alert.csv   the rule next to the ENS, LR_ALL and LR_ASB thresholds set to its DEV_TEST alert rate: the
                          same measures, the ROC point, and model-minus-rule differences with paired bootstrap 95%
                          CIs; DEV_TEST rows marked as the derivation set
  decision_curve.csv      net benefit at threshold probabilities 0.002 to 0.098 by 0.002 and 0.10 to 0.50 by 0.01 of
                          ENS, LR_ALL and LR_ASB alerting at p >= pt, of the fixed alerts (the rule, and ENS at its
                          YOUDEN, SENS80 and SENS90 thresholds, the same alert at every pt), of alerting everyone and
                          of alerting no one; all but the last two with pointwise bootstrap 95% limits (one strategy
                          at one pt: not a simultaneous band and not limits of a difference of two, so overlapping
                          bands do not show similar net benefit; net benefit is not compared formally), the alert
                          the same in every resample; the DEV_TEST values of the fixed ENS lines are derivation
                          results

The two outcomes share their persons, so the rule alerts the same persons for both: its alerts, its target alert rate
and the DEV_TEST n are equal between the outcomes, while its events and the model thresholds differ.

Rates are left blank in a row whose confusion counts include a count of 1-9, together with the count that would give
it by difference from events or non-events, and so is a net benefit at such counts (with its limits). In a rule row
with such a count the other confusion counts are blank too, 0 included (the counts of 1-9 are written <10; n and
events stay): the size of the rule's alert group can be read from the other outcome's row of the same dataset, and
with that size any other count shown in the row would give the masked one by difference. A fixed alert with such a
count is blank at every threshold probability, since two of its net benefits would give its TP and FP.

Run from the working folder:
    s=s05_thresholds; python $s.py > logs/$s.out 2> cache/$s.err
"""
import json
import os
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

MODELS = ("ENS", "LR_ALL", "LR_ASB")
RULE_KEYS = ("BMI25",)                      # literal because cohort.py is imported only inside main(); asserted there
FIXED = ("YOUDEN", "SENS80", "SENS90")      # ENS thresholds whose alert decision_curve.csv also holds at every pt
GRID = np.r_[np.arange(1, 50) / 500, np.arange(10, 51) / 100]  # 0.002 to 0.098 by 0.002, 0.10 to 0.50 by 0.01
RATES = ("alert_rate", "se", "sp", "ppv", "npv", "alerts_per_case")
DIFFS = ("alert_rate", "se", "sp", "ppv", "npv")
CELLS = ["tp", "fp", "tn", "fn"]
PARTNERS = (("tp", "fn"), ("fp", "tn"))     # each pair adds up to a total of the row (events, non-events)
RATE_COLS = [f"{k}{s}" for k in RATES for s in ("", "_lo", "_hi")]
DIFF_COLS = [f"d_{k}{s}" for k in DIFFS for s in ("", "_lo", "_hi")]
OP_COLS = (["outcome", "dataset", "model", "threshold_name", "threshold", "derivation", "n", "events"] + RATE_COLS
           + CELLS + ["n_boot"])
CMP_COLS = (["row_type", "outcome", "dataset", "strategy", "rule", "model", "threshold_name", "threshold",
             "derivation", "dev_test_target_rate", "dev_test_achieved_rate", "n", "events"] + RATE_COLS
            + ["fpr", "tpr"] + CELLS + DIFF_COLS + ["n_boot"])
FIXED_COLS = [f"nb_rule_{r.lower()}" for r in RULE_KEYS] + [f"nb_ens_{t.lower()}" for t in FIXED]
DC_COLS = (["outcome", "dataset", "threshold_prob", "n", "events", "prevalence"]
           + [f"{c}{s}" for c in [f"nb_{m.lower()}" for m in MODELS] + FIXED_COLS for s in ("", "_lo", "_hi")]
           + ["nb_all", "nb_none", "n_boot"])
JSON_NAME = "thresholds.json"
TABLES = {"operating_points.csv": (OP_COLS, ["n", "events"] + CELLS),
          "comparators_alert.csv": (CMP_COLS, ["n", "events"] + CELLS),
          "decision_curve.csv": (DC_COLS, ["n", "events"])}


def ratio(a, b):
    return float(a / b) if b > 0 else np.nan


def confusion(y, flag):
    """tp, fp, tn, fn of a 0/1 alert."""
    pos = y == 1
    tp, fp = int(np.count_nonzero(flag & pos)), int(np.count_nonzero(flag & ~pos))
    return dict(tp=tp, fp=fp, tn=int(np.count_nonzero(~flag & ~pos)), fn=int(np.count_nonzero(~flag & pos)))


def small(c):
    """True when one of tp, fp, tn, fn is 1-9."""
    return any(0 < int(c[k]) < runlog.SMALL for k in CELLS)


def blank_rates(row, rule=False):
    """The row with its rates, their CIs and its ROC point blank, and with the partner of a count of 1-9 blank when
    the partner is 10 or more (it would give that count by difference from events or non-events). In a rule row
    every count that is not 1-9 is blank, 0 included: the rule alerts the same persons for both outcomes, so the size
    of its alert group, read from the other outcome's row, and a shown count would give a masked one by difference."""
    out = {**row, **{c: None for c in RATE_COLS + ["fpr", "tpr"]}}
    for a, b in PARTNERS:
        for k, other in ((a, b), (b, a)):
            if 0 < int(row[k]) < runlog.SMALL <= int(row[other]):
                out[other] = None
    if rule:
        out.update({k: None for k in CELLS if not 0 < int(row[k]) < runlog.SMALL})
    return out


def flag_stat(key):
    """fn(y, flag) for common.paired_boot: one operating characteristic of a 0/1 alert."""
    def stat(y, flag):
        pos = y == 1
        tp = np.count_nonzero(flag & pos)
        fp = np.count_nonzero(flag) - tp
        fn = np.count_nonzero(pos) - tp
        tn = len(y) - tp - fp - fn
        return {"alert_rate": ratio(tp + fp, len(y)), "se": ratio(tp, tp + fn), "sp": ratio(tn, tn + fp),
                "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn)}[key]
    return stat


# ------------------------------------------------------------------------------------------------ inputs
def predictions(C, perf, comp, outcome, dataset, y, keys):
    """({model: p}, {model: sha}) of ENS, LR_ALL and LR_ASB, once persons, labels and hashes agree with the frame and
    with performance.csv and model_comparison.csv."""
    p, sha = {}, {}
    for model in MODELS:
        p[model], sha[model] = C.checked_preds(model, outcome, dataset, keys, y,
                                               C.recorded_sha(model, outcome, dataset, perf, comp))
    return p, sha


def load(C, perf, comp, outcome, dataset):
    """Labels, ENS, LR_ALL and LR_ASB predictions, rule alerts and prediction hashes of one outcome and dataset."""
    X, y, keys = C.checked_frame(outcome, dataset)
    p, sha = predictions(C, perf, comp, outcome, dataset, y, keys)
    flags = {rule: C.rule_flag(rule, X).astype(bool) for rule in C.RULES}
    return dict(y=y.to_numpy(float), p=p, flags=flags, sha=sha)


# ------------------------------------------------------------------------------------------------ thresholds
def described(common, y, p, t, with_se):
    """thresholds.json entry of a threshold: its value and DEV_TEST alert rate (and sensitivity)."""
    assert np.isfinite(t) and 0 <= t <= 1, "a threshold is missing or outside 0-1"
    c = common.confusion_at(y, p, t)
    hide = small(c)
    out = dict(threshold=float(t), dev_test_alert_rate=None if hide else c["alert_rate"])
    if with_se:
        out["dev_test_se"] = None if hide else c["se"]
    return out


def derive(common, C, d):
    """thresholds.json entry of one outcome, from its DEV_TEST persons only."""
    y, p, flags = d["y"], d["p"], d["flags"]
    entry = {"derivation": {"dataset": "DEV_TEST", "n": runlog.mask_count(len(y)),
                            "events": runlog.mask_count(int(y.sum())), "preds_sha256": dict(d["sha"])},
             "rules": {}}
    target, rule_small = {}, {}
    for rule, flag in flags.items():
        target[rule], rule_small[rule] = float(flag.mean()), small(confusion(y, flag))
        entry["rules"][rule] = {"dev_test_alert_rate": None if rule_small[rule] else target[rule]}
    for model in MODELS:
        out = {}
        if model == "ENS":
            out["YOUDEN"] = described(common, y, p[model], common.youden_threshold(y, p[model]), False)
            for name, s in C.SENS_TARGETS.items():
                t = common.threshold_for_sensitivity(y, p[model], s)
                out[name] = described(common, y, p[model], t, True)
                assert np.mean(p[model][y == 1] >= t) >= s, f"{name}: sensitivity below its target"
        for rule in C.RULES:
            t, achieved = common.alert_match_threshold(p[model], target[rule])
            c = common.confusion_at(y, p[model], t)
            assert abs(c["alert_rate"] - achieved) <= 1e-12, "the alert rate of a threshold differs from its search"
            hide = small(c)
            out[f"MATCH_{rule}"] = dict(
                threshold=float(t), target_rate=None if rule_small[rule] else target[rule],
                achieved_rate=None if hide else achieved,
                difference=None if hide or rule_small[rule] else achieved - target[rule])
        entry[model] = out
    return entry


def checked_thresholds(C, path, synthetic, B):
    """thresholds.json, once the DEV_TEST prediction files have the SHA-256 it records."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc.get("synthetic") is bool(synthetic), "thresholds.json and this run differ in SYNTHETIC"
    assert doc.get("n_boot") == B, "thresholds.json and this run differ in bootstrap replicates"
    off = 0
    for outcome in C.OUTCOMES:
        der = doc[outcome]["derivation"]
        assert der["dataset"] == "DEV_TEST", "thresholds.json names another derivation set"
        for model in MODELS:
            off += C.read_preds(model, outcome, "DEV_TEST").attrs["preds_sha256"] != der["preds_sha256"][model]
    assert off == 0, f"{off:,} DEV_TEST prediction file(s) differ from thresholds.json"
    return doc


def thresholds_of(doc, outcome):
    """{(model, threshold name): threshold} of one outcome, in thresholds.json order."""
    return {(model, name): float(v["threshold"]) for model in MODELS for name, v in doc[outcome][model].items()}


# ------------------------------------------------------------------------------------------------ tables
def operating_rows(common, B, seed, outcome, dataset, d, thr):
    """(operating_points.csv rows, results by (model, threshold name)) of one outcome and dataset."""
    y = d["y"]
    rows, res = [], {}
    for (model, name), t in thr.items():
        r = common.operating_point_ci(y, d["p"][model], t, B, seed)
        res[(model, name)] = r
        row = dict(outcome=outcome, dataset=dataset, model=model, threshold_name=name, threshold=t,
                   derivation=dataset == "DEV_TEST", n=len(y), events=int(y.sum()), **r, n_boot=B)
        rows.append(blank_rates(row) if small(r) else row)
    return rows, res


def comparator_rows(common, B, seed, outcome, dataset, d, thr, res, doc):
    """comparators_alert.csv rows of one outcome and dataset: the rule, then ENS, LR_ALL and LR_ASB at its alert rate."""
    y = d["y"]
    base = dict(outcome=outcome, dataset=dataset, derivation=dataset == "DEV_TEST", n=len(y), events=int(y.sum()),
                n_boot=B)
    rows = []
    for rule, flag in d["flags"].items():
        name = f"MATCH_{rule}"
        target = doc[outcome]["rules"][rule]["dev_test_alert_rate"]
        r = common.rule_point_ci(y, flag, B, seed)
        row = dict(base, row_type="rule", strategy=f"RULE_{rule}", rule=rule, dev_test_target_rate=target, **r,
                   fpr=1 - r["sp"], tpr=r["se"])
        rows.append(blank_rates(row, rule=True) if small(r) else row)
        for model in MODELS:
            m, t = res[(model, name)], thr[(model, name)]
            row = dict(base, row_type="model", strategy=f"{model}@{name}", rule=rule, model=model,
                       threshold_name=name, threshold=t, dev_test_target_rate=target,
                       dev_test_achieved_rate=doc[outcome][model][name]["achieved_rate"], **m, fpr=1 - m["sp"],
                       tpr=m["se"])
            if small(m):
                row = blank_rates(row)
            elif not small(r):
                alert = d["p"][model] >= t
                for key in DIFFS:
                    est, lo, hi = common.paired_boot(y, alert, flag, flag_stat(key), B, seed)
                    row.update({f"d_{key}": est, f"d_{key}_lo": lo, f"d_{key}_hi": hi})
            rows.append(row)
    return rows


def alert_counts(y, p, grid):
    """(tp, fp) of the alert p >= pt at each threshold probability pt."""
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    return len(pos) - np.searchsorted(pos, grid, "left"), len(neg) - np.searchsorted(neg, grid, "left")


def curve_rows(common, B, seed, outcome, dataset, d, thr):
    """decision_curve.csv rows of one outcome and dataset: ENS, LR_ALL and LR_ASB alerting at p >= pt, then the fixed
    alerts (the rule, and ENS at its FIXED thresholds in thr, the same alert at every pt), all with bootstrap 95%
    limits (the alert the same in every resample). A net benefit at counts of 1-9 is blank, and so are its limits; a
    fixed alert with such a count is blank at every pt."""
    y = d["y"]
    n, events = len(y), int(y.sum())
    cols, hidden = {}, 0
    for model in MODELS:
        nb = common.net_benefit_ci(y, d["p"][model], GRID, B, seed)
        if model == "ENS":
            nb_all, nb_none = [r["nb_all"] for r in nb], [r["nb_none"] for r in nb]
        tp, fp = alert_counts(y, d["p"][model], GRID)
        cells = np.stack([tp, fp, n - events - fp, events - tp])
        hide = ((cells > 0) & (cells < runlog.SMALL)).any(axis=0)
        for s in ("", "_lo", "_hi"):
            cols[f"nb_{model.lower()}{s}"] = np.where(hide, np.nan, [r[f"nb_model{s}"] for r in nb])
        hidden += int(hide.sum())
    alerts = [d["flags"][rule] for rule in RULE_KEYS] + [d["p"]["ENS"] >= thr[("ENS", name)] for name in FIXED]
    for col, alert in zip(FIXED_COLS, alerts):
        nb = common.fixed_alert_nb_ci(y, alert, GRID, B, seed)
        hide = small(confusion(y, alert))
        for s in ("", "_lo", "_hi"):
            cols[f"{col}{s}"] = np.full(len(GRID), np.nan) if hide else [r[f"nb_rule{s}"] for r in nb]
        hidden += len(GRID) if hide else 0
    out = pd.DataFrame(dict(outcome=outcome, dataset=dataset, threshold_prob=GRID, n=n, events=events,
                            prevalence=events / n, **cols, nb_all=nb_all, nb_none=nb_none, n_boot=B))
    return out[DC_COLS], hidden


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    import paths_local as P
    import common
    import cohort as C

    C.remove_files([os.path.join(runlog.HERE, name) for name in [JSON_NAME, *TABLES]])
    assert tuple(C.RULES) == RULE_KEYS, "the rules differ from the decision_curve.csv columns"
    B = common.n_boot(P.SYNTHETIC)
    perf, comp = C.read_table("performance.csv"), C.read_table("model_comparison.csv")
    say(f"bootstrap replicates: {B:,}")

    dev, doc = {}, {"synthetic": bool(P.SYNTHETIC), "n_boot": B}
    for outcome in C.OUTCOMES:
        t1 = time.time()
        dev[outcome] = load(C, perf, comp, outcome, "DEV_TEST")
        doc[outcome] = derive(common, C, dev[outcome])
        y = dev[outcome]["y"]
        say(f"{outcome} DEV_TEST: persons {count_text(len(y))}, events {count_text(int(y.sum()))}; "
            f"{len(thresholds_of(doc, outcome)):,} thresholds from the ENS, LR_ALL and LR_ASB predictions "
            f"({time.time() - t1:.0f} s)")
    path = os.path.join(runlog.HERE, JSON_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, allow_nan=False)
        fh.write("\n")
    derived = {o: thresholds_of(doc, o) for o in C.OUTCOMES}

    doc = checked_thresholds(C, path, P.SYNTHETIC, B)
    assert all(thresholds_of(doc, o) == derived[o] for o in C.OUTCOMES), "thresholds.json reads back different"
    assert all(dev[o]["sha"] == doc[o]["derivation"]["preds_sha256"] for o in C.OUTCOMES), \
        "the DEV_TEST predictions in memory differ from thresholds.json"
    say(f"thresholds.json written and read back: the {len(C.OUTCOMES) * len(MODELS):,} DEV_TEST prediction files "
        f"agree with its SHA-256 values; the thresholds are applied unchanged to every dataset")

    op, cmp_, curves = [], [], []
    blank_nb = 0
    for outcome in C.OUTCOMES:
        thr = thresholds_of(doc, outcome)
        for dataset in C.PRED_DATASETS:
            t1 = time.time()
            d = dev[outcome] if dataset == "DEV_TEST" else load(C, perf, comp, outcome, dataset)
            rows, res = operating_rows(common, B, C.SEED, outcome, dataset, d, thr)
            op += rows
            cmp_ += comparator_rows(common, B, C.SEED, outcome, dataset, d, thr, res, doc)
            curve, hidden = curve_rows(common, B, C.SEED, outcome, dataset, d, thr)
            curves.append(curve)
            blank_nb += hidden
            say(f"{outcome} {dataset}: persons {count_text(len(d['y']))}, events {count_text(int(d['y'].sum()))} "
                f"({time.time() - t1:.0f} s)")

    out = {"operating_points.csv": pd.DataFrame(op, columns=OP_COLS),
           "comparators_alert.csv": pd.DataFrame(cmp_, columns=CMP_COLS),
           "decision_curve.csv": pd.concat(curves, ignore_index=True)}
    for name, (cols, counts) in TABLES.items():
        runlog.write_table(out[name][cols], name, counts)
    blank_rows = sum(r["se"] is None for r in op + cmp_)
    say(f"rows with a confusion count of 1-9, rates left blank: {blank_rows:,}; net benefits left blank: {blank_nb:,}")
    say("tables written: " + ", ".join([JSON_NAME, *TABLES]))
    say(f"s05 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
