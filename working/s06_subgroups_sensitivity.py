"""s06 - the ensemble in subgroups and in two sensitivity analyses (run by the data holder).

Reads the ensemble predictions of s03 and, through cohort.outcome_frame, the persons of cache/dev.parquet and
cache/knhanes.parquet. The predictions of each dataset are checked against its persons and labels, its counts in
cohort_flow.csv and their hashes in performance.csv. Writes two aggregate tables:

  subgroups.csv     KN in all, by age band, by sex, by survey period and by alcohol-drinking group (with its unknown
                    level): persons, events, prevalence, mean prediction, AUROC and O:E with 95% CIs from resamples
                    drawn within the subgroup
  sensitivity.csv   CHECKUP_DAY: DEV_TEST persons whose urine test fell on a health-checkup day, and those tested on
                    another day; WEIGHTED: KN without and with the survey weight (wt_itvex divided by the number of
                    pooled survey years), with 95% CIs from record resamples that keep each record's weight

Groups with fewer than 10 events or non-events get counts only, with events blank when there are 10 or more of them
and the non-events are 1-9 (n - events would give that count). The two groups of the DEV_TEST split by check-up day
get counts only together, with events blank, when either has 1-9 persons, events or non-events, since the dataset
totals of performance.csv would give that count by difference. The weighted intervals ignore the survey strata and
clusters. The All rows of subgroups.csv repeat performance.csv and calibration.csv (asserted). These rows are
descriptive and carry no P values.

Run from the working folder:
    s=s06_subgroups_sensitivity; python $s.py > logs/$s.out 2> cache/$s.err
"""
import os
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

MODEL = "ENS"
KN_SETS = ("KN",)
MIN_CLASS = 10              # fewest events and non-events of a group with metrics
TOL = 1e-9                  # All rows against the s03 tables (same resamples; only the summation order differs)
SMALL_NOTE = "fewer than 10 events or non-events"
PAIR_NOTE = "counts only: one of the two groups has 1-9 persons, events or non-events"
UNWEIGHTED_NOTE = "unweighted, as in the main analyses; 95% CIs from record resamples"
SUB_COLS = ["outcome", "dataset", "model", "subgroup_var", "subgroup", "n", "events", "prevalence", "mean_pred",
            "auroc", "auroc_lo", "auroc_hi", "oe", "oe_lo", "oe_hi", "n_boot", "note"]
SENS_COLS = ["analysis", "outcome", "dataset", "model", "subgroup", "weighted", "n", "events", "prevalence",
             "prevalence_lo", "prevalence_hi", "mean_pred", "auroc", "auroc_lo", "auroc_hi", "oe", "oe_lo", "oe_hi",
             "n_boot", "note"]
TABLES = {"subgroups.csv": (SUB_COLS, ["n", "events"]), "sensitivity.csv": (SENS_COLS, ["n", "events"])}
CHECKUP_GROUPS = (("Checkup day", True), ("Other day", False))
SEXES = ("Men", "Women")
REPEATS = (("performance.csv", ("prevalence", "mean_pred", "auroc", "auroc_lo", "auroc_hi")),
           ("calibration.csv", ("oe", "oe_lo", "oe_hi")))


def ensemble_row(table, name, outcome, dataset):
    """The one ENS row of an outcome and dataset in an s03 table."""
    row = table[(table["outcome"] == outcome) & (table["dataset"] == dataset) & (table["model"] == MODEL)]
    assert len(row) == 1, f"{name} has {len(row):,} ensemble rows for one outcome and dataset"
    return row.iloc[0]


def dataset_frame(C, perf, outcome, dataset):
    """(y, p, keys) of one outcome and dataset, once the s03 predictions agree with outcome_frame, cohort_flow.csv and
    performance.csv."""
    X, y, keys = C.checked_frame(outcome, dataset)
    bad = int((~X["male"].isin([0.0, 1.0])).sum())
    assert bad == 0, f"{outcome} {dataset}: {count_text(bad)} sex values are not 0 or 1"
    p, _ = C.checked_preds(MODEL, outcome, dataset, keys, y, C.recorded_sha(MODEL, outcome, dataset, perf))
    return y.to_numpy(float), p, keys


def group_stats(common, B, seed, y, p, w=None, prevalence_ci=False):
    """Counts, prevalence, mean prediction, AUROC and O:E with 95% CIs of one group; weights w, when given, stay with
    their records in every resample. Counts only when the group has fewer than MIN_CLASS events or non-events."""
    n, events = len(y), int(y.sum())
    if min(events, n - events) < MIN_CLASS:
        return dict(n=n, events=events, note=SMALL_NOTE)
    wv = np.ones(n) if w is None else w
    prev = common.weighted_prevalence(y, wv, B if prevalence_ci else 0, seed)
    auc = common.auroc_ci(y, p, B, seed) if w is None else common.weighted_auroc(y, p, wv, B, seed)
    oe = common.weighted_oe(y, p, wv, B, seed)
    out = dict(n=n, events=events, mean_pred=float(p.mean()) if w is None else float(wv @ p / wv.sum()), n_boot=B,
               note="")
    for key, (est, lo, hi) in (("prevalence", prev), ("auroc", auc), ("oe", oe)):
        out.update({key: est, f"{key}_lo": lo, f"{key}_hi": hi})
    return out


def subgroup_masks(C, keys):
    """[(subgroup_var, subgroup, mask)]: everyone, then the age bands, the sexes, the survey periods and the
    alcohol-drinking groups (the unknown level included)."""
    out = [("all", "All", np.ones(len(keys), bool))]
    for var, levels in (("age_band", C.AGE_BANDS), ("sex", SEXES), ("period", C.KN_PERIODS),
                        ("drink_group", C.DRINK_LEVELS)):
        v = keys[var].to_numpy(object)
        masks = [(var, lev, v == lev) for lev in levels]
        left = len(v) - sum(int(m.sum()) for _, _, m in masks)
        assert left == 0, f"{count_text(left)} persons outside the {var} groups"
        out += masks
    return out


def checkup_flags(keys):
    """Checkup-day flag of each DEV_TEST person (checkup date on the urine-test day)."""
    c = keys["checkup_day"]
    miss = int(pd.isna(c).sum())
    assert miss == 0, f"{count_text(miss)} DEV_TEST persons without a checkup-day flag"
    return c.to_numpy().astype(bool)


def guard_pair(rows):
    """Both groups of a split of a dataset as counts only, events blank, when either has 1-9 persons, events or
    non-events; the other n is blank too when an n is 1-9."""
    small = [[0 < k < runlog.SMALL for k in (r["n"], r["events"], r["n"] - r["events"])] for r in rows]
    if not any(map(any, small)):
        return rows
    n_small = any(s[0] for s in small)
    return [dict(n=r["n"] if s[0] or not n_small else None, events=None, note=PAIR_NOTE) for r, s in zip(rows, small)]


def pooled_weights(keys, years):
    """(wt_itvex / years, mask of records with a finite positive weight)."""
    w = pd.to_numeric(keys["wt_itvex"], errors="coerce").to_numpy(float) / years
    return w, np.isfinite(w) & (w > 0)


def weighted_note(years, left_out):
    text = (f"survey weight wt_itvex / {years} ({years} survey years pooled); 95% CIs from record resamples that "
            f"keep each record's weight, ignoring strata (kstrata) and clusters (psu), so they are approximate")
    return text + ("; records without a positive weight left out" if left_out else "")


def differs_from_s03(s03, outcome, dataset, r):
    """Number of All-row values that differ from performance.csv and calibration.csv."""
    off = 0
    for name, cols in REPEATS:
        src = ensemble_row(s03[name], name, outcome, dataset)
        off += sum(not np.isclose(float(src[c]), r[c], rtol=0, atol=TOL) for c in cols)
    return off


def output_table(rows, cols):
    """The rows as one table; n_boot stays a whole number (blank where a row has counts only), and events is blank
    where it is 10 or more and the non-events are 1-9 (n - events would give them)."""
    df = pd.DataFrame(rows, columns=cols)
    df["n_boot"] = df["n_boot"].astype("Int64")
    n, events = (pd.to_numeric(df[c]) for c in ("n", "events"))
    df["events"] = events.where(~((events >= runlog.SMALL) & (n - events).between(1, runlog.SMALL - 1)))
    return df


def main():
    t0 = time.time()
    import paths_local as P
    import common
    import cohort as C

    C.remove_files([os.path.join(runlog.HERE, name) for name in TABLES])
    B = common.n_boot(P.SYNTHETIC)
    s03 = {name: C.read_table(name) for name, _ in REPEATS}
    perf = s03["performance.csv"]
    years = len(P.KN_YEARS)
    say(f"bootstrap replicates: {B:,}")

    sub, checkup, weighted = [], [], []
    for outcome in C.OUTCOMES:
        t1 = time.time()
        y, p, keys = dataset_frame(C, perf, outcome, "DEV_TEST")
        flag = checkup_flags(keys)
        pair = guard_pair([group_stats(common, B, C.SEED, y[flag == v], p[flag == v], prevalence_ci=True)
                           for _, v in CHECKUP_GROUPS])
        for (label, _), r in zip(CHECKUP_GROUPS, pair):
            checkup.append({"analysis": "CHECKUP_DAY", "outcome": outcome, "dataset": "DEV_TEST", "model": MODEL,
                            "subgroup": label, "weighted": False, **r})
        small = sum(r["note"] != "" for r in pair)
        say(f"{outcome} DEV_TEST: checkup day and other days, {small:,} of 2 with counts only "
            f"({time.time() - t1:.0f} s)")

        for dataset in KN_SETS:
            t1 = time.time()
            y, p, keys = dataset_frame(C, perf, outcome, dataset)
            key = {"outcome": outcome, "dataset": dataset, "model": MODEL}
            groups = subgroup_masks(C, keys)
            small = 0
            for var, label, m in groups:
                r = group_stats(common, B, C.SEED, y[m], p[m])
                if var == "all" and not r["note"]:
                    off = differs_from_s03(s03, outcome, dataset, r)
                    assert off == 0, (f"{outcome} {dataset}: {off:,} values of the All row differ from performance.csv "
                                      f"or calibration.csv; run s03 again")
                small += r["note"] == SMALL_NOTE
                sub.append({**key, "subgroup_var": var, "subgroup": label, **r})
            say(f"{outcome} {dataset}: persons {count_text(len(y))}, events {count_text(int(y.sum()))}; "
                f"{len(groups):,} groups, {small:,} with counts only ({time.time() - t1:.0f} s)")

            t1 = time.time()
            w, ok = pooled_weights(keys, years)
            base = {"analysis": "WEIGHTED", **key, "subgroup": "All"}
            r = group_stats(common, B, C.SEED, y, p, prevalence_ci=True)
            weighted.append({**base, "weighted": False, **r, "note": r["note"] or UNWEIGHTED_NOTE})
            r = group_stats(common, B, C.SEED, y[ok], p[ok], w[ok], prevalence_ci=True)
            weighted.append({**base, "weighted": True, **r, "note": r["note"] or weighted_note(years, not ok.all())})
            say(f"{outcome} {dataset} weighted: records with a positive weight {count_text(int(ok.sum()))}, without "
                f"{count_text(int((~ok).sum()))} ({time.time() - t1:.0f} s)")

    out = {"subgroups.csv": sub, "sensitivity.csv": checkup + weighted}
    for name, (cols, counts) in TABLES.items():
        runlog.write_table(output_table(out[name], cols), name, counts)
    say("tables written: " + ", ".join(TABLES))
    say(f"s06 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
