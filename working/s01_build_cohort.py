"""s01 - development cohort, eligible KNHANES adults, the person-level split and the cohort tables
(run by the data holder).

Writes the person-level caches cache/dev.parquet and cache/knhanes.parquet, and these aggregate tables:

  cohort_flow.csv                flow steps, exclusion reasons and analysis populations (n, events)
  cohort_checks.csv              consistency counts, rule counts, unit medians and the split
  baseline_characteristics.csv   persons per sample and dataset, inputs and outcomes in stored units
  input_levels.csv               levels of the inputs, with the KNHANES codes behind each level
  knhanes_coding.csv             KNHANES codes per year and variable, and the value each one becomes
  enzyme_by_year.csv             ALT and AST per year (hospital data: and check-up day): persons, events, share above
                                 40 U/L and median

When a check stops the run, cohort_checks.csv (and knhanes_coding.csv, once the survey files are read) are written
up to that point, with a last row saying so. Command, in the working folder:
    python s01_build_cohort.py > logs/s01_build_cohort.out 2> cache/s01_build_cohort.err
"""
import os
import time

import numpy as np
import pandas as pd

import runlog
from runlog import count_text, say

OUTPUTS = ("cohort_flow.csv", "cohort_checks.csv", "baseline_characteristics.csv", "input_levels.csv",
           "knhanes_coding.csv", "enzyme_by_year.csv")
FLOW_COLS = ["source", "sample", "order", "key", "parent", "kind", "label", "n", "excluded", "events"]
CHECK_COLS = ["source", "item", "n", "value", "note"]
BASE_COLS = ["sample", "dataset", "variable", "level", "n", "pct", "mean", "sd", "median", "q1", "q3", "unit"]
LEVEL_COLS = ["source", "input", "level", "n", "pct", "kn_codes"]
CODING_COLS = ["section", "year", "variable", "target", "code", "maps_to", "n_records", "n_adults"]
ENZYME_COLS = ["source", "enzyme", "year", "checkup_day", "n", "events", "prop_above_40", "median"]
BANDS = ["not above 40 U/L", "above 40 U/L"]
STATS = ("mean", "sd", "median", "q1", "q3")
# levels that the descriptive tables (baseline_characteristics.csv, input_levels.csv) count together, before masking:
# level text -> the level whose row holds it (the display reads these rows through display.COUNTED_TOGETHER); the model
# inputs and the caches keep every level
POOLED_LEVELS = {"ua_urobilinogen": {"0": "0.5", "3": "2"}}


def masked_pct(df):
    """pct blank where the row's n is 1-9."""
    small = df["n"].between(1, runlog.SMALL - 1)
    df.loc[small, "pct"] = np.nan
    return df


def write_checks(rows):
    runlog.write_table(pd.DataFrame(rows, columns=CHECK_COLS), "cohort_checks.csv", ["n"])


def write_coding(rows):
    """knhanes_coding.csv; n_records is left blank where n_adults is 10 or more and the records of other ages,
    n_records - n_adults, are 1-9 (they would be a count read by difference), so that n_adults stays complete."""
    t = pd.DataFrame(rows, columns=CODING_COLS).astype({"n_records": float, "n_adults": float})
    rest = t["n_records"] - t["n_adults"]
    t.loc[(t["n_adults"] >= runlog.SMALL) & rest.between(1, runlog.SMALL - 1), "n_records"] = np.nan
    runlog.write_table(t, "knhanes_coding.csv", ["n_records", "n_adults"])


def check_row(ck, source, item, n=None, value=None, note="logged"):
    """Append one row of cohort_checks.csv."""
    ck.append(dict(source=source, item=item, n=None if n is None else int(n), value=value, note=note))


def enzyme_labels(C, d, outcome):
    """0/1 labels of one enzyme for every row of a cohort frame (each row has a value and can be labelled)."""
    o = C.OUTCOMES[outcome]
    return C.outcome_label(d[o["value"]], d[o["cmp"]], o["threshold"])


# ------------------------------------------------------------------------------------------------ split
def split_checks(C, dev, ck):
    """SPLIT rows; asserts disjoint roles, that every person has both labels, and the stratification of the split."""
    role = dev["role"].to_numpy(object)
    tr, te = role == "DEV_TRAIN", role == "DEV_TEST"
    assert (tr | te).all(), "development cohort persons without a role"
    both = len(set(dev["id"].to_numpy()[tr]) & set(dev["id"].to_numpy()[te]))
    ck += [dict(source="SPLIT", item="development cohort persons", n=len(dev), value=None, note="logged"),
           dict(source="SPLIT", item="training persons", n=int(tr.sum()), value=None, note="logged"),
           dict(source="SPLIT", item="test persons", n=int(te.sum()), value=None, note="logged"),
           dict(source="SPLIT", item="persons in both training and test", n=both, value=None, note="asserted 0")]
    assert both == 0, f"{count_text(both)} persons in both training and test"
    lacking = int((~(dev["in_alt"].to_numpy(bool) & dev["in_ast"].to_numpy(bool))).sum())
    ck.append(dict(source="SPLIT", item="development cohort persons without both an ALT and an AST label", n=lacking,
                   value=None, note="asserted 0"))
    assert lacking == 0, f"{count_text(lacking)} development cohort persons without both an ALT and an AST label"
    st = dev["stratum"].to_numpy(int)
    worst = 0.0
    for k in range(4):
        text = f"ALT {BANDS[k // 2]}, AST {BANDS[k % 2]}"
        nk = int((st == k).sum())
        for name, m in (("training", tr), ("test", te)):
            ck.append(dict(source="SPLIT", item=f"{name} persons in stratum {k}", n=int((m & (st == k)).sum()),
                           value=text, note="logged"))
        if nk:
            worst = max(worst, abs(int((te & (st == k)).sum()) - nk * te.sum() / len(dev)))
    ck.append(dict(source="SPLIT", item="largest gap between a stratum's test persons and its share (persons)", n=None,
                   value=round(worst, 2), note="asserted below 1"))
    assert worst < 1, "a stratum's test persons differ from its share by one person or more"
    off = int((st != 2 * enzyme_labels(C, dev, "ALT") + enzyme_labels(C, dev, "AST")).sum())
    ck.append(dict(source="SPLIT", item="persons whose stratum disagrees with the ALT and AST labels", n=off,
                   value=None, note="asserted 0"))
    assert off == 0, f"{count_text(off)} persons whose stratum disagrees with the ALT and AST labels"


# ------------------------------------------------------------------------------------------------ cohort details
def kn_availability(kn_flow, ck):
    """KN rows of adults at kn_anthro with both enzymes and with neither (kn_cohort writes the rows with HE_alt only and
    HE_ast only)."""
    n = {r["key"]: r["n"] for r in kn_flow}
    check_row(ck, "KN", "adults at kn_anthro with both HE_alt and HE_ast", n["kn_enzymes"])
    check_row(ck, "KN", "adults at kn_anthro with neither HE_alt nor HE_ast", n["kn_no_alt_ast"])


def crosstab_rows(C, dev, kn, ck):
    """Persons by ALT label and AST label, in the development cohort and among the eligible KNHANES adults."""
    words = ("not above 40", "above 40")
    for source, who, d in (("DEV", "development cohort persons", dev), ("KN", "eligible adults", kn)):
        alt, ast = enzyme_labels(C, d, "ALT"), enzyme_labels(C, d, "AST")
        for i, alt_word in enumerate(words):
            for j, ast_word in enumerate(words):
                check_row(ck, source, f"{who} with ALT {alt_word} and AST {ast_word}",
                          int(((alt == i) & (ast == j)).sum()))


def checkup_rows(C, dev, ck):
    """Development cohort persons per urine-test period and those of them whose urine test fell on a check-up day."""
    period = C.urine_period(dev["urine_year"].to_numpy(int))
    checkup = dev["checkup_day"].to_numpy(bool)
    for p in C.URINE_PERIODS:
        m = period == p
        check_row(ck, "DEV", f"development cohort persons with a urine test in {p}", int(m.sum()))
        check_row(ck, "DEV", f"development cohort persons with a urine test in {p} on a check-up day",
                  int((m & checkup).sum()))


def bmi_record_rows(C, dev, ck):
    """BMI gap bands (1-7 and 8-30 days; dev_cohort logs the urine day and the 1-30 days) and nursing-conflict counts of
    the development cohort. The BMI join is repeated for the cohort persons on their urine days and must give the BMI
    already held in the cohort frame."""
    import pyarrow.parquet as pq
    ids = dev["id"].to_numpy(np.int64)
    proc = pd.read_csv(C.P.DATA_PATH, usecols=[C.COL_ID, C.COL_DAY], low_memory=False, encoding="utf-8-sig")
    proc["id"] = pd.to_numeric(proc[C.COL_ID], errors="coerce").astype(np.int64)
    one = proc.drop_duplicates("id", keep="first")          # the first row per person is the record used in the cohort
    one = one.assign(day=C._day(one[C.COL_DAY]))
    j = dev[["id"]].merge(one[["id", "day"]], on="id", how="left", validate="one_to_one")
    days = j["day"].to_numpy()
    bmi, lag, _ = C.bmi_as_of(ids, days, C._read_anthro(ids, []))
    differ = int((~np.isclose(bmi, dev["bmi"].to_numpy(float))).sum())
    check_row(ck, "DEV", "development cohort persons whose BMI differs when the join is repeated", differ,
              note="asserted 0")
    assert differ == 0, f"{count_text(differ)} development cohort persons whose BMI differs when the join is repeated"
    for text, m in (("1-7 days", (lag >= 1) & (lag <= 7)), ("8-30 days", (lag >= 8) & (lag <= 30))):
        check_row(ck, "DEV", f"development cohort persons with a BMI {text} before the urine day", int(m.sum()))
    if "nursing_conflict" not in pq.read_schema(C.P.ANTHRO_DAY).names:
        check_row(ck, "DEV", "nursing_conflict column not in the anthropometry table", None, "absent")
        return
    a = pq.read_table(C.P.ANTHRO_DAY, columns=["id", "date", "nursing_conflict"]).to_pandas(date_as_object=False)
    a["id"] = a["id"].astype(np.int64)
    a["date"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    a = a[a["id"].isin(ids)]
    conflict = a.assign(conflict=a["nursing_conflict"].astype("boolean").fillna(False).astype(bool))
    conflict = conflict.groupby(["id", "date"], as_index=False)["conflict"].max()   # any row of the person-day
    record = pd.DataFrame({"id": ids, "date": pd.Series(days) - pd.to_timedelta(lag, unit="D")})
    record = record.merge(conflict, on=["id", "date"], how="left", validate="many_to_one")
    check_row(ck, "DEV", "development cohort persons whose BMI record has a nursing-conflict flag",
              int(record["conflict"].eq(True).sum()))


def rule_rows(C, dev, kn, kn_flow, ck):
    """RULE rows of the data rules whose counts are not made inside dev_cohort or kn_cohort."""
    def rule(item, n, value):
        check_row(ck, "RULE", item, n, value, "rule")

    for name, o in C.OUTCOMES.items():
        cmp_text = dev[o["cmp"]].astype("string").fillna("=").to_numpy(object)
        rule(f"development cohort persons whose {name} is censored and labelled from its comparator",
             int((cmp_text != "=").sum()), "labelled from the comparator")
    ranges = (f"height {C.HEIGHT_RANGE[0]:g}-{C.HEIGHT_RANGE[1]:g} cm, weight {C.WEIGHT_RANGE[0]:g}-"
              f"{C.WEIGHT_RANGE[1]:g} kg, BMI {C.BMI_RANGE[0]:g}-{C.BMI_RANGE[1]:g} kg/m2")
    n = {r["key"]: r["n"] for r in kn_flow}
    rule("adults at kn_inputs with height, weight or BMI outside the ranges (excluded)", n["kn_anthro_range"], ranges)
    pregnant = next(r["n"] for r in ck if r["item"] == "eligible adults with pregnancy yes (counted, not excluded)")
    rule("eligible adults with pregnancy yes (pregnancy is not an exclusion)", pregnant, "not excluded")
    rule("eligible adults in 2022-2024 whose ALT and AST are used as published",
         int((kn["year"].to_numpy(int) >= 2022).sum()), "no conversion")
    rule("persons split into training and test within the four strata of ALT above 40 by AST above 40", None,
         "80/20, seed 42")


# ------------------------------------------------------------------------------------------------ baseline
def summary(v):
    v = v[~np.isnan(v)]
    if len(v) < 100:
        return {s: np.nan for s in STATS}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(mean=v.mean(), sd=v.std(ddof=1), median=med, q1=q1, q3=q3)


def as_levels(C, values):
    """Coded values as level text, '(missing)' where NaN."""
    v = pd.Series(np.asarray(values, float))
    lut = {u: C.level_text(u) for u in v.dropna().unique()}
    return v.map(lut).fillna("(missing)").to_numpy(object)


def descriptive_levels(C, c, values):
    """as_levels() of input c, with the levels that the descriptive tables count together (POOLED_LEVELS)."""
    lev, pooled = as_levels(C, values), POOLED_LEVELS.get(c)
    return lev if pooled is None else pd.Series(lev).map(lambda x: pooled.get(x, x)).to_numpy(object)


def ordered(keys, order=None):
    """Levels to write: all of order (zero counts included), then other numeric levels, then '(...)' levels."""
    first = list(order or [])
    rest = [k for k in keys if k not in first]
    num = sorted((k for k in rest if not k.startswith("(")), key=lambda k: (pd.to_numeric(k, errors="coerce"), k))
    return first + num + sorted(k for k in rest if k.startswith("("))


def describe(rows, C, sample, dataset, d, source):
    """Rows of baseline_characteristics.csv for one sample and dataset."""
    n = len(d)

    def add(variable, level, k, stats=None, unit=""):
        rows.append(dict(sample=sample, dataset=dataset, variable=variable, level=level, n=int(k),
                         pct=100.0 * k / n if n else np.nan, **(stats or {s: np.nan for s in STATS}), unit=unit))

    def cont(variable, values, unit, absent=None):
        v = np.asarray(values, float)
        add(variable, "", int((~np.isnan(v)).sum()), summary(v), unit)
        if absent:
            add(variable, absent, int(np.isnan(v).sum()))

    def cat(variable, values, order=None):
        vc = pd.Series(values, dtype=object).fillna("(missing)").value_counts()
        for k in ordered(list(vc.index), order):
            add(variable, k, int(vc.get(k, 0)))

    add("persons", "", n)
    cont("age", d["age"], "years")
    cat("age_band", C.age_band(d["age"].to_numpy(float)), C.AGE_BANDS)
    cat("male", as_levels(C, d["male"]))
    for c in C.DIPSTICK:
        cat(c, descriptive_levels(C, c, d[c]))
    for c in ("ua_pH", "ua_sg"):
        cont(c, d[c], "")
    cont("bmi", d["bmi"], "kg/m2", "(missing)")
    cont("alt", d["alt_value"], "U/L", "(not measured)")
    cont("ast", d["ast_value"], "U/L", "(not measured)")
    labels = {}
    for key, o in (("alt_label", C.OUTCOMES["ALT"]), ("ast_label", C.OUTCOMES["AST"])):
        m = d[o["sample"]].to_numpy(bool)
        lab = np.full(n, "(not measured)", dtype=object)
        y = C.outcome_label(d[o["value"]].to_numpy()[m], d[o["cmp"]].to_numpy(object)[m], o["threshold"])
        lab[m] = np.where(y == 1, "1", "0")
        cat(key, lab, ["0", "1", "(not measured)"])
        labels[key] = lab
    a, b = labels["alt_label"], labels["ast_label"]
    both = np.where((a == "(not measured)") | (b == "(not measured)"), "(not measured)",
                    np.where((a == "1") & (b == "1"), "1", "0"))
    cat("both_label", both, ["0", "1", "(not measured)"])
    if source == "DEV":
        cat("checkup_day", np.where(d["checkup_day"].to_numpy(bool), "1", "0"), ["0", "1"])
        cat("urine_period", C.urine_period(d["urine_year"].to_numpy(int)), C.URINE_PERIODS)
    else:
        cat("period", d["period"].to_numpy(object), C.KN_PERIODS)


def e1_masks(C, adults, dev, ck):
    """Masks of the adult frame for the descriptive contrasts: E1A, both enzymes classifiable or not; E1B, among adults
    with both enzymes and complete dipstick and sex, a BMI or none. E1B_BMI must be the development cohort."""
    both = adults["in_alt"].to_numpy(bool) & adults["in_ast"].to_numpy(bool)
    complete = adults[C.DIPSTICK + ["age", "male"]].notna().all(axis=1).to_numpy(bool)
    has_bmi = adults["has_bmi"].to_numpy(bool)
    masks = {"E1A_ELIGIBLE": both, "E1A_NOT": ~both, "E1B_BMI": both & complete & has_bmi,
             "E1B_NO_BMI": both & complete & ~has_bmi}
    off = abs(int(masks["E1B_BMI"].sum()) - len(dev))
    check_row(ck, "DEV", "persons counted differently in the adult frame and the development cohort", off,
              note="asserted 0")
    assert off == 0, f"{count_text(off)} persons counted differently in the adult frame and the development cohort"
    return masks


def baseline(C, dev, kn, adults, masks):
    """Rows of one analysis sample (ANY: datasets DEV, DEV_TRAIN, DEV_TEST and KN) and of the descriptive contrasts
    (samples E1A_ELIGIBLE, E1A_NOT, E1B_BMI, E1B_NO_BMI; dataset DEV)."""
    rows = []
    role = dev["role"].to_numpy(object)
    for dataset, m in (("DEV", np.ones(len(dev), bool)), ("DEV_TRAIN", role == "DEV_TRAIN"),
                       ("DEV_TEST", role == "DEV_TEST")):
        describe(rows, C, "ANY", dataset, dev[m], "DEV")
    describe(rows, C, "ANY", "KN", kn, "KN")
    for sample, m in masks.items():
        describe(rows, C, sample, "DEV", adults[m], "DEV")
    out = masked_pct(pd.DataFrame(rows, columns=BASE_COLS))
    out["pct"] = out["pct"].astype(float).round(2)
    for c in STATS:                     # rounded for display only after any unit conversion (s08)
        out[c] = out[c].astype(float).round(8)
    return out


def input_levels(C, dev, kn):
    rows = []
    for source, d in (("DEV", dev), ("KN", kn)):
        for c in C.DIPSTICK + ["male"]:
            lev = descriptive_levels(C, c, d[c])
            if source == "KN":
                raw = (np.where(d["male"].to_numpy(float) == 1, "1", "2") if c == "male"
                       else C.code_texts(d[C.KN_RAW[c]]).to_numpy(object))
            vc = pd.Series(lev).value_counts()
            for k in ordered(list(vc.index)):
                codes = ""
                if source == "KN":
                    found = pd.unique(raw[lev == k])
                    codes = ",".join(sorted(found, key=lambda x: (pd.to_numeric(x, errors="coerce"), x)))
                rows.append(dict(source=source, input=c, level=k, n=int(vc[k]), pct=100.0 * vc[k] / len(d),
                                 kn_codes=codes))
    out = masked_pct(pd.DataFrame(rows, columns=LEVEL_COLS))
    out["pct"] = out["pct"].round(2)
    return out


def enzyme_by_year(C, dev, kn):
    """Rows of enzyme_by_year.csv: per enzyme and year (development cohort: also per check-up day; KNHANES: eligible
    adults) the persons, events, share above 40 U/L and median of the stored values. Masking: events blank when
    n - events is 1-9, the share blank when n, events or n - events is 1-9, the median only from 100 values."""
    rows = []

    def add(source, enzyme, year, checkup_day, values, labels):
        n, events = len(labels), int(np.sum(labels))
        rows.append(dict(source=source, enzyme=enzyme, year=int(year), checkup_day=checkup_day, n=n, events=events,
                         prop_above_40=events / n, median=float(np.median(values)) if len(values) >= 100 else np.nan))

    for enzyme, o in C.OUTCOMES.items():
        y = enzyme_labels(C, dev, enzyme)
        v = dev[o["value"]].to_numpy(float)
        year = dev["urine_year"].to_numpy(int)
        checkup = dev["checkup_day"].to_numpy(bool)
        for yr in sorted(set(year.tolist())):
            for flag, text in ((False, "0"), (True, "1")):
                m = (year == yr) & (checkup == flag)
                if m.any():
                    add("DEV", enzyme, yr, text, v[m], y[m])
        y = enzyme_labels(C, kn, enzyme)
        v = kn[o["value"]].to_numpy(float)
        year = kn["year"].to_numpy(int)
        for yr in sorted(set(year.tolist())):
            m = year == yr
            add("KN", enzyme, yr, "", v[m], y[m])
    out = pd.DataFrame(rows, columns=ENZYME_COLS).astype({"events": float})
    rest = (out["n"] - out["events"]).between(1, runlog.SMALL - 1)
    small = rest | out["n"].between(1, runlog.SMALL - 1) | out["events"].between(1, runlog.SMALL - 1)
    out.loc[small, "prop_above_40"] = np.nan
    out.loc[rest, "events"] = np.nan
    out["prop_above_40"] = out["prop_above_40"].astype(float).round(4)
    out["median"] = out["median"].astype(float).round(2)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    import cohort as C
    for f in OUTPUTS:
        p = os.path.join(runlog.HERE, f)
        if os.path.exists(p):
            os.remove(p)
    for p in (C.DEV_CACHE, C.KN_CACHE):
        if os.path.exists(p):
            os.remove(p)

    checks, coding = [], []
    try:
        table = C.load_code_table(checks=checks)
        say(f"code table: {count_text(len(table))} rows")
        dev, dev_flow, _, adults = C.dev_cohort(checks)
        dev.insert(1, "role", pd.array(C.person_split(dev), dtype="string"))
        split_checks(C, dev, checks)
        dev_flow += C.dev_final_rows(dev)
        train = int((dev["role"] == "DEV_TRAIN").sum())
        say(f"development cohort: {count_text(len(dev))} persons; training {count_text(train)}, "
            f"test {count_text(len(dev) - train)} ({time.time() - t0:.0f} s)")
        kn, kn_flow, _, _ = C.kn_cohort(table, checks, coding)
        say(f"KNHANES eligible adults: {count_text(len(kn))} ({time.time() - t0:.0f} s)")
        masks = e1_masks(C, adults, dev, checks)
        kn_availability(kn_flow, checks)
        crosstab_rows(C, dev, kn, checks)
        checkup_rows(C, dev, checks)
        bmi_record_rows(C, dev, checks)
        rule_rows(C, dev, kn, kn_flow, checks)
    except Exception:
        checks.append(dict(source="RUN", item="s01 stopped at a failed check; rows above were written before it",
                           n=None, value=None, note="stopped"))
        write_checks(checks)
        if coding:
            write_coding(coding)
        raise

    dev_flow.sort(key=lambda r: r["order"])
    flow = pd.DataFrame(dev_flow + kn_flow, columns=FLOW_COLS)
    runlog.write_table(flow, "cohort_flow.csv", ["n", "excluded", "events"])
    write_checks(checks)
    write_coding(coding)
    base = baseline(C, dev, kn, adults, masks)
    runlog.write_table(base, "baseline_characteristics.csv", ["n"])
    levels = input_levels(C, dev, kn)
    runlog.write_table(levels, "input_levels.csv", ["n"])
    enzymes = enzyme_by_year(C, dev, kn)
    runlog.write_table(enzymes, "enzyme_by_year.csv", ["n", "events"])
    say(f"tables: flow {count_text(len(flow))} rows, checks {count_text(len(checks))}, "
        f"baseline {count_text(len(base))}, input levels {count_text(len(levels))}, coding {count_text(len(coding))}, "
        f"enzyme years {count_text(len(enzymes))}")

    C.write_cache(dev, kn)
    for outcome in C.OUTCOMES:
        for dataset in C.DATASETS:
            X, y, keys = C.outcome_frame(outcome, dataset)
            missing = int(X.isna().sum().sum())
            assert missing == 0, f"{count_text(missing)} missing inputs in an outcome frame"
            assert len(keys) == len(y) == len(X), "outcome frame parts differ in length"
            C.check_counts(outcome, dataset, len(y), int(y.sum()))
            say(f"{outcome} {dataset}: persons {count_text(len(y))}, events {count_text(int(y.sum()))}")
    for dataset in C.DATASETS:          # the two outcomes share their persons and rows; only the labels differ
        xa, _, ka = C.outcome_frame("ALT", dataset)
        xb, _, kb = C.outcome_frame("AST", dataset)
        assert xa.equals(xb) and ka.equals(kb), f"{dataset}: the ALT and AST frames differ in persons or order"
    say(f"s01 done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    runlog.run(main, __file__)
