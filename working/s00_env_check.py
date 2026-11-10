"""s00 - interpreter, packages and data layout (run by the data holder).

Reads file metadata and single columns of the hospital files and the KNHANES survey files. Writes aggregate tables
only: counts of 1-9 are shown as "<10", percentiles are given only for 100 or more values, and no minimum or
maximum is written.

  environment.txt          interpreter and package versions
  knhanes_variables.csv    year, variable, label, file_rows: survey variables for eligibility, outcomes and inputs
  knhanes_values.csv       year, variable, value, n: those variables with few distinct values
  knhanes_numeric.csv      year, variable, n, n_missing, n_nonnumeric, p01, p50, p99: the others
  dev_schema.csv           file, column, dtype: hospital columns the study reads, all anthropometry columns (no values)
  dev_values.csv           file, variable, value, n: selected categorical columns
  dev_numeric.csv          file, variable, n, n_missing, n_nonnumeric, p01, p50, p99: selected numeric columns
  data_checks.csv          source, item, n, note: row and person counts used to plan the cohort, and counts of the
                           survey enzymes at 1,000 U/L or more and at possible special codes

Run from the working folder:
    python s00_env_check.py > logs/s00_env_check.out 2> cache/s00_env_check.err
"""
import importlib.metadata as md
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

import paths_local as P
import runlog
from runlog import say

PACKAGES = ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "pyreadstat",
            "pyarrow", "matplotlib", "cycler", "pillow", "scipy", "joblib"]

# Survey variables: identifiers and design, sex and age, height, weight and BMI, ALT and AST, pregnancy, alcohol
# frequency, and the nine dipstick fields. Variables are also picked by label (Korean keyword: pregnancy).
KN_NAME = re.compile(r"^(id|year|sex|age|psu|kstrata|wt_itvex|wt_tot|HE_alt|HE_ast|HE_ht|HE_wt|HE_BMI|HE_prg|BD1_11|"
                     r"HE_Upro|HE_Uglu|HE_Uket|HE_Ubil|HE_Ubld|HE_Uro|HE_Unitr|HE_Uph|HE_Usg)$", re.I)
KN_LABEL = re.compile("임신")
KN_NO_VALUES = {"id", "psu", "kstrata"}          # listed by name only
# Summarised with percentiles, never value-counted (a list of values would show the largest one).
KN_NUMERIC = {"age", "wt_itvex", "wt_tot", "he_alt", "he_ast", "he_ht", "he_wt", "he_bmi"}
KN_ENZYMES = ("HE_alt", "HE_ast")
KN_SPECIAL_CODES = (888, 999, 8888, 9999)        # values that could be survey special codes rather than U/L
HANGUL = re.compile("[가-힣]")

PROC_WANT = ["일련번호", "성별", "age_at_urine_test", "days_difference", "소변검사일자", "ua_glucose", "ua_protein",
             "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen", "ua_nitrite", "ua_pH", "ua_sg",
             "ALT", "AST"]
PROC_CATEGORICAL = ["성별", "ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen",
                    "ua_nitrite", "ua_pH", "ua_sg"]
PROC_NUMERIC = ["age_at_urine_test", "days_difference", "ALT", "AST"]
RAW_WANT = ["일련번호", "소변검사일시", "ua_nitrite", "ua_glucose"]
LAB_ANALYTES = ("alt", "ast")
# The columns the study reads from each hospital file; dev_schema.csv lists only these, never the names of other
# laboratory columns.
PROC_SCHEMA = PROC_WANT + ["수진일자"]
LAB_SCHEMA = (["id", "date"] + [f"{a}{suffix}" for a in LAB_ANALYTES for suffix in ("", "_cmp", "_nrep", "_last")]
              + ["age", "adult", "is_checkup"])
EPISODE_SCHEMA = ["id", "dt", "nitrite_pos"]
MAX_DISTINCT = 60


def fmt(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def col(d, name):
    """Column of d, or an all-missing column when the file does not have it."""
    return d[name] if name in d else pd.Series(np.nan, index=d.index)


def numeric_row(key, var, s):
    s = pd.Series(s)
    x = pd.to_numeric(s, errors="coerce")
    n = int(x.notna().sum())
    p = np.nanpercentile(x, [1, 50, 99]) if n >= 100 else [np.nan] * 3
    return (key, var, n, int(s.isna().sum()), int((s.notna() & x.isna()).sum()), *[float(v) for v in p])


def value_rows(key, var, s, max_distinct=MAX_DISTINCT):
    """(value rows, numeric row): value counts when the column has at most max_distinct values, else a numeric row."""
    s = pd.Series(s)
    vc = s.value_counts(dropna=True)
    if len(vc) > max_distinct:
        return [], numeric_row(key, var, s)
    items = list(vc.items())
    try:
        items.sort(key=lambda kv: float(kv[0]))
    except (TypeError, ValueError):
        items.sort(key=lambda kv: str(kv[0]))
    rows = [(key, var, fmt(v), int(n)) for v, n in items]
    rows.append((key, var, "(missing)", int(s.isna().sum())))
    return rows, None


def text_rows(key, var, s):
    """Free-text values: those seen at least 10 times, the rest pooled, so that no rare string is written."""
    s = pd.Series(s).astype("string").str.strip()
    vc = s.value_counts(dropna=True)
    rows = [(key, var, str(v), int(n)) for v, n in vc.items() if n >= runlog.SMALL]
    rare = vc[vc < runlog.SMALL]
    rows.append((key, var, f"(other: {len(rare)} distinct values, each seen <10 times)", int(rare.sum())))
    rows.append((key, var, "(missing)", int(s.isna().sum())))
    return rows


def schema_rows(checks, file, dtypes, want):
    """dev_schema.csv rows of the columns the study reads, in that order (no other column name is written); data_checks
    rows give the number of the file's other columns and name any wanted column the file lacks."""
    missing = [c for c in want if c not in dtypes]
    checks.append((file, "other columns of the file (not listed)", sum(c not in want for c in dtypes), ""))
    checks.append((file, "columns the study reads that the file lacks", len(missing), ", ".join(missing)))
    return [(file, c, str(dtypes[c])) for c in want if c in dtypes]


# ------------------------------------------------------------------------------------------------ sections
def environment():
    lines = [f"python {sys.version.split()[0]}", f"executable {sys.executable}"]
    for p in PACKAGES:
        try:
            lines.append(f"{p} {md.version(p)}")
        except md.PackageNotFoundError:
            lines.append(f"{p} MISSING")
    with open(os.path.join(runlog.HERE, "environment.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    say(f"environment: {len(lines)} lines; missing packages {sum(l.endswith('MISSING') for l in lines)}")


def _kn_reader(path):
    import pyreadstat
    return pyreadstat.read_xport if path.lower().endswith(".xpt") else pyreadstat.read_sas7bdat


def kn_meta(path):
    """(names, labels, rows, encoding used). Tries the file's own encoding, then CP949 and EUC-KR, and finally
    latin-1 with the labels re-decoded as CP949."""
    read = _kn_reader(path)
    for enc in (None, "cp949", "euc-kr"):
        try:
            _, m = read(path, metadataonly=True) if enc is None else read(path, metadataonly=True, encoding=enc)
        except Exception:  # noqa: BLE001 - try the next encoding
            continue
        labels = list(m.column_labels) if m.column_labels else [None] * len(m.column_names)
        if any(l and HANGUL.search(l) for l in labels):
            return list(m.column_names), labels, m.number_rows, enc or "file"
    _, m = read(path, metadataonly=True, encoding="latin1")
    labels = [l.encode("latin1").decode("cp949", errors="replace") if l else l for l in (m.column_labels or [])]
    return list(m.column_names), labels, m.number_rows, "latin1>cp949"


def knhanes(checks):
    var_rows, val_rows, num_rows = [], [], []
    for year in P.KN_YEARS:
        path = P.KN_FILES[year]
        names, labels, nrows, enc = kn_meta(path)
        keep = [(n, l) for n, l in zip(names, labels) if KN_NAME.match(n) or (l and KN_LABEL.search(l))]
        cols = [n for n, _ in keep if n.lower() not in KN_NO_VALUES]
        try:                                    # CP949 (latin-1 garbles the Korean text columns)
            df, _ = _kn_reader(path)(path, usecols=cols, encoding="cp949")
            data_enc = "cp949"
        except Exception:  # noqa: BLE001 - fall back, and say so in data_checks.csv
            df, _ = _kn_reader(path)(path, usecols=cols, encoding="latin1")
            data_enc = "latin1"
        nrows = int(nrows) if nrows is not None else len(df)   # transport files carry no row count
        var_rows += [(year, n, l, nrows) for n, l in keep]
        for c in cols:
            if c.lower() in KN_NUMERIC:
                num_rows.append(numeric_row(year, c, df[c]))
                continue
            if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                val_rows += text_rows(year, f"{c} (text)", df[c])
                continue
            r, num = value_rows(year, c, df[c])
            val_rows += r
            if num:
                num_rows.append(num)
        checks += [("knhanes", f"{year} file rows", int(nrows), ""), ("knhanes", f"{year} variables kept", len(keep), ""),
                   ("knhanes", f"{year} label encoding", None, enc), ("knhanes", f"{year} data encoding", None, data_enc)]
        age = next((c for c in cols if c.lower() == "age"), None)
        if age:
            a = pd.to_numeric(df[age], errors="coerce")
            checks += [("knhanes", f"{year} age >= 19", int((a >= 19).sum()), ""),
                       ("knhanes", f"{year} age missing", int(a.isna().sum()), "")]
        for name in KN_ENZYMES:                 # counts the percentile tables cannot show (no value is written)
            c = next((x for x in cols if x.lower() == name.lower()), None)
            if c is None:
                checks.append(("knhanes", f"{year} {name} not in the file", None, "absent"))
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            checks.append(("knhanes", f"{year} {name} 1,000 U/L or more", int((v >= 1000).sum()), ""))
            for code in KN_SPECIAL_CODES:
                checks.append(("knhanes", f"{year} {name} equal to {code}", int((v == code).sum()),
                               "possible special code"))
        say(f"knhanes {year}: rows {nrows:,}; variables kept {len(keep)}")
    runlog.write_aggregate(pd.DataFrame(var_rows, columns=["year", "variable", "label", "file_rows"]),
                           "knhanes_variables.csv", count_cols=["file_rows"])
    runlog.write_aggregate(pd.DataFrame(val_rows, columns=["year", "variable", "value", "n"]),
                           "knhanes_values.csv", count_cols=["n"])
    runlog.write_aggregate(pd.DataFrame(num_rows, columns=["year", "variable", "n", "n_missing", "n_nonnumeric",
                                                           "p01", "p50", "p99"]),
                           "knhanes_numeric.csv", count_cols=["n", "n_missing", "n_nonnumeric"])


def hospital(checks, schema, values, numeric):
    # ---- processed development file
    head = pd.read_csv(P.DATA_PATH, nrows=2000, low_memory=False)
    schema += schema_rows(checks, "processed", dict(head.dtypes.items()), PROC_SCHEMA)
    use = [c for c in PROC_WANT if c in head.columns]
    d = pd.read_csv(P.DATA_PATH, usecols=use, low_memory=False)
    ids = col(d, "일련번호")
    per = ids.value_counts()
    age = pd.to_numeric(col(d, "age_at_urine_test"), errors="coerce")
    dd = pd.to_numeric(col(d, "days_difference"), errors="coerce")
    sex_missing = ~col(d, "성별").isin(["M", "F"])
    adult_dd0 = (age >= 19) & (age <= 110) & (dd == 0)
    glu4 = pd.to_numeric(col(d, "ua_glucose"), errors="coerce") == 4
    checks += [("processed", "rows", len(d)), ("processed", "persons", int(ids.nunique())),
               ("processed", "persons with more than one row", int((per > 1).sum())),
               ("processed", "rows age < 19", int((age < 19).sum())),
               ("processed", "rows age > 110", int((age > 110).sum())),
               ("processed", "rows age missing", int(age.isna().sum())),
               ("processed", "rows days_difference == 0", int((dd == 0).sum())),
               ("processed", "rows days_difference missing", int(dd.isna().sum())),
               ("processed", "rows sex not M or F", int(sex_missing.sum())),
               ("processed", "rows ua_glucose == 4", int(glu4.sum())),
               ("processed", "adult rows with days_difference == 0", int(adult_dd0.sum())),
               ("processed", "adult rows with days_difference == 0 and ua_glucose == 4", int((adult_dd0 & glu4).sum())),
               ("processed", "adult rows with days_difference == 0 and sex not M or F",
                int((adult_dd0 & sex_missing).sum()))]
    for c in PROC_CATEGORICAL:
        if c in d:
            r, num = value_rows("processed", c, d[c])
            values += r
            if num:
                numeric.append(num)
    if "소변검사일자" in d:
        yr = pd.to_datetime(d["소변검사일자"].astype(str), errors="coerce", format="mixed").dt.year
        values += value_rows("processed", "urine test year", yr)[0]
    numeric += [numeric_row("processed", c, d[c]) for c in PROC_NUMERIC if c in d]
    say(f"processed file: rows {len(d):,}; persons {ids.nunique():,}")
    del d

    # ---- raw first-visit file
    rhead = pd.read_csv(P.RAW_FIRST, nrows=2000, dtype=str, encoding="utf-8-sig")
    schema += schema_rows(checks, "raw_first_visit", dict.fromkeys(rhead.columns, "text"), RAW_WANT)
    r = pd.read_csv(P.RAW_FIRST, usecols=[c for c in RAW_WANT if c in rhead.columns], dtype=str,
                    encoding="utf-8-sig")
    checks += [("raw_first_visit", "rows", len(r)),
               ("raw_first_visit", "persons", int(r["일련번호"].nunique()) if "일련번호" in r else 0)]
    for c in ("ua_nitrite", "ua_glucose"):
        if c in r:
            values += text_rows("raw_first_visit", f"{c} (text)", r[c])
    say(f"raw first-visit file: rows {len(r):,}")
    del r

    # ---- atlas caches
    import pyarrow.parquet as pq
    sch = pq.read_schema(P.LAB_DAY)
    schema += schema_rows(checks, "lab_day_wide", {f.name: f.type for f in sch}, LAB_SCHEMA)
    checks.append(("lab_day_wide", "rows (person-days)", pq.ParquetFile(P.LAB_DAY).metadata.num_rows))
    for a in LAB_ANALYTES:                      # one analyte at a time keeps memory low
        if a not in sch.names:
            continue
        lab = pq.read_table(P.LAB_DAY, columns=[c for c in (a, f"{a}_cmp", f"{a}_nrep") if c in sch.names]).to_pandas()
        v = pd.to_numeric(lab[a], errors="coerce")
        checks.append(("lab_day_wide", f"person-days with {a}", int(v.notna().sum())))
        numeric.append(numeric_row("lab_day_wide", a, lab[a]))
        for extra in (f"{a}_cmp", f"{a}_nrep"):
            if extra in lab:
                values += value_rows("lab_day_wide", extra, lab[extra])[0]
        cmp_col = f"{a}_cmp"
        if cmp_col in lab:
            for sign in ("<", ">"):
                sel = lab[cmp_col].astype("string") == sign
                values += value_rows("lab_day_wide", f"{a} value where {a}_cmp is '{sign}'",
                                     v[sel.fillna(False).to_numpy()])[0]
        del lab, v
    say("lab_day_wide: done")
    dsch = pq.read_schema(P.DIPSTICK_EPISODE)
    schema += schema_rows(checks, "dipstick_episode_wide", {f.name: f.type for f in dsch}, EPISODE_SCHEMA)


def anthropometry(checks, schema, values, numeric):
    """The atlas anthropometry table: schema, rows, repeated (id, date) pairs, counts per value of source, bmi_derived
    and nursing_conflict, rows per calendar year, and BMI percentiles (from 100 values; never a minimum or maximum)."""
    import pyarrow.parquet as pq
    sch = pq.read_schema(P.ANTHRO_DAY)
    schema += [("anthro_day", f.name, str(f.type)) for f in sch]
    checks.append(("anthro_day", "rows (person-days)", pq.ParquetFile(P.ANTHRO_DAY).metadata.num_rows))
    use = [c for c in ("id", "date", "bmi", "bmi_derived", "source", "nursing_conflict") if c in sch.names]
    a = pq.read_table(P.ANTHRO_DAY, columns=use).to_pandas()
    if "id" in a and "date" in a:
        checks.append(("anthro_day", "rows repeating a person and day (id, date)",
                       int(a.duplicated(["id", "date"]).sum())))
    if "date" in a:
        values += value_rows("anthro_day", "calendar year", pd.to_datetime(a["date"], errors="coerce").dt.year)[0]
    if "source" in a:
        values += text_rows("anthro_day", "source (text)", a["source"])
    for c in ("nursing_conflict", "bmi_derived"):
        if c in a:
            r, num = value_rows("anthro_day", c, a[c])
            values += r
            if num:
                numeric.append(num)
    if "bmi" in a:
        numeric.append(numeric_row("anthro_day", "bmi", a["bmi"]))
    if "bmi" in a and "bmi_derived" in a:       # the flag itself has no percentiles: BMI for each value of the flag
        for flag in (True, False):
            sel = (a["bmi_derived"] == flag).fillna(False).to_numpy(bool)
            numeric.append(numeric_row("anthro_day", f"bmi where bmi_derived is {flag}", a.loc[sel, "bmi"]))
    say(f"anthro_day: rows {len(a):,}")


def main():
    checks, schema, values, numeric = [], [], [], []
    failed = 0
    for name, fn in (("environment", environment), ("knhanes", lambda: knhanes(checks)),
                     ("hospital", lambda: hospital(checks, schema, values, numeric)),
                     ("anthropometry", lambda: anthropometry(checks, schema, values, numeric))):
        try:
            fn()
            checks.append(("s00", f"section {name}", None, "ok"))
        except Exception as e:  # noqa: BLE001 - recorded, traceback to standard error, other sections still run
            failed += 1
            checks.append(("s00", f"section {name}", None, f"failed: {type(e).__name__}"))
            traceback.print_exc()
            say(f"section {name} FAILED ({type(e).__name__}); see cache/s00_env_check.err")
    runlog.write_aggregate(pd.DataFrame(schema, columns=["file", "column", "dtype"]), "dev_schema.csv")
    runlog.write_aggregate(pd.DataFrame(values, columns=["file", "variable", "value", "n"]), "dev_values.csv",
                           count_cols=["n"])
    runlog.write_aggregate(pd.DataFrame(numeric, columns=["file", "variable", "n", "n_missing", "n_nonnumeric",
                                                          "p01", "p50", "p99"]),
                           "dev_numeric.csv", count_cols=["n", "n_missing", "n_nonnumeric"])
    checks = [c if len(c) == 4 else (*c, "") for c in checks]
    runlog.write_aggregate(pd.DataFrame(checks, columns=["source", "item", "n", "note"]), "data_checks.csv",
                           count_cols=["n"])
    if failed:
        raise RuntimeError(f"{failed} section(s) failed")
    say("s00 done")


if __name__ == "__main__":
    runlog.run(main, __file__)
