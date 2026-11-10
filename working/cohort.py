"""Cohorts, person-level caches and shared helpers of the scripts that read individual-level data (s01-s07).

Paths come from paths_local; assertion messages carry counts only.
"""
import hashlib
import os

import numpy as np
import pandas as pd

import paths_local as P
import runlog

# ------------------------------------------------------------------------------------------------ constants
DIPSTICK = ["ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen", "ua_nitrite",
            "ua_pH", "ua_sg"]
INPUTS = DIPSTICK + ["age", "male", "bmi"]
_INPUT_TEXTS = ["Urine glucose", "Urine protein", "Urine ketones", "Urine bilirubin", "Urine blood",
                "Urine urobilinogen", "Urine nitrite", "Urine pH", "Urine specific gravity", "Age", "Male sex", "BMI"]
assert len(_INPUT_TEXTS) == len(INPUTS), "INPUT_LABELS and INPUTS differ in length"
INPUT_LABELS = dict(zip(INPUTS, _INPUT_TEXTS))
OUTCOMES = {"ALT": dict(value="alt_value", cmp="alt_cmp", sample="in_alt", threshold=40.0, unit="U/L"),
            "AST": dict(value="ast_value", cmp="ast_cmp", sample="in_ast", threshold=40.0, unit="U/L")}
DATASETS = ["DEV_TRAIN", "DEV_TEST", "KN"]
MODELS = ["ENS", "LR_ASB", "LR_ALL"]
RULES = {"BMI25": ("bmi", 25.0)}
THRESHOLD_NAMES = ["YOUDEN", "SENS80", "SENS90", "MATCH_BMI25"]
LR_INPUTS = {"LR_ASB": ["age", "male", "bmi"], "LR_ALL": list(INPUTS)}
SENS_TARGETS = {"SENS80": 0.80, "SENS90": 0.90}
SEED, ADULT_AGE, AGE_MAX = 42, 19, 110
STUDY_START, STUDY_END = pd.Timestamp(2010, 1, 1), pd.Timestamp(2025, 8, 14)
GLU4_RULE = "merge"              # hospital urine glucose 4 -> 3 ("keep" leaves it at 4)
SEX_MISSING_RULE = "exclude"     # hospital sex other than M/F: missing input ("female" sets 0)
HEIGHT_RANGE = (130.0, 219.9)    # cm, both ends included (hospital anthropometry table and KNHANES)
WEIGHT_RANGE = (30.0, 249.9)     # kg, both ends included
BMI_RANGE = (12.0, 60.0)         # kg/m2, both ends included
BMI_LOOKBACK_DAYS = 30           # BMI of the urine day, else the latest of days -30 to -1; never after the urine day
KN_RAW = {"ua_glucose": "he_uglu", "ua_protein": "he_upro", "ua_ketone": "he_uket", "ua_bilirubin": "he_ubil",
          "ua_blood": "he_ubld", "ua_urobilinogen": "he_uro", "ua_nitrite": "he_unitr", "ua_pH": "he_uph",
          "ua_sg": "he_usg"}
KN_GRADED = {0: 0, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}
KN_URO = {0: 0.5, 1: 0.5, 2: 1, 3: 2, 4: 3, 5: 3}
AGE_BANDS = ["19-39", "40-59", "60-79", "80+"]
KN_PERIODS = ["2016-2018", "2019-2021", "2022-2024"]
URINE_PERIODS = ["2010-2013", "2014-2017", "2018-2021", "2022-2025"]
SHAP_N = 5000

ROLES = ("pregnancy", "alcohol")
MEANINGS = ("yes", "no", "not_applicable", "unknown")
DRINK_LEVELS = ("none_or_rare", "monthly", "weekly", "unknown")      # meanings of the alcohol role of the code table
NOT_LISTED = "not in table"
CMP_SIGNS = ("=", "<", "<=", ">", ">=")
PRED_DATASETS = ("DEV_TEST", "KN")

# hospital file columns
COL_ID, COL_DAY, COL_RAW_TIME = "일련번호", "소변검사일자", "소변검사일시"
COL_SEX, COL_CHECKUP = "성별", "수진일자"
PROC_COLS = [COL_ID, COL_DAY, "age_at_urine_test", COL_SEX, "days_difference"] + DIPSTICK + [COL_CHECKUP]
PROC_NEAREST = {"ALT": "ALT", "AST": "AST"}      # nearest-day values: one agreement count, if present
LAB_COLS = ["id", "date", "alt", "alt_cmp", "alt_nrep", "alt_last", "ast", "ast_cmp", "ast_nrep", "ast_last", "age",
            "adult", "is_checkup"]
ANTHRO_CHECKUP_WORDS = ("check", "검진")           # a source text holding one of these names a check-up record
KN_BASE = (["id", "sex", "age"] + [KN_RAW[c] for c in DIPSTICK]
           + ["he_alt", "he_ast", "he_ht", "he_wt", "he_bmi", "wt_itvex", "psu", "kstrata"])

DEV_CACHE = os.path.join(runlog.CACHE, "dev.parquet")
KN_CACHE = os.path.join(runlog.CACHE, "knhanes.parquet")
DEV_COLUMNS = ["id", "role"] + INPUTS + ["alt_value", "alt_cmp", "in_alt", "ast_value", "ast_cmp", "in_ast", "stratum",
                                         "checkup_day", "urine_year"]
KN_COLUMNS = ["id", "year", "period"] + INPUTS + ["alt_value", "alt_cmp", "in_alt", "ast_value", "ast_cmp", "in_ast",
                                                  "drink_group", "wt_itvex", "psu", "kstrata"]
_MEMO = {}
MODEL_FOLDER = "models"


def check_paths():
    """Positive check of the local settings (skipped in synthetic runs): MODEL_DIR is the folder named MODEL_FOLDER and
    KN_CODE_TABLE lies inside the notes folder next to this working folder. Messages name the setting, never a path."""
    if P.SYNTHETIC:
        return
    assert os.path.basename(os.path.normpath(P.MODEL_DIR)) == MODEL_FOLDER, \
        "MODEL_DIR is not a folder named MODEL_FOLDER"
    notes = os.path.realpath(os.path.join(runlog.HERE, os.pardir, "notes"))
    table = os.path.realpath(P.KN_CODE_TABLE)
    assert os.path.commonpath([notes, table]) == notes, \
        "KN_CODE_TABLE is not inside the notes folder next to the working folder"


check_paths()


# ------------------------------------------------------------------------------------------------ small helpers
def _check(rows, source, item, n=None, value=None, note="logged"):
    rows.append(dict(source=source, item=item, n=None if n is None else int(n), value=value, note=note))


def _zero(rows, source, item, n):
    """Add a check row that must be 0; return '' when it is, else the stop message (count and item).

    Used as `assert not (stop := _zero(...)), stop`, so that a failed check names its own line in the status file."""
    n = int(n)
    _check(rows, source, item, n, note="asserted 0")
    return "" if n == 0 else f"{runlog.count_text(n)} {item}"


def _values(x):
    """A flat float array (non-numbers -> NaN)."""
    s = x if isinstance(x, pd.Series) else pd.Series(np.ravel(np.asarray(x, dtype=object)))
    return pd.to_numeric(s, errors="coerce").astype(float).to_numpy()


def _texts(x, n):
    """A flat object array of stripped comparator texts, '=' where blank (n entries)."""
    if isinstance(x, str) or x is None:
        return np.full(n, x if x else "=", dtype=object)
    s = x if isinstance(x, pd.Series) else pd.Series(np.ravel(np.asarray(x, dtype=object)))
    s = s.astype("string").str.strip()
    return s.where(s.notna() & (s != ""), "=").to_numpy(object)


def _day(s):
    """Calendar day of a date column (text, or yyyymmdd numbers); NaT where unreadable."""
    if pd.api.types.is_numeric_dtype(s):
        s = pd.Series([f"{int(v)}" if pd.notna(v) and float(v).is_integer() else None for v in s], index=s.index,
                      dtype="string")
    else:
        s = s.astype("string").str.strip()
    return pd.to_datetime(s, errors="coerce", format="mixed").dt.normalize().astype("datetime64[ns]")


def level_text(v):
    """A coded value as text: '0', '0.5', '1.005'."""
    return format(float(v), ".10g")


def code_texts(values):
    """KNHANES codes as text, same index: numbers as level_text, missing as 'blank', other text stripped."""
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    num = pd.to_numeric(s, errors="coerce").astype(float)
    out = pd.Series("blank", index=s.index, dtype=object)
    ok = num.notna().to_numpy()
    out[ok] = [level_text(v) for v in num.to_numpy()[ok]]
    txt = ~ok & s.notna().to_numpy()
    if txt.any():
        t = s[txt].astype(str).str.strip()
        out[txt] = t.where(t != "", "blank").to_numpy(object)
    return out


def kn_period(year):
    y = np.asarray(year, int)
    return np.select([y <= 2018, y <= 2021], KN_PERIODS[:2], KN_PERIODS[2]).astype(object)


def urine_period(year):
    y = np.asarray(year, int)
    return np.select([y <= 2013, y <= 2017, y <= 2021], URINE_PERIODS[:3], URINE_PERIODS[3]).astype(object)


# ------------------------------------------------------------------------------------------------ pure rules
def is_adult(age):
    """Adult flag, ADULT_AGE <= age <= AGE_MAX; NaN or text gives False."""
    if isinstance(age, pd.Series):
        a = pd.to_numeric(age, errors="coerce").astype(float)
        return (a >= ADULT_AGE) & (a <= AGE_MAX)
    a = _values(age)
    ok = (a >= ADULT_AGE) & (a <= AGE_MAX)
    return bool(ok[0]) if np.ndim(age) == 0 else ok.reshape(np.shape(age))


def age_band(age):
    """AGE_BANDS labels (None below 19 or when missing); a Series keeps its index."""
    a = _values(age)
    out = np.full(len(a), None, dtype=object)
    for lab, lo in zip(AGE_BANDS, (19, 40, 60, 80)):
        out[a >= lo] = lab
    return pd.Series(out, index=age.index) if isinstance(age, pd.Series) else out


def nitrite_from_text(text):
    """1.0 for POSITIVE or +, 0.0 for NEGATIVE, - or 0, else NaN (spaces and case ignored); keeps the index."""
    s = text if isinstance(text, pd.Series) else pd.Series(np.ravel(np.asarray(text, dtype=object)))
    u = s.astype("string").str.strip().str.upper()
    v = np.full(len(u), np.nan)
    v[u.isin(["POSITIVE", "+"]).fillna(False).to_numpy(bool)] = 1.0
    v[u.isin(["NEGATIVE", "-", "0"]).fillna(False).to_numpy(bool)] = 0.0
    return pd.Series(v, index=s.index)


def knhanes_dipstick_map(raw):
    """The 9 dipstick inputs from KNHANES codes (lower-case KN_RAW columns of raw); codes outside the maps -> NaN."""
    graded = {float(k): float(v) for k, v in KN_GRADED.items()}
    uro = {float(k): float(v) for k, v in KN_URO.items()}
    out = pd.DataFrame(index=raw.index)
    for c in DIPSTICK:
        x = pd.to_numeric(raw[KN_RAW[c]], errors="coerce").astype(float)
        if c == "ua_urobilinogen":
            out[c] = x.map(uro)
        elif c == "ua_nitrite":
            out[c] = np.where(x >= 1, 1.0, np.where(x == 0, 0.0, np.nan))
        elif c in ("ua_pH", "ua_sg"):
            out[c] = x
        else:
            out[c] = x.map(graded)
    return out.astype(float)


def label_parts(value, cmp, threshold):
    """(labels, undetermined): outcome_label without the stop; undetermined marks values no rule can label."""
    v = _values(value)
    c = _texts(cmp, len(v))
    assert len(c) == len(v), "values and comparators differ in length"
    out = np.full(len(v), np.nan)
    has = ~np.isnan(v)
    eq = has & (c == "=")
    out[eq] = (v[eq] > threshold).astype(float)
    out[has & np.isin(c, ["<", "<="]) & (v <= threshold)] = 0.0
    out[has & (c == ">") & (v >= threshold)] = 1.0
    out[has & (c == ">=") & (v > threshold)] = 1.0
    return out, has & np.isnan(out)


def outcome_label(value, cmp, threshold):
    """1/0 labels in stored units; NaN where the value is missing; stops on censored values it cannot label.

    '=' or no comparator: 1 if value > threshold, else 0. '<' or '<=' L: 0 if L <= threshold. '>' L: 1 if
    L >= threshold. '>=' L: 1 if L > threshold."""
    out, bad = label_parts(value, cmp, threshold)
    k = int(bad.sum())
    assert k == 0, f"{runlog.count_text(k)} censored or unknown-comparator values whose label cannot be determined"
    return out


def strata(alt_value, alt_cmp, ast_value, ast_cmp):
    """Split stratum 0-3 = 2 x ALT label + AST label (labels 0/1; both enzymes present and determinate)."""
    a = outcome_label(alt_value, alt_cmp, OUTCOMES["ALT"]["threshold"])
    b = outcome_label(ast_value, ast_cmp, OUTCOMES["AST"]["threshold"])
    assert not (np.isnan(a) | np.isnan(b)).any(), "a person without both labels"
    return (2 * a + b).astype(np.int8)


def combine_meanings(meanings, n):
    """One meaning per person from several variables of a role: yes, else unknown, else no, else not_applicable."""
    out = np.full(n, "not_applicable", dtype=object)
    if not meanings:
        return out
    m = np.column_stack([np.asarray(x, dtype=object) for x in meanings])
    for level in ("no", "unknown", "yes"):           # later levels take precedence
        out[(m == level).any(axis=1)] = level
    return out


def code_meanings(values, rows):
    """(codes, meanings): KNHANES codes as text and the meaning each has in the code-table rows of one variable and
    year (missing for a code the table does not list); the index of values is kept."""
    codes = code_texts(values)
    return codes, codes.map(dict(zip(rows["code"], rows["meaning"])))


def n_trials():
    """Optuna trials per learner: 25 (LIV_SYNTH_TRIALS only in synthetic runs)."""
    if P.SYNTHETIC and os.environ.get("LIV_SYNTH_TRIALS"):
        return int(os.environ["LIV_SYNTH_TRIALS"])
    return 25


# ------------------------------------------------------------------------------------------------ KNHANES files
def _kn_reader(path):
    """(read function, encoding, label): SAS files in CP949; transport files only in synthetic runs."""
    import pyreadstat
    if path.lower().endswith(".xpt"):
        assert P.SYNTHETIC, "transport files are read only in synthetic runs"
        return pyreadstat.read_xport, "utf-8", "read_xport utf-8"
    return pyreadstat.read_sas7bdat, "cp949", "read_sas7bdat cp949"


def kn_names(year):
    """Variable names of one KNHANES file (metadata only)."""
    path = P.KN_FILES[year]
    read, enc, _ = _kn_reader(path)
    _, meta = read(path, metadataonly=True, encoding=enc)
    return list(meta.column_names)


def read_knhanes_raw(year, columns):
    """The named variables of one survey year, names matched case-insensitively and returned in lower case, with
    year set from the file key; attrs["names"] maps each lower-case name to the name in the file."""
    path = P.KN_FILES[year]
    read, enc, _ = _kn_reader(path)
    names = {}
    for c in kn_names(year):
        names.setdefault(c.lower(), c)
    want = list(dict.fromkeys(str(c).lower() for c in columns))
    missing = [c for c in want if c not in names]
    assert not missing, f"{len(missing):,} requested variable(s) missing from a survey file"
    use = [names[c] for c in want] + ([names["year"]] if "year" in names and "year" not in want else [])
    df, _ = read(path, usecols=use, encoding=enc)
    df.columns = [c.lower() for c in df.columns]
    if "year" in df:
        bad = int((pd.to_numeric(df["year"], errors="coerce") != year).sum())
        assert bad == 0, f"{runlog.count_text(bad)} survey records carry a year other than their file's"
    df["year"] = year
    df = df[want + ([] if "year" in want else ["year"])].copy()
    df.attrs["names"] = {c: names[c] for c in want}
    return df


def _years(text):
    """Survey years of a code-table cell ('2016-2024', '2016,2017,2018', '2016-2018,2020'); None if unreadable."""
    out = []
    try:
        for part in str(text).replace(" ", "").split(","):
            if "-" in part:
                a, b = (int(x) for x in part.split("-"))
                if a > b:
                    return None
                out += list(range(a, b + 1))
            elif part:
                out.append(int(part))
    except ValueError:
        return None
    return out or None


def _read_code_table(path):
    """(text table, encoding): UTF-8 (a byte-order mark allowed), or CP949, the plain CSV of Korean-locale Excel, only
    when the file is not UTF-8; no third encoding is tried."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp949"), "cp949"


def load_code_table(path=None, checks=None):
    """The validated KNHANES code table: one row per (role, variable, year, code) with its meaning. The
    encoding it was read with is a check row, appended to checks when given, before the table is validated."""
    t, encoding = _read_code_table(path or P.KN_CODE_TABLE)
    _check([] if checks is None else checks, "KN", "code table encoding", value=encoding)
    t.columns = [str(c).strip() for c in t.columns]
    assert list(t.columns) == ["role", "variable", "years", "code", "meaning", "note"], \
        "code table header differs from role, variable, years, code, meaning, note"
    t = t.apply(lambda s: s.astype(str).str.strip())
    bad = int((~t["role"].isin(ROLES)).sum())
    assert bad == 0, f"code table: {bad:,} row(s) with an unknown role"
    ok = np.where(t["role"] == "alcohol", t["meaning"].isin(DRINK_LEVELS), t["meaning"].isin(MEANINGS))
    bad = int((~ok).sum())
    assert bad == 0, f"code table: {bad:,} row(s) with a meaning not allowed for their role"
    bad = int((t["variable"] == "").sum())
    assert bad == 0, f"code table: {bad:,} row(s) without a variable"
    years = t["years"].map(_years)
    bad = int(years.isna().sum())
    assert bad == 0, f"code table: {bad:,} row(s) with unreadable years"
    bad = sum(1 for ys in years for y in ys if y not in P.KN_YEARS)
    assert bad == 0, f"code table: {bad:,} year(s) outside the survey years"

    def code(c):
        if c.lower() == "blank":
            return "blank"
        try:
            v = float(c)
        except ValueError:
            return None
        return level_text(v) if np.isfinite(v) else None
    codes = t["code"].map(code)
    bad = int(codes.isna().sum())
    assert bad == 0, f"code table: {bad:,} code(s) that are neither a number nor 'blank'"
    bad = int(((codes == "blank") & (t["meaning"] != "unknown")).sum())
    assert bad == 0, f"code table: {bad:,} blank code(s) not listed as unknown"
    rows = [(r, v, v.lower(), y, c, m) for r, v, ys, c, m in zip(t["role"], t["variable"], years, codes, t["meaning"])
            for y in ys]
    out = pd.DataFrame(rows, columns=["role", "variable", "var", "year", "code", "meaning"])
    bad = int(out.duplicated(["var", "year", "code"]).sum())
    assert bad == 0, f"code table: {bad:,} repeated (variable, year, code) entries"
    bad = int((out.groupby("var")["role"].nunique() > 1).sum())
    assert bad == 0, f"code table: {bad:,} variable(s) listed under two roles"
    have = set(zip(out["year"], out["role"]))
    bad = sum((y, r) not in have for y in P.KN_YEARS for r in ROLES)
    assert bad == 0, f"code table: {bad:,} (year, role) pair(s) without a variable"
    bad = int((out[out["role"] == "alcohol"].groupby("year")["var"].nunique() > 1).sum())
    assert bad == 0, f"code table: {bad:,} year(s) with more than one alcohol variable"
    return out


# ------------------------------------------------------------------------------------------------ flow rows
class _Flow:
    """Rows of cohort_flow.csv; excluded is the drop from the previous step of the chain."""

    def __init__(self, source):
        self.source, self.rows, self.n = source, [], {}

    def step(self, order, sample, key, label, mask, after=None):
        n = int(np.sum(mask))
        excluded = None if after is None else self.n[after] - n
        self.rows.append(dict(source=self.source, sample=sample, order=order, key=key, parent=after or "",
                              kind="step", label=label, n=n, excluded=excluded, events=None))
        self.n[key] = n

    def reasons(self, order, sample, step, items):
        """Exclusion reasons of step (list of (key, label, mask), mutually exclusive); their sum must match."""
        total = 0
        for key, label, mask in items:
            n = int(np.sum(mask))
            total += n
            self.rows.append(dict(source=self.source, sample=sample, order=order, key=key, parent=step,
                                  kind="reason", label=label, n=n, excluded=None, events=None))
        drop = self.rows[[r["key"] for r in self.rows].index(step)]["excluded"]
        assert total == drop, f"{runlog.count_text(abs(total - drop))} persons differ between the reasons and the step"

    def final(self, order, sample, key, label, parent, n, events=None):
        self.rows.append(dict(source=self.source, sample=sample, order=order, key=key, parent=parent, kind="final",
                              label=label, n=int(n), excluded=None, events=None if events is None else int(events)))


# ------------------------------------------------------------------------------------------------ development
def _read_lab(ids):
    import pyarrow.parquet as pq
    lab = pq.read_table(P.LAB_DAY, columns=LAB_COLS).to_pandas(date_as_object=False)
    lab["id"] = lab["id"].astype(np.int64)
    lab["date"] = pd.to_datetime(lab["date"]).astype("datetime64[ns]")
    return lab[lab["date"].notna() & lab["id"].isin(ids)].reset_index(drop=True)


def _read_episodes():
    import pyarrow.parquet as pq
    ep = pq.read_table(P.DIPSTICK_EPISODE, columns=["id", "dt", "nitrite_pos"]).to_pandas()
    ep["id"] = ep["id"].astype(np.int64)
    ep["dt"] = pd.to_datetime(ep["dt"]).astype("datetime64[ns]")
    ep["nitrite_pos"] = pd.to_numeric(ep["nitrite_pos"], errors="coerce").astype(float)
    ep = ep[ep["dt"].notna()]
    n_dup = int(ep.duplicated(["id", "dt"]).sum())
    return ep.drop_duplicates(["id", "dt"], keep="first"), n_dup


def _between(x, lo, hi):
    """Elements of the array x within [lo, hi] (NaN gives False)."""
    return (x >= lo) & (x <= hi)


def _read_anthro(ids, ck):
    """The anthropometry rows of these persons that can give a BMI, for bmi_as_of (columns id, date, bmi, source). Rows
    with a height, weight or BMI outside HEIGHT_RANGE, WEIGHT_RANGE or BMI_RANGE are dropped and counted in a RULE row;
    rows without a BMI value are dropped and counted."""
    import pyarrow.parquet as pq
    a = pq.read_table(P.ANTHRO_DAY, columns=["id", "date", "height", "weight", "bmi", "source"]).to_pandas(
        date_as_object=False)
    a["id"] = a["id"].astype(np.int64)
    a["date"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    a = a[a["date"].notna() & a["id"].isin(ids)].reset_index(drop=True)
    h, w, b = (pd.to_numeric(a[c], errors="coerce").astype(float).to_numpy() for c in ("height", "weight", "bmi"))
    outside = ((~np.isnan(h) & ~_between(h, *HEIGHT_RANGE)) | (~np.isnan(w) & ~_between(w, *WEIGHT_RANGE))
               | (~np.isnan(b) & ~_between(b, *BMI_RANGE)))
    keep = ~outside & ~np.isnan(b)
    _check(ck, "DEV", "anthropometry rows of the processed persons", len(a))
    _check(ck, "RULE", "anthropometry rows dropped before the BMI join: height, weight or BMI outside the ranges",
           int(outside.sum()), f"height {HEIGHT_RANGE[0]:g}-{HEIGHT_RANGE[1]:g} cm, weight {WEIGHT_RANGE[0]:g}-"
           f"{WEIGHT_RANGE[1]:g} kg, BMI {BMI_RANGE[0]:g}-{BMI_RANGE[1]:g} kg/m2", "rule")
    _check(ck, "DEV", "anthropometry rows without a BMI value (not usable)", int((~outside & np.isnan(b)).sum()))
    a = a.loc[keep, ["id", "date", "bmi", "source"]].reset_index(drop=True)
    a["bmi"] = b[keep]
    rank = _source_rank(a["source"])
    _check(ck, "DEV", "usable anthropometry rows whose source names a check-up record", int((rank == 0).sum()))
    _check(ck, "DEV", "usable anthropometry rows with another source", int((rank == 1).sum()))
    _check(ck, "RULE", "usable anthropometry rows that repeat a person and day (the check-up row is used)",
           int(a.duplicated(["id", "date"]).sum()), "check-up first", "rule")
    return a


def _source_rank(source):
    """0 where the source text of an anthropometry row names a check-up record, else 1 (check-up rows come first)."""
    s = pd.Series(np.asarray(source, dtype=object)).astype("string").str.lower()
    return np.where(s.str.contains("|".join(ANTHRO_CHECKUP_WORDS), regex=True, na=False).to_numpy(bool), 0, 1)


def bmi_as_of(ids, days, anthro, lookback=BMI_LOOKBACK_DAYS):
    """(bmi, lag, source), arrays over the index persons: the BMI of the index day, else the latest of the days from
    `lookback` days before it to the day before (both ends included), never one dated after the index day. Several rows
    for one person and day: the check-up row first, then file order. NaN, NaN and None where no row qualifies. lag is
    the number of days from the BMI record to the index day. anthro holds usable rows (id, date, bmi, source)."""
    idx = pd.DataFrame({"id": np.asarray(ids).astype(np.int64), "_i": np.arange(len(ids)),
                        "index_day": pd.to_datetime(np.asarray(days)).astype("datetime64[ns]")})
    a = anthro[["id", "date", "bmi", "source"]].copy()
    a["id"] = a["id"].astype(np.int64)
    a["date"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    a["_rank"] = _source_rank(a["source"])
    m = idx.merge(a, on="id", how="inner")
    gap = (m["index_day"] - m["date"]).dt.days.to_numpy(float)
    in_window = _between(gap, 0, lookback)
    hit = m[in_window].assign(_lag=gap[in_window]).sort_values(["_i", "_lag", "_rank"], kind="mergesort")
    hit = hit.drop_duplicates("_i", keep="first")
    n = len(idx)
    bmi, lag = np.full(n, np.nan), np.full(n, np.nan)
    source = np.full(n, None, dtype=object)
    i = hit["_i"].to_numpy(int)
    bmi[i], lag[i] = hit["bmi"].to_numpy(float), hit["_lag"].to_numpy(float)
    source[i] = hit["source"].to_numpy(object)
    bad = int(((lag < 0) | (lag > lookback)).sum())
    assert bad == 0, f"{runlog.count_text(bad)} BMI values dated after the index day or before the window"
    return bmi, lag, source


def _cmp_counts(ck, item, value, cmp, mask):
    c = pd.Series(_texts(cmp, len(mask))[mask & ~np.isnan(value)])
    for sign in CMP_SIGNS:
        _check(ck, "DEV", f"{item} comparator '{sign}'", int((c == sign).sum()))
    _check(ck, "DEV", f"{item} comparator other", int((~c.isin(CMP_SIGNS)).sum()))


def dev_cohort(checks=None):
    """(dev, flow_rows, check_rows, adults): the development cohort (cache columns except role) and its table rows;
    adults is the frame of the persons at the dev_adult step (inputs, has_bmi, in_alt, in_ast, enzyme values and
    comparators, checkup_day, urine_year), kept in memory for the descriptive contrasts of s01 and never written to
    cache. Check rows are appended to checks, when given, as they are made."""
    ck = [] if checks is None else checks
    assert GLU4_RULE in ("merge", "keep") and SEX_MISSING_RULE in ("exclude", "female"), "unknown rule setting"

    # processed file: one record per person (first row in file order) after the repeated-row check
    head = pd.read_csv(P.DATA_PATH, nrows=0, encoding="utf-8-sig").columns
    nearest = [c for c in PROC_NEAREST.values() if c in head]
    d = pd.read_csv(P.DATA_PATH, usecols=PROC_COLS + nearest, low_memory=False, encoding="utf-8-sig")
    pid = pd.to_numeric(d[COL_ID], errors="coerce")
    assert not (stop := _zero(ck, "DEV", "processed rows without a readable person number", pid.isna().sum())), stop
    d["id"] = pid.astype(np.int64)
    d["day"] = _day(d[COL_DAY])
    d["ck_day"] = _day(d[COL_CHECKUP])
    for c in DIPSTICK + ["age_at_urine_test", "days_difference"] + nearest:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["sex"] = d[COL_SEX].astype("string").str.strip()
    per = d["id"].value_counts()
    _check(ck, "DEV", "processed rows", len(d))
    _check(ck, "DEV", "persons", len(per))
    _check(ck, "DEV", "persons with repeated rows", int((per > 1).sum()))
    u = d[["id", "day", "age_at_urine_test", "sex"] + DIPSTICK].drop_duplicates()
    differ_ids = u.loc[u["id"].duplicated(), "id"].unique()
    u = d[["id", "ck_day", "days_difference"]].drop_duplicates()
    other_ids = u.loc[u["id"].duplicated(), "id"].unique()
    one = d.drop_duplicates("id", keep="first").reset_index(drop=True)
    del d, u
    n = len(one)
    ids = one["id"].to_numpy()
    day = one["day"].to_numpy()
    differ = np.isin(ids, differ_ids)
    _check(ck, "DEV", "persons whose repeated rows differ in inputs or urine day", int(differ.sum()))
    _check(ck, "DEV", "persons whose repeated rows differ in checkup date or days_difference",
           int(np.isin(ids, other_ids).sum()))

    # raw first-visit file
    raw = pd.read_csv(P.RAW_FIRST, usecols=[COL_ID, COL_RAW_TIME, "ua_nitrite"], dtype=str, encoding="utf-8-sig")
    rid = pd.to_numeric(raw[COL_ID].str.strip(), errors="coerce")
    assert not (stop := _zero(ck, "DEV", "raw first-visit rows without a readable person number",
                              rid.isna().sum())), stop
    raw = pd.DataFrame({"id": rid.astype(np.int64),
                        "raw_dt": pd.to_datetime(raw[COL_RAW_TIME], errors="coerce",
                                                 format="mixed").astype("datetime64[ns]"),
                        "nit_text": raw["ua_nitrite"]})
    assert not (stop := _zero(ck, "DEV", "raw first-visit person numbers repeated", raw["id"].duplicated().sum())), stop
    _check(ck, "DEV", "raw first-visit persons", len(raw))
    _check(ck, "DEV", "raw first-visit persons not in the processed file", int((~raw["id"].isin(ids)).sum()))
    j = one[["id"]].merge(raw, on="id", how="left", validate="one_to_one", indicator=True)
    has_raw = (j["_merge"] == "both").to_numpy()
    raw_dt = j["raw_dt"].to_numpy()
    raw_same = has_raw & (pd.Series(raw_dt).dt.normalize().to_numpy() == day)
    _check(ck, "DEV", "persons without a raw first-visit record", int((~has_raw).sum()))
    _check(ck, "DEV", "persons whose raw first-visit day differs from the urine day", int((has_raw & ~raw_same).sum()))

    # nitrite: processed value, else the raw text
    text = nitrite_from_text(j["nit_text"]).to_numpy(float)
    proc = one["ua_nitrite"].to_numpy(float)
    have = ~np.isnan(proc)
    nit_disagree = have & ~np.isnan(text) & (proc != text)
    nitrite = np.where(have, proc, text)
    for item, m in (("nitrite processed missing, text 1", ~have & (text == 1)),
                    ("nitrite processed missing, text 0", ~have & (text == 0)),
                    ("nitrite processed missing, text not readable", ~have & np.isnan(text)),
                    ("nitrite processed present, text not readable", have & np.isnan(text)),
                    ("nitrite processed and text disagree", nit_disagree)):
        _check(ck, "DEV", f"persons with {item}", int(m.sum()))
    del raw, j

    # atlas episodes: nitrite_pos at the raw first-visit time (cross-check only)
    ep, n_dup = _read_episodes()
    e = pd.DataFrame({"id": ids, "dt": raw_dt}).merge(ep, on=["id", "dt"], how="left", validate="many_to_one")
    atlas_nit = e["nitrite_pos"].to_numpy(float)
    del ep, e

    # atlas lab_day_wide on the urine day: blood-test day, ALT and AST, exact age, checkup flag
    lab = _read_lab(ids)
    k = pd.DataFrame({"id": ids, "date": day}).merge(lab, on=["id", "date"], how="left", validate="many_to_one",
                                                      indicator=True)
    del lab
    has_lab = (k["_merge"] == "both").to_numpy()
    alt_v = pd.to_numeric(k["alt"], errors="coerce").astype(float).to_numpy()
    ast_v = pd.to_numeric(k["ast"], errors="coerce").astype(float).to_numpy()
    has_alt = has_lab & ~np.isnan(alt_v)
    has_ast = has_lab & ~np.isnan(ast_v)
    alt_c = np.where(has_alt, _texts(k["alt_cmp"], n), None)
    ast_c = np.where(has_ast, _texts(k["ast_cmp"], n), None)
    _, bad_alt = label_parts(np.where(has_alt, alt_v, np.nan), alt_c, OUTCOMES["ALT"]["threshold"])
    _, bad_ast = label_parts(np.where(has_ast, ast_v, np.nan), ast_c, OUTCOMES["AST"]["threshold"])
    in_alt = has_alt & ~bad_alt
    in_ast = has_ast & ~bad_ast
    atlas_adult = k["adult"].astype("boolean")
    atlas_age = pd.to_numeric(k["age"], errors="coerce").astype(float).to_numpy()
    atlas_ck = k["is_checkup"].astype("boolean").fillna(False).to_numpy(bool)
    alt_nrep = pd.to_numeric(k["alt_nrep"], errors="coerce").to_numpy(float)
    ast_nrep = pd.to_numeric(k["ast_nrep"], errors="coerce").to_numpy(float)
    alt_last = pd.to_numeric(k["alt_last"], errors="coerce").astype(float).to_numpy()
    ast_last = pd.to_numeric(k["ast_last"], errors="coerce").astype(float).to_numpy()
    del k

    # health check-up on the urine day (the checkup date of the processed file)
    ck_day = one["ck_day"].to_numpy()
    checkup = ~np.isnat(ck_day) & (ck_day == day)

    # BMI: the urine day, else the latest of the 30 days before it, never after it (anthropometry table)
    bmi, bmi_lag, bmi_src = bmi_as_of(ids, day, _read_anthro(ids, ck))
    has_bmi = ~np.isnan(bmi)

    # inputs, with the two rules
    age = one["age_at_urine_test"].to_numpy(float)
    sex_mf = one["sex"].isin(["M", "F"]).fillna(False).to_numpy(bool)
    male = np.where(one["sex"].eq("M").fillna(False).to_numpy(bool), 1.0,
                    np.where(one["sex"].eq("F").fillna(False).to_numpy(bool), 0.0, np.nan))
    if SEX_MISSING_RULE == "female":
        male = np.where(np.isnan(male), 0.0, male)
    X = pd.DataFrame({c: one[c].to_numpy(float) for c in DIPSTICK})
    X["ua_nitrite"] = nitrite
    glu4 = (X["ua_glucose"] == 4).to_numpy()
    if GLU4_RULE == "merge":
        X.loc[glu4, "ua_glucose"] = 3.0
    X["age"] = age
    X["male"] = male
    X["bmi"] = bmi
    X = X[INPUTS]
    complete = X[DIPSTICK + ["age", "male"]].notna().all(axis=1).to_numpy()
    dip_complete = X[DIPSTICK].notna().all(axis=1).to_numpy()

    # flow
    f = _Flow("DEV")
    s1 = np.ones(n, bool)
    before, after, nodate = day < STUDY_START.to_datetime64(), day > STUDY_END.to_datetime64(), np.isnat(day)
    s2 = ~before & ~after & ~nodate
    s3 = s2 & has_lab
    s4 = s3 & is_adult(age)
    s5 = s4 & in_alt & in_ast
    s6 = s5 & complete
    cohort = s6 & has_bmi
    no_sex = s5 & np.isnan(male)
    f.step(1, "ALL", "dev_persons", "Persons with a first urinalysis and a blood test within 365 days", s1)
    f.step(2, "ALL", "dev_period", "Urine test from January 2010 to 14 August 2025", s2, "dev_persons")
    f.reasons(2, "ALL", "dev_period", [("dev_period_before", "Urine test before January 2010", before),
                                       ("dev_period_after", "Urine test after 14 August 2025", after),
                                       ("dev_period_nodate", "Urine-test date not readable", nodate)])
    f.step(3, "ALL", "dev_blood_day", "Blood test on the urine-test day", s3, "dev_period")
    f.step(4, "ALL", "dev_adult", "Aged 19-110 years at the urine test", s4, "dev_blood_day")
    f.reasons(4, "ALL", "dev_adult", [("dev_age_under19", "Aged under 19 years", s3 & (age < ADULT_AGE)),
                                      ("dev_age_over110", "Aged over 110 years", s3 & (age > AGE_MAX)),
                                      ("dev_age_missing", "Age not recorded", s3 & np.isnan(age))])
    f.step(5, "ALL", "dev_enzymes", "Same-day ALT and AST, both classifiable", s5, "dev_adult")
    f.reasons(5, "ALL", "dev_enzymes",
              [("dev_no_alt_ast", "Neither ALT nor AST on the urine-test day", s4 & ~has_alt & ~has_ast),
               ("dev_ast_only", "AST only (no ALT on the urine-test day)", s4 & ~has_alt & has_ast),
               ("dev_alt_only", "ALT only (no AST on the urine-test day)", s4 & has_alt & ~has_ast),
               ("dev_enzyme_undetermined", "ALT or AST censored and not classifiable",
                s4 & has_alt & has_ast & (bad_alt | bad_ast))])
    f.step(6, "ALL", "dev_inputs", "All dipstick fields and sex recorded", s6, "dev_enzymes")
    f.reasons(6, "ALL", "dev_inputs", [("dev_missing_sex", "Sex not recorded", no_sex),
                                       ("dev_missing_dipstick", "Dipstick field missing", s5 & ~no_sex & ~complete)])
    f.step(7, "ALL", "dev_bmi", "BMI on the urine-test day or in the 30 days before it (development cohort)", cohort,
           "dev_inputs")
    f.reasons(7, "ALL", "dev_bmi",
              [("dev_bmi_none", "No BMI on the urine-test day or in the 30 days before it", s6 & ~has_bmi)])
    for c in DIPSTICK + ["age", "male"]:
        _check(ck, "DEV", f"persons at dev_enzymes with {c} missing", int((s5 & X[c].isna()).sum()))
    _check(ck, "DEV", "persons at dev_enzymes with a dipstick field missing", int((s5 & ~dip_complete).sum()))
    _check(ck, "RULE", "ua_glucose 4 before the rule, persons at dev_enzymes", int((s5 & glu4).sum()), GLU4_RULE,
           "rule")
    _check(ck, "RULE", "sex not M or F before the rule, persons at dev_enzymes", int((s5 & ~sex_mf).sum()),
           SEX_MISSING_RULE, "rule")
    _check(ck, "RULE", "ua_glucose 4 before the rule, development cohort", int((cohort & glu4).sum()), GLU4_RULE,
           "rule")
    _check(ck, "RULE", "persons at dev_adult with only one of same-day ALT and AST (both are required)",
           int((s4 & (has_alt != has_ast)).sum()), "both required", "rule")
    full_inputs = s4 & complete & has_bmi            # adults with complete inputs and a BMI: the cost of requiring both
    _check(ck, "RULE", "adults with complete inputs and a BMI with ALT only",
           int((full_inputs & has_alt & ~has_ast).sum()), "both required", "rule")
    _check(ck, "RULE", "adults with complete inputs and a BMI with AST only",
           int((full_inputs & ~has_alt & has_ast).sum()), "both required", "rule")
    for label, m in (("both ALT and AST", has_alt & has_ast), ("ALT only", has_alt & ~has_ast),
                     ("AST only", ~has_alt & has_ast), ("neither ALT nor AST", ~has_alt & ~has_ast)):
        for when, day_mask in (("a check-up day", checkup), ("another day", ~checkup)):
            _check(ck, "DEV", f"persons at dev_adult with {label} on the urine day, {when}",
                   int((s4 & m & day_mask).sum()))

    # cross-checks over the persons of the study period, then within the development cohort
    dd = one["days_difference"].to_numpy(float)
    for item, m in (("lab row on the urine day and days_difference == 0", has_lab & (dd == 0)),
                    ("lab row on the urine day and days_difference != 0", has_lab & ~np.isnan(dd) & (dd != 0)),
                    ("no lab row on the urine day and days_difference == 0", ~has_lab & (dd == 0)),
                    ("no lab row on the urine day and days_difference != 0", ~has_lab & ~np.isnan(dd) & (dd != 0)),
                    ("days_difference missing", np.isnan(dd)),
                    ("same-day ALT or AST and days_difference != 0",
                     (has_alt | has_ast) & ~np.isnan(dd) & (dd != 0))):
        _check(ck, "DEV", f"persons in the study period with {item}", int((s2 & m).sum()))
    assert not (stop := _zero(ck, "DEV", "development cohort persons whose repeated rows differ in inputs or urine day",
                              (cohort & differ).sum())), stop
    assert not (stop := _zero(ck, "DEV", "development cohort persons without a raw first-visit record",
                              (cohort & ~has_raw).sum())), stop
    assert not (stop := _zero(ck, "DEV",
                              "development cohort persons whose raw first-visit day differs from the urine day",
                              (cohort & has_raw & ~raw_same).sum())), stop
    assert not (stop := _zero(ck, "DEV", "development cohort persons whose processed and text nitrite disagree",
                              (cohort & nit_disagree).sum())), stop
    for item, m in (("nitrite filled from the text as 1", ~have & (nitrite == 1)),
                    ("nitrite filled from the text as 0", ~have & (nitrite == 0)),
                    ("nitrite 1", nitrite == 1)):
        _check(ck, "DEV", f"development cohort persons with {item}", int((cohort & m).sum()))
    both = cohort & ~np.isnan(atlas_nit)
    _check(ck, "DEV", "atlas dipstick episodes repeated on (id, dt), dropped", n_dup)
    _check(ck, "DEV", "development cohort persons with an atlas nitrite_pos", int(both.sum()))
    _check(ck, "DEV", "development cohort persons whose nitrite equals the atlas nitrite_pos",
           int((both & (atlas_nit == nitrite)).sum()))
    _check(ck, "DEV", "development cohort persons whose nitrite differs from the atlas nitrite_pos",
           int((both & (atlas_nit != nitrite)).sum()))
    _check(ck, "DEV", "development cohort persons without an atlas nitrite_pos",
           int((cohort & np.isnan(atlas_nit)).sum()))
    assert not (stop := _zero(ck, "DEV", "development cohort persons who are minors by the atlas exact age",
                              (cohort & atlas_adult.eq(False).fillna(False).to_numpy(bool)).sum())), stop
    _check(ck, "DEV", "development cohort persons without an atlas exact age",
           int((cohort & atlas_adult.isna().to_numpy()).sum()))
    _check(ck, "DEV", "development cohort persons whose age differs from the atlas exact age by more than 1 year",
           int((cohort & (np.abs(age - atlas_age) > 1)).sum()))

    # same-day enzymes in the development cohort
    for name, v, c, m, nrep, last, has, bad in (
            ("ALT", alt_v, alt_c, cohort & in_alt, alt_nrep, alt_last, has_alt, bad_alt),
            ("AST", ast_v, ast_c, cohort & in_ast, ast_nrep, ast_last, has_ast, bad_ast)):
        cut = OUTCOMES[name]["threshold"]
        txt = _texts(c, n)
        eq = m & (txt == "=")
        _check(ck, "DEV", f"development cohort persons with a same-day {name}", int(m.sum()))
        _check(ck, "DEV", f"development cohort same-day {name} exactly 40", int((eq & (v == cut)).sum()))
        _check(ck, "DEV", f"development cohort same-day {name} above 40 and below 41",
               int((eq & (v > cut) & (v < cut + 1)).sum()))
        _check(ck, "DEV", f"development cohort same-day {name} not a whole number",
               int((eq & (v != np.floor(v))).sum()))
        _check(ck, "DEV", f"development cohort same-day {name} 1,000 U/L or more", int((m & (v >= 1000)).sum()))
        _cmp_counts(ck, f"development cohort same-day {name}", v, c, m)
        _check(ck, "DEV", f"development cohort same-day {name} from repeated draws (nrep > 1)",
               int((m & (nrep > 1)).sum()))
        _check(ck, "DEV", f"development cohort persons whose first and last same-day {name} draws differ on the "
               "above-40 cut", int((m & ~np.isnan(last) & ((v > cut) != (last > cut))).sum()))
        _check(ck, "DEV", f"persons at dev_adult with a same-day {name} that no rule can label", int((s4 & bad).sum()))
        here = s4 & has
        for text, mask in (("'=' above 40 (label 1)", (txt == "=") & (v > cut)),
                           ("'=' at or below 40 (label 0)", (txt == "=") & (v <= cut)),
                           ("'<' or '<=' with a limit at or below 40 (label 0)",
                            np.isin(txt, ["<", "<="]) & (v <= cut)),
                           ("'<' or '<=' with a limit above 40 (no label)", np.isin(txt, ["<", "<="]) & (v > cut)),
                           ("'>' with a limit at or above 40 (label 1)", (txt == ">") & (v >= cut)),
                           ("'>' with a limit below 40 (no label)", (txt == ">") & (v < cut)),
                           ("'>=' with a limit above 40 (label 1)", (txt == ">=") & (v > cut)),
                           ("'>=' with a limit at or below 40 (no label)", (txt == ">=") & (v <= cut)),
                           ("another comparator (no label)", ~np.isin(txt, list(CMP_SIGNS)))):
            _check(ck, "DEV", f"persons at dev_adult with a same-day {name} {text}", int((here & mask).sum()))
    assert not (stop := _zero(ck, "DEV", "development cohort same-day ALT values whose label cannot be determined",
                              (cohort & bad_alt).sum())), stop
    assert not (stop := _zero(ck, "DEV", "development cohort same-day AST values whose label cannot be determined",
                              (cohort & bad_ast).sum())), stop
    assert not (stop := _zero(ck, "DEV", "development cohort persons without a classifiable same-day ALT",
                              (cohort & ~in_alt).sum())), stop
    assert not (stop := _zero(ck, "DEV", "development cohort persons without a classifiable same-day AST",
                              (cohort & ~in_ast).sum())), stop
    for name, v in (("ALT", alt_v), ("AST", ast_v)):
        if PROC_NEAREST[name] not in one:
            _check(ck, "DEV", f"processed {name} column not in the file", None, "absent")
            continue
        pv = one[PROC_NEAREST[name]].to_numpy(float)
        m = cohort & (dd == 0) & ~np.isnan(pv) & ~np.isnan(v)
        eq = m & (np.abs(pv - v) < 1e-9)
        _check(ck, "DEV", f"development cohort processed {name} equal to the same-day value", int(eq.sum()))
        _check(ck, "DEV", f"development cohort processed {name} different from the same-day value",
               int((m & ~eq).sum()))
    assert not (stop := _zero(ck, "DEV", "persons with a BMI dated after the urine day or more than 30 days before it",
                              ((bmi_lag < 0) | (bmi_lag > BMI_LOOKBACK_DAYS)).sum())), stop
    named_checkup = has_bmi & (_source_rank(bmi_src) == 0)
    _check(ck, "DEV", "development cohort persons with a BMI on the urine day", int((cohort & (bmi_lag == 0)).sum()))
    _check(ck, "DEV", "development cohort persons with a BMI 1-30 days before the urine day",
           int((cohort & (bmi_lag > 0)).sum()))
    _check(ck, "DEV", "development cohort persons whose BMI comes from a record naming a check-up",
           int((cohort & named_checkup).sum()))
    _check(ck, "DEV", "development cohort persons whose BMI comes from another record",
           int((cohort & has_bmi & ~named_checkup).sum()))
    _check(ck, "DEV", "development cohort persons whose checkup date is the urine day", int((cohort & checkup).sum()))
    _check(ck, "DEV", "development cohort persons with atlas is_checkup on the urine day",
           int((cohort & atlas_ck).sum()))
    _check(ck, "DEV", "development cohort persons with a checkup date on the urine day and atlas is_checkup",
           int((cohort & checkup & atlas_ck).sum()))
    _check(ck, "DEV", "development cohort persons with a checkup date on the urine day only",
           int((cohort & checkup & ~atlas_ck).sum()))
    _check(ck, "DEV", "development cohort persons with atlas is_checkup only",
           int((cohort & ~checkup & atlas_ck).sum()))

    # the cohort frame, in processed-file order
    sel = np.flatnonzero(cohort)
    dev = pd.DataFrame({"id": ids[sel].astype(np.int64)})
    for c in INPUTS:
        dev[c] = X[c].to_numpy(float)[sel]
    dev["alt_value"] = alt_v[sel]
    dev["alt_cmp"] = pd.array(alt_c[sel], dtype="string")
    dev["in_alt"] = in_alt[sel]
    dev["ast_value"] = ast_v[sel]
    dev["ast_cmp"] = pd.array(ast_c[sel], dtype="string")
    dev["in_ast"] = in_ast[sel]
    dev["stratum"] = strata(dev["alt_value"], dev["alt_cmp"], dev["ast_value"], dev["ast_cmp"])
    dev["checkup_day"] = checkup[sel]
    dev["urine_year"] = pd.DatetimeIndex(day[sel]).year.astype(np.int16)
    assert dev["id"].is_unique, "development cohort has repeated persons"
    assert not (stop := _zero(ck, "DEV", "development cohort persons outside 19-110 years by age_at_urine_test",
                              (~is_adult(dev["age"])).sum())), stop
    for item, col, lo, hi in (("same-day ALT median (U/L)", "alt_value", 10, 40),
                              ("same-day AST median (U/L)", "ast_value", 10, 40),
                              ("BMI median (kg/m2)", "bmi", 20, 28)):
        assert not (stop := _unit(ck, f"development cohort {item}", dev[col].to_numpy(float), lo, hi)), stop

    # the persons at dev_adult, in processed-file order, for the descriptive contrasts of s01
    at = np.flatnonzero(s4)
    adults = pd.DataFrame({c: X[c].to_numpy(float)[at] for c in INPUTS})
    adults["has_bmi"] = has_bmi[at]
    adults["in_alt"] = in_alt[at]
    adults["alt_value"] = alt_v[at]
    adults["alt_cmp"] = pd.array(alt_c[at], dtype="string")
    adults["in_ast"] = in_ast[at]
    adults["ast_value"] = ast_v[at]
    adults["ast_cmp"] = pd.array(ast_c[at], dtype="string")
    adults["checkup_day"] = checkup[at]
    adults["urine_year"] = pd.DatetimeIndex(day[at]).year.astype(np.int16)
    return dev, f.rows, ck, adults


def _unit(ck, item, values, lo, hi):
    """Add a UNIT row (the median is written from 100 values on); return '' when the median is within lo-hi, else the
    stop message."""
    v = values[~np.isnan(values)]
    med = float(np.median(v)) if len(v) else np.nan
    _check(ck, "UNIT", item, len(v), round(med, 2) if len(v) >= 100 else None, f"asserted {lo}-{hi}")
    return "" if len(v) and lo <= med <= hi else f"{item}: outside {lo}-{hi} ({runlog.count_text(len(v))} values)"


def dev_final_rows(dev):
    """The final cohort_flow.csv rows of the development data (needs the role column): one row per outcome and role,
    n the same for both outcomes, events by outcome; parent dev_bmi."""
    f = _Flow("DEV")
    train = (dev["role"] == "DEV_TRAIN").to_numpy(bool)
    for sample, o in OUTCOMES.items():
        m = dev[o["sample"]].to_numpy(bool)
        y = np.zeros(len(dev))
        y[m] = outcome_label(dev[o["value"]].to_numpy()[m], dev[o["cmp"]].to_numpy(object)[m], o["threshold"])
        for role, rm, label in (("DEV_TRAIN", train, "Training set"), ("DEV_TEST", ~train, "Test set")):
            sel = m & rm
            f.final(8, sample, f"{sample}_{role}", label, "dev_bmi", sel.sum(), y[sel].sum())
    return f.rows


def person_split(dev, seed=SEED):
    """Role of each person (DEV_TRAIN or DEV_TEST): 20 % of the cohort to testing, stratified on stratum, seed 42."""
    from sklearn.model_selection import train_test_split
    st = dev["stratum"].to_numpy()
    counts = np.bincount(st.astype(int), minlength=4)
    small = int(((counts > 0) & (counts < 2)).sum())
    assert small == 0, f"{small:,} split strata hold a single person"
    _, te = train_test_split(np.arange(len(dev)), test_size=0.2, random_state=seed, stratify=st)
    role = np.full(len(dev), "DEV_TRAIN", dtype=object)
    role[te] = "DEV_TEST"
    return role


# ------------------------------------------------------------------------------------------------ KNHANES
def _coding_rows(cod, section, year, variable, target, codes, maps_to, adult):
    g = pd.DataFrame({"code": np.asarray(codes, dtype=object), "maps_to": np.asarray(maps_to, dtype=object),
                      "adult": np.asarray(adult, bool)})
    t = g.groupby(["code", "maps_to"], sort=False).agg(n_records=("adult", "size"), n_adults=("adult", "sum"))
    t = t.reset_index()
    t["_k"] = pd.to_numeric(t["code"], errors="coerce")
    for _, r in t.sort_values(["_k", "code"], na_position="last").iterrows():
        cod.append(dict(section=section, year=year, variable=variable, target=target, code=r["code"],
                        maps_to=r["maps_to"], n_records=int(r["n_records"]), n_adults=int(r["n_adults"])))


def kn_cohort(table, checks=None, coding=None):
    """(kn, flow_rows, check_rows, coding_rows): eligible KNHANES adults (cache columns, plus the raw dipstick codes
    he_*, which write_cache drops) and their table rows; checks and coding, when given, are filled as they are made."""
    ck = [] if checks is None else checks
    cod = [] if coding is None else coding
    missing_listed = missing_base = 0
    for year in P.KN_YEARS:
        have = {c.lower() for c in kn_names(year)}
        missing_listed += len(set(table.loc[table["year"] == year, "var"]) - have)
        missing_base += len(set(KN_BASE) - have)
    assert not (stop := _zero(ck, "KN", "code-table variables missing from a survey file (variable x year)",
                              missing_listed)), stop
    assert not (stop := _zero(ck, "KN",
                              "input, outcome or design variables missing from a survey file (variable x year)",
                              missing_base)), stop

    parts = []
    unlisted = unlisted_codes = 0
    for year in P.KN_YEARS:
        tv = table[table["year"] == year]
        df = read_knhanes_raw(year, KN_BASE + list(dict.fromkeys(tv["var"])))
        names = df.attrs.get("names", {})
        adult = (pd.to_numeric(df["age"], errors="coerce") >= ADULT_AGE).to_numpy()
        _check(ck, "KN", f"survey records {year}", len(df))
        for role in ROLES:
            got = []
            for var in dict.fromkeys(tv.loc[tv["role"] == role, "var"]):
                codes, meaning = code_meanings(df[var], tv[tv["var"] == var])
                miss = meaning.isna().to_numpy()
                unlisted += int(miss.sum())
                unlisted_codes += int(codes[miss].nunique())
                _coding_rows(cod, "alcohol" if role == "alcohol" else "eligibility", year, names.get(var, var), role,
                             codes, meaning.fillna(NOT_LISTED), adult)
                got.append(meaning.to_numpy(object))
            if role == "pregnancy":
                df["pregnancy"] = combine_meanings(got, len(df))
            else:
                df["drink_group"] = got[0]         # one alcohol variable per year (load_code_table)
        mapped = knhanes_dipstick_map(df)
        for c in DIPSTICK:
            rawv = KN_RAW[c]
            maps = code_texts(mapped[c]).replace("blank", "missing")
            _coding_rows(cod, "dipstick", year, names.get(rawv, rawv), c, code_texts(df[rawv]), maps, adult)
        parts.append(df[KN_BASE + ["year", "pregnancy", "drink_group"]])
    _check(ck, "KN", "survey file reader", value=_kn_reader(P.KN_FILES[P.KN_YEARS[0]])[2])
    _check(ck, "KN", "survey records whose file year column differs from the file's year", 0, note="asserted 0")
    _check(ck, "KN", "distinct pregnancy or alcohol codes not in the code table", unlisted_codes)
    assert not (stop := _zero(ck, "KN", "survey records with a pregnancy or alcohol code not in the code table",
                              unlisted)), stop

    r = pd.concat(parts, ignore_index=True)
    nrec = len(r)
    age = pd.to_numeric(r["age"], errors="coerce").astype(float).to_numpy()
    sexv = pd.to_numeric(r["sex"], errors="coerce").astype(float).to_numpy()
    male = np.where(sexv == 1, 1.0, np.where(sexv == 2, 0.0, np.nan))
    ht = pd.to_numeric(r["he_ht"], errors="coerce").astype(float).to_numpy()
    wt = pd.to_numeric(r["he_wt"], errors="coerce").astype(float).to_numpy()
    he_bmi = pd.to_numeric(r["he_bmi"], errors="coerce").astype(float).to_numpy()
    he_alt = pd.to_numeric(r["he_alt"], errors="coerce").astype(float).to_numpy()
    he_ast = pd.to_numeric(r["he_ast"], errors="coerce").astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        bmi = wt / (ht / 100.0) ** 2                  # a height of 0 gives inf, which the BMI range removes
    X = knhanes_dipstick_map(r)
    X["age"] = age
    X["male"] = male
    X["bmi"] = bmi
    X = X[INPUTS]
    _check(ck, "KN", "survey records with age missing", int(np.isnan(age).sum()))
    _check(ck, "KN", "survey records with a sex code other than 1 or 2", int(np.isnan(male).sum()))
    for c in DIPSTICK[:7]:
        rawv = pd.to_numeric(r[KN_RAW[c]], errors="coerce").to_numpy(float)
        _check(ck, "KN", f"survey records with a {KN_RAW[c]} code outside the map (missing)",
               int((~np.isnan(rawv) & X[c].isna().to_numpy()).sum()))

    # flow: records -> adults -> inputs -> height and weight -> ALT and AST
    f = _Flow("KN")
    adult = age >= ADULT_AGE
    complete = X[DIPSTICK + ["age", "male"]].notna().all(axis=1).to_numpy()
    no_sex = adult & np.isnan(male)
    s3 = adult & complete
    has_anthro = ~np.isnan(ht) & ~np.isnan(wt)
    in_range = _between(ht, *HEIGHT_RANGE) & _between(wt, *WEIGHT_RANGE) & _between(bmi, *BMI_RANGE)
    s4 = s3 & in_range
    has_alt, has_ast = ~np.isnan(he_alt), ~np.isnan(he_ast)
    s5 = s4 & has_alt & has_ast
    f.step(1, "ALL", "kn_records", "KNHANES participants 2016-2024", np.ones(nrec, bool))
    f.step(2, "ALL", "kn_adult", "Aged 19 years or older", adult, "kn_records")
    f.reasons(2, "ALL", "kn_adult", [("kn_age_under19", "Aged under 19 years", age < ADULT_AGE),
                                     ("kn_age_missing", "Age not recorded", np.isnan(age))])
    f.step(3, "ALL", "kn_inputs", "All dipstick fields and sex recorded", s3, "kn_adult")
    f.reasons(3, "ALL", "kn_inputs", [("kn_missing_sex", "Sex not recorded", no_sex),
                                      ("kn_missing_dipstick", "Dipstick field missing", adult & ~no_sex & ~complete)])
    f.step(4, "ALL", "kn_anthro", "Height and weight recorded and within the ranges", s4, "kn_inputs")
    f.reasons(4, "ALL", "kn_anthro", [("kn_anthro_missing", "Height or weight not recorded", s3 & ~has_anthro),
                                      ("kn_anthro_range", "Height, weight or BMI outside the ranges",
                                       s3 & has_anthro & ~in_range)])
    f.step(5, "ALL", "kn_enzymes", "ALT and AST recorded", s5, "kn_anthro")
    f.reasons(5, "ALL", "kn_enzymes", [("kn_no_alt_ast", "Neither ALT nor AST recorded", s4 & ~has_alt & ~has_ast),
                                       ("kn_ast_only", "AST only (ALT not recorded)", s4 & ~has_alt & has_ast),
                                       ("kn_alt_only", "ALT only (AST not recorded)", s4 & has_alt & ~has_ast)])
    for name, v in (("ALT", he_alt), ("AST", he_ast)):
        y = outcome_label(v[s5], "=", OUTCOMES[name]["threshold"])
        f.final(6, name, f"{name}_KN", "Eligible adults", "kn_enzymes", s5.sum(), y.sum())
    _check(ck, "KN", "adults at kn_anthro with HE_alt only", int((s4 & has_alt & ~has_ast).sum()))
    _check(ck, "KN", "adults at kn_anthro with HE_ast only", int((s4 & ~has_alt & has_ast).sum()))
    _check(ck, "RULE", "adults at kn_anthro with only one of HE_alt and HE_ast (both are required)",
           int((s4 & (has_alt != has_ast)).sum()), "both required", "rule")
    for name, v in (("ALT", he_alt), ("AST", he_ast)):
        cut = OUTCOMES[name]["threshold"]
        _check(ck, "KN", f"eligible adults {name} exactly 40", int((s5 & (v == cut)).sum()))
        _check(ck, "KN", f"eligible adults {name} above 40 and below 41", int((s5 & (v > cut) & (v < cut + 1)).sum()))
        _check(ck, "KN", f"eligible adults {name} not a whole number", int((s5 & (v != np.floor(v))).sum()))
        _check(ck, "KN", f"eligible adults {name} 1,000 U/L or more", int((s5 & (v >= 1000)).sum()))
    gap = np.abs(bmi - he_bmi)
    _check(ck, "KN", "eligible adults with a recorded HE_BMI", int((s5 & ~np.isnan(he_bmi)).sum()))
    _check(ck, "KN", "eligible adults whose computed BMI is within 0.1 of HE_BMI", int((s5 & (gap <= 0.1)).sum()))
    _check(ck, "KN", "eligible adults whose computed BMI differs from HE_BMI by more than 0.5",
           int((s5 & (gap > 0.5)).sum()))
    preg = r["pregnancy"].to_numpy(object)
    _check(ck, "KN", "eligible adults with pregnancy yes (counted, not excluded)", int((s5 & (preg == "yes")).sum()))
    _check(ck, "KN", "eligible women with pregnancy status unknown (counted, not excluded)",
           int((s5 & (preg == "unknown") & (male == 0)).sum()))
    drink = r["drink_group"].to_numpy(object)
    for level in DRINK_LEVELS:
        _check(ck, "KN", f"eligible adults with drink_group {level}", int((s5 & (drink == level)).sum()))

    # eligible adults
    e = np.flatnonzero(s5)
    kn = pd.DataFrame({"id": r["id"].astype(str).to_numpy()[e], "year": r["year"].to_numpy()[e].astype(np.int16)})
    kn["period"] = kn_period(kn["year"])
    for c in INPUTS:
        kn[c] = X[c].to_numpy(float)[e]
    kn["alt_value"] = he_alt[e]
    kn["alt_cmp"] = pd.array(["="] * len(e), dtype="string")
    kn["in_alt"] = np.ones(len(e), bool)
    kn["ast_value"] = he_ast[e]
    kn["ast_cmp"] = pd.array(["="] * len(e), dtype="string")
    kn["in_ast"] = np.ones(len(e), bool)
    kn["drink_group"] = pd.array(drink[e], dtype="string")
    for c in ("wt_itvex", "psu", "kstrata"):
        kn[c] = pd.to_numeric(r[c], errors="coerce").astype(float).to_numpy()[e]
        _check(ck, "KN", f"eligible adults with {c} missing or not a number", int(kn[c].isna().sum()))
    _check(ck, "KN", "eligible adults with wt_itvex not above 0", int((kn["wt_itvex"] <= 0).sum()))
    for c in DIPSTICK:
        kn[KN_RAW[c]] = pd.to_numeric(r[KN_RAW[c]], errors="coerce").astype(float).to_numpy()[e]
    assert not (stop := _zero(ck, "KN", "eligible adults with a repeated survey id", kn["id"].duplicated().sum())), stop
    assert not (stop := _zero(ck, "KN", "eligible adults aged under 19 or without an age",
                              (~(kn["age"] >= ADULT_AGE)).sum())), stop
    assert not (stop := _zero(ck, "KN", "eligible adults with a he_unitr code other than 0 or 1",
                              (~kn["he_unitr"].isin([0, 1])).sum())), stop
    assert not (stop := _zero(ck, "KN", "eligible adults with a urine pH outside 4-10",
                              (~kn["ua_pH"].between(4, 10)).sum())), stop
    assert not (stop := _zero(ck, "KN", "eligible adults with a urine specific gravity outside 1.000-1.060",
                              (~kn["ua_sg"].between(1.0, 1.06)).sum())), stop
    assert not (stop := _zero(ck, "KN", "eligible adults with a drink_group outside the levels",
                              (~kn["drink_group"].isin(DRINK_LEVELS)).sum())), stop
    for item, col, lo, hi in (("ALT median (U/L)", "alt_value", 10, 40), ("AST median (U/L)", "ast_value", 10, 40),
                              ("BMI median (kg/m2)", "bmi", 20, 28)):
        assert not (stop := _unit(ck, f"eligible adults {item}", kn[col].to_numpy(float), lo, hi)), stop
    return kn, f.rows, ck, cod


# ------------------------------------------------------------------------------------------------ caches
def write_cache(dev, kn):
    """Write cache/dev.parquet and cache/knhanes.parquet with the columns DEV_COLUMNS and KN_COLUMNS."""
    os.makedirs(runlog.CACHE, exist_ok=True)
    for frame, cols, path in ((dev, DEV_COLUMNS, DEV_CACHE), (kn, KN_COLUMNS, KN_CACHE)):
        lost = [c for c in cols if c not in frame]
        assert not lost, f"{len(lost):,} cache column(s) missing"
        frame[cols].to_parquet(path, index=False)
    _MEMO.clear()


def read_cache():
    """(dev, kn) from cache/ (read once per process)."""
    if not _MEMO:
        _MEMO["dev"] = pd.read_parquet(DEV_CACHE)
        _MEMO["kn"] = pd.read_parquet(KN_CACHE)
    return _MEMO["dev"], _MEMO["kn"]


def outcome_frame(outcome, dataset):
    """(X, y, keys): the dataset's persons who have the outcome's lab, selected before labelling, in cache order;
    DEV_TRAIN in a seeded random order (RandomState(SEED)), so that the recipe's 5 calibration folds are random."""
    o = OUTCOMES[outcome]
    dev, kn = read_cache()
    if dataset in ("DEV_TRAIN", "DEV_TEST"):
        d = dev[((dev["role"] == dataset) & dev[o["sample"]]).to_numpy(bool)].reset_index(drop=True)
        if dataset == "DEV_TRAIN":      # CalibratedClassifierCV(cv=5) splits in row order, without shuffling
            d = d.iloc[np.random.RandomState(SEED).permutation(len(d))].reset_index(drop=True)
        extra = ["role", "checkup_day", "urine_year"]
    else:
        assert dataset == "KN", "unknown dataset"
        d = kn[kn[o["sample"]].to_numpy(bool)].reset_index(drop=True)
        extra = ["year", "period", "drink_group", "wt_itvex", "psu", "kstrata"]
    y = outcome_label(d[o["value"]], d[o["cmp"]], o["threshold"])
    miss = int(np.isnan(y).sum())
    assert miss == 0, f"{runlog.count_text(miss)} persons of an outcome sample without a label"
    X = d[INPUTS].astype(float)
    keys = pd.DataFrame({"row": np.arange(len(d), dtype=np.int64), "id": d["id"].to_numpy(),
                         "age_band": age_band(d["age"].to_numpy(float)),
                         "sex": np.where(d["male"].to_numpy(float) == 1, "Men", "Women")})
    for c in extra:
        keys[c] = d[c].to_numpy()
    return X, pd.Series(y.astype(np.int64), name="y"), keys


def train_xy(outcome):
    """(X, y) of DEV_TRAIN for one outcome, exactly as passed to the recipe."""
    X, y, _ = outcome_frame(outcome, "DEV_TRAIN")
    return X, y


def check_counts(outcome, dataset, n, events):
    """Assert that n and events equal the cohort_flow.csv row <OUTCOME>_<DATASET> (a '<10' cell means 1-9)."""
    f = pd.read_csv(os.path.join(runlog.HERE, "cohort_flow.csv"), dtype=str, keep_default_na=False)
    row = f[(f["kind"] == "final") & (f["key"] == f"{outcome}_{dataset}")]
    assert len(row) == 1, f"cohort_flow.csv has {len(row):,} rows for one outcome and dataset"
    for col, val in (("n", int(n)), ("events", int(events))):
        cell = row[col].iloc[0]
        ok = 1 <= val <= 9 if cell == "<10" else (cell != "" and int(float(cell)) == val)
        assert ok, f"{outcome} {dataset}: {col} differs from cohort_flow.csv"


def cal_folds(model, X, y):
    """The (fit rows, calibration rows) of each of the model's 5 folds, from StratifiedKFold(5) over X in row order
    as the calibrated model split it; each fold's isotonic map is fitted again and must equal the model's."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold
    yv = np.asarray(y)
    folds = list(StratifiedKFold(n_splits=5).split(X, yv))
    assert len(folds) == len(model.calibrated_classifiers_), "calibration fold count differs from the model"
    for k, (_, cal) in enumerate(folds):
        cc = model.calibrated_classifiers_[k]
        iso = IsotonicRegression(out_of_bounds="clip").fit(cc.estimator.predict_proba(X.iloc[cal])[:, 1], yv[cal])
        a, b = iso.y_thresholds_, cc.calibrators[0].y_thresholds_
        assert len(a) == len(b) and np.allclose(a, b), "a calibration fold differs from the model"
    return folds


# ------------------------------------------------------------------------------------------------ models, predictions
def model_path(model, outcome):
    assert model in MODELS and outcome in OUTCOMES, "unknown model or outcome"
    name = f"ens_{outcome}.joblib" if model == "ENS" else f"lr_{model}_{outcome}.joblib"
    return os.path.join(P.MODEL_DIR, name)


def load_model(model, outcome):
    import joblib
    return joblib.load(model_path(model, outcome))


def rule_flag(rule, X):
    """0/1 alert of a rule: X[col] >= cut."""
    col, cut = RULES[rule]
    return (X[col].to_numpy(float) >= cut).astype(np.int8)


def preds_path(model, outcome, dataset):
    assert model in MODELS and outcome in OUTCOMES and dataset in PRED_DATASETS, "unknown prediction file"
    return os.path.join(runlog.CACHE, f"preds_{model}_{outcome}_{dataset}.parquet")


def preds_sha256(y, p):
    """SHA-256 of p (little-endian float64 bytes) followed by y (int8 bytes)."""
    pb = np.ascontiguousarray(np.asarray(p, dtype="<f8")).tobytes()
    yb = np.ascontiguousarray(np.asarray(y).astype(np.int8)).tobytes()
    return hashlib.sha256(pb + yb).hexdigest()


def write_preds(model, outcome, dataset, keys, y, p):
    """Write one prediction file (row, id, y, p) and return its preds_sha256."""
    yv = np.asarray(y).astype(np.int8)
    pv = np.asarray(p, dtype=float)
    assert len(keys) == len(yv) == len(pv), "prediction, label and key lengths differ"
    out = pd.DataFrame({"row": keys["row"].to_numpy(np.int64), "id": keys["id"].to_numpy(), "y": yv, "p": pv})
    out.to_parquet(preds_path(model, outcome, dataset), index=False)
    return preds_sha256(yv, pv)


def read_preds(model, outcome, dataset):
    """The (row, id, y, p) frame of one prediction file; attrs["preds_sha256"] holds its recomputed hash."""
    out = pd.read_parquet(preds_path(model, outcome, dataset))
    assert list(out.columns) == ["row", "id", "y", "p"], "prediction file columns differ from row, id, y, p"
    assert (out["row"].to_numpy() == np.arange(len(out))).all(), "prediction rows are not in outcome_frame order"
    out.attrs["preds_sha256"] = preds_sha256(out["y"], out["p"])
    return out


# ------------------------------------------------------------------------------------------------ checks of s02-s07
MIN_CLASS = 10              # fewest events and non-events of an evaluated dataset


def file_sha256(path):
    """SHA-256 of a file."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def array_sha256(a, dtype):
    """SHA-256 of an array as contiguous bytes of dtype (row-major for a matrix)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes()).hexdigest()


def train_hashes(X, y, keys):
    """SHA-256 of the training inputs (float64, row-major), labels (int8) and ids (int64), as in train_summary.csv."""
    return dict(sha256_inputs=array_sha256(X.to_numpy(), "<f8"), sha256_labels=array_sha256(y.to_numpy(), "i1"),
                sha256_ids=array_sha256(keys["id"].to_numpy(), "<i8"))


def remove_files(paths):
    """Delete those of the files that exist."""
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def read_table(name):
    """An aggregate table of the working folder with every cell as text."""
    return pd.read_csv(os.path.join(runlog.HERE, name), dtype=str, keep_default_na=False)


def summary_row(summary, outcome):
    """The train_summary.csv row of one outcome, written by a run with the same SYNTHETIC setting."""
    row = summary[summary["outcome"] == outcome]
    assert len(row) == 1, f"train_summary.csv has {len(row):,} rows for one outcome"
    row = row.iloc[0]
    assert row["synthetic"] == str(bool(P.SYNTHETIC)), "train_summary.csv and this run differ in SYNTHETIC"
    return row


def check_training(row, X, y, keys):
    """Assert that the training persons have the SHA-256 values of their train_summary.csv row."""
    now = train_hashes(X, y, keys)
    off = sum(now[k] != row[k] for k in now)
    assert off == 0, f"{off:,} of the training inputs, labels and ids differ from train_summary.csv; fit with s02 again"


def load_ensemble(outcome, row):
    """The calibrated ensemble of one outcome, once its file has the SHA-256 of its train_summary.csv row; classes 0/1
    and inputs INPUTS."""
    assert file_sha256(model_path("ENS", outcome)) == row["sha256_model"], \
        "the ensemble file differs from train_summary.csv"
    model = load_model("ENS", outcome)
    assert type(model).__name__ == "CalibratedClassifierCV", "the ensemble file holds another kind of model"
    assert [int(c) for c in model.classes_] == [0, 1], "the ensemble's classes are not 0 and 1"
    assert list(model.feature_names_in_) == INPUTS, "the ensemble's inputs differ from INPUTS"
    return model


def checked_frame(outcome, dataset, min_class=MIN_CLASS):
    """(X, y, keys) of outcome_frame, once inputs are complete, labels 0/1, n and events as in cohort_flow.csv, and
    events and non-events at least min_class."""
    X, y, keys = outcome_frame(outcome, dataset)
    n, events = len(y), int(y.sum())
    where = f"{outcome} {dataset}:"
    miss = int(X.isna().sum().sum())
    assert miss == 0, f"{where} {runlog.count_text(miss)} missing inputs"
    bad = int((~y.isin([0, 1])).sum())
    assert bad == 0, f"{where} {runlog.count_text(bad)} labels are not 0 or 1"
    check_counts(outcome, dataset, n, events)
    assert min(events, n - events) >= min_class, f"{where} fewer than {min_class:,} events or non-events"
    return X, y, keys


def recorded_sha(model, outcome, dataset, perf, comp=None):
    """preds_sha256 of a model's evaluated predictions: performance.csv (ENS) or model_comparison.csv (regressions)."""
    if model == "ENS":
        t, name = perf, "performance.csv"
    else:
        t, name = comp[comp["row_type"] == "model"], "model_comparison.csv"
    t = t[(t["outcome"] == outcome) & (t["dataset"] == dataset) & (t["model"] == model)]
    assert len(t) == 1, f"{name} has {len(t):,} rows for one model, outcome and dataset"
    return t["preds_sha256"].iloc[0]


def checked_preds(model, outcome, dataset, keys, y, recorded):
    """(p, sha) of one prediction file, once it holds the persons and labels of keys and y, its hash is recorded and
    every prediction lies within 0-1."""
    f = read_preds(model, outcome, dataset)
    where = f"{outcome} {dataset} {model}:"
    same = (len(f) == len(keys) and np.array_equal(f["id"].to_numpy(), keys["id"].to_numpy())
            and np.array_equal(f["y"].to_numpy(np.int64), np.asarray(y).astype(np.int64)))
    assert same, f"{where} the predictions belong to other persons or labels; run the script that wrote them again"
    sha = f.attrs["preds_sha256"]
    assert sha == recorded, f"{where} the prediction file differs from the one that was evaluated"
    p = f["p"].to_numpy(float)
    bad = int((~np.isfinite(p) | (p < 0) | (p > 1)).sum())
    assert bad == 0, f"{where} {runlog.count_text(bad)} predictions missing or outside 0-1"
    return p, sha
