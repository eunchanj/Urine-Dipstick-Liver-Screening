"""Shared display helpers for s08-s11: roots, aggregate readers, number formats, labels, small-cell checks, the text and
word checks, markdown tables, and table and figure fragments with their index files.

Every function is pure except the explicit writers write_text(), write_fragment() and reset_index(), and
word_patterns(), which imports the optional local file text_patterns.py on first use (or says once that the word
checks are skipped).
Importing this module imports aggregate_guard, but every script still imports aggregate_guard as its first
statement.

Roots (taken from aggregate_guard)
  AGG_DIR   $LIV_AGG_DIR, or working/: the aggregate tables of AGGREGATES (read_table() refuses any other name)
  OUT_ROOT  $LIV_OUT_ROOT, or the folder above working/: outputs go to OUT_ROOT/manuscript, figures, supplement
  LOG_DIR   where aggregate_guard writes opened_<script>.txt
  To use other folders, set LIV_AGG_DIR and LIV_OUT_ROOT.

Cells
  read_table() keeps every cell as text: a blank cell becomes None and a masked count stays the text '<10' (a count
  of 1-9). It refuses a table whose cells break the cell format (cell_problems: a count cell other than blank, '<10',
  0 or a whole number of 10 or more; '<10' outside the count columns of AGGREGATES; a word in a column of numbers),
  naming the table, the column and the data row. num() turns a cell into a float (None when blank) and raises
  MaskedValueError on '<10'; count() gives an int, '<10' or None. Both kinds of unknown cell are shown as NA or
  '<10', never as zero.

Number formats (ASCII apart from the × of a small P value; rounding is half up on the shortest decimal text of the
value; commas only in numbers of five or more digits)
  fmt(0.36712, 3) -> '0.367'          fmt_ci(0.367, 0.341, 0.392) -> '0.367 (0.341, 0.392)'
  fmt_ci(-2.46, -2.58, -2.33, 2) -> '-2.46 (-2.58, -2.33)'     fmt_ci(0.027, 0.011, 0.039, signed=True) ->
  '+0.027 (0.011, 0.039)' (a CI is always 'estimate (lower, upper)': a comma and a space, never a hyphen or 'to')
  median_iqr(11.5, 10.5, 12.5) -> '11.5 [10.5, 12.5]' (the row says MEDIAN_LABEL)
  fmt_p(0.0473) -> '0.047'   fmt_p(3.5e-08) -> '3.5×10^-8^' (below 0.001: exact, two significant figures, with the
  superscript of the text build; a P of 0, below what the computation can hold, is '<1×10^-300^')
  fmt_pct(0.4567) -> '45.7' (a proportion shown as a percentage with one decimal; no % sign: accuracy measures,
  prevalence, mean predicted risk)     fmt_pct_ci(0.452, 0.436, 0.474) -> '45.2 (43.6, 47.4)'
  fmt_count(12345) -> '12,345'   fmt_count(4500) -> '4500'   fmt_count('<10') -> '<10'   fmt_count(5) -> '<10'
  fmt_n_share(2946, 100000, '2.95') -> '2946 (2.9)' and fmt_n_share(36000, 300000) -> '36,000 (12)': a percentage
  that describes people (fmt_people_pct) is a whole number from 10 up and has one decimal under 10, tested on the value
  rounded to one decimal (9.96 -> '10'), and one that would round to 100 without being 100 keeps one decimal ('99.9').
  It is rounded once from the counts; the two-decimal pct cell is only checked against it (rounded again it would give
  3.0). A share above 0 that rounds to 0 is '<0.1'.
  NA ('NA') for a blank cell; '<10' passes through fmt(), fmt_ci(), fmt_pct() and fmt_count(), while fmt_p() and
  fmt_pct_ci() raise on it (read_table() keeps '<10' to the count columns).

Small cells (simple suppression; within one table or one figure panel no count of 1-9 may follow)
  masked_sum_range(total, shown, k) and masked_follow(k, lo, hi): can the k masked cells of a block be worked out
  from its shown total; panel_heights(cells, total): bar heights of a figure panel rounded to 2 significant figures
  (sig_round, sig_interval), after that check.

Text and word checks
  text_problems(s): dashes, non-ASCII signs, 'matched', American spellings, claims that no displayed string may make,
  a comma in a number of fewer than five digits, a number not followed by one space before its unit, and a panel
  reference in parentheses. forbidden_problems(text): the word lists of text_patterns.py (a number alone in a table
  cell of numbers is data); skipped with a one-line message when that local file is missing. check_legend(s): a title,
  legend or note does not refer to the text. md_table checks every cell; write_fragment the title, body and notes;
  write_text every .md and .txt output.

Units: ALT and AST are in U/L and BMI in kg/m² as stored; nothing is converted, and a number and its unit are written
with one space (40 U/L, 25 kg/m²).

Index formats (UTF-8 CSV, header as below, one row per fragment, rows sorted by order)
  (i)  tables_index.csv, one per target folder, written by s08 through write_fragment(kind="table"):
         OUT_ROOT/manuscript/tables/tables_index.csv (target main), OUT_ROOT/supplement/tables/tables_index.csv (si)
       columns id, target, order, title, file, sources:
         id       stable key, [a-z][a-z0-9_]*; the fragment file is <id>.md in the same folder
         target   'main' or 'si' (the same in every row of one file)
         order    table number within the target (1, 2, ...; unique in the file): label 'Table <order>' (main) or
                  'Supplementary Table S<order>' (si)
         title    caption text without the label
         file     fragment file name relative to the index folder (<id>.md)
         sources  aggregate tables the numbers come from, ';'-separated names of AGGREGATES
  (ii) figures_index.csv, one per target folder, written by s09 through write_fragment(kind="figure"):
         OUT_ROOT/figures/figures_index.csv (main), OUT_ROOT/supplement/figures/figures_index.csv (si)
       columns id, target, order, title, legend, files, sources:
         id, target, order, title, sources as in (i); label 'Figure <order>' or 'Supplementary Figure S<order>'
         legend   the legend after the title, one paragraph (the legend text and its notes joined by spaces)
         files    the four image files relative to the index folder, ';'-separated in the order pdf;png;eps;tif
                  (<id>.pdf;<id>.png;<id>.eps;<id>.tif, as written by figstyle.save_all(fig, figure_base(id, target)));
                  a figure saved as separate parts (FIGURE_PARTS, empty in this paper) lists the four files of each
                  part in turn, under one row and one legend (<id>_a.pdf;...;<id>_a.tif;<id>_b.pdf;...;<id>_b.tif,
                  from figure_base(id, target, part))
       The legend fragment <id>.md and the visible strings <id>_text.txt (one <id>_<part>_text.txt per part) lie next
       to the images.
  Fragment file <id>.md: line 1 '**<label>.** <title>', a blank line, the body (tables: the markdown table; figures:
  the legend text), then each note as its own paragraph. No source names or comments, so that s10 can place it verbatim.
  reset_index(kind, target, ids) removes an index file and every output of its ids and of `ids` at the start of a
  run, so that a run that stops leaves no out-of-date fragment, image or index.
"""
import csv
import io
import json
import os
import re
import sys
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import aggregate_guard as _guard

WK, ROOT = _guard.WK, _guard.ROOT
AGG_DIR, OUT_ROOT, LOG_DIR = _guard.AGG_DIR, _guard.OUT_ROOT, _guard.LOG_DIR
OUTPUT_DIRS = _guard.OUTPUT_DIRS
SCRIPT = _guard.SCRIPT

MASKED = "<10"
NA = "NA"
GE, LE, TIMES = "\u2265", "\u2264", "\u00d7"
BMI_UNIT = "kg/m\u00b2"             # the unit of BMI wherever it is shown

# ------------------------------------------------------------------------------------------------ aggregate tables
# name: (writer, count columns, header or None)
AGGREGATES = {
    "environment.txt": ("s00", (), None),
    "knhanes_variables.csv": ("s00", ("file_rows",), ("year", "variable", "label", "file_rows")),
    "knhanes_values.csv": ("s00", ("n",), ("year", "variable", "value", "n")),
    "knhanes_numeric.csv": ("s00", ("n", "n_missing", "n_nonnumeric"), (
        "year", "variable", "n", "n_missing", "n_nonnumeric", "p01", "p50", "p99")),
    "dev_schema.csv": ("s00", (), ("file", "column", "dtype")),
    "dev_values.csv": ("s00", ("n",), ("file", "variable", "value", "n")),
    "dev_numeric.csv": ("s00", ("n", "n_missing", "n_nonnumeric"), (
        "file", "variable", "n", "n_missing", "n_nonnumeric", "p01", "p50", "p99")),
    "data_checks.csv": ("s00", ("n",), ("source", "item", "n", "note")),
    "cohort_flow.csv": ("s01", ("n", "excluded", "events"), (
        "source", "sample", "order", "key", "parent", "kind", "label", "n", "excluded", "events")),
    "cohort_checks.csv": ("s01", ("n",), ("source", "item", "n", "value", "note")),
    "baseline_characteristics.csv": ("s01", ("n",), (
        "sample", "dataset", "variable", "level", "n", "pct", "mean", "sd", "median", "q1", "q3", "unit")),
    "input_levels.csv": ("s01", ("n",), ("source", "input", "level", "n", "pct", "kn_codes")),
    "knhanes_coding.csv": ("s01", ("n_records", "n_adults"), (
        "section", "year", "variable", "target", "code", "maps_to", "n_records", "n_adults")),
    "enzyme_by_year.csv": ("s01", ("n", "events"), (
        "source", "enzyme", "year", "checkup_day", "n", "events", "prop_above_40", "median")),
    "hyperparameters.csv": ("s02", (), ("outcome", "learner", "param", "value", "tuned", "search")),
    "train_summary.csv": ("s02", ("n_train", "events_train", "calibration_fold_events_min"), (
        "outcome", "n_train", "events_train", "prevalence_train", "learners", "trials_per_learner", "tuning_folds",
        "seed", "sampler", "voting", "calibration", "calibration_folds", "class_weighting",
        "cv_auroc_lgb", "cv_auroc_xgb", "cv_auroc_cat", "calibration_fold_events_min", "fit_seconds", "model_file",
        "sha256_model", "sha256_inputs", "sha256_labels", "sha256_ids", "sha256_cohort_py", "sha256_recipe_py",
        "sha256_s02", "versions", "synthetic")),
    "performance.csv": ("s03", ("n", "events"), (
        "outcome", "dataset", "model", "n", "events", "prevalence", "mean_pred", "auroc", "auroc_lo", "auroc_hi",
        "auprc", "auprc_lo", "auprc_hi", "n_boot", "preds_sha256")),
    "calibration.csv": ("s03", ("n", "events", "n_p0", "events_p0", "n_p1", "nonevents_p1"), (
        "outcome", "dataset", "model", "n", "events", "prevalence", "mean_pred", "oe", "oe_lo", "oe_hi", "citl",
        "citl_lo", "citl_hi", "slope", "slope_lo", "slope_hi", "brier", "brier_lo", "brier_hi", "scaled_brier",
        "scaled_brier_lo", "scaled_brier_hi", "n_p0", "events_p0", "n_p1", "nonevents_p1", "slope_excl01",
        "slope_excl01_lo", "slope_excl01_hi", "n_boot")),
    "calibration_bins.csv": ("s03", ("n", "events"), (
        "outcome", "dataset", "model", "bins_requested", "bin", "n", "events", "mean_pred", "observed", "observed_lo",
        "observed_hi")),
    "prediction_histogram.csv": ("s03", ("n", "events"), (
        "outcome", "dataset", "model", "bin_lo", "bin_hi", "n", "events")),
    "roc_points.csv": ("s03", (), ("outcome", "dataset", "model", "fpr", "tpr")),
    "model_comparison.csv": ("s04", ("n", "events"), (
        "row_type", "outcome", "dataset", "model", "model_b", "n", "events", "auroc", "auroc_lo", "auroc_hi", "auprc",
        "brier", "scaled_brier", "oe", "citl", "slope", "delta", "delta_lo", "delta_hi", "delong_delta", "delong_se",
        "delong_p", "n_boot", "preds_sha256")),
    "roc_points_lr.csv": ("s04", (), ("outcome", "dataset", "model", "fpr", "tpr")),
    "lr_coefficients.csv": ("s04", ("n_train", "events_train"), (
        "outcome", "model", "term", "coef_std", "coef_unit", "center", "scale", "n_train", "events_train")),
    "thresholds.json": ("s05", ("n", "events"), None),
    "operating_points.csv": ("s05", ("n", "events", "tp", "fp", "tn", "fn"), (
        "outcome", "dataset", "model", "threshold_name", "threshold", "derivation", "n", "events", "alert_rate",
        "alert_rate_lo", "alert_rate_hi", "se", "se_lo", "se_hi", "sp", "sp_lo", "sp_hi", "ppv", "ppv_lo", "ppv_hi",
        "npv", "npv_lo", "npv_hi", "alerts_per_case", "alerts_per_case_lo", "alerts_per_case_hi", "tp", "fp", "tn",
        "fn", "n_boot")),
    "comparators_alert.csv": ("s05", ("n", "events", "tp", "fp", "tn", "fn"), (
        "row_type", "outcome", "dataset", "strategy", "rule", "model", "threshold_name", "threshold", "derivation",
        "dev_test_target_rate", "dev_test_achieved_rate", "n", "events", "alert_rate", "alert_rate_lo",
        "alert_rate_hi", "se", "se_lo", "se_hi", "sp", "sp_lo", "sp_hi", "ppv", "ppv_lo", "ppv_hi", "npv", "npv_lo",
        "npv_hi", "alerts_per_case", "alerts_per_case_lo", "alerts_per_case_hi", "fpr", "tpr", "tp", "fp", "tn", "fn",
        "d_alert_rate", "d_alert_rate_lo", "d_alert_rate_hi", "d_se", "d_se_lo", "d_se_hi", "d_sp", "d_sp_lo",
        "d_sp_hi", "d_ppv", "d_ppv_lo", "d_ppv_hi", "d_npv", "d_npv_lo", "d_npv_hi", "n_boot")),
    "decision_curve.csv": ("s05", ("n", "events"), (
        "outcome", "dataset", "threshold_prob", "n", "events", "prevalence", "nb_ens", "nb_ens_lo", "nb_ens_hi",
        "nb_lr_all", "nb_lr_all_lo", "nb_lr_all_hi", "nb_lr_asb", "nb_lr_asb_lo", "nb_lr_asb_hi", "nb_rule_bmi25",
        "nb_rule_bmi25_lo", "nb_rule_bmi25_hi", "nb_ens_youden", "nb_ens_youden_lo", "nb_ens_youden_hi",
        "nb_ens_sens80", "nb_ens_sens80_lo", "nb_ens_sens80_hi", "nb_ens_sens90", "nb_ens_sens90_lo",
        "nb_ens_sens90_hi", "nb_all", "nb_none", "n_boot")),
    "subgroups.csv": ("s06", ("n", "events"), (
        "outcome", "dataset", "model", "subgroup_var", "subgroup", "n", "events", "prevalence", "mean_pred", "auroc",
        "auroc_lo", "auroc_hi", "oe", "oe_lo", "oe_hi", "n_boot", "note")),
    "sensitivity.csv": ("s06", ("n", "events"), (
        "analysis", "outcome", "dataset", "model", "subgroup", "weighted", "n", "events", "prevalence",
        "prevalence_lo", "prevalence_hi", "mean_pred", "auroc", "auroc_lo", "auroc_hi", "oe", "oe_lo", "oe_hi",
        "n_boot", "note")),
    "shap_summary.csv": ("s07", ("n_records",), (
        "outcome", "dataset", "n_records", "n_learners", "input", "label", "mean_abs_shap", "rank",
        "mean_abs_shap_lgb", "mean_abs_shap_xgb", "mean_abs_shap_cat")),
    "shap_by_level.csv": ("s07", ("n",), ("outcome", "dataset", "input", "level", "n", "mean_shap")),
}


class MaskedValueError(ValueError):
    """A masked count ('<10') was used as a number."""


def agg_path(name):
    """Path of an aggregate table in AGG_DIR; refuses names outside AGGREGATES."""
    if name not in AGGREGATES:
        raise KeyError(f"{name!r} is not an aggregate table of AGGREGATES")
    return os.path.join(AGG_DIR, name)


def header_difference(got, want):
    """What differs between a header and the expected one: missing columns, columns it lacks, or the order."""
    got, want = list(got), list(want)
    missing, extra = [c for c in want if c not in got], [c for c in got if c not in want]
    parts = ([f"missing {missing}"] if missing else []) + ([f"not expected {extra}"] if extra else [])
    return "; ".join(parts) or "the same columns in another order"


COUNT_CELL = re.compile(r"0|[1-9][0-9]+")          # a count cell: 0 or a whole number of 10 or more (or blank, '<10')
# columns that hold words and numbers by design; any other column that holds a number holds only numbers and blanks
MIXED_COLUMNS = frozenset({
    ("knhanes_values.csv", "value"), ("dev_values.csv", "value"), ("cohort_checks.csv", "value"),
    ("baseline_characteristics.csv", "level"), ("input_levels.csv", "kn_codes"), ("knhanes_coding.csv", "code"),
    ("knhanes_coding.csv", "maps_to"), ("hyperparameters.csv", "value"), ("shap_by_level.csv", "level")})


def count_cell_ok(text):
    """True for the text of a good count cell: blank, '<10', 0 or a whole number of 10 or more, in digits
    only (no sign, decimal point, separator or leading zero)."""
    t = str(text).strip()
    return t in ("", MASKED) or COUNT_CELL.fullmatch(t) is not None


def _is_number(text):
    try:
        Decimal(str(text).strip())
    except InvalidOperation:
        return False
    return True


def cell_problems(name, df, counts=None):
    """The cells of aggregate table `name`, read as text, that break the cell format, as (column, data row counted
    from 1, what) in file order; `counts` are its count columns (default: those of AGGREGATES). A count column holds
    only what count_cell_ok() takes (the text of a refused count cell is not given: it may be a count of 1-9); '<10',
    as a whole cell, stands in no other column; and a column that holds a number holds only numbers and blanks, apart
    from MIXED_COLUMNS (the refused cell is quoted, up to 40 characters)."""
    counts = AGGREGATES[name][1] if counts is None else counts
    found = []
    for j, col in enumerate(df.columns):
        cells = df[col].astype(str).str.strip()
        if col in counts:
            bad = ~cells.map(count_cell_ok)
            found += [(i, j, col, "not a count cell (blank, '<10', 0 or a whole number of 10 or more)")
                      for i in bad.to_numpy().nonzero()[0]]
            continue
        masked = cells == MASKED
        found += [(i, j, col, "'<10' outside a count column") for i in masked.to_numpy().nonzero()[0]]
        number = cells.map(_is_number)
        if (name, col) in MIXED_COLUMNS or not number.any():
            continue
        word = ~(number | masked | (cells == ""))
        found += [(i, j, col, f"a word in a column of numbers ({cells.iloc[i][:40]!r})")
                  for i in word.to_numpy().nonzero()[0]]
    return [(col, int(i) + 1, what) for i, _j, col, what in sorted(found)]


def read_table(name):
    """An aggregate CSV with every cell as text: blank -> None, '<10' kept. The header must be that of AGGREGATES (the
    error lists the missing, extra or reordered columns), and the cells must keep the cell format
    (cell_problems; the error names the column and the data row, counted from 1, of the first cell that breaks it).
    The table name is kept in
    df.attrs['table'] for messages."""
    import pandas as pd
    spec = AGGREGATES[name] if name in AGGREGATES else None
    if spec is None or spec[2] is None:
        raise KeyError(f"{name!r} is not an aggregate CSV of AGGREGATES")
    df = pd.read_csv(agg_path(name), dtype=str, keep_default_na=False, na_filter=False)
    if tuple(df.columns) != spec[2]:
        raise ValueError(f"{name}: header differs from AGGREGATES: {header_difference(df.columns, spec[2])}")
    bad = cell_problems(name, df, spec[1])
    if bad:
        col, row, what = bad[0]
        more = f" ({len(bad) - 1} more cell(s) of this table also break it)" if len(bad) > 1 else ""
        raise ValueError(f"{name}: column {col}, data row {row}: {what} (see cell_problems){more}")
    out = df.astype(object).where(df != "", None)
    out.attrs["table"] = name
    return out


def read_json(name):
    """thresholds.json (JSON null -> None)."""
    if not name.endswith(".json"):
        raise KeyError(f"{name!r} is not a JSON aggregate")
    with open(agg_path(name), encoding="utf-8") as fh:
        return json.load(fh)


def select(df, **equal):
    """Rows whose columns equal the given text values (None matches a blank cell)."""
    mask = None
    for col, value in equal.items():
        m = df[col].isna() if value is None else (df[col] == str(value))
        mask = m if mask is None else (mask & m)
    return df if mask is None else df[mask]


def one(df, **equal):
    """The single row (a dict) whose columns equal the given values; LookupError (naming the table read by
    read_table) unless exactly one row matches."""
    rows = select(df, **equal)
    if len(rows) != 1:
        table = getattr(df, "attrs", {}).get("table")
        raise LookupError(f"{len(rows)} rows {f'of {table} ' if table else ''}match {equal}")
    return rows.iloc[0].to_dict()


def single(values, what):
    """The one value that the items of `values` share, blanks (None) left out (the n_boot of every row a display
    draws from, for example); ValueError naming `what` unless exactly one value is left."""
    vals = list(dict.fromkeys(v for v in values if v is not None))
    if len(vals) != 1:
        raise ValueError(f"{what}: expected one value, found {len(vals)}")
    return vals[0]


# ------------------------------------------------------------------------------------------------ cells
def is_blank(v):
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return isinstance(v, str) and v.strip() == ""


def is_masked(v):
    return isinstance(v, str) and v.strip() == MASKED


def num(v):
    """Float of a cell; None when blank; MaskedValueError on '<10'."""
    if is_blank(v):
        return None
    if is_masked(v):
        raise MaskedValueError("a masked count ('<10') is not a number")
    x = float(v)
    return None if x != x else x


def count(v):
    """Count cell -> int, '<10' or None (blank)."""
    if is_blank(v):
        return None
    if is_masked(v):
        return MASKED
    d = _decimal(v)
    if d != d.to_integral_value():
        raise ValueError("a count cell holds a fraction")
    return int(d)


def flag(v):
    """'True' / 'False' cell -> bool; None when blank."""
    if is_blank(v):
        return None
    t = str(v).strip()
    if t not in ("True", "False"):
        raise ValueError(f"not a True/False cell: {t!r}")
    return t == "True"


# ------------------------------------------------------------------------------------------------ number formats
def _decimal(x):
    if isinstance(x, Decimal):
        return x
    if isinstance(x, str):
        try:
            return Decimal(x.strip())
        except InvalidOperation:
            raise ValueError(f"not a number: {x!r}") from None
    if isinstance(x, bool):
        raise TypeError("a bool is not a number to display")
    return Decimal(repr(float(x)))


def fmt(x, d=3, na=NA):
    """x with d decimals (half up); NA when blank, '<10' passed through, never '-0.000'."""
    if is_blank(x):
        return na
    if is_masked(x):
        return MASKED
    v = _decimal(x)
    if not v.is_finite():
        return na
    q = v.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return f"{q:f}"


def fmt_signed(x, d=3, na=NA):
    """As fmt(), with '+' before a positive value (differences)."""
    s = fmt(x, d, na)
    if s in (na, MASKED) or s.startswith("-") or _decimal(s) == 0:
        return s
    return "+" + s


def fmt_range(lo, hi, d=3, na=NA):
    """The two bounds of a CI or the two quartiles: '0.341, 0.392'; '-2.58, -2.33'."""
    a, b = fmt(lo, d, na), fmt(hi, d, na)
    if na in (a, b):
        return na
    return f"{a}, {b}"


def fmt_ci(est, lo, hi, d=3, na=NA, signed=False):
    """'0.367 (0.341, 0.392)'; the estimate alone when a bound is blank; NA when the estimate is blank."""
    e = (fmt_signed if signed else fmt)(est, d, na)
    if e in (na, MASKED) or is_blank(lo) or is_blank(hi):
        return e
    return f"{e} ({fmt_range(lo, hi, d, na)})"


def fmt_p(p, na=NA):
    """A P value: three decimals from 0.001 up; below that the exact value to two significant figures in scientific
    form, '3.5×10^-8^' (^...^ is the superscript of the text build); a P of 0 (below what the computation can hold)
    as '<1×10^-300^'. NA when blank."""
    if is_blank(p):
        return na
    v = _decimal(p)
    if v >= Decimal("0.001"):
        return fmt(v, 3, na)
    if v == 0:
        return f"<1{TIMES}10^-300^"
    exponent = v.adjusted()
    mantissa = v.scaleb(-exponent).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if mantissa >= 10:                  # 9.96 rounds up to the next power of ten
        mantissa, exponent = Decimal("1.0"), exponent + 1
    return f"{mantissa}{TIMES}10^{exponent}^"


def fmt_pct(prop, d=1, na=NA):
    """A proportion (0-1) as a percentage with d decimals, no % sign."""
    if is_blank(prop) or is_masked(prop):
        return na if is_blank(prop) else MASKED
    return fmt(_decimal(prop) * 100, d, na)


def fmt_pct_ci(est, lo, hi, d=1, na=NA, signed=False):
    """Proportions shown as percentages with their CI: '45.2 (43.6, 47.4)'."""
    def pct(v):
        return None if is_blank(v) else _decimal(v) * 100
    return fmt_ci(pct(est), pct(lo), pct(hi), d, na, signed)


def fmt_count(n, na=NA):
    """A count, with a comma only in numbers of five or more digits (4500, 12,345); '<10' for a masked cell and for 1-9;
    NA when blank."""
    c = count(n) if not isinstance(n, int) or isinstance(n, bool) else n
    if c is None:
        return na
    if c == MASKED or 1 <= c <= 9:
        return MASKED
    return f"{c:,}" if c >= 10000 else str(c)


SHARE_TOLERANCE = Decimal("0.0051")     # a two-decimal pct cell against the share worked out from its counts


def _whole(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else count(v)


def share_pct(n, total, pct_cell=None):
    """n / total x 100 (a Decimal, not rounded) from a count cell and the total it is a share of; None when either
    is unknown (blank or '<10') or the total is 0. The two-decimal pct cell of the aggregate (0-100), when given,
    must agree with it within SHARE_TOLERANCE (ValueError otherwise), so a wrong denominator stops the script."""
    c, t = _whole(n), _whole(total)
    if not isinstance(c, int) or not isinstance(t, int) or t <= 0:
        return None
    exact = Decimal(c) * 100 / Decimal(t)
    if not is_blank(pct_cell):
        if is_masked(pct_cell):
            raise MaskedValueError("a pct cell holds '<10'")
        if abs(exact - _decimal(pct_cell)) > SHARE_TOLERANCE:
            raise ValueError("a pct cell does not agree with its count and total")
    return exact


def fmt_people_pct(pct, na=NA):
    """A percentage (0-100) that describes people: a whole number from 10 up, one decimal under 10, with the test on
    the value rounded to one decimal (9.96 -> '10'). A value that would round to 100 without being 100 keeps one decimal
    ('99.9'), and one above 0 that rounds to 0 is '<0.1', so that it never reads as 0.0. Rounded once, from the value
    given. NA when blank, '<10' passed through."""
    one = fmt(pct, 1, na)
    if one in (na, MASKED):
        return one
    v = _decimal(pct)
    if v > 0 and _decimal(one) == 0:
        return "<0.1"
    whole = fmt(v, 0, na)
    return one if _decimal(one) < 10 or (_decimal(whole) == 100 and v != 100) else whole


def fmt_share(n, total, pct_cell=None, na=NA):
    """'12' or '2.9': the share n / total as a percentage that describes people (fmt_people_pct), rounded once from the
    counts (share_pct). The pct cells of baseline_characteristics.csv and input_levels.csv hold two decimals, and
    rounding them again to one is 0.1 off for about 1 cell in 20 (2.946 is stored as 2.95, which would show as 3.0, not
    2.9). NA when the share is unknown."""
    return fmt_people_pct(share_pct(n, total, pct_cell), na)


def fmt_n_share(n, total, pct_cell=None, na=NA):
    """'12,345 (12)' from a count and the total it is a share of (the share rounded once, fmt_share); '<10' alone for
    a masked count or one of 1-9; NA when blank. A cell 'n/N (%)' is this cell with N written after n."""
    c = fmt_count(n, na)
    if c in (na, MASKED):
        return c
    return f"{c} ({fmt_share(n, total, pct_cell, na)})"


def mean_sd(mean, sd, d=1, na=NA):
    """'45.0 (12.5)'."""
    m = fmt(mean, d, na)
    return m if m == na or is_blank(sd) else f"{m} ({fmt(sd, d, na)})"


MEDIAN_LABEL = "median [quartile 1, quartile 3]"           # what a row of median_iqr() cells says it shows


def median_iqr(median, q1, q3, d=1, na=NA):
    """'45.0 [35.5, 57.5]'; the median alone when a quartile is blank; NA when the median is blank."""
    m = fmt(median, d, na)
    if m in (na, MASKED) or is_blank(q1) or is_blank(q3):
        return m
    return f"{m} [{fmt_range(q1, q3, d, na)}]"


# ------------------------------------------------------------------------------------------------ labels
INPUTS = ("ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen", "ua_nitrite",
          "ua_pH", "ua_sg", "age", "male", "bmi")
DIPSTICK = INPUTS[:9]
GRADED = ("ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen")
NUMERIC_LEVELS = {"ua_pH": 1, "ua_sg": 3}      # input -> decimals of its level labels
OUTCOMES = ("ALT", "AST")
RULES = ("BMI25",)

OUTCOME_SHORT = {"ALT": "ALT", "AST": "AST"}
OUTCOME_LABELS = {"ALT": "ALT >40 U/L", "AST": "AST >40 U/L"}
# column headers, table rows and panel titles (the short forms)
DATASET_LABELS = {
    "DEV": "Development cohort",
    "DEV_TRAIN": "Development training set",
    "DEV_TEST": "Development test set",
    "KN": "KNHANES",
}
# the one long form of the KNHANES population for every title, legend and note (written to stand inside a sentence;
# capitalise the first letter at the start of one)
DATASET_LONG = {"KN": "eligible KNHANES adults"}
MODEL_LABELS = {
    "ENS": "Ensemble",
    "LR_ASB": "Logistic regression: age, sex and BMI",
    "LR_ALL": "Logistic regression: all 12 inputs",
}
# row labels of the threshold tables (logistic regression spelled out)
MODEL_SHORT = {"ENS": "Ensemble", "LR_ASB": "Logistic regression, age, sex and BMI",
               "LR_ALL": "Logistic regression, 12 inputs"}
# what the ensemble is, without a claim about its calibration
MODEL_LONG = {"ENS": "Soft-voting ensemble of the LightGBM, XGBoost and CatBoost models"}
RULE_LABELS = {"BMI25": f"BMI {GE}25 {BMI_UNIT}"}
# what the rule is, for the title, legend or note that first names it: the Korean criterion for obesity, not a diagnosis
RULE_LONG = {"BMI25": f"BMI {GE}25 {BMI_UNIT}, the Korean Society for the Study of Obesity criterion for obesity"}
# the ensemble thresholds set in the development test set by the Youden index or for a sensitivity
THRESHOLD_LABELS = {
    "YOUDEN": "Youden index threshold",
    "SENS80": "Threshold for 80% sensitivity",
    "SENS90": "Threshold for 90% sensitivity",
}
# the rule and each model at the threshold set by that rule (the points of the ROC panels, the rows of the rule tables)
STRATEGY_LABELS = {
    "RULE_BMI25": f"Rule: BMI {GE}25 {BMI_UNIT}",
    "ENS@MATCH_BMI25": f"Ensemble, threshold set by the BMI {GE}25 {BMI_UNIT} rule",
    "LR_ALL@MATCH_BMI25": f"Logistic regression, 12 inputs, threshold set by the BMI {GE}25 {BMI_UNIT} rule",
    "LR_ASB@MATCH_BMI25": f"Logistic regression, age, sex and BMI, threshold set by the BMI {GE}25 {BMI_UNIT} rule",
}
INPUT_LABELS = {
    "ua_glucose": "Urine glucose", "ua_protein": "Urine protein", "ua_ketone": "Urine ketones",
    "ua_bilirubin": "Urine bilirubin", "ua_blood": "Urine blood", "ua_urobilinogen": "Urine urobilinogen",
    "ua_nitrite": "Urine nitrite", "ua_pH": "Urine pH", "ua_sg": "Urine specific gravity", "age": "Age",
    "male": "Men", "bmi": "BMI",
}
_GRADES = {"0": "Negative", "0.5": "Trace", "1": "1+", "2": "2+", "3": "3+"}
LEVEL_LABELS = {
    # model values; '3+ or above' for glucose, protein and blood, where a higher reading maps to 3 (hospital glucose 4+,
    # KNHANES codes 4 and 5); ketones, bilirubin and urobilinogen are labelled '3+'
    "ua_glucose": {**_GRADES, "3": "3+ or above"},
    "ua_protein": {**_GRADES, "3": "3+ or above"},
    "ua_ketone": dict(_GRADES),
    "ua_bilirubin": dict(_GRADES),
    "ua_blood": {**_GRADES, "3": "3+ or above"},
    # level 0.5 holds the normal readings of the hospital records and KNHANES codes 0 (negative) and 1 (trace); 0 is a
    # rare hospital value
    "ua_urobilinogen": {"0": "Negative", "0.5": "Normal", "1": "1+", "2": "2+", "3": "3+"},
    "ua_nitrite": {"0": "Negative", "1": "Positive"},
    "male": {"0": "Women", "1": "Men"},
    # the SHAP bands of shap_by_level.csv; the top band joins obesity classes II and III (say so in the legend)
    "bmi": {"<18.5": "<18.5", "18.5-22.9": "18.5-22.9", "23.0-24.9": "23.0-24.9", "25.0-29.9": "25.0-29.9",
            "30+": f"{GE}30"},
}
AGE_BAND_LABELS = {"19-39": "19-39 years", "40-59": "40-59 years", "60-79": "60-79 years", "80+": f"{GE}80 years"}
SEX_LABELS = {"Men": "Men", "Women": "Women"}
PERIOD_LABELS = {"2016-2018": "2016-2018", "2019-2021": "2019-2021", "2022-2024": "2022-2024"}
URINE_PERIOD_LABELS = {"2010-2013": "2010-2013", "2014-2017": "2014-2017", "2018-2021": "2018-2021",
                       "2022-2025": "2022-2025"}
# the alcohol groups of the KNHANES code table (subgroups.csv, drink_group)
DRINK_LABELS = {"none_or_rare": "None or rare", "monthly": "Monthly", "weekly": "Weekly", "unknown": "Unknown"}
SUBGROUP_VAR_LABELS = {"all": "All", "age_band": "Age group", "sex": "Sex", "period": "Survey period",
                       "drink_group": "Alcohol drinking frequency"}
SENSITIVITY_LABELS = {
    "CHECKUP_DAY": "Index urinalysis on a health check-up day",
    "WEIGHTED": "Survey-weighted estimates",
    "Checkup day": "Index urinalysis on a health check-up day",
    "Other day": "Index urinalysis on another day",
}
MEANING_LABELS = {"yes": "Yes", "no": "No", "not_applicable": "Not applicable", "unknown": "Unknown"}
BASELINE_LABELS = {
    "persons": "People", "age": "Age, years", "age_band": "Age group", "male": "Men",
    "alt": "ALT, U/L", "ast": "AST, U/L", "alt_label": OUTCOME_LABELS["ALT"], "ast_label": OUTCOME_LABELS["AST"],
    "both_label": "ALT and AST both >40 U/L", "index_urinalysis": "Index urinalysis",
    "checkup_day": "On a health check-up day", "period": "Survey period", **INPUT_LABELS, "bmi": f"BMI, {BMI_UNIT}",
}
NOT_MEASURED = "Not measured"
MISSING_LABEL = "Not recorded"
FLOW_LABELS = {
    # one age box in the flow figure: the over-110 reason alone could be worked out by subtraction
    "dev_adult_excluded": "Aged <19 or >110 years",
}
ABBREVIATIONS = {
    "ALT": "alanine aminotransferase",
    "AST": "aspartate aminotransferase",
    "AUPRC": "area under the precision-recall curve",
    "AUROC": "area under the receiver operating characteristic curve",
    "BMI": "body mass index",
    "CI": "confidence interval",
    "KNHANES": "Korea National Health and Nutrition Examination Survey",
    "LR": "logistic regression",
    "NPV": "negative predictive value",
    "O:E": "ratio of observed to expected events",
    "PPV": "positive predictive value",
    "SD": "standard deviation",
    "SHAP": "Shapley additive explanations",
}


def level_key(v):
    """Canonical text of a level ('1.000' -> '1', '1.010' -> '1.01'); other text unchanged."""
    t = str(v).strip()
    try:
        d = Decimal(t)
    except InvalidOperation:
        return t
    return format(d.normalize(), "f") if d.is_finite() else t


def _level_number(v):
    try:
        return Decimal(level_key(v))
    except InvalidOperation:
        return None


def level_label(input_name, level):
    """Display label of one level of an input ('0.5' -> 'Trace'; SG '1.01' -> '1.010')."""
    key = level_key(level)
    if input_name in NUMERIC_LEVELS:
        return fmt(key, NUMERIC_LEVELS[input_name]) if _level_number(key) is not None else key
    return LEVEL_LABELS.get(input_name, {}).get(key, key)


def subgroup_label(var, level):
    """'age_band', '80+' -> '>=80 years' (with the sign)."""
    if var == "all":
        return "All"
    table = {"age_band": AGE_BAND_LABELS, "sex": SEX_LABELS, "period": PERIOD_LABELS,
             "drink_group": DRINK_LABELS}.get(var, {})
    return table.get(level, level)


def abbreviation_note(*keys):
    """'Abbreviations: AUROC, area under ...; CI, confidence interval.' for the keys given, in alphabetical order: the
    closing note of every figure legend, in the same form as the closing note of every table
    (s08_tables.abbreviations)."""
    keys = sorted(set(keys), key=str.lower)
    unknown = [k for k in keys if k not in ABBREVIATIONS]
    if unknown:
        raise KeyError(f"no abbreviation text for {unknown}")
    return "Abbreviations: " + "; ".join(f"{k}, {ABBREVIATIONS[k]}" for k in keys) + "."


# ------------------------------------------------------------------------------------------------ sparse levels
# Sparse hospital levels pooled for display, as inclusive ranges of model values: bilirubin trace with 1+; SG 1.000 and
# 1.001 with 1.005. The groups of Table 1 and the supplementary tables of the levels are planned by
# s08_tables.plan_levels, the one planner of pooled levels.
SPARSE_POOLS = {
    "ua_bilirubin": ((Decimal("0.5"), Decimal("1")),),
    "ua_sg": ((Decimal("1"), Decimal("1.005")),),
}
# Rows of the descriptive tables (baseline_characteristics.csv in every sample and dataset, input_levels.csv) that s01
# counts together before masking: the level cell of the row -> the model values it holds. The planner takes such a row
# as one level and names it from these values (held_levels).
COUNTED_TOGETHER = {"ua_urobilinogen": {"0.5": ("0", "0.5"), "2": ("2", "3")}}


def ordered_levels(levels):
    """Levels sorted by value (numbers first, in order; other text after, in the given order)."""
    keys = list(dict.fromkeys(level_key(v) for v in levels))
    nums = sorted((k for k in keys if _level_number(k) is not None), key=_level_number)
    return nums + [k for k in keys if _level_number(k) is None]


def held_levels(input_name, level):
    """The model values that a row of the descriptive tables holds: the values s01 counted together
    (COUNTED_TOGETHER), or the level itself."""
    key = level_key(level)
    return COUNTED_TOGETHER.get(input_name, {}).get(key, (key,))


def pooled_label(input_name, levels, universe=None):
    """Label of a group of levels: one level as level_label(); graded inputs 'Trace or 1+', '2+ or above',
    'Negative to 1+'; pH and SG '<=1.005', '>=8.5', '1.010-1.020'. `universe` gives the ends of the scale (default:
    the graded scale of LEVEL_LABELS, or the group itself)."""
    levels = ordered_levels(levels)
    if len(levels) == 1:
        return level_label(input_name, levels[0])
    if universe is None:
        universe = LEVEL_LABELS.get(input_name, {}).keys() if input_name in GRADED else levels
    universe = ordered_levels(universe)
    top, bottom = levels[-1] == universe[-1], levels[0] == universe[0]
    first, last = level_label(input_name, levels[0]), level_label(input_name, levels[-1])
    if input_name in NUMERIC_LEVELS:
        if bottom and not top:
            return f"{LE}{last}"
        if top and not bottom:
            return f"{GE}{first}"
        return f"{first}-{last}"
    if top:
        return f"{first} or above"
    last = last[0].lower() + last[1:] if last[:1].isalpha() else last
    return f"{first} or {last}" if len(levels) == 2 else f"{first} to {last}"


# ------------------------------------------------------------------------------------------------ small cells
# Simple suppression. A count of 1-9 is shown as '<10', and within one table or one figure panel no shown value may
# let it be worked out; differences across tables or figures are accepted. A block is a set of cells whose total is
# shown: a reader who knows the total and the shown cells (or the interval each shown cell lies in) narrows the sum of
# the masked cells.
def masked_sum_range(total, shown, k):
    """(lo, hi): the values the sum of the k masked cells (each 1-9) of a block can take for a reader who knows the
    block's `total` and that each shown cell lies in its interval. shown: (lo, hi) for every other cell of the
    block (lo == hi for a cell shown exactly, (0, 0) for an empty one). ValueError when no sum fits, that is, when the
    cells cannot add up to the total."""
    t, k = int(total), int(k)
    lo = max(k, t - sum(int(h) for _, h in shown))
    hi = min(9 * k, t - sum(int(a) for a, _ in shown))
    if lo > hi:
        raise ValueError(f"the shown cells and {k} masked cell(s) of 1-9 cannot add up to the total")
    return lo, hi


def masked_follow(k, lo, hi):
    """True when k masked cells, whose sum a reader can narrow to [lo, hi] (masked_sum_range), are each known: only
    one sum is possible, and it is one cell's count (k = 1) or k (each cell 1) or 9k (each cell 9)."""
    return k > 0 and lo == hi and (k == 1 or lo in (k, 9 * k))


def sig_round(n, digits=2):
    """A count rounded half up to `digits` significant figures; counts below 10**digits are unchanged. With 2: 57 ->
    57, 114 -> 110, 115 -> 120, 995 -> 1,000, 12,345 -> 12,000."""
    n = int(n)
    if n < 0:
        raise ValueError("a count cannot be negative")
    if n < 10 ** digits:
        return n
    unit = 10 ** (len(str(n)) - digits)
    return int((Decimal(n) / unit).quantize(Decimal(1), rounding=ROUND_HALF_UP)) * unit


def sig_interval(n, digits=2):
    """(lo, hi): every count that sig_round() turns into the same value as n, which is all a reader can tell of n from
    a bar drawn at that height (1,000 -> (995, 1,049); 57 -> (57, 57))."""
    n = int(n)
    r = sig_round(n, digits)
    lo, hi = 0, n                          # the smallest count that rounds to r or more
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if sig_round(mid, digits) >= r else (mid + 1, hi)
    first = lo
    lo, hi = n, 2 * r + 10                 # the largest count that rounds to r or less
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if sig_round(mid, digits) <= r else (lo, mid - 1)
    return first, lo


def panel_heights(cells, total, digits=2, where="panel"):
    """Bar heights of one figure panel whose cells add up to a total shown elsewhere (a histogram of predicted risk:
    the people of the panel, as in Table 2), after the panel's small-cell check.

    cells: the count cells of the bars (int, '<10' or the text of a count cell; a count of 1-9 counts as '<10').
    Returns one height per cell: the count rounded to `digits` significant figures (sig_round; counts below 10**digits
    as they are) for a shown cell, None for a masked one (drawn as a bar of 1-9, its count not shown). A reader then
    knows each shown count only within its sig_interval(). ValueError when a cell is blank or the cells cannot add up
    to `total`; AssertionError when a masked cell follows from the total (masked_follow): a display decision is needed
    (change the display, or hide a neighbouring bar of at least 11 people in a style and legend entry of its own)."""
    if is_blank(total) or is_masked(total):
        raise ValueError(f"{where}: the panel total is needed for the small-cell check")
    heights, shown, k = [], [], 0
    for c in cells:
        v = c if isinstance(c, int) and not isinstance(c, bool) else count(c)
        if v is None:
            raise ValueError(f"{where}: a blank cell (the panel cannot be checked)")
        if v == MASKED or 1 <= v <= 9:
            k += 1
            heights.append(None)
        else:
            heights.append(sig_round(v, digits))
            shown.append(sig_interval(v, digits))
    try:
        lo, hi = masked_sum_range(count(total) if isinstance(total, str) else total, shown, k)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    if masked_follow(k, lo, hi):
        raise AssertionError(f"{where}: a masked band follows from the total; a display decision is needed")
    return heights


# ------------------------------------------------------------------------------------------------ text check
_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")
_ALLOWED_NON_ASCII = set("\u2265\u2264\u00d7\u00b1\u00b0\u00b5\u03bc\u00b2")
_AMERICAN = {
    r"hemoglobin": "haemoglobin", r"analyz(?:e|ed|es|ing)": "analyse", r"randomiz(?:e|ed|ation)": "randomise",
    r"color(?:s|ed)?": "colour", r"center(?:s|ed)?": "centre", r"standardiz(?:e|ed|ation)": "standardise",
    r"summariz(?:e|ed)": "summarise", r"categoriz(?:e|ed)": "categorise", r"characteriz(?:e|ed)": "characterise",
    r"optimiz(?:e|ed|ation)": "optimise", r"minimiz(?:e|ed)": "minimise", r"maximiz(?:e|ed)": "maximise",
    r"recogniz(?:e|ed)": "recognise", r"organization": "organisation", r"behavior": "behaviour",
    r"favor(?:able)?": "favour", r"label(?:ed|ing)": "labelled", r"model(?:ed|ing)": "modelled",
    r"gray": "grey", r"fiber": "fibre", r"liter": "litre", r"anemia": "anaemia", r"edema": "oedema",
    r"pediatric": "paediatric", r"(?:hyper|hypo)?glycemi(?:a|c)": "glycaemia", r"aging": "ageing",
    r"tumor": "tumour", r"utiliz(?:e|ed|ation)": "utilise", r"visualiz(?:e|ed|ation)": "visualise",
    r"normaliz(?:e|ed)": "normalise", r"hospitaliz(?:ed|ation)": "hospitalised", r"enrollment": "enrolment",
    r"catalog": "catalogue",
}
_AMERICAN_RX = re.compile(r"(?i)\b(" + "|".join(_AMERICAN) + r")\b")
# claims that no displayed string may make
_CLAIMS_RX = re.compile(
    r"(?i)\bindependent\b|\bblinded\b|\bpre-?specified\b|\bfrozen\b|\boutperform\w*|\bsuperior\b|\brobust\w*|"
    r"\bstable over time\b|\bnationally representative\b|\bundiagnosed\b|\bdeployab\w*|\bready for use\b|"
    r"\bclinically useful\b|\brule[- ]?out\b|\brestores? calibration\b|\bwell[- ]calibrated\b|\bcost[- ]?saving\b|"
    r"\breduc(?:e|es|ed|ing) testing\b")
_REFERS_TO_TEXT = re.compile(r"(?i)\b(?:the|main) text\b")     # a table or figure is read without the text
_COMMA_FOUR_DIGITS = re.compile(r"(?<![\d,.])\d,\d{3}(?![\d,])")   # 4,500: the comma belongs to five digits or more
_UNIT_SPACE = re.compile(r"(?<![A-Za-z])\d(?:\s{2,})?(?:U/L|kg/m[2\u00b2]|kg\b|cm\b|mm\b|years?\b|days?\b)")
_PANEL_PARENS = re.compile(r"\bFigures?\s+S?\d+\([a-z]\)")


def text_problems(s, allow=()):
    """Display-rule problems of one displayed string (empty list when it may be shown): a dash or minus sign other
    than the ASCII hyphen, other non-ASCII signs, 'matched', claims that no displayed string may make (independent,
    blinded, outperform, superior, robust, undiagnosed, deployable, rule out, well calibrated, ...), American spellings,
    a comma in a number of fewer than five digits, a number and its unit without exactly one space between them, and a
    panel cross-reference in parentheses. `allow`: phrases removed before the check."""
    s = str(s)
    for phrase in allow:
        s = s.replace(phrase, "")
    out = []
    if _DASHES.search(s):
        out.append("a dash or minus sign other than the ASCII hyphen")
    odd = sorted({c for c in s if ord(c) > 127 and c not in _ALLOWED_NON_ASCII and not _DASHES.match(c)})
    if odd:
        out.append("non-ASCII character(s) " + ", ".join(f"U+{ord(c):04X}" for c in odd))
    if re.search(r"(?i)\bmatched\b", s):
        out.append("'matched': thresholds are 'set to flag the proportion nearest to that of the rule' in the "
                   "development test set")
    for m in _CLAIMS_RX.finditer(s):
        out.append(f"{m.group(0)!r}: a claim that no displayed string may make")
    if _COMMA_FOUR_DIGITS.search(s):
        out.append("a comma in a number of fewer than five digits")
    if _UNIT_SPACE.search(s):
        out.append("a number and its unit need one space between them")
    if _PANEL_PARENS.search(s):
        out.append("a panel reference in parentheses: write 'Figure 2a'")
    for m in _AMERICAN_RX.finditer(s):
        word = m.group(1)
        brit = next(v for k, v in _AMERICAN.items() if re.fullmatch(f"(?i){k}", word))
        out.append(f"American spelling {word!r} (British: {brit})")
    return out


def check_text(s, where="text", allow=()):
    """Raise ValueError when a displayed string breaks a display rule (text_problems)."""
    problems = text_problems(s, allow)
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems) + f" in {str(s)[:80]!r}")
    return s


def check_legend(s, where="legend"):
    """Raise ValueError when a title, legend or note of a table or figure refers to the text ('the text', 'main text'):
    each is read without it. Text outside the fragments, such as the items of cohort_checks.csv, may use the words, so
    this is not part of text_problems()."""
    m = _REFERS_TO_TEXT.search(str(s))
    if m:
        raise ValueError(f"{where}: refers to {m.group(0)!r}; a table or figure is read without the text")
    return s


# ------------------------------------------------------------------------------------------------ word check
# The word lists live in text_patterns.py, an optional local quality-check file that is not part of this repository;
# it is imported on first use, and without it the word checks are skipped with a one-line message.
NUMBER_ONLY = re.compile(r"\d+\.\d+")                       # what the number alternatives of the word lists match
NUMERIC_CELL = re.compile("(?:[0-9.,+<>()\\[\\]%\\s\u2264\u2265-]|NA)+")
FOOTNOTE_MARK = re.compile(r"\^[a-z]\^")                    # a note marker of Footnotes ('^a^'), set aside in a cell
_PIPE = re.compile(r"(?<!\\)\|")
# colour markup in a table cell (tables have no colour)
_COLOUR_MARKUP = re.compile(r"(?i)</?\s*(?:span|font)\b|\bstyle\s*=|\bcolou?r\s*=|\\colou?r\b")
_WORDS = []                                                 # [the patterns] once word_patterns() has looked


def word_patterns():
    """(name, compiled pattern) of the word lists of text_patterns.py (imported on first use). () when that file is
    missing: the word checks are then skipped, and the first call says so in one line on standard error."""
    if not _WORDS:
        try:
            import text_patterns
        except ModuleNotFoundError as e:
            if e.name != "text_patterns":
                raise
            sys.stderr.write(f"{SCRIPT}: text_patterns.py is missing, so the word checks were skipped\n")
            _WORDS.append(())
        else:
            _WORDS.append(tuple(text_patterns.FORBIDDEN))
    return _WORDS[0]


def numeric_spans(text):
    """(start, end) spans of `text` that hold only numbers (digits, '(lower, upper)', '[q1, q3]', signs, '%', '<10',
    NA): the cells of its pipe-table lines that hold nothing else (note markers such as '^a^' set aside, so a cell of
    numbers that carries a note is still data), and whole lines that hold nothing else (a tick label or a value in a
    figure). A number that a word list names is data there, not text."""
    spans, pos = [], 0
    for line in str(text).splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body.lstrip().startswith("|"):
            cuts = [m.start() for m in _PIPE.finditer(body)]
            for a, b in zip(cuts, cuts[1:]):
                cell = FOOTNOTE_MARK.sub("", body[a + 1:b]).strip()
                if cell and NUMERIC_CELL.fullmatch(cell):
                    spans.append((pos + a + 1, pos + b))
        elif body.strip() and NUMERIC_CELL.fullmatch(body.strip()):
            spans.append((pos, pos + len(body)))
        pos += len(line)
    return spans


def word_hits(text):
    """Matches of the word patterns in `text`: ([(line number, pattern name, matched text), ...], number left out).
    Each matched span is listed once, under the first pattern that matches it. A match that is only a number
    (NUMBER_ONLY) inside a numeric span (numeric_spans) is data: it is counted as left out, not listed."""
    patterns = word_patterns()
    text = str(text)
    if not patterns or not text:
        return [], 0
    spans = numeric_spans(text)
    hits, exempt, seen = [], 0, set()
    for name, rx in patterns:
        for m in rx.finditer(text):
            if m.span() in seen:
                continue
            seen.add(m.span())
            if NUMBER_ONLY.fullmatch(m.group(0)) and any(a <= m.start() and m.end() <= b for a, b in spans):
                exempt += 1
                continue
            hits.append((text.count("\n", 0, m.start()) + 1, name, m.group(0)))
    return hits, exempt


def forbidden_problems(text):
    """Word-list problems of one displayed text, 'line <n>: <pattern> pattern matches <word>' (empty when it may be
    shown, and when text_patterns.py is missing)."""
    return [f"line {ln}: {name} pattern matches {word!r}" for ln, name, word in word_hits(text)[0]]


def check_words(text, where="text"):
    """Raise ValueError when a displayed text holds a word of the word lists (forbidden_problems)."""
    problems = forbidden_problems(text)
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))
    return text


# ------------------------------------------------------------------------------------------------ markdown
class Footnotes:
    """Table footnotes marked with superscript letters (pandoc '^a^') in order of first use."""

    def __init__(self):
        self._keys, self._text = [], {}

    def mark(self, key, text=None):
        """Marker of a note; its text is given on first use."""
        if key not in self._text:
            if text is None:
                raise KeyError(f"footnote {key!r} has no text yet")
            self._keys.append(key)
            self._text[key] = text
        return f"^{chr(ord('a') + self._keys.index(key))}^"

    def lines(self):
        return [f"^{chr(ord('a') + i)}^ {self._text[k]}" for i, k in enumerate(self._keys)]


def _cell(c):
    if not isinstance(c, str):
        raise TypeError(f"table cells must be formatted text, not {type(c).__name__}: use fmt(), fmt_count() ...")
    return c.replace("|", "\\|").replace("\n", " ")


def md_table(header, rows, align=None, widths=None):
    """Pipe table (pandoc). Cells must be text (format numbers first); align: one letter per column, l, r or c
    (default l); widths: relative widths written as separator dashes (default: the longest cell of each column),
    which pandoc uses when a row is longer than its column limit. Every cell passes check_text(), and none may carry
    colour markup (tables have no colour)."""
    ncol = len(header)
    for i, r in enumerate(rows):
        if len(r) != ncol:
            raise ValueError(f"row {i + 1} has {len(r)} cells, the header {ncol}")
    head = [_cell(h) for h in header]
    body = [[_cell(c) for c in r] for r in rows]
    for i, c in enumerate(head + [c for r in body for c in r]):
        check_text(c, "table cell")
        if _COLOUR_MARKUP.search(c):
            raise ValueError(f"table cell {c[:40]!r} carries colour markup; tables have no colour")
    align = (align or "l" * ncol)
    if len(align) != ncol or set(align) - set("lrc"):
        raise ValueError("align needs one of l, r, c per column")
    if widths is None:
        widths = [max([len(head[j])] + [len(r[j]) for r in body]) for j in range(ncol)]
    seps = []
    for a, w in zip(align, widths):
        w = max(3, int(w))
        seps.append({"l": ":" + "-" * (w - 1), "r": "-" * (w - 1) + ":", "c": ":" + "-" * (w - 2) + ":"}[a])
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(seps) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ outputs
KINDS = ("table", "figure")
TARGETS = ("main", "si")
FRAGMENT_DIRS = {("table", "main"): ("manuscript", "tables"), ("table", "si"): ("supplement", "tables"),
                 ("figure", "main"): ("figures",), ("figure", "si"): ("supplement", "figures")}
INDEX_NAMES = {"table": "tables_index.csv", "figure": "figures_index.csv"}
INDEX_COLUMNS = {"table": ("id", "target", "order", "title", "file", "sources"),
                 "figure": ("id", "target", "order", "title", "legend", "files", "sources")}
FIGURE_TYPES = (".pdf", ".png", ".eps", ".tif")
# a figure saved as separate parts: id -> part letters. One legend and one index row; each part has its own files
# <id>_<part>.pdf ... and <id>_<part>_text.txt. Every figure of this paper is one file, so no id is listed
FIGURE_PARTS = {}
TEXT_SUFFIX = "_text.txt"          # <id>_text.txt: the visible strings of a figure (figstyle.text_inventory_path)
WORD_CHECKED = (".md", ".txt")     # outputs of write_text that are text (tables, legends, supplement, figure strings)
_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def out_path(*parts, create=True):
    """Absolute path under one of the output folders of OUT_ROOT (parents created)."""
    p = os.path.normpath(os.path.join(OUT_ROOT, *parts))
    rel = os.path.relpath(p, OUT_ROOT).split(os.sep)
    if rel[0] not in OUTPUT_DIRS:
        raise ValueError(f"outputs go to {', '.join(OUTPUT_DIRS)} of OUT_ROOT")
    if create:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_text(path, text):
    """Write a UTF-8 text output under OUT_ROOT's output folders (atomically); returns the path. A .md or .txt output
    first passes check_words() (CSV outputs such as numbers.csv are data and are not word-checked here)."""
    rel = os.path.relpath(os.path.abspath(path), OUT_ROOT)
    path = out_path(*rel.split(os.sep))
    if path.lower().endswith(WORD_CHECKED):
        check_words(text, os.path.relpath(path, OUT_ROOT))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, path)
    return path


def fragment_dir(kind, target, create=True):
    if (kind, target) not in FRAGMENT_DIRS:
        raise ValueError(f"kind must be one of {KINDS} and target one of {TARGETS}")
    d = os.path.join(OUT_ROOT, *FRAGMENT_DIRS[(kind, target)])
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def index_path(kind, target):
    return os.path.join(fragment_dir(kind, target, create=False), INDEX_NAMES[kind])


def label(kind, target, order):
    """'Table 1', 'Figure 3', 'Supplementary Table S2', 'Supplementary Figure S1'."""
    stem = "Table" if kind == "table" else "Figure"
    return f"{stem} {int(order)}" if target == "main" else f"Supplementary {stem} S{int(order)}"


def panel_label(panel):
    """The lower-case letter of a figure panel ('a', 'b', ...), from its position (0 is 'a') or its letter:
    figstyle.panel_title draws it in bold at the top left-hand corner of the panel, legends use legend_panel(), and
    panel_ref() gives the form for cross-references."""
    letter = chr(ord("a") + panel) if isinstance(panel, int) else str(panel)
    if not re.fullmatch("[a-z]", letter):
        raise ValueError(f"no panel label for {panel!r}")
    return letter


def legend_panel(panel):
    """The panel letter of a legend, bold in markdown: '**a**'."""
    return f"**{panel_label(panel)}**"


def panel_ref(target, order, panel):
    """A cross-reference to a panel: 'Figure 2a', 'Supplementary Figure S2a'."""
    return label("figure", target, order) + panel_label(panel)


def figure_stems(fig_id):
    """File names without extension of a figure: its id, or <id>_<part> for each part of a figure of FIGURE_PARTS."""
    return [f"{fig_id}_{part}" for part in FIGURE_PARTS[fig_id]] if fig_id in FIGURE_PARTS else [fig_id]


def figure_base(fig_id, target, part=None):
    """Path without extension for figstyle.save_all(): <figure folder>/<id>; <figure folder>/<id>_<part> for a part of
    a figure saved as separate files (FIGURE_PARTS), which has no file of its own without a part."""
    if not _ID.match(fig_id):
        raise ValueError(f"figure id {fig_id!r} must match {_ID.pattern}")
    stem = fig_id if part is None else f"{fig_id}_{part}"
    if stem not in figure_stems(fig_id):
        raise ValueError(f"figure {fig_id!r} is saved as {figure_stems(fig_id)}, not as {stem!r}")
    return os.path.join(fragment_dir("figure", target), stem)


def read_index(kind, target):
    """Rows (dicts) of an index file; [] when it does not exist."""
    p = index_path(kind, target)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and tuple(rows[0].keys()) != INDEX_COLUMNS[kind]:
        raise ValueError(f"{p}: header differs from the index format")
    return rows


def _write_index(kind, target, rows):
    p = index_path(kind, target)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=INDEX_COLUMNS[kind], lineterminator="\n")
    w.writeheader()
    for r in sorted(rows, key=lambda r: int(r["order"])):
        w.writerow(r)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    os.replace(tmp, p)
    return p


def output_names(kind, frag_id):
    """File names of every output of one fragment id: <id>.md and, for a figure, its four image files and
    <id>_text.txt (TEXT_SUFFIX), and those of each part for a figure of FIGURE_PARTS (it lists its id's own image files
    too, so that a run removes any such file it finds)."""
    if not _ID.match(frag_id):
        raise ValueError(f"id {frag_id!r} must match {_ID.pattern}")
    names = [f"{frag_id}.md"]
    if kind == "figure":
        for stem in dict.fromkeys([frag_id] + figure_stems(frag_id)):
            names += [f"{stem}{ext}" for ext in FIGURE_TYPES] + [f"{stem}{TEXT_SUFFIX}"]
    return names


def reset_index(kind, target, ids=()):
    """Remove an index file and every output of the ids it lists and of `ids` (the ids the calling script writes, so
    that outputs are removed also when a stopped run already removed the index): <id>.md and, for a figure, the
    image files of its index row and the image files and visible strings of output_names(). Call it at the start of a
    run that rebuilds them, so that a run that stops leaves no out-of-date fragment, image or index."""
    if kind not in KINDS or target not in TARGETS:
        raise ValueError(f"kind must be one of {KINDS} and target one of {TARGETS}")
    d = fragment_dir(kind, target, create=False)
    rows = read_index(kind, target)
    names = set()
    for i in [r["id"] for r in rows] + list(ids):
        names.update(output_names(kind, i))
    if kind == "figure":
        names.update(f for r in rows for f in (r.get("files") or "").split(";") if f)
    for n in sorted(names):
        if os.path.basename(n) != n or n in (".", ".."):
            raise ValueError(f"{index_path(kind, target)}: {n!r} is not a file name of its folder")
        p = os.path.join(d, n)
        if os.path.isfile(p):
            os.remove(p)
    if os.path.exists(index_path(kind, target)):
        os.remove(index_path(kind, target))


def write_fragment(kind, id, title, markdown, notes, sources, *, target, order, files=None):
    """Write one table or figure-legend fragment (<id>.md) and add or replace its row in the index file.

    kind 'table' or 'figure'; id [a-z][a-z0-9_]*; title: caption without the label; markdown: the table (tables) or
    the legend text (figures); notes: list of note paragraphs; sources: aggregate table names the numbers come from;
    target 'main' or 'si'; order: number within the target; files (figures only): the four paths returned by
    figstyle.save_all(fig, figure_base(id, target)) or, for a figure of FIGURE_PARTS, the four paths of each part in
    turn (paths_a + paths_b, from figure_base(id, target, part)). The title, the body and every note pass check_text(),
    check_words() (in the body a number alone in a table cell of numbers is data) and check_legend(). Returns the
    fragment path."""
    if kind not in KINDS or target not in TARGETS:
        raise ValueError(f"kind must be one of {KINDS} and target one of {TARGETS}")
    if not _ID.match(id):
        raise ValueError(f"fragment id {id!r} must match {_ID.pattern}")
    order = int(order)
    if order < 1:
        raise ValueError("order starts at 1")
    if isinstance(notes, str):
        notes = [notes]
    notes = [n for n in notes if n]
    title = " ".join(str(title).split())
    if not title:
        raise ValueError("a fragment needs a title")
    unknown = [s for s in sources if s not in AGGREGATES]
    if not sources or unknown:
        raise ValueError(f"sources must name aggregate tables (unknown: {unknown})")
    for text, where in [(title, "title"), (markdown, "body")] + [(n, "note") for n in notes]:
        check_text(text, f"{id} {where}")
        check_words(text, f"{id} {where}")
        check_legend(text, f"{id} {where}")
    d = fragment_dir(kind, target)
    row = {"id": id, "target": target, "order": str(order), "title": title, "sources": ";".join(sources)}
    if kind == "table":
        row["file"] = f"{id}.md"
    else:
        files = list(files or [])
        want = [f"{stem}{ext}" for stem in figure_stems(id) for ext in FIGURE_TYPES]
        if [os.path.basename(f) for f in files] != want:
            raise ValueError(f"figure {id!r} needs the files {';'.join(want)}, in that order")
        for f in files:
            if os.path.dirname(os.path.abspath(f)) != d or not os.path.exists(f):
                raise ValueError(f"figure files must exist in {d}")
        row["legend"] = " ".join([" ".join(markdown.split())] + [" ".join(n.split()) for n in notes]).strip()
        row["files"] = ";".join(os.path.basename(f) for f in files)
    rows = [r for r in read_index(kind, target) if r["id"] != id]
    clash = [r["id"] for r in rows if int(r["order"]) == order]
    if clash:
        raise ValueError(f"{label(kind, target, order)} is already used by {clash[0]!r}")
    text = "\n\n".join([f"**{label(kind, target, order)}.** {title}", markdown.strip()] + [n.strip() for n in notes])
    path = os.path.join(d, f"{id}.md")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    os.replace(tmp, path)
    _write_index(kind, target, rows + [row])
    return path


# ------------------------------------------------------------------------------------------------ running
def run(main):
    """Call main() and exit: 0 when it returns None or 0; on an exception, the traceback (standard error), a one-line
    FAIL summary and exit 1. main() prints its own one-line summary."""
    t0 = time.time()
    try:
        rc = main()
    except Exception as e:  # noqa: BLE001 - reported, then a non-zero exit
        traceback.print_exc()
        print(f"{SCRIPT}: FAIL {type(e).__name__} ({time.time() - t0:.0f} s)")
        sys.exit(1)
    if rc not in (None, 0):
        print(f"{SCRIPT}: FAIL exit {rc} ({time.time() - t0:.0f} s)")
        sys.exit(int(rc) if isinstance(rc, int) else 1)
    sys.exit(0)
