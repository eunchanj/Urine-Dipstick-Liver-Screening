"""Values the text may quote, collected from the aggregate tables with their sources: manuscript/numbers.csv.

numbers.csv is an internal file for tracing the numbers of the text back to the aggregate tables. It has the status of
the aggregate CSVs: it is not published.

Nothing new is computed here. Every value is one cell of an aggregate table of AGG_DIR, carried as it
is written, apart from the pooled dipstick counts of Table 1 and Supplementary Table S1: those are the sums and
differences of cells that s08_tables.level_plans works out for the tables, and each is written with its derivation. The
display text comes from the formats of display.py (a baseline .pct is shown as its count over the persons of its sample,
rounded once, as in Table 1). The numbers of the text can be looked up in this file.

Run (it reads only the aggregate tables of AGG_DIR and writes under OUT_ROOT, see display.py):
    PYTHONDONTWRITEBYTECODE=1 python s11_numbers.py

numbers.csv (OUT_ROOT/manuscript/, UTF-8, one row per key, keys unique, rows in the order of the families below; it
is removed when a run starts, so a run that stops leaves none)
  key            dotted id of [A-Za-z0-9_] parts, such as ALT.KN.ENS.auroc
  value          the source cell as written: '<10' for a masked count, blank for a blank cell (never a number in
                 their place); for thresholds.json the number as JSON writes it, blank for null (s11 stops when
                 the field is missing); for a derived dipstick key the count its derivation gives
  lo, hi         the source cells of the 95% CI when the key has one; blank otherwise
  display        the value as the text may quote it: the estimate, then '(lo, hi)' when lo and hi are given ('+'
                 before a positive difference, never before a bound); counts with a comma from five digits and
                 '<10' for 1-9; proportions as percentages without the % sign; P values as display.fmt_p writes them
                 (exact, in scientific form below 0.001, '3.5×10^-8^'); 'NA' for a blank cell
  unit           unit of display (persons, %, percentage points, years, U/L, kg/m2, ...); blank for the AUROC, O:E and
                 other ratios
  source_file    the aggregate table
  source_row     the columns that pick the row, 'column=value;column=value' (an empty value picks a blank cell);
                 for thresholds.json 'path=ALT/ENS/YOUDEN' (its keys from the top; 'path=' for the top level); for a
                 derived dipstick key the columns its terms share, 'sample=ANY;variable=<input>'
  source_column  the column of value; 'col;col_lo;col_hi' for a key with a CI; for thresholds.json the field; for a
                 derived dipstick key its derivation, 'n = <terms>' or 'pct = (<terms>) / <D>:persons'

Key families (O outcome, D dataset, M model, T threshold name; decimals of display in brackets; 'people': a percentage
that describes people, display.fmt_share, a whole number from 10 up and one decimal under 10)
  flow.<step>.n, flow.<step>.excluded, flow.<reason>.n, flow.<final>.n, flow.<final>.events     cohort_flow.csv
  checks.<dev|rule|split|kn>.<item>                               cohort_checks.csv, the items of CHECK_ITEMS
  baseline.<sample>.<D>.persons.n                                  baseline_characteristics.csv: sample ANY and the four
      descriptive contrasts E1A_ELIGIBLE, E1A_NOT, E1B_BMI and E1B_NO_BMI (dataset DEV)
  baseline.<sample>.<D>.<variable>.<level>.n | .pct (people)      categorical rows (male: women, men; alt_label,
      ast_label and both_label: neg, pos, not_measured; checkup_day: no, yes; age_band, urine_period and period by
      their levels)
  baseline.<sample>.<D>.age.mean | .sd | .median | .q1 | .q3 (0)      years
  baseline.<sample>.<D>.bmi.mean | .sd | .median | .q1 | .q3 (1)      kg/m2
  baseline.<sample>.<D>.<alt|ast>.mean | .sd | .median | .q1 | .q3 (1)     U/L
  baseline.<sample>.<D>.<bmi|alt|ast>.n | .pct, and .missing | .not_measured .n | .pct
      every .pct: value is the two-decimal pct cell; display is the share n / persons of the sample and dataset
      rounded once (display.fmt_share), as Table 1 shows it (the cell, rounded again, is 0.1 off now and then); a share
      that would round to 100 without being 100 keeps one decimal, and a share above 0 that rounds to 0 is shown as
      '<0.1'
  baseline.ANY.<D>.<input>.<group>.n | .pct (people)              the dipstick rows of Table 1 and Table S1
      the display groups of s08_tables.level_plans for sample ANY, the one sample of those tables: every input, for
      D = DEV_TRAIN, DEV_TEST, KN and DEV (urine pH and specific gravity: DEV and KN), such as
      baseline.ANY.DEV_TEST.ua_glucose.trace_or_1plus.n.
      <group> is the row label as a key part ('Trace or above' -> trace_or_above, '<=1.005' -> le1_005,
      '1.006-1.010' -> 1_006_1_010). A group of one level with its own row is that row, as for the other levels. Any
      other group is derived: in its terms, <D>:<level> is the n cell of that level and <D>:persons the persons row
      of D, added with their signs; they are the levels of the group, or the column total less the other groups, or the
      development cohort less the training or test set, as s08 works the count out ('n = 0' when no level of the group
      has a row in D). A group shown as '<10' has its .n key only.
  baseline.ANY.<D>.<input>.at_or_above.n | .pct (people)          the one row of each dipstick input in Table 1
      (s08_tables.DIPSTICK_SUMMARY: the levels at or above the cut of the row, D = DEV_TRAIN, DEV_TEST, KN), derived
      from the groups of the plan like a pooled group (the sum of the groups from the cut up, or the column total less
      the groups below it or, with a group across the cut, less the levels below the cut, s08_tables.at_or_above); no
      key when the table shows NA or a count of 1-9
  baseline.ANY.<D>.<ua_pH|ua_sg>.median | .q1 | .q3 (pH 1, specific gravity 3)   the median [quartile 1, quartile 3]
      rows of Table 1, for D = DEV_TRAIN, DEV_TEST, KN
  baseline.ANY.KN.ua_sg.above_1_030.n | .pct (people)             eligible adults with a specific gravity above 1.030
      (derived: the sum of those levels; Table S1 shows them in intervals)
  train.<O>.<column>                                              train_summary.csv; prevalence_train (people), as
                                                                   the events of the ensemble table show it
  <O>.<D>.ENS.n | events | prevalence (1) | mean_pred (1) | auroc (3) | auprc (3) | n_boot      performance.csv
  <O>.<D>.ENS.oe | citl | slope | slope_excl01 (2) | brier | scaled_brier (3) | n_p0 | events_p0 | n_p1 |
      nonevents_p1                                                calibration.csv
  <O>.<D>.<LR model>.auroc | auprc | brier | scaled_brier (3) | oe | citl | slope (2)          model_comparison.csv
  <O>.<D>.<A>_vs_<B>.delta_auroc (3, signed) | delong_p                                       model_comparison.csv
  <O>.<D>.<M>.<T>.alert_rate | se | sp | ppv | npv (1) | alerts_per_case (2, as in the tables: PER_CASE_FORMAT)
      T YOUDEN, SENS80, SENS90 from operating_points.csv; T MATCH_BMI25 from comparators_alert.csv, with
      d_alert_rate | d_se | d_sp | d_ppv | d_npv (2, percentage points: model minus rule)
  <O>.<D>.RULE_BMI25.<metric>                                     comparators_alert.csv, rule rows
  <O>.KN.RULE_BMI25.alert_rate_2dp (2)                            the KNHANES rule alert rates to two decimals, a form
      that no table shows: the tables give .alert_rate, with one decimal
      (s11 stops when a MATCH row of operating_points.csv has no comparators_alert.csv row)
  thresholds.n_boot; thresholds.<O>.rules.BMI25.dev_test_alert_rate (1)                       thresholds.json
  thresholds.<O>.<M>.<T>.threshold (4, probability) | threshold_pct (2) | dev_test_alert_rate | dev_test_se (1)
      | target_rate | achieved_rate (1) | difference (3, percentage points), for M = ENS, LR_ALL, LR_ASB
  dca.<O>.<D>.<tag>.threshold_prob | nb_ens | nb_lr_all | nb_lr_asb | nb_rule_bmi25 | nb_ens_youden |
      nb_ens_sens80 | nb_ens_sens90 | nb_all | nb_none (4)                                     decision_curve.csv
      threshold_prob is the grid value as a percentage, without a trailing '.0' (0.048 -> 4.8, 0.21 -> 21). Tags
      p02, p05, p10, p15, p20 and p30: the threshold probabilities 0.02 ... 0.30 of every dataset, when they are on
      the grid, with the 95% CIs of the seven strategies that have them; tags t<NNN>: every other grid point of
      KN, in thousandths (t048 = 0.048), point values only
  subgroup.<O>.<D>.<M>.<variable>.<level>.n | events | prevalence (1) | mean_pred (1) | auroc (3) | oe (2)
  sens.<analysis>.<O>.<D>.<M>.<group>.<metric>, metrics as for subgroups, prevalence with its CI   sensitivity.csv
      (analysis CHECKUP_DAY, groups checkup_day, other_day; WEIGHTED, groups weighted, unweighted)
  shap.<O>.<D>.n_records | n_learners; shap.<O>.<D>.<input>.rank | mean_abs (3, log-odds)       shap_summary.csv

Left out (counted as withheld in the summary line)
  - the reasons of a flow step when one of them is masked or blank: the step's excluded count stands for them, as the
    single age box of Figure 1 does; otherwise the masked reason follows by subtraction;
  - the items of a set of cohort_checks.csv rows that together split a total given elsewhere (CHECK_PARTS: persons
    at dev_adult by the availability of the two enzymes, the BMI record of the development cohort, the urine periods,
    the cross-tabulation of the two enzymes and the split strata), when one of them is masked or blank; and the items
    of the availability of the two enzymes when a reason of the flow step dev_enzymes or kn_enzymes is masked or blank
    (with those counts, the flow step would give the masked reason by subtraction: CHECK_FLOW_PARTS);
  - the levels of a variable of Table 1 and the baseline tables of the supplement, in any sample and dataset (with the
    measured and not measured rows of BMI, ALT and AST) when one of them is masked or blank (s11 stops when one of those
    two rows is missing);
  - in subgroups.csv (per subgroup variable) and in the CHECKUP_DAY rows of sensitivity.csv (per outcome, dataset and
    model), when one group's count is masked or blank: the counts of every group and the rates that carry them
    (prevalence, O:E), since the totals of the groups are given by other tables (s11 stops when a group of
    CHECKUP_DAY has no row). This is more than the displays hide (the supplementary tables of the subgroups and of the
    sensitivity analyses hide only the cells that s08_tables.hide_for_block picks, Figure 4 the column of the subgroup
    variable), so a value they show can have no key here, and the text does not quote it;
  - in the WEIGHTED rows of sensitivity.csv, when the records without a positive weight, their events or their
    non-events number 1-9 (or a count is unknown): the weighted row's n, events and prevalence, by the rule of the
    supplementary table of the sensitivity analyses (s08_tables.weighted_hidden); its mean predicted risk, AUROC and
    O:E stay;
  - the count of eligible adults above a specific gravity of 1.030 when one of its levels is masked.
Not taken: the cohort_checks.csv items outside CHECK_ITEMS (among them the comparator-sign and label-branch rows of the
two enzymes, which are not published), the dipstick levels of the four descriptive contrasts, the confusion counts tp,
fp, tn and fn (the tables show rates: only the note of a row whose rates are not shown, s08_tables.counts_note, gives
that row's counts, and the text does not quote them), and rows that repeat another table (the ENS rows of
model_comparison.csv, the MATCH rows of operating_points.csv, which comparators_alert.csv repeats, the 'All' rows of
subgroups.csv).
"""
import aggregate_guard  # noqa: F401  (first statement: refuses individual-level files, records the files opened)

import csv
import io
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import display as D
import s08_tables as T     # the level plans of the dipstick rows (one planner for every table and this file)

OUT_PARTS = ("manuscript", "numbers.csv")
COLUMNS = ("key", "value", "lo", "hi", "display", "unit", "source_file", "source_row", "source_column")
KEY = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")
# item texts of cohort_checks.csv with '1,000 U/L': they stand in selectors, which are not displayed words, and the text
# check leaves out only these forms of the comma
INTERNAL_NAMES = {"source_row": ("1,000 U/L",)}


def _ci(d, signed=False):
    return lambda v, lo, hi: D.fmt_ci(v, lo, hi, d, signed=signed)


def _pct(d, signed=False):
    return lambda v, lo, hi: D.fmt_pct_ci(v, lo, hi, d, signed=signed)


# display formats: name -> function(value, lo, hi) of source cells
FORMATS = {
    "count": lambda v, lo, hi: D.fmt_count(v),          # persons, events, records: '12,345', '4500', '<10'
    "int": lambda v, lo, hi: D.fmt(v, 0),               # settings and ranks (folds, trials, seed, rank)
    "r0": _ci(0), "r1": _ci(1), "r2": _ci(2), "r3": _ci(3), "r4": _ci(4),
    "d3": _ci(3, signed=True),                          # AUROC differences
    "pct1": _pct(1), "pct2": _pct(2),                   # proportions (0-1) shown as percentages
    "pp2": _pct(2, signed=True), "pp3": _pct(3, signed=True),   # differences of proportions, percentage points
    # grid probabilities as percentages: 0.048 -> 4.8, 0.21 -> 21 (the grid steps of 0.002 need one decimal)
    "pct_grid": lambda v, lo, hi: re.sub(r"\.0$", "", D.fmt_pct(v, 1)),
    "p": lambda v, lo, hi: D.fmt_p(v),
}
PER_CASE_FORMAT = f"r{T.PER_CASE_DECIMALS}"      # alerts per case (1/PPV), with the decimals of the tables

# the persons at dev_adult by the enzymes they have on the urine day, and the kind of day (key part, words of the item)
AVAILABILITY = (("both", "both ALT and AST"), ("alt_only", "ALT only"), ("ast_only", "AST only"),
                ("neither", "neither ALT nor AST"))
DAY_WORDS = (("checkup", "a check-up day"), ("other", "another day"))
URINE_PERIODS = ("2010-2013", "2014-2017", "2018-2021", "2022-2025")
# the four cells of ALT above 40 U/L by AST above 40 U/L (key part, words of the item) and the split strata (0-3)
ALT_AST_CELLS = (("neg_neg", "ALT not above 40 and AST not above 40"), ("neg_pos", "ALT not above 40 and AST above 40"),
                 ("pos_neg", "ALT above 40 and AST not above 40"), ("pos_pos", "ALT above 40 and AST above 40"))


def _enzyme_items(name):
    """The cohort_checks.csv items of one enzyme, as (key, source, item, unit), with a key part in lower case."""
    e = name.lower()
    return (
        (f"dev.cohort_same_day_{e}", "DEV", f"development cohort persons with a same-day {name}", "persons"),
        (f"dev.cohort_{e}_exactly_40", "DEV", f"development cohort same-day {name} exactly 40", "persons"),
        (f"dev.cohort_{e}_above_40_below_41", "DEV", f"development cohort same-day {name} above 40 and below 41",
         "persons"),
        (f"dev.cohort_{e}_not_whole", "DEV", f"development cohort same-day {name} not a whole number", "persons"),
        (f"dev.cohort_{e}_1000_or_more", "DEV", f"development cohort same-day {name} 1,000 U/L or more", "persons"),
        (f"dev.cohort_{e}_repeated_draws", "DEV", f"development cohort same-day {name} from repeated draws (nrep > 1)",
         "persons"),
        (f"dev.cohort_{e}_first_last_differ", "DEV",
         f"development cohort persons whose first and last same-day {name} draws differ on the above-40 cut",
         "persons"),
        (f"dev.adult_{e}_no_rule", "DEV", f"persons at dev_adult with a same-day {name} that no rule can label",
         "persons"),
        (f"kn.{e}_exactly_40", "KN", f"eligible adults {name} exactly 40", "persons"),
        (f"kn.{e}_above_40_below_41", "KN", f"eligible adults {name} above 40 and below 41", "persons"),
        (f"kn.{e}_not_whole", "KN", f"eligible adults {name} not a whole number", "persons"),
        (f"kn.{e}_1000_or_more", "KN", f"eligible adults {name} 1,000 U/L or more", "persons"),
        (f"kn.anthro_{e}_only", "KN", f"adults at kn_anthro with HE_{e} only", "persons"),
    )


# cohort_checks.csv items the text may quote: (key, source, item, unit); 'survey records <year>' rows are added
CHECK_ITEMS = (
    ("dev.processed_rows", "DEV", "processed rows", "rows"),
    ("dev.persons", "DEV", "persons", "persons"),
    ("dev.persons_repeated_rows", "DEV", "persons with repeated rows", "persons"),
    ("dev.raw_first_visit_persons", "DEV", "raw first-visit persons", "persons"),
    ("dev.raw_first_visit_not_processed", "DEV", "raw first-visit persons not in the processed file", "persons"),
    ("dev.cohort_checkup_day", "DEV", "development cohort persons whose checkup date is the urine day", "persons"),
    ("dev.cohort_nitrite_from_text_1", "DEV", "development cohort persons with nitrite filled from the text as 1",
     "persons"),
    ("dev.cohort_nitrite_from_text_0", "DEV", "development cohort persons with nitrite filled from the text as 0",
     "persons"),
    ("dev.cohort_nitrite_1", "DEV", "development cohort persons with nitrite 1", "persons"),
    ("rule.glucose_4plus_cohort", "RULE", "ua_glucose 4 before the rule, development cohort", "persons"),
    ("rule.sex_missing_enzymes", "RULE", "sex not M or F before the rule, persons at dev_enzymes", "persons"),
    ("rule.adult_only_one_enzyme", "RULE",
     "persons at dev_adult with only one of same-day ALT and AST (both are required)", "persons"),
    ("rule.complete_inputs_alt_only", "RULE", "adults with complete inputs and a BMI with ALT only", "persons"),
    ("rule.complete_inputs_ast_only", "RULE", "adults with complete inputs and a BMI with AST only", "persons"),
    ("rule.kn_only_one_enzyme", "RULE",
     "adults at kn_anthro with only one of HE_alt and HE_ast (both are required)", "persons"),
    ("kn.psu_not_numeric", "KN", "eligible adults with psu missing or not a number", "persons"),
    ("kn.pregnancy_yes", "KN", "eligible adults with pregnancy yes (counted, not excluded)", "persons"),
    ("dev.enzymes_missing_glucose", "DEV", "persons at dev_enzymes with ua_glucose missing", "persons"),
    ("dev.enzymes_missing_protein", "DEV", "persons at dev_enzymes with ua_protein missing", "persons"),
    ("dev.enzymes_missing_ketone", "DEV", "persons at dev_enzymes with ua_ketone missing", "persons"),
    ("dev.enzymes_missing_bilirubin", "DEV", "persons at dev_enzymes with ua_bilirubin missing", "persons"),
    ("dev.enzymes_missing_blood", "DEV", "persons at dev_enzymes with ua_blood missing", "persons"),
    ("dev.enzymes_missing_urobilinogen", "DEV", "persons at dev_enzymes with ua_urobilinogen missing", "persons"),
    ("dev.enzymes_missing_nitrite", "DEV", "persons at dev_enzymes with ua_nitrite missing", "persons"),
    ("dev.enzymes_missing_ph", "DEV", "persons at dev_enzymes with ua_pH missing", "persons"),
    ("dev.enzymes_missing_sg", "DEV", "persons at dev_enzymes with ua_sg missing", "persons"),
) + tuple(x for name in D.OUTCOMES for x in _enzyme_items(name)) + tuple(
    (f"dev.adult_{a}_{d}", "DEV", f"persons at dev_adult with {what} on the urine day, {day}", "persons")
    for a, what in AVAILABILITY for d, day in DAY_WORDS) + (
    ("dev.cohort_bmi_same_day", "DEV", "development cohort persons with a BMI on the urine day", "persons"),
    ("dev.cohort_bmi_1_30_days", "DEV", "development cohort persons with a BMI 1-30 days before the urine day",
     "persons"),
    ("dev.cohort_bmi_1_7_days", "DEV", "development cohort persons with a BMI 1-7 days before the urine day",
     "persons"),
    ("dev.cohort_bmi_8_30_days", "DEV", "development cohort persons with a BMI 8-30 days before the urine day",
     "persons"),
    ("dev.cohort_bmi_checkup_record", "DEV",
     "development cohort persons whose BMI comes from a record naming a check-up", "persons"),
    ("dev.cohort_bmi_other_record", "DEV", "development cohort persons whose BMI comes from another record", "persons"),
) + tuple(x for p in URINE_PERIODS for x in (
    (f"dev.urine_{p.replace('-', '_')}", "DEV", f"development cohort persons with a urine test in {p}", "persons"),
    (f"dev.urine_{p.replace('-', '_')}_checkup", "DEV",
     f"development cohort persons with a urine test in {p} on a check-up day", "persons"),
)) + tuple(
    (f"dev.alt_ast_{c}", "DEV", f"development cohort persons with {what}", "persons") for c, what in ALT_AST_CELLS
) + tuple(
    (f"kn.alt_ast_{c}", "KN", f"eligible adults with {what}", "persons") for c, what in ALT_AST_CELLS
) + tuple((f"split.{s}_stratum_{k}", "SPLIT", f"{word} persons in stratum {k}", "persons")
          for s, word in (("training", "training"), ("test", "test")) for k in range(4))
# sets of the items above that split one total given elsewhere (the flow, or another item): when one of a set is masked
# or blank, every item of the set is left out, or the masked item would follow by subtraction
CHECK_PARTS = (
    tuple(f"dev.adult_{a}_{d}" for a, _ in AVAILABILITY for d, _ in DAY_WORDS) + ("rule.adult_only_one_enzyme",),
    ("dev.cohort_bmi_same_day", "dev.cohort_bmi_1_30_days"),
    ("dev.cohort_bmi_1_7_days", "dev.cohort_bmi_8_30_days"),
    ("dev.cohort_bmi_checkup_record", "dev.cohort_bmi_other_record"),
    tuple(f"dev.urine_{p.replace('-', '_')}" for p in URINE_PERIODS),
    tuple(f"dev.urine_{p.replace('-', '_')}_checkup" for p in URINE_PERIODS),
    tuple(f"dev.alt_ast_{c}" for c, _ in ALT_AST_CELLS),
    tuple(f"kn.alt_ast_{c}" for c, _ in ALT_AST_CELLS),
    tuple(f"split.training_stratum_{k}" for k in range(4)),
    tuple(f"split.test_stratum_{k}" for k in range(4)),
    ("rule.kn_only_one_enzyme", "kn.anthro_alt_only", "kn.anthro_ast_only"),
)
# the items left out when a reason of the flow step is masked or blank, by (source, step) of cohort_flow.csv: with the
# step's count they would give the masked reason by subtraction
CHECK_FLOW_PARTS = {
    ("DEV", "dev_enzymes"): tuple(f"dev.adult_{a}_{d}" for a, _ in AVAILABILITY for d, _ in DAY_WORDS),
    ("KN", "kn_enzymes"): ("rule.kn_only_one_enzyme", "kn.anthro_alt_only", "kn.anthro_ast_only"),
}
FLOW_FINALS = ("DEV_TRAIN", "DEV_TEST", "KN")       # the final rows of cohort_flow.csv, for each outcome
# fields of thresholds.json besides 'threshold', by the start of a threshold's name: (field, format, unit)
_RATE = ("pct1", "%")
THRESHOLD_FIELDS = (
    ("YOUDEN", (("dev_test_alert_rate", *_RATE),)),
    ("SENS", (("dev_test_alert_rate", *_RATE), ("dev_test_se", *_RATE))),
    ("MATCH_", (("target_rate", *_RATE), ("achieved_rate", *_RATE), ("difference", "pp3", "percentage points"))),
)
SURVEY_RECORDS = re.compile(r"^survey records (\d{4})$")

# baseline_characteristics.csv
LEVEL_KEYS = {
    "male": {"0": "women", "1": "men"},
    "alt_label": {"0": "neg", "1": "pos"},
    "ast_label": {"0": "neg", "1": "pos"},
    "both_label": {"0": "neg", "1": "pos"},
    "checkup_day": {"0": "no", "1": "yes"},
}
CATEGORICAL = ("age_band", "male", "alt_label", "ast_label", "both_label", "checkup_day", "urine_period", "period")
SUMMARY_STATS = ("mean", "sd", "median", "q1", "q3")
CONTINUOUS = {
    # variable: [(stat, key suffix, format, unit)]; the stored units are years, kg/m2 and U/L
    "age": [(s, "", "r0", "years") for s in SUMMARY_STATS],
    "bmi": [(s, "", "r1", "kg/m2") for s in SUMMARY_STATS],
    "alt": [(s, "", "r1", "U/L") for s in SUMMARY_STATS],
    "ast": [(s, "", "r1", "U/L") for s in SUMMARY_STATS],
}
MEASURED = ("bmi", "alt", "ast")                    # their blank-level row counts the persons with a value
UNMEASURED = {"bmi": "(missing)", "alt": "(not measured)", "ast": "(not measured)"}  # the other row
# the sample whose dipstick rows the tables show (Table 1 and Table S1), with the columns (keys of
# s08_tables.LEVEL_COLUMNS) that have keys: every column of its plans
DIPSTICK_KEYS = {"ANY": tuple(T.LEVEL_COLUMNS)}
# dipstick rows of Table 1 shown as median [quartile 1, quartile 3]: input -> decimals, and the datasets of that table
MEDIAN_INPUTS = {"ua_pH": 1, "ua_sg": 3}
MEDIAN_DATASETS = {"ANY": ("DEV_TRAIN", "DEV_TEST", "KN")}
SG_TOP = Decimal("1.030")                           # the top strip step of the hospital records
TERM_ORDER = ("DEV", "DEV_TRAIN", "DEV_TEST", "KN")

OP_RATES = ("alert_rate", "se", "sp", "ppv", "npv")
OP_DIFFS = ("d_alert_rate", "d_se", "d_sp", "d_ppv", "d_npv")
KN_DATASETS = ("KN",)
DCA_POINTS = (("p02", 0.02), ("p05", 0.05), ("p10", 0.10), ("p15", 0.15), ("p20", 0.20), ("p30", 0.30))
DCA_TAG_DATASET = "KN"                              # every grid point of this dataset has its keys (t<NNN>)
# net benefits with 95% limits (the curves of the three models, the rule and the three fixed ensemble alerts), and the
# two reference strategies (point estimates only)
DCA_WITH_CI = ("nb_ens", "nb_lr_all", "nb_lr_asb", "nb_rule_bmi25", "nb_ens_youden", "nb_ens_sens80", "nb_ens_sens90")
DCA_REFERENCE = ("nb_all", "nb_none")
WEIGHTED_HIDDEN = ("n", "events", "prevalence")     # what the sensitivity table shows as NA in a weighted row
# the groups of the split of sensitivity.csv; the split needs one row for each
SPLIT_GROUPS = {"CHECKUP_DAY": ("Checkup day", "Other day")}


# ------------------------------------------------------------------------------------------------ collecting
def _unknown(cell):
    """A masked or blank cell: its number is not known (never zero)."""
    return D.is_blank(cell) or D.is_masked(cell)


def _text(cell):
    return "" if D.is_blank(cell) else str(cell)


def _json_cell(v):
    """A JSON scalar as text, the way JSON writes it; None for null."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"thresholds.json: {v!r} is not a number")
    return json.dumps(v)


def _selector(sel):
    parts = []
    for col, v in sel.items():
        t = _text(v)
        if ";" in t or "\n" in t or "=" in col:
            raise ValueError(f"a row key cannot be written as a selector: {col}={t!r}")
        parts.append(f"{col}={t}")
    return ";".join(parts)


def _slug(text):
    """Key part of a level or group: '19-39' -> '19_39', '80+' -> '80plus', '(not measured)' -> 'not_measured'."""
    t = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower().replace("+", "plus")).strip("_")
    if not t:
        raise ValueError(f"no key part for {text!r}")
    return t


def _group_slug(label):
    """Key part of the row label of a dipstick group: '<=1.005' -> 'le1_005', '>=8.5' -> 'ge8_5' (the signs as
    Unicode in the label), '1.006-1.010' -> '1_006_1_010', 'Trace or 1+' -> 'trace_or_1plus'."""
    return _slug(str(label).replace(D.LE, "le").replace(D.GE, "ge"))


def _rows(name, ids):
    """(record, selector) pairs of an aggregate table, in file order; the columns `ids` must pick one row each."""
    df = D.read_table(name)
    out, seen = [], set()
    for rec in df.to_dict("records"):
        rec = {k: (None if D.is_blank(v) else v) for k, v in rec.items()}
        sel = {c: rec[c] for c in ids}
        k = tuple(_text(v) for v in sel.values())
        if k in seen:
            raise ValueError(f"{name}: more than one row with {dict(zip(ids, k))}")
        seen.add(k)
        out.append((rec, sel))
    return out


def _outcome_rows(name, ids):
    """_rows of a table with rows for each outcome: stops with LookupError, naming the table, when the outcomes in its
    rows are not exactly display.OUTCOMES, so that numbers.csv cannot lack the keys of one of the two enzymes."""
    recs = _rows(name, ids)
    found = {rec["outcome"] for rec, _ in recs}
    if found != set(D.OUTCOMES):
        raise LookupError(f"{name}: rows of the outcomes {sorted(map(str, found))}, expected {list(D.OUTCOMES)}")
    return recs


class Numbers:
    """The rows of numbers.csv, with counts of masked, blank and withheld values."""

    def __init__(self):
        self.rows, self.keys, self.tables = [], set(), []
        self.withheld = 0
        self.match_rows = set()     # MATCH rows of operating_points.csv still to be found in comparators_alert.csv

    def _add(self, key, fmt, unit, table, selector, columns, value, lo=None, hi=None, display=None):
        if not KEY.match(key):
            raise ValueError(f"key {key!r} is not a dotted id")
        if key in self.keys:
            raise ValueError(f"key {key!r} is given twice")
        if table not in self.tables:
            self.tables.append(table)
        row = {"key": key, "value": _text(value), "lo": _text(lo), "hi": _text(hi),
               "display": FORMATS[fmt](value, lo, hi) if display is None else display, "unit": unit,
               "source_file": table, "source_row": selector, "source_column": columns}
        if D.is_masked(value) and row["display"] != D.MASKED:
            raise ValueError(f"{key}: a masked cell must be shown as '<10'")
        for name, text in row.items():
            D.check_text(text, f"numbers.csv {key} ({name})", allow=INTERNAL_NAMES.get(name, ()))
        self.keys.add(key)
        self.rows.append(row)

    def cell(self, key, fmt, unit, table, rec, sel, column, ci=False):
        """One key from the cell `column` of a table row (with its `_lo` and `_hi` cells when ci)."""
        lo, hi = (rec[f"{column}_lo"], rec[f"{column}_hi"]) if ci else (None, None)
        cols = f"{column};{column}_lo;{column}_hi" if ci else column
        self._add(key, fmt, unit, table, _selector(sel), cols, rec[column], lo, hi)

    def share(self, key, table, rec, sel, total):
        """The pct cell of a row as value, displayed as the share n / total rounded once (display.fmt_share), as
        Tables 1, S4 and S5 show it; the pct cell must agree with n and total to its two decimals."""
        self._add(key, None, "%", table, _selector(sel), "pct", rec["pct"],
                  display=D.fmt_share(rec["n"], total, rec["pct"]))

    def derived(self, key, table, selector, terms, value, total=None, total_term=None):
        """A derived count (value: an int, or '<10') with its terms, 'n = <terms>'; with total, its share instead
        (display.fmt_share), 'pct = (<terms>) / <total_term>', value still the count."""
        if total is None:
            self._add(key, "count", "persons", table, selector, f"n = {terms}", value)
        else:
            self._add(key, None, "%", table, selector, f"pct = ({terms}) / {total_term}", value,
                      display=D.fmt_share(value, total))

    def json_cell(self, key, fmt, unit, path, obj, field):
        """One key from a field of thresholds.json: a JSON null gives a blank value, a missing field stops the run."""
        if not isinstance(obj, dict) or field not in obj:
            raise KeyError(f"thresholds.json: {'/'.join(path) or 'the top level'} has no field {field!r}")
        self._add(key, fmt, unit, "thresholds.json", f"path={'/'.join(path)}", field, _json_cell(obj[field]))

    def withhold(self, n=1):
        self.withheld += n

    def csv_text(self):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(self.rows)
        return buf.getvalue()


def _partition_hidden(members):
    """Columns to leave out for every row of a set of groups that split one total given elsewhere: the counts and
    the rates that carry them when a group's n is unknown; the events and those rates when only events are."""
    if any(_unknown(m["n"]) for m in members):
        return {"n", "events", "prevalence", "oe"}
    if any(_unknown(m["events"]) for m in members):
        return {"events", "prevalence", "oe"}
    return set()


# ------------------------------------------------------------------------------------------------ dipstick terms
def _combine(a, b, sign):
    """Terms a + sign x b (term -> coefficient), without the terms that cancel."""
    out = dict(a)
    for term, c in b.items():
        out[term] = out.get(term, 0) + sign * c
    return {term: c for term, c in out.items() if c != 0}


def _plan_terms(plan):
    """The derivation of each group count of an s08 level plan: {(column, group index): {(column, level): coefficient}}
    (level None: the column total), in the order of s08_tables.LevelPlan._resolve: the levels of the group when all
    are known; then, repeatedly, the column total less the other groups when one group of a column is open, and the
    relations between the columns (development cohort = training set + test set). A group shown as '<10' has none."""
    cols, n = list(plan.cells), len(plan.groups)
    terms = {}
    for k in cols:
        for i, group in enumerate(plan.groups):
            if all(T.is_int(plan.cells[k].get(lev, 0)) for lev in group):
                terms[(k, i)] = {(k, lev): 1 for lev in group if lev in plan.cells[k]}
    changed = True
    while changed:
        changed = False
        for k in cols:
            open_ = [i for i in range(n) if (k, i) not in terms]
            if len(open_) == 1:
                t = {(k, None): 1}
                for j in range(n):
                    if j != open_[0]:
                        t = _combine(t, terms[(k, j)], -1)
                terms[(k, open_[0])] = t
                changed = True
        for s, a, b in T.RELATIONS:
            if not all(x in plan.cells for x in (s, a, b)):
                continue
            for i in range(n):
                missing = [x for x in (s, a, b) if (x, i) not in terms]
                if len(missing) != 1:
                    continue
                if missing[0] == s:
                    terms[(s, i)] = _combine(terms[(a, i)], terms[(b, i)], 1)
                elif missing[0] == a:
                    terms[(a, i)] = _combine(terms[(s, i)], terms[(b, i)], -1)
                else:
                    terms[(b, i)] = _combine(terms[(s, i)], terms[(a, i)], -1)
                changed = True
    return terms


def _terms_value(terms, plan):
    """The count the terms give, from the cells and totals of the plan; None when a term is not a whole count."""
    total = 0
    for (k, lev), c in terms.items():
        v = plan.totals[k] if lev is None else plan.cells[k][lev]
        if not T.is_int(v):
            return None
        total += c * v
    return total


def _terms_text(terms, raw):
    """'DEV:0 + DEV:0.5 - DEV_TRAIN:0 - DEV_TRAIN:0.5': each term as <dataset>:<level as written> (<dataset>:persons
    for a column total), in the order of TERM_ORDER and the levels; '0' for no term."""
    def order(item):
        (k, lev), _ = item
        return TERM_ORDER.index(T.LEVEL_COLUMNS[k]), lev is not None, Decimal(lev) if lev is not None else 0
    words = []
    for (k, lev), c in sorted(terms.items(), key=order):
        ds = T.LEVEL_COLUMNS[k]
        name = f"{ds}:persons" if lev is None else f"{ds}:{raw[(ds, lev)]}"
        word = name if abs(c) == 1 else f"{abs(c)}*{name}"
        words.append((word if c > 0 else f"-{word}") if not words else f"{'+' if c > 0 else '-'} {word}")
    return " ".join(words) if words else "0"


# ------------------------------------------------------------------------------------------------ sections
def _merged_steps():
    """(source, step) of every step of cohort_flow.csv with a masked or blank reason: its reasons are left out."""
    reasons = {}
    for rec, _ in _rows("cohort_flow.csv", ("source", "key")):
        if rec["kind"] == "reason":
            reasons.setdefault((rec["source"], rec["parent"]), []).append(rec)
    return {p for p, rs in reasons.items() if any(_unknown(r["n"]) for r in rs)}


def flow(nb):
    name = "cohort_flow.csv"
    recs = _rows(name, ("source", "key"))
    finals = {rec["key"] for rec, _ in recs if rec["kind"] == "final"}
    missing = [f"{o}_{d}" for o in D.OUTCOMES for d in FLOW_FINALS if f"{o}_{d}" not in finals]
    if missing:
        raise LookupError(f"{name}: no final row for {missing}")
    merged = _merged_steps()
    for rec, sel in recs:
        base = f"flow.{rec['key']}"
        kind = rec["kind"]
        if kind == "step":
            nb.cell(f"{base}.n", "count", "persons", name, rec, sel, "n")
            if rec["parent"] is not None:
                nb.cell(f"{base}.excluded", "count", "persons", name, rec, sel, "excluded")
        elif kind == "reason":
            if (rec["source"], rec["parent"]) in merged:
                nb.withhold()
            else:
                nb.cell(f"{base}.n", "count", "persons", name, rec, sel, "n")
        elif kind == "final":
            nb.cell(f"{base}.n", "count", "persons", name, rec, sel, "n")
            nb.cell(f"{base}.events", "count", "persons", name, rec, sel, "events")
        else:
            raise ValueError(f"{name}: unknown kind {kind!r}")


def checks(nb):
    name = "cohort_checks.csv"
    recs = _rows(name, ("source", "item"))
    by_item = {(r["source"], r["item"]): (r, s) for r, s in recs}
    cells = {}
    for key, source, item, _ in CHECK_ITEMS:
        if (source, item) not in by_item:
            raise LookupError(f"{name}: no row {source} / {item!r}")
        cells[key] = by_item[(source, item)][0]["n"]
    gone = {k for part in CHECK_PARTS if any(_unknown(cells[x]) for x in part) for k in part}
    gone |= {k for step, part in CHECK_FLOW_PARTS.items() if step in _merged_steps() for k in part}
    for key, source, item, unit in CHECK_ITEMS:
        if key in gone:                     # one item of its set is masked or blank
            nb.withhold()
            continue
        rec, sel = by_item[(source, item)]
        nb.cell(f"checks.{key}", "count", unit, name, rec, sel, "n")
    for rec, sel in recs:
        m = SURVEY_RECORDS.match(rec["item"] or "")
        if rec["source"] == "KN" and m:
            nb.cell(f"checks.kn.survey_records_{m.group(1)}", "count", "records", name, rec, sel, "n")


def baseline(nb):
    name = "baseline_characteristics.csv"
    groups = {}
    for rec, sel in _rows(name, ("sample", "dataset", "variable", "level")):
        groups.setdefault((rec["sample"], rec["dataset"], rec["variable"]), []).append((rec, sel))
    # the denominator of every pct cell: the persons of its sample and dataset
    persons = {(s, d): D.count(m[0][0]["n"]) for (s, d, v), m in groups.items() if v == "persons" and len(m) == 1}
    lacking = [(d, v) for d in MEDIAN_DATASETS["ANY"] for o in D.OUTCOMES for v in (o.lower(), f"{o.lower()}_label")
               if ("ANY", d, v) not in groups]
    if lacking:
        raise LookupError(f"{name}: sample ANY has no rows of {lacking}")
    for (sample, dataset, var), members in groups.items():
        base = f"baseline.{sample}.{dataset}.{var}"
        if var == "persons":
            if len(members) != 1:
                raise ValueError(f"{name}: {sample} {dataset} has {len(members)} persons rows")
            rec, sel = members[0]
            nb.cell(f"{base}.n", "count", "persons", name, rec, sel, "n")
            continue
        if var not in CONTINUOUS and var not in CATEGORICAL:
            continue                        # dipstick inputs: dipstick() takes the rows of the tables
        if var in MEASURED and sorted(_text(rec["level"]) for rec, _ in members) != ["", UNMEASURED[var]]:
            raise LookupError(f"{name}: {sample} {dataset} {var} needs a measured and a {UNMEASURED[var]} row")
        hide = any(_unknown(rec["n"]) for rec, _ in members)
        for rec, sel in members:
            level = rec["level"]
            if level is None and var in CONTINUOUS:
                for stat, suffix, fmt, unit in CONTINUOUS[var]:
                    nb.cell(f"{base}.{stat}{suffix}", fmt, unit, name, rec, sel, stat)
                if var not in MEASURED:
                    continue
                part = base
            else:
                if level is None:
                    raise ValueError(f"{name}: {sample} {dataset} {var} has a row without a level")
                part = f"{base}.{LEVEL_KEYS.get(var, {}).get(level) or _slug(level)}"
            if hide:
                nb.withhold(2)
                continue
            if not isinstance(persons.get((sample, dataset)), int):
                raise LookupError(f"{name}: no persons count for {sample} {dataset}")
            nb.cell(f"{part}.n", "count", "persons", name, rec, sel, "n")
            nb.share(f"{part}.pct", name, rec, sel, persons[(sample, dataset)])


def dipstick(nb):
    """The dipstick rows of Table 1 and Table S1, for the one sample of those tables (DIPSTICK_KEYS, sample ANY): the
    groups of s08's level plans in the columns that have keys, the median [quartile 1, quartile 3] rows of urine pH and
    specific gravity (MEDIAN_DATASETS), and the eligible adults above a specific gravity of 1.030."""
    name = "baseline_characteristics.csv"
    rows = {}           # (sample, dataset, variable, level key or None) -> (record, selector)
    for rec, sel in _rows(name, ("sample", "dataset", "variable", "level")):
        if rec["sample"] in DIPSTICK_KEYS and (rec["variable"] in D.DIPSTICK or rec["variable"] == "persons"):
            lev = None if rec["level"] is None else D.level_key(rec["level"])
            rows[(rec["sample"], rec["dataset"], rec["variable"], lev)] = (rec, sel)
    src = T.Sources()
    for sample, keys in DIPSTICK_KEYS.items():
        plans, _ = T.level_plans(src, sample)
        for var, plan in plans.items():
            if var in D.DIPSTICK:
                _plan_keys(nb, name, rows, sample, var, plan, [k for k in plan.cells if k in keys])
        for ds in MEDIAN_DATASETS[sample]:
            for var, d in MEDIAN_INPUTS.items():
                if (sample, ds, var, None) not in rows:
                    raise LookupError(f"{name}: no {sample} {ds} {var} summary row")
                rec, sel = rows[(sample, ds, var, None)]
                for stat in ("median", "q1", "q3"):
                    nb.cell(f"baseline.{sample}.{ds}.{var}.{stat}", f"r{d}", "", name, rec, sel, stat)
        if sample == "ANY":
            _above_sg(nb, name, rows)
            _summary_keys(nb, name, rows, plans)


def _plan_keys(nb, name, rows, sample, var, plan, cols):
    """The .n and .pct keys of the groups of one level plan of `sample` in the columns `cols`: a group of one level
    with its own row from that row, any other group derived from the cells of the sample (_plan_terms)."""
    terms = _plan_terms(plan)
    lev_raw = {(d, lev): r["level"] for (s, d, v, lev), (r, _) in rows.items()
               if s == sample and v == var and lev is not None}
    shared = _selector({"sample": sample, "variable": var})
    for k in cols:
        ds, total = T.LEVEL_COLUMNS[k], plan.totals[k]
        for i, group in enumerate(plan.groups):
            base = f"baseline.{sample}.{ds}.{var}.{_group_slug(plan.label(i))}"
            if len(group) == 1 and (sample, ds, var, group[0]) in rows:     # one level with its own row
                rec, sel = rows[(sample, ds, var, group[0])]
                nb.cell(f"{base}.n", "count", "persons", name, rec, sel, "n")
                if not _unknown(rec["n"]):
                    nb.share(f"{base}.pct", name, rec, sel, total)
                continue
            t = terms.get((k, i))
            if t is None:                                                   # shown as '<10'
                if plan.texts[k][i] != D.MASKED:
                    raise ValueError(f"{sample} {var} {ds}: the group {plan.label(i)!r} has no count and is not "
                                     "'<10'")
                t = {(k, lev): 1 for lev in group if lev in plan.cells[k]}
                nb.derived(f"{base}.n", name, shared, _terms_text(t, lev_raw), D.MASKED)
                continue
            value = _terms_value(t, plan)
            if value is None or value != plan.values[k][i]:
                raise ValueError(f"{sample} {var} {ds}: the terms of {plan.label(i)!r} do not give the count of s08")
            if 1 <= value <= 9:
                raise ValueError(f"{sample} {var} {ds}: a pooled count of 1-9 follows from the shown counts")
            text = _terms_text(t, lev_raw)
            nb.derived(f"{base}.n", name, shared, text, str(value))
            nb.derived(f"{base}.pct", name, shared, text, str(value), total, f"{ds}:persons")


def _at_or_above_terms(plan, terms, k, cut):
    """The terms of the count of s08_tables.at_or_above in column k: the groups from the cut up, or the column total
    less the groups below it (terms: _plan_terms of the plan); with a group across the cut, the column total less the
    levels below the cut when all of them are known; None when the count stays unknown."""
    x = Decimal(cut)
    above, below, across = [], [], False
    for i, (group, v) in enumerate(zip(plan.groups, plan.values[k])):
        side = {Decimal(lev) >= x for lev in group}
        if len(side) != 1:
            across = True
        else:
            (above if side == {True} else below).append((i, v))
    if across:
        lows = [lev for lev in plan.cells[k] if Decimal(lev) < x]
        if lows and all(T.is_int(plan.cells[k][lev]) for lev in lows):
            return _combine({(k, None): 1}, {(k, lev): 1 for lev in lows}, -1)
        return None
    for part, start, sign in ((above, {}, 1), (below, {(k, None): 1}, -1)):
        if all(T.is_int(v) for _, v in part):
            t = start
            for i, _ in part:
                if (k, i) not in terms:
                    return None
                t = _combine(t, terms[(k, i)], sign)
            return t
    return None


def _summary_keys(nb, name, rows, plans):
    """The one row of each dipstick input of Table 1 (s08_tables.DIPSTICK_SUMMARY): the people at or above the cut of
    the row, in the columns of Table 1, written like a pooled group from the groups of the level plan (the sum of the
    groups from the cut up, or the column total less the groups below it). No key when the table shows NA (a group lies
    on both sides of the cut, or the count stays unknown) or a count of 1-9."""
    for var, cut in T.DIPSTICK_SUMMARY.items():
        plan = plans[var]
        terms = _plan_terms(plan)
        lev_raw = {(d, lev): r["level"] for (s, d, v, lev), (r, _) in rows.items()
                   if s == "ANY" and v == var and lev is not None}
        shared = _selector({"sample": "ANY", "variable": var})
        for k in T.BASELINE_TABLE_KEYS:
            ds, total = T.LEVEL_COLUMNS[k], plan.totals[k]
            base = f"baseline.ANY.{ds}.{var}.at_or_above"
            value = T.at_or_above(plan, k, cut)
            t = _at_or_above_terms(plan, terms, k, cut)
            if value is None or t is None:
                if value is not None:
                    raise ValueError(f"{var} {ds}: no terms for the count of its row in Table 1")
                nb.withhold(2)
                continue
            if _terms_value(t, plan) != value:
                raise ValueError(f"{var} {ds}: the terms of its row in Table 1 do not give the count of s08")
            if 1 <= value <= 9:             # Table 1 shows '<10'
                nb.withhold(2)
                continue
            text = _terms_text(t, lev_raw)
            nb.derived(f"{base}.n", name, shared, text, str(value))
            nb.derived(f"{base}.pct", name, shared, text, str(value), total, f"{ds}:persons")


def _above_sg(nb, name, rows):
    """The eligible adults above a specific gravity of 1.030 (sample ANY, KN; Table S1 shows them in intervals): the
    sum of those levels, withheld when one of them is masked."""
    ds = "KN"
    above = sorted(((lev, rec) for (s, d, v, lev), (rec, _) in rows.items()
                    if s == "ANY" and d == ds and v == "ua_sg" and lev is not None and Decimal(lev) > SG_TOP),
                   key=lambda x: Decimal(x[0]))
    if any(_unknown(rec["n"]) for _, rec in above):
        nb.withhold(2)
        return
    value = sum(D.count(rec["n"]) for _, rec in above)        # shown counts: 0 or 10 or more each
    text = " + ".join(f"{ds}:{rec['level']}" for _, rec in above) or "0"
    shared = _selector({"sample": "ANY", "variable": "ua_sg"})
    base = f"baseline.ANY.{ds}.ua_sg.above_1_030"
    nb.derived(f"{base}.n", name, shared, text, str(value))
    nb.derived(f"{base}.pct", name, shared, text, str(value), D.count(rows[("ANY", ds, "persons", None)][0]["n"]),
               f"{ds}:persons")


def train(nb):
    name = "train_summary.csv"
    for rec, sel in _outcome_rows(name, ("outcome",)):
        base = f"train.{rec['outcome']}"
        for col, fmt, unit in (("n_train", "count", "persons"), ("events_train", "count", "persons")):
            nb.cell(f"{base}.{col}", fmt, unit, name, rec, sel, col)
        # the events as a share of the people of the training set, as the table of the ensemble details shows them
        nb._add(f"{base}.prevalence_train", None, "%", name, _selector(sel), "prevalence_train",
                rec["prevalence_train"], display=D.fmt_share(rec["events_train"], rec["n_train"],
                                                             D.fmt_pct(rec["prevalence_train"], 2, None)))
        for col, fmt, unit in (("cv_auroc_lgb", "r3", ""), ("cv_auroc_xgb", "r3", ""), ("cv_auroc_cat", "r3", ""),
                               ("calibration_fold_events_min", "count", "persons"),
                               ("trials_per_learner", "int", "trials"), ("tuning_folds", "int", "folds"),
                               ("calibration_folds", "int", "folds"), ("seed", "int", "")):
            nb.cell(f"{base}.{col}", fmt, unit, name, rec, sel, col)


def performance(nb):
    name = "performance.csv"
    for rec, sel in _outcome_rows(name, ("outcome", "dataset", "model")):
        base = f"{rec['outcome']}.{rec['dataset']}.{rec['model']}"
        nb.cell(f"{base}.n", "count", "persons", name, rec, sel, "n")
        nb.cell(f"{base}.events", "count", "persons", name, rec, sel, "events")
        nb.cell(f"{base}.prevalence", "pct1", "%", name, rec, sel, "prevalence")
        nb.cell(f"{base}.mean_pred", "pct1", "%", name, rec, sel, "mean_pred")
        nb.cell(f"{base}.auroc", "r3", "", name, rec, sel, "auroc", ci=True)
        nb.cell(f"{base}.auprc", "r3", "", name, rec, sel, "auprc", ci=True)
        nb.cell(f"{base}.n_boot", "count", "resamples", name, rec, sel, "n_boot")


def calibration(nb):
    name = "calibration.csv"
    for rec, sel in _outcome_rows(name, ("outcome", "dataset", "model")):
        base = f"{rec['outcome']}.{rec['dataset']}.{rec['model']}"
        for col, fmt in (("oe", "r2"), ("citl", "r2"), ("slope", "r2"), ("slope_excl01", "r2"), ("brier", "r3"),
                         ("scaled_brier", "r3")):
            nb.cell(f"{base}.{col}", fmt, "", name, rec, sel, col, ci=True)
        for col in ("n_p0", "events_p0", "n_p1", "nonevents_p1"):
            nb.cell(f"{base}.{col}", "count", "persons", name, rec, sel, col)


def model_comparison(nb):
    name = "model_comparison.csv"
    for rec, sel in _outcome_rows(name, ("row_type", "outcome", "dataset", "model", "model_b")):
        od = f"{rec['outcome']}.{rec['dataset']}"
        if rec["row_type"] == "model":
            if rec["model"] == "ENS":
                continue                    # repeats performance.csv and calibration.csv
            base = f"{od}.{rec['model']}"
            nb.cell(f"{base}.auroc", "r3", "", name, rec, sel, "auroc", ci=True)
            for col, fmt in (("auprc", "r3"), ("brier", "r3"), ("scaled_brier", "r3"), ("oe", "r2"),
                             ("citl", "r2"), ("slope", "r2")):
                nb.cell(f"{base}.{col}", fmt, "", name, rec, sel, col)
        elif rec["row_type"] == "delta":
            base = f"{od}.{rec['model']}_vs_{rec['model_b']}"
            nb._add(f"{base}.delta_auroc", "d3", "", name, _selector(sel), "delta;delta_lo;delta_hi",
                    rec["delta"], rec["delta_lo"], rec["delta_hi"])
            nb.cell(f"{base}.delong_p", "p", "", name, rec, sel, "delong_p")
        else:
            raise ValueError(f"{name}: unknown row_type {rec['row_type']!r}")


def _operating(nb, name, base, rec, sel, diffs=False):
    for col in OP_RATES:
        nb.cell(f"{base}.{col}", "pct1", "%", name, rec, sel, col, ci=True)
    nb.cell(f"{base}.alerts_per_case", PER_CASE_FORMAT, "alerts per case", name, rec, sel, "alerts_per_case", ci=True)
    if diffs:
        for col in OP_DIFFS:
            nb.cell(f"{base}.{col}", "pp2", "percentage points", name, rec, sel, col, ci=True)


def operating_points(nb):
    name = "operating_points.csv"
    for rec, sel in _outcome_rows(name, ("outcome", "dataset", "model", "threshold_name")):
        if rec["threshold_name"].startswith("MATCH_"):
            # comparators_alert.csv repeats these rows, with the differences from the rule; comparators() checks
            # that each of them is there
            nb.match_rows.add((rec["outcome"], rec["dataset"], rec["model"], rec["threshold_name"]))
            continue
        base = f"{rec['outcome']}.{rec['dataset']}.{rec['model']}.{rec['threshold_name']}"
        _operating(nb, name, base, rec, sel)


def comparators(nb):
    name = "comparators_alert.csv"
    for rec, sel in _outcome_rows(name, ("outcome", "dataset", "strategy")):
        od = f"{rec['outcome']}.{rec['dataset']}"
        if rec["row_type"] == "rule":
            _operating(nb, name, f"{od}.{rec['strategy']}", rec, sel)
            if rec["dataset"] in KN_DATASETS:
                nb.cell(f"{od}.{rec['strategy']}.alert_rate_2dp", "pct2", "%", name, rec, sel, "alert_rate", ci=True)
        elif rec["row_type"] == "model":
            _operating(nb, name, f"{od}.{rec['model']}.{rec['threshold_name']}", rec, sel, diffs=True)
            nb.match_rows.discard((rec["outcome"], rec["dataset"], rec["model"], rec["threshold_name"]))
        else:
            raise ValueError(f"{name}: unknown row_type {rec['row_type']!r}")
    if nb.match_rows:
        raise LookupError(f"{name}: no row for {len(nb.match_rows)} MATCH row(s) of operating_points.csv, such as "
                          f"{sorted(nb.match_rows)[0]}")


def _threshold_fields(name):
    """The fields of thresholds.json that a threshold of this name holds besides 'threshold', with the
    format and unit of each key; a field that is missing stops the run (Numbers.json_cell)."""
    for start, fields in THRESHOLD_FIELDS:
        if name.startswith(start):
            return fields
    raise LookupError(f"thresholds.json: a threshold named {name!r} is not in THRESHOLD_FIELDS")


def _block(obj, field, where):
    """A block of thresholds.json that must be there (every outcome holds its rules and the three models)."""
    if not isinstance(obj, dict) or field not in obj:
        raise LookupError(f"thresholds.json: {where} has no {field!r}")
    return obj[field]


def thresholds(nb):
    t = D.read_json("thresholds.json")
    nb.json_cell("thresholds.n_boot", "count", "resamples", [], t, "n_boot")
    for o in D.OUTCOMES:
        out = _block(t, o, "the top level")
        for rule, obj in _block(out, "rules", o).items():
            nb.json_cell(f"thresholds.{o}.rules.{rule}.dev_test_alert_rate", "pct1", "%", [o, "rules", rule], obj,
                         "dev_test_alert_rate")
        for model in ("ENS", "LR_ALL", "LR_ASB"):
            for tname, obj in _block(out, model, o).items():
                base, path = f"thresholds.{o}.{model}.{tname}", [o, model, tname]
                nb.json_cell(f"{base}.threshold", "r4", "probability", path, obj, "threshold")
                nb.json_cell(f"{base}.threshold_pct", "pct2", "%", path, obj, "threshold")
                # the development test set alert rates to one decimal, as Table 3 and the supplementary tables show them
                for field, fmt, unit in _threshold_fields(tname):
                    nb.json_cell(f"{base}.{field}", fmt, unit, path, obj, field)


def _thousandths(cell):
    """A grid probability in whole thousandths ('0.048' -> 48); ValueError off the 0.001 grid."""
    x = Decimal(str(cell).strip()) * 1000
    q = x.quantize(Decimal(1), rounding=ROUND_HALF_UP)
    if abs(x - q) > Decimal("1e-6"):
        raise ValueError(f"decision_curve.csv: threshold_prob {cell!r} is not on a grid of 0.001")
    return int(q)


def decision_curve(nb):
    name = "decision_curve.csv"
    for rec, sel in _outcome_rows(name, ("outcome", "dataset", "threshold_prob")):
        x = D.num(rec["threshold_prob"])
        tag = next((t for t, p in DCA_POINTS if x is not None and abs(x - p) < 1e-9), None)
        with_ci = tag is not None
        if tag is None and rec["dataset"] == DCA_TAG_DATASET and x is not None:
            tag = f"t{_thousandths(rec['threshold_prob']):03d}"
        if tag is None:
            continue
        base = f"dca.{rec['outcome']}.{rec['dataset']}.{tag}"
        nb.cell(f"{base}.threshold_prob", "pct_grid", "%", name, rec, sel, "threshold_prob")
        for col in DCA_WITH_CI:
            nb.cell(f"{base}.{col}", "r4", "", name, rec, sel, col, ci=with_ci)
        for col in DCA_REFERENCE:
            nb.cell(f"{base}.{col}", "r4", "", name, rec, sel, col)


def _group_metrics(nb, name, base, rec, sel, hidden, prevalence_ci=False):
    for col, fmt, unit, ci in (("n", "count", "persons", False), ("events", "count", "persons", False),
                               ("prevalence", "pct1", "%", prevalence_ci), ("mean_pred", "pct1", "%", False),
                               ("auroc", "r3", "", True), ("oe", "r2", "", True)):
        if col in hidden:
            nb.withhold()
        else:
            nb.cell(f"{base}.{col}", fmt, unit, name, rec, sel, col, ci=ci)


def subgroups(nb):
    name = "subgroups.csv"
    parts = {}
    for rec, sel in _outcome_rows(name, ("outcome", "dataset", "model", "subgroup_var", "subgroup")):
        if rec["subgroup_var"] != "all":     # 'All' repeats performance.csv and calibration.csv
            parts.setdefault((rec["outcome"], rec["dataset"], rec["model"], rec["subgroup_var"]), []).append(
                (rec, sel))
    for (o, d, m, var), members in parts.items():
        hidden = _partition_hidden([r for r, _ in members])
        for rec, sel in members:
            base = f"subgroup.{o}.{d}.{m}.{var}.{_slug(rec['subgroup'])}"
            _group_metrics(nb, name, base, rec, sel, hidden)


def sensitivity(nb):
    name = "sensitivity.csv"
    recs = _outcome_rows(name, ("analysis", "outcome", "dataset", "model", "subgroup", "weighted"))
    parts, pairs = {}, {}
    for rec, _ in recs:
        a = rec["analysis"]
        if a in SPLIT_GROUPS:                        # groups that split the dataset
            parts.setdefault((a, rec["outcome"], rec["dataset"], rec["model"], rec["weighted"]), []).append(rec)
        elif a == "WEIGHTED":
            pairs.setdefault((rec["outcome"], rec["dataset"], rec["model"]), {})[D.flag(rec["weighted"])] = rec
        else:
            raise ValueError(f"{name}: unknown analysis {a!r}")
    for (a, o, d, m, w), members in parts.items():      # _partition_hidden reads every group of a split
        if sorted(_text(r["subgroup"]) for r in members) != sorted(SPLIT_GROUPS[a]):
            raise LookupError(f"{name}: {a} {o} {d} {m}{' weighted' if D.flag(w) else ''} needs one row for each of "
                              f"{' and '.join(SPLIT_GROUPS[a])}")
    for rec, sel in recs:
        a, o, d, m = rec["analysis"], rec["outcome"], rec["dataset"], rec["model"]
        if a == "WEIGHTED":
            pair = pairs[(o, d, m)]
            if set(pair) != {False, True}:
                raise LookupError(f"{name}: WEIGHTED {o} {d} {m} needs one unweighted and one weighted row")
            w = D.flag(rec["weighted"])
            hidden = set(WEIGHTED_HIDDEN) if w and T.weighted_hidden(pair[False], pair[True]) else set()
            base = f"sens.WEIGHTED.{o}.{d}.{m}.{'weighted' if w else 'unweighted'}"
            _group_metrics(nb, name, base, rec, sel, hidden, prevalence_ci=True)
        else:                                        # CHECKUP_DAY
            group = _slug(rec["subgroup"]) + ("_weighted" if D.flag(rec["weighted"]) else "")
            hidden = _partition_hidden(parts[(a, o, d, m, rec["weighted"])])
            _group_metrics(nb, name, f"sens.{a}.{o}.{d}.{m}.{group}", rec, sel, hidden, prevalence_ci=True)


def shap(nb):
    name = "shap_summary.csv"
    recs = _outcome_rows(name, ("outcome", "dataset", "input"))
    first = {}
    for rec, sel in recs:
        od = (rec["outcome"], rec["dataset"])
        if od not in first:
            first[od] = (rec, sel)
        elif (rec["n_records"], rec["n_learners"]) != (first[od][0]["n_records"], first[od][0]["n_learners"]):
            raise ValueError(f"{name}: n_records or n_learners differ within {od}")
    for (o, d), (rec, sel) in first.items():
        nb.cell(f"shap.{o}.{d}.n_records", "count", "persons", name, rec, sel, "n_records")
        nb.cell(f"shap.{o}.{d}.n_learners", "int", "learners", name, rec, sel, "n_learners")
    for rec, sel in recs:
        base = f"shap.{rec['outcome']}.{rec['dataset']}.{rec['input']}"
        nb.cell(f"{base}.rank", "int", "", name, rec, sel, "rank")
        nb.cell(f"{base}.mean_abs", "r3", "log-odds", name, rec, sel, "mean_abs_shap")


SECTIONS = (flow, checks, baseline, dipstick, train, performance, calibration, model_comparison, operating_points,
            comparators, thresholds, decision_curve, subgroups, sensitivity, shap)


def collect():
    nb = Numbers()
    for section in SECTIONS:
        section(nb)
    return nb


def main():
    out = D.out_path(*OUT_PARTS)
    if os.path.isfile(out):
        os.remove(out)              # first: a run that stops leaves no out-of-date numbers.csv
    nb = collect()
    D.write_text(out, nb.csv_text())
    masked = sum(1 for r in nb.rows if r["value"] == D.MASKED)
    blank = sum(1 for r in nb.rows if r["value"] == "")
    print(f"s11_numbers: {len(nb.rows):,} values from {len(nb.tables)} aggregate tables in {'/'.join(OUT_PARTS)} "
          f"({masked} masked, {blank} blank, {nb.withheld} withheld)")


if __name__ == "__main__":
    D.run(main)
