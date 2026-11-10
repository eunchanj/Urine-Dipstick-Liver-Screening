"""Tables 1-3 and the Supplementary Tables, from the aggregate tables only.

Main tables (manuscript/tables/ with tables_index.csv; the three together in manuscript/tables.md), each with both
outcomes in display order (display.OUTCOMES, ALT first) and with equal weight:
  Table 1 baseline     characteristics of the development training and test sets and of the eligible KNHANES adults
                       (sample ANY of baseline_characteristics.csv): age, sex, BMI, one row per dipstick input, ALT and
                       AST as medians and as the numbers above 40 U/L, and the descriptive row of both above 40 U/L
  Table 2 performance  AUROC of the three models with the three differences (the formal tests of the study: 3 pairs
                       x 2 data sets x 2 outcomes), and calibration, in four value columns: ALT and AST, in the
                       development test set and in KNHANES (performance, model_comparison, calibration)
  Table 3 comparators  per outcome, in KNHANES: the BMI rule, the model thresholds set in the development test set
                       to flag the proportion nearest to that of the rule, and the ensemble at the thresholds set by
                       the Youden index and for 80% and 90% sensitivity (comparators_alert.csv, operating_points.csv)
Supplementary Tables (supplement/tables/ with tables_index.csv): SI_ORDER below. Those that repeat by outcome give ALT
first. Tables have no footnotes: what a table says about its cells stands in its legend (class Legend).

Every number shown comes from an aggregate table. The script formats numbers and pools sparse levels: a pooled count is
the sum of its levels or, when one of them is masked, the column total minus the other groups, or the development
cohort minus the training set; its share is the count over the column total. plan_levels() chooses one grouping per
input for every table that shows its levels (level_plans(): Supplementary Table S1), so that no count of 1-9 is shown
and, within a table, no shown value lets one be worked out (from a column total, or between the development cohort
and its training and test sets); block_problems() and hide_for_block() do the same for the other groups of counts
(unknown_ranges(): what a reader can tell of each unknown cell of a group). Subtraction across tables is accepted.

Run (reads working/, writes the output folders next to it; bytecode writing off from the start):
    PYTHONDONTWRITEBYTECODE=1 python s08_tables.py
"""
import aggregate_guard  # noqa: F401  (first statement: refuses individual-level files, records the files opened)

import itertools
import math
import os
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import display as D

MASKED, GE, TIMES = D.MASKED, D.GE, D.TIMES
BASELINE, LEVELS, CHECKS = "baseline_characteristics.csv", "input_levels.csv", "cohort_checks.csv"
CODING = "knhanes_coding.csv"

MAIN_ORDER = {"baseline": 1, "performance": 2, "comparators": 3}
# the order of the Supplementary Tables (the citations and s10 take these numbers)
SI_ORDER = {"inputs": 1, "knhanes_codes": 2, "alcohol_codes": 3, "enzyme_availability": 4, "bmi_availability": 5,
            "bmi_source": 6, "enzyme_crosstab": 7, "ensemble_details": 8, "lr_coefficients": 9, "operating_points": 10,
            "comparator_differences": 11, "subgroups": 12, "sensitivity_analyses": 13, "enzyme_by_year": 14}
THRESHOLD_DECIMALS = 4      # every threshold (a predicted risk) is printed to 4 decimals, so that it can be applied
PER_CASE_DECIMALS = 2       # alerts per case (1/PPV), wherever shown
PEOPLE = D.BASELINE_LABELS["persons"]                       # 'People': displayed strings say people, never persons
# the one sentence that names the development cohort, here and in the Methods, Figure 1 and the table legends
DEV_POPULATION = ("adults with same-day ALT and AST results that could both be classified, complete dipstick results "
                  "and sex, and a BMI recorded on the index day or in the 30 days before it")
BMI_WINDOW = ("BMI recorded on the urinalysis day or, if none, the most recent in the 30 days before it (days -30 to 0 "
              "inclusive); none after the index day")
KN_ELIGIBLE = ("eligible KNHANES adults (aged 19 years or older; all dipstick fields and sex recorded; height, weight "
               "and BMI within the accepted ranges; ALT and AST both recorded)")


def si(table_id):
    """A supplementary table as cited anywhere ('Supplementary Table S6')."""
    return D.label("table", "si", SI_ORDER[table_id])


def main_table(table_id):
    """'Table 2' for a main table id."""
    return D.label("table", "main", MAIN_ORDER[table_id])


# ------------------------------------------------------------------------------------------------ small formats
def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def lc(text):
    """First letter in lower case (labels used inside a sentence), acronyms kept."""
    return text if text[:2].isupper() else text[:1].lower() + text[1:]


def cap(text):
    """First letter in upper case (a label of display.DATASET_LONG at the start of a sentence)."""
    return text[:1].upper() + text[1:]


def join_and(items):
    """The items joined by commas, with 'and' (and no comma) before the last one."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def threshold_text(cell):
    """A threshold (predicted risk) with THRESHOLD_DECIMALS decimals."""
    return D.fmt(cell, THRESHOLD_DECIMALS)


def sig(text, digits=3):
    """A hyperparameter value: whole numbers as written, other numbers to `digits` significant digits, text as is."""
    t = str(text).strip()
    try:
        d = Decimal(t)
    except InvalidOperation:
        return t
    if not d.is_finite() or d == d.to_integral_value() and "." not in t and "e" not in t.lower():
        return t
    if d == 0:
        return "0"
    q = d.quantize(Decimal(1).scaleb(d.adjusted() - digits + 1), rounding=ROUND_HALF_UP)
    if q.adjusted() > d.adjusted():          # rounding carried into a new digit (0.0999 -> 0.100)
        q = q.quantize(Decimal(1).scaleb(q.adjusted() - digits + 1), rounding=ROUND_HALF_UP)
    return format(q, "f") if q.as_tuple().exponent <= 0 else format(q.quantize(Decimal(1)), "f")


def n_with_pct(n, total, pct_cell=None):
    """'12,345 (12)', '456 (2.9)'; '<10' alone (display.fmt_n_share): the share is the count over the column total,
    rounded once as a percentage that describes people; the pct cell of the aggregate, when given, must agree with the
    count and total to its two decimals. A share at or above 99.95% and below 100%, which at one decimal would read as
    a whole hundred, is NEAR_WHOLE ('12,345 (>99.9)'; share_note says so). A total that is itself masked gives the count
    alone."""
    if not is_int(total):
        return D.fmt_count(n)
    shown = D.fmt_n_share(n, total, pct_cell)
    exact = D.share_pct(n, total) if " (" in shown else None
    if exact is not None and Decimal("99.95") <= exact < 100:
        return f"{shown.split(' (', 1)[0]} ({NEAR_WHOLE})"
    return shown


def n_of_total(n, total):
    """'n/N (%)' from a count and its total; '<10' or NA alone (a total that is masked: the count alone)."""
    shown = n_with_pct(n, total)
    if shown in (MASKED, D.NA) or " (" not in shown:
        return shown
    n_txt, pct_txt = shown.split(" (", 1)
    return f"{n_txt}/{D.fmt_count(total)} ({pct_txt}"


def section(label, ncol, style="**"):
    return [f"{style}{label}{style}"] + [""] * (ncol - 1)


def col_widths(header, rows, cap=40):
    """Separator widths for md_table: the longest cell of a column (the label column capped), at least its longest
    word."""
    def plain(s):
        return re.sub(r"\^[^^]*\^|\*", "", s)
    out = []
    for j in range(len(header)):
        cells = [plain(r[j]) for r in rows if r[j]]
        longest = max([len(c) for c in cells] + [3])
        word = max([len(w) for c in cells + [plain(header[j])] for w in c.split()] + [3])
        out.append(max(word, min(longest, cap if j == 0 else 28)))
    return out


def table(header, rows, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    return D.md_table(header, rows, align=align, widths=col_widths(header, rows))


class Legend:
    """Sentences of a table legend, each kept once, in the order of first use: the tables have no footnotes, so what a
    table says about its cells stands in its legend."""

    def __init__(self):
        self.sentences = []

    def add(self, sentence):
        if sentence not in self.sentences:
            self.sentences.append(sentence)


# NA in the Abbreviations line of a table that prints it, unless a sentence of its legend that opens 'NA:' defines it;
# a table may give NA a text of its own (abbreviations, na=)
NA_TEXT = {"NA": "not available"}
NA_COUNTS_ONLY = "not estimated (counts only)"
SHARE_NOTE = "A share shown as <0.1 is above 0 and below 0.05%."
NEAR_WHOLE = ">99.9"         # a share at or above 99.95% and below 100% (n_with_pct)
NEAR_WHOLE_NOTE = f"A share shown as {NEAR_WHOLE} is below 100% and at or above 99.95%."
# why a cell shows NA or a legend sentence leaves a count out, one sentence each (the tables that can show them say so
# in their legends); NA is named only where a cell of the table prints it and the Abbreviations line, or a legend
# sentence that opens 'NA:', defines it
HIDDEN_NOTE = "A count is withheld where, with the totals of the table, it would let a count of 1-9 be worked out."
HIDDEN_NA_NOTE = ("NA in place of a count: withheld, since with the totals of the table it would let a count of 1-9 "
                  "be worked out.")
SMALL_STATS_NOTE = "NA in a row of means, medians or quartiles: fewer than 100 values."


def shows_na(rows):
    """True when a cell of `rows` shows NA."""
    return any(c == D.NA for r in rows for c in r[1:])


def abbreviations(keys, extra=None, rows=(), na=NA_TEXT["NA"]):
    """'Abbreviations: AUROC, area under ...; CI, confidence interval.' (display.ABBREVIATIONS and `extra`); NA is
    added, with the text `na`, when a cell of `rows` shows it, and left out with na=None (a legend sentence that opens
    'NA:' defines it)."""
    texts = dict(D.ABBREVIATIONS)
    texts.update({"NA": na} if na else {})
    texts.update(extra or {})
    keys = sorted(set(keys) | ({"NA"} if na and shows_na(rows) else set()), key=str.lower)
    return "Abbreviations: " + "; ".join(f"{k}, {texts[k]}" for k in keys) + "."


def share_note(rows):
    """What a share shown as '<0.1' (display.fmt_share) or as NEAR_WHOLE (n_with_pct) means, for the shares that the
    cells of `rows` show, in one paragraph; an empty string when they show neither."""
    cells = [c for r in rows for c in r]
    return " ".join(note for mark, note in (("(<0.1)", SHARE_NOTE), (f"({NEAR_WHOLE})", NEAR_WHOLE_NOTE))
                    if any(mark in c for c in cells))


single = D.single       # the one value that many rows share (n_boot of a table, for example)


def boot_note(n_boot):
    return f"95% CIs: percentile bootstrap with {D.fmt_count(n_boot)} resamples."


# the two outcomes, as the notes of every table that shows them say it
OUTCOME_NOTE = ("ALT >40 U/L and AST >40 U/L are two separate binary outcomes in the same people, each with its own "
                "models and thresholds.")
# in the notes of every table that shows or names events
EVENTS_NOTE = "Events: people with the outcome."


# ------------------------------------------------------------------------------------------------ aggregate reader
class Sources:
    """Aggregate tables read once through display.read_table."""

    def __init__(self):
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            self.cache[name] = D.read_table(name)
        return self.cache[name]

    def rows(self, name, **equal):
        return D.select(self(name), **equal).to_dict("records")

    def one(self, name, **equal):
        return D.one(self(name), **equal)


# ------------------------------------------------------------------------------------------------ blocks of counts
# A block is a group of count cells whose total a reader knows (a column total, an All row, a data set). The reader
# sees each shown cell, '<10' for a count of 1-9, NA for a blank cell or for a cell hidden here, and the total.
def is_small(v):
    """A masked cell ('<10'), or a count of 1-9 (shown as '<10')."""
    return v == MASKED or (is_int(v) and 1 <= v <= 9)


def unknown_ranges(values, total, hide=()):
    """{index: (lo, hi)}: the counts that each unknown cell of a block can hold for a reader who knows the block's
    total and its shown cells. A masked cell holds 1-9; a blank cell (None) any count; a cell hidden by
    hide_for_block (its index in `hide`) 10 or more, since a count of 1-9 is shown as '<10' and a 0 is never hidden
    there (a hidden 0 counts as known). For the masked cells this is the rule of display.masked_sum_range and
    display.masked_follow. ValueError when the cells cannot add up to the total."""
    t = int(total)
    bounds, known = {}, 0
    for i, v in enumerate(values):
        if is_small(v):
            bounds[i] = (1, 9)
        elif v is None:
            bounds[i] = (0, t)
        elif i in hide and v != 0:
            bounds[i] = (10, t)
        else:
            known += v
    resid = t - known
    low, high = sum(a for a, _ in bounds.values()), sum(b for _, b in bounds.values())
    if not low <= resid <= high:
        raise ValueError("the cells of a block cannot add up to its total")
    return {i: (max(a, resid - (high - b)), min(b, resid - (low - a))) for i, (a, b) in bounds.items()}


def block_problems(shown, total, hidden=()):
    """Why a masked shown count, or a count of 1-9 among the cells of the block that are not shown at all (`hidden`;
    a hidden 0 counts as known), could be worked out from the shown `total` (empty list when none can). `shown`: the
    cells shown (int, '<10', or None for NA)."""
    shown, rest = list(shown), [v for v in hidden if v != 0]
    ranges = unknown_ranges(shown + [None] * len(rest), total)
    fixed = {i for i, (lo, hi) in ranges.items() if lo == hi}
    out = []
    if any(i in fixed and is_small(v) for i, v in enumerate(shown)):
        out.append("a masked count follows from the total")
    if any(len(shown) + j in fixed and is_small(v) for j, v in enumerate(rest)):
        out.append("a hidden count of 1-9 follows from the total")
    return out


def require_block(shown, total, hidden, where):
    """Stop, naming `where`, when block_problems() finds a count that could be worked out, or when the cells of the
    block cannot add up to its total."""
    try:
        problems = block_problems(shown, total, hidden)
    except ValueError as e:                  # the cells cannot add up to the total
        raise ValueError(f"{where}: {e}") from None
    if problems:
        raise ValueError(f"{where}: {problems[0]}; a display decision is needed")


def hide_for_block(values, total, hide=()):
    """Indexes of the cells of a block to show as NA (those of `hide` and further ones), so that no unknown cell
    (masked, blank or hidden) can be worked out from the block's total (unknown_ranges): while one of them has a
    single possible count, the smallest shown count of 10 or more is hidden too. A 0 is never hidden, since a reader
    who knows that the hidden cells hold 0 or 10 or more could then tell it."""
    hide = set(hide)
    while True:
        if all(lo < hi for lo, hi in unknown_ranges(values, total, hide).values()):
            return hide
        free = [i for i, v in enumerate(values) if i not in hide and is_int(v) and v >= 10]
        if not free:
            raise ValueError("a block cannot be shown without revealing a masked count")
        hide.add(min(free, key=lambda i: values[i]))


def hidden_cells(values, total, where, hide=()):
    """hide_for_block (with the cells of `hide` hidden from the start), naming `where` when no cell can be hidden to
    protect a masked count (a display decision) or the cells cannot add up to the total."""
    try:
        return hide_for_block(values, total, hide)
    except ValueError as e:
        raise ValueError(f"{where}: {e}; a display decision is needed") from None


def weighted_hidden(unweighted, weighted):
    """True when the weighted row of a WEIGHTED pair of sensitivity.csv (the supplementary table of the sensitivity
    analyses) shows its n, events and prevalence as NA: a count of the pair is masked or blank (unknown, never 0), or
    the records without a positive weight, their events or their non-events number 1-9. s11_numbers withholds the same
    cells."""
    counts = [D.count(r[c]) for r in (unweighted, weighted) for c in ("n", "events")]
    if not all(is_int(c) for c in counts):
        return True
    dn, de = counts[0] - counts[2], counts[1] - counts[3]
    return any(1 <= x <= 9 for x in (dn, de, dn - de))


# ------------------------------------------------------------------------------------------------ pooled levels
# Columns that show the levels of an input, their totals, and the relation a reader can use between them:
# DEV (development cohort, Supplementary Table S1) is the training set plus the test set, level by level.
LEVEL_COLUMNS = {"TR": "DEV_TRAIN", "TE": "DEV_TEST", "KN": "KN", "DEV": "DEV"}
RELATIONS = (("DEV", "TR", "TE"),)
GRADED_SHOWN = D.GRADED + ("ua_nitrite",)
NUMERIC_INPUTS = ("ua_pH", "ua_sg")
PLAN_INPUTS = GRADED_SHOWN + NUMERIC_INPUTS + ("male",)


def _num(level):
    try:
        return Decimal(D.level_key(level))
    except InvalidOperation:
        return None


def partition(universe, ranges):
    """Levels in order, split into groups: the levels inside one inclusive (lo, hi) range form a group."""
    groups, last = [], None
    for lev in universe:
        x = _num(lev)
        rng = next((r for r in ranges if x is not None and r[0] <= x <= r[1]), None)
        if rng is not None and rng == last:
            groups[-1].append(lev)
        else:
            groups.append([lev])
        last = rng
    return [tuple(g) for g in groups]


def sg_ranges(universe):
    """Display ranges of urine specific gravity: KNHANES records steps of 0.001 and the hospital strip steps of 0.005,
    so values are shown in intervals of 0.005 ending at a strip step (the first interval holds 1.000-1.005)."""
    out = set()
    step, base = Decimal("0.005"), Decimal("1.000")
    for lev in universe:
        x = _num(lev)
        if x is None:
            continue
        hi = base + step * max(1, math.ceil((x - base) / step))
        out.add((base if hi == base + step else hi - step + Decimal("0.001"), hi))
    return tuple(sorted(out))


def _compositions(total, k):
    """k-tuples of whole numbers 1-9 that sum to total."""
    if k == 0:
        if total == 0:
            yield ()
        return
    for first in range(1, 10):
        rest = total - first
        if k - 1 <= rest <= 9 * (k - 1):
            for tail in _compositions(rest, k - 1):
                yield (first,) + tail


def reader_check(texts, totals, limit=500000, where="levels"):
    """Cells shown '<10' that a reader can work out from the shown counts, the column totals and the relations
    between the columns (RELATIONS): a list of (column, group index). Its errors name `where` (the table and input)
    and, when the counts of one column do not add up, that column."""
    keys = list(texts)
    ngroups = len(next(iter(texts.values())))
    unknown = {k: [i for i, t in enumerate(texts[k]) if t == MASKED] for k in keys}
    known = {k: [None if t == MASKED else int(t.replace(",", "")) for t in texts[k]] for k in keys}
    options = {}
    for k in keys:
        resid = totals[k] - sum(v for v in known[k] if v is not None)
        options[k] = list(_compositions(resid, len(unknown[k])))
        if not options[k]:
            raise ValueError(f"{where}: the shown counts of {LEVEL_COLUMNS.get(k, k)} do not add up to its total")
    active = [k for k in keys if unknown[k]]
    if math.prod(len(options[k]) for k in active) > limit:
        raise ValueError(f"{where}: too many masked cells to check")
    rel = [r for r in RELATIONS if all(x in texts for x in r)]
    seen = {(k, i): set() for k in active for i in unknown[k]}
    feasible = False
    for combo in itertools.product(*(options[k] for k in active)):
        full = {k: list(known[k]) for k in keys}
        for k, vals in zip(active, combo):
            for i, v in zip(unknown[k], vals):
                full[k][i] = v
        if any(full[s][i] != full[a][i] + full[b][i] for s, a, b in rel for i in range(ngroups)):
            continue
        feasible = True
        for k in active:
            for i in unknown[k]:
                seen[(k, i)].add(full[k][i])
    if not feasible:
        raise ValueError(f"{where}: the shown counts contradict the relations between the columns")
    return [ki for ki, vals in seen.items() if len(vals) == 1]


class LevelPlan:
    """Display groups of one input, shared by every table that shows its levels (sample ANY of
    baseline_characteristics.csv: Supplementary Table S1). base: the groups given by the starting ranges (the display
    intervals of specific gravity)."""

    def __init__(self, name, universe, groups, cells, totals, pcts, base=(), sample="ANY"):
        self.name, self.universe, self.groups, self.base = name, universe, groups, tuple(base)
        self.cells, self.totals, self.pcts, self.sample = cells, totals, pcts, sample
        self.where = f"{BASELINE} {sample} {name}"        # the table, sample and input, for the error messages
        self.values = self._resolve()
        self.texts = {k: [self._text(k, i) for i in range(len(groups))] for k in cells}

    def _parts(self, key, i):
        return [self.cells[key].get(lev, 0) for lev in self.groups[i]]

    def _resolve(self):
        val = {k: [sum(p) if all(is_int(x) for x in p) else None
                   for p in (self._parts(k, i) for i in range(len(self.groups)))] for k in self.cells}
        changed = True
        while changed:
            changed = False
            for k, row in val.items():
                open_ = [i for i, v in enumerate(row) if v is None]
                if len(open_) == 1:
                    row[open_[0]] = self.totals[k] - sum(v for v in row if v is not None)
                    changed = True
            for s, a, b in RELATIONS:
                if not all(x in val for x in (s, a, b)):
                    continue
                for i in range(len(self.groups)):
                    trio = [val[s][i], val[a][i], val[b][i]]
                    if trio.count(None) != 1:
                        continue
                    if trio[0] is None:
                        val[s][i] = trio[1] + trio[2]
                    elif trio[1] is None:
                        val[a][i] = trio[0] - trio[2]
                    else:
                        val[b][i] = trio[0] - trio[1]
                    changed = True
        for k, row in val.items():
            for i, v in enumerate(row):
                if v is None:
                    continue
                parts = self._parts(k, i)
                known = sum(x for x in parts if is_int(x))
                masked = sum(1 for x in parts if x == MASKED)
                if not known + masked <= v <= known + 9 * masked:
                    raise ValueError(f"{self.where}: the levels of {LEVEL_COLUMNS.get(k, k)} do not add up "
                                     "to its total")
        for s, a, b in RELATIONS:
            if all(x in val for x in (s, a, b)):
                for i in range(len(self.groups)):
                    trio = (val[s][i], val[a][i], val[b][i])
                    if None not in trio and trio[0] != trio[1] + trio[2]:
                        raise ValueError(f"{self.where}: {LEVEL_COLUMNS[s]} is not {LEVEL_COLUMNS[a]} plus "
                                         f"{LEVEL_COLUMNS[b]}")
        return val

    def _text(self, key, i):
        v = self.values[key][i]
        if v is not None:
            return D.fmt_count(v)
        parts = self._parts(key, i)
        if sum(1 for x in parts if x == MASKED) == 1 and all(x == 0 or x == MASKED for x in parts):
            return MASKED
        return None

    def problems(self):
        """Group indexes that cannot be shown, or whose '<10' a reader could work out."""
        bad = [i for k in self.texts for i, t in enumerate(self.texts[k]) if t is None]
        return bad or [i for _, i in reader_check(self.texts, self.totals, where=self.where)]

    def held(self, i):
        """The model values of group i, in order (a row that s01 counted together holds several)."""
        return D.ordered_levels(v for lev in self.groups[i] for v in D.held_levels(self.name, lev))

    def pooled(self):
        """Groups pooled because levels are sparse: every group of several levels or holding a row that s01 counted
        together, except the display intervals of specific gravity as they were given."""
        if self.name == "ua_sg":
            return [g for g in self.groups if g not in self.base]
        return [g for i, g in enumerate(self.groups) if len(self.held(i)) > 1]

    def label(self, i):
        universe = self.universe if self.name in NUMERIC_INPUTS else None
        return D.pooled_label(self.name, self.held(i), universe)

    def show(self, key, i):
        """'n (%)' of group i in column key."""
        text = self.texts[key][i]
        if text == MASKED:
            return MASKED
        g = self.groups[i]
        cell = self.cells[key].get(g[0], 0) if len(g) == 1 else None
        pct = self.pcts[key].get(g[0]) if len(g) == 1 and is_int(cell) and cell > 0 else None
        return n_with_pct(self.values[key][i], self.totals[key], pct)


def plan_levels(name, cells, totals, pcts, ranges=(), sample="ANY"):
    """One grouping of the levels of an input for all the columns given (cells: {column: {level: int or '<10'}}) of
    one sample: start from `ranges` (display.SPARSE_POOLS, or the intervals of specific gravity) and merge a group
    that cannot be shown, or whose '<10' could be worked out, with its smaller neighbour until every column can be
    shown."""
    universe = D.ordered_levels(lev for c in cells.values() for lev in c)
    groups = base = partition(universe, ranges)
    while True:
        plan = LevelPlan(name, universe, groups, cells, totals, pcts, base, sample)
        bad = plan.problems()
        if not bad:
            return plan
        if len(groups) < 2:
            raise ValueError(f"{sample} {name}: no grouping of its levels can be shown")
        i = bad[0]
        size = [sum(c.get(lev, 0) for c in cells.values() for lev in g if is_int(c.get(lev, 0))) for g in groups]
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if size[i - 1] <= size[i + 1] else i + 1
        a, b = sorted((i, j))
        groups = groups[:a] + [groups[a] + groups[b]] + groups[b + 1:]


def level_plans(src, sample="ANY"):
    """(plans, codes): the LevelPlan of every input of PLAN_INPUTS, in that order, for the rows of `sample` of
    baseline_characteristics.csv, and the KNHANES codes of each level. Every plan covers the columns of LEVEL_COLUMNS
    (the numeric inputs only DEV and KN), so that the relation between them (training set + test set = development
    cohort) is used and checked. The counts of DEV and KN must equal input_levels.csv, which also gives the codes."""
    totals = {k: D.count(src.one(BASELINE, sample=sample, dataset=ds, variable="persons", level=None)["n"])
              for k, ds in LEVEL_COLUMNS.items()}
    plans, codes = {}, {}
    for name in PLAN_INPUTS:
        cells, pcts = {}, {}
        for k, ds in LEVEL_COLUMNS.items():
            rows = [r for r in src.rows(BASELINE, sample=sample, dataset=ds, variable=name) if r["level"] is not None]
            cells[k] = {D.level_key(r["level"]): D.count(r["n"]) for r in rows}
            pcts[k] = {D.level_key(r["level"]): r["pct"] for r in rows}
        for source, k in (("DEV", "DEV"), ("KN", "KN")):
            rows = src.rows(LEVELS, source=source, input=name)
            if {D.level_key(r["level"]): D.count(r["n"]) for r in rows} != cells[k]:
                raise ValueError(f"{LEVELS} {source} {name} differs from {BASELINE} ({LEVEL_COLUMNS[k]})")
        kn_rows = src.rows(LEVELS, source="KN", input=name)
        codes[name] = {D.level_key(r["level"]): r["kn_codes"] for r in kn_rows}
        keys = ("DEV", "KN") if name in NUMERIC_INPUTS else tuple(LEVEL_COLUMNS)
        cells = {k: cells[k] for k in keys}
        universe = D.ordered_levels(lev for c in cells.values() for lev in c)
        ranges = sg_ranges(universe) if name == "ua_sg" else D.SPARSE_POOLS.get(name, ())
        plans[name] = plan_levels(name, cells, {k: totals[k] for k in keys}, {k: pcts[k] for k in keys}, ranges, sample)
    return plans, codes


def pooled_note(plans, names):
    """'urine bilirubin, trace or above; urine urobilinogen, negative or normal and 2+ or above' (empty when no input
    has pooled levels)."""
    parts = []
    for name in names:
        plan = plans[name]
        labels = [lc(plan.label(i)) for i, g in enumerate(plan.groups) if g in plan.pooled()]
        if labels:
            parts.append(f"{lc(D.INPUT_LABELS[name])}, {join_and(labels)}")
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------ Table 1
BASE_KEYS = {"TR": "DEV_TRAIN", "TE": "DEV_TEST", "KN": "KN"}
SOURCE_OF = {"TR": "DEV", "TE": "DEV", "KN": "KN"}
BASELINE_TABLE_KEYS = ("TR", "TE", "KN")        # the columns of Table 1
TABLE1_SAMPLE = "ANY"                           # Table 1: the people of the development cohort and KNHANES adults
# the sample of baseline_characteristics.csv that Table 1 describes -> (target, id); a dict of one entry, so that the
# sample cannot silently share a key with another
BASELINE_TABLES = {TABLE1_SAMPLE: ("main", "baseline")}
# each dipstick input as one row of Table 1: the levels at or above this model value (for bilirubin and urobilinogen
# 1+ and above, urobilinogen level 0.5 holding the normal readings of the hospital records and KNHANES codes 0 and 1;
# nitrite has one positive level). The levels themselves are in Supplementary Table S1
DIPSTICK_SUMMARY = {"ua_glucose": "0.5", "ua_protein": "0.5", "ua_ketone": "0.5", "ua_bilirubin": "1",
                    "ua_blood": "0.5", "ua_urobilinogen": "1", "ua_nitrite": "1"}
ENZYME_ROWS = (("alt", "alt_label"), ("ast", "ast_label"))
DIPSTICK_NA_NOTE = ("NA in a dipstick row: not shown, because levels with 1-9 people lie on both sides of the cut of "
                    "that row in that column.")


def summary_label(name):
    """The row label of a dipstick input in Table 1: 'Urine glucose, trace or above', 'Urine nitrite, positive'."""
    cut = Decimal(DIPSTICK_SUMMARY[name])
    levels = [lev for lev in D.ordered_levels(D.LEVEL_LABELS[name]) if _num(lev) >= cut]
    return f"{D.INPUT_LABELS[name]}, {lc(D.pooled_label(name, levels))}"


def at_or_above(plan, key, cut):
    """People of column `key` at or above model value `cut` of the input of `plan`, from the groups of its level plan:
    the sum of the groups from the cut up, or the column total less the groups below it; with a group across the cut
    (its levels are shown together elsewhere), the column total less the levels below the cut, when all of them are
    known. None when the count stays unknown."""
    x = Decimal(cut)
    above, below, across = [], [], False
    for g, v in zip(plan.groups, plan.values[key]):
        nums = [_num(lev) for lev in g]
        if all(n >= x for n in nums):
            above.append(v)
        elif all(n < x for n in nums):
            below.append(v)
        else:
            across = True
    if across:
        lows = [c for lev, c in plan.cells.get(key, {}).items() if _num(lev) is not None and _num(lev) < x]
        return plan.totals[key] - sum(lows) if lows and all(is_int(c) for c in lows) else None
    if all(is_int(v) for v in above):
        return sum(above)
    if all(is_int(v) for v in below):
        return plan.totals[key] - sum(below)
    return None


def baseline_table(src, plans, keys, target, sample=TABLE1_SAMPLE):
    """Rows and legend of Table 1 (keys TR, TE, KN of sample TABLE1_SAMPLE; plans: the level plans of the sample, for
    the one row of each dipstick input). Every row it shows or checks is read with row() (display.one), so that a
    missing row stops the script naming the table and the row."""
    legend = Legend()
    ncol = len(keys) + 1
    dev_cols = [k for k in keys if SOURCE_OF[k] == "DEV"]
    kn_cols = [k for k in keys if SOURCE_OF[k] == "KN"]

    def row(k, variable, level=None):
        return src.one(BASELINE, sample=sample, dataset=BASE_KEYS[k], variable=variable, level=level)

    def levels(k, variable):
        return {r["level"]: r for r in src.rows(BASELINE, sample=sample, dataset=BASE_KEYS[k], variable=variable)
                if r["level"] is not None}

    def where(k, what):
        return f"{BASELINE} {sample} {BASE_KEYS[k]} {what}"

    persons = {k: D.count(row(k, "persons")["n"]) for k in keys}

    def level_cell(k, variable, level):
        """'n (%)' of one level of the table's sample in column k."""
        r = row(k, variable, level)
        return n_with_pct(r["n"], persons[k], r["pct"])

    rows = [[f"{PEOPLE}, n"] + [D.fmt_count(persons[k]) for k in keys]]
    rows.append(["Age, years, mean (SD)"] + [D.mean_sd(row(k, "age")["mean"], row(k, "age")["sd"], 0) for k in keys])

    def category_block(title, variable, order, labels, only=None):
        """Rows of the levels in `order` (labels from `labels`) of one categorical variable, under a section row
        unless title is None (they then belong to the section row above them)."""
        if title is not None:
            rows.append(section(title, ncol))
        for k in keys:
            if only and SOURCE_OF[k] != only:
                continue
            shown = [D.count(row(k, variable, x)["n"]) for x in order]
            hidden = [D.count(r["n"]) for x, r in levels(k, variable).items() if x not in order]
            require_block(shown, persons[k], hidden, where(k, variable))
        for x in order:
            cells = ["" if (only and SOURCE_OF[k] != only) else level_cell(k, variable, x) for k in keys]
            rows.append([labels[x]] + cells)

    band_labels = dict(D.AGE_BAND_LABELS)
    if kn_cols:
        legend.add("KNHANES records ages of 80 years or more as 80.")
    category_block("Age group, n (%)", "age_band", list(D.AGE_BAND_LABELS), band_labels)

    women = []
    for k in keys:
        men = D.count(row(k, "male", "1")["n"])
        require_block([D.count(row(k, "male", "0")["n"])], persons[k], [men], where(k, "women"))
        women.append(level_cell(k, "male", "0"))
    rows.append(["Women, n (%)"] + women)

    cells = []
    for k in keys:
        r = row(k, "bmi")
        if D.count(row(k, "bmi", "(missing)")["n"]) != 0 or D.count(r["n"]) != persons[k]:
            raise ValueError(f"{where(k, 'bmi')}: every person of the analysis population has a BMI")
        cells.append(D.mean_sd(r["mean"], r["sd"], 1))
    rows.append([f"{D.BASELINE_LABELS['bmi']}, mean (SD)"] + cells)

    rows.append(section("Urine dipstick results, n (%)", ncol))
    for name in DIPSTICK_SUMMARY:
        cells = []
        for k in keys:
            n = at_or_above(plans[name], k, DIPSTICK_SUMMARY[name])
            if n is None:
                legend.add(DIPSTICK_NA_NOTE)
                cells.append(D.NA)
                continue
            require_block([n], persons[k], [persons[k] - n], where(k, name))
            cells.append(n_with_pct(n, persons[k]))
        rows.append([summary_label(name)] + cells)
    for name, d in (("ua_pH", 1), ("ua_sg", 3)):
        cells = []
        for k in keys:
            r = row(k, name)
            cells.append(D.median_iqr(r["median"], r["q1"], r["q3"], d))
        rows.append([f"{D.INPUT_LABELS[name]}, {D.MEDIAN_LABEL}"] + cells)

    def enzyme_rows(variable, label_variable):
        """The median and the count above 40 U/L of one enzyme: every person of the population has a same-day value
        and a label (the '(not measured)' rows of baseline_characteristics.csv are 0 here)."""
        cells_m, cells_e = [], []
        for k in keys:
            r = row(k, variable)
            if D.count(row(k, variable, "(not measured)")["n"]) != 0 or D.count(r["n"]) != persons[k]:
                raise ValueError(f"{where(k, variable)}: every person of the analysis population has a value")
            cells_m.append(D.median_iqr(r["median"], r["q1"], r["q3"], 1))
            cells_e.append(event_cell(k, label_variable))
        return cells_m, cells_e

    def event_cell(k, variable):
        """'n (%)' of the people above 40 U/L in column k ('(not measured)' must be 0)."""
        pos, neg, gone = (row(k, variable, x) for x in ("1", "0", "(not measured)"))
        if D.count(gone["n"]) != 0:
            raise ValueError(f"{where(k, variable)}: '(not measured)' is not 0 in the analysis population")
        require_block([D.count(pos["n"])], persons[k], [D.count(neg["n"])], where(k, variable))
        return n_with_pct(pos["n"], persons[k], pos["pct"])

    rows.append(section("ALT and AST", ncol))
    for variable, label_variable in ENZYME_ROWS:
        medians, events = enzyme_rows(variable, label_variable)
        rows.append([f"{D.BASELINE_LABELS[variable]}, {D.MEDIAN_LABEL}"] + medians)
        rows.append([f"{D.BASELINE_LABELS[label_variable]}, n (%)"] + events)
    rows.append([f"{D.BASELINE_LABELS['both_label']}, n (%)"] + [event_cell(k, "both_label") for k in keys])

    if dev_cols:
        cells = []
        for k in keys:
            if k not in dev_cols:
                cells.append("")
                continue
            other = D.count(row(k, "checkup_day", "0")["n"])
            require_block([D.count(row(k, "checkup_day", "1")["n"])], persons[k], [other], where(k, "checkup day"))
            cells.append(level_cell(k, "checkup_day", "1"))
        rows.append(section(f"{D.BASELINE_LABELS['index_urinalysis']}, n (%)", ncol))
        rows.append([D.BASELINE_LABELS["checkup_day"]] + cells)
        category_block(None, "urine_period", list(D.URINE_PERIOD_LABELS),
                       {k: f"In {v}" for k, v in D.URINE_PERIOD_LABELS.items()}, only="DEV")
    if kn_cols:
        category_block(f"{D.BASELINE_LABELS['period']}, n (%)", "period", list(D.PERIOD_LABELS), D.PERIOD_LABELS,
                       only="KN")
    return rows, legend


def write_baseline(src, plans, sample=TABLE1_SAMPLE):
    """Table 1 (sample TABLE1_SAMPLE: the people of the development cohort and of the eligible KNHANES adults); plans:
    the level plans of that sample (level_plans)."""
    target, table_id = BASELINE_TABLES[sample]
    keys = BASELINE_TABLE_KEYS
    rows, legend = baseline_table(src, plans, keys, target, sample)
    header = ["Characteristic"] + [D.DATASET_LABELS[BASE_KEYS[k]] for k in keys]
    notes = [(f"Values are n (%) unless stated otherwise. Development training and test sets: {DEV_POPULATION}, "
              "split person by person within the four combinations of ALT and AST above or not above 40 U/L (hospital "
              f"records). KNHANES column: {KN_ELIGIBLE}, in the surveys of 2016 to 2024. The index urinalysis is each "
              "person's first urinalysis in the hospital records."),
             (f"The urine dipstick results are shown as one row per input ({si('inputs')} shows the levels, the model "
              "values and the KNHANES codes); the cut of each row is named in the row, and urobilinogen level 0.5 "
              "holds the normal readings of the hospital records and KNHANES codes 0 (negative) and 1 (trace). The "
              "ALT and AST rows are given in the same form; the row of both enzymes above 40 U/L is descriptive and is "
              "not an outcome.")]
    notes += [share_note(rows)] + legend.sentences + [abbreviations(["ALT", "AST", "BMI", "KNHANES", "SD"], rows=rows)]
    title = ("Characteristics of the development training and test sets (hospital records) and of "
             f"{D.DATASET_LONG['KN']}")
    order = (MAIN_ORDER if target == "main" else SI_ORDER)[table_id]
    return dict(kind="table", id=table_id, title=title, markdown=table(header, rows), notes=notes, sources=[BASELINE],
                target=target, order=order)


# ------------------------------------------------------------------------------------------------ Table S1
def write_inputs(src, plans, codes):
    header = ["Input and level", "Model value", "KNHANES codes", f"{D.DATASET_LABELS['DEV']}, n (%)",
              f"{D.DATASET_LABELS['KN']}, n (%)"]
    rows = []
    for name in GRADED_SHOWN + NUMERIC_INPUTS + ("male",):
        plan = plans[name]
        rows.append(section(D.INPUT_LABELS[name] if name != "male" else "Sex", len(header), "*"))
        for i, g in enumerate(plan.groups):
            if name in NUMERIC_INPUTS:
                value, code = "", ""
            else:
                value = ", ".join(plan.held(i))
                code = ", ".join(c.strip() for x in g for c in (codes[name].get(x) or "").split(",") if c.strip())
            label = plan.label(i) if name != "male" else D.LEVEL_LABELS["male"][g[0]]
            rows.append([label, value, code, plan.show("DEV", i), plan.show("KN", i)])
    pooled = pooled_note(plans, PLAN_INPUTS)
    sizes = plans["male"].totals                # the column totals, which the level plans already count as known
    notes = [(f"Development cohort: {D.fmt_count(sizes['DEV'])} people, the training and test sets together (hospital "
              f"records). KNHANES: {D.fmt_count(sizes['KN'])} {KN_ELIGIBLE}. Model value: the value an input takes in "
              "the models; dipstick grades are coded 0 (negative), 0.5 (trace), 1, 2 and 3 (1+ to 3+ or above), and "
              "urobilinogen level 0.5 holds the normal readings of the hospital records and KNHANES codes 0 (negative) "
              "and 1 (trace). KNHANES codes: the survey codes seen at that level among eligible adults (sex: 1, men; "
              "2, women). Age and BMI enter the models as recorded (KNHANES records ages of 80 years or more as 80).")]
    if pooled:
        notes.append(f"Levels with few people are shown together so that no cell shows 1-9 people: {pooled}; a "
                     f"pooled row lists its model values and codes. {si('knhanes_codes')} gives the codes by "
                     "survey year.")
    notes.append("Urine pH and specific gravity enter the models as recorded. Specific gravity is recorded in steps "
                 "of 0.001 in KNHANES and in strip steps of 0.005 in the hospital records, apart from rare lower "
                 "values that are shown with the lowest level; it is shown here in intervals of 0.005 that end at a "
                 "strip step.")
    notes += [share_note(rows), abbreviations(["ALT", "AST", "BMI", "KNHANES"], rows=rows)]
    title = ("Inputs of the models: levels, model values and KNHANES codes, with counts of people by level for the "
             f"development cohort and for {D.DATASET_LONG['KN']}")
    return dict(kind="table", id="inputs", title=title, markdown=table(header, rows, "lllrr"), notes=notes,
                sources=[LEVELS, BASELINE], target="si", order=SI_ORDER["inputs"])


# ------------------------------------------------------------------------------------------------ Tables S2 and S3
CODE_INPUTS = D.GRADED + ("ua_nitrite", "ua_pH")
ADULTS_NA_NOTE = ("NA: withheld, since with the other counts of the survey year it would let a count of 1-9 be "
                  "worked out.")         # it defines NA, so the Abbreviations line leaves NA out


def code_order(code):
    x = _num(code)
    return (x is None, x if x is not None else Decimal(0), code)


def write_knhanes_codes(src):
    rows_all = src.rows(CODING, section="dipstick")
    years = sorted({r["year"] for r in rows_all}, key=int)
    header = ["KNHANES code", "Model value"] + years
    rows = []
    for name in CODE_INPUTS:
        part = [r for r in rows_all if r["target"] == name]
        if not part:
            continue
        variables = sorted({r["variable"] for r in part})
        rows.append(section(f"{D.INPUT_LABELS[name]} ({', '.join(variables)})", len(header), "*"))
        hide_zero = name != "ua_pH"
        codes = sorted({r["code"] for r in part if r["code"] != "blank" and not (hide_zero and r["code"] == "0")},
                       key=code_order)
        for code in codes:
            maps = {r["maps_to"] for r in part if r["code"] == code}
            if len(maps) != 1:
                raise ValueError(f"{name} code {code}: more than one model value")
            m = maps.pop()
            if m == "missing":
                value = D.MISSING_LABEL
            elif name == "ua_pH":
                value = D.level_label(name, m)
            elif name == "ua_urobilinogen" and code == "1":
                # code 1 is a trace reading, whose model value is the level of the normal readings
                value = f"{D.level_key(m)} (Trace)"
            else:
                value = f"{D.level_key(m)} ({D.level_label(name, m)})"
            cells = []
            for y in years:
                hit = [r for r in part if r["year"] == y and r["code"] == code]
                if len(hit) > 1:
                    raise ValueError(f"knhanes_coding.csv: {len(hit)} rows for {name} code {code} in {y}")
                cells.append(D.fmt_count(hit[0]["n_adults"]) if hit else "0")
            rows.append([code, value] + cells)
    sg = src.rows(CODING, section="dipstick", target="ua_sg")
    fine = all(any(_num(r["code"]) is not None and (_num(r["code"]) * 1000) % 5 != 0
                   and D.count(r["n_adults"]) != 0 for r in sg if r["year"] == y) for y in years)
    sg_text = ("Urine specific gravity is recorded as a value in steps of 0.001 in every survey year and enters the "
               "models as recorded." if fine else "Urine specific gravity enters the models as recorded.")
    # the total over all years gives no year's negative readings: the rows of each input leave at least one code a year
    # unlisted beside its masked cells
    adults = D.fmt_count(src.one("cohort_flow.csv", key="kn_adult")["n"])
    n_years = {9: "nine"}.get(len(years), len(years))
    notes = [("Adults (aged 19 years or older) in each survey year with each urine dipstick code, as written in the "
              f"survey files, and the value it takes in the models ({adults} adults over the {n_years} survey years). "
              "Code 0 (negative readings, which for urobilinogen are the normal readings) and blank results are not "
              "listed, since with the number of adults in a survey year they would give the counts of 1-9 (shown as "
              "<10) by subtraction. " + sg_text)]
    if shows_na(rows):
        notes.append(ADULTS_NA_NOTE)
    notes.append(abbreviations(["KNHANES"], rows=rows, na=None))
    title = "KNHANES urine dipstick codes by survey year"
    return dict(kind="table", id="knhanes_codes", title=title, markdown=table(header, rows, "ll" + "r" * len(years)),
                notes=notes, sources=[CODING, "cohort_flow.csv"], target="si", order=SI_ORDER["knhanes_codes"])


CODE_SECTIONS = (("alcohol", "Frequency of alcohol drinking"), ("eligibility", "Pregnancy"))
GROUP_LABELS = {"alcohol": D.DRINK_LABELS, "eligibility": D.MEANING_LABELS}


def code_blocks_hidden(blocks, where):
    """For one survey year, {block: indexes of the cells to show as NA} of the blocks of Supplementary Table S3
    (block: its cells as int, '<10' or None). The alcohol and pregnancy blocks count the same adults of the year, so a
    reader knows that their sums are equal: when the sums leave one possible total (a block shown in full against a
    block with one '<10', or two ranges that meet in one value), each block is checked against that total with
    hide_for_block."""
    spans = {}
    for name, values in blocks.items():
        known = sum(v for v in values if is_int(v) and not is_small(v))
        small = sum(1 for v in values if is_small(v))
        spans[name] = (known + small, None if any(v is None for v in values) else known + 9 * small)
    tops = [hi for _, hi in spans.values() if hi is not None]
    total = max(lo for lo, _ in spans.values())
    if not tops or min(tops) != total:      # no single total (or blocks with different adults: nothing follows)
        return {}
    return {name: hidden_cells(values, total, f"{where} {name}") for name, values in blocks.items()}


def write_alcohol_codes(src):
    """The KNHANES alcohol frequency codes with the group of each (the alcohol groups of the subgroup results), and the
    pregnancy codes. Every code seen in a survey file is listed, with the adults of each survey year (n_adults is
    complete in knhanes_coding.csv); the two blocks count the same adults of a year, so code_blocks_hidden keeps a '<10'
    of one from following from the other, and a cell hidden so shows NA (ADULTS_NA_NOTE)."""
    rows_all = [r for r in src.rows(CODING) if r["section"] in dict(CODE_SECTIONS)]
    years = sorted({r["year"] for r in rows_all}, key=int)
    header = ["KNHANES code", "Group"] + years
    order, cells = {}, {}
    for sect, title in CODE_SECTIONS:
        part = [r for r in rows_all if r["section"] == sect]
        order[sect] = sorted({r["code"] for r in part}, key=code_order)
        for code in order[sect]:
            for y in years:
                hit = [r for r in part if r["year"] == y and r["code"] == code]
                if len(hit) > 1:
                    raise ValueError(f"{CODING}: {len(hit)} rows for {title} code {code} in {y}")
                cells[(sect, code, y)] = D.count(hit[0]["n_adults"]) if hit else 0
    hidden = set()
    for y in years:
        blocks = {sect: [cells[(sect, c, y)] for c in order[sect]] for sect in order}
        for sect, hide in code_blocks_hidden(blocks, f"{CODING} {y}").items():
            hidden |= {(sect, order[sect][i], y) for i in hide}
    rows = []
    for sect, title in CODE_SECTIONS:
        part = [r for r in rows_all if r["section"] == sect]
        variables = sorted({r["variable"] for r in part})
        rows.append(section(f"{title} ({', '.join(variables)})", len(header), "*"))
        for code in order[sect]:
            maps = {r["maps_to"] for r in part if r["code"] == code}
            if len(maps) != 1:
                raise ValueError(f"{title} code {code}: more than one group")
            group = GROUP_LABELS[sect][maps.pop()]
            rows.append(["Blank" if code == "blank" else code, group] + [
                D.NA if (sect, code, y) in hidden else D.fmt_count(cells[(sect, code, y)]) for y in years])
    notes = [("Adults (aged 19 years or older) in each survey year with each code, as written in the survey files, "
              "and the group or meaning it has in the analyses. The groups of frequency of alcohol drinking are "
              f"the alcohol groups of {si('subgroups')}. Pregnancy is counted and is not an exclusion in either data "
              "source (the hospital records have no information on pregnancy).")]
    if shows_na(rows):
        notes.append(ADULTS_NA_NOTE)
    notes.append(abbreviations(["KNHANES"], rows=rows, na=None))
    title = "KNHANES codes of the frequency of alcohol drinking and of pregnancy, by survey year"
    return dict(kind="table", id="alcohol_codes", title=title, markdown=table(header, rows, "ll" + "r" * len(years)),
                notes=notes, sources=[CODING], target="si", order=SI_ORDER["alcohol_codes"])


# ------------------------------------------------------------------------------------------------ Tables S4 and S5
def contrast_rows(src, samples, legend, bmi_block, enzymes):
    """Rows of a descriptive contrast of two samples of baseline_characteristics.csv (dataset DEV), one column each:
    people, age, sex and the index urinalysis, with `bmi_block` the rows of BMI recorded or not, and with `enzymes` the
    ALT and AST rows of those with both enzymes. Every level of a variable shown is a row of its own, so that
    hide_for_block can keep a count of 1-9 from following from the column total: such a cell is NA (HIDDEN_NA_NOTE), and
    so is a mean, median or quartile of fewer than 100 values (SMALL_STATS_NOTE)."""
    ncol = len(samples) + 1
    totals = {s: D.count(src.one(BASELINE, sample=s, dataset="DEV", variable="persons", level=None)["n"])
              for s in samples}

    def level_rows(s, variable):
        return {r["level"]: r for r in src.rows(BASELINE, sample=s, dataset="DEV", variable=variable)}

    def stat_row(label, variable, fmt_cell):
        cells = []
        for s in samples:
            r = level_rows(s, variable).get(None)
            text = D.NA if r is None or r["n"] is None or r["median"] is None else fmt_cell(r)
            if text == D.NA:
                legend.add(SMALL_STATS_NOTE)
            cells.append(text)
        return [label] + cells

    def block(title, variable, levels, labels):
        """A section row (unless title is None) and one row per level of `levels` ('(missing)' and '(not measured)'
        only when some column has people there)."""
        data = {s: level_rows(s, variable) for s in samples}
        special = ("(missing)", "(not measured)")
        kept = [lev for lev in levels if lev not in special
                or any(lev in data[s] and D.count(data[s][lev]["n"]) != 0 for s in samples)]
        columns = {}
        for s in samples:
            values = [D.count(data[s][lev]["n"]) if lev in data[s] else 0 for lev in kept]
            hide = hidden_cells(values, totals[s], f"{BASELINE} {s} {variable}") if is_int(totals[s]) else set()
            cells = []
            for i, (lev, v) in enumerate(zip(kept, values)):
                if i in hide or v is None:
                    legend.add(HIDDEN_NA_NOTE)
                    cells.append(D.NA)
                elif is_small(v):
                    cells.append(MASKED)
                else:
                    cells.append(n_with_pct(v, totals[s], data[s][lev]["pct"] if lev in data[s] else None))
            columns[s] = cells
        rows = [] if title is None else [section(title, ncol)]
        rows += [[labels[lev]] + [columns[s][i] for s in samples] for i, lev in enumerate(kept)]
        return rows

    rows = [[f"{PEOPLE}, n"] + [D.fmt_count(totals[s]) for s in samples]]
    rows.append(stat_row("Age, years, mean (SD)", "age", lambda r: D.mean_sd(r["mean"], r["sd"], 0)))
    rows += block("Age group, n (%)", "age_band", list(D.AGE_BAND_LABELS), D.AGE_BAND_LABELS)
    rows += block("Sex, n (%)", "male", ["0", "1", "(missing)"],
                  {"0": "Women", "1": "Men", "(missing)": D.MISSING_LABEL})
    if bmi_block:
        rows += block("BMI, n (%)", "bmi", [None, "(missing)"], {None: "Recorded", "(missing)": D.MISSING_LABEL})
    rows.append(stat_row(f"{D.BASELINE_LABELS['bmi']}, {D.MEDIAN_LABEL}", "bmi",
                         lambda r: D.median_iqr(r["median"], r["q1"], r["q3"], 1)))
    if enzymes:
        for variable, label_variable in ENZYME_ROWS:
            rows.append(stat_row(f"{D.BASELINE_LABELS[variable]}, {D.MEDIAN_LABEL}", variable,
                                 lambda r: D.median_iqr(r["median"], r["q1"], r["q3"], 1)))
            rows += block(None, label_variable, ["1", "0", "(not measured)"],
                          {"1": D.BASELINE_LABELS[label_variable] + ", n (%)",
                           "0": D.BASELINE_LABELS[label_variable].replace(">40", "not above 40") + ", n (%)",
                           "(not measured)": D.NOT_MEASURED + ", n (%)"})
    rows += block(f"{D.BASELINE_LABELS['index_urinalysis']}, n (%)", "checkup_day", ["1", "0"],
                  {"1": D.BASELINE_LABELS["checkup_day"], "0": "On another day"})
    rows += block(None, "urine_period", list(D.URINE_PERIOD_LABELS),
                  {k: f"In {v}" for k, v in D.URINE_PERIOD_LABELS.items()})
    return rows


def undetermined_hidden(src, items, values):
    """Indexes of `items` (the availability counts of availability_note) to leave out when the adults with both enzymes
    whose results could not be classified number 1-9 on a day or over both days: they are the adults with both enzymes
    less the first column of the table, day by day, so the two 'both ALT and AST' cells go; and on such a day they are
    also the second column less the adults with one enzyme or neither, so that one of those counts goes when all three
    are known."""
    both = "both ALT and AST"
    days = (("a check-up day", "1"), ("another day", "0"))

    def column(sample):
        return {d: D.count(src.one(BASELINE, sample=sample, dataset="DEV", variable="checkup_day", level=lev)["n"])
                for d, lev in days}

    def value(group, day):
        return values[items.index((group, day))]

    first, second = column("E1A_ELIGIBLE"), column("E1A_NOT")
    gaps = {d: value(both, d) - first[d] for d, _ in days if is_int(value(both, d)) and is_int(first[d])}
    if not (any(1 <= g <= 9 for g in gaps.values()) or 1 <= sum(gaps.values()) <= 9):
        return set()
    hide = {items.index((both, d)) for d, _ in days}
    for d, _ in days:
        if d in gaps and 1 <= gaps[d] <= 9 and is_int(second[d]):
            others = [("ALT only", d), ("AST only", d), ("neither ALT nor AST", d)]
            cells = [None] + [value(*x) for x in others]    # None: the adults whose results could not be classified
            for i in hidden_cells(cells, second[d], f"enzyme availability, {d}"):
                if i > 0:
                    hide.add(items.index(others[i - 1]))
    return hide


def availability_note(src, legend):
    """The sentence on the adults with only one enzyme: the counts of cohort_checks.csv (persons at dev_adult with a
    same-day ALT, AST or neither, by the day of the urinalysis), against the adults at that step; a count that would
    give a count of 1-9 from that total, or from the two columns of the table (undetermined_hidden), is left out, and
    the legend says so. Then the two rows of RULE that count the adults with complete inputs and a BMI that the
    requirement of both enzymes left out, unless both are 0."""
    adults = D.count(src.one("cohort_flow.csv", key="dev_adult")["n"])
    groups = (("both ALT and AST", "both ALT and AST"), ("ALT only", "ALT only"), ("AST only", "AST only"),
              ("neither ALT nor AST", "neither"))
    days = ("a check-up day", "another day")
    items = [(g, d) for g, _ in groups for d in days]
    values = [D.count(src.one(CHECKS, source="DEV", item=f"persons at dev_adult with {g} on the urine day, {d}")["n"])
              for g, d in items]
    hide = hidden_cells(values, adults, "enzyme availability among adults", undetermined_hidden(src, items, values))
    shown = {it: ("withheld" if i in hide else D.fmt_count(v)) for i, (it, v) in enumerate(zip(items, values))}
    if hide and HIDDEN_NA_NOTE not in legend.sentences:     # a cell printing NA gives the reason already
        legend.add(HIDDEN_NOTE)
    parts = [f"{word}, {shown[(g, days[0])]} and {shown[(g, days[1])]}" for g, word in groups]
    results = (f"Same-day results of the {D.fmt_count(adults)} adults at this step, on a health check-up day and on "
               f"another day: {'; '.join(parts)}.")
    only = {w: D.count(src.one(CHECKS, source="RULE", item=f"adults with complete inputs and a BMI with {w} only")["n"])
            for w in ("ALT", "AST")}
    if all(v == 0 for v in only.values()):          # no one with complete inputs and a BMI was left out for this
        return results
    return (f"{results} Among the adults with complete inputs and a BMI, {D.fmt_count(only['ALT'])} had ALT only and "
            f"{D.fmt_count(only['AST'])} had AST only, and these people were left out by the requirement of both "
            "enzymes.")


def write_enzyme_availability(src):
    legend = Legend()
    samples = ("E1A_ELIGIBLE", "E1A_NOT")
    rows = contrast_rows(src, samples, legend, bmi_block=True, enzymes=False)
    header = ["Characteristic", "ALT and AST both classifiable", "Not both classifiable"]
    undetermined = D.count(src.one("cohort_flow.csv", key="dev_enzyme_undetermined")["n"])
    clause = "" if undetermined == 0 else (" (the second column also holds adults with both enzymes when a value "
                                           "could not be classified)")
    notes = [("Values are n (%) unless stated otherwise. Adults in the hospital records with a first urinalysis in the "
              "study period, a blood test on the day of the urinalysis and an age of 19-110 years: those with same-day "
              f"ALT and AST results that could both be classified, against the others{clause}. "
              f"{availability_note(src, legend)} The index urinalysis is each person's first urinalysis in the "
              "hospital records.")]
    notes += [share_note(rows)] + legend.sentences + [abbreviations(["ALT", "AST", "BMI", "SD"], rows=rows)]
    title = ("Adults with same-day ALT and AST results that could both be classified, and adults without them, in the "
             "hospital records")
    return dict(kind="table", id="enzyme_availability", title=title, markdown=table(header, rows), notes=notes,
                sources=[BASELINE, CHECKS, "cohort_flow.csv"], target="si", order=SI_ORDER["enzyme_availability"])


def write_bmi_availability(src):
    legend = Legend()
    samples = ("E1B_BMI", "E1B_NO_BMI")
    rows = contrast_rows(src, samples, legend, bmi_block=False, enzymes=True)
    header = ["Characteristic", "BMI recorded", "No BMI recorded"]
    notes = [(f"Values are n (%) unless stated otherwise. Adults with same-day ALT and AST results that could both be "
              "classified and with complete dipstick results and sex (hospital records), with and without a BMI "
              f"recorded on the index day or in the 30 days before it; those with a BMI are the development cohort of "
              f"{main_table('baseline')}. {OUTCOME_NOTE} The index urinalysis is each person's first urinalysis in "
              "the hospital records.")]
    notes += [share_note(rows)] + legend.sentences + [abbreviations(["ALT", "AST", "BMI", "SD"], rows=rows)]
    title = ("Adults with and without a recent BMI among those with both enzymes and complete dipstick results and "
             "sex, in the hospital records")
    return dict(kind="table", id="bmi_availability", title=title, markdown=table(header, rows), notes=notes,
                sources=[BASELINE], target="si", order=SI_ORDER["bmi_availability"])


# ------------------------------------------------------------------------------------------------ Tables S6 and S7
def write_bmi_source(src):
    """The timing and source of the BMI record of each person of the development cohort, and the share of index
    urinalyses on a health check-up day by period of the urinalysis (cohort_checks.csv)."""
    legend = Legend()
    header = ["Item", f"{D.DATASET_LABELS['DEV']}, n (%)"]
    total = D.count(src.one(BASELINE, sample="ANY", dataset="DEV", variable="persons", level=None)["n"])

    def count_of(item):
        return D.count(src.one(CHECKS, source="DEV", item=f"development cohort persons {item}")["n"])

    def block(title, items):
        values = [count_of(item) for _, item in items]
        hide = hidden_cells(values, total, f"{CHECKS} {title}")
        out = [section(title, 2)]
        for i, ((label, _), v) in enumerate(zip(items, values)):
            if i in hide:
                legend.add(HIDDEN_NA_NOTE)
            out.append([label, D.NA if i in hide else n_with_pct(v, total)])
        return out

    rows = block("Timing of the BMI record", (
        ("On the day of the index urinalysis", "with a BMI on the urine day"),
        ("1-7 days before it", "with a BMI 1-7 days before the urine day"),
        ("8-30 days before it", "with a BMI 8-30 days before the urine day")))
    rows += block("Source of the BMI record", (
        ("A record that names a health check-up", "whose BMI comes from a record naming a check-up"),
        ("Another record", "whose BMI comes from another record")))
    rows.append(section("Index urinalysis on a health check-up day, by period of the urinalysis, n/N (%)", 2))
    for period, label in D.URINE_PERIOD_LABELS.items():
        people = count_of(f"with a urine test in {period}")
        day = count_of(f"with a urine test in {period} on a check-up day")
        other = people - day if is_int(people) and is_int(day) else None
        hide = hidden_cells([day, other], people, f"{CHECKS} check-up days {period}") if is_int(people) else set()
        if 0 in hide:
            legend.add(HIDDEN_NA_NOTE)
        rows.append([label, D.NA if 0 in hide else n_of_total(day, people)])
    notes = [(f"Development cohort (hospital records): {D.fmt_count(total)} {DEV_POPULATION}. {BMI_WINDOW}. "
              "The timing and source describe the BMI record used for each person; several records on one day: the "
              "record that names a health check-up first. Percentages of the shares by period are of the people "
              "with an index urinalysis in that period.")]
    notes += [share_note(rows)] + legend.sentences + [abbreviations(["ALT", "AST", "BMI"], rows=rows)]
    title = ("Timing and source of the BMI record, and index urinalyses on a health check-up day by period of the "
             "urinalysis, in the development cohort")
    return dict(kind="table", id="bmi_source", title=title, markdown=table(header, rows), notes=notes,
                sources=[CHECKS, BASELINE], target="si", order=SI_ORDER["bmi_source"])


def write_enzyme_crosstab(src):
    """ALT above 40 U/L by AST above 40 U/L in the development cohort and the eligible KNHANES adults (the four strata
    of the split, cohort_checks.csv): a descriptive table, not an outcome."""
    legend = Legend()
    header = ["Data set and ALT", "AST not above 40 U/L, n (%)", "AST above 40 U/L, n (%)"]
    rows = []
    for source, ds, stem in (("DEV", "DEV", "development cohort persons"), ("KN", "KN", "eligible adults")):
        total = D.count(src.one(BASELINE, sample="ANY", dataset=ds, variable="persons", level=None)["n"])
        rows.append(section(f"{D.DATASET_LABELS[ds]}: {D.fmt_count(total)} people", 3))
        words = ("not above 40", "above 40")
        values = [D.count(src.one(CHECKS, source=source, item=f"{stem} with ALT {a} and AST {b}")["n"])
                  for a in words for b in words]
        hide = hidden_cells(values, total, f"{CHECKS} {ds} ALT by AST")
        if hide:
            legend.add(HIDDEN_NA_NOTE)
        cell = {i: (D.NA if i in hide else n_with_pct(v, total)) for i, v in enumerate(values)}
        rows.append(["ALT not above 40 U/L", cell[0], cell[1]])
        rows.append(["ALT above 40 U/L", cell[2], cell[3]])
    notes = [(f"ALT above 40 U/L by AST above 40 U/L: the four combinations used to split the development cohort into "
              f"training and test sets. Development cohort: {DEV_POPULATION} (hospital records). KNHANES: "
              f"{D.DATASET_LONG['KN']}. Percentages are of the people of each data set. A descriptive table, not an "
              "outcome: no analysis combined the two enzymes.")]
    notes += [share_note(rows)] + legend.sentences + [abbreviations(["ALT", "AST", "BMI", "KNHANES"], rows=rows)]
    title = "ALT above 40 U/L by AST above 40 U/L in the development cohort and in eligible KNHANES adults"
    return dict(kind="table", id="enzyme_crosstab", title=title, markdown=table(header, rows), notes=notes,
                sources=[CHECKS, BASELINE], target="si", order=SI_ORDER["enzyme_crosstab"])


# ------------------------------------------------------------------------------------------------ Table S8
LEARNER_LABELS = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
LIBRARY_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost", "optuna": "Optuna",
                  "scikit-learn": "scikit-learn"}
QUIET_PARAMS = ("verbose", "verbosity", "n_jobs", "thread_count", "use_label_encoder")
TRAINING_OUTCOMES = "Outcomes in the development training set: ALT >40 U/L and AST >40 U/L."


def search_text(text):
    """'int 100-300 step 50' -> '100-300 (integer, step 50)'; 'float 0.02-0.1 log' -> '0.02-0.1 (log scale)'."""
    if text is None:
        return "Not tuned"
    m = re.fullmatch(r"(int|float) (\S+?)-(\S+?)(?: step (\S+))?( log)?", text.strip())
    if not m:
        raise ValueError(f"unknown search space {text!r}")
    kind, lo, hi, step, log = m.groups()
    extra = (["integer"] if kind == "int" else []) + ([f"step {step}"] if step else []) + (["log scale"] if log
                                                                                          else [])
    return f"{lo}-{hi}" + (f" ({', '.join(extra)})" if extra else "")


def versions_text(text):
    parts = []
    for item in str(text).split(";"):
        lib, _, ver = item.partition("=")
        parts.append(f"{LIBRARY_LABELS.get(lib.strip(), lib.strip())} {ver.strip()}")
    return "; ".join(parts)


def write_ensemble_details(src):
    ts = {o: src.one("train_summary.csv", outcome=o) for o in D.OUTCOMES}
    hp = src.rows("hyperparameters.csv")
    header = ["Item", "Search space"] + [D.OUTCOME_SHORT[o] for o in D.OUTCOMES]
    rows = [section("Training (development training set)", 4)]

    def both(fn_):
        return [fn_(ts[o]) for o in D.OUTCOMES]

    rows.append([f"{PEOPLE}, n", ""] + both(lambda t: D.fmt_count(t["n_train"])))
    rows.append(["Events, n (%)", ""] + both(lambda t: n_with_pct(D.count(t["events_train"]), D.count(t["n_train"]),
                                                                  D.fmt_pct(t["prevalence_train"], 2, None))))
    rows.append(["Optuna trials per learner", ""] + both(lambda t: t["trials_per_learner"]))
    rows.append(["Cross-validation folds for tuning (stratified)", ""] + both(lambda t: t["tuning_folds"]))
    rows.append(["Sampler (seed)", ""] + both(lambda t: f"{t['sampler']} ({t['seed']})"))
    rows.append(["Voting of the three learners", ""] + both(lambda t: t["voting"].capitalize()))
    rows.append(["Calibration (folds)", ""] + both(lambda t: f"{t['calibration'].capitalize()} "
                                                             f"({t['calibration_folds']})"))
    rows.append(["Class weighting", ""] + both(lambda t: t["class_weighting"].capitalize()))
    rows.append(["Fewest events in a calibration fold", ""] + both(lambda t: D.fmt_count(
        t["calibration_fold_events_min"])))
    for lr in ("lgb", "xgb", "cat"):
        rows.append([f"Best mean cross-validation AUROC: {LEARNER_LABELS[lr]}", ""] + both(
            lambda t, lr=lr: D.fmt(t[f"cv_auroc_{lr}"], 3)))
    vers = {versions_text(ts[o]["versions"]) for o in D.OUTCOMES}
    if len(vers) != 1:
        raise ValueError("the two ensembles were fitted with different software versions")
    for learner in ("lgb", "xgb", "cat"):
        rows.append(section(LEARNER_LABELS[learner], 4))
        params = []
        for r in hp:
            if r["learner"] == learner and r["param"] not in QUIET_PARAMS and r["param"] not in params:
                params.append(r["param"])
        for p in params:
            got = {o: [r for r in hp if r["learner"] == learner and r["param"] == p and r["outcome"] == o]
                   for o in D.OUTCOMES}
            searches = {r["search"] for o in D.OUTCOMES for r in got[o]}
            if len(searches) != 1 or any(len(got[o]) != 1 for o in D.OUTCOMES):
                raise ValueError(f"{learner} {p}: one row per outcome with one search space expected")
            rows.append([p, search_text(searches.pop())] + [sig(got[o][0]["value"]) for o in D.OUTCOMES])
    notes = [("The ensemble averages the predicted probabilities of LightGBM, XGBoost and CatBoost (soft voting) "
              "and is calibrated by isotonic regression in cross-validation folds; the final prediction is the mean "
              "of the calibrated fold ensembles. Hyperparameters were tuned by Optuna on the mean AUROC of "
              "stratified cross-validation in the development training set; values are shown to three significant "
              "digits. Settings that only control messages, threads or the encoding of the class labels are not "
              f"listed. Software: {vers.pop()}. {TRAINING_OUTCOMES} {EVENTS_NOTE}"),
             abbreviations(["ALT", "AST", "AUROC", "TPE"], {"TPE": "tree-structured Parzen estimator"}, rows)]
    title = "The ensemble: training, tuned hyperparameters and settings of the three learners"
    return dict(kind="table", id="ensemble_details", title=title, markdown=table(header, rows, "llrr"), notes=notes,
                sources=["train_summary.csv", "hyperparameters.csv"], target="si", order=SI_ORDER["ensemble_details"])


# ------------------------------------------------------------------------------------------------ Table S9
LR_MODELS = ("LR_ASB", "LR_ALL")
COEF_NA_NOTE = "NA: not shown, because fewer than 10 people in the training set had one value of this input."
# the standard of error of the table: the coefficients have no interval
COEF_POINT_NOTE = ("Coefficients are point estimates of the penalised fit; no confidence intervals were computed for "
                   "them.")


def write_lr_coefficients(src):
    legend = Legend()
    header = ["Term"] + [f"{D.OUTCOME_SHORT[o]}: {w}" for o in D.OUTCOMES for w in ("standardised", "per unit")]
    rows, sizes = [], {}
    for o in D.OUTCOMES:
        rs = src.rows("lr_coefficients.csv", outcome=o)
        sizes[o] = (single([D.count(r["n_train"]) for r in rs], "n_train"),
                    single([D.count(r["events_train"]) for r in rs], "events_train"))
    for m in LR_MODELS:
        rows.append(section(D.MODEL_LABELS[m], 5))
        terms = [r["term"] for r in src.rows("lr_coefficients.csv", outcome=D.OUTCOMES[0], model=m)]
        for term in terms:
            cells = []
            for o in D.OUTCOMES:
                r = src.one("lr_coefficients.csv", outcome=o, model=m, term=term)
                cells.append(D.fmt(r["coef_std"], 3))
                unit = D.fmt(r["coef_unit"], 4)
                if unit == D.NA:
                    legend.add(COEF_NA_NOTE)
                cells.append(unit)
            label = "Intercept" if term == "(intercept)" else D.INPUT_LABELS[term]
            rows.append([label] + cells)
    fitted = "; ".join(f"{D.OUTCOME_SHORT[o]}: {D.fmt_count(sizes[o][0])} people, {D.fmt_count(sizes[o][1])} events"
                       for o in D.OUTCOMES)
    notes = [(f"Fitted in the development training set ({fitted}) with an L2 (ridge) penalty on the standardised "
              f"coefficients (C = 1), without interactions and without a calibration step. {TRAINING_OUTCOMES} "
              f"{EVENTS_NOTE} Standardised: per standard deviation of the input in the training set (intercept: at the "
              "training means). Per unit: per increase of 1 in the model value (dipstick grades coded 0, 0.5, 1, 2 and "
              "3, so that negative to trace and trace to 1+ are each an increase of 0.5 and add half the coefficient "
              f"to the log odds), per unit of pH and of specific gravity, per year of age, per 1 {D.BMI_UNIT} of BMI, "
              "and men versus women; the per-unit intercept applies to the uncentred inputs, so that the log odds is "
              f"the intercept plus the sum of each per-unit coefficient times the input. {COEF_POINT_NOTE}")]
    notes += legend.sentences + [abbreviations(["ALT", "AST", "BMI"], rows=rows)]
    title = "Coefficients of the logistic regression comparators"
    return dict(kind="table", id="lr_coefficients", title=title, markdown=table(header, rows), notes=notes,
                sources=["lr_coefficients.csv"], target="si", order=SI_ORDER["lr_coefficients"])


# ------------------------------------------------------------------------------------------------ Table 2
PERF_MODELS = ("LR_ASB", "LR_ALL", "ENS")
# besides the stepwise rows, the ensemble against the model with age, sex and BMI
SECOND_STEP = ("ENS", "LR_ASB")
# the calibration block: the ensemble with its 95% CIs (calibration.csv), the others as point estimates
CAL_MODELS = ("ENS", "LR_ALL", "LR_ASB")
COLUMN_DATASET = {"DEV_TEST": "development test set", "KN": D.DATASET_LABELS["KN"]}
NA_COUNT_NOTE = "NA: not shown because a cell of the two-by-two table of alerts and outcomes held 1-9 people."
# calibration measures: (row label, column of calibration.csv and model_comparison.csv, decimals), and their terms
CAL_MEASURES = (("O:E ratio", "oe", 2), ("Calibration-in-the-large", "citl", 2), ("Calibration slope", "slope", 2),
                ("Brier score", "brier", 3), ("Scaled Brier score", "scaled_brier", 3))
CALIBRATION_TERMS = (
    "O:E, observed events divided by the sum of the predicted risks; calibration-in-the-large, intercept of a "
    "logistic recalibration with the slope held at 1 (ideal 0; negative when the predicted risks are too high); "
    "calibration slope, coefficient of the log odds of the predicted risk in a logistic model of the outcome (ideal 1; "
    "above 1 when the predicted risks are bunched too closely, below 1 when they are scattered too widely; a predicted "
    "risk of exactly 0 or 1 enters at log odds of -20.7 or 20.7); scaled Brier score, the share of the Brier score "
    f"of predicting the prevalence for every person, prevalence {TIMES} (1 - prevalence), that the model removes "
    "(ideal 1; 0 when it removes none; negative when it does worse).")
CHECKUP_GROUPS = (("Checkup day", "on a health check-up day"), ("Other day", "on another day"))


def column_header(outcome, dataset):
    return f"{D.OUTCOME_SHORT[outcome]}: {COLUMN_DATASET[dataset]}"


def na_note(text, legend, note=NA_COUNT_NOTE):
    """`text`; when it is NA, `note` goes into the legend (why the estimate is not shown)."""
    if text == D.NA:
        legend.add(note)
    return text


def performance_table(src, cols):
    """Rows of Table 2: the columns `cols` ((outcome, dataset) pairs in display order)."""
    legend = Legend()
    ncol = len(cols) + 1
    perf = {c: src.one("performance.csv", outcome=c[0], dataset=c[1], model="ENS") for c in cols}
    cal = {c: src.one("calibration.csv", outcome=c[0], dataset=c[1], model="ENS") for c in cols}

    def mc(c, m, b=None):
        if b is None:
            return src.one("model_comparison.csv", row_type="model", outcome=c[0], dataset=c[1], model=m)
        return src.one("model_comparison.csv", row_type="delta", outcome=c[0], dataset=c[1], model=m, model_b=b)

    for c in cols:
        a, b = D.num(perf[c]["auroc"]), D.num(mc(c, "ENS")["auroc"])
        if a is None or b is None or abs(a - b) > 1e-9:
            raise ValueError(f"{c}: ensemble AUROC differs between performance.csv and model_comparison.csv")
    rows = [[f"{PEOPLE} (events), n"] + [f"{D.fmt_count(perf[c]['n'])} ({D.fmt_count(perf[c]['events'])})"
                                         for c in cols]]
    rows.append(["Prevalence, %"] + [D.fmt_pct(perf[c]["prevalence"]) for c in cols])
    rows.append(["Ensemble, mean predicted risk, %"] + [D.fmt_pct(cal[c]["mean_pred"]) for c in cols])
    rows.append(section("Discrimination, AUROC (95% CI)", ncol))
    previous = None
    for m in PERF_MODELS:
        rows.append([D.MODEL_LABELS[m]] + [D.fmt_ci(mc(c, m)["auroc"], mc(c, m)["auroc_lo"], mc(c, m)["auroc_hi"], 3)
                                           for c in cols])
        if previous:
            d = {c: mc(c, m, previous) for c in cols}
            rows.append(["Difference from the model above (95% CI)"] + [
                D.fmt_ci(d[c]["delta"], d[c]["delta_lo"], d[c]["delta_hi"], 3, signed=True) for c in cols])
            rows.append(["P value"] + [D.fmt_p(d[c]["delong_p"]) for c in cols])
        previous = m
    first, second = SECOND_STEP
    d = {c: mc(c, first, second) for c in cols}
    rows.append([f"{D.MODEL_LABELS[first]} minus {lc(D.MODEL_LABELS[second])} (95% CI)"] + [
        D.fmt_ci(d[c]["delta"], d[c]["delta_lo"], d[c]["delta_hi"], 3, signed=True) for c in cols])
    rows.append(["P value"] + [D.fmt_p(d[c]["delong_p"]) for c in cols])
    rows.append(section("Calibration and Brier scores", ncol))
    for m in CAL_MODELS:
        rows.append(section(D.MODEL_LABELS[m] + (" (95% CI)" if m == "ENS" else " (point estimates)"), ncol, "*"))
        for label, key, dec in CAL_MEASURES:
            cells = []
            for c in cols:
                if m == "ENS":
                    a, b = D.num(cal[c][key]), D.num(mc(c, m)[key])
                    if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):     # two blanks agree
                        raise ValueError(f"{c} {key}: the ensemble differs between calibration.csv and "
                                         "model_comparison.csv")
                    cells.append(D.fmt_ci(cal[c][key], cal[c][f"{key}_lo"], cal[c][f"{key}_hi"], dec))
                else:
                    cells.append(D.fmt(mc(c, m)[key], dec))
            rows.append([label] + cells)
    n_boot = single([D.count(perf[c]["n_boot"]) for c in cols] + [D.count(cal[c]["n_boot"]) for c in cols]
                    + [D.count(r["n_boot"]) for r in src.rows("model_comparison.csv") if (r["outcome"], r["dataset"])
                       in cols], "n_boot")
    header = ["Measure"] + [column_header(*c) for c in cols]
    return header, rows, legend, n_boot


MODEL_NOTE = (f"Ensemble: {lc(D.MODEL_LONG['ENS'])} on the 12 inputs (9 urine dipstick results, age, sex and BMI), "
              "with isotonic calibration fitted in cross-validation folds of the development training set. Each "
              "logistic regression model uses the inputs named.")
# the KNHANES calibration rows of Table 2: no updating was done in KNHANES
AS_THEY_STAND_NOTE = "In KNHANES, the predicted risks are used as they stand, without recalibration."
PER_CASE_NOTE = "Alerts per case: alerts per person with the outcome among those flagged (1/PPV)."
DELTA_NOTE = ("Difference in AUROC with its paired bootstrap 95% CI: the model minus the model above it, or the "
              "ensemble minus the model named. Differences were computed from unrounded AUROCs, so they may depart by "
              "0.001 from the difference of the AUROCs shown. P value: two-sided DeLong test for two correlated AUROCs "
              "(alpha 0.05). "
              "P values were not adjusted for multiple comparisons.")
PERFORMANCE_ABBREVIATIONS = ["ALT", "AST", "AUROC", "BMI", "CI", "KNHANES", "O:E"]
PERFORMANCE_SOURCES = ["performance.csv", "model_comparison.csv", "calibration.csv"]


def write_performance(src):
    """Table 2: both outcomes (ALT first) in the development test set and in the eligible KNHANES adults."""
    cols = [(o, d) for o in D.OUTCOMES for d in ("DEV_TEST", "KN")]
    header, rows, legend, n_boot = performance_table(src, cols)
    notes = [(f"{OUTCOME_NOTE} {EVENTS_NOTE} Each analysis of a column uses the people and events of its first row. "
              f"KNHANES columns: {D.DATASET_LONG['KN']}. {MODEL_NOTE} {AS_THEY_STAND_NOTE} {boot_note(n_boot)}"),
             DELTA_NOTE, CALIBRATION_TERMS]
    notes += legend.sentences + [abbreviations(PERFORMANCE_ABBREVIATIONS, rows=rows)]
    title = (f"Discrimination and calibration of the ensemble and two logistic regression models for "
             f"{D.OUTCOME_LABELS['ALT']} and {D.OUTCOME_LABELS['AST']}: development test set and "
             f"{D.DATASET_LONG['KN']}")
    return dict(kind="table", id="performance", title=title, markdown=table(header, rows), notes=notes,
                sources=PERFORMANCE_SOURCES, target="main", order=MAIN_ORDER["performance"])


# ------------------------------------------------------------------------------------------------ Table 3
STRATEGIES = ("RULE_BMI25", "ENS@MATCH_BMI25", "LR_ALL@MATCH_BMI25", "LR_ASB@MATCH_BMI25")
METRICS = (("alert_rate", "Alert rate"), ("se", "Sensitivity"), ("sp", "Specificity"), ("ppv", "PPV"),
           ("npv", "NPV"))
CONFUSION = ("tp", "fp", "tn", "fn")
SE_THRESHOLDS = ("YOUDEN", "SENS80", "SENS90")    # the ensemble operating points of Table 3
SE_AFTER = "LR_ASB@MATCH_BMI25"                   # the last row of the rule and its model thresholds
SE_NOTE = ("Ensemble rows of the Youden index threshold and of the thresholds for 80% and 90% sensitivity: set in the "
           "development test set (the Youden index threshold as the one that maximises sensitivity + specificity - 1, "
           "the others as the largest with at least 80% or 90% sensitivity there; not set by the rule's alert rate) "
           "and applied unchanged to KNHANES; shown beside the rule for comparison. No paired comparison with the "
           "rule was made.")


def threshold_label(t):
    """A threshold name inside a row label: 'Youden index threshold', 'threshold for 80% sensitivity'."""
    label = D.THRESHOLD_LABELS[t]
    return label if label.startswith("Youden") else lc(label)


def target_note(src):
    """The note of the two columns of the development test set in Table 3, with the people of that set (the same for
    both outcomes). At one decimal, a step of 0.1 is at least 30 people there, so no count of 1-9 follows from a
    percentage."""
    people = D.fmt_count(single([D.count(r["n"]) for r in src.rows("comparators_alert.csv", dataset="DEV_TEST")], "n"))
    return (f"Proportion of people flagged in the development test set ({people} people): by the rule (target), and by "
            "the model threshold (achieved; for the Youden index threshold and the thresholds for 80% and 90% "
            "sensitivity, which have no target, the proportion they flag).")


def counts_note(r):
    """Note of a row of comparators_alert.csv or operating_points.csv whose rates are blank because a confusion count
    is 1-9: its four counts as the aggregate gives them ('<10' masked, a blank cell not shown); nothing is worked out
    from them."""
    def part(cell, where):
        return f"{where}, {'not shown' if D.is_blank(cell) else D.fmt_count(cell)}"
    return ("Rates not shown, because a cell of the two-by-two table of alerts and outcomes held 1-9 people. People "
            f"with the outcome: {part(r['tp'], 'flagged')}; {part(r['fn'], 'not flagged')}. People without the "
            f"outcome: {part(r['fp'], 'flagged')}; {part(r['tn'], 'not flagged')}. A count not shown would give "
            "the masked one by subtraction from the totals in the heading row.")


def rate_cells(r, legend, label, where):
    """The five rates and alerts per case of a row: when a confusion count is 1-9, NA, with a sentence in the legend
    (`label`: the row it speaks of) that gives the counts (counts_note; the shown counts are first checked against
    n and events); otherwise the formatted values, NA noted when blank."""
    if any(D.fmt_count(r[c]) == MASKED for c in CONFUSION):
        n, events = D.count(r["n"]), D.count(r["events"])
        if not (is_int(n) and is_int(events)):
            raise ValueError(f"{where}: n and events are needed to show the counts of a masked row")
        cell = {c: D.count(r[c]) for c in CONFUSION}
        require_block([cell["tp"], cell["fn"]], events, (), f"{where} (people with the outcome)")
        require_block([cell["fp"], cell["tn"]], n - events, (), f"{where} (people without the outcome)")
        if r["alert_rate"] is not None:
            raise ValueError(f"{where}: a row with a count of 1-9 shows its rates")
        legend.add(f"{label}: {counts_note(r)}")
        return [D.NA] * (len(METRICS) + 1)
    cells = [na_note(D.fmt_pct_ci(r[k], r[f"{k}_lo"], r[f"{k}_hi"]), legend) for k, _ in METRICS]
    cells.append(na_note(D.fmt_ci(r["alerts_per_case"], r["alerts_per_case_lo"], r["alerts_per_case_hi"],
                                  PER_CASE_DECIMALS), legend))
    return cells


def se_rows(src, outcome, dataset, n, events, legend, boots):
    """The ensemble at the thresholds of SE_THRESHOLDS (operating_points.csv), in the columns of Table 3: threshold,
    no target, the alert rate in the development test set, then the results in `dataset`."""
    out = []
    for t in SE_THRESHOLDS:
        op = src.one("operating_points.csv", outcome=outcome, dataset=dataset, model="ENS", threshold_name=t)
        dev = src.one("operating_points.csv", outcome=outcome, dataset="DEV_TEST", model="ENS", threshold_name=t)
        if (D.count(op["n"]), D.count(op["events"])) != (n, events):
            raise ValueError(f"{outcome} {dataset} {t}: people or events differ from comparators_alert.csv")
        if abs(D.num(op["threshold"]) - D.num(dev["threshold"])) > 1e-12:
            raise ValueError(f"{outcome} {t}: the threshold differs between data sets")
        boots.append(D.count(op["n_boot"]))
        label = f"{D.MODEL_SHORT['ENS']}, {threshold_label(t)}"
        cells = [threshold_text(op["threshold"]), "", na_note(D.fmt_pct(dev["alert_rate"]), legend)]
        out.append([label] + cells + rate_cells(op, legend, f"{D.OUTCOME_SHORT[outcome]}, {lc(label)}",
                                                f"{outcome} {dataset} ENS {t}"))
    return out


def comparators_table(src, dataset, legend):
    """Rows of Table 3: a section per outcome, in display order: the rule and the models at the thresholds it sets, then
    the ensemble at the thresholds of SE_THRESHOLDS."""
    header = ["Strategy", "Threshold", "Development test set: target alert rate, %",
              "Development test set: achieved alert rate, %"] + [
        f"{'KNHANES: alert rate' if k == 'alert_rate' else lab}, % (95% CI)" for k, lab in METRICS] + [
        "Alerts per case (95% CI)"]
    rule_rates = {src.one("comparators_alert.csv", outcome=o, dataset="DEV_TEST", strategy=STRATEGIES[0])
                  ["dev_test_target_rate"] for o in D.OUTCOMES} - {None}
    if len(rule_rates) > 1:
        raise ValueError("the rule alerts the same people for both outcomes, but its target rates differ")
    rows, boots = [], []
    for o in D.OUTCOMES:
        rs = src.rows("comparators_alert.csv", outcome=o, dataset=dataset)
        n = single([D.count(r["n"]) for r in rs], "n")
        ev = single([D.count(r["events"]) for r in rs], "events")
        rows.append(section(f"{D.OUTCOME_LABELS[o]}: {D.fmt_count(n)} people, {D.fmt_count(ev)} events", len(header)))
        for s in STRATEGIES:
            r = src.one("comparators_alert.csv", outcome=o, dataset=dataset, strategy=s)
            boots.append(D.count(r["n_boot"]))
            if r["row_type"] == "rule":
                label, thr, achieved = f"{D.RULE_LABELS[r['rule']]} (rule)", "", ""
            else:
                label = D.MODEL_SHORT[r["model"]]
                thr, achieved = threshold_text(r["threshold"]), na_note(D.fmt_pct(r["dev_test_achieved_rate"]), legend)
            cells = [thr, na_note(D.fmt_pct(r["dev_test_target_rate"]), legend), achieved]
            rows.append([label] + cells + rate_cells(r, legend, f"{D.OUTCOME_SHORT[o]}, {lc(label)}",
                                                     f"{o} {dataset} {s}"))
            if s == SE_AFTER:
                rows += se_rows(src, o, dataset, n, ev, legend, boots)
    return header, rows, single(boots, "n_boot")


def comparator_notes(n_boot):
    """The first note of Table 3."""
    return (f"{OUTCOME_NOTE} {EVENTS_NOTE} The rule alerts the same people for both outcomes: its alerts are the "
            "same for ALT and AST and only its accuracy differs between the enzymes. Each model "
            "threshold set by the rule was chosen in the development test set, separately for each outcome, to flag "
            "the proportion of people nearest to that flagged by the rule (ties to the larger threshold); every "
            "threshold was applied unchanged to KNHANES, where the proportions flagged by the rule and by the "
            f"models differ. Alert rate to alerts per case: results in {D.DATASET_LONG['KN']}; the differences between "
            f"each model and the rule are in {si('comparator_differences')}. Alert when the predicted risk is at or "
            f"above the threshold. {PER_CASE_NOTE} Models as in {main_table('performance')}. {boot_note(n_boot)}")


COMPARATOR_ABBREVIATIONS = ["ALT", "AST", "BMI", "CI", "KNHANES", "NPV", "PPV"]
COMPARATOR_SOURCES = ["comparators_alert.csv", "operating_points.csv"]


def write_comparators(src):
    """Table 3: the rule, the model thresholds set by it and the ensemble operating points, in KNHANES, per outcome."""
    legend = Legend()
    header, rows, n_boot = comparators_table(src, "KN", legend)
    notes = [comparator_notes(n_boot), target_note(src), SE_NOTE] + legend.sentences + [
        abbreviations(COMPARATOR_ABBREVIATIONS, rows=rows)]
    title = (f"The rule {D.RULE_LONG['BMI25']}, and model thresholds set in the development test set: results in "
             f"{D.DATASET_LONG['KN']}")
    return dict(kind="table", id="comparators", title=title, markdown=table(header, rows), notes=notes,
                sources=COMPARATOR_SOURCES, target="main", order=MAIN_ORDER["comparators"])


# ------------------------------------------------------------------------------------------------ Table S10
DIFF_DATASETS = ("DEV_TEST", "KN")
DIFF_DATASET_LABELS = {"DEV_TEST": "Development test set", "KN": D.DATASET_LABELS["KN"]}
OP_ROWS = (("ENS", "YOUDEN"), ("ENS", "SENS80"), ("ENS", "SENS90"), ("ENS", "MATCH_BMI25"), ("LR_ALL", "MATCH_BMI25"),
           ("LR_ASB", "MATCH_BMI25"))
DERIVATION_NOTE = ("In the development test set the thresholds were set, so its rows are in-sample derivation "
                   "results; the same thresholds were applied unchanged to KNHANES.")


def operating_point_label(model, threshold_name):
    if threshold_name == "MATCH_BMI25":
        return D.STRATEGY_LABELS[f"{model}@MATCH_BMI25"]
    return f"{D.MODEL_SHORT[model]}, {threshold_label(threshold_name)}"


def write_operating_points(src):
    """Every operating point of operating_points.csv, by outcome and data set: the ensemble at its three thresholds
    and at the threshold set by the rule, and the two logistic regression models at the threshold set by the rule."""
    legend = Legend()
    header = ["Operating point", "Threshold"] + [f"{lab}, % (95% CI)" for _, lab in METRICS] + [
        "Alerts per case (95% CI)"]
    rows, boots = [], []
    for o in D.OUTCOMES:
        rows.append(section(D.OUTCOME_LABELS[o], len(header)))
        for ds in DIFF_DATASETS:
            rs = {(m, t): src.one("operating_points.csv", outcome=o, dataset=ds, model=m, threshold_name=t)
                  for m, t in OP_ROWS}
            n = single([D.count(r["n"]) for r in rs.values()], "n")
            ev = single([D.count(r["events"]) for r in rs.values()], "events")
            derivation = {D.flag(r["derivation"]) for r in rs.values()}
            if derivation != {ds == "DEV_TEST"}:
                raise ValueError(f"{o} {ds}: the derivation flag of operating_points.csv")
            mark = " (derivation set)" if ds == "DEV_TEST" else ""
            rows.append(section(f"{DIFF_DATASET_LABELS[ds]}{mark}: {D.fmt_count(n)} people, {D.fmt_count(ev)} events",
                                len(header), "*"))
            for (m, t), r in rs.items():
                boots.append(D.count(r["n_boot"]))
                label = operating_point_label(m, t)
                cells = [threshold_text(r["threshold"])]
                cells += rate_cells(r, legend, f"{D.OUTCOME_SHORT[o]}, {lc(DIFF_DATASET_LABELS[ds])}, {lc(label)}",
                                    f"{o} {ds} {m} {t}")
                rows.append([label] + cells)
    notes = [(f"{OUTCOME_NOTE} {EVENTS_NOTE} Each row is a model at a threshold set in the development test set: the "
              "Youden index threshold and the largest thresholds with at least 80% or 90% sensitivity for the "
              "ensemble, and, for each model, the threshold set to flag the proportion of people nearest to that of "
              f"the rule ({D.RULE_LONG['BMI25']}; ties to the larger threshold). Alert when the predicted risk is at "
              f"or above the threshold. {PER_CASE_NOTE} {boot_note(single(boots, 'n_boot'))} {DERIVATION_NOTE} Models "
              f"as in {main_table('performance')}. KNHANES: {D.DATASET_LONG['KN']}.")]
    notes += legend.sentences + [abbreviations(["ALT", "AST", "BMI", "CI", "KNHANES", "NPV", "PPV"], rows=rows)]
    title = ("Operating points of the ensemble and of the logistic regression models at the thresholds set in the "
             "development test set: development test set and KNHANES, by outcome")
    return dict(kind="table", id="operating_points", title=title, markdown=table(header, rows), notes=notes,
                sources=["operating_points.csv"], target="si", order=SI_ORDER["operating_points"])


# ------------------------------------------------------------------------------------------------ Table S11
DIFF_DECIMALS = 2           # percentage points: two decimals, so that a small CI is not printed as one value
DIFF_NA_NOTE = ("NA: not shown because a cell of the two-by-two table of alerts and outcomes of the model or of its "
                "rule held 1-9 people.")
DIFF_DERIVATION_NOTE = ("In the development test set (the derivation set) the thresholds were set to flag the "
                        "proportion nearest to the rule's, so the difference in alert rate is the gap left by tied "
                        "predictions.")


def write_comparator_differences(src):
    """The notes point to the tables with the people and events of each data set and do not repeat them: beside
    differences to 2 decimals, within one table, they would give net differences of 1-9 people between a model and its
    rule."""
    legend = Legend()
    header = ["Model and data set"] + [f"{lab}, difference (95% CI)" for _, lab in METRICS]
    rows, boots = [], []
    for o in D.OUTCOMES:
        rows.append(section(D.OUTCOME_LABELS[o], len(header)))
        for rule in D.RULES:
            rows.append(section(f"Thresholds set by the rule {D.RULE_LABELS[rule]}", len(header), "*"))
            for m in ("ENS", "LR_ALL", "LR_ASB"):
                for ds in DIFF_DATASETS:
                    r = src.one("comparators_alert.csv", outcome=o, dataset=ds, strategy=f"{m}@MATCH_{rule}")
                    boots.append(D.count(r["n_boot"]))
                    label = f"{D.MODEL_SHORT[m]}: {lc(DIFF_DATASET_LABELS[ds])}"
                    if D.flag(r["derivation"]):
                        legend.add(DIFF_DERIVATION_NOTE)
                    cells = [na_note(D.fmt_pct_ci(r[f"d_{k}"], r[f"d_{k}_lo"], r[f"d_{k}_hi"], DIFF_DECIMALS,
                                                  signed=True), legend, DIFF_NA_NOTE) for k, _ in METRICS]
                    rows.append([label] + cells)
    notes = [(f"Differences in percentage points, to {DIFF_DECIMALS} decimals: the model at its threshold minus the "
              f"rule ({D.RULE_LONG['BMI25']}), on the same people; 95% CIs from "
              f"{D.fmt_count(single(boots, 'n_boot'))} paired bootstrap resamples. Thresholds as in "
              f"{main_table('comparators')}. The people and events of each data set are in "
              f"{main_table('performance')}. {OUTCOME_NOTE} {EVENTS_NOTE} KNHANES: {D.DATASET_LONG['KN']}.")]
    notes += legend.sentences + [abbreviations(["ALT", "AST", "BMI", "CI", "KNHANES", "NPV", "PPV"], rows=rows)]
    title = "Differences between each model threshold and the BMI rule, on the same people"
    return dict(kind="table", id="comparator_differences", title=title, markdown=table(header, rows), notes=notes,
                sources=["comparators_alert.csv"], target="si", order=SI_ORDER["comparator_differences"])


# ------------------------------------------------------------------------------------------------ Table S12
SUB_VARS = ("age_band", "sex", "period", "drink_group")
SUB_LEVELS = {"age_band": list(D.AGE_BAND_LABELS), "sex": list(D.SEX_LABELS), "period": list(D.PERIOD_LABELS),
              "drink_group": list(D.DRINK_LABELS)}
SUB_DROP_NOTE = ("The group of unknown alcohol drinking frequency is shown as counts only (its prevalence, AUROC and "
                 "O:E are NA).")
SUB_HIDDEN_NOTE = ("NA in place of a count, or a prevalence: withheld, since with the All row it would let a count of "
                   "1-9 in this group of rows be worked out.")
SUB_FEW_NOTE = "NA in the AUROC and O:E columns: fewer than 10 events or non-events in the subgroup, counts only."


def write_subgroups(src):
    """The KNHANES subgroups of subgroups.csv, by outcome. Within each subgroup variable the n and events cells are
    the shown cells of a block whose total is the All row: a level of 1-9 people stays '<10', and the smallest shown
    count of 10 or more goes to NA (hide_for_block), so that the masked level cannot be worked out from All minus the
    other levels."""
    legend = Legend()
    header = ["Subgroup", f"{PEOPLE}, n", "Events, n", "Prevalence, %", "AUROC (95% CI)", "O:E (95% CI)"]
    rows, boots = [], []
    for o in D.OUTCOMES:
        rows.append(section(D.OUTCOME_LABELS[o], len(header)))
        allr = src.one("subgroups.csv", outcome=o, dataset="KN", model="ENS", subgroup_var="all")
        blocks = [("all", [allr])]
        for var in SUB_VARS:
            blocks.append((var, [src.one("subgroups.csv", outcome=o, dataset="KN", model="ENS",
                                         subgroup_var=var, subgroup=lev) for lev in SUB_LEVELS[var]]))
        for var, rs in blocks:
            hide_n, hide_e = set(), set()
            if var != "all":
                rows.append(section(D.SUBGROUP_VAR_LABELS[var], len(header), "*"))
                hide_n = hide_for_block([D.count(r["n"]) for r in rs], D.count(allr["n"]))
                hide_e = hide_for_block([D.count(r["events"]) for r in rs], D.count(allr["events"]), hide_n)
            for i, r in enumerate(rs):
                if r["n_boot"] is not None:
                    boots.append(D.count(r["n_boot"]))
                if i in hide_n or i in hide_e:
                    legend.add(SUB_HIDDEN_NOTE)
                n_txt = D.NA if i in hide_n else D.fmt_count(r["n"])
                e_txt = D.NA if i in hide_e else D.fmt_count(r["events"])
                prev = D.NA if i in hide_e else D.fmt_pct(r["prevalence"])
                if var == "drink_group" and r["subgroup"] == "unknown":
                    legend.add(SUB_DROP_NOTE)           # counts only, whatever the size of the group
                    prev, auroc, oe = D.NA, D.NA, D.NA
                elif r["auroc"] is None:
                    legend.add(SUB_FEW_NOTE)
                    auroc, oe = D.NA, D.NA
                else:
                    auroc = D.fmt_ci(r["auroc"], r["auroc_lo"], r["auroc_hi"], 3)
                    oe = D.fmt_ci(r["oe"], r["oe_lo"], r["oe_hi"], 2)
                label = "All" if var == "all" else D.subgroup_label(var, r["subgroup"])
                rows.append([label, n_txt, e_txt, prev, auroc, oe])
    notes = [(f"Ensemble in {D.DATASET_LONG['KN']}. The row All gives the whole sample; events are the people with the "
              f"outcome. {OUTCOME_NOTE} 95% CIs: percentile bootstrap with {D.fmt_count(single(boots, 'n_boot'))} "
              "resamples drawn within each subgroup. O:E, observed events divided by the sum of the predicted risks. "
              "KNHANES records ages of 80 years or more as 80. The alcohol drinking groups are those of the survey "
              f"codes in {si('alcohol_codes')}.")]
    # NA marks a group shown as counts only, unless a count or prevalence is withheld too (then the general text)
    na = NA_TEXT["NA"] if SUB_HIDDEN_NOTE in legend.sentences else NA_COUNTS_ONLY
    notes += legend.sentences + [abbreviations(["ALT", "AST", "AUROC", "CI", "KNHANES", "O:E"], rows=rows, na=na)]
    title = ("Discrimination and calibration of the ensemble by age group, sex, survey period and frequency of alcohol "
             f"drinking in {D.DATASET_LONG['KN']}")
    return dict(kind="table", id="subgroups", title=title, markdown=table(header, rows), notes=notes,
                sources=["subgroups.csv"], target="si", order=SI_ORDER["subgroups"])


# ------------------------------------------------------------------------------------------------ Table S13
PAIR_NOTE = ("NA in a row of a pair of groups: counts only, because one of the two groups (days of the index "
             "urinalysis) has 1-9 people, events or non-events, which the totals of the data set would give by "
             "subtraction.")
FEW_NOTE = "NA in a row with counts: fewer than 10 events or non-events in the group, counts only."
SENS_HIDDEN_NOTE = "NA in place of a count: not shown, since with the totals it would let a count of 1-9 be worked out."


def write_sensitivity_analyses(src):
    legend = Legend()
    header = ["Analysis and group", f"{PEOPLE}, n", "Events, n", "Prevalence, % (95% CI)", "Mean predicted risk, %",
              "O:E (95% CI)", "AUROC (95% CI)"]
    rows, boots = [], []

    def why_of(rs):
        """Note of the counts-only rows of a pair of groups: the pair rule (it blanks the events of both groups), or
        too few events or non-events in one group."""
        return PAIR_NOTE if any(r["events"] is None for r in rs) else FEW_NOTE

    def stats_cells(r, hide_n=False, hide_e=False, why=FEW_NOTE):
        """People, events, prevalence (95% CI), mean predicted risk, O:E (95% CI) and AUROC (95% CI) of an ensemble
        row; counts only, with the note `why`, when its rates are blank."""
        if r["n_boot"] is not None:
            boots.append(D.count(r["n_boot"]))
        if hide_n or hide_e:
            legend.add(SENS_HIDDEN_NOTE)
        n_txt = D.NA if hide_n else D.fmt_count(r["n"])
        e_txt = D.NA if (hide_n or hide_e) else D.fmt_count(r["events"])
        if r["auroc"] is None:
            legend.add(why)
            return [n_txt, e_txt, D.NA, D.NA, D.NA, D.NA]
        prev = D.NA if (hide_n or hide_e) else D.fmt_pct_ci(r["prevalence"], r["prevalence_lo"], r["prevalence_hi"])
        return [n_txt, e_txt, prev, D.fmt_pct(r["mean_pred"]), D.fmt_ci(r["oe"], r["oe_lo"], r["oe_hi"], 2),
                D.fmt_ci(r["auroc"], r["auroc_lo"], r["auroc_hi"], 3)]

    def counts_hidden(rs, dataset, outcome):
        """Cells of the n and events of a pair of groups to show as NA, against the totals of the data set."""
        perf = src.one("performance.csv", outcome=outcome, dataset=dataset, model="ENS")
        hide_n = hide_for_block([D.count(r["n"]) for r in rs], D.count(perf["n"]))
        return hide_n, hide_for_block([D.count(r["events"]) for r in rs], D.count(perf["events"]), hide_n)

    rows.append(section(f"{D.SENSITIVITY_LABELS['CHECKUP_DAY']} (development test set)", len(header)))
    for o in D.OUTCOMES:
        rs = [src.one("sensitivity.csv", analysis="CHECKUP_DAY", outcome=o, dataset="DEV_TEST", model="ENS",
                      subgroup=g) for g, _ in CHECKUP_GROUPS]
        hide_n, hide_e = counts_hidden(rs, "DEV_TEST", o)
        for i, (r, (_, words)) in enumerate(zip(rs, CHECKUP_GROUPS)):
            rows.append([f"{D.OUTCOME_SHORT[o]}: index urinalysis {words}"]
                        + stats_cells(r, i in hide_n, i in hide_e, why_of(rs)))
    rows.append(section(f"{D.SENSITIVITY_LABELS['WEIGHTED']} (KNHANES)", len(header)))
    weighted_note = ("Weighted by the KNHANES interview and examination weights (wt_itvex) pooled over the survey "
                     "years; the 95% CIs come from resamples of the records that keep each record's weight and ignore "
                     "the strata and clusters of the survey design, so they are approximate.")
    left = False
    for o in D.OUTCOMES:
        pair = {w: src.one("sensitivity.csv", analysis="WEIGHTED", outcome=o, dataset="KN", model="ENS",
                           subgroup="All", weighted=w) for w in ("False", "True")}
        hide = weighted_hidden(pair["False"], pair["True"])
        left = left or pair["True"]["n"] != pair["False"]["n"]
        for w, word in (("False", "unweighted"), ("True", "weighted")):
            rows.append([f"{D.OUTCOME_SHORT[o]}: {word}"] + stats_cells(pair[w], hide and w == "True"))
    kept = ("Records without a positive weight were left out of the weighted analyses." if left else
            "Every record had a positive weight.")
    notes = [(f"Ensemble. {OUTCOME_NOTE} {EVENTS_NOTE} The health check-up day analysis splits the development test "
              "set by the day of the index urinalysis; the two groups also differ in the period of the urinalysis, so "
              "this comparison is descriptive. " + weighted_note + " Weighted prevalence, mean predicted risk and O:E "
              "are weighted means, and the weighted AUROC weights each pair of people by the product of their weights. "
              f"{kept} Unweighted rows repeat {main_table('performance')}. 95% CIs from "
              f"{D.fmt_count(single(boots, 'n_boot'))} bootstrap resamples. KNHANES: {D.DATASET_LONG['KN']}.")]
    notes += legend.sentences + [abbreviations(["ALT", "AST", "AUROC", "CI", "KNHANES", "O:E"], rows=rows)]
    title = ("Sensitivity analyses: index urinalysis on a health check-up day in the development test set, and survey "
             "weights in KNHANES")
    return dict(kind="table", id="sensitivity_analyses", title=title, markdown=table(header, rows), notes=notes,
                sources=["sensitivity.csv", "performance.csv"], target="si", order=SI_ORDER["sensitivity_analyses"])


# ------------------------------------------------------------------------------------------------ Table S14
YEAR_NA_NOTE = ("NA: a median needs at least 100 values, and a count of 10 or more is left out where the people not "
                "above 40 U/L number 1-9 (it would give them by subtraction).")


def write_enzyme_by_year(src):
    """The KNHANES medians and the numbers above 40 U/L of ALT and AST by survey year (enzyme_by_year.csv, source KN):
    a descriptive table. The cells of enzyme_by_year.csv are shown as they stand, with the blanks of its masking rule as
    NA."""
    legend = Legend()
    rows_all = src.rows("enzyme_by_year.csv", source="KN")
    years = sorted({r["year"] for r in rows_all}, key=int)
    header = ["Survey year", f"{PEOPLE}, n"]
    for o in D.OUTCOMES:
        header += [f"{D.OUTCOME_SHORT[o]}, U/L, median", f"{D.OUTCOME_LABELS[o]}, n (%)"]
    rows = []
    for y in years:
        by = {o: src.one("enzyme_by_year.csv", source="KN", enzyme=o, year=y) for o in D.OUTCOMES}
        n = single([by[o]["n"] for o in D.OUTCOMES], f"people of {y}")
        cells = [y, D.fmt_count(n)]
        for o in D.OUTCOMES:
            med = D.fmt(by[o]["median"], 1)
            events = n_with_pct(by[o]["events"], D.count(n))
            cells += [med, events]
        rows.append(cells)
    if shows_na(rows):
        legend.add(YEAR_NA_NOTE)
    notes = [(f"{cap(D.DATASET_LONG['KN'])} by survey year: {OUTCOME_NOTE} A descriptive table of the enzyme values as "
              "they were published in each survey year.")]
    notes += legend.sentences + [abbreviations(["ALT", "AST", "KNHANES"], rows=rows)]
    title = f"ALT and AST by survey year: medians and numbers above 40 U/L, in {D.DATASET_LONG['KN']}"
    return dict(kind="table", id="enzyme_by_year", title=title, markdown=table(header, rows), notes=notes,
                sources=["enzyme_by_year.csv"], target="si", order=SI_ORDER["enzyme_by_year"])


# ------------------------------------------------------------------------------------------------ main
def main():
    # first: remove every output this script writes, so that a run that stops leaves no out-of-date table
    for target, ids in (("main", MAIN_ORDER), ("si", SI_ORDER)):
        D.reset_index("table", target, ids=ids)
    tables_md = D.out_path("manuscript", "tables.md", create=False)
    if os.path.exists(tables_md):
        os.remove(tables_md)
    src = Sources()
    plans, codes = level_plans(src)                         # sample ANY: Supplementary Table S1
    specs = [write_baseline(src, plans), write_performance(src), write_comparators(src),
             write_inputs(src, plans, codes), write_knhanes_codes(src), write_alcohol_codes(src),
             write_enzyme_availability(src), write_bmi_availability(src), write_bmi_source(src),
             write_enzyme_crosstab(src), write_ensemble_details(src), write_lr_coefficients(src),
             write_operating_points(src), write_comparator_differences(src), write_subgroups(src),
             write_sensitivity_analyses(src), write_enzyme_by_year(src)]
    ids = {target: sorted(s["id"] for s in specs if s["target"] == target) for target in D.TARGETS}
    if ids != {"main": sorted(MAIN_ORDER), "si": sorted(SI_ORDER)}:
        raise ValueError(f"the tables written differ from MAIN_ORDER and SI_ORDER: {ids}")
    paths = {}
    for spec in specs:
        paths[spec["id"]] = D.write_fragment(spec["kind"], spec["id"], spec["title"], spec["markdown"],
                                             spec["notes"], spec["sources"], target=spec["target"],
                                             order=spec["order"])
    main_specs = sorted((s for s in specs if s["target"] == "main"), key=lambda s: s["order"])
    texts = []
    for s in main_specs:
        with open(paths[s["id"]], encoding="utf-8") as fh:
            texts.append(fh.read().strip())
    D.write_text(tables_md, "\n\n".join(texts))
    pooled = sum(1 for p in plans.values() if p.pooled())
    n_si = sum(1 for s in specs if s["target"] == "si")
    print(f"s08_tables: {len(main_specs)} main and {n_si} supplementary tables; pooled levels in {pooled} inputs "
          "(Supplementary Table S1)")


if __name__ == "__main__":
    D.run(main)
