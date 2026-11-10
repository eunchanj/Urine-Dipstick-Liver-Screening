"""Figures 1-4 and the Supplementary Figures, from the aggregate tables only.

Run (it reads only the aggregate tables of AGG_DIR and writes under OUT_ROOT, see display.py):
    PYTHONDONTWRITEBYTECODE=1 python s09_figures.py

Figures (id, target and number). figstyle.save_all writes <id>.pdf, .png, .eps and .tif; display.write_fragment writes
the legend fragment <id>.md and the row of figures_index.csv. The main figures use the main style of figstyle. ALT and
AST are reported with equal weight, in display.OUTCOMES order (ALT first):
  flow                       main 1  one flow of participants for both outcomes: cohort_flow.csv, with the two raw
                                     first-visit rows of cohort_checks.csv above dev_persons
  roc_calibration_alt        main 2  ALT, one column per dataset: ROC panels (a development test set, b KNHANES, with
                                     the rule point; roc_points.csv, roc_points_lr.csv, comparators_alert.csv,
                                     performance.csv, model_comparison.csv) over the calibration panels (c, d;
                                     calibration_bins.csv, calibration.csv) with the histogram of predicted risk
                                     (prediction_histogram.csv)
  roc_calibration_ast        main 3  AST: the same layout, axes and line forms as Figure 2
  subgroups                  main 4  KNHANES forest of the ensemble AUROC by age group, sex, survey period and alcohol
                                     drinking frequency: panel a ALT, panel b AST, on one AUROC axis (subgroups.csv)
  decision_curves            si 1    net benefit in KNHANES and the development test set, per outcome, two rows of
                                     panels each (decision_curve.csv)
  prediction_distributions   si 2    people per band of predicted risk in the two populations, per outcome
                                     (prediction_histogram.csv)
  shap_summary               si 3    mean absolute SHAP value of each input (shap_summary.csv)
  shap_levels                si 4    mean SHAP value at each input level (shap_by_level.csv)

Every figure gives its sample size, in the figure or in its legend. The legends of Figures 2 and 3 print the people and
point to Table 2 for the events: beside a ROC panel, the events would give counts of 1-9 from the curve within one
panel, as (1 - sensitivity) x events.

Nothing is computed beyond formatting and the rounding of drawn values: every number shown is a cell of an aggregate
table, or (one joint exclusion of Figure 1) the difference of two counts shown in it. Checks that stop the run: the
trapezoid area of every plotted ROC curve lies within ROC_TOL of its AUROC; the flow figure shows no count of 0-9 and no
count that follows by subtraction from the others within it (a step that removes 1-9 people is drawn together with the
next step, the events and non-events of each outcome in the final boxes are of 10 or more), and every count equals its
cell; every text of the flow figure lies inside its box and no box, connector or text touches another; a count of the
subgroup figure is shown only when no count of the same column and subgroup variable is hidden, and the groups that are
not drawn (the unknown alcohol frequency) are taken into account; every histogram panel passes display.panel_heights (no
band of 1-9 people follows from the panel total, the ensemble n of performance.csv) and its bars are drawn at counts
rounded to 2 significant figures; the calibration points and their limits are drawn rounded to 0.001 (at the decile
sizes of this study, the Wilson limits then no longer give each decile's n and events exactly); a fixed alert of
decision_curve.csv is withheld at every threshold probability or at none; a legend has at most 350
words; figstyle.save_all (no transparency, black text, no overlapping or clipped text, the display rules and the word
lists of every visible string, and for a main figure its canvas, Arial, text of at least 7 pt and lines of at least 1
pt).
Blank and '<10' cells are unknown: they are left out or marked, never drawn as zero.
"""
import aggregate_guard  # noqa: F401  (first statement: refuses individual-level files, records the files opened)

import math
import re

import display as D
import figstyle as F

import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

GE, TIMES = D.GE, D.TIMES
ROC_TOL = 0.002          # trapezoid area of the 0.005 grid against the AUROC
ROC_AREA = []            # (curve, |trapezoid area - AUROC|) of every plotted curve, for the summary line
CAL_STEP = 3             # decimals of the drawn observed proportions and their limits (see draw_calibration)
LEGEND_WORDS = 350       # the most words of a figure legend (its title, text and notes)
LR_DRAWN = ("LR_ASB", "LR_ALL")          # drawn under the ensemble, in this order
OUT_WORD = dict(D.OUTCOME_SHORT)         # ALT, AST
OUTCOME_TEXT = {
    ("ALT", "DEV"): f"Same-day {D.OUTCOME_LABELS['ALT']}",
    ("ALT", "KN"): D.OUTCOME_LABELS["ALT"],
    ("AST", "DEV"): f"Same-day {D.OUTCOME_LABELS['AST']}",
    ("AST", "KN"): D.OUTCOME_LABELS["AST"],
}
THRESHOLD = dict(D.OUTCOME_LABELS)       # ALT >40 U/L, AST >40 U/L
SUBGROUP_VARS = ("age_band", "sex", "period", "drink_group")
# figure numbers (index 'order'); s10_supplement.FIGURES lists the supplementary ids in this order
MAIN_ORDER = {"flow": 1, "roc_calibration_alt": 2, "roc_calibration_ast": 3, "subgroups": 4}
SI_ORDER = {"decision_curves": 1, "prediction_distributions": 2, "shap_summary": 3, "shap_levels": 4}
ROC_FIGURES = {"ALT": "roc_calibration_alt", "AST": "roc_calibration_ast"}
# tables of s08 cited by the legends, as (target, number) of s08_tables.MAIN_ORDER / SI_ORDER (s08 is not imported):
# the models and the people with each outcome (Figures 2
# to 4 and the decision curves), the rule and the ensemble at its threshold (Figures 2 and 3), the O:E and the counts of
# each subgroup (Figure 4)
S08_TABLES = {"performance": ("main", 2), "comparators": ("main", 3), "subgroups": ("si", 12)}
COMPARATOR_TABLE = {"ALT": "comparators", "AST": "comparators"}    # one table holds both outcomes
PERFORMANCE_TABLE = {"ALT": "performance", "AST": "performance"}


def order_of(fig_id, target):
    return (MAIN_ORDER if target == "main" else SI_ORDER)[fig_id]


def figure_ref(fig_id, target):
    """'Figure 2', 'Supplementary Figure S1'."""
    return D.label("figure", target, order_of(fig_id, target))


def figure_cite(fig_id, target):
    """A figure cited inside a sentence, where Figure is abbreviated: 'Fig. 2', 'Supplementary Fig. S1' (the label
    that opens a legend keeps 'Figure')."""
    return figure_ref(fig_id, target).replace("Figure", "Fig.", 1)


def table_ref(table_id):
    """'Table 3', 'Supplementary Table S12'."""
    target, number = S08_TABLES[table_id]
    return D.label("table", target, number)


def performance_ref(outcomes):
    """The table of the models and of the people and events of `outcomes`: 'Table 2' (it holds both outcomes)."""
    return D.single([table_ref(PERFORMANCE_TABLE[o]) for o in outcomes], "performance table of the outcomes")


def people_text(n):
    """The people of one population: '12,345 people'."""
    return f"{D.fmt_count(n)} people"


def events_pointer(outcomes):
    """Where a legend that prints the people only (see the top of this file) sends the reader for the events: 'the
    number with the outcome is in Table 2', 'the numbers with each outcome are in Table 2'."""
    many = len(outcomes) > 1
    return (f"the number{'s' if many else ''} with {'each' if many else 'the'} outcome {'are' if many else 'is'} in "
            f"{performance_ref(outcomes)}")


class Store:
    """Aggregate tables, each read once (display.read_table: cells as text, blank -> None, '<10' kept)."""

    def __init__(self):
        self.tables = {}

    def __call__(self, name):
        if name not in self.tables:
            self.tables[name] = D.read_table(name)
        return self.tables[name]


# ------------------------------------------------------------------------------------------------ cells and checks
def fval(cell):
    """Float of a cell for drawing; NaN when the cell is blank or masked (unknown, never zero)."""
    if D.is_blank(cell) or D.is_masked(cell):
        return math.nan
    return float(cell)


def column(rows, name):
    return np.array([fval(v) for v in rows[name]], float)


def source_of(dataset):
    return "DEV" if dataset.startswith("DEV") else "KN"


def trapezoid(x, y):
    """Area under the polyline through (x, y), taken in the order of x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def check_roc_area(name, fpr, tpr, auroc, tol=ROC_TOL):
    """The trapezoid area of a plotted ROC curve lies within tol of the AUROC of its table
    (recorded for the summary)."""
    diff = abs(trapezoid(fpr, tpr) - auroc)
    ROC_AREA.append((name, diff))
    if not diff <= tol:
        raise AssertionError(f"{name}: the trapezoid area of the ROC grid differs from the AUROC by {diff:.5f} "
                             f"(tolerance {tol})")
    return diff


def showable(cell):
    """A count that a figure may show: a whole number of 10 or more."""
    c = D.count(cell) if not isinstance(cell, int) else cell
    return isinstance(c, int) and c >= 10


def pct_formatter(decimals=0):
    return FuncFormatter(lambda v, _pos: D.fmt(round(v * 100, 9), decimals))


def axis_count_formatter():
    """Tick labels of a count axis ('1', '100', '10,000'): axis positions, not counts of people."""
    return FuncFormatter(lambda v, _pos: D.fmt_count(int(round(v))) if v >= 10 else D.fmt(v, 0))


def _luminance(colour):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in mcolors.to_rgb(colour)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ink(colour, contrast=3.0):
    """The colour, darkened in 5% steps until its contrast with white is at least `contrast` (lines and markers in
    the light dataset colours stay legible); colours that already reach it are returned unchanged."""
    rgb = mcolors.to_rgb(colour)
    for k in range(21):
        c = tuple(v * (1 - 0.05 * k) for v in rgb)
        if 1.05 / (_luminance(c) + 0.05) >= contrast:
            return mcolors.to_hex(c)
    return F.BLACK


def join_or(words):
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " or " + words[-1]


def lower_first(s):
    """'Same-day ALT' -> 'same-day ALT'; a first word with a capital after its first letter ('ALT', 'KNHANES') is
    left as it is."""
    first = s.split(" ", 1)[0]
    return s if any(ch.isupper() for ch in first[1:]) else s[:1].lower() + s[1:]


def drawing_axes(fig, width_mm, height_mm):
    """Axes over the whole figure in millimetres (1 data unit = 1 mm), without axis lines or ticks."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    return ax


def publish(fig, fig_id, target, order, title, legend, notes, sources):
    """Check and save a figure (figstyle.save_all) and write its legend fragment and index row. `fig` is the figure,
    or {part: figure} for a figure saved as separate files (display.FIGURE_PARTS), which has one legend and one row.
    The title, legend and notes are checked first (display rules, word lists, the legend limit of LEGEND_WORDS words),
    so that a text the fragment would refuse leaves no image without its legend. A main figure is made, drawn and
    published inside its figstyle.main_style() block."""
    for text, where in [(title, "title"), (legend, "legend")] + [(n, "note") for n in notes]:
        D.check_text(text, f"{fig_id} {where}")
        D.check_words(text, f"{fig_id} {where}")
        D.check_legend(text, f"{fig_id} {where}")
    parts = [len(t.split()) for t in [title, legend] + list(notes)]
    if sum(parts) > LEGEND_WORDS:
        raise AssertionError(f"{fig_id}: the legend has {sum(parts)} words (at most {LEGEND_WORDS}): title {parts[0]}, "
                             f"legend {parts[1]}, notes {parts[2:]}")
    paths = []
    for part, f in (fig if isinstance(fig, dict) else {None: fig}).items():
        F.assert_opaque(f)
        paths += F.save_all(f, D.figure_base(fig_id, target, part))
        plt.close(f)
    D.write_fragment("figure", fig_id, title, legend, notes, list(dict.fromkeys(sources)), target=target,
                     order=order, files=paths)
    return f"{D.label('figure', target, order)} ({fig_id})"


# ============================================================================================ Figure 1: flow
class FlowError(ValueError):
    """The flow figure cannot be drawn from cohort_flow.csv without showing a small count, or the table does not add
    up."""


class FlowSizes:
    """The sizes of the flow figure: millimetres, with text and lines in points. The room inside and between the boxes
    grows with the text; `width`, the columns and `split_stem` (one line of the split note, or two) are set for the
    figure."""

    def __init__(self, width, max_height, fs, head_fs, lw, heavy_lw, dev_right, kn_left, col_gap, kn_excl, split_stem):
        k = fs / 7.0
        self.width, self.max_height = width, max_height     # canvas
        self.fs, self.head_fs = fs, head_fs                 # box text, stream headers
        self.lw, self.heavy_lw = lw, heavy_lw               # lines; the outline of a box of the 'heavy' style
        self.line_mm = fs * 1.26 / 72 * 25.4                # line pitch inside a box
        self.base_mm = 0.36 * fs / 72 * 25.4                # baseline below the centre of a line
        self.pad_x, self.pad_y = 1.3 * k, 0.9 * k           # room inside a box
        self.count_gap = 1.5 * k                            # least room between a label and its count
        self.rounding = 0.7 * k                             # corners of a box
        self.ex_margin = 1.1 * k                            # least room from an exclusion box to the boxes around it
        self.arrow_min = 3.6 * k                            # least length of an arrow between two boxes
        self.arrow_head = 5.5 * k                           # arrow head (matplotlib's mutation scale)
        self.split_stem = split_stem                        # development split: stem to the branch line
        self.split_gap = split_stem + 4.5 * k               # development split: whole gap
        self.note_dx = 1.3 * k                              # split note to the stem
        self.head_gap = 2.4 * k                             # stream header to the first box
        self.left, self.right = 0.4, width - 0.4            # outer edges of the two streams
        self.dev_right, self.kn_left = dev_right, kn_left   # inner edges of the two streams
        self.col_gap, self.kn_excl = col_gap, kn_excl       # main to exclusion column; KNHANES exclusion column


# Figure 1 at the 'flow' canvas of figstyle (183 by 230 mm at most): 7 pt text, 1 pt lines; the development stream is
# the wider one, since it has more steps and longer exclusion texts
FLOW_SIZE = FlowSizes(*F.MAIN_CANVAS["flow"], F.MAIN_TEXT_PT, F.MAIN_TEXT_PT, F.MAIN_LINE_PT, 1.5, 104.0, 108.0, 4.0,
                      34.0, 6.0)
RAW_ITEMS = ("raw first-visit persons", "raw first-visit persons not in the processed file",
             "persons without a raw first-visit record")
DEV_CHAIN = ("dev_persons", "dev_period", "dev_blood_day", "dev_adult", "dev_enzymes", "dev_inputs", "dev_bmi")
KN_CHAIN = ("kn_records", "kn_adult", "kn_inputs", "kn_anthro", "kn_enzymes")
# box text of each step (written for this figure; a key not listed takes its label from cohort_flow.csv, through
# own_terms)
STEP_TEXT = {
    "raw": "People with a first urinalysis in the hospital records",
    "dev_persons": "First urinalysis with a blood test within 365 days",
    "dev_blood_day": "Blood test on the day of the urinalysis",
    "dev_adult": "Aged 19-110 years at the urinalysis",
    "dev_enzymes": "Same-day ALT and AST, both classifiable",
    "dev_inputs": "All dipstick results and sex recorded",
    "dev_bmi": "Development cohort: BMI on the day of the urinalysis or in the 30 days before it",
    "dev_train": "Training set",
    "dev_test": "Test set",
    "kn_adult": "Aged 19 years or older",
    "kn_inputs": "All dipstick results and sex recorded",
    "kn_anthro": "BMI available: height and weight recorded and within the accepted ranges",
    "kn_enzymes": "Eligible adults: ALT and AST recorded",
}
# exclusion text of a step without reason rows
NONE_TEXT = {
    # people of the raw first-visit table who are not in the processed file: no blood test within 365 days of the first
    # urinalysis (no blood test at all, or only one more than 365 days away), as the count of cohort_checks.csv says
    "raw": "No blood test within 365 days of the first urinalysis",
    "dev_blood_day": "No blood test on the day of the urinalysis",
}
REASON_TEXT = {
    "dev_no_alt_ast": "Neither same-day ALT nor AST",
    "dev_ast_only": "AST only (no same-day ALT)",
    "dev_alt_only": "ALT only (no same-day AST)",
    "dev_enzyme_undetermined": "Censored and not classifiable",
    "dev_missing_dipstick": "One or more dipstick results missing",
    "dev_bmi_none": "No BMI on the day of the urinalysis or in the 30 days before it",
    "kn_missing_dipstick": "One or more dipstick results missing",
    "kn_no_alt_ast": "Neither ALT nor AST recorded",
    "kn_ast_only": "AST only (ALT not recorded)",
    "kn_alt_only": "ALT only (AST not recorded)",
    "kn_anthro_range": "Height, weight or BMI outside the accepted ranges",
}
# one line for the reasons of a step when one of them is masked (otherwise their labels joined), and for a step drawn
# together with the next one
MERGED_TEXT = {
    "dev_period": "Urinalysis outside the study period or its date not readable",
    "dev_enzymes": "One or both enzymes missing on the day of the urinalysis or not classifiable",
    "dev_inputs": "Sex not recorded or one or more dipstick results missing",
    "kn_adult": "Aged under 19 years or age not recorded",
    "kn_inputs": "Sex not recorded or one or more dipstick results missing",
    "kn_anthro": "No BMI: height or weight not recorded or outside the accepted ranges",
    "kn_enzymes": "One or both enzymes not recorded",
}
# the enzyme line when no one's results were undetermined (the count of dev_enzyme_undetermined is 0)
ENZYMES_MISSING_TEXT = "One or both enzymes missing on the day of the urinalysis"
# the rows of cohort_checks.csv whose value cell gives the accepted ranges of height, weight and BMI (KNHANES, hospital)
RANGE_ITEMS = ("adults at kn_inputs with height, weight or BMI outside the ranges (excluded)",
               "anthropometry rows dropped before the BMI join: height, weight or BMI outside the ranges")
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
KEEP_TOGETHER = (re.compile(rf"\b(\d{{1,2}}) ({MONTHS}) (\d{{4}})\b"), re.compile(rf"\b({MONTHS}) (\d{{4}})\b"),
                 re.compile(r"(\d) (h|years|days|U/L|kg/m[2²])\b"))
NBSP = "\x1f"     # stands for a space that a line must not break at, while wrapping only
SINGLE_LINE = {"dev_adult"}   # the age exclusion is one box: its over-110 reason alone could be worked out
SPLIT_NOTE = "Random split by person, stratified by ALT and AST above 40 U/L"
DEV_HEADER = "Development: hospital records"
KN_HEADER = "External validation: KNHANES, used only for evaluation"
BOX_STYLES = {
    "step": dict(face="#FFFFFF", edge=F.BLACK),
    "excl": dict(face="#FFFFFF", edge=F.GREY),
    "dev_train": dict(face="#EDEDED", edge=F.BLACK),
    "dev_test": dict(face=F.tint(F.DATASET_COLOURS["DEV_TEST"], 0.8), edge=F.BLACK),
    "kn": dict(face=F.tint(F.DATASET_COLOURS["KN"], 0.75), edge=F.BLACK),
}


class Box:
    """One box of the flow figure. items: ('c', text) centred and wrapped, ('n', text) centred and never wrapped (a
    count), ('p', label, count) label on the left and count on the right of the label's last line (a wrapped label
    reads on to its count), ('l', text) on the left. cells: the counts it shows, as (text shown, table, row, column),
    for the number check."""

    def __init__(self, items, style, cells=()):
        self.items, self.style, self.cells = list(items), style, list(cells)
        self.lines, self.min_height = [], 0.0
        self.x0 = self.x1 = self.y0 = self.y1 = None
        self.patch, self.texts, self.sizes = None, [], None

    def layout(self, measure, x0, x1):
        self.x0, self.x1, self.sizes = x0, x1, measure.sizes
        inner = x1 - x0 - 2 * self.sizes.pad_x
        lines = []
        for it in self.items:
            if it[0] == "n":
                lines.append(("c", it[1], None))
            elif it[0] in ("c", "l"):
                lines += [(it[0], s, None) for s in measure.wrap(it[1], inner)]
            else:
                parts = measure.wrap(it[1], inner - measure.width(it[2]) - self.sizes.count_gap)
                lines += [("l", s, None) for s in parts[:-1]]
                lines.append(("p", parts[-1], it[2]))
        self.lines = lines
        return self

    @property
    def natural_height(self):
        return len(self.lines) * self.sizes.line_mm + 2 * self.sizes.pad_y

    @property
    def height(self):
        return max(self.natural_height, self.min_height)

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2

    def draw(self, ax, y_top):
        self.y1, self.y0 = y_top, y_top - self.height
        st, sz = BOX_STYLES[self.style], self.sizes
        self.patch = FancyBboxPatch((self.x0, self.y0), self.x1 - self.x0, self.height,
                                    boxstyle=f"round,pad=0,rounding_size={sz.rounding}", facecolor=st["face"],
                                    edgecolor=st["edge"], linewidth=sz.heavy_lw if st.get("heavy") else sz.lw, zorder=2)
        ax.add_patch(self.patch)
        top = y_top - (self.height - self.natural_height) / 2 - sz.pad_y
        kw = dict(fontsize=sz.fs, va="baseline", zorder=3)
        for i, (kind, left, right) in enumerate(self.lines):
            yb = top - (i + 0.5) * sz.line_mm - sz.base_mm
            if kind == "c":
                self.texts.append(ax.text(self.xc, yb, left, ha="center", **kw))
            else:
                self.texts.append(ax.text(self.x0 + sz.pad_x, yb, left, ha="left", **kw))
            if right:
                self.texts.append(ax.text(self.x1 - sz.pad_x, yb, right, ha="right", **kw))
        return self


class Measure:
    """Widths of texts (millimetres) at the figure's resolution, and greedy word wrapping to a width. `sizes`: the
    FlowSizes of the figure (the boxes laid out with it take theirs from it)."""

    def __init__(self, fig, sizes=FLOW_SIZE):
        self.fig, self.sizes, self.size, self.cache = fig, sizes, sizes.fs, {}
        self.renderer = fig.canvas.get_renderer()

    def width(self, s, weight="normal"):
        key = (s, weight)
        if key not in self.cache:
            w, _h, _d = self.renderer.get_text_width_height_descent(
                s, FontProperties(size=self.size, weight=weight), ismath=False)
            self.cache[key] = w / self.fig.dpi * 25.4
        return self.cache[key]

    def wrap(self, text, width, balance=True):
        """Lines of at most `width` mm (a single word wider than that stays whole); a date such as '14 August
        2025' and a number with its unit ('8 h', '365 days') are never broken. Text that needs two lines is
        broken where the longer line is shortest (balance=True), so no line holds a single trailing word."""
        text = str(text)
        for rx in KEEP_TOGETHER:
            text = rx.sub(lambda m: NBSP.join(m.groups()), text)
        words = [w for w in text.split(" ") if w]
        lines, cur = [], ""
        for word in words:
            cand = f"{cur} {word}" if cur else word
            if cur and self.width(cand.replace(NBSP, " ")) > width:
                lines.append(cur)
                cur = word
            else:
                cur = cand
        lines = lines + [cur] if cur else lines
        if balance and len(lines) == 2:
            best = None
            for i in range(1, len(words)):
                a, b = " ".join(words[:i]), " ".join(words[i:])
                wa, wb = self.width(a.replace(NBSP, " ")), self.width(b.replace(NBSP, " "))
                if max(wa, wb) <= width and (best is None or max(wa, wb) < best[0]):
                    best = (max(wa, wb), [a, b])
            if best:
                lines = best[1]
        return [s.replace(NBSP, " ") for s in lines]


def _int_count(row, col, what):
    c = D.count(row[col])
    if not isinstance(c, int):
        raise FlowError(f"{row['key']}: {col} is {'masked' if c == D.MASKED else 'blank'} ({what})")
    return c


def _shown(n, where):
    """A count shown in the flow figure: a whole number of 10 or more."""
    if not (isinstance(n, int) and n >= 10):
        raise FlowError(f"{where}: the flow figure would show the count {D.fmt_count(n)}")
    return D.fmt_count(n)


def step_excluded(row):
    """The people a step removes (its excluded cell): an int or '<10'; FlowError when the cell is blank."""
    c = D.count(row["excluded"])
    if c is None:
        raise FlowError(f"{row['key']}: the exclusions of the step are blank")
    return c


def removes_few(excluded):
    """True for a step that removes 1-9 people (a masked cell or a count of 1-9)."""
    return excluded == D.MASKED or (isinstance(excluded, int) and 1 <= excluded <= 9)


def own_terms(label):
    """A cohort_flow.csv label in this figure's words: 'Urine test ...' -> 'Urinalysis ...', 'Urine-test date ...' ->
    'Date of the urinalysis ...', 'Persons' -> 'People' (the dates and counts stay those of the table). FlowError for
    a label that would still say 'urine test' or 'dipstick field missing': it needs a text in this file."""
    text = re.sub(r"^Urine-test date\b", "Date of the urinalysis", str(label))
    text = re.sub(r"^Urine test\b", "Urinalysis", text)
    text = re.sub(r"\bPersons\b", "People", re.sub(r"\bpersons\b", "people", text))
    if re.search(r"(?i)urine[- ]test|dipstick field missing", text):
        raise FlowError(f"cohort_flow.csv label {label!r} has no wording for the flow figure")
    return text


def step_label(key, row):
    return STEP_TEXT.get(key) or own_terms(row["label"])


def merged_text(key, reasons):
    """The one line for the reasons of step `key` (rows of its reasons) in MERGED_TEXT; the enzyme step of the
    development stream leaves out 'not classifiable' when no one's results were undetermined."""
    if key == "dev_enzymes" and any(r["key"] == "dev_enzyme_undetermined" and D.count(r["n"]) == 0 for r in reasons):
        return ENZYMES_MISSING_TEXT
    return MERGED_TEXT.get(key)


def age_exclusion_text(reasons):
    """One line for the age exclusion: the age reasons with persons, in words."""
    has = {r["key"]: (D.count(r["n"]) == D.MASKED or (isinstance(D.count(r["n"]), int) and D.count(r["n"]) > 0))
           for r in reasons}
    under, over, missing = has.get("dev_age_under19"), has.get("dev_age_over110"), has.get("dev_age_missing")
    if under and over:
        text = D.FLOW_LABELS["dev_adult_excluded"]
    elif under:
        text = "Aged <19 years"
    elif over:
        text = "Aged >110 years"
    else:
        text = ""
    if missing:
        text = f"{text}, or age not recorded" if text else "Age not recorded"
    return text or None


def exclusion_box(step, reasons):
    """Exclusion box between `step` and its parent ('Excluded' and the count, then the reasons), or None when the
    step removes no one. A reason with no persons is left out; when a reason is masked or 1-9 (or the step is in
    SINGLE_LINE) the reasons are named on one line under the step's own count, so no masked count follows by
    subtraction."""
    key = step["key"]
    total = _int_count(step, "excluded", "exclusions of the step")
    if total == 0:
        return None
    text_total = _shown(total, f"{key} excluded")
    cells = [(text_total, "cohort_flow.csv", key, "excluded")]
    parts = []
    for r in reasons:
        c = D.count(r["n"])
        if c is None:
            raise FlowError(f"{r['key']}: blank reason count")
        if c != 0:
            parts.append((r, c))
    small = [r["key"] for r, c in parts if c == D.MASKED or c <= 9]
    known = sum(c for _, c in parts if isinstance(c, int))
    k_masked = sum(1 for _, c in parts if c == D.MASKED)
    if parts and not (known + k_masked <= total <= known + 9 * k_masked):
        raise FlowError(f"{key}: the reasons do not add up to the step's exclusions")
    items = [("p", "Excluded", text_total)]
    labels = [REASON_TEXT.get(r["key"]) or own_terms(r["label"]) for r, _ in parts]
    if not parts:
        if key not in NONE_TEXT:
            raise FlowError(f"{key}: no reason rows and no text for its exclusions")
        items.append(("l", NONE_TEXT[key]))
    elif key in SINGLE_LINE or small or len(parts) == 1:
        text = age_exclusion_text(reasons) if key == "dev_adult" else None
        if not text and len(labels) == 1:
            text = labels[0]
        text = text or merged_text(key, reasons) or join_or([labels[0]] + [lower_first(s) for s in labels[1:]])
        items.append(("l", text))
    else:
        for (r, c), label in zip(parts, labels):
            shown = _shown(c, r["key"])
            items.append(("p", label, shown))
            cells.append((shown, "cohort_flow.csv", r["key"], "n"))
    return Box(items, "excl", cells)


def reasons_line(key, reasons):
    """One line for the people a step removes: its text in MERGED_TEXT or NONE_TEXT, or its reasons joined."""
    labels = [REASON_TEXT.get(r["key"]) or own_terms(r["label"]) for r in reasons.get(key, [])]
    return merged_text(key, reasons.get(key, [])) or NONE_TEXT.get(key) or join_or(
        [labels[0]] + [lower_first(s) for s in labels[1:]])


def joint_exclusion_box(first, few, key, n, reasons):
    """The exclusion box of a step that removes 1-9 people drawn with the next step (see _chain_boxes): one count, the
    people between `first` (the step before them) and `key`, and one line for the reasons of the two steps."""
    shown = _shown(n[first] - n[key], f"{few} and {key} excluded together")
    text = f"{reasons_line(few, reasons)}; {lower_first(reasons_line(key, reasons))}"
    return Box([("p", "Excluded", shown), ("l", text)], "excl", [(shown, "difference", (first, key), "n")])


def flow_model(flow, checks):
    """Boxes of the flow figure of both outcomes (Figure 1) from cohort_flow.csv and cohort_checks.csv (no drawing). The
    development stream runs to the development cohort and its random split, whose two boxes give the people and the
    events of each outcome (the same people for ALT and AST); the KNHANES stream runs to the eligible adults, whose box
    gives the same. Raises FlowError when the table does not add up or when a box would show a count of 0-9 or a count
    that follows from the others of the figure.

    Returns a dict: dev (main boxes), dev_excl (exclusion box after each main box but the last), dev_split (training,
    test), kn, kn_excl, raw (whether the raw first-visit rows are drawn), folded (steps drawn with the next step),
    notes (printed when the run ends)."""
    records = flow.to_dict("records")
    rows = {r["key"]: r for r in records}
    finals = [f"{o}_{d}" for o in D.OUTCOMES for d in ("DEV_TRAIN", "DEV_TEST", "KN")]
    missing = [k for k in DEV_CHAIN + KN_CHAIN + tuple(finals) if k not in rows]
    if missing:
        raise FlowError(f"cohort_flow.csv lacks the rows {missing}")
    not_final = [k for k in finals if rows[k]["kind"] != "final"]
    if not_final:
        raise FlowError(f"{not_final} are not final rows")
    reasons = {}
    for r in records:
        if r["kind"] == "reason":
            reasons.setdefault(r["parent"], []).append(r)
    n = {k: _int_count(r, "n", "persons") for k, r in rows.items() if r["kind"] != "reason"}
    notes, folded = [], []

    def follows(key, parent):
        """The step removes parent n - its n persons, as its excluded cell says (1-9 when the cell is masked)."""
        ex, gone = step_excluded(rows[key]), n[parent] - n[key]
        if gone < 0 or (gone != ex if isinstance(ex, int) else not 1 <= gone <= 9):
            raise FlowError(f"{key}: n does not follow from {parent} and the exclusions")

    for chain in (DEV_CHAIN, KN_CHAIN):
        for parent, key in zip(chain, chain[1:]):
            if rows[key]["parent"] != parent:
                raise FlowError(f"{key}: parent {rows[key]['parent']!r}, expected {parent!r}")
            follows(key, parent)

    def step_box(key, label, style="step"):
        """Main box of a step: its label and n."""
        shown = _shown(n[key], key)
        return Box([("c", label), ("n", f"n = {shown}")], style, [(shown, "cohort_flow.csv", key, "n")])

    def final_box(label, keys, style):
        """Box of an analysis population: its label, the people (the same for both outcomes) and, on a line each, the
        events of each outcome (keys: its final row of each outcome, ALT then AST). Non-events of 1-9 would follow from
        n and the events, so each is 10 or more."""
        first = keys[0]
        if any(n[k] != n[first] for k in keys):
            raise FlowError(f"{keys}: the outcomes differ in their people")
        shown = _shown(n[first], first)
        items, cells = [("c", label), ("n", f"n = {shown}")], [(shown, "cohort_flow.csv", first, "n")]
        for outcome, key in zip(D.OUTCOMES, keys):
            ev = _int_count(rows[key], "events", "events")
            shown_ev = _shown(ev, f"{key} events")
            _shown(n[key] - ev, f"{key} persons without the outcome")
            items.append(("n", f"{D.OUTCOME_LABELS[outcome]}: {shown_ev}"))
            cells.append((shown_ev, "cohort_flow.csv", key, "events"))
        return Box(items, style, cells)

    # development: raw first-visit rows (cohort_checks.csv) above dev_persons, when they add up
    raw = _raw_rows(checks, n["dev_persons"], notes)
    dev, dev_excl = [], []
    if raw:
        raw_n, out_n = raw
        shown = _shown(raw_n, "raw first-visit persons")
        dev.append(Box([("c", STEP_TEXT["raw"]), ("n", f"n = {shown}")], "step",
                       [(shown, "cohort_checks.csv", RAW_ITEMS[0], "n")]))
        shown = _shown(out_n, "raw first-visit persons not in the processed file")
        dev_excl.append(Box([("p", "Excluded", shown), ("l", NONE_TEXT["raw"])], "excl",
                            [(shown, "cohort_checks.csv", RAW_ITEMS[1], "n")]))
    _chain_boxes(DEV_CHAIN, rows, reasons, n, dev, dev_excl, step_box, folded)

    # training and test sets of the development cohort (one split, the same people for both outcomes)
    for o in D.OUTCOMES:
        if n[f"{o}_DEV_TRAIN"] + n[f"{o}_DEV_TEST"] != n["dev_bmi"]:
            raise FlowError(f"the {o} training and test persons do not add up to the development cohort")
    split = [final_box(STEP_TEXT["dev_train"], tuple(f"{o}_DEV_TRAIN" for o in D.OUTCOMES), "dev_train"),
             final_box(STEP_TEXT["dev_test"], tuple(f"{o}_DEV_TEST" for o in D.OUTCOMES), "dev_test")]

    # KNHANES: the common chain down to the eligible adults, whose box gives the events of each outcome
    kn, kn_excl = [], []
    kn_keys = tuple(f"{o}_KN" for o in D.OUTCOMES)
    if any(n[k] != n[KN_CHAIN[-1]] for k in kn_keys):
        raise FlowError("the KNHANES analysis populations differ from the eligible adults")
    _chain_boxes(KN_CHAIN, rows, reasons, n, kn, kn_excl, step_box, folded,
                 last_box=lambda key, label: final_box(label, kn_keys, "kn"))
    return dict(dev=dev, dev_excl=dev_excl, dev_split=split, kn=kn, kn_excl=kn_excl, raw=bool(raw), folded=folded,
                notes=notes)


def _chain_boxes(keys, rows, reasons, n, mains, excls, step_box, folded, last_box=None):
    """Main and exclusion boxes of a chain of steps (appended to mains and excls). The first step has no exclusion
    box of its own; a later step that removes no one is named in its parent's box (both hold the same persons). A step
    that removes 1-9 people is drawn with the next step, in one main box naming both and one exclusion box counting
    both (the people of the two boxes around it would otherwise give the masked count by difference); it is added to
    `folded`, and FlowError stops the figure when no next step shows exclusions of its own. `last_box(key, label)`
    makes the main box of the last step of the chain (default: step_box)."""
    few = None                     # a step of 1-9 exclusions, drawn with the next step
    for i, key in enumerate(keys):
        row = rows[key]
        make = last_box if last_box is not None and i == len(keys) - 1 else step_box
        if i == 0:
            mains.append(make(key, step_label(key, row)))
            continue
        ex = step_excluded(row)
        if removes_few(ex):
            if few is not None or i == len(keys) - 1:
                raise FlowError(f"{key}: a step that removes 1-9 people cannot be drawn with the next step; a display "
                                "decision is needed")
            few = key
            folded.append(key)
            continue
        if few is not None:
            if ex == 0:
                raise FlowError(f"{few}: the next step removes no one, so 1-9 people would show by difference; a "
                                "display decision is needed")
            excls.append(joint_exclusion_box(keys[i - 2], few, key, n, reasons))
            box = make(key, step_label(key, row))
            box.items.insert(0, ("c", step_label(few, rows[few])))
            mains.append(box)
            few = None
            continue
        if ex == 0:
            if i == len(keys) - 1 and last_box is not None:      # the parent's box becomes the last box of the chain
                prev = mains.pop()
                box = make(key, step_label(key, row))
                box.items[:0] = [it for it in prev.items if it[0] == "c"]
                mains.append(box)
                continue
            box = mains[-1]
            box.items.insert(sum(1 for it in box.items if it[0] == "c"), ("c", step_label(key, row)))
            box.cells.append((D.fmt_count(n[key]), "cohort_flow.csv", key, "n"))
            continue
        excls.append(exclusion_box(row, reasons.get(key, [])))
        mains.append(make(key, step_label(key, row)))


def _raw_rows(checks, dev_persons, notes):
    """(raw first-visit persons, of them not in the processed file) from cohort_checks.csv when they lead exactly to
    dev_persons; None otherwise (the figure then starts at dev_persons, and a note says why)."""
    def item(text):
        r = D.select(checks, source="DEV", item=text)
        return D.count(r.iloc[0]["n"]) if len(r) == 1 else None

    raw, out, no_raw = (item(t) for t in RAW_ITEMS)
    if not (isinstance(raw, int) and isinstance(out, int) and raw >= 10 and out >= 10):
        notes.append("the raw first-visit rows of cohort_checks.csv are missing or masked: the flow figure starts at "
                     "dev_persons")
        return None
    if raw - out != dev_persons or no_raw not in (0, None):
        notes.append("the raw first-visit rows of cohort_checks.csv do not lead exactly to dev_persons: the flow "
                     "figure starts at dev_persons")
        return None
    return raw, out


def flow_cells_check(model, store):
    """Every count shown in the flow figure equals the cell it comes from (or, for the joint exclusion of two steps,
    the difference of their people) and is drawn in its box."""
    flow = store("cohort_flow.csv")
    checks = store("cohort_checks.csv")
    boxes = _all_boxes(model)
    n_checked = 0
    for b in boxes:
        drawn = [t.get_text() for t in b.texts]
        for shown, table, key, col in b.cells:
            if table == "cohort_flow.csv":
                cell = D.one(flow, key=key)[col]
            elif table == "cohort_checks.csv":
                cell = D.one(checks, source="DEV", item=key)[col]
            else:                                       # 'difference': (step before the pair, last step of the pair)
                a, z = key
                cell = D.count(D.one(flow, key=a)["n"]) - D.count(D.one(flow, key=z)["n"])
            if D.fmt_count(cell) != shown:
                raise AssertionError(f"flow figure: {key} {col} shows {shown}, its cell holds {D.fmt_count(cell)}")
            if not any(shown in t.split() or t.endswith(" " + shown) or t == shown for t in drawn):
                raise AssertionError(f"flow figure: the count {shown} of {key} is not drawn in its box")
            n_checked += 1
    return n_checked


def _all_boxes(model):
    out = list(model["dev"]) + [b for b in model["dev_excl"] if b is not None] + list(model["dev_split"])
    out += list(model["kn"]) + [b for b in model["kn_excl"] if b is not None]
    return out


def _segment_hits(p, q, box, inset=0.25):
    """True when the axis-parallel segment p-q passes through the inside of a box (touching an edge is allowed)."""
    x0, x1, y0, y1 = box.x0 + inset, box.x1 - inset, box.y0 + inset, box.y1 - inset
    (ax_, ay), (bx, by) = p, q
    if abs(ax_ - bx) < 1e-9:
        lo, hi = sorted((ay, by))
        return x0 < ax_ < x1 and lo < y1 and hi > y0
    lo, hi = sorted((ax_, bx))
    return y0 < ay < y1 and lo < x1 and hi > x0


def flow_layout_problems(fig, boxes, segments, free_texts):
    """The flow figure: texts outside their boxes, boxes that touch, connectors through boxes, free texts on boxes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.axes[0].transData.inverted()
    out = []
    for b in boxes:
        for t in b.texts:
            if F.text_outside_box(t, b.patch, pad_pt=1.0):
                out.append(f"text {t.get_text()[:30]!r} runs outside its box")
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a.x0 < b.x1 + 0.5 and b.x0 < a.x1 + 0.5 and a.y0 < b.y1 + 0.5 and b.y0 < a.y1 + 0.5:
                out.append(f"boxes {a.lines[0][1][:20]!r} and {b.lines[0][1][:20]!r} touch")
    for p, q in segments:
        for b in boxes:
            if _segment_hits(p, q, b):
                out.append(f"a connector passes through the box {b.lines[0][1][:24]!r}")
    for t in free_texts:
        e = t.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inv.transform((e.x0, e.y0)), inv.transform((e.x1, e.y1))
        for b in boxes:
            if x0 < b.x1 + 0.4 and b.x0 < x1 + 0.4 and y0 < b.y1 + 0.4 and b.y0 < y1 + 0.4:
                out.append(f"text {t.get_text()[:24]!r} touches the box {b.lines[0][1][:24]!r}")
        for p, q in segments:
            if (min(p[0], q[0]) < x1 + 0.3 and max(p[0], q[0]) > x0 - 0.3 and min(p[1], q[1]) < y1 + 0.3
                    and max(p[1], q[1]) > y0 - 0.3):
                out.append(f"text {t.get_text()[:24]!r} touches a connector")
    return out


def draw_flow(model, sizes=FLOW_SIZE):
    """The flow figure in the sizes of `sizes` (FlowSizes) from the boxes of flow_model(): development stream (left) and
    KNHANES stream (right), each a column of main boxes with their exclusions to the right; below the development cohort
    its random split, with the events of each outcome in the box of each set."""
    W = sizes.width
    fig = F.new_figure(W, sizes.max_height, layout=None)
    ax = drawing_axes(fig, W, sizes.max_height)
    m = Measure(fig, sizes)
    dx0, dx1, kx0, kx1 = sizes.left, sizes.dev_right, sizes.kn_left, sizes.right
    ws = (dx1 - dx0 - sizes.col_gap) / 2             # development column width (main boxes, training, test)
    # KNHANES: the main column is wider than the exclusion column, so that its longer labels stay on one line
    wke = sizes.kn_excl                              # KNHANES exclusion column width
    wk = kx1 - kx0 - sizes.col_gap - wke             # KNHANES main column width
    dev_main, dev_ex = (dx0, dx0 + ws), (dx1 - ws, dx1)
    kn_main, kn_ex = (kx0, kx0 + wk), (kx1 - wke, kx1)

    dev, dev_excl = model["dev"], model["dev_excl"]
    kn, kn_excl = model["kn"], model["kn_excl"]
    for b in dev:
        b.layout(m, *dev_main)
    for b in dev_excl:
        if b is not None:
            b.layout(m, *dev_ex)
    split = model["dev_split"]
    for (x0, x1), b in zip((dev_main, dev_ex), split):
        b.layout(m, x0, x1)
    for b in kn:
        b.layout(m, *kn_main)
    for b in kn_excl:
        if b is not None:
            b.layout(m, *kn_ex)
    # equal heights in the row of the split
    h = max(b.natural_height for b in split)
    for b in split:
        b.min_height = h

    def gaps(excls):
        return [sizes.arrow_min if e is None else max(e.height + 2 * sizes.ex_margin, sizes.arrow_min) for e in excls]

    dev_gaps, kn_gaps = gaps(dev_excl), gaps(kn_excl)
    dev_h = sum(b.height for b in dev) + sum(dev_gaps) + sizes.split_gap + split[0].height
    kn_h = sum(b.height for b in kn) + sum(kn_gaps)
    head = 1.0 + sizes.head_fs / 72 * 25.4 * 1.15 + sizes.head_gap
    H = math.ceil(head + max(dev_h, kn_h) + 1.0)
    if H > sizes.max_height:
        raise FlowError(f"the flow figure needs {H} mm, more than {sizes.max_height:.0f} mm")
    # the shorter stream spreads its spare height over its gaps
    if dev_h < kn_h:
        dev_gaps = [g + (kn_h - dev_h) / len(dev_gaps) for g in dev_gaps]
    else:
        kn_gaps = [g + (dev_h - kn_h) / len(kn_gaps) for g in kn_gaps]
    fig.set_size_inches(W * F.MM, H * F.MM)
    ax.set_ylim(0, H)

    segments, free = [], []

    def line(points):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=F.BLACK, linewidth=sizes.lw, solid_capstyle="butt", zorder=1)
        segments.extend(zip(points[:-1], points[1:]))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=F.BLACK, linewidth=sizes.lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=sizes.arrow_head))
        segments.append(((x0, y0), (x1, y1)))

    def chain(mains, excls, gap_list, y):
        for i, b in enumerate(mains):
            b.draw(ax, y)
            if i < len(mains) - 1:
                nxt = b.y0 - gap_list[i]
                arrow(b.xc, b.y0, b.xc, nxt)
                e = excls[i]
                if e is not None:
                    ym = (b.y0 + nxt) / 2
                    e.draw(ax, ym + e.height / 2)
                    arrow(b.xc, ym, e.x0, ym)
                y = nxt
        return mains[-1]

    top = H - head
    free.append(ax.text(dx0, H - 1.0, DEV_HEADER, ha="left", va="top", fontsize=sizes.head_fs, fontweight="bold"))
    free.append(ax.text(kx0, H - 1.0, KN_HEADER, ha="left", va="top", fontsize=sizes.head_fs, fontweight="bold"))

    # development stream and its random split
    cohort = chain(dev, dev_excl, dev_gaps, top)
    y_bar = cohort.y0 - sizes.split_stem
    line([(cohort.xc, cohort.y0), (cohort.xc, y_bar)])
    line([(split[0].xc, y_bar), (split[1].xc, y_bar)])
    y_split = cohort.y0 - sizes.split_gap
    for b in split:
        arrow(b.xc, y_bar, b.xc, y_split)
        b.draw(ax, y_split)
    x_note = cohort.xc + sizes.note_dx
    free.append(ax.text(x_note, (cohort.y0 + y_bar) / 2, "\n".join(m.wrap(SPLIT_NOTE, dx1 - x_note)), ha="left",
                        va="center", fontsize=sizes.fs))

    # KNHANES stream, down to the eligible adults
    chain(kn, kn_excl, kn_gaps, top)

    boxes = _all_boxes(model)
    problems = flow_layout_problems(fig, boxes, segments, free)
    if problems:
        raise AssertionError("flow figure layout: " + "; ".join(problems))
    return fig


def ranges_note(checks):
    """The sentence of the Figure 1 legend with the accepted ranges of height, weight and BMI, from the two RULE rows of
    cohort_checks.csv that hold them (one for the hospital records, one for KNHANES: they must be the same). Stops
    with ValueError when the cell is not written as expected or the two differ."""
    kn, hospital = (D.one(checks, source="RULE", item=item)["value"] for item in RANGE_ITEMS)
    m = re.fullmatch(r"height (\S+) cm, weight (\S+) kg, BMI (\S+) kg/m2", str(kn))
    if kn != hospital or not m:
        raise ValueError(f"cohort_checks.csv: the accepted ranges are not the same in both sources or not as expected: "
                         f"{kn!r}, {hospital!r}")
    return (f"BMI in KNHANES was calculated from measured height and weight. The accepted ranges, as in the hospital "
            f"records, were height {m.group(1)} cm, weight {m.group(2)} kg and BMI {m.group(3)} {D.BMI_UNIT}.")


def figure_flow(store):
    """Figure 1: one flow of participants for both outcomes (the cohort and its split are joint, so it takes no
    outcome)."""
    fig_id, target = "flow", "main"
    flow, checks = store("cohort_flow.csv"), store("cohort_checks.csv")
    model = flow_model(flow, checks)
    both = f"{D.OUTCOME_LABELS['ALT']} and {D.OUTCOME_LABELS['AST']}"
    title = f"Flow of participants through the development and external validation data for {both}"
    legend = (
        "Left, development data from the hospital records. People of the development cohort were assigned at random, "
        "person by person, to a training set (80%) or a test set (20%) within the four groups formed by ALT and AST "
        "each above or not above 40 U/L. Right, external validation data from KNHANES, used only to evaluate the "
        "models. Both enzymes were required of every person, so the people of each final box (n) are the same for "
        f"{D.OUTCOME_LABELS['ALT']} and {D.OUTCOME_LABELS['AST']}, two separate outcomes; the numbers under n are the "
        "people with each outcome. Exclusions were applied in the order shown, so each person is counted once.")
    notes = ["People aged under 19 years and over 110 years are counted together.", ranges_note(checks)]
    if model["folded"]:
        notes.append("Where a step removed fewer than 10 people, it is shown with the next step and one count covers "
                     "both.")
    notes.append(f"The analysis populations are the training set, the test set and {D.DATASET_LONG['KN']}.")
    notes.append(D.abbreviation_note("ALT", "AST", "BMI", "KNHANES"))
    with F.main_style("flow"):
        fig = draw_flow(model)
        n_cells = flow_cells_check(model, store)
        label = publish(fig, fig_id, target, order_of(fig_id, target), title, legend, notes,
                        ["cohort_flow.csv", "cohort_checks.csv"])
    return label, model["notes"] + [f"{figure_ref(fig_id, target)}: {n_cells} counts checked against their cells"]


# ================================================================================================ ROC and calibration
POINT_ORDER = ("RULE_BMI25", "ENS@MATCH_BMI25")
ROC_DATASETS = ("DEV_TEST", "KN")      # the columns of Figures 2 and 3, left to right
POINT_DATASET = "KN"                   # the rule point stands on the ROC panel of KNHANES only
HATCH = "//////"
# a band of 1-9 people: hatched in the supplementary figures; in a main figure a plain fill of its own, far from the
# shown bands also in grey scale, since no bar has an outline there and hatching of 1 pt would fill such a narrow bar
MASKED_BAR = {"si": dict(facecolor="#FFFFFF", edgecolor=F.GREY, linewidth=0.3, hatch=HATCH),
              "main": dict(facecolor="#404040", edgecolor="none", linewidth=0)}
# what a legend says of the bands of 1-9 people (the key of the figure shows their bar)
MASKED_TEXT = "counts of 1-9 people are masked"


def point_style(strategy):
    """The marker of a strategy in a ROC panel, in the style in force: filled for the rule, open for the ensemble at the
    threshold set by the rule."""
    rule = F.RULE_MARKERS["BMI25"]
    if strategy.startswith("RULE"):
        return dict(marker=rule["marker"], markersize=F.line_pt(4.2, 5.5), markerfacecolor=rule["color"],
                    markeredgecolor=rule["color"], markeredgewidth=F.line_pt(0.6))
    return dict(marker=rule["marker"], markersize=F.line_pt(7.6, 9.0), markerfacecolor="none",
                markeredgecolor=F.MODEL_COLOURS["ENS"], markeredgewidth=F.line_pt(1.0, 1.5))


def curve_width(model):
    """The width of a model's ROC curve, in the style in force."""
    return F.line_pt(1.4, 1.6) if model == "ENS" else F.line_pt(1.0)


def roc_curve(store, outcome, dataset, model):
    table = "roc_points.csv" if model == "ENS" else "roc_points_lr.csv"
    rows = D.select(store(table), outcome=outcome, dataset=dataset, model=model)
    fpr, tpr = column(rows, "fpr"), column(rows, "tpr")
    if len(rows) < 2 or not (np.isfinite(fpr).all() and np.isfinite(tpr).all()):
        raise ValueError(f"{table}: no complete curve for {outcome} {dataset} {model}")
    o = np.argsort(fpr, kind="stable")
    return fpr[o], tpr[o]


def auroc_row(store, outcome, dataset, model):
    if model == "ENS":
        return D.one(store("performance.csv"), outcome=outcome, dataset=dataset, model="ENS")
    return D.one(store("model_comparison.csv"), row_type="model", outcome=outcome, dataset=dataset, model=model)


def draw_roc(ax, store, outcome, dataset, panel, missing):
    """The ROC panel of one dataset, with the rule points on the KNHANES panel; `panel` is its label
    (display.panel_label). A point without a value is added to `missing` as (panel, strategy)."""
    ax.plot([0, 1], [0, 1], color=F.GREY, linewidth=F.line_pt(0.6), linestyle=(0, (3, 2)), zorder=1)
    handles, labels = [], []
    for model in LR_DRAWN + ("ENS",):
        fpr, tpr = roc_curve(store, outcome, dataset, model)
        r = auroc_row(store, outcome, dataset, model)
        auroc = D.num(r["auroc"])
        if auroc is None:
            raise ValueError(f"{'performance.csv' if model == 'ENS' else 'model_comparison.csv'}: blank AUROC for "
                             f"{outcome} {dataset} {model}, whose ROC curve is drawn")
        check_roc_area(f"{outcome} {dataset} {model}", fpr, tpr, auroc)
        h, = ax.plot(fpr, tpr, color=F.MODEL_COLOURS[model], linestyle=F.MODEL_LINESTYLES[model],
                     linewidth=curve_width(model), zorder=3 if model == "ENS" else 2)
        handles.insert(0, h)
        labels.insert(0, D.fmt_ci(r["auroc"], r["auroc_lo"], r["auroc_hi"], 3))
    ca = store("comparators_alert.csv")
    for strategy in (POINT_ORDER if dataset == POINT_DATASET else ()):
        r = D.one(ca, outcome=outcome, dataset=dataset, strategy=strategy)
        x, y = fval(r["fpr"]), fval(r["tpr"])
        if math.isfinite(x) and math.isfinite(y):
            ax.plot([x], [y], linestyle="none", zorder=6 if strategy.startswith("ENS") else 5, clip_on=False,
                    **point_style(strategy))
        else:
            missing.append((panel, strategy))
    leg = ax.legend(handles, labels, title="AUROC (95% CI)", loc="lower right", frameon=False, handlelength=2.4,
                    borderaxespad=0.4, alignment="left")
    leg.set_zorder(7)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: D.fmt(v, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: D.fmt(v, 1)))
    ax.set_xlabel("False-positive rate (1 - specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_aspect("equal", adjustable="box")
    F.panel_title(ax, panel, f"{D.DATASET_LABELS[dataset]}: {OUT_WORD[outcome]}")


def calibration_top(store, outcomes, datasets):
    """Upper limit of the observed-proportion axis of the calibration panels: 0.5 when every decile's upper 95% limit is
    at most 0.45 for each outcome and dataset given (Figures 2 and 3 give the same limit from all of them)."""
    rows = D.select(store("calibration_bins.csv"), model="ENS")
    rows = rows[rows["outcome"].isin(list(outcomes)) & rows["dataset"].isin(list(datasets))]
    hi = column(rows, "observed_hi")
    hi = hi[np.isfinite(hi)]
    return 0.5 if len(hi) and float(hi.max()) <= 0.45 else 1.0


def on_grid(artists, decimals=CAL_STEP):
    """True when every y value drawn by the given lines and line collections lies on the grid of `decimals` decimals
    (within 1e-9)."""
    ys = []
    for a in artists:
        if hasattr(a, "get_segments"):
            ys += [np.asarray(s, float)[:, 1] for s in a.get_segments() if len(s)]
        else:
            ys.append(np.asarray(a.get_ydata(), float))
    ys = np.concatenate(ys) if ys else np.empty(0)
    return bool(np.all(np.abs(ys - np.round(ys, decimals)) < 1e-9))


def draw_calibration(ax, store, outcome, dataset, panel, top, missing):
    """Observed proportion against mean predicted risk by decile, with Wilson 95% limits. The proportions and limits
    are drawn rounded to CAL_STEP decimals (drawn exactly, they would give the n and events of each decile, and
    with the events of Table 2 those of a decile left out; rounded, they no longer give them exactly at the decile
    sizes of this study); the cells decide which deciles are drawn. `panel` is its label (display.panel_label)."""
    rows = D.select(store("calibration_bins.csv"), outcome=outcome, dataset=dataset, model="ENS")
    x, y = column(rows, "mean_pred"), column(rows, "observed")
    lo, hi = column(rows, "observed_lo"), column(rows, "observed_hi")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if (~ok).any():
        missing.append((panel, int((~ok).sum())))
    colour = ink(F.DATASET_COLOURS[dataset])
    yd, lod, hid = (np.round(v[ok], CAL_STEP) for v in (y, lo, hi))
    ax.plot([0, 1], [0, 1], color=F.GREY, linewidth=F.line_pt(0.7), linestyle=(0, (3, 2)), zorder=1)
    joined, = ax.plot(x[ok], yd, color=colour, linewidth=F.line_pt(0.8), zorder=2)
    yerr = np.vstack([np.clip(yd - lod, 0, None), np.clip(hid - yd, 0, None)])
    eb = ax.errorbar(x[ok], yd, yerr=yerr, fmt="o", markersize=F.line_pt(3.0, 4.5), color=colour,
                     markerfacecolor=colour, markeredgecolor=colour, ecolor=colour, elinewidth=F.line_pt(0.8),
                     capsize=F.line_pt(1.6, 2.5), capthick=F.line_pt(0.8), zorder=3)
    points, caps, bars = eb.lines
    # in the main style a marker has no edge, so that it leaves the narrow limits of a large decile to be seen
    points.set_markeredgewidth(F.line_pt(points.get_markeredgewidth(), 0.0))
    if not on_grid([joined, points, *caps, *bars]):
        raise AssertionError(f"calibration panel {outcome} {dataset}: drawn values not rounded to {CAL_STEP} "
                             "decimals")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02 * top, top)          # room below 0, so a decile at 0 is drawn whole
    step = 0.1 if top <= 0.5 else 0.2
    ax.yaxis.set_major_locator(FixedLocator(np.round(np.arange(0, top + 1e-9, step), 6)))
    ax.yaxis.set_major_formatter(pct_formatter())
    ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Observed proportion (%)")
    F.panel_title(ax, panel, f"{D.DATASET_LABELS[dataset]}: {OUT_WORD[outcome]}")
    return eb


def calibration_stats(ax, store, outcome, dataset):
    """The O:E and calibration slope of a calibration panel with their CIs, in a place clear of its marks. Call it when
    the figure holds everything else, its key included, since the key takes room from the panels."""
    c = D.one(store("calibration.csv"), outcome=outcome, dataset=dataset, model="ENS")
    stats = (f"O:E {D.fmt_ci(c['oe'], c['oe_lo'], c['oe_hi'], 2)}\n"
             f"Calibration slope {D.fmt_ci(c['slope'], c['slope_lo'], c['slope_hi'], 2)}")
    place_clear_text(ax, stats, ((0.98, 0.04, "right", "bottom"), (0.02, 0.97, "left", "top"),
                                 (0.98, 0.97, "right", "top")))


def place_clear_text(ax, text, candidates):
    """Put a text in the first candidate place (axes fraction x, y, ha, va) where it covers no plotted mark. The axes
    needs its title first: the messages of figstyle.text_collisions name an axes by it."""
    fig = ax.figure
    for x, y, ha, va in candidates:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=F.text_pt(7.0), linespacing=1.3,
                    zorder=6)
        hit = [p for p in F.text_collisions(fig) if f"{ax.get_title(loc='left')[:24]!r}" in p
               and repr(t.get_text()[:30]) in p and "covers" in p]
        if not hit:
            return t
        t.remove()
    raise AssertionError(f"no place clear of the marks for {text[:30]!r}")


def histogram_bands(store, outcome, dataset):
    """(lower limits, upper limits, counts) of prediction_histogram.csv; a count is an int or '<10'."""
    rows = D.select(store("prediction_histogram.csv"), outcome=outcome, dataset=dataset, model="ENS")
    counts = [D.count(v) for v in rows["n"]]
    if not len(rows) or any(c is None for c in counts):
        raise ValueError(f"prediction_histogram.csv: no rows or a blank n for {outcome} {dataset}")
    return column(rows, "bin_lo"), column(rows, "bin_hi"), counts


def histogram_panel(store, outcome, dataset):
    """(lower limits, upper limits, bar heights, band widths) of one histogram panel. The heights come from
    display.panel_heights: each shown count rounded to 2 significant figures (counts under 100 as they are) and None
    for a band of 1-9 people, after its check that no such band follows from the panel total, the ensemble n of
    performance.csv (shown in the performance tables and the flow figure); it raises AssertionError when one
    would. The widths are bin_hi - bin_lo rounded to 9 decimals (raw float differences are not compared)."""
    lo, hi, counts = histogram_bands(store, outcome, dataset)
    total = D.one(store("performance.csv"), outcome=outcome, dataset=dataset, model="ENS")["n"]
    heights = D.panel_heights(counts, total, where=f"prediction_histogram.csv {outcome} {dataset}")
    return lo, hi, heights, [float(w) for w in np.round(hi - lo, 9)]


def draw_histogram(axh, lo, hi, heights, dataset, peak, target="si"):
    """People per band of predicted risk on a log scale shared by the panels of a figure (`peak`: the largest drawn
    height of those panels). A shown band is a bar up to its rounded count (histogram_panel), without an outline in a
    main figure; a band of 1-9 people is a bar of MASKED_BAR up to 9 (its count is not known); an empty band has no
    bar. Returns whether a band of 1-9 people is drawn."""
    colour = F.DATASET_COLOURS[dataset]
    face = F.tint(colour, 0.35)
    shown = dict(edgecolor=ink(colour), linewidth=0.3) if target == "si" else dict(edgecolor="none", linewidth=0)
    masked = False
    for a, b, h in zip(lo, hi, heights):
        if h is None:
            axh.add_patch(Rectangle((a, 1), b - a, 8, zorder=2, **MASKED_BAR[target]))
            masked = True
        elif h > 0:
            axh.add_patch(Rectangle((a, 1), b - a, h - 1, facecolor=face, zorder=2, **shown))
    axh.set_yscale("log")
    top = max(peak, 10) * 1.8
    decades = int(math.floor(math.log10(top)))
    axh.set_ylim(1, top)
    step = 1 if decades <= 3 else 2
    axh.yaxis.set_major_locator(FixedLocator([10 ** k for k in range(0, decades + 1, step)]))
    axh.yaxis.set_minor_locator(NullLocator())
    axh.yaxis.set_major_formatter(axis_count_formatter())
    axh.set_xlim(0, 1)
    axh.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    axh.xaxis.set_major_formatter(pct_formatter())
    axh.set_xlabel("Predicted risk (%)")
    axh.set_ylabel("People")
    return masked


ROC_H = 196                  # height (mm) of Figures 2 and 3
ROC_RATIOS = [1.0, 0.70, 0.24]    # the shares of the ROC panel, the calibration panel and its histogram


def roc_calibration_layout():
    """(the figure, the (ROC, calibration, histogram) axes of each dataset of ROC_DATASETS): one column per dataset."""
    fig = F.new_figure("double", ROC_H)
    gs = fig.add_gridspec(3, len(ROC_DATASETS), height_ratios=ROC_RATIOS)
    axes = []
    for j in range(len(ROC_DATASETS)):
        ax_c = fig.add_subplot(gs[1, j])
        axes.append((fig.add_subplot(gs[0, j]), ax_c, fig.add_subplot(gs[2, j], sharex=ax_c)))
    return fig, axes


def roc_key(missing):
    """Handles and labels of a key: the three ROC curves, then the strategies drawn as points on the KNHANES panel. A
    strategy missing from that panel (`missing`, as draw_roc fills it) has no entry, since no mark answers to it; a
    note of the legend names it."""
    models = ("ENS", "LR_ALL", "LR_ASB")
    handles = [Line2D([], [], color=F.MODEL_COLOURS[m], linestyle=F.MODEL_LINESTYLES[m], linewidth=curve_width(m))
               for m in models]
    gone = {s for _panel, s in missing}
    points = [s for s in POINT_ORDER if s not in gone]
    return ([*handles, *[Line2D([], [], linestyle="none", **point_style(s)) for s in points]],
            [*[D.MODEL_LABELS[m] for m in models], *[D.STRATEGY_LABELS[s] for s in points]])


def calibration_key(marks):
    """Handles and labels of a key: the observed proportions (`marks`: the error bars of every calibration panel, drawn
    side by side in the key because each panel has its own colour) and the reference line."""
    return ([marks, Line2D([], [], color=F.GREY, linewidth=F.line_pt(0.7), linestyle=(0, (3, 2)))],
            ["Observed proportion by decile of predicted risk (95% CI)", "Perfect calibration"])


def figure_roc_calibration(store, outcome):
    """Figure 2 (ALT) or Figure 3 (AST): discrimination and calibration of the ensemble for one outcome in the two
    datasets, in one column each: the ROC panels (a, b) above the calibration panels (c, d) with their histograms. The
    two figures have the same layout and the same axes (the calibration axis and the histogram axis are set from all of
    their panels)."""
    fig_id, target = ROC_FIGURES[outcome], "main"
    missing_points, missing_bins, masked_any = [], [], False
    top = calibration_top(store, D.OUTCOMES, ROC_DATASETS)
    hist = {d: histogram_panel(store, outcome, d) for d in ROC_DATASETS}
    # the log axis is scaled to the largest drawn (rounded) height of Figures 2 and 3, so that its limit gives away no
    # exact count and the two figures share the axis
    peak = max([h for o in D.OUTCOMES for d in ROC_DATASETS for h in histogram_panel(store, o, d)[2] if h] + [10])
    width = D.single([w for d in ROC_DATASETS for w in hist[d][3]], "band width of prediction_histogram.csv")
    with F.main_style("panel"):
        fig, axes = roc_calibration_layout()
        cal_marks = []
        for j, (dataset, (ax_r, ax_c, ax_h)) in enumerate(zip(ROC_DATASETS, axes)):
            draw_roc(ax_r, store, outcome, dataset, D.panel_label(j), missing_points)
            ax_r.set_anchor("N")
            cal_marks.append(draw_calibration(ax_c, store, outcome, dataset, D.panel_label(2 + j), top, missing_bins))
            lo, hi, heights, _widths = hist[dataset]
            masked_any = draw_histogram(ax_h, lo, hi, heights, dataset, peak, target) or masked_any
        roc_handles, roc_labels = roc_key(missing_points)
        cal_handles, cal_labels = calibration_key(tuple(cal_marks))
        if masked_any:
            cal_handles.append(Patch(**MASKED_BAR[target]))
            cal_labels.append("Band of predicted risk with 1-9 people")
        # filled column by column: the curves, the observed proportions and the reference line, then the points
        fig.legend(roc_handles[:3] + cal_handles[:2] + roc_handles[3:] + cal_handles[2:],
                   roc_labels[:3] + cal_labels[:2] + roc_labels[3:] + cal_labels[2:],
                   loc="outside upper center", ncol=2, frameon=False, handlelength=2.6, columnspacing=2.0,
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        for dataset, (_ax_r, ax_c, _ax_h) in zip(ROC_DATASETS, axes):
            calibration_stats(ax_c, store, outcome, dataset)
        for _ax_r, ax_c, ax_h in axes:
            fig.align_ylabels([ax_c, ax_h])
        title, legend, notes = roc_calibration_text(store, outcome, missing_points, missing_bins, width, masked_any)
        sources = ["roc_points.csv", "roc_points_lr.csv", "performance.csv", "model_comparison.csv",
                   "comparators_alert.csv", "calibration_bins.csv", "calibration.csv", "prediction_histogram.csv"]
        return publish(fig, fig_id, target, order_of(fig_id, target), title, legend, notes, sources)


def roc_calibration_text(store, outcome, missing_points, missing_bins, width, masked_any):
    """Title, legend and notes of Figure 2 or 3 (panels as in figure_roc_calibration): the data set of each column of
    panels, then what each pair of panels shows."""
    label = D.OUTCOME_LABELS[outcome]
    long_kn = D.DATASET_LONG["KN"]
    roc_panels, cal_panels = [D.legend_panel(j) for j in (0, 1)], [D.legend_panel(j) for j in (2, 3)]
    # the two panels of each data set: its ROC panel above its calibration panel (one column per data set)
    column = {d: f"{roc_panels[j]}, {cal_panels[j]}" for j, d in enumerate(ROC_DATASETS)}
    point_panel = roc_panels[ROC_DATASETS.index(POINT_DATASET)]      # the panel with the rule points
    # one number of resamples for every CI of the figure: the calibration rows and the AUROC rows of the curves
    boot = [D.one(store(t), outcome=outcome, dataset=d, model="ENS")["n_boot"]
            for t in ("calibration.csv", "performance.csv") for d in ROC_DATASETS]
    boot += [D.one(store("model_comparison.csv"), row_type="model", outcome=outcome, dataset=d, model=m)["n_boot"]
             for d in ROC_DATASETS for m in LR_DRAWN]
    n_boot = D.fmt_count(D.single([D.count(v) for v in boot], f"n_boot of the rows drawn for {outcome}"))
    title = f"Discrimination and calibration of the ensemble for {label} in the development test set and in {long_kn}"
    # the people only, with the totals that display.panel_heights checks each histogram panel against (see the top of
    # this file for the events)
    sizes = {d: D.one(store("performance.csv"), outcome=outcome, dataset=d, model="ENS")["n"] for d in ROC_DATASETS}
    # the marks and lines are named by the key of the figure; the legend keeps the diagonal of the ROC panels, whose
    # line the key names only as perfect calibration. The way the thresholds were set is in the Methods and Table 3.
    # The risks drawn are the ensemble's own; no recalibration or other updating was done in KNHANES
    masked = f"; {MASKED_TEXT}" if masked_any else ""
    legend = (
        f"{column['DEV_TEST']} {D.DATASET_LABELS['DEV_TEST']} ({people_text(sizes['DEV_TEST'])}); {column['KN']} "
        f"{long_kn} ({people_text(sizes['KN'])}); {events_pointer((outcome,))}. {', '.join(roc_panels)} Receiver "
        "operating characteristic curves of the ensemble and two logistic regression models, with the AUROC (95% CI) "
        f"of each model; the diagonal shows no discrimination (AUROC 0.5); {point_panel} also marks the rule "
        f"({D.RULE_LONG['BMI25']}) and the ensemble at its threshold set by the rule "
        f"({table_ref(COMPARATOR_TABLE[outcome])}). {', '.join(cal_panels)} Calibration of the ensemble: observed "
        "proportion with the outcome against mean predicted risk by decile of predicted risk, with Wilson 95% CIs as "
        f"error bars; O:E and the calibration slope with percentile bootstrap 95% CIs ({n_boot} resamples); in KNHANES "
        "the predicted risks are used without recalibration. Below each calibration plot, the number of people in each "
        f"{D.fmt(width * 100, 0)}% band of predicted risk (log scale){masked}. Models as in "
        f"{performance_ref((outcome,))}.")
    notes = []
    if missing_points:
        what = "; ".join(f"{lower_first(D.STRATEGY_LABELS[s])} {D.legend_panel(panel)}" for panel, s in missing_points)
        notes.append("Not shown because a cell of its two-by-two table of alerts and outcomes held 1-9 people: "
                     f"{what}.")
    if missing_bins:
        what = "; ".join(f"{n} decile{'s' if n > 1 else ''} in {D.legend_panel(panel)}" for panel, n in missing_bins)
        notes.append(f"Deciles with fewer than 10 people with or without the outcome are not shown: {what}.")
    notes.append(D.abbreviation_note(outcome, "AUROC", "BMI", "CI", "O:E", "KNHANES"))
    return title, legend, notes


# ================================================================================================ Figure 4: subgroups
def forest_rows(store, outcome, dataset="KN"):
    """The rows of one panel of the subgroup figure, in the order of the table: 'all', then for each variable of
    SUBGROUP_VARS a header row and its levels. `drawn` is False for the level of unknown alcohol frequency, which the
    figure does not show (its counts are in the supplementary table); forest_counts still takes it into account, so
    that an undrawn count of 1-9 cannot be worked out from the total and the levels drawn."""
    rows = D.select(store("subgroups.csv"), outcome=outcome, dataset=dataset, model="ENS").to_dict("records")
    alls = [r for r in rows if r["subgroup_var"] == "all"]
    if len(alls) != 1:
        raise ValueError(f"subgroups.csv: {len(alls)} 'all' rows for {outcome} {dataset}")
    out = [dict(alls[0], header=False, drawn=True)]
    for var in SUBGROUP_VARS:
        group = [r for r in rows if r["subgroup_var"] == var]
        if not group:
            raise ValueError(f"subgroups.csv: no {var} rows for {outcome} {dataset}")
        out.append(dict(subgroup_var=var, subgroup=None, header=True, drawn=True))
        out += [dict(r, header=False, drawn=not (var == "drink_group" and r["subgroup"] == "unknown")) for r in group]
    return out


def forest_counts(rows):
    """Texts of the n and events columns, keyed by (subgroup_var, subgroup, column). A count is shown when it is 10
    or more, the count of the 'all' row in its column is shown, and so is every count of its column and subgroup
    variable (including a level that is not drawn): a hidden count would follow from the others and the total. Hidden
    counts are ''."""
    out = {}
    data = [r for r in rows if not r["header"]]
    total = next(r for r in data if r["subgroup_var"] == "all")
    for col in ("n", "events"):
        total_ok = showable(total[col])
        out[("all", total["subgroup"], col)] = D.fmt_count(total[col]) if total_ok else ""
        for var in SUBGROUP_VARS:
            group = [r for r in data if r["subgroup_var"] == var]
            ok = total_ok and all(showable(r[col]) for r in group)
            for r in group:
                out[(var, r["subgroup"], col)] = D.fmt_count(r[col]) if ok else ""
    return out


def forest_hidden_problems(rows, texts):
    """Counts of a figure that reveal a hidden one: a column of a subgroup variable with some counts shown and some
    hidden, or any shown count below 10."""
    out = []
    data = [r for r in rows if not r["header"]]
    for col in ("n", "events"):
        for var in SUBGROUP_VARS:
            shown = [texts[(var, r["subgroup"], col)] != "" for r in data if r["subgroup_var"] == var]
            if any(shown) and not all(shown):
                out.append(f"{var} {col}: some counts shown, some hidden")
        for r in data:
            t = texts[(r["subgroup_var"], r["subgroup"], col)]
            if t and (t == D.MASKED or int(t.replace(",", "")) < 10):
                out.append(f"{r['subgroup']} {col}: {t}")
    return out


def forest_axis(blocks):
    """Lower limit and ticks of the AUROC axis of a subgroup figure: 0.5, or lower to hold every lower 95% limit (to the
    0.1 below it); ticks every 0.1, or every 0.2 when that would give more than six."""
    los = [fval(r["auroc_lo"]) for rows in blocks.values() for r in rows if not r["header"]]
    los = [v for v in los if math.isfinite(v)]
    x_lo = min([0.5] + [math.floor(v * 10 + 1e-9) / 10 for v in los])
    step = 0.1 if round((1.0 - x_lo) / 0.1) + 1 <= 6 else 0.2
    ticks = [round(1.0 - k * step, 6) for k in range(int(round((1.0 - x_lo) / step)) + 1)][::-1]
    return x_lo, ticks


def forest_panel(fig, ax_t, panel, rows, texts, outcome, x_lo, x_ticks, layout, W, H):
    """One forest panel of the subgroup figure, with `rows` in order; `layout` holds its positions (mm, measured up from
    the bottom of the figure). Returns whether a group has no estimate."""
    y_top, y_bottom = layout["y_top"], layout["y_bottom"]
    label_x, n_right, ev_right = layout["label_x"], layout["n_right"], layout["ev_right"]
    f0, f1, au_x = layout["f0"], layout["f1"], layout["au_x"]
    kw = dict(fontsize=F.MAIN_TEXT_PT, va="center")
    head = fig.add_axes([label_x / W, layout["y_head"] / H, (W - label_x - 0.5) / W, 0.5 / H])
    head.set_axis_off()
    head.set_xticks([])
    head.set_yticks([])
    F.panel_title(head, panel, OUTCOME_TEXT[(outcome, "KN")])
    for i, r in enumerate(rows):
        y = y_top - (i + 0.5) * layout["row"]
        if r["header"]:
            ax_t.text(label_x, y, D.SUBGROUP_VAR_LABELS[r["subgroup_var"]], ha="left", fontweight="bold", **kw)
        elif r["subgroup_var"] == "all":
            ax_t.text(label_x, y, D.subgroup_label("all", r["subgroup"]), ha="left", fontweight="bold", **kw)
        else:
            ax_t.text(label_x + 3.5, y, D.subgroup_label(r["subgroup_var"], r["subgroup"]), ha="left", **kw)
    yh = y_top + layout["head_h"] / 2
    ax_t.text(n_right, yh, "n", ha="right", **kw)
    ax_t.text(ev_right, yh, "Events", ha="right", **kw)
    ax_t.text(au_x, yh, "AUROC (95% CI)", ha="left", **kw)
    axf = fig.add_axes([f0 / W, y_bottom / H, (f1 - f0) / W, (y_top - y_bottom) / H])
    axf.set_xlim(x_lo, 1.0)
    axf.set_ylim(y_bottom, y_top)
    for side in ("left", "right", "top"):
        axf.spines[side].set_visible(False)
    axf.yaxis.set_major_locator(NullLocator())
    axf.xaxis.set_major_locator(FixedLocator(x_ticks))
    axf.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: D.fmt(v, 1)))
    axf.set_xlabel("AUROC")
    colour = ink(F.DATASET_COLOURS["KN"])
    total = rows[0]
    if math.isfinite(fval(total["auroc"])):
        axf.plot([fval(total["auroc"])] * 2, [y_bottom, y_top], color=F.GREY, linewidth=F.MAIN_LINE_PT,
                 linestyle=(0, (1, 1.5)), zorder=1)
    not_estimated = False
    for i, r in enumerate(rows):
        if r["header"]:
            continue
        y = y_top - (i + 0.5) * layout["row"]
        key = (r["subgroup_var"], r["subgroup"])
        ax_t.text(n_right, y, texts[key + ("n",)], ha="right", **kw)
        ax_t.text(ev_right, y, texts[key + ("events",)], ha="right", **kw)
        est, lo, hi = fval(r["auroc"]), fval(r["auroc_lo"]), fval(r["auroc_hi"])
        if math.isfinite(est):
            ax_t.text(au_x, y, D.fmt_ci(r["auroc"], r["auroc_lo"], r["auroc_hi"], 3), ha="left", **kw)
            if math.isfinite(lo) and math.isfinite(hi):
                axf.plot([max(lo, x_lo), min(hi, 1.0)], [y, y], color=colour, linewidth=2.0, zorder=2,
                         solid_capstyle="butt")
            if x_lo <= est <= 1.0:
                # no marker edge, so that a marker leaves a narrow CI to be seen
                axf.plot([est], [y], linestyle="none", zorder=3, color=colour, markeredgewidth=0,
                         marker="D" if r["subgroup_var"] == "all" else "s",
                         markersize=5.0 if r["subgroup_var"] == "all" else 4.5)
        else:
            ax_t.text(au_x, y, "Not estimated", ha="left", **kw)
            not_estimated = True
    return not_estimated


def figure_subgroups(store):
    """Figure 4: two forest panels one above the other (a ALT, b AST) on one AUROC axis (forest_axis over both panels),
    the AUROC of the ensemble by subgroup in KNHANES, with the n and events of each group."""
    fig_id, target = "subgroups", "main"
    W = F.MAIN_CANVAS["panel"][0]
    all_rows = {o: forest_rows(store, o) for o in D.OUTCOMES}
    drawn = {o: [r for r in rows if r["drawn"]] for o, rows in all_rows.items()}
    a, b = ([(r["subgroup_var"], r["subgroup"]) for r in drawn[o]] for o in D.OUTCOMES)
    if a != b:
        raise AssertionError(f"{figure_ref(fig_id, target)}: the two panels do not show the same groups in one order")
    texts = {o: forest_counts(all_rows[o]) for o in D.OUTCOMES}
    for o in D.OUTCOMES:
        problems = forest_hidden_problems(all_rows[o], texts[o])
        if problems:
            raise AssertionError(f"{figure_ref(fig_id, target)} {o} counts: " + "; ".join(problems))
    hidden = any(t == "" for o in D.OUTCOMES for t in texts[o].values())
    x_lo, x_ticks = forest_axis(drawn)
    nrow = len(a)
    ROW, TITLE_H, HEAD_H, AXIS_H, GAP = 4.2, 5.5, 4.6, 11.5, 3.0
    PANEL = TITLE_H + HEAD_H + nrow * ROW + AXIS_H
    H = round(2 * PANEL + GAP, 1)
    label_x, label_w = 3.5, 33.0
    bx = label_x + label_w
    columns = dict(label_x=label_x, n_right=bx + 11.0, ev_right=bx + 23.0, f0=bx + 27.0, f1=W - 33.0, au_x=W - 31.0)
    not_estimated = False
    with F.main_style("panel"):
        fig = F.new_figure(W, H, layout=None)
        ax_t = drawing_axes(fig, W, H)
        for j, o in enumerate(D.OUTCOMES):
            y_panel = H - j * (PANEL + GAP)                       # top of the panel
            y_top = y_panel - TITLE_H - HEAD_H
            layout = dict(columns, row=ROW, head_h=HEAD_H, y_top=y_top, y_bottom=y_top - nrow * ROW,
                          y_head=y_panel - TITLE_H)
            none = forest_panel(fig, ax_t, j, drawn[o], texts[o], o, x_lo, x_ticks, layout, W, H)
            not_estimated = not_estimated or none
        # one number of resamples for every group drawn (header rows have none; counts-only groups a blank cell)
        n_boot = D.fmt_count(D.single([D.count(r["n_boot"]) for o in D.OUTCOMES for r in drawn[o] if not r["header"]],
                                      "n_boot of subgroups.csv"))
        long_kn = D.DATASET_LONG["KN"]
        title = (f"Discrimination of the ensemble for {D.OUTCOME_LABELS['ALT']} and {D.OUTCOME_LABELS['AST']} by age "
                 f"group, sex, survey period and alcohol drinking frequency in {long_kn}")
        overall = D.subgroup_label("all", "All")
        subgroup_table = table_ref("subgroups")
        legend = (
            f"{D.legend_panel(0)} {D.OUTCOME_LABELS['ALT']}; {D.legend_panel(1)} {D.OUTCOME_LABELS['AST']}. The two "
            "panels show the same groups in the same order on one AUROC axis. Values are the AUROC (95% CI); error "
            f"bars are the 95% CIs. Row '{overall}' gives the whole sample; n is the "
            f"number of people in the group and events the number with the outcome. 95% CIs are from {n_boot} "
            "bootstrap resamples drawn within each group (percentile method). KNHANES records ages of 80 years or more "
            "as 80. People with an unknown frequency of alcohol drinking are not drawn; their number is in "
            f"{subgroup_table}. "
            f"The O:E ratio of each group is given in {subgroup_table}. Models as in {performance_ref(D.OUTCOMES)}.")
        notes = []
        if hidden:
            notes.append("Counts below 10 are not shown, and neither are the other counts of the same column and "
                         "subgroup variable, from which they could be worked out.")
        if not_estimated:
            notes.append("Not estimated: fewer than 10 people with or without the outcome in the group.")
        notes.append(D.abbreviation_note("ALT", "AST", "AUROC", "CI", "KNHANES", "O:E"))
        return publish(fig, fig_id, target, order_of(fig_id, target), title, legend, notes, ["subgroups.csv"])


# ================================================================================================ decision curves
DCA_DATASETS = ("KN", "DEV_TEST")
DCA_H = 156                  # height (mm) of the figure
# Two rows of panels per population, so that no panel holds more than three bands at 183 mm: the models (the curves of
# the ensemble and of the logistic regression with all 12 inputs, alert at p >= pt, with their bands, and the three
# fixed ensemble alerts as lines), then the ensemble with the logistic regression with age, sex and BMI and the rule
# (bands); alerting everyone and alerting no one in both. Columns: the outcomes in display.OUTCOMES order (ALT, AST).
DCA_ROWS = (("nb_ens", "nb_lr_all", "nb_ens_youden", "nb_ens_sens80", "nb_ens_sens90", "nb_all", "nb_none"),
            ("nb_ens", "nb_lr_asb", "nb_rule_bmi25", "nb_all", "nb_none"))
DCA_LINES = {
    "nb_ens": dict(color=F.MODEL_COLOURS["ENS"], linestyle="-", linewidth=1.3, zorder=6),
    "nb_lr_all": dict(color=F.MODEL_COLOURS["LR_ALL"], linestyle="--", linewidth=1.0, zorder=5),
    "nb_lr_asb": dict(color=F.MODEL_COLOURS["LR_ASB"], linestyle="-.", linewidth=1.0, zorder=5),
    "nb_ens_youden": dict(color=ink(F.OKABE_ITO["orange"]), linestyle=(0, (6, 1.6)), linewidth=1.0, zorder=4),
    "nb_ens_sens80": dict(color=ink(F.OKABE_ITO["reddish_purple"]), linestyle=(0, (4, 1.4, 1, 1.4)), linewidth=1.0,
                          zorder=4),
    "nb_ens_sens90": dict(color=ink(F.OKABE_ITO["sky_blue"]), linestyle=(0, (1, 1.1)), linewidth=1.3, zorder=4),
    "nb_rule_bmi25": dict(color=F.RULE_MARKERS["BMI25"]["color"], linestyle=(0, (1, 1.2)), linewidth=1.2, zorder=4),
    "nb_all": dict(color=F.GREY, linestyle="-", linewidth=0.8, zorder=3),
    "nb_none": dict(color=F.BLACK, linestyle="-", linewidth=0.6, zorder=3),
}
# pointwise 95% bands, opaque (tints or hatching): the narrow band of the rule lies over the ensemble's
DCA_BANDS = {
    "nb_ens": dict(facecolor=F.tint(F.MODEL_COLOURS["ENS"], 0.72), edgecolor="none", linewidth=0, zorder=1),
    "nb_lr_all": dict(facecolor="none", edgecolor=F.tint(F.MODEL_COLOURS["LR_ALL"], 0.35), linewidth=0, hatch="////",
                      zorder=2),
    "nb_lr_asb": dict(facecolor="none", edgecolor=F.tint(F.MODEL_COLOURS["LR_ASB"], 0.35), linewidth=0, hatch=r"\\\\",
                      zorder=2),
    "nb_rule_bmi25": dict(facecolor=F.tint(F.RULE_MARKERS["BMI25"]["color"], 0.8), edgecolor="none", linewidth=0,
                          zorder=2),
}
DCA_LABELS = {
    "nb_ens": "Ensemble (95% CI)", "nb_lr_all": "Logistic regression: all 12 inputs (95% CI)",
    "nb_lr_asb": "Logistic regression: age, sex and BMI (95% CI)",
    "nb_ens_youden": f"Ensemble, {D.THRESHOLD_LABELS['YOUDEN']}",
    "nb_ens_sens80": f"Ensemble, {lower_first(D.THRESHOLD_LABELS['SENS80'])}",
    "nb_ens_sens90": f"Ensemble, {lower_first(D.THRESHOLD_LABELS['SENS90'])}",
    "nb_rule_bmi25": f"{D.STRATEGY_LABELS['RULE_BMI25']} (95% CI)",
    "nb_all": "Alert everyone", "nb_none": "Alert no one",
}
# legend entries in the order of its three columns (a legend is filled column by column)
DCA_LEGEND = ("nb_ens", "nb_lr_all", "nb_lr_asb", "nb_ens_youden", "nb_ens_sens80", "nb_ens_sens90", "nb_rule_bmi25",
              "nb_all", "nb_none")
# the strategies in a sentence (notes)
DCA_NAMES = {
    "nb_ens": "the ensemble", "nb_lr_all": "logistic regression with all 12 inputs",
    "nb_lr_asb": "logistic regression with age, sex and BMI",
    "nb_ens_youden": f"the ensemble at the {D.THRESHOLD_LABELS['YOUDEN']}",
    "nb_ens_sens80": f"the ensemble at the {lower_first(D.THRESHOLD_LABELS['SENS80'])}",
    "nb_ens_sens90": f"the ensemble at the {lower_first(D.THRESHOLD_LABELS['SENS90'])}",
    "nb_rule_bmi25": f"the {D.RULE_LABELS['BMI25']} rule",
    "nb_all": "alerting everyone", "nb_none": "alerting no one",
}
# the alerts that flag the same people at every threshold probability
FIXED_ALERTS = ("nb_rule_bmi25", "nb_ens_youden", "nb_ens_sens80", "nb_ens_sens90")
DCA_MODELS = ("nb_ens", "nb_lr_all", "nb_lr_asb")       # curves that follow the threshold probability (alert p >= pt)
DOT_ZORDER = max(s["zorder"] for s in DCA_LINES.values()) + 1      # lone values, above every line of a panel


def join_and(words):
    words = [w for w in words if w]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def decision_limits(rows):
    """Net-benefit axis of the panels of one population and outcome: up to 1.12 x the prevalence, or to 1.02 x the
    largest value or upper 95% limit drawn when that is higher (no curve or band leaves a panel at its upper edge);
    down to the lowest value of the models with their CIs (and at least 0.12 x the prevalence below 0), but not below
    -2 x the prevalence. Returns (lower, upper, whether a model curve or band leaves the panel at the bottom)."""
    prev = fval(rows["prevalence"].iloc[0])
    lows = np.concatenate([column(rows, c) for m in DCA_MODELS for c in (m, m + "_lo")])
    lows = lows[np.isfinite(lows)]
    low = float(lows.min()) if len(lows) else 0.0
    highs = np.concatenate([column(rows, c) for c in DCA_LINES] + [column(rows, c + "_hi") for c in DCA_BANDS])
    highs = highs[np.isfinite(highs)]
    high = float(highs.max()) if len(highs) else 0.0
    ymin = max(min(-0.12 * prev, 1.1 * low), -2 * prev)
    return ymin, max(1.12 * prev, 1.02 * high), low < ymin


def lone_points(v):
    """Values with a withheld (blank) value or the end of the grid on both sides, which a line cannot show."""
    ok = np.isfinite(v)
    return ok & ~np.r_[False, ok[:-1]] & ~np.r_[ok[1:], False]


def partly_withheld(rows):
    """The columns of a fixed alert (net benefit, _lo, _hi) withheld at some threshold probabilities only. s05
    withholds a fixed alert with its limits at every threshold probability or at none, since two of its net
    benefits would give its true and false positives."""
    cols = [c for a in FIXED_ALERTS for c in (a, a + "_lo", a + "_hi")]
    return [c for c in cols if 0 < int(np.isfinite(column(rows, c)).sum()) < len(rows)]


def decision_panel(ax, rows, pt, cols, ymin):
    """One decision-curve panel: the bands of the strategies `cols`, then their lines, a value between two withheld
    ones as a dot above every line (blank values are withheld, never zero; a fixed alert is blank at
    every threshold probability or at none, which partly_withheld checks first). Returns ({col: 'absent', 'gaps' or
    'whole'}, whether a dot was drawn, whether a model curve or band leaves the panel at its lower edge)."""
    status, dotted, cut = {}, False, False
    for col in cols:
        if col in DCA_BANDS:
            lo, hi = column(rows, col + "_lo"), column(rows, col + "_hi")
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                ax.fill_between(pt, np.where(ok, lo, 0), np.where(ok, hi, 0), where=ok, interpolate=False,
                                **DCA_BANDS[col])
    for col in cols:
        v = column(rows, col)
        ok = np.isfinite(v)
        if not ok.any():
            status[col] = "absent"
            continue
        status[col] = "whole" if ok.all() else "gaps"
        style = DCA_LINES[col]
        ax.plot(pt, v, **style)
        lone = lone_points(v)
        if lone.any():
            ax.plot(pt[lone], v[lone], linestyle="none", marker="o", markersize=3.8, markerfacecolor=style["color"],
                    markeredgecolor="#FFFFFF", markeredgewidth=0.5, zorder=DOT_ZORDER)
            dotted = True
        if col in DCA_MODELS:
            low = np.concatenate([v[ok], column(rows, col + "_lo")])
            cut = cut or bool(np.nanmin(low) < ymin)
    return status, dotted, cut


def figure_decision_curves(store):
    dc = store("decision_curve.csv")
    blocks = {}                  # the rows of each population and outcome, checked before anything is drawn
    for ds in DCA_DATASETS:
        for o in D.OUTCOMES:
            rows = D.select(dc, outcome=o, dataset=ds)
            if not len(rows):
                raise ValueError(f"decision_curve.csv: no rows for {o} {ds}")
            partly = partly_withheld(rows)
            if partly:
                raise ValueError(f"decision_curve.csv: {', '.join(partly)} of {o} {ds} withheld at some threshold "
                                 "probabilities only (a fixed alert is withheld at every one or at none)")
            blocks[(ds, o)] = rows
    n_out, n_kind = len(D.OUTCOMES), len(DCA_ROWS)
    nrows = len(DCA_DATASETS) * n_kind
    fig, axes = F.subplots(nrows, n_out, width="double", height_mm=DCA_H)
    drawn, absent, gaps, dots, cut_models = set(), {}, set(), set(), []
    for i, ds in enumerate(DCA_DATASETS):
        for j, o in enumerate(D.OUTCOMES):
            rows = blocks[(ds, o)]
            pt = column(rows, "threshold_prob")
            ymin, ymax, _cut = decision_limits(rows)
            for k, cols in enumerate(DCA_ROWS):
                r = n_kind * i + k
                ax, panel = axes[r, j], n_out * r + j
                letter = D.legend_panel(panel)
                status, dotted, cut = decision_panel(ax, rows, pt, cols, ymin)
                for col, s in status.items():
                    if s == "absent":
                        absent.setdefault(col, []).append(letter)
                        continue
                    drawn.add(col)
                    if s == "gaps":
                        gaps.add(letter)
                if dotted:
                    dots.add(letter)
                if cut:
                    cut_models.append(letter)
                ax.set_xlim(0, float(np.nanmax(pt)))
                ax.set_ylim(ymin, ymax)
                ax.xaxis.set_major_locator(FixedLocator(np.round(np.arange(0, float(np.nanmax(pt)) + 1e-9, 0.1),
                                                                 6)))
                ax.xaxis.set_major_formatter(pct_formatter())
                if k == n_kind - 1:
                    ax.set_xlabel("Threshold probability (%)")
                ax.set_ylabel("Net benefit")
                if k == 0:
                    F.panel_title(ax, panel, f"{D.DATASET_LABELS[ds]}: {OUT_WORD[o]}")
                else:
                    first = D.panel_label(n_out * (n_kind * i) + j)
                    F.panel_title(ax, panel, f"As {first}: ensemble, logistic regression (age, sex and BMI) and rule")
    handles, labels = [], []
    for col in DCA_LEGEND:
        if col not in drawn:
            continue
        line = Line2D([], [], **{k: v for k, v in DCA_LINES[col].items() if k != "zorder"})
        if col in DCA_BANDS:
            handles.append((Patch(**{k: v for k, v in DCA_BANDS[col].items() if k != "zorder"}), line))
        else:
            handles.append(line)
        labels.append(DCA_LABELS[col])
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, handlelength=2.8,
               columnspacing=1.6, handler_map={tuple: HandlerTuple(ndivide=1)})
    n_boot = D.fmt_count(D.single([D.count(v) for v in dc["n_boot"]], "n_boot of decision_curve.csv"))
    per = n_kind * n_out
    span = {ds: f"{D.legend_panel(per * i)}-{D.legend_panel(per * i + per - 1)}" for i, ds in enumerate(DCA_DATASETS)}
    kind = [join_and([D.legend_panel(n_out * r + j) for r in range(nrows) if r % n_kind == k for j in range(n_out)])
            for k in range(n_kind)]
    cal = "Figs. " + join_and([D.panel_ref(t, order_of(f, t), p).split(" ", 1)[1] for f, t, p in (
        ("roc_calibration_alt", "main", "c"), ("roc_calibration_alt", "main", "d"),
        ("roc_calibration_ast", "main", "c"), ("roc_calibration_ast", "main", "d"))])
    sides = ("left", "right")       # the columns, one per outcome in display.OUTCOMES order
    columns = join_and([f"{THRESHOLD[o]} in the {side} column" for o, side in zip(D.OUTCOMES, sides)])
    title = ("Decision curves of the ensemble, two logistic regression models, three fixed ensemble alerts and the "
             f"{D.RULE_LABELS['BMI25']} rule")
    # the people of each population: the same for both outcomes, read from the rows of each block
    sizes = {ds: D.single([D.count(v) for o in D.OUTCOMES for v in blocks[(ds, o)]["n"]],
                          f"n of decision_curve.csv, {ds}") for ds in DCA_DATASETS}
    legend = (
        f"Net benefit against threshold probability in {D.DATASET_LONG['KN']} ({people_text(sizes['KN'])}), "
        f"{span['KN']}, and the development test set ({people_text(sizes['DEV_TEST'])}), {span['DEV_TEST']}, for "
        f"{columns}; {events_pointer(D.OUTCOMES)}. Panels {kind[0]}: {DCA_NAMES['nb_ens']}, "
        f"{DCA_NAMES['nb_lr_all']} and three fixed ensemble alerts; panels {kind[1]}: {DCA_NAMES['nb_ens']}, "
        f"{DCA_NAMES['nb_lr_asb']} and {DCA_NAMES['nb_rule_bmi25']}. On a model's curve, a person is flagged when the "
        "predicted probability, used as it stands (without recalibration in KNHANES), is at or above the threshold "
        f"probability pt; net benefit = true positives/n - false positives/n {TIMES} pt/(1 - pt), with n the people of "
        "the panel. The fixed alerts (ensemble thresholds set in the development test set for the Youden index and for "
        "80% or 90% sensitivity, so their results there are derivation results) and the rule flag the same people at "
        "every pt; a fixed-alert line above the ensemble curve reflects calibration between pt and the fixed threshold "
        f"(calibration panels of {cal}), not better discrimination. Bands are pointwise percentile bootstrap 95% CIs "
        f"({n_boot} resamples), not limits of a difference between strategies: overlapping bands do not show similar "
        "net benefit, and net benefit was not compared formally. Breaks in a curve and missing strategies are values "
        f"withheld because a count of 1-9 people entered them, not zeros. Models as in {performance_ref(D.OUTCOMES)}.")
    notes = []
    if cut_models:
        notes.append(f"In {join_and(cut_models)}, the lower edge also cuts the curve or band of a model.")
    if gaps:
        notes.append(f"Curves with withheld values: {join_and(sorted(gaps))}.")
    if dots:
        notes.append(f"A dot marks a value with withheld values on both sides: {join_and(sorted(dots))}.")
    if absent:
        notes.append("Withheld at every threshold probability: " + "; ".join(
            f"{DCA_NAMES[col]}, {join_and(panels)}" for col, panels in absent.items()) + ".")
    notes.append(D.abbreviation_note("ALT", "AST", "BMI", "CI", "KNHANES"))
    return publish(fig, "decision_curves", "si", order_of("decision_curves", "si"), title, legend, notes,
                   ["decision_curve.csv"])


# ================================================================================================ predicted risk
PD_DATASETS = ("DEV_TEST", "KN")       # within each band of predicted risk, the left and the right bar
PD_H = 72                              # height (mm) of the figure


def figure_prediction_distributions(store):
    """Supplementary Figure: people per band of predicted risk of the ensemble in the two populations, one panel per
    outcome (a ALT, b AST). Each pair of half-width bars fills one band; the heights come from histogram_panel (2
    significant figures, the check of display.panel_heights) and are drawn on one log axis for the two panels."""
    hist = {(o, d): histogram_panel(store, o, d) for o in D.OUTCOMES for d in PD_DATASETS}
    peak = max([h for v in hist.values() for h in v[2] if h] + [10])
    width = D.single([w for v in hist.values() for w in v[3]], "band width of prediction_histogram.csv")
    fig, axes = F.subplots(1, len(D.OUTCOMES), width="double", height_mm=PD_H)
    masked_any = False
    for j, o in enumerate(D.OUTCOMES):
        ax = axes[j]
        for k, d in enumerate(PD_DATASETS):
            lo, hi, heights, _w = hist[(o, d)]
            half = (hi - lo) / 2.0
            masked_any = draw_histogram(ax, lo + k * half, lo + (k + 1) * half, heights, d, peak, "si") or masked_any
        F.panel_title(ax, j, D.OUTCOME_LABELS[o])
    handles = [Patch(facecolor=F.tint(F.DATASET_COLOURS[d], 0.35), edgecolor=ink(F.DATASET_COLOURS[d]), linewidth=0.3)
               for d in PD_DATASETS]
    labels = [D.DATASET_LABELS[d] for d in PD_DATASETS]
    if masked_any:
        handles.append(Patch(**MASKED_BAR["si"]))
        labels.append("Band of predicted risk with 1-9 people")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    sizes = {d: D.single([D.count(D.one(store("performance.csv"), outcome=o, dataset=d, model="ENS")["n"])
                          for o in D.OUTCOMES], f"n of performance.csv, {d}") for d in PD_DATASETS}
    long_kn = D.DATASET_LONG["KN"]
    title = (f"Distribution of the predicted risk of the ensemble for {D.OUTCOME_LABELS['ALT']} and "
             f"{D.OUTCOME_LABELS['AST']} in the development test set and in {long_kn}")
    legend = (
        f"Number of people in each {D.fmt(width * 100, 0)}% band of predicted risk of the ensemble (log scale; bar "
        f"heights rounded to 2 significant figures) for {D.OUTCOME_LABELS['ALT']}, {D.legend_panel(0)}, and "
        f"{D.OUTCOME_LABELS['AST']}, {D.legend_panel(1)}. {D.DATASET_LABELS['DEV_TEST']} "
        f"({people_text(sizes['DEV_TEST'])}); {long_kn} ({people_text(sizes['KN'])}). In KNHANES, the ensemble's "
        "predicted risks are used as they stand, without recalibration.")
    notes = [f"{MASKED_TEXT.capitalize()}."] if masked_any else []
    notes.append(D.abbreviation_note("ALT", "AST", "KNHANES"))
    return publish(fig, "prediction_distributions", "si", order_of("prediction_distributions", "si"), title, legend,
                   notes, ["prediction_histogram.csv", "performance.csv"])


# ================================================================================================ SHAP
LEARNERS = (("mean_abs_shap_lgb", "LightGBM", "o"), ("mean_abs_shap_xgb", "XGBoost", "^"),
            ("mean_abs_shap_cat", "CatBoost", "s"))
SHAP_PANELS = (("ua_glucose", "ua_protein", "ua_ketone", "ua_bilirubin", "ua_blood", "ua_urobilinogen"),
               ("ua_nitrite", "ua_pH", "ua_sg", "age", "male", "bmi"))
# nitrite, pH and specific gravity are dipstick fields too: panel a holds the graded ones
SHAP_PANEL_TITLES = ("Graded dipstick fields", "Other dipstick fields, age, sex and BMI")
assert sorted(sum(SHAP_PANELS, ())) == sorted(D.INPUTS), "SHAP_PANELS must hold every input once"
# each outcome's symbol (the key of the figure names it); within a level, the outcomes are drawn top down in
# display.OUTCOMES order (SHAP_OFFSETS)
SHAP_STYLE = {"ALT": dict(marker="o", color=F.OKABE_ITO["blue"]),
              "AST": dict(marker="^", color=F.OKABE_ITO["vermillion"])}
SHAP_OFFSETS = (0.17, -0.17)
INPUT_HEAD = {**D.INPUT_LABELS, "age": "Age, years", "male": "Sex", "bmi": f"BMI, {D.BMI_UNIT}"}


def figure_shap_summary(store):
    ss = store("shap_summary.csv")
    fig, axes = F.subplots(1, 2, width="double", height_mm=90)
    for j, o in enumerate(D.OUTCOMES):
        ax = axes[j]
        rows = D.select(ss, outcome=o).to_dict("records")
        if sorted(r["input"] for r in rows) != sorted(D.INPUTS):
            raise ValueError(f"shap_summary.csv: the inputs of {o} differ from the 12 inputs")
        rows.sort(key=lambda r: (int(r["rank"]), D.INPUTS.index(r["input"])))
        y = np.arange(len(rows))[::-1].astype(float)
        mean = np.array([fval(r["mean_abs_shap"]) for r in rows])
        ax.barh(y, mean, height=0.62, color=F.tint(F.MODEL_COLOURS["ENS"], 0.5),
                edgecolor=F.MODEL_COLOURS["ENS"], linewidth=0.4, zorder=2)
        for col, _name, marker in LEARNERS:
            ax.plot([fval(r[col]) for r in rows], y, linestyle="none", marker=marker, markersize=3.0,
                    markerfacecolor="#FFFFFF", markeredgecolor=F.BLACK, markeredgewidth=0.6, zorder=3)
        ax.yaxis.set_major_locator(FixedLocator(y))
        ax.set_yticklabels([D.INPUT_LABELS[r["input"]] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(0, None)
        ax.set_xlabel("Mean absolute SHAP value (log-odds scale)")
        F.panel_title(ax, j, D.OUTCOME_LABELS[o])
    n_learn = D.fmt_count(D.single([D.count(v) for v in ss["n_learners"]], "n_learners of shap_summary.csv"))
    handles = [Patch(facecolor=F.tint(F.MODEL_COLOURS["ENS"], 0.5), edgecolor=F.MODEL_COLOURS["ENS"],
                     linewidth=0.4)]
    labels = [f"Mean over the {n_learn} base learners"]
    for _col, name, marker in LEARNERS:
        handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=3.0, markerfacecolor="#FFFFFF",
                              markeredgecolor=F.BLACK, markeredgewidth=0.6))
        labels.append(name)
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    sampled = []
    for o in D.OUTCOMES:      # people sampled: one number for the rows of each outcome
        n = D.single([D.count(v) for v in D.select(ss, outcome=o)["n_records"]], f"n_records of shap_summary.csv, {o}")
        sampled.append(f"{OUT_WORD[o]} {D.fmt_count(n)}")
    n_rec = "; ".join(sampled)
    # the values explain the base learners' log odds before the isotonic step, not the ensemble's predicted risks
    title = "Mean absolute SHAP value of each input for the base learners of the ensemble in the development test set"
    per_panel = join_and([f"{lower_first(OUTCOME_TEXT[(o, 'DEV')])} {D.legend_panel(j)}"
                          for j, o in enumerate(D.OUTCOMES)])
    legend = (
        "Mean absolute SHAP value of each input on the log-odds scale of the base learners, for "
        f"{per_panel}, in "
        f"a random sample of people of the development test set (people sampled: {n_rec}). Bars show the mean "
        f"over the {n_learn} base learners of the ensemble (each of the three learners fitted in each calibration "
        "fold), with the mean over the folds of each learner. SHAP values describe the base learners "
        f"before the isotonic calibration step. Inputs are ordered by their mean absolute SHAP value. The input "
        f"'{D.INPUT_LABELS['male']}' is sex, 1 for men and 0 for women.")
    notes = [D.abbreviation_note("ALT", "AST", "BMI", "SHAP")]
    return publish(fig, "shap_summary", "si", order_of("shap_summary", "si"), title, legend, notes,
                   ["shap_summary.csv"])


def shap_level_label(input_name, level):
    if input_name == "age":
        return f"{GE}{level[:-1]}" if level.endswith("+") else level
    return D.level_label(input_name, level)


def shap_rows(sbl, inputs):
    """Rows of one panel: for each input a header row, then its levels that have a mean SHAP value in at least one
    outcome (levels with fewer than 10 sampled people have none), in order."""
    out = []
    for inp in inputs:
        rows = D.select(sbl, input=inp)
        levels = [lv for lv in dict.fromkeys(rows["level"])
                  if any(math.isfinite(fval(v)) for v in D.select(rows, level=lv)["mean_shap"])]
        if inp != "age":
            levels = [lv for lv in D.ordered_levels(levels)]
            levels = [next(x for x in rows["level"] if D.level_key(x) == k) for k in levels]
        out.append(dict(header=True, input=inp, label=INPUT_HEAD[inp]))
        out += [dict(header=False, input=inp, level=lv, label=shap_level_label(inp, lv)) for lv in levels]
    return out


def figure_shap_levels(store):
    sbl = store("shap_by_level.csv")
    panels = [shap_rows(sbl, inputs) for inputs in SHAP_PANELS]
    fig, axes = F.subplots(1, 2, width="double", height_mm=172)
    omitted = any(D.is_masked(v) for v in sbl["n"])      # levels with 1-9 sampled people have no mean SHAP value
    for j, rows in enumerate(panels):
        ax = axes[j]
        y = np.arange(len(rows))[::-1].astype(float)
        ax.axvline(0, color=F.GREY, linewidth=0.6, zorder=1)
        for o, offset in zip(D.OUTCOMES, SHAP_OFFSETS):
            st = SHAP_STYLE[o]
            for inp in dict.fromkeys(r["input"] for r in rows):
                ys, xs = [], []
                for yy, r in zip(y, rows):
                    if r["header"] or r["input"] != inp:
                        continue
                    cell = D.select(sbl, outcome=o, input=inp, level=r["level"])
                    v = fval(cell["mean_shap"].iloc[0]) if len(cell) == 1 else math.nan
                    ys.append(yy + offset)
                    xs.append(v)
                ax.plot(xs, ys, color=st["color"], linewidth=0.6, zorder=2)
                ax.plot(xs, ys, linestyle="none", marker=st["marker"], markersize=3.2, color=st["color"], zorder=3)
        ax.yaxis.set_major_locator(FixedLocator(y))
        ax.set_yticklabels([r["label"] for r in rows])
        for tick, r in zip(ax.get_yticklabels(), rows):
            if r["header"]:
                tick.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel("Mean SHAP value (log-odds scale)")
        F.panel_title(ax, j, SHAP_PANEL_TITLES[j])
    handles = [Line2D([], [], color=SHAP_STYLE[o]["color"], marker=SHAP_STYLE[o]["marker"], markersize=3.2,
                      linewidth=0.6) for o in D.OUTCOMES]
    labels = [OUTCOME_TEXT[(o, "DEV")] for o in D.OUTCOMES]
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    title = "Mean SHAP value at each level of each input in the development test set"
    outcomes = join_and([lower_first(OUTCOME_TEXT[(o, "DEV")]) for o in D.OUTCOMES])
    panels = join_and([f"{D.legend_panel(j)} {lower_first(t)}" for j, t in enumerate(SHAP_PANEL_TITLES)])
    legend = (
        "Mean SHAP value (log-odds scale of the base learners) of the people at each level of each input, for "
        f"{outcomes}, in the samples of {figure_cite('shap_summary', 'si')}: {panels}. Values "
        "above 0 "
        "mark levels at which the input's mean contribution to the base learners' log odds (before the isotonic "
        "calibration step) was positive, and values below 0 levels at which it was negative; SHAP values show "
        f"associations within the models only. Age is shown in 5-year bands and BMI in bands of {D.BMI_UNIT}; the top "
        f"band ({GE}30 {D.BMI_UNIT}) joins the obesity classes II and III.")
    notes = []
    if omitted:
        notes.append("Levels held by fewer than 10 sampled people are left out.")
    notes.append(D.abbreviation_note("ALT", "AST", "BMI", "SHAP"))
    return publish(fig, "shap_levels", "si", order_of("shap_levels", "si"), title, legend, notes,
                   ["shap_by_level.csv"])


# ================================================================================================ main
def main():
    # first: remove both indexes and every output of their ids and of the ids written here, so that a run that stops
    # leaves no out-of-date legend, image or index (s10 then stops on what is missing)
    for target, ids in (("main", MAIN_ORDER), ("si", SI_ORDER)):
        D.reset_index("figure", target, ids=ids)
    store = Store()
    made, notes = [], []
    label, flow_notes = figure_flow(store)
    made.append(label)
    notes += flow_notes
    for outcome in D.OUTCOMES:
        made.append(figure_roc_calibration(store, outcome))
    made.append(figure_subgroups(store))
    made.append(figure_decision_curves(store))
    made.append(figure_prediction_distributions(store))
    made.append(figure_shap_summary(store))
    made.append(figure_shap_levels(store))
    for n in dict.fromkeys(notes):
        print("note: " + n)
    worst = max(ROC_AREA, key=lambda c: c[1])
    n_main = sum(1 for m in made if m.startswith("Figure "))
    print(f"s09_figures: {n_main} main and {len(made) - n_main} supplementary figures; ROC area check on "
          f"{len(ROC_AREA)} curves, largest |trapezoid area - AUROC| {worst[1]:.5f} ({worst[0]}; tolerance "
          f"{ROC_TOL})")


if __name__ == "__main__":
    D.run(main)
