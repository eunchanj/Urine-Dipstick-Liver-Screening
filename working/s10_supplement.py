"""The Supplementary Information assembled from the supplementary table fragments of s08 and the supplementary figure
fragments of s09: supplement/Supplementary_Information.md and supplement/supplement_index.csv.

Run (once s08 and s09 have written their fragments; it opens output files only, no aggregate table; bytecode writing
off from the start):
    PYTHONDONTWRITEBYTECODE=1 python s10_supplement.py

Reads (outputs of s08 and s09, through display.read_index and plain reads of the fragment files)
  OUT_ROOT/supplement/tables/tables_index.csv and the <id>.md table fragments it lists
  OUT_ROOT/supplement/figures/figures_index.csv, its <id>.md legend fragments, and whether the <id>.pdf, .png, .eps
  and .tif files it lists exist and are not empty (the images themselves are not opened)
Writes (both removed first, so that a run that stops leaves no out-of-date output)
  OUT_ROOT/supplement/Supplementary_Information.md: a title line, the contents by purpose (where the sections are, then
  one line per group of GROUPS, with the short topics of its tables and figures), then
    1 Methods details        headings with [[TEXT: ...]] placeholders for the methods text
    2 Supplementary tables   the table fragments, in TABLES order, as s08 wrote them
    3 Supplementary figures  for each figure in FIGURES order its PNG (figures/<id>.png, empty alternative text so
                             that pandoc adds no second caption), then its legend fragment as s09 wrote it
  Sections are numbered without the S of the table and figure labels (1, 1.4, 2), so that no number names both a
  section and a table or figure.
  OUT_ROOT/supplement/supplement_index.csv: one row per section, table and figure, in document order, columns
    number   '1' or '1.4' for sections; 'Supplementary Table S3' and 'Supplementary Figure S3' for tables and figures
    kind     section, table or figure
    id       section key or fragment id
    title    heading, or caption title without the label
    sources  aggregate tables the numbers come from (';'-separated, as in the fragment index; blank for sections)

STRUCTURE (METHODS, TABLES, FIGURES) sets the order. Each table and figure is one fragment id; its number is its place
in TABLES or FIGURES, and the fragment must carry that number (its index 'order' and the
'**Supplementary Table S<n>.**' label of its first line), since s08 and s09 also cite the numbers. The script stops,
listing every problem, when a fragment id is not in STRUCTURE, when an id of STRUCTURE has no fragment or no group of
GROUPS, when a number differs, or when a fragment and its index row disagree. With --allow-missing (partial runs) an id
without a fragment becomes a [[MISSING: ...]] placeholder that keeps its number.

Tokens, resolved in the text of this script and in the fragments: {{table:<id>}} -> 'Supplementary Table S3',
{{figure:<id>}} -> 'Supplementary Figure S2' (a panel letter follows the token: {{figure:<id>}}a -> 'Supplementary
Figure S2a'), {{section:<key>}} -> 'section 1.4' (cited elsewhere as 'section 1.4 of the Supplementary Information').
A token that does not resolve stops the script.

Checks of the assembled text (each problem is listed with its place; any problem stops the script):
  words        the word lists of text_patterns.py, through display.word_hits; a number that a list names counts as
               data, not text, in a table cell holding only numbers or on a line of its own:
               such matches are counted in the summary line instead. text_patterns.py is an optional local file,
               not part of this repository: without it the word checks are skipped, and display says so in one line
  display      display.text_problems on every line (dashes, other non-ASCII signs, 'matched', the claims that no
               displayed string may make, American spellings, a comma in a number of fewer than five digits, a number
               and its unit without one space between them, a panel reference in parentheses)
  citations    no supplementary item cited without the word 'Supplementary', and no other name for the supplement
  placeholders no unresolved token, and no [[...]] other than the TEXT and MISSING placeholders
  values       no 'nan', and no table cell that shows a count of 1-9 with its percentage
  layout       sections, subsections and the table and figure captions numbered in order, once each
"""
import aggregate_guard  # noqa: F401  (first statement: refuses individual-level files, records the files opened)

import argparse
import collections
import csv
import io
import os
import re

import display as D

DOC_TITLE = "Supplementary Information"
SUPPLEMENT_MD = "Supplementary_Information.md"
SUPPLEMENT_INDEX = "supplement_index.csv"
INDEX_COLUMNS = ("number", "kind", "id", "title", "sources")
CONTENTS_HEADING = "Contents by purpose"
KINDS = ("table", "figure")
WRITERS = {"table": "s08_tables.py", "figure": "s09_figures.py"}
ID = re.compile(r"^[a-z][a-z0-9_]*$")


def text_placeholder(what):
    """A paragraph still to be written: '[[TEXT: <what to write>]]'."""
    return f"[[TEXT: {what}]]"


def ref(kind, key):
    """Token of a table, figure or section of the supplement ('{{table:inputs}}'), resolved when it is rendered."""
    return "{{" + kind + ":" + key + "}}"


def main_ref(kind, key):
    """Label of a main table or figure that a directive cites ('Figure 1'), from its number in MAIN_ITEMS."""
    return D.label(kind, "main", MAIN_ITEMS[(kind, key)])


def lc(text):
    """First letter in lower case (a topic inside a sentence), acronyms kept."""
    return text if text[:2].isupper() else text[:1].lower() + text[1:]


def refs(items):
    """Tokens of {kind: keys}, in that order: commas, with 'and' (and no comma) before the last one."""
    tokens = [ref(kind, key) for kind, keys in items.items() for key in keys]
    return tokens[0] if len(tokens) == 1 else ", ".join(tokens[:-1]) + " and " + tokens[-1]


# ------------------------------------------------------------------------------------------------ structure
# the numbers of the main tables and figures that the directives below cite: those of MAIN_ORDER in s08_tables.py and
# s09_figures.py (they must agree)
MAIN_ITEMS = {("table", "baseline"): 1, ("table", "comparators"): 3, ("figure", "flow"): 1, ("figure", "subgroups"): 4}
# where the outcomes appear (the section on the outcomes names these items): the items that show ALT and AST, repeated
# for each (ALT then AST, as two blocks, panels a and b or two columns of panels), drawn together in each panel, or
# side by side in one table, and those that show neither (the inputs, their codes and the records of BMI)
BOTH_OUTCOMES = {"table": ("enzyme_availability", "bmi_availability", "enzyme_crosstab", "ensemble_details",
                           "lr_coefficients", "operating_points", "comparator_differences", "subgroups",
                           "sensitivity_analyses", "enzyme_by_year"),
                 "figure": ("decision_curves", "prediction_distributions", "shap_summary", "shap_levels")}
NO_OUTCOME = {"table": ("inputs", "knhanes_codes", "alcohol_codes", "bmi_source")}
# section 1: (key, heading, what the paragraph should describe).
METHODS = (
    ("development", "Development data",
     "Describe the hospital records, the study period, the choice of the index urinalysis and of the same-day blood "
     "test, the adults included (adults with same-day ALT and AST results that could both be classified, complete "
     "dipstick results and sex, and a BMI recorded on the index day or in the 30 days before it), the people left out "
     f"({ref('table', 'enzyme_availability')} and {ref('table', 'bmi_availability')}), and the split of people into "
     f"the training and test sets within the four groups of ALT and AST above or not above 40 U/L "
     f"({main_ref('figure', 'flow')})."),
    ("survey", "KNHANES participants and survey items",
     "Describe the survey years, the eligible adults (aged 19 years or more, with complete dipstick results and sex, "
     "height and weight recorded and within the accepted ranges, and ALT and AST both recorded), the items on "
     f"pregnancy and alcohol drinking frequency with the handling of each code ({ref('table', 'alcohol_codes')}; "
     "pregnancy was counted and was not a reason for exclusion), and how ALT and AST were measured, with the change "
     f"of laboratory in 2022 ({main_ref('figure', 'flow')})."),
    ("dipstick", "Urine dipstick inputs",
     "Describe how each dipstick field was recorded in each source and converted to model values, by source and "
     f"survey year ({ref('table', 'inputs')} and {ref('table', 'knhanes_codes')})."),
    ("bmi", "Body mass index",
     "Give the rule of each source. Hospital records: BMI recorded on the urinalysis day or, if none, the most recent "
     "in the 30 days before it (days -30 to 0 inclusive); none after the index day. KNHANES: height and weight "
     "measured on the survey day. Give the ranges of height, weight and BMI that were accepted, the source and timing "
     f"of the hospital records with the share of index urinalyses on a health check-up day "
     f"({ref('table', 'bmi_source')}), and the comparison of the adults with and without a BMI "
     f"({ref('table', 'bmi_availability')})."),
    ("outcomes", "Outcomes",
     f"Describe the two outcomes, {D.OUTCOME_LABELS['ALT']} and {D.OUTCOME_LABELS['AST']}, with equal weight and in "
     "the same words for each: two binary outcomes of the same people, each with its own models, neither ranked above "
     "the other, not joined into one outcome or one alert. Give the rule of both outcomes in both sources: positive "
     "when the stored value is above 40 U/L (40.0 U/L is not, 40.5 U/L is). Give the measurements of each source: "
     "hospital records, same-day results, a value reported beyond a measurement limit being classified from its "
     "comparator; KNHANES, the published values, with no conversion in the 2022-2024 surveys. Say that "
     f"{ref('table', 'enzyme_crosstab')} and the row of both enzymes above 40 U/L in {main_ref('table', 'baseline')} "
     f"are descriptive and are not outcomes. Say where the outcomes appear: both outcomes, ALT first, in "
     f"{refs(BOTH_OUTCOMES)}; no outcome in {refs(NO_OUTCOME)}."),
    ("ensemble", "Ensemble model",
     "Describe, for each outcome, the hyperparameter search for each learner, soft voting, isotonic calibration, class "
     f"weighting and random seeds ({ref('table', 'ensemble_details')})."),
    ("comparison", "Logistic regression models and the rule",
     "Describe the two logistic regression models, on age, sex and BMI and on all 12 inputs, for each outcome: "
     "standardised inputs with an L2 penalty (C = 1) and no calibration step; age, BMI, urine pH, urine specific "
     "gravity and the graded dipstick results as linear terms, without interactions. Describe the rule, "
     f"{D.RULE_LONG['BMI25']} ({ref('table', 'lr_coefficients')})."),
    ("thresholds", "Thresholds",
     "Describe how each threshold was set in the development test set, separately for each outcome (Youden index, 80% "
     "and 90% sensitivity, and the thresholds that flag the proportion nearest to that of the rule, with how a tie "
     f"was broken) and applied unchanged to KNHANES ({main_ref('table', 'comparators')} and "
     f"{ref('table', 'operating_points')}, with the differences from the rule in "
     f"{ref('table', 'comparator_differences')}). Say that the rule flags the same people for both outcomes."),
    ("measures", "Performance measures and confidence intervals",
     "Describe discrimination, calibration-in-the-large, the calibration slope, the observed-to-expected ratio, the "
     "Brier scores, the bootstrap confidence intervals, the comparison of areas under the curve (the DeLong test and "
     "the paired bootstrap for the three differences, in each of the two data sets and for each outcome, 12 tests, "
     "two-sided, with P values not adjusted for multiple comparisons) and the decision curves (net benefit from each "
     "model's predicted probability as it stands, without recalibration in KNHANES, and of the fixed alerts at the "
     "Youden and 80% and 90% sensitivity thresholds; masked values are gaps, not zero), for "
     f"{D.DATASET_LONG['KN']} and the development test set ({ref('figure', 'decision_curves')}). Describe the "
     f"distributions of predicted risk ({ref('figure', 'prediction_distributions')}). Give the formats of the tables "
     "and figures: a confidence interval as estimate (lower, upper); mean (SD), with age as whole numbers; median "
     f"[quartile 1, quartile 3]; ALT and AST in U/L and BMI in {D.BMI_UNIT}, as stored; percentages that describe "
     "people as whole numbers from 10 up and with one decimal under 10 (a share that would round to 100 without being "
     "100 keeps "
     "one decimal, and a share below 0.05% is shown as <0.1); accuracy measures, prevalence and mean predicted risk "
     "with one decimal; differences in percentage points and alerts per case with two decimals; thresholds with "
     "four."),
    ("additional", "Subgroups and sensitivity analyses",
     f"Describe the subgroups by age group, sex, survey period and alcohol drinking frequency in "
     f"{D.DATASET_LONG['KN']} ({main_ref('figure', 'subgroups')}a for ALT, {main_ref('figure', 'subgroups')}b for AST "
     f"and {ref('table', 'subgroups')} for both outcomes; people whose drinking frequency is unknown are given as "
     "counts only); the index urinalysis on a health check-up day against another day in the development test set and "
     f"the survey-weighted estimates ({ref('table', 'sensitivity_analyses')}); and the ALT and AST values by survey "
     f"year ({ref('table', 'enzyme_by_year')})."),
    ("explanation", "SHAP values",
     "Describe the sample of the development test set (the same people for both outcomes), the base learners explained "
     "(the mean log odds of the 15 base learners, before the isotonic calibration step), the scale of the values and "
     "how they were summarised, with the bands of age and of BMI (the top BMI band joins obesity classes II and III); "
     f"SHAP values show associations within the models only ({ref('figure', 'shap_summary')}a for ALT, "
     f"{ref('figure', 'shap_summary')}b for AST and {ref('figure', 'shap_levels')} for both outcomes)."),
    ("counts", "Small counts",
     "Describe how counts of 1-9 are shown and how sparse levels are pooled for display."),
    ("software", "Software", "Give the software and package versions."),
)
# sections 2 and 3: (fragment id, topic). The topic stands in the contents by purpose (GROUPS, which give its context)
# and names a missing fragment in messages and placeholders; a present fragment is shown with the title of its index
# row. One id per table or figure; add ids here when s08 or s09 split a topic into several tables or figures, in the
# order they should be numbered. The ids and numbers are those of s08_tables.SI_ORDER and s09_figures.SI_ORDER (each
# fragment must carry its number).
TABLES = (
    ("inputs", "Inputs and their coding"),
    ("knhanes_codes", "KNHANES dipstick codes by survey year"),
    ("alcohol_codes", "KNHANES alcohol frequency codes and their groups, and the pregnancy codes"),
    ("enzyme_availability", "Adults with same-day ALT and AST both classifiable against those without"),
    ("bmi_availability", "Adults with a BMI against those without"),
    ("bmi_source", "BMI source and interval, and the share of health check-up days by period"),
    ("enzyme_crosstab", "ALT above 40 U/L by AST above 40 U/L in each data set"),
    ("ensemble_details", "Training and hyperparameters of the ensemble"),
    ("lr_coefficients", "Coefficients of the logistic regression models"),
    ("operating_points", "Operating points of the ensemble and of the models at the threshold set by the rule"),
    ("comparator_differences", "Differences between each model threshold and the rule"),
    ("subgroups", "Results by subgroup"),
    ("sensitivity_analyses", "Sensitivity analyses"),
    ("enzyme_by_year", "KNHANES ALT and AST by survey year"),
)
FIGURES = (
    ("decision_curves", "Decision curves"),
    ("prediction_distributions", "Distributions of predicted risk"),
    ("shap_summary", "Mean absolute SHAP values of the inputs"),
    ("shap_levels", "Mean SHAP values at each level of the inputs"),
)
# the sections in order: (key, heading, kind of the entries, entries)
STRUCTURE = (
    ("methods", "Methods details", "text", METHODS),
    ("tables", "Supplementary tables", "table", TABLES),
    ("figures", "Supplementary figures", "figure", FIGURES),
)
HEAD = f"# {DOC_TITLE}\n\n{DOC_TITLE} for: " + text_placeholder("title of the article")
# the contents at the head of the supplement, a short map by purpose: where the sections are, then one line per group,
# as (group, its tables, its figures). Every table and figure has one group, in the order of its number; the full
# titles are in the document and in supplement_index.csv.
CONTENTS_LEAD = (f"The methods details are in {ref('section', 'methods')}, the tables in {ref('section', 'tables')} "
                 f"and the figures in {ref('section', 'figures')}. The TRIPOD+AI and TRIPOD+AI for Abstracts "
                 f"checklists are in section {len(STRUCTURE) + 1}. By purpose:")
GROUPS = (
    ("Inputs and models", ("inputs", "knhanes_codes", "alcohol_codes", "ensemble_details", "lr_coefficients"),
     ("shap_summary", "shap_levels")),
    ("People", ("enzyme_availability", "bmi_availability", "bmi_source", "enzyme_crosstab", "enzyme_by_year"), ()),
    ("More results, both outcomes", ("operating_points", "comparator_differences", "subgroups", "sensitivity_analyses"),
     ("decision_curves", "prediction_distributions")),
)

# ------------------------------------------------------------------------------------------------ text checks
# The word lists are those of text_patterns.py, applied through display.word_hits (which exempts a number alone in a
# table cell of numbers); the patterns below are this script's own checks.
SMALL_COUNT = re.compile(r"[1-9] \(\d+(?:\.\d+)?\)")        # a count of 1-9 with its percentage
NAN = re.compile(r"(?i)\bnan\b")
TOKEN = re.compile(r"\{\{(table|figure|section):([a-z][a-z0-9_]*)\}\}")
BRACES = re.compile(r"\{\{|\}\}")
PLACEHOLDER = re.compile(r"\[\[(TEXT|MISSING): (?:(?!\[\[|\]\]).)+\]\]")     # single brackets may stand inside
PIPE = re.compile(r"(?<!\\)\|")
# a supplementary item cited without the word 'Supplementary', or another name for the supplement
OTHER_CITATION = re.compile(r"(?<!Supplementary )\b(?:Table|Figure)s? S\d+|\bData S\d+\b|"
                            r"(?i:\bsupporting information\b)")


class AssemblyError(ValueError):
    """The supplement cannot be assembled; the message lists every problem found."""

    def __init__(self, what, problems):
        super().__init__(f"{what} ({len(problems)} problem(s)):\n  " + "\n  ".join(problems))
        self.problems = list(problems)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ------------------------------------------------------------------------------------------------ numbering
def numbering(structure=None):
    """Numbers of the supplement: ({(kind, id): n} for tables and figures, {section key: '1' or '1.4'} in document
    order). Sections carry no S, so that their numbers differ from the labels of the tables and figures
    (Supplementary Table S1)."""
    structure = STRUCTURE if structure is None else structure
    numbers, sections = {}, {}
    for i, (key, _heading, kind, items) in enumerate(structure, 1):
        sections[key] = str(i)
        for j, item in enumerate(items, 1):
            if kind == "text":
                sections[item[0]] = f"{i}.{j}"
            else:
                numbers[(kind, item[0])] = j
    return numbers, sections


def check_structure(structure=None, head=None, groups=None):
    """Stop (ValueError) unless STRUCTURE has the sections text, table, figure in that order, unique slug keys and ids,
    tokens in its own text that all resolve, and each table and figure in one group of GROUPS, in the order of its
    number."""
    structure = STRUCTURE if structure is None else structure
    head = HEAD if head is None else head
    groups = GROUPS if groups is None else groups
    problems = []
    if [s[2] for s in structure] != ["text", "table", "figure"]:
        problems.append("the sections must hold text, then tables, then figures")
    keys = [s[0] for s in structure] + [item[0] for s in structure for item in s[3]]
    problems += [f"key or id {k!r} is used twice" for k in sorted({k for k in keys if keys.count(k) > 1})]
    problems += [f"key or id {k!r} is not a slug ({ID.pattern})" for k in keys if not ID.match(str(k))]
    for s in structure:
        if not s[3]:
            problems.append(f"section {s[0]!r} has no entries")
        for item in s[3]:
            if len(item) != (3 if s[2] == "text" else 2) or not all(isinstance(x, str) and x.strip() for x in item):
                problems.append(f"entry {item!r} of section {s[0]!r} has the wrong form")
    numbers, sections = numbering(structure)
    for j, kind in enumerate(KINDS, 1):
        listed = [fid for group in groups for fid in group[j]]
        problems += [f"{kind} {fid!r} of GROUPS is not in STRUCTURE" for fid in listed if (kind, fid) not in numbers]
        problems += [f"{kind} {fid!r} stands in {listed.count(fid)} groups of GROUPS"
                     for (k, fid) in numbers if k == kind and listed.count(fid) != 1]
        for group in groups:
            got = [numbers[(kind, fid)] for fid in group[j] if (kind, fid) in numbers]
            if got != sorted(got):
                problems.append(f"group {group[0]!r}: its {kind}s are not in the order of their numbers")
    texts = ([("title", head), (CONTENTS_HEADING, CONTENTS_LEAD)]
             + [(item[0], item[2]) for s in structure if s[2] == "text" for item in s[3]])
    for where, text in texts:
        for kind, key in TOKEN.findall(text):
            if (key not in sections) if kind == "section" else ((kind, key) not in numbers):
                problems.append(f"{where}: token {ref(kind, key)} names nothing in STRUCTURE")
        if BRACES.search(TOKEN.sub("", text)):
            problems.append(f"{where}: braces that are not a token")
    if problems:
        raise AssemblyError("STRUCTURE of s10_supplement.py", problems)


# ------------------------------------------------------------------------------------------------ fragments
Fragment = collections.namedtuple("Fragment", "kind id order title sources text image")


def _relative(path):
    return os.path.relpath(path, D.OUT_ROOT).replace(os.sep, "/")


def read_fragments(kind):
    """The supplementary fragments of one kind as {id: Fragment}, with the problems of the index file and its fragments
    (a list of strings). Figure images are checked for existence and size only."""
    index = D.index_path(kind, "si")
    folder = D.fragment_dir(kind, "si", create=False)
    supplement = os.path.join(D.OUT_ROOT, "supplement")
    try:
        rows = D.read_index(kind, "si")
    except ValueError:
        return {}, [f"{_relative(index)}: the header differs from the index format of display.py"]
    found, orders, problems = {}, {}, []
    for line, r in enumerate(rows, 2):
        fid = r.get("id") or ""
        at = f"{_relative(index)} line {line} ({fid or 'no id'})"
        if not ID.match(fid):
            problems.append(f"{at}: the id is not a slug")
            continue
        if fid in found:
            problems.append(f"{at}: the id is listed twice")
            continue
        if r.get("target") != "si":
            problems.append(f"{at}: target {r.get('target')!r}, expected 'si'")
        try:
            order = int(r.get("order") or "")
        except ValueError:
            problems.append(f"{at}: order {r.get('order')!r} is not a whole number")
            continue
        if order in orders:
            problems.append(f"{at}: {D.label(kind, 'si', order)} is also used by {orders[order]!r}")
        orders[order] = fid
        sources = [s for s in (r.get("sources") or "").split(";") if s]
        unknown = [s for s in sources if s not in D.AGGREGATES]
        if not sources or unknown:
            problems.append(f"{at}: sources must name aggregate tables (unknown: {unknown})")
        title = " ".join((r.get("title") or "").split())
        if not title:
            problems.append(f"{at}: no title")
        image = None
        if kind == "table":
            if r.get("file") != f"{fid}.md":
                problems.append(f"{at}: file {r.get('file')!r}, expected {fid + '.md'!r}")
        else:
            want = [fid + ext for ext in D.FIGURE_TYPES]
            if (r.get("files") or "").split(";") != want:
                problems.append(f"{at}: files {r.get('files')!r}, expected {';'.join(want)!r}")
            for name in want:
                p = os.path.join(folder, name)
                if not os.path.isfile(p) or os.path.getsize(p) == 0:
                    problems.append(f"{at}: {name} is missing or empty")
            image = os.path.relpath(os.path.join(folder, fid + ".png"), supplement).replace(os.sep, "/")
        path = os.path.join(folder, f"{fid}.md")
        if not os.path.isfile(path):
            problems.append(f"{at}: fragment {fid}.md is missing")
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        first, _, body = text.partition("\n\n")
        if first != f"**{D.label(kind, 'si', order)}.** {title}":
            problems.append(f"{at}: the first line of {fid}.md is not "
                            f"'**{D.label(kind, 'si', order)}.** <index title>'")
        if not body.strip():
            problems.append(f"{at}: {fid}.md has nothing after its caption")
        elif kind == "table" and not any(x.lstrip().startswith("|") for x in body.splitlines()):
            problems.append(f"{at}: {fid}.md holds no pipe table")
        elif kind == "figure" and " ".join(body.split()) != " ".join((r.get("legend") or "").split()):
            problems.append(f"{at}: the legend in {fid}.md differs from the legend of the index row")
        found[fid] = Fragment(kind, fid, order, title, ";".join(sources), text, image)
    return found, problems


def match_structure(found, allow_missing=False, structure=None):
    """Problems of the fragments against STRUCTURE (ids it lacks, numbers that differ, and, unless allow_missing, ids
    without a fragment), and the (kind, id) of the ids without a fragment."""
    numbers, _ = numbering(structure)
    problems, missing = [], []
    for kind in KINDS:
        expected = {fid: n for (k, fid), n in numbers.items() if k == kind}
        for fid, frag in sorted(found[kind].items(), key=lambda kv: kv[1].order):
            label = D.label(kind, "si", frag.order)
            if fid not in expected:
                problems.append(f"{kind} {fid!r} ({label} of {WRITERS[kind]}) is not in STRUCTURE of s10_supplement.py")
            elif frag.order != expected[fid]:
                problems.append(f"{kind} {fid!r} is {label} in its index, but the supplement numbers it "
                                f"{D.label(kind, 'si', expected[fid])}: give it order {expected[fid]} in "
                                f"{WRITERS[kind]} (or change STRUCTURE)")
        missing += [(kind, fid) for fid, _n in sorted(expected.items(), key=lambda kv: kv[1]) if fid not in found[kind]]
    if missing and not allow_missing:
        names = ", ".join(f"{D.label(k, 'si', numbers[(k, f)])} {f!r} ({WRITERS[k]})" for k, f in missing)
        problems.append(f"no fragment yet for {names}: run s08_tables.py and s09_figures.py first "
                        f"(--allow-missing is for a partial run)")
    return problems, missing


# ------------------------------------------------------------------------------------------------ document
class Doc:
    """The supplement written in STRUCTURE order: headings, paragraphs and placed fragments, each kept as a part with
    its place (for messages); tokens are resolved when the parts are rendered. Also collects the rows of
    supplement_index.csv and the topics of the contents by purpose."""

    def __init__(self, structure=None):
        self.numbers, self.sections = numbering(structure)
        self._sections_left = list(self.sections)
        self._left = {k: [fid for (kk, fid) in self.numbers if kk == k] for k in KINDS}
        self.parts, self.rows, self.topics = [], [], {}

    def heading(self, key, title):
        """Heading of the next section ('## 2 Supplementary tables') or subsection ('### 1.4 Outcomes')."""
        _require(self._sections_left and self._sections_left[0] == key,
                 f"heading {key!r} is not the next section of STRUCTURE")
        self._sections_left.pop(0)
        number = self.sections[key]
        level = 3 if "." in number else 2
        self.parts.append((f"{number} {title}", f"{'#' * level} {number} {title}"))
        self.rows.append([number, "section", key, title, ""])

    def paragraph(self, place, text):
        self.parts.append((place, text))

    def place(self, kind, fid, frag=None, topic=None):
        """The next table or figure: its fragment as written (a figure after its image), or, when frag is None, a
        [[MISSING: ...]] placeholder under its number and topic."""
        _require(self._left[kind] and self._left[kind][0] == fid, f"{kind} {fid!r} placed out of order")
        self._left[kind].pop(0)
        n = self.numbers[(kind, fid)]
        label = D.label(kind, "si", n)
        if frag is None:
            title, sources = topic, ""
            body = (f"**{label}.** {topic}\n\n[[MISSING: no fragment {fid!r} in "
                    f"{_relative(D.index_path(kind, 'si'))} (written by {WRITERS[kind]})]]")
        else:
            title, sources = frag.title, frag.sources
            body = frag.text if frag.image is None else f"![]({frag.image})\n\n{frag.text}"
        self.parts.append((f"{label} ({fid})", body))
        self.topics[(kind, fid)] = topic
        self.rows.append([label, kind, fid, title, sources])

    def resolve(self, text):
        """Tokens replaced by their labels ('Supplementary Table S3', 'Supplementary Figure S2', 'section 1.4'); a token
        that names nothing stays (and fails the checks)."""
        def label(m):
            kind, key = m.groups()
            if kind == "section":
                return f"section {self.sections[key]}" if key in self.sections else m.group(0)
            n = self.numbers.get((kind, key))
            return m.group(0) if n is None else D.label(kind, "si", n)
        return TOKEN.sub(label, text)

    def contents(self):
        """The contents by purpose: where the sections are, then one line per group with the topics of its items."""
        lines = [f"## {CONTENTS_HEADING}", "", CONTENTS_LEAD, ""]
        for group, tables, figures in GROUPS:
            items = [(kind, fid) for kind, ids in zip(KINDS, (tables, figures)) for fid in ids]
            lines.append(f"- **{group}:** "
                         + "; ".join(f"{ref(k, f)}, {lc(self.topics[(k, f)])}" for k, f in items) + ".")
        return "\n".join(lines)

    def render(self, head):
        """[(place, markdown)] of the whole document in order, tokens resolved."""
        _require(not self._sections_left and not any(self._left.values()), "the supplement is not complete")
        parts = [("title", head), (CONTENTS_HEADING, self.contents())] + self.parts
        return [(place, self.resolve(md).strip()) for place, md in parts]


def build(found, structure=None):
    """The Doc of the supplement from the fragments found (missing ones as placeholders)."""
    structure = STRUCTURE if structure is None else structure
    doc = Doc(structure)
    for key, heading, kind, items in structure:
        doc.heading(key, heading)
        for item in items:
            if kind == "text":
                sub, title, what = item
                doc.heading(sub, title)
                doc.paragraph(f"{doc.sections[sub]} {title}", text_placeholder(what))
            else:
                fid, topic = item
                doc.place(kind, fid, found[kind].get(fid), topic)
    return doc


# ------------------------------------------------------------------------------------------------ checks
def _cells(line):
    """(start, end) of the cells of one pipe-table line (positions within the line)."""
    cuts = [m.start() for m in PIPE.finditer(line)]
    return [(a + 1, b) for a, b in zip(cuts, cuts[1:])]


def check_part(place, text):
    """(problems, numbers exempted) of one rendered part of the supplement or of the index file: the word lists through
    display.word_hits (a number that a list names is data in a numeric table cell or on a line of its own), then this
    script's checks of each line."""
    hits, exempt = D.word_hits(text)
    problems = [f"{place}, line {ln}: {name} pattern matches {word!r}" for ln, name, word in hits]
    for i, line in enumerate(text.splitlines(), 1):
        at = f"{place}, line {i}"
        problems += [f"{at}: {p}" for p in D.text_problems(line)]
        if OTHER_CITATION.search(line):
            problems.append(f"{at}: a supplementary item cited without the word 'Supplementary', or another name for "
                            "the supplement")
        if BRACES.search(line):
            problems.append(f"{at}: unresolved token or stray braces")
        if re.search(r"\[\[|\]\]", PLACEHOLDER.sub("", line)):
            problems.append(f"{at}: a [[...]] that is not a TEXT or MISSING placeholder")
        if NAN.search(line):
            problems.append(f"{at}: 'nan' (a value that was not formatted)")
        if line.lstrip().startswith("|"):
            for a, b in _cells(line):
                if SMALL_COUNT.fullmatch(line[a:b].strip()):
                    problems.append(f"{at}: a cell shows a count of 1-9 with its percentage")
    return problems, exempt


def check_layout(text, structure=None):
    """Problems of the order of headings and captions in the assembled text."""
    numbers, sections = numbering(structure)
    problems = []
    heads = re.findall(r"(?m)^(#+) (.*)$", text)
    want = [("#", DOC_TITLE), ("##", CONTENTS_HEADING)]
    titles = {**{s[0]: s[1] for s in (structure or STRUCTURE)},
              **{item[0]: item[1] for s in (structure or STRUCTURE) if s[2] == "text" for item in s[3]}}
    want += [("###" if "." in n else "##", f"{n} {titles[k]}") for k, n in sections.items()]
    if heads != want:
        problems.append("headings are not the title, the contents and the sections of STRUCTURE in order (a fragment "
                        "may hold a heading)")
    for kind, word in (("table", "Table"), ("figure", "Figure")):
        n = sum(1 for k, _ in numbers if k == kind)
        caps = re.findall(rf"(?m)^\*\*Supplementary {word} S(\d+)\.\*\* ", text)
        if caps != [str(i) for i in range(1, n + 1)]:
            problems.append(f"{word.lower()} captions are numbered {caps}, expected 1-{n} once each, in order")
    n_fig = sum(1 for k, _ in numbers if k == "figure")
    if len(re.findall(r"(?m)^!\[\]\(", text)) > n_fig:
        problems.append("more images than figures")
    return problems


def index_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(INDEX_COLUMNS)
    w.writerows(rows)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ main
def output_paths():
    return [D.out_path("supplement", name, create=False) for name in (SUPPLEMENT_MD, SUPPLEMENT_INDEX)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble the supplement from the supplementary fragments of s08 and s09.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="for a partial run: show an id of STRUCTURE without a fragment as a placeholder")
    a = ap.parse_args(argv)

    for p in output_paths():        # first, so that a run that stops (STRUCTURE too) leaves no out-of-date output
        if os.path.exists(p):
            os.remove(p)
    check_structure()

    found, problems = {}, []
    for kind in KINDS:
        found[kind], p = read_fragments(kind)
        problems += p
    p, missing = match_structure(found, a.allow_missing)
    problems += p
    if problems:
        raise AssemblyError("the supplementary fragments do not fit STRUCTURE", problems)

    doc = build(found)
    parts = doc.render(HEAD)
    text = "\n\n".join(md for _, md in parts) + "\n"
    table = index_csv(doc.rows)
    problems, exempt = [], 0
    for place, md in parts + [(SUPPLEMENT_INDEX, table)]:
        p, e = check_part(place, md)
        problems += p
        exempt += e
    problems += check_layout(text)
    if problems:
        raise AssemblyError("the supplement breaks the text rules", problems)

    md_path, index_path = output_paths()
    D.write_text(md_path, text)
    D.write_text(index_path, table)
    n_text = len(re.findall(r"\[\[TEXT: ", text))
    summary = (f"s10_supplement: {DOC_TITLE} with {len(TABLES)} tables and {len(FIGURES)} figures in "
               f"{len(STRUCTURE)} sections ({len(METHODS)} methods subsections); {n_text} text placeholders")
    if missing:
        summary += f"; {len(missing)} missing fragment(s) shown as placeholders"
    if exempt:
        summary += f"; {exempt} numeric table cell(s) hold a number that a word list names (data, not text)"
    print(summary)


if __name__ == "__main__":
    D.run(main)
