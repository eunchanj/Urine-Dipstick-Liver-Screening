"""Figure style for s09-s10: matplotlib on the Agg backend, black text in one sans-serif font (Arial when present), the
Okabe-Ito colours, no transparency, figure widths of 89 mm (single column) and 183 mm (double column), lower-case bold
panel letters, and the checks that run before a figure is saved.

Two styles. STYLE, in force from import, is that of the supplementary figures, with every text at TEXT_PT (7 pt), the
size of the text of the main figures: all figures of the paper use one type size, the bold panel letters included.
MAIN_STYLE is that of the main figures: Arial, all visible text at least 7 pt and all lines at least 1 pt at the file's
native size, on a canvas of MAIN_CANVAS. A figure takes it by being made, drawn and saved inside a block:
    with F.main_style("flow"):           # or F.style_for(target, canvas), for a builder with a supplementary twin
        fig = F.new_figure(183, 200, layout=None)
        ...
        paths = F.save_all(fig, D.figure_base("flow", "main"))

save_all(fig, base) checks the figure (assert_opaque, assert_no_collisions, black text, and display.text_problems
and display.forbidden_problems on every visible string, which include texts taken from the aggregate tables such as
the flow labels; inside a main_style() block also main_style_problems, the font, sizes and line widths of every visible
artist), then writes base.pdf, base.png (300 dpi), base.eps (vector; checked by eps_problems: no
raster, Type 42 fonts) and base.tif (300 dpi, LZW), plus base_text.txt (the visible strings, one per line, for the word
checks). PNG and TIFF come from one 300 dpi rendering and are RGB without an alpha
channel; the PDF and EPS carry no creation date, so two runs write the same bytes.
"""
import contextlib
import io
import os
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.axis import Axis  # noqa: E402
from matplotlib.collections import Collection, LineCollection, PathCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import display as D  # noqa: E402

MM = 1 / 25.4
SINGLE_MM, DOUBLE_MM = 89.0, 183.0
MAX_HEIGHT_MM = 230.0        # assumed usable page height
DPI = 300

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
BLACK = OKABE_ITO["black"]
GREY = "#808080"
LIGHT_GREY = "#D9D9D9"
CYCLE = [OKABE_ITO[k] for k in ("blue", "vermillion", "bluish_green", "orange", "reddish_purple", "sky_blue",
                                "yellow", "black")]
MODEL_COLOURS = {"ENS": OKABE_ITO["blue"], "LR_ALL": OKABE_ITO["vermillion"], "LR_ASB": OKABE_ITO["bluish_green"]}
MODEL_LINESTYLES = {"ENS": "-", "LR_ALL": "--", "LR_ASB": "-."}
DATASET_COLOURS = {"DEV_TEST": OKABE_ITO["orange"], "KN": OKABE_ITO["blue"]}
RULE_MARKERS = {"BMI25": dict(marker="s", color=BLACK)}
SKIP_CHECK = "skip-collision-check"   # set_gid() value of a text that may overlap marks (it is still checked for text)


def _font():
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


FONT = _font()
TEXT_PT = 7.0                # the one type size of every figure: titles, axis labels, ticks, keys, panel letters
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"], "font.size": TEXT_PT,
    "axes.titlesize": TEXT_PT, "axes.titlelocation": "left", "axes.labelsize": TEXT_PT, "xtick.labelsize": TEXT_PT,
    "ytick.labelsize": TEXT_PT, "legend.fontsize": TEXT_PT, "legend.title_fontsize": TEXT_PT,
    "figure.titlesize": TEXT_PT,
    "text.color": BLACK, "axes.labelcolor": BLACK, "axes.titlecolor": BLACK, "axes.edgecolor": BLACK,
    "xtick.color": BLACK, "ytick.color": BLACK, "xtick.labelcolor": BLACK, "ytick.labelcolor": BLACK,
    "legend.labelcolor": BLACK, "legend.edgecolor": BLACK, "legend.facecolor": "white", "legend.framealpha": 1.0,
    "legend.frameon": False, "legend.fancybox": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.edgecolor": "white", "savefig.transparent": False, "savefig.dpi": DPI, "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5, "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2,
    "lines.markersize": 4, "patch.linewidth": 0.6, "hatch.linewidth": 0.5, "hatch.color": BLACK,
    "errorbar.capsize": 2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": LIGHT_GREY, "grid.linewidth": 0.5, "grid.alpha": 1.0,
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": False, "mathtext.default": "regular",
    "axes.prop_cycle": cycler(color=CYCLE),
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def apply():
    """Set the style (done on import; call again after plt.rcdefaults())."""
    matplotlib.rcParams.update(STYLE)


apply()

# the main figures: one sans-serif font and these sizes at the file's native size; lines of at least 1 pt
MAIN_FONT = "Arial"
MAIN_TEXT_PT, MAIN_LINE_PT = TEXT_PT, 1.0
PANEL_LETTER_PT, PANEL_GAP_PT = TEXT_PT, 4.0    # the bold panel letter: the size of the other text; its gap to the axes
# the largest (width, height) in mm: 'panel' for the figures of panels, 'flow' for the flow diagram
MAIN_CANVAS = {"panel": (DOUBLE_MM, MAX_HEIGHT_MM), "flow": (DOUBLE_MM, MAX_HEIGHT_MM)}
MAIN_STYLE = {
    **STYLE, "font.sans-serif": [MAIN_FONT], "font.size": MAIN_TEXT_PT, "axes.titlesize": MAIN_TEXT_PT,
    "axes.labelsize": MAIN_TEXT_PT, "xtick.labelsize": MAIN_TEXT_PT, "ytick.labelsize": MAIN_TEXT_PT,
    "legend.fontsize": MAIN_TEXT_PT, "legend.title_fontsize": MAIN_TEXT_PT, "figure.titlesize": MAIN_TEXT_PT,
    "axes.linewidth": MAIN_LINE_PT, "xtick.major.width": MAIN_LINE_PT, "ytick.major.width": MAIN_LINE_PT,
    "xtick.minor.width": MAIN_LINE_PT, "ytick.minor.width": MAIN_LINE_PT, "lines.linewidth": MAIN_LINE_PT,
    "lines.markeredgewidth": MAIN_LINE_PT, "patch.linewidth": MAIN_LINE_PT, "hatch.linewidth": MAIN_LINE_PT,
    "grid.linewidth": MAIN_LINE_PT,
}
_CANVAS = []                 # the canvas of each main_style() block in force, innermost last


@contextlib.contextmanager
def main_style(canvas="panel"):
    """The style of a main figure for everything inside the block: make, draw and save the figure in it, since
    matplotlib reads most settings when an artist is made and some when it is drawn. canvas: a key of MAIN_CANVAS.
    Inside the block size() allows that canvas, and save_all() asserts main_style_problems() and the rules of
    eps_problems()."""
    if canvas not in MAIN_CANVAS:
        raise ValueError(f"canvas must be one of {sorted(MAIN_CANVAS)}")
    _CANVAS.append(canvas)
    try:
        with matplotlib.rc_context(MAIN_STYLE):
            yield
    finally:
        _CANVAS.pop()


def style_for(target, canvas="panel"):
    """The block of a figure of `target`, for a builder shared by a main figure and its supplementary twin:
    main_style(canvas) for 'main'; for 'si' a block that changes nothing, since STYLE is in force outside
    main_style()."""
    if target not in D.TARGETS:
        raise ValueError(f"target must be one of {D.TARGETS}")
    return main_style(canvas) if target == "main" else contextlib.nullcontext()


def text_pt(si_pt, main_pt=MAIN_TEXT_PT):
    """A text size set by hand in a shared builder: `si_pt` in the supplementary style, `main_pt` in main_style()."""
    return main_pt if _CANVAS else si_pt


def line_pt(si_pt, main_pt=MAIN_LINE_PT):
    """A line width set by hand in a shared builder: `si_pt` in the supplementary style, `main_pt` in main_style()."""
    return main_pt if _CANVAS else si_pt


def tint(colour, f=0.6):
    """An opaque colour mixed with white (f = share of white), for bands and fills instead of transparency."""
    r, g, b = mcolors.to_rgb(colour)
    return mcolors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def size(width="single", height_mm=60.0):
    """(width, height) in inches; width 'single' (89 mm), 'double' (183 mm) or millimetres; at most 183 mm by 230 mm, or
    the canvas of the main_style() block in force."""
    w = {"single": SINGLE_MM, "double": DOUBLE_MM}.get(width, width)
    w = float(w)
    max_w, max_h = MAIN_CANVAS[_CANVAS[-1]] if _CANVAS else (DOUBLE_MM, MAX_HEIGHT_MM)
    if not 0 < w <= max_w or not 0 < float(height_mm) <= max_h:
        raise ValueError(f"figures are at most {max_w:.0f} mm wide and {max_h:.0f} mm high")
    return w * MM, float(height_mm) * MM


def new_figure(width="single", height_mm=60.0, layout="constrained", **kwargs):
    """A figure of column width (see size()); constrained layout keeps labels inside the figure."""
    return plt.figure(figsize=size(width, height_mm), layout=layout, **kwargs)


def subplots(nrows=1, ncols=1, width="single", height_mm=60.0, layout="constrained", **kwargs):
    return plt.subplots(nrows, ncols, figsize=size(width, height_mm), layout=layout, **kwargs)


def panel_title(ax, panel, title=""):
    """Label a panel: its lower-case bold letter (display.panel_label, from its position or letter; PANEL_LETTER_PT, the
    size of the other text) at the top left-hand corner, just outside the axes on the line of the title, and `title`,
    when given, as the left-aligned axes title. Returns the Text of the letter."""
    if title:
        ax.set_title(title, loc="left")
    return ax.annotate(D.panel_label(panel), xy=(0, 1), xycoords="axes fraction",
                       xytext=(-PANEL_GAP_PT, matplotlib.rcParams["axes.titlepad"]), textcoords="offset points",
                       ha="right", va="baseline", fontsize=PANEL_LETTER_PT, fontweight="bold")


# ------------------------------------------------------------------------------------------------ opacity
def _rgba(value):
    try:
        arr = mcolors.to_rgba_array(value)
    except (ValueError, TypeError):
        return np.empty((0, 4))
    return np.asarray(arr, float).reshape(-1, 4)


def _what(a):
    name = type(a).__name__
    text = a.get_text() if hasattr(a, "get_text") and isinstance(a, Text) else ""
    lab = a.get_label() if hasattr(a, "get_label") else ""
    extra = text or (lab if lab and not str(lab).startswith("_") else "")
    return f"{name} {extra[:30]!r}" if extra else name


def transparency_problems(fig):
    """Artists with an alpha other than None or 1, or colours whose alpha lies strictly between 0 and 1 (alpha 0 is
    'no paint', such as an unfilled marker; it is allowed)."""
    out = []
    artists = list(fig.findobj(include_self=True))
    artists += [t.get_bbox_patch() for t in fig.findobj(Text) if t.get_bbox_patch() is not None]
    for a in artists:
        if hasattr(a, "get_visible") and not a.get_visible():
            continue
        alpha = a.get_alpha() if hasattr(a, "get_alpha") else None
        if alpha is not None and not np.all(np.asarray(alpha, float) == 1.0):
            out.append(f"{_what(a)}: alpha {alpha}")
        for name in ("facecolor", "edgecolor", "color", "markerfacecolor", "markeredgecolor", "markerfacecoloralt"):
            getter = getattr(a, "get_" + name, None)
            if getter is None:
                continue
            try:
                value = getter()
            except (TypeError, ValueError, AttributeError):
                continue
            a_ch = _rgba(value)[:, 3]
            if np.any((a_ch > 0) & (a_ch < 1)):
                out.append(f"{_what(a)}: {name} with alpha {float(a_ch[(a_ch > 0) & (a_ch < 1)][0]):.2f}")
        if hasattr(a, "get_array") and type(a).__name__ == "AxesImage":
            arr = np.asarray(a.get_array())
            if arr.ndim == 3 and arr.shape[-1] == 4 and np.any(np.asarray(arr[..., 3], float) < 1):
                out.append(f"{_what(a)}: image with an alpha channel")
    return out


def assert_opaque(fig):
    problems = transparency_problems(fig)
    if problems:
        raise AssertionError("transparency in the figure: " + "; ".join(problems))


def text_colour_problems(fig):
    """Drawn texts that are not black."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [f"{_what(t)} is not black" for t, _ in drawn_texts(fig, renderer)
            if mcolors.to_hex(t.get_color(), keep_alpha=False) != BLACK]


# ------------------------------------------------------------------------------------------------ text placement
def _axes_name(fig, ax):
    title = ax.get_title(loc="left") or ax.get_title() or ax.get_title(loc="right")
    return f"axes {fig.axes.index(ax) + 1}" + (f" {title[:24]!r}" if title else "")


def _in_view(t, ax, axis):
    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    v = t.get_position()[0 if axis == "x" else 1]
    span = (hi - lo) or 1.0
    return lo - 1e-9 * span <= v <= hi + 1e-9 * span


def drawn_texts(fig, renderer=None):
    """(Text, where) for every visible, non-empty text the figure draws: figure texts and titles, axes titles, axis
    labels, tick labels within the view limits, offset texts, axes texts and annotations, legend texts and titles."""
    out = []

    def add(t, where):
        if t is not None and t.get_visible() and t.get_text().strip():
            out.append((t, where))

    for t in fig.texts:
        add(t, "figure")
    for name in ("_suptitle", "_supxlabel", "_supylabel"):
        add(getattr(fig, name, None), "figure")
    for leg in fig.legends:
        for t in leg.get_texts() + [leg.get_title()]:
            add(t, "figure legend")
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        w = _axes_name(fig, ax)
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None), ax.xaxis.label,
                  ax.yaxis.label, ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            add(t, w)
        for axis, labels in (("x", ax.get_xticklabels(which="both")), ("y", ax.get_yticklabels(which="both"))):
            for t in labels:
                if _in_view(t, ax, axis):
                    add(t, w)
        for t in ax.texts:
            add(t, w)
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for t in leg.get_texts() + [leg.get_title()]:
                add(t, w + " legend")
    return out


def figure_texts(fig):
    """The visible strings of a figure, in drawing order, without repeats."""
    fig.canvas.draw()
    return list(dict.fromkeys(t.get_text() for t, _ in drawn_texts(fig, fig.canvas.get_renderer())))


def _densify(xy, step):
    """The polyline resampled so that consecutive points lie at most `step` display pixels apart."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return xy
    seg = np.diff(xy, axis=0)
    k = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step).astype(int), 1)
    pts = [xy[:1]] + [a + d * (np.arange(1, n + 1)[:, None] / n) for a, d, n in zip(xy[:-1], seg, k)]
    return np.vstack(pts)


def _marks(ax, step):
    """Plotted points of an axes in display pixels, and its filled polygons (paths in display pixels)."""
    pts, polys = [], []
    for ln in ax.lines:
        if ln.get_visible():
            xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
            pts.append(_densify(xy, step) if ln.get_linestyle() not in ("None", "none", "", " ") else xy)
    for c in ax.collections:
        if not c.get_visible():
            continue
        if isinstance(c, LineCollection):
            for s in c.get_segments():
                pts.append(_densify(c.get_transform().transform(s), step))
        elif isinstance(c, PathCollection):
            offsets = c.get_offsets()
            if len(offsets):
                pts.append(c.get_offset_transform().transform(np.asarray(offsets, float)))
        elif isinstance(c, PolyCollection):
            polys += [c.get_transform().transform_path(p) for p in c.get_paths()]
    pts = np.vstack([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.empty((0, 2))
    return pts, polys


def text_collisions(fig, pad_pt=1.0, marks=True):
    """Problems of text placement: two drawn texts that overlap; a text that runs off the figure; an axes text or a
    legend that covers a plotted line, marker or filled band of its axes (marks=True; texts with gid SKIP_CHECK are
    exempt from the mark check). Returns a list of messages (empty when the figure is clear)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = pad_pt * fig.dpi / 72.0
    texts = drawn_texts(fig, renderer)
    boxes = [(t, w, t.get_window_extent(renderer)) for t, w in texts]
    out = []
    fb = fig.bbox
    for t, w, b in boxes:
        if b.x0 < fb.x0 - px or b.y0 < fb.y0 - px or b.x1 > fb.x1 + px or b.y1 > fb.y1 + px:
            out.append(f"{w} | {_what(t)} runs off the figure")
    for i, (t1, w1, b1) in enumerate(boxes):
        for t2, w2, b2 in boxes[i + 1:]:
            if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > px and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > px:
                out.append(f"{w1} | {_what(t1)} overlaps {_what(t2)} ({w2})")
    if marks:
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            pts, polys = _marks(ax, step=1.0)
            labels = [(t, t.get_window_extent(renderer)) for t in ax.texts
                      if t.get_visible() and t.get_text().strip() and t.get_gid() != SKIP_CHECK]
            leg = ax.get_legend()
            if leg is not None and leg.get_visible():
                labels.append((leg, leg.get_window_extent(renderer)))
            for t, b in labels:
                x0, y0, x1, y1 = b.x0 - px, b.y0 - px, b.x1 + px, b.y1 + px
                name = "legend" if t is leg else _what(t)
                if len(pts):
                    inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
                    if inside.any():
                        out.append(f"{_axes_name(fig, ax)} | {name} covers {int(inside.sum())} plotted point(s)")
                if polys:
                    gx, gy = np.meshgrid(np.linspace(x0, x1, 21), np.linspace(y0, y1, 9))
                    grid = np.column_stack([gx.ravel(), gy.ravel()])
                    if any(p.contains_points(grid).any() for p in polys):
                        out.append(f"{_axes_name(fig, ax)} | {name} covers a filled band")
    return out


def assert_no_collisions(fig, **kwargs):
    problems = text_collisions(fig, **kwargs)
    if problems:
        raise AssertionError("text placement: " + "; ".join(problems))


def text_outside_box(text, patch, pad_pt=1.0):
    """True when a text is not inside a box (a patch), with pad_pt of room: for flow diagrams."""
    fig = text.get_figure()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = pad_pt * fig.dpi / 72.0
    tb, pb = text.get_window_extent(r), patch.get_window_extent(r)
    return tb.x0 < pb.x0 + px or tb.x1 > pb.x1 - px or tb.y0 < pb.y0 + px or tb.y1 > pb.y1 - px


# ------------------------------------------------------------------------------------------------ main figures
def _painted(fig):
    """(artist, where) for every artist that a draw of the figure paints: what is hidden and what lies under it is left
    out, and so are the axis, spines and background of an axes switched off (set_axis_off, set_frame_on) and the ticks
    outside the view limits. Call it after a draw, which places the ticks."""
    out = []

    def walk(a, where):
        if not a.get_visible():
            return
        out.append((a, where))
        kids = list(a.get_children())
        if isinstance(a, Axes):
            where = _axes_name(fig, a) if a in fig.axes else "inset axes"
            off = [] if a.axison and a.get_frame_on() else [*a.spines.values(), a.patch]
            off += [] if a.axison else [a.xaxis, a.yaxis]
            kids = [c for c in kids if not any(c is x for x in off)]
        elif isinstance(a, Axis):
            lo, hi = sorted(a.get_view_interval())
            slack = 1e-9 * ((hi - lo) or 1.0)
            kids = [a.label, a.offsetText] + [t for t in a.get_major_ticks() + a.get_minor_ticks()
                                              if t.get_loc() is None or lo - slack <= t.get_loc() <= hi + slack]
        elif isinstance(a, Text):
            kids += [p for p in (a.get_bbox_patch(), getattr(a, "arrow_patch", None)) if p is not None]
        for c in kids:
            walk(c, where)

    walk(fig, "figure")
    return out


def _has_paint(colour):
    return bool(np.any(_rgba(colour)[:, 3] > 0))


def _line_widths(a):
    """(what, width in points) of each line an artist paints: its line, its marker edge, its outline, its hatching. A
    width of 0 or an edge without colour paints nothing (bars and bands without an outline, filled markers)."""
    out = []
    if isinstance(a, Line2D):
        if len(a.get_xydata()):
            if a.get_linestyle() not in ("None", "none", "", " ") and _has_paint(a.get_color()):
                out.append(("line", a.get_linewidth()))
            if (a.get_marker() not in ("None", "none", "", " ", None) and a.get_markersize() > 0
                    and _has_paint(a.get_markeredgecolor())):
                out.append(("marker edge", a.get_markeredgewidth()))
    elif isinstance(a, Patch):
        if a.get_linestyle() not in ("None", "none", "", " ") and _has_paint(a.get_edgecolor()):
            out.append(("outline", a.get_linewidth()))
    elif isinstance(a, Collection) and len(a.get_paths()) and _has_paint(a.get_edgecolor()):
        out += [("line", float(w)) for w in np.atleast_1d(a.get_linewidth())]
    if isinstance(a, (Patch, Collection)) and a.get_hatch():
        out.append(("hatching", a.get_hatch_linewidth()))
    return [(what, w) for what, w in out if w > 0]


def _text_style_problems(t):
    """The font and size of one text against the rules of a main figure."""
    font = font_manager.get_font(font_manager.findfont(t.get_fontproperties()))
    out = []
    if font.family_name != MAIN_FONT:
        out.append(f"font {font.family_name}, not {MAIN_FONT}")
    else:
        missing = sorted({c for c in t.get_text() if c != "\n" and font.get_char_index(ord(c)) == 0})
        if missing:
            out.append(f"{MAIN_FONT} has no {', '.join(f'U+{ord(c):04X}' for c in missing)}")
    if t.get_fontsize() < MAIN_TEXT_PT - 1e-9:
        out.append(f"{t.get_fontsize():g} pt, below {MAIN_TEXT_PT:g} pt")
    return out


def main_style_problems(fig, canvas="panel"):
    """What keeps a figure from the rules of a main figure at its native size: its canvas (MAIN_CANVAS[canvas]), the
    font and size of every visible text, and the width of every visible line (axes, ticks, curves, reference lines,
    error bars, box outlines, arrows, marker edges, hatching). A list of messages, each once (empty when the figure is
    clear)."""
    fig.canvas.draw()
    out = []
    w_mm, h_mm = (float(v) / MM for v in fig.get_size_inches())
    max_w, max_h = MAIN_CANVAS[canvas]
    if w_mm > max_w + 1e-6 or h_mm > max_h + 1e-6:
        out.append(f"canvas of {w_mm:.1f} x {h_mm:.1f} mm, more than {max_w:.0f} x {max_h:.0f} mm")
    for a, where in _painted(fig):
        if isinstance(a, Text) and a.get_text().strip():
            out += [f"{where} | {_what(a)}: {p}" for p in _text_style_problems(a)]
        out += [f"{where} | {_what(a)}: {what} of {w:g} pt, below {MAIN_LINE_PT:g} pt"
                for what, w in _line_widths(a) if w < MAIN_LINE_PT - 1e-9]
    return list(dict.fromkeys(out))


def assert_main_style(fig, canvas="panel"):
    problems = main_style_problems(fig, canvas)
    if problems:
        raise AssertionError("main figure style: " + "; ".join(problems))


# what the program of an EPS file says (matplotlib writes each operator after its operands)
_EPS_RASTER = re.compile(rb"(?<![\w/])(?:colorimage|imagemask|image)(?!\w)")
_EPS_SELECT = re.compile(rb"/(\S+) ([0-9.]+) selectfont")
_EPS_WIDTH = re.compile(rb"([0-9.]+) setlinewidth")
_EPS_BOX = re.compile(rb"%%HiResBoundingBox: (\S+) (\S+) (\S+) (\S+)")


def eps_problems(eps, canvas=None):
    """What keeps EPS data (bytes) from the format wanted: a raster image, a font of a type other than 42, or a font
    that is selected but not embedded (PostScript has no transparency: assert_opaque checks what would be flattened).
    With `canvas`, also the rules of a main figure as the file states them: the bounding box, the name and size of every
    font selected and every line width set. A list of messages."""
    code = b"\n".join(ln for ln in eps.splitlines() if not ln.lstrip().startswith(b"%"))
    out = ["a raster image"] if _EPS_RASTER.search(code) else []
    selected = set(_EPS_SELECT.findall(code))
    out += [f"a font of Type {k.decode()}" for k in sorted(set(re.findall(rb"/FontType (\d+) def", code)) - {b"42"})]
    out += [f"font {name.decode()} is not embedded"
            for name in sorted({n for n, _ in selected} - set(re.findall(rb"/FontName /(\S+) def", code)))]
    if canvas is None:
        return out
    box = _EPS_BOX.search(eps)
    left, bottom, right, top = (float(v) for v in box.groups()) if box else (0.0, 0.0, 0.0, 0.0)
    w_mm, h_mm = (right - left) / 72 * 25.4, (top - bottom) / 72 * 25.4
    max_w, max_h = MAIN_CANVAS[canvas]
    if not box or w_mm > max_w + 1e-3 or h_mm > max_h + 1e-3:
        out.append(f"bounding box of {w_mm:.1f} x {h_mm:.1f} mm, more than {max_w:.0f} x {max_h:.0f} mm"
                   if box else "no bounding box")
    for name, pt in sorted(selected):
        if not name.startswith(MAIN_FONT.encode()):
            out.append(f"font {name.decode()}, not {MAIN_FONT}")
        if float(pt) < MAIN_TEXT_PT - 1e-3:
            out.append(f"text of {float(pt):g} pt, below {MAIN_TEXT_PT:g} pt")
    thin = sorted({float(w) for w in _EPS_WIDTH.findall(code) if 0 < float(w) < MAIN_LINE_PT - 1e-3})
    out += [f"a line of {w:g} pt, below {MAIN_LINE_PT:g} pt" for w in thin]
    return out


# ------------------------------------------------------------------------------------------------ saving
def text_inventory_path(base):
    """base_text.txt: the visible strings of a figure (display.TEXT_SUFFIX; display.reset_index removes it)."""
    return base + D.TEXT_SUFFIX


def save_all(fig, base, check=True, collisions=True, marks=True, allow=()):
    """Check the figure, then write base.pdf, base.png, base.eps, base.tif and base_text.txt; returns the four image
    paths (pdf, png, eps, tif). `base` must lie in an output folder of OUT_ROOT (see display.figure_base). The checks
    (check=True): opacity, black text, text placement (collisions, marks), and display.text_problems (`allow`:
    phrases it skips) and display.forbidden_problems on every visible string; the word check of the text inventory
    (display.write_text) runs even with check=False. A figure of the main folder is saved inside a main_style() block,
    and no other figure is: main_style_problems(), transparency_problems() and eps_problems(eps, canvas) then stop it
    even with check=False, and the EPS of every figure passes eps_problems(eps). The layout of the last check draw is
    kept for the four files (layout engine 'none' from then on)."""
    from PIL import Image

    base = os.path.abspath(base)
    D.out_path(*os.path.relpath(base + ".pdf", D.OUT_ROOT).split(os.sep))   # refuses places outside the outputs
    canvas = _CANVAS[-1] if _CANVAS else None
    if (os.path.dirname(base) == os.path.abspath(D.fragment_dir("figure", "main", create=False))) != bool(canvas):
        raise AssertionError(f"{os.path.basename(base)}: a figure of the main folder is saved inside main_style(), and "
                             "no other figure is")
    texts = figure_texts(fig)
    problems = main_style_problems(fig, canvas) + transparency_problems(fig) if canvas else []
    if check:
        assert_opaque(fig)
        problems += text_colour_problems(fig)
        if collisions:
            problems += text_collisions(fig, marks=marks)
        for s in texts:
            problems += [f"{s[:40]!r}: {p}" for p in D.text_problems(s, allow) + D.forbidden_problems(s)]
    if problems:
        raise AssertionError(f"{os.path.basename(base)}: " + "; ".join(problems))
    # keep the layout of the last draw (the checked one): a constrained layout moves by a fraction of a pixel at each
    # further draw, so the four files would differ from the checked figure and from each other
    fig.set_layout_engine("none")
    paths = [base + ext for ext in D.FIGURE_TYPES]
    buf = io.BytesIO()
    fig.savefig(buf, format="eps")
    eps = b"".join(line for line in buf.getvalue().splitlines(keepends=True) if not line.startswith(b"%%CreationDate"))
    problems = eps_problems(eps, canvas)
    if problems:
        raise AssertionError(f"{os.path.basename(base)}.eps: " + "; ".join(problems))
    fig.savefig(paths[0], format="pdf", metadata={"Creator": None, "CreationDate": None})
    with open(paths[2], "wb") as fh:
        fh.write(eps)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        if im.mode == "RGBA" and im.getchannel("A").getextrema()[0] < 255:
            raise AssertionError(f"{os.path.basename(base)}: the rendering has transparent pixels")
        rgb = im.convert("RGB")
    rgb.save(paths[1], format="PNG", dpi=(DPI, DPI))
    rgb.save(paths[3], format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    D.write_text(text_inventory_path(base), "\n".join(texts))
    return tuple(paths)
