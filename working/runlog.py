"""Run wrapper and output helpers for the scripts that read individual-level data (s00-s07).

These scripts are run by the data holder in a terminal, with standard output sent to logs/ and standard error to
cache/:

    python sNN_name.py > logs/sNN_name.out 2> cache/sNN_name.err

- run() writes logs/<script>.status: "OK", or "FAIL <exception type> <file>:<line>" with no message text. The
  exception is re-raised, so its traceback goes to standard error (cache/), which stays on your computer.
- Printed lines and assertion messages carry counts only, never data values.
- mask_count() and mask_counts() show counts of 1-9 as "<10" in every file written outside cache/.
"""
import os
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
CACHE = os.path.join(HERE, "cache")
SMALL = 10
# Column names that would identify a person or a visit; aggregate outputs must not carry them.
ID_LIKE = {"id", "일련번호", "소변검사일자", "소변검사일시", "date", "dt", "수진일자", "검사일자"}


def run(main, script_file):
    """Call main(); record OK or the exception type and the innermost line of this folder's code."""
    name = os.path.splitext(os.path.basename(script_file))[0]
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    status = os.path.join(LOGS, f"{name}.status")
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - the type is recorded, then re-raised
        frames = traceback.extract_tb(e.__traceback__)
        own = [f for f in frames if os.path.abspath(f.filename).startswith(HERE)]
        fr = own[-1] if own else (frames[-1] if frames else None)
        where = f"{os.path.basename(fr.filename)}:{fr.lineno}" if fr else "?"
        with open(status, "w") as fh:
            fh.write(f"FAIL {type(e).__name__} {where}\n")
        raise
    with open(status, "w") as fh:
        fh.write("OK\n")


def mask_count(n):
    """A count for an output file: 1-9 -> '<10'; 0 and counts of 10 or more unchanged (as int)."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return n
    n = int(n)
    return "<10" if 0 < n < SMALL else n


def count_text(n):
    """A count for a printed line or an assertion message: '<10' for 1-9, else with thousands separators."""
    m = mask_count(n)
    return m if isinstance(m, str) else f"{m:,}"


def mask_counts(df, cols):
    """Copy of df with the count columns cols passed through mask_count."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(mask_count).astype(object)
    return out


def write_aggregate(df, name, count_cols=()):
    """Write an aggregate table to the working folder (top level). Refuses identifier or date columns; masks the
    listed count columns."""
    bad = [c for c in df.columns if str(c).strip().lower() in ID_LIKE]
    assert not bad, f"aggregate output has {len(bad)} identifier or date column(s)"
    out = mask_counts(df, count_cols) if count_cols else df
    path = os.path.join(HERE, name)
    out.to_csv(path, index=False)
    return path


def write_table(df, name, count_cols=()):
    """write_aggregate, with the counts of a count column that also has blank cells kept as whole numbers."""
    out = mask_counts(df, count_cols)
    for c in count_cols:
        out[c] = pd.Series([v if isinstance(v, str) or v is None or pd.isna(v) else int(v) for v in out[c]],
                           index=out.index, dtype=object)
    return write_aggregate(out, name)


def say(*parts):
    """Print a progress line (counts and labels only)."""
    print(*parts, flush=True)


def as_frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)
