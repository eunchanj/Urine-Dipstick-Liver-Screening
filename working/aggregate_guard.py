"""Access guard for the scripts that read only aggregate tables (s08-s11); imported as their first statement.

    import aggregate_guard  # noqa: F401  (first statement of every s08-s11 script)

On import it
  (a) adds a hook with sys.addaudithook that refuses, with PermissionError, to open (built-in open, io.open, os.open,
      FileIO and the opener used by imports), list, change, copy or rename a path that falls under one of the RULES
      below, and refuses a subprocess whose arguments name such a path. The rules are decided from the text of the
      path first, so a refused path is never touched, and then again after symlinks are resolved. The macOS
      data-volume spelling of a path (/System/Volumes/Data followed by the absolute path: one file through a
      firmlink, which realpath does not resolve) is judged as the path without that prefix. The message names the
      rule only, never the path;
  (b) replaces readers that would bypass the hook (C-level file access) with functions that raise PermissionError:
      pandas.read_parquet, read_pickle, read_feather, read_orc, read_hdf, read_sas, read_spss, read_stata,
      joblib.load, pickle.load, pickle.loads, pickle.Unpickler, and the pyarrow functions and classes of
      PYARROW_READERS that open a path from C code (memory maps, OSFile, input_stream, the IPC file and stream
      readers, read_tensor, read_schema). pandas, joblib and pyarrow are not imported here: their readers are
      replaced as soon as they are imported (or at once when already imported). It also refuses, with ImportError,
      to import the modules in REFUSED_IMPORTS (paths_local, cohort, the SAS, transport, parquet and other C-level
      readers with pyarrow's private modules behind them, the learning libraries); sqlite3.connect is checked like
      an open;
  (c) records every path the process opens or tries to open (the event comes before the file system answers, so a
      missing bytecode file that an import looks for is listed too), absolute, symlinks resolved, once per path,
      with the modes seen, and writes them at exit to LOG_DIR/opened_<script>.txt (<script>: the main script's stem).

Limits (what the hook cannot see; the display scripts need none of these):
  - a relative path is judged against the working folder of the process (os.getcwd()), and a call that names a
    folder by an open file descriptor (os.open or os.listdir with dir_fd, a descriptor from os.open) is judged by the
    part it names, not by the folder the descriptor stands for;
  - a subprocess is checked only through the words of its arguments that hold a path separator or end in a refused
    file type, each judged against this process's working folder: a bare file name, a word resolved against the
    child's own working folder, the commands a shell runs and the files a child process opens are not checked (the
    child runs without this hook). Subprocesses are allowed; each is listed in
    the log by program name;
  - calls that raise no event are not seen: os.stat and os.path.exists, file access from C code that bypasses
    Python's open (extension modules, the C module _pickle), and a hard link to a refused file opened by its own
    name (only symbolic links are resolved). That is why the readers in (b) are replaced and the imports in
    REFUSED_IMPORTS refused. Other C-level readers and writers of the extension modules that take a path are not
    replaced, for example pyarrow.lib's private reader classes, CompressedInputStream and MemoryMappedFile (the
    display scripts call no pyarrow function);
  - bytecode writing is switched off only once this module has loaded, so its own first import can still write
    working/__pycache__/: run the scripts with PYTHONDONTWRITEBYTECODE=1 (or python -B).

Roots (display.py takes them from here):
  WK              the folder of this file (working/)
  ROOT            its parent, the folder above working/
  AGG_DIR         $LIV_AGG_DIR, or WK: the aggregate tables written by s00-s07 (top-level .csv, .json, .txt)
  OUT_ROOT        $LIV_OUT_ROOT, or ROOT: outputs go to its folders OUTPUT_DIRS
  LOG_DIR         $LIV_LOG_DIR; else WK/logs when OUT_ROOT is ROOT, else OUT_ROOT/logs
  PROJECT         the nearest folder above WK that holds .git (None when there is none)

Rules (first match refuses; judged with letter case ignored, as on the macOS file system, and without a leading
/System/Volumes/Data, see (a)):
  cache folder                cache/ of WK and of AGG_DIR
  learner training folder     any folder named catboost_info
  numbered project folder     any path part that starts with 00_ or 03_ (the raw-data and model folders of the authors'
                              project)
  individual-level file name  a name that starts with preds_, ends with _preds.csv, is psi_sample.csv or contains
                              _snapshot_
  individual-level file type  .parquet, .joblib, .pkl, .pickle, .sas7bdat, .xpt, .feather
  traceback file              *.err
  local path settings         a name that starts with paths_local
  project outside ROOT        any other path of PROJECT outside it (reads under a root given to allow_read() pass)
  aggregate folder write      creating, changing or removing a file directly in AGG_DIR or WK

Log format (LOG_DIR/opened_<script>.txt, UTF-8): '#' comment lines, then one line per path,
'kind<TAB>modes<TAB>path', sorted by kind (KINDS order) and path. Modes: r read, w written (or created by a rename
or copy), m renamed away, d deleted. Kinds: see KINDS. Files of the Python installation are one line per
installation folder, 'library<TAB><modes><TAB><folder> (<n> files)'. Refused attempts are lines
'refused<TAB><count><TAB><rule>' (no path); subprocesses are lines 'process<TAB>-<TAB><program name>'.
"""
import atexit
import errno
import importlib.abc
import os
import re
import site
import sys

_SEP = os.sep
OUTPUT_DIRS = ("manuscript", "figures", "supplement")
INPUT_TYPES = (".csv", ".json", ".txt")
PERSON_TYPES = (".parquet", ".joblib", ".pkl", ".pickle", ".sas7bdat", ".xpt", ".feather")
NUMBERED = ("00_", "03_")
KINDS = ("input", "output", "log", "code", "library", "system", "aggregate-folder", "project", "process", "refused")
KIND_TEXT = {
    "input": "aggregate table at the top level of AGG_DIR",
    "output": "file in an output folder of OUT_ROOT",
    "log": "file in LOG_DIR",
    "code": "Python source or bytecode outside the Python installation",
    "library": "file of the Python installation",
    "system": "other file outside the project, AGG_DIR and the output folders",
    "aggregate-folder": "other file under AGG_DIR (display scripts must not read it)",
    "project": "other file of the project (display scripts must not read it)",
    "process": "subprocess started (program name only)",
    "refused": "access refused by a rule (count and rule; the path is not written)",
}

R_CACHE = "cache folder"
R_TRAINING = "learner training folder (catboost_info)"
R_NUMBERED = "numbered project data or model folder (00_*, 03_*)"
R_NAME = "individual-level file name (preds_*, *_preds.csv, psi_sample.csv, *_snapshot_*)"
R_TYPE = "individual-level file type (parquet, joblib, pickle, SAS, transport, feather)"
R_ERR = "traceback file (*.err)"
R_PATHS = "local path settings (paths_local)"
R_PROJECT = "project file outside the folder above working/"
R_AGG_WRITE = "writing into the aggregate folder"
RULES = (R_CACHE, R_TRAINING, R_NUMBERED, R_NAME, R_TYPE, R_ERR, R_PATHS, R_PROJECT, R_AGG_WRITE)

REFUSED_IMPORTS = {
    "paths_local": "local path settings",
    "cohort": "code that reads individual-level data",
    "pyreadstat": "SAS and transport file reader",
    "pyarrow.parquet": "parquet reader",
    "pyarrow.dataset": "parquet reader",
    "pyarrow.feather": "feather reader",
    "pyarrow.orc": "ORC reader",
    "pyarrow.fs": "file access outside Python's open",
    "pyarrow.csv": "file access outside Python's open",
    "pyarrow.json": "file access outside Python's open",
    "pyarrow.substrait": "file access outside Python's open",
    # the private modules behind them (pandas loads none of these)
    "pyarrow._parquet": "parquet reader",
    "pyarrow._dataset": "parquet reader",
    "pyarrow._feather": "feather reader",
    "pyarrow._orc": "ORC reader",
    "pyarrow._fs": "file access outside Python's open",
    "pyarrow._csv": "file access outside Python's open",
    "pyarrow._json": "file access outside Python's open",
    "pyarrow._substrait": "file access outside Python's open",
    "fastparquet": "parquet reader",
    "polars": "file access outside Python's open",
    "duckdb": "file access outside Python's open",
    "h5py": "file access outside Python's open",
    "tables": "file access outside Python's open",
    "lightgbm": "learning library",
    "xgboost": "learning library",
    "catboost": "learning library",
    "shap": "learning library",
}
PANDAS_READERS = ("read_parquet", "read_pickle", "read_feather", "read_orc", "read_hdf", "read_sas", "read_spss",
                  "read_stata")
_PANDAS_HOMES = ("pandas", "pandas.io.api", "pandas.io.parquet", "pandas.io.pickle", "pandas.io.feather_format",
                 "pandas.io.orc", "pandas.io.pytables", "pandas.io.sas", "pandas.io.sas.sasreader", "pandas.io.spss",
                 "pandas.io.stata")
# pandas loads pyarrow and pyarrow.ipc itself, so the readers that open a path from C code are replaced instead of
# refused, under each name a caller can reach them by (pyarrow re-exports the IPC reader classes)
_IPC_READERS = ("RecordBatchFileReader", "RecordBatchStreamReader")
PYARROW_READERS = {"pyarrow": ("memory_map", "create_memory_map", "OSFile", "input_stream") + _IPC_READERS,
                   "pyarrow.lib": ("memory_map", "create_memory_map", "OSFile", "input_stream", "read_schema"),
                   "pyarrow.ipc": ("open_file", "open_stream", "read_tensor", "read_schema") + _IPC_READERS}


class GuardImportError(ImportError):
    """Import refused by aggregate_guard."""


# ------------------------------------------------------------------------------------------------ paths and roots
def _lexical(path):
    """Absolute, normalised text of a path argument (no file-system access); None for file descriptors."""
    if path is None:
        path = "."
    if isinstance(path, int):
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    if not os.path.isabs(text):
        try:
            text = os.path.join(os.getcwd(), text)
        except OSError:
            text = os.path.join(_SEP, text)
    return os.path.normpath(text)


DATA_VOLUME = _SEP + _SEP.join(("system", "volumes", "data"))    # macOS: /System/Volumes/Data, in lower case


def _plain(path):
    """Lower-case text of an absolute path without a leading /System/Volumes/Data: on macOS that data-volume spelling
    and the path without it name the same file (a firmlink, which realpath leaves as it is)."""
    low = path.lower()
    while low == DATA_VOLUME or low.startswith(DATA_VOLUME + _SEP):
        low = low[len(DATA_VOLUME):] or _SEP
    return low


def _under(path, root):
    """path equals root or lies below it (both absolute; compared without letter case or the data-volume prefix)."""
    if root is None:
        return False
    p, r = _plain(path), _plain(root).rstrip(_SEP)
    return p == r or p.startswith(r + _SEP)


def _project_root(start):
    d = start
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


WK = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(WK)
PROJECT = _project_root(WK)
AGG_DIR = WK
CACHE_DIRS = (os.path.join(WK, "cache"),)
_EXTRA_READ = []


def rule_for(path, writing=False, listing=False):
    """The rule (text) that refuses `path`, or None. `path` must be absolute and normalised; decided from its text
    alone (letter case and a leading /System/Volumes/Data aside, _plain). `listing` (directory listings and
    subprocess arguments) skips the project rule and the write rule."""
    low = _plain(path)
    for c in CACHE_DIRS:
        if _under(low, c):
            return R_CACHE
    parts = low.split(_SEP)
    if "catboost_info" in parts:
        return R_TRAINING
    if any(p.startswith(NUMBERED) for p in parts):
        return R_NUMBERED
    base = parts[-1]
    if base.startswith("preds_") or base.endswith("_preds.csv") or base == "psi_sample.csv" or "_snapshot_" in base:
        return R_NAME
    if base.endswith(PERSON_TYPES):
        return R_TYPE
    if base.endswith(".err"):
        return R_ERR
    if base.startswith("paths_local"):
        return R_PATHS
    if listing:
        return None
    if (PROJECT is not None and _under(low, PROJECT) and not _under(low, ROOT)
            and not (not writing and any(_under(low, r) for r in _EXTRA_READ))):
        return R_PROJECT
    if writing and os.path.dirname(low) in (_plain(AGG_DIR), _plain(WK)):
        return R_AGG_WRITE
    return None


def _root_from_env(name, default):
    """A root from the environment: refused when its text, or then its resolved form, falls under a rule."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    lex = _lexical(os.path.expanduser(value))
    rule = rule_for(lex, listing=True)
    real = None if rule else os.path.realpath(lex)
    rule = rule or rule_for(real, listing=True)
    if rule:
        raise PermissionError(errno.EACCES, f"aggregate_guard refused {name} ({rule})")
    return real


AGG_DIR = _root_from_env("LIV_AGG_DIR", WK)
CACHE_DIRS = tuple(dict.fromkeys((os.path.join(WK, "cache"), os.path.join(AGG_DIR, "cache"))))
OUT_ROOT = _root_from_env("LIV_OUT_ROOT", ROOT)
LOG_DIR = _root_from_env("LIV_LOG_DIR", os.path.join(WK, "logs") if _plain(OUT_ROOT) == _plain(ROOT)
                         else os.path.join(OUT_ROOT, "logs"))
for _name, _root in (("LIV_AGG_DIR", AGG_DIR), ("LIV_OUT_ROOT", OUT_ROOT), ("LIV_LOG_DIR", LOG_DIR)):
    _rule = rule_for(_root)
    if _rule:
        raise PermissionError(errno.EACCES, f"aggregate_guard refused {_name} ({_rule})")


def _python_roots():
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except AttributeError:
        pass
    return tuple(sorted({os.path.realpath(r) for r in roots if r}))


PYTHON_ROOTS = _python_roots()


def _script_name():
    name = os.environ.get("LIV_GUARD_NAME", "").strip()
    if not name:
        main = sys.modules.get("__main__")
        f = getattr(main, "__file__", None)
        if not f and sys.argv and sys.argv[0] not in ("", "-c", "-m"):
            f = sys.argv[0]
        name = os.path.splitext(os.path.basename(f))[0] if f else "python"
    return re.sub(r"[^\w.-]", "_", name)


SCRIPT = _script_name()


def kind_of(path):
    """Kind (KINDS) of a recorded path (absolute, resolved)."""
    low = _plain(path)
    if _under(low, LOG_DIR):
        return "log"
    if os.path.dirname(low) == _plain(AGG_DIR) and low.endswith(INPUT_TYPES):
        return "input"
    in_python = any(_under(low, r) for r in PYTHON_ROOTS)
    if low.endswith((".py", ".pyc")) and not in_python:
        return "code"
    if in_python:
        return "library"
    if any(_under(low, os.path.join(OUT_ROOT, d)) for d in OUTPUT_DIRS):
        return "output"
    if _under(low, AGG_DIR):
        return "aggregate-folder"
    if _under(low, PROJECT) or _under(low, ROOT):
        return "project"
    return "system"


def allow_read(root):
    """Let this process read files under `root` (a folder of the project outside ROOT). The other rules
    still apply; writing there stays refused."""
    lex = _lexical(root)
    rule = rule_for(lex, listing=True) or rule_for(os.path.realpath(lex), listing=True)
    if rule:
        raise PermissionError(errno.EACCES, f"aggregate_guard refused allow_read ({rule})")
    _EXTRA_READ.append(os.path.realpath(lex))


# ------------------------------------------------------------------------------------------------ (a) the hook
_OPENED = {}       # resolved path -> set of modes
_REFUSED = {}      # rule -> count
_PROCESSES = set()
_BUSY = [False]
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_LIST_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"})
_CHANGE_EVENTS = frozenset({"os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod", "os.chown", "os.utime",
                            "os.chflags", "os.lchflags", "os.setxattr", "os.removexattr", "shutil.rmtree"})
_PAIR_EVENTS = frozenset({"os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.copymode",
                          "shutil.copystat", "shutil.copytree", "shutil.move"})
_PROCESS_EVENTS = frozenset({"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.spawn",
                             "os.startfile"})
_EVENTS = frozenset({"open", "sqlite3.connect"}) | _LIST_EVENTS | _CHANGE_EVENTS | _PAIR_EVENTS | _PROCESS_EVENTS


def _refuse(rule):
    _REFUSED[rule] = _REFUSED.get(rule, 0) + 1
    raise PermissionError(errno.EACCES, f"aggregate_guard refused this access ({rule})")


def _checked(path, writing=False, listing=False):
    """Resolved path once both its text and its resolved form pass the rules; None for file descriptors."""
    lex = _lexical(path)
    if lex is None:
        return None
    rule = rule_for(lex, writing, listing)
    if rule:
        _refuse(rule)
    real = os.path.realpath(lex)
    if real != lex:
        rule = rule_for(real, writing, listing)
        if rule:
            _refuse(rule)
    return real


def _note(path, mode):
    """Record a path; pseudo-file names such as '<unknown>' (looked up when a traceback is printed) are skipped."""
    if path is not None and not re.fullmatch(r"<[^<>]*>", os.path.basename(path)):
        _OPENED.setdefault(path, set()).add(mode)


def _writing(mode, flags):
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    return bool((flags or 0) & _WRITE_FLAGS)


def _tokens(value):
    """Path-like words of a subprocess argument (a string, a path or a list of them)."""
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out += _tokens(v)
        return out
    try:
        text = os.fsdecode(os.fspath(value))
    except TypeError:
        return []
    return [w.strip("'\"") for w in text.split() if _SEP in w or w.lower().endswith(PERSON_TYPES + (".err",))]


def _program_name(program, argv):
    """Base name of the program a subprocess runs (from the executable, or the first word of its arguments)."""
    for value in (program, argv):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            words = os.fsdecode(os.fspath(value)).split()
        except TypeError:
            continue
        if words:
            return os.path.basename(words[0])
    return "?"


def _hook(event, args):
    if event not in _EVENTS or _BUSY[0]:
        return
    if event == "open":
        path, mode, flags = args
        writing = _writing(mode, flags)
        _note(_checked(path, writing), "w" if writing else "r")
    elif event == "sqlite3.connect":
        if args and args[0] not in (":memory:", "", b":memory:", b""):
            _note(_checked(args[0], writing=True), "w")
    elif event in _LIST_EVENTS:
        _checked(args[0] if args else None, listing=True)
    elif event in _CHANGE_EVENTS:
        if args and not isinstance(args[0], int):
            real = _checked(args[0], writing=True)
            if event in ("os.remove", "shutil.rmtree", "os.rmdir") and real in _OPENED:
                _note(real, "d")
    elif event in _PAIR_EVENTS:
        src, dst = args[0], args[1]
        moving = event in ("os.rename", "shutil.move")
        real_src = _checked(src, writing=moving)
        real_dst = _checked(dst, writing=True)
        if moving and real_src in _OPENED:
            _note(real_src, "m")
        if event in ("os.rename", "shutil.move", "shutil.copyfile", "shutil.copytree"):
            _note(real_dst, "w")
    else:
        if event == "subprocess.Popen":            # (executable, args, cwd, env)
            program, argv, words = args[0], args[1], [args[0], args[1], args[2]]
        elif event == "os.system":                 # (command,)
            program, argv, words = args[0], None, [args[0]]
        elif event == "os.spawn":                  # (mode, path, args, env)
            program, argv, words = args[1], args[2], [args[1], args[2]]
        else:                                      # (path, args or operation, ...)
            program, argv, words = args[0], args[1] if len(args) > 1 else None, list(args[:2])
        for w in _tokens(words):
            _checked(w, listing=True)
        _PROCESSES.add(_program_name(program, argv))


# ------------------------------------------------------------------------------------------------ (b) readers
def _refusing(name):
    def refused(*args, **kwargs):
        _REFUSED[f"reader {name}"] = _REFUSED.get(f"reader {name}", 0) + 1
        raise PermissionError(errno.EACCES, f"aggregate_guard refused {name} (reader of individual-level file types)")
    refused.__name__ = refused.__qualname__ = name.rsplit(".", 1)[-1]
    refused.refused_by_aggregate_guard = True
    return refused


def _patch_pandas(_module=None):
    for home in _PANDAS_HOMES:
        m = sys.modules.get(home)
        if m is not None:
            for name in PANDAS_READERS:
                if hasattr(m, name):
                    setattr(m, name, _refusing(f"pandas.{name}"))


def _patch_joblib(_module=None):
    for home in ("joblib", "joblib.numpy_pickle"):
        m = sys.modules.get(home)
        if m is not None and hasattr(m, "load"):
            m.load = _refusing("joblib.load")


def _patch_pyarrow(_module=None):
    for home, names in PYARROW_READERS.items():
        m = sys.modules.get(home)
        if m is not None:
            for name in names:
                if hasattr(m, name):
                    setattr(m, name, _refusing(f"{home}.{name}"))


def _patch_pickle():
    import pickle

    class RefusedUnpickler(pickle.Unpickler):
        def __init__(self, *args, **kwargs):
            _refusing("pickle.Unpickler")()

    pickle.load = _refusing("pickle.load")
    pickle.loads = _refusing("pickle.loads")
    pickle.Unpickler = RefusedUnpickler


_AFTER_IMPORT = {"pandas": _patch_pandas, "joblib": _patch_joblib, "pyarrow": _patch_pyarrow}


class _PatchingLoader(importlib.abc.Loader):
    """Runs the module's own loader, then replaces its readers."""

    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            spec.loader = self._loader
        module.__loader__ = self._loader
        self._loader.exec_module(module)
        self._patch(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _ImportGate(importlib.abc.MetaPathFinder):
    """First finder on sys.meta_path: refuses REFUSED_IMPORTS and wraps the loaders of the modules of _AFTER_IMPORT
    (pandas, joblib, pyarrow) so that their readers are replaced once the module has run."""

    def find_spec(self, fullname, path=None, target=None):
        for name, why in REFUSED_IMPORTS.items():
            if fullname == name or fullname.startswith(name + "."):
                _REFUSED[f"import {name}"] = _REFUSED.get(f"import {name}", 0) + 1
                raise GuardImportError(f"aggregate_guard refused to import {fullname} ({why})", name=fullname)
        patch = _AFTER_IMPORT.get(fullname)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PatchingLoader(spec.loader, patch)
                return spec
        return None


# ------------------------------------------------------------------------------------------------ (c) the log
def opened():
    """{resolved path: modes text} recorded so far."""
    return {p: "".join(sorted(m, key="rwmd".index)) for p, m in _OPENED.items()}


def refused():
    """{rule: count} of refused accesses, imports and reader calls so far."""
    return dict(_REFUSED)


def log_path():
    return os.path.join(LOG_DIR, f"opened_{SCRIPT}.txt")


def log_lines():
    """Lines of the log. Files of the Python installation are counted per installation folder instead of listed: their
    package-internal names would only fill the log."""
    rows, libs = [], {}
    for p, m in opened().items():
        kind = kind_of(p)
        if kind == "library":
            root = min((r for r in PYTHON_ROOTS if _under(p, r)), key=len)
            n, modes = libs.get(root, (0, ""))
            libs[root] = (n + 1, "".join(sorted(set(modes + m), key="rwmd".index)))
        else:
            rows.append((kind, m, p))
    rows += [("library", modes, f"{root} ({n:,} files)") for root, (n, modes) in libs.items()]
    rows += [("process", "-", name) for name in sorted(_PROCESSES)]
    rows += [("refused", str(n), rule) for rule, n in _REFUSED.items()]
    rows.sort(key=lambda r: (KINDS.index(r[0]), r[2]))
    head = [f"# aggregate_guard: paths opened by {SCRIPT}; one line per path: kind<TAB>modes<TAB>path",
            "# modes: r read, w written or created, m renamed away, d deleted",
            "# AGG_DIR " + AGG_DIR,
            "# OUT_ROOT " + OUT_ROOT]
    head += [f"# kind {k}: {KIND_TEXT[k]}" for k in KINDS]
    return head + ["\t".join(r) for r in rows]


def _write_log():
    _BUSY[0] = True
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        tmp = os.path.join(LOG_DIR, f".opened_{SCRIPT}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines()) + "\n")
        os.replace(tmp, log_path())
    except OSError as e:
        sys.stderr.write(f"aggregate_guard: the list of opened files was not written ({type(e).__name__})\n")
    finally:
        _BUSY[0] = False
    if _REFUSED:
        sys.stderr.write(f"aggregate_guard: {sum(_REFUSED.values())} access(es) refused: "
                         + "; ".join(sorted(_REFUSED)) + "\n")


def _install():
    if getattr(sys, "_aggregate_guard_installed", False):
        return
    sys._aggregate_guard_installed = True
    sys.dont_write_bytecode = True      # no __pycache__ writes next to the code from here on
    _patch_pickle()
    _patch_pandas()
    _patch_joblib()
    _patch_pyarrow()
    sys.meta_path.insert(0, _ImportGate())
    sys.addaudithook(_hook)
    atexit.register(_write_log)


_install()
