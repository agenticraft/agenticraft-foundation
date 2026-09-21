#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every Python file states the licence it is published under.

The packaging metadata declares one licence for the whole distribution.
That is enough for an installer and not enough for anything that reads a
file on its own: an SBOM generator, a vendoring tool, a reviewer looking
at a single module in a code search. The convention that answers them is
one comment line at the top of the file naming the same SPDX identifier
the metadata declares.

The identifier is read from ``pyproject.toml`` rather than written into
this script, so the header and the declared licence cannot drift apart:
relicensing the project is one edit to the metadata, and this gate then
reports every file still carrying the old identifier.

Rules
-----

``SPDX001`` a Python file with no SPDX identifier.
``SPDX002`` an identifier that is not the file's first comment line (a
shebang and an encoding declaration may precede it, and nothing else
may).
``SPDX003`` an identifier naming a licence other than the declared one.
``SPDX004`` a file whose bytes are not UTF-8, so no identifier can be
read from it at all.

Usage
-----

::

    python3 scripts/check_spdx_headers.py              # judge the tree
    python3 scripts/check_spdx_headers.py FILE...      # judge these files
    python3 scripts/check_spdx_headers.py --fix        # stamp what SPDX001 names
    python3 scripts/check_spdx_headers.py --self-test  # prove the rules can fail

``--fix`` only adds a header that is absent. A misplaced or mismatched
identifier is someone's decision to revisit, not a line to rewrite
unasked, so SPDX002 and SPDX003 are reported either way.

Exit codes: ``0`` clean, ``1`` violation, ``2`` no measurement could be
taken: the self-test failed, the tree declares no licence to compare
against, or the run had no Python file to judge. A gate that read nothing
reports that rather than certifying a tree it never opened.
"""

from __future__ import annotations

import argparse
import codecs
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Directories a tool owns: its cache, its metadata. The name belongs to
#: the tool that writes it, so no contributor gives a source directory one
#: of them and they are pruned wherever they sit. This is the whole of what
#: the walk skips on a name; everything else it skips on evidence.
TOOL_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
    }
)

#: Directories the build backend and the docs builder write into. Unlike
#: the names above, these are names a contributor may give a package or a
#: test fixture, so they are pruned only at the root, where the tools that
#: write them work, and judged as source anywhere below it.
OUTPUT_DIRS = frozenset({"build", "dist", "site"})

#: What makes a directory a virtual environment whatever it is called: the
#: interpreter writes this file when it creates one. ``venv`` and ``.venv``
#: are conventions a contributor may borrow for a package or a fixture;
#: this file is the fact, so the walk prunes on it instead.
ENVIRONMENT_MARKER = "pyvenv.cfg"

#: A dependency tree a package manager installed, and the manifest that
#: sits beside the one it installed for. A directory a contributor happens
#: to name ``node_modules`` has no manifest beside it and is judged.
DEPENDENCY_DIR = "node_modules"
PACKAGE_MANIFEST = "package.json"

#: The suffix of a file this gate judges.
SOURCE_SUFFIX = ".py"

MARKER = "SPDX-License-Identifier"

#: A comment line that sets out to be the header: it starts the line and
#: the marker opens it. The rule is textual because its readers are -- an
#: SBOM generator scrapes this line rather than parsing the file -- so a
#: line a scraper would read as a header is judged as one wherever it
#: sits, inside a docstring included. Prose that names the convention
#: without standing at the head of a line is not an attempt at a header: a
#: docstring or a comment carrying the marker mid-line, a note indented
#: inside a function. Neither is diagnosed as a misplaced header, and
#: ``--fix`` stamps the file above it.
_HEADER_ATTEMPT = re.compile(rf"^#\s*{MARKER}\b")

#: A line boundary as the interpreter reads one. ``str.splitlines`` also
#: breaks on characters the tokenizer does not treat as boundaries -- a
#: form feed, which source carries as a page separator -- so counting lines
#: with it would report a line number the file does not have.
_LINE_END = re.compile(r"\r\n|\r|\n")

#: An encoding declaration as PEP 263 spells it. The interpreter reads it
#: on the first line, or the second when a shebang takes the first, and
#: nowhere else, so the header goes below it rather than through it.
_CODING_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-_.a-zA-Z0-9]+")

#: The one accepted spelling of the header line.
_HEADER_LINE = re.compile(rf"^#\s*{MARKER}:\s*(?P<identifier>\S+)\s*$")

#: ``license = "Apache-2.0"`` in the ``[project]`` table. Scoped to that
#: table so a licence named anywhere else in the file -- a tool's own
#: configuration, a classifier list -- is not mistaken for the
#: declaration. Both table forms are recognised, since an array of tables
#: opens a scope of its own and a licence inside one is not the
#: project's.
_COMMENT = r"\s*(#.*)?$"
_TABLE = re.compile(rf"^\[(?P<name>[^\[\]]+)\]{_COMMENT}")
_ARRAY_TABLE = re.compile(rf"^\[\[(?P<name>[^\[\]]+)\]\]{_COMMENT}")
_LICENCE = re.compile(rf"^license\s*=\s*[\"'](?P<identifier>[^\"']+)[\"']{_COMMENT}")


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str

    def render(self) -> str:
        return f"{self.rule} {self.path}: {self.detail}"


# ---------------------------------------------------------------------------
# Pure functions -- the self-test drives these before any real file is read
# ---------------------------------------------------------------------------


def declared_identifier(pyproject_text: str) -> str | None:
    """The SPDX identifier the packaging metadata declares, if it does."""
    table = ""
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        header = _TABLE.match(stripped) or _ARRAY_TABLE.match(stripped)
        if header:
            table = header.group("name").strip()
            continue
        if table != "project":
            continue
        licence = _LICENCE.match(stripped)
        if licence:
            return licence.group("identifier").strip()
    return None


def decode_source(data: bytes) -> str | None:
    """The file's text, or ``None`` when the bytes are not UTF-8.

    A byte-order mark is an encoding artefact rather than source, so it is
    read off here and the rules see the text below it; ``--fix`` writes
    the mark back as it found it.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def undecodable(path: str) -> Violation:
    """The verdict on a file whose bytes the gate cannot read as UTF-8."""
    return Violation("SPDX004", path, "is not valid UTF-8, so it states no identifier")


def source_lines(text: str) -> list[str]:
    """The file's lines as the interpreter counts them."""
    return _LINE_END.split(text)


def line_ending(text: str) -> str:
    """The terminator this file already uses, so ``--fix`` never mixes two."""
    match = _LINE_END.search(text)
    return match.group(0) if match else "\n"


def header_position(text: str) -> int:
    """The line index the identifier has to occupy in this file.

    A shebang has to stay on the first line for the kernel to read it, and
    an encoding declaration is only honoured on the first line or the
    second, so a file carrying either keeps it where the interpreter looks
    and takes the header on the line below.
    """
    lines = source_lines(text)
    position = 1 if text.startswith("#!") else 0
    if position < len(lines) and _CODING_COOKIE.match(lines[position]):
        position += 1
    return position


def header_violation(path: str, text: str, expected: str) -> Violation | None:
    """The rule this file breaks, or ``None`` when it carries the header."""
    lines = source_lines(text)
    found = [index for index, line in enumerate(lines) if _HEADER_ATTEMPT.match(line)]
    if not found:
        return Violation("SPDX001", path, f"no {MARKER}: {expected} header")

    index = found[0]
    position = header_position(text)
    if index != position:
        return Violation(
            "SPDX002",
            path,
            f"the identifier is on line {index + 1}; it has to be the first "
            f"comment line, on line {position + 1}",
        )

    match = _HEADER_LINE.match(lines[index])
    if match is None:
        return Violation(
            "SPDX002",
            path,
            f"line {index + 1} is not a header comment: {lines[index]!r}",
        )
    identifier = match.group("identifier")
    if identifier != expected:
        return Violation(
            "SPDX003",
            path,
            f"declares {identifier}; this project is published under {expected}",
        )
    return None


def stamp(text: str, expected: str) -> str:
    """The same file with the header inserted on the line it belongs on."""
    ending = line_ending(text)
    header = f"# {MARKER}: {expected}{ending}"
    offset = 0
    for _ in range(header_position(text)):
        boundary = _LINE_END.search(text, offset)
        if boundary is None:
            return f"{text}{ending}{header}"
        offset = boundary.end()
    return f"{text[:offset]}{header}{text[offset:]}"


def is_pruned(name: str, directory: Path, root: Path) -> bool:
    """Whether the walk skips the subdirectory ``name`` found in ``directory``.

    Three kinds of directory hold no source of this project, and each is
    recognised by what it is rather than by what someone might have called
    it. A tool's own cache and metadata carry a name that tool owns, so
    they are skipped wherever they sit -- and they are the only thing a
    name alone decides. A virtual environment is a directory holding
    ``pyvenv.cfg``, whatever it is named, and a dependency tree is a
    ``node_modules`` beside the manifest it was installed for. Build output
    is skipped only at the root, where the build backend and the docs
    builder write it.

    So a package or a fixture a contributor names ``venv``,
    ``node_modules``, ``build`` or ``site`` carries none of that evidence
    and is judged like any other source: no Python directory escapes this
    gate by its name alone.
    """
    if name in TOOL_DIRS:
        return True
    if (directory / name / ENVIRONMENT_MARKER).is_file():
        return True
    if name == DEPENDENCY_DIR and (directory / PACKAGE_MANIFEST).is_file():
        return True
    return name in OUTPUT_DIRS and directory == root


def is_source(path: Path) -> bool:
    """Whether this file is Python source the gate judges."""
    return path.suffix == SOURCE_SUFFIX


def self_test() -> list[str]:
    """Prove each rule fails on a file that violates it, and only then.

    The rules are proved against synthetic text, and the walk -- whose
    prune rules read the filesystem -- against a synthetic tree built in a
    temporary directory. Both run before any file of this project is read.
    """
    problems: list[str] = []
    expected = "Apache-2.0"
    header = f"# {MARKER}: {expected}\n"
    mention = f'"""Prose naming {MARKER} for a reader."""\n'
    commented = (
        f'"""A module nobody stamped."""\n\n\ndef read() -> None:\n'
        f"    # we rely on {MARKER} headers\n"
        f"    return None\n"
    )
    indented = (
        f'"""A module nobody stamped."""\n\n\ndef read() -> None:\n'
        f"    # {MARKER}: {expected}\n"
        f"    return None\n"
    )
    cookie = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Runnable."""\n'
    windows = '#!/usr/bin/env python3\r\n"""Runnable."""\r\n'

    cases = {
        "SPDX001": '"""A module nobody stamped."""\n',
        "SPDX002": f'"""A module stamped below its docstring."""\n\n{header}',
        "SPDX003": f"# {MARKER}: BUSL-1.1\n",
    }
    for rule, text in cases.items():
        violation = header_violation("synthetic.py", text, expected)
        if violation is None or violation.rule != rule:
            problems.append(f"{rule} did not fire on a file that violates it: {text!r}")

    if decode_source(b"\xff\xfe" + mention.encode("utf-8")) is not None:
        problems.append("bytes that are not UTF-8 decoded as if they were, so SPDX004 cannot fire")
    if undecodable("synthetic.py").rule != "SPDX004":
        problems.append("the verdict on a file that is not UTF-8 is not SPDX004")
    if decode_source(header.encode("utf-8-sig")) != header:
        problems.append("a byte-order mark was read as part of the source below it")

    for description, text in {
        "prose naming the identifier": mention,
        "a comment naming the identifier in passing": commented,
        "a header line indented inside a function": indented,
    }.items():
        read_as = header_violation("synthetic.py", text, expected)
        if read_as is None or read_as.rule != "SPDX001":
            problems.append(f"{description} was read as a header rather than as prose")

    for description, text in {
        "a stamped module": f'{header}"""Stamped."""\n',
        "a stamped script": f'#!/usr/bin/env python3\n{header}"""Stamped."""\n',
        "a stamped empty file": header,
        "a stamped module below an encoding declaration": (
            f'# -*- coding: utf-8 -*-\n{header}"""Stamped."""\n'
        ),
    }.items():
        violation = header_violation("synthetic.py", text, expected)
        if violation is not None:
            problems.append(f"{description} was reported: {violation.render()}")

    for description, text in {
        "an empty file": "",
        "a script": '#!/usr/bin/env python3\n"""Runnable."""\n',
        "a script whose last line has no terminator": "#!/usr/bin/env python3",
        "a module whose prose names the identifier": mention,
        "a module whose comment names the identifier": commented,
        "a module whose indented comment spells the header": indented,
        "a script with an encoding declaration": cookie,
        "a file with CRLF line endings": windows,
    }.items():
        stamped = stamp(text, expected)
        violation = header_violation("synthetic.py", stamped, expected)
        if violation is not None:
            problems.append(f"stamping {description} left it in violation: {violation.render()}")

    if source_lines(stamp(cookie, expected))[1] != "# -*- coding: utf-8 -*-":
        problems.append(
            "stamping moved the encoding declaration off the lines the interpreter reads it on"
        )
    if "\n" in stamp(windows, expected).replace("\r\n", ""):
        problems.append("stamping a file with CRLF line endings mixed a bare LF into it")

    if not is_source(Path("module.py")) or is_source(Path("pyproject.toml")):
        problems.append(f"the gate does not judge exactly the {SOURCE_SUFFIX} files")

    metadata = (
        "[build-system]\n"
        'license = "MIT"\n'
        "\n"
        "[project]  # the distribution\n"
        'name = "synthetic"\n'
        f'license = "{expected}"  # the licence it is published under\n'
    )
    declared = declared_identifier(metadata)
    if declared != expected:
        problems.append(
            f"the declared licence read as {declared!r}; the [project] table "
            f"says {expected!r} and another table's key is a decoy"
        )
    if declared_identifier('[project]\nname = "synthetic"\n') is not None:
        problems.append("a project declaring no licence read as declaring one")
    vendored = '[project]\nname = "synthetic"\n\n[[tool.vendor.items]]\nlicense = "MIT"\n'
    if declared_identifier(vendored) is not None:
        problems.append("a licence in an array of tables read as the project's own declaration")

    with tempfile.TemporaryDirectory(prefix="check-spdx-headers-") as scratch:
        problems.extend(walk_self_test(Path(scratch)))

    return problems


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def python_files(root: Path) -> list[Path]:
    """Every Python file under ``root`` that belongs to this project."""
    found: list[Path] = []
    for directory, subdirectories, names in os.walk(root):
        here = Path(directory)
        subdirectories[:] = sorted(
            name for name in subdirectories if not is_pruned(name, here, root)
        )
        found.extend(sorted(here / name for name in names if is_source(here / name)))
    return found


def relative_name(path: Path, root: Path) -> str:
    """How a path is named in a report: relative to the root when it lies there."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def select(root: Path, paths: list[Path]) -> tuple[list[Path], str | None]:
    """The files this run judges, or the reason it can judge none.

    A run that judges nothing is not a clean run: reporting success over an
    empty selection certifies a tree the gate never opened. Every way of
    arriving at nothing -- a path naming no file, whether it is absent or a
    directory wearing the source suffix; a path that is not Python source;
    a tree holding no Python file -- is a measurement that could not be
    taken rather than a verdict, so each is named here and none of them
    reaches the report.
    """
    if not paths:
        walked = python_files(root)
        if not walked:
            return [], f"no {SOURCE_SUFFIX} file under {root}"
        return walked, None

    absent = [path for path in paths if not path.is_file()]
    if absent:
        return [], "no file at " + ", ".join(str(path) for path in absent)

    given = [path.resolve() for path in paths if is_source(path)]
    if not given:
        named = ", ".join(str(path) for path in paths)
        return [], f"no {SOURCE_SUFFIX} file among the paths given: {named}"
    return given, None


def walk_self_test(root: Path) -> list[str]:
    """Prove the walk reaches every source directory and prunes only the rest.

    Two prune rules read the filesystem -- an environment is a directory
    holding ``pyvenv.cfg``, a dependency tree is a ``node_modules`` beside
    a manifest -- and the promise that no name alone hides source is a
    claim about a tree, not about a string. So this builds one in a
    temporary directory, walks it, and throws it away.
    """
    problems: list[str] = []
    judged = {
        Path("module.py"),
        Path("src") / "pkg" / "venv" / "module.py",
        Path("src") / "pkg" / DEPENDENCY_DIR / "module.py",
        Path("src") / "build" / "module.py",
        Path("tests") / "fixtures" / "venv" / "module.py",
        Path(".github") / "scripts" / "module.py",
    }
    pruned = {
        Path(".venv") / "lib" / "module.py",
        Path("env") / "lib" / "module.py",
        Path("app") / DEPENDENCY_DIR / "dependency" / "module.py",
        Path("__pycache__") / "module.py",
        Path(".mypy_cache") / "module.py",
        Path("build") / "module.py",
    }
    for relative in sorted(judged | pruned):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('"""Synthetic."""\n', encoding="utf-8")
    for environment in (".venv", "env"):
        marker = root / environment / ENVIRONMENT_MARKER
        marker.write_text("home = /synthetic\n", encoding="utf-8")
    (root / "app" / PACKAGE_MANIFEST).write_text('{"name": "synthetic"}\n', encoding="utf-8")
    empty = root / "empty"
    empty.mkdir()
    directory_named_like_source = root / f"package{SOURCE_SUFFIX}"
    directory_named_like_source.mkdir()

    walked = {path.relative_to(root) for path in python_files(root)}
    for relative in sorted(judged - walked):
        problems.append(
            f"{relative} escaped the walk; a name a contributor may choose is not a prune"
        )
    for relative in sorted(walked & pruned):
        problems.append(f"{relative} was walked into; it is generated or vendored, not source")

    for description, selection in {
        "a tree holding source": select(root, []),
        "a Python file named on the command line": select(root, [root / "module.py"]),
    }.items():
        if selection[1] is not None:
            problems.append(f"{description} read as nothing to judge: {selection[1]}")
    for description, selection in {
        "a tree holding no Python file": select(empty, []),
        "a path that is not there": select(root, [root / "absent.py"]),
        "a directory named instead of a file": select(root, [root / "src"]),
        "a directory wearing the source suffix": select(root, [directory_named_like_source]),
    }.items():
        if selection[1] is None:
            problems.append(f"{description} read as a measurement the gate could take")
    return problems


def evaluate(
    root: Path, files: list[Path], expected: str, *, fix: bool
) -> tuple[list[Violation], int]:
    """Judge every file given, stamping the ones SPDX001 names when asked to."""
    violations: list[Violation] = []
    stamped = 0
    for path in files:
        name = relative_name(path, root)
        data = path.read_bytes()
        text = decode_source(data)
        if text is None:
            violations.append(undecodable(name))
            continue
        violation = header_violation(name, text, expected)
        if violation is None:
            continue
        if fix and violation.rule == "SPDX001":
            mark = codecs.BOM_UTF8 if data.startswith(codecs.BOM_UTF8) else b""
            path.write_bytes(mark + stamp(text, expected).encode("utf-8"))
            stamped += 1
            continue
        violations.append(violation)
    return violations, stamped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Every Python file states the licence it is published under."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "the files to judge (default: every Python file under --root); "
            "a path that is not Python source is skipped as long as the run "
            "also names source, and a run handed no source at all reports "
            "that it took no measurement rather than a clean tree"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="the project to judge (default: the repository this script lives in)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="stamp the files that carry no header",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run only the self-test and report whether the rules can fail",
    )
    args = parser.parse_args(argv)

    problems = self_test()
    if problems:
        for problem in problems:
            print(f"SELF-TEST FAILURE: {problem}", file=sys.stderr)
        print(
            "the SPDX header gate cannot prove its own rules fail; refusing to certify anything",
            file=sys.stderr,
        )
        return 2
    if args.self_test:
        print("check_spdx_headers: self-test passed -- every rule can fail.")
        return 0

    root = args.root.resolve()
    metadata = root / "pyproject.toml"
    if not metadata.is_file():
        print(
            f"SPDX HEADER GATE FAILURE: {metadata} does not exist, so there is "
            f"no declared licence to compare the headers against.",
            file=sys.stderr,
        )
        return 2
    expected = declared_identifier(metadata.read_text(encoding="utf-8"))
    if expected is None:
        print(
            f"SPDX HEADER GATE FAILURE: {metadata} names no license in its "
            f"[project] table, so there is no identifier the headers can be "
            f"held to.",
            file=sys.stderr,
        )
        return 2

    files, unmeasurable = select(root, args.paths)
    if unmeasurable is not None:
        print(
            f"SPDX HEADER GATE FAILURE: {unmeasurable}, so nothing was judged.",
            file=sys.stderr,
        )
        return 2
    violations, stamped = evaluate(root, files, expected, fix=args.fix)
    if stamped:
        print(f"check_spdx_headers: stamped {stamped} file(s) with {expected}.")
    if violations:
        print(
            f"check_spdx_headers: {len(violations)} file(s) do not carry the "
            f"{expected} identifier this project declares:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation.render()}", file=sys.stderr)
        return 1

    print(f"check_spdx_headers: {len(files)} Python file(s) carry {expected}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
