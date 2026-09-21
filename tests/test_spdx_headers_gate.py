# SPDX-License-Identifier: Apache-2.0
"""The licence-header gate: what it accepts, what it refuses, what it stamps.

The gate is a script rather than a module, so it is driven here the way CI
drives it -- as a subprocess whose exit code is the verdict.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = REPO_ROOT / "scripts" / "check_spdx_headers.py"

#: The smallest project metadata the gate needs: it reads the identifier it
#: expects from the licence the packaging metadata declares, so the header
#: and the metadata cannot drift apart.
PYPROJECT = '[project]\nname = "synthetic"\nlicense = "Apache-2.0"\n'

HEADER = "# SPDX-License-Identifier: Apache-2.0\n"


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the gate as CI runs it and return the completed process."""
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def synthetic_tree(root: Path, module: str) -> Path:
    """A one-module project whose licence metadata the gate can read."""
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    source = root / "module.py"
    source.write_text(module, encoding="utf-8")
    return source


def test_spdx_gate_self_test_proves_every_rule_can_fail() -> None:
    result = run_gate("--self-test")
    assert result.returncode == 0, result.stderr


def test_spdx_gate_flags_a_file_without_the_identifier(tmp_path: Path) -> None:
    synthetic_tree(tmp_path, '"""No header at all."""\n')

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1
    assert "SPDX001" in result.stderr
    assert "module.py" in result.stderr


def test_spdx_gate_flags_an_identifier_that_is_not_the_first_comment(
    tmp_path: Path,
) -> None:
    synthetic_tree(tmp_path, f'"""Docstring first."""\n\n{HEADER}')

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1
    assert "SPDX002" in result.stderr


def test_spdx_gate_flags_an_identifier_for_another_licence(tmp_path: Path) -> None:
    synthetic_tree(tmp_path, "# SPDX-License-Identifier: BUSL-1.1\n")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1
    assert "SPDX003" in result.stderr


def test_spdx_gate_accepts_a_stamped_file(tmp_path: Path) -> None:
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_spdx_gate_refuses_a_tree_that_declares_no_licence(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 2
    assert "license" in result.stderr


def test_spdx_gate_fix_stamps_a_file_below_its_shebang(tmp_path: Path) -> None:
    source = synthetic_tree(tmp_path, '#!/usr/bin/env python3\n"""Runnable."""\n')

    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0

    assert source.read_text(encoding="utf-8").splitlines()[:2] == [
        "#!/usr/bin/env python3",
        "# SPDX-License-Identifier: Apache-2.0",
    ]
    assert run_gate("--root", str(tmp_path)).returncode == 0


def test_spdx_gate_accepts_this_repository() -> None:
    result = run_gate()

    assert result.returncode == 0, result.stderr


BUILD_OUTPUT_NAMES = ("build", "dist", "site")


def test_spdx_gate_judges_a_package_named_like_build_output(tmp_path: Path) -> None:
    """``build``, ``dist`` and ``site`` are ordinary Python identifiers.

    A package or fixture carrying one of those names below the root is
    source, and the gate that claims no Python directory can escape it has
    to judge it.
    """
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    for directory in BUILD_OUTPUT_NAMES:
        package = tmp_path / "src" / "pkg" / directory
        package.mkdir(parents=True)
        (package / "module.py").write_text('"""Unstamped."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1, result.stdout
    for directory in BUILD_OUTPUT_NAMES:
        assert str(Path("src") / "pkg" / directory / "module.py") in result.stderr


def test_spdx_gate_skips_build_output_at_the_root(tmp_path: Path) -> None:
    """The root is where the build backend and mkdocs write, so it is pruned."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    for directory in BUILD_OUTPUT_NAMES:
        output = tmp_path / directory / "lib"
        output.mkdir(parents=True)
        (output / "copied.py").write_text('"""Build output."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 0, result.stderr


TOOL_CACHE_NAMES = (".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "__pycache__")

ENVIRONMENT_NAMES = (".venv", "venv", "env")


def test_spdx_gate_skips_tool_caches_at_any_depth(tmp_path: Path) -> None:
    """A tool owns these names and writes what is under them; none of it is source."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    for directory in TOOL_CACHE_NAMES:
        cache = tmp_path / "src" / "pkg" / directory / "inner"
        cache.mkdir(parents=True)
        (cache / "generated.py").write_text('"""Not ours."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_spdx_gate_skips_a_virtual_environment_whatever_it_is_called(tmp_path: Path) -> None:
    """An environment is pruned on the file that makes it one, not on its name."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    for directory in ENVIRONMENT_NAMES:
        environment = tmp_path / "src" / "pkg" / directory
        (environment / "lib").mkdir(parents=True)
        (environment / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
        (environment / "lib" / "installed.py").write_text('"""Not ours."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_spdx_gate_judges_a_package_named_like_an_environment(tmp_path: Path) -> None:
    """``venv`` and ``node_modules`` are names a contributor may give a directory.

    Neither name is owned by a tool the way a cache name is, so a gate that
    claims no Python directory escapes it by name alone has to judge them.
    """
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    unstamped = (
        Path("src") / "pkg" / "venv" / "hidden.py",
        Path("src") / "pkg" / "node_modules" / "hidden.py",
        Path("tests") / "fixtures" / "venv" / "case.py",
    )
    for relative in unstamped:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('"""Unstamped."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1, result.stdout
    for relative in unstamped:
        assert str(relative) in result.stderr


def test_spdx_gate_skips_a_dependency_tree_beside_its_manifest(tmp_path: Path) -> None:
    """``node_modules`` is a dependency tree where a manifest says it is."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    application = tmp_path / "app"
    vendored = application / "node_modules" / "dependency"
    vendored.mkdir(parents=True)
    (application / "package.json").write_text('{"name": "app"}\n', encoding="utf-8")
    (vendored / "helper.py").write_text('"""Not ours."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_spdx_gate_fix_stamps_a_file_that_only_mentions_the_marker(tmp_path: Path) -> None:
    """Prose naming the convention is not a header and must not read as one."""
    source = synthetic_tree(tmp_path, '"""Docs mention SPDX-License-Identifier for readers."""\n')

    assert run_gate("--root", str(tmp_path)).returncode == 1
    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0

    assert source.read_text(encoding="utf-8").splitlines()[0] == HEADER.rstrip("\n")


def test_spdx_gate_reads_a_licence_declared_beside_a_comment(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]  # the distribution\nname = "synthetic"\n'
        'license = "Apache-2.0"  # the one this project is published under\n',
        encoding="utf-8",
    )
    (tmp_path / "module.py").write_text(f'{HEADER}"""Stamped."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 0, result.stderr


def test_spdx_gate_ignores_a_licence_in_an_array_of_tables(tmp_path: Path) -> None:
    """A licence under ``[[tool.x.items]]`` is not the project's declaration."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "synthetic"\n\n[[tool.vendor.items]]\nlicense = "MIT"\n',
        encoding="utf-8",
    )
    (tmp_path / "module.py").write_text(f'{HEADER}"""Stamped."""\n', encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 2
    assert "license" in result.stderr


def test_spdx_gate_reports_a_file_it_cannot_decode(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (tmp_path / "module.py").write_bytes(b"# SPDX-License-Identifier: Apache-2.0\n# \xff\xfe\n")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1
    assert "SPDX004" in result.stderr
    assert "Traceback" not in result.stderr


def test_spdx_gate_fix_stamps_below_a_byte_order_mark(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    source = tmp_path / "module.py"
    source.write_bytes(b"\xef\xbb\xbf#!/usr/bin/env python3\n" + b'"""Runnable."""\n')

    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0

    written = source.read_bytes()
    assert written.startswith(b"\xef\xbb\xbf#!/usr/bin/env python3\n")
    assert written.decode("utf-8-sig").splitlines()[1] == HEADER.rstrip("\n")
    assert run_gate("--root", str(tmp_path)).returncode == 0


def test_spdx_gate_judges_only_the_paths_it_is_given(tmp_path: Path) -> None:
    """The commit hook passes the staged files, so untracked work is not judged."""
    stamped = synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    scratch = tmp_path / "scratch.py"
    scratch.write_text('"""Untracked."""\n', encoding="utf-8")

    assert run_gate("--root", str(tmp_path), str(stamped)).returncode == 0
    assert run_gate("--root", str(tmp_path)).returncode == 1

    flagged = run_gate("--root", str(tmp_path), str(scratch))
    assert flagged.returncode == 1
    assert "scratch.py" in flagged.stderr


def test_spdx_gate_refuses_a_run_that_judged_no_file(tmp_path: Path) -> None:
    """A gate that read nothing says so; a clean verdict would certify nothing."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    package = tmp_path / "src"
    package.mkdir()
    (package / "module.py").write_text(f'{HEADER}"""Stamped."""\n', encoding="utf-8")

    for path in (package, tmp_path / "pyproject.toml"):
        result = run_gate("--root", str(tmp_path), str(path))

        assert result.returncode == 2, result.stdout
        assert "nothing was judged" in result.stderr


def test_spdx_gate_refuses_a_tree_holding_no_python_file(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 2
    assert "nothing was judged" in result.stderr


def test_spdx_gate_reports_a_path_that_is_not_there(tmp_path: Path) -> None:
    """A mistyped path is a verdict the gate states, not a stack trace."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')

    result = run_gate("--root", str(tmp_path), str(tmp_path / "gone.py"))

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "gone.py" in result.stderr


def test_spdx_gate_fix_stamps_a_file_whose_comment_mentions_the_marker(tmp_path: Path) -> None:
    """A comment about the convention is prose, not an identifier out of place."""
    source = synthetic_tree(
        tmp_path,
        '"""Unstamped."""\n\n\ndef read() -> None:\n'
        "    # we rely on SPDX-License-Identifier headers\n"
        "    return None\n",
    )

    first = run_gate("--root", str(tmp_path))
    assert first.returncode == 1
    assert "SPDX001" in first.stderr

    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0
    assert source.read_text(encoding="utf-8").splitlines()[0] == HEADER.rstrip("\n")


def test_spdx_gate_fix_stamps_a_file_whose_indented_comment_spells_the_header(
    tmp_path: Path,
) -> None:
    """The header opens a line of its own; a note inside a function is prose.

    An indented comment spelling the identifier is a reminder a contributor
    left, not the file's header -- an SBOM scraper reading the top of the file
    sees nothing -- so the file is unstamped and ``--fix`` stamps it.
    """
    source = synthetic_tree(
        tmp_path,
        '"""Unstamped."""\n\n\ndef read() -> None:\n'
        "    # SPDX-License-Identifier: Apache-2.0\n"
        "    return None\n",
    )

    first = run_gate("--root", str(tmp_path))
    assert first.returncode == 1
    assert "SPDX001" in first.stderr

    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0
    assert source.read_text(encoding="utf-8").splitlines()[0] == HEADER.rstrip("\n")


def test_spdx_gate_reports_a_directory_wearing_the_source_suffix(tmp_path: Path) -> None:
    """A path that names no file is a verdict the gate states, not a stack trace."""
    synthetic_tree(tmp_path, f'{HEADER}"""Stamped."""\n')
    package = tmp_path / "package.py"
    package.mkdir()

    result = run_gate("--root", str(tmp_path), str(package))

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "package.py" in result.stderr


def test_spdx_gate_flags_a_header_comment_out_of_place(tmp_path: Path) -> None:
    """An identifier below the docstring is still a misplaced header, not prose."""
    synthetic_tree(tmp_path, '"""Docstring first."""\n\n# SPDX-License-Identifier Apache-2.0\n')

    result = run_gate("--root", str(tmp_path))

    assert result.returncode == 1
    assert "SPDX002" in result.stderr


def test_spdx_gate_fix_keeps_the_line_ending_the_file_uses(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    source = tmp_path / "module.py"
    source.write_bytes(b'#!/usr/bin/env python3\r\n"""Runnable."""\r\n')

    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0

    assert source.read_bytes() == (
        b'#!/usr/bin/env python3\r\n# SPDX-License-Identifier: Apache-2.0\r\n"""Runnable."""\r\n'
    )
    assert run_gate("--root", str(tmp_path)).returncode == 0


def test_spdx_gate_fix_keeps_an_encoding_declaration_where_python_reads_it(
    tmp_path: Path,
) -> None:
    """PEP 263 honours the coding cookie on the first two lines and nowhere else."""
    source = synthetic_tree(
        tmp_path,
        '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Runnable."""\n',
    )

    assert run_gate("--root", str(tmp_path), "--fix").returncode == 0

    assert source.read_text(encoding="utf-8").splitlines()[:3] == [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        HEADER.rstrip("\n"),
    ]
    assert run_gate("--root", str(tmp_path)).returncode == 0
