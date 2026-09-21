# Development Scripts

Gates for agenticraft-foundation. Each one runs on its own, has a caller in
CI, and proves its own rules can fail before it certifies anything.

---

## Available Scripts

### check_spdx_headers.py

Holds every Python file in the repository to the SPDX identifier
`pyproject.toml` declares. The identifier has to be the file's first comment
line, below a shebang when the file has one.

```bash
python3 scripts/check_spdx_headers.py              # judge the tree
python3 scripts/check_spdx_headers.py FILE...      # judge these files
python3 scripts/check_spdx_headers.py --fix        # stamp what carries no header
python3 scripts/check_spdx_headers.py --self-test  # prove the rules can fail
```

Rules:

| Rule | Fires on |
| --- | --- |
| `SPDX001` | a Python file with no SPDX identifier |
| `SPDX002` | an identifier that is not the first comment line, below any shebang or encoding declaration |
| `SPDX003` | an identifier naming a licence other than the declared one |
| `SPDX004` | a file whose bytes are not UTF-8, so it states no identifier |

An identifier is a line `#` starts and the marker opens. The rule is textual
because the tools that read these headers are: an SBOM generator scrapes the
line rather than parsing the file. So prose that names the convention without
standing at the head of a line -- a docstring or a comment carrying the marker
mid-line, a note indented inside a function -- is not a header, and the file it
sits in is judged as the unstamped file it is and `--fix` stamps it. A line the
marker opens reads as a header to a scraper wherever it sits, a docstring
included, so `SPDX002` calling such a line misplaced is the rule holding rather
than missing.

`--fix` adds a header that is absent and nothing else: a misplaced or
mismatched identifier is someone's decision to revisit, not a line to rewrite
unasked, so `SPDX002` and `SPDX003` are reported either way. It writes the
header below any shebang and any PEP 263 encoding declaration, where the
interpreter still reads both, and in the line ending the file already uses.

Given no paths the gate walks the whole tree, pruning a directory on what it
is rather than on what it is called. A tool's own cache or metadata -- `.git`,
`.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.tox`, `__pycache__` -- is
pruned wherever it sits, because the name belongs to the tool that writes it;
that is the whole of what a name alone decides. A virtual environment is
pruned on the `pyvenv.cfg` that makes it one, whatever it is named, and a
`node_modules` on the `package.json` beside it. `build/`, `dist/` and `site/`
are pruned only at the root, where the build backend and MkDocs write them. So
a package or a fixture a contributor names `venv`, `node_modules`, `build` or
`site` carries none of that evidence and is judged as the source it is: no
Python directory a contributor adds can escape the gate by its name alone.

The evidence for an environment is the `pyvenv.cfg` the interpreter writes when
it creates one. A conda or mamba environment writes no such file, so one placed
inside the working tree under a name no tool owns would be walked and judged as
source. None sits here, and the answer if one ever does is to keep it out of
the tree rather than to teach the gate another name.

The expected identifier is read from the `[project]` table rather than written
into the script, so relicensing the project is one edit to the metadata and
this gate then names every file still carrying the old identifier.

Exit codes: `0` clean, `1` violation, `2` no measurement could be taken — the
self-test failed, the tree declares no licence to hold the headers to, or the
run had no Python file to judge (a path naming no file, a path that is not
`.py`, a tree holding no source). The self-test runs before any real file is
read and the walk half of it is proved on a tree built in a temporary
directory, so neither a gate that has stopped working nor a run that read
nothing is ever mistaken for a clean tree.

Callers: the `SPDX headers` step of the `lint` job in
`.github/workflows/ci.yml`, which judges the whole tree; the
`check-spdx-headers` hook in `.pre-commit-config.yaml`, which hands over the
staged `.py` files -- the suffix the gate judges, rather than pre-commit's
wider `python` tag, which an extensionless script with a python shebang also
carries -- so untracked scratch work never blocks a commit and a staged file
the gate does not judge never stalls one; and the
pre-PR checklist in `CONTRIBUTING.md`. Pinned by
`tests/test_spdx_headers_gate.py`, which drives it the way CI does.
