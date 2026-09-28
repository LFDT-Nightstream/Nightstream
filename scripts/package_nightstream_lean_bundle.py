#!/usr/bin/env python3
"""Create a focused, cache-free Nightstream Lean review bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PurePosixPath("nightstream-lean-review")
LEAN_ROOT = Path("formal/nightstream-lean")
PAPER_ROOTS = (
    Path("docs/superneo-paper"),
    Path("docs/hypernova-paper"),
)
STRUCTURE_FILES = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("SECURITY.md"),
    Path("CONTRIBUTING.md"),
    Path("LICENSE"),
    Path("scripts/audit_formal_lean.sh"),
    Path("scripts/package_nightstream_lean_bundle.py"),
    Path("wiki/formal/index.md"),
    Path("wiki/architecture/index.md"),
    Path("wiki/protocol/index.md"),
    Path("wiki/protocol/superneo-folding.md"),
    Path("wiki/protocol/hypernova-ivc.md"),
    Path("wiki/protocol/transcript-and-digests.md"),
    Path("wiki/development/testing.md"),
    Path("wiki/glossary.md"),
)

EXCLUDED_DIR_NAMES = {
    ".cache",
    ".git",
    ".idea",
    ".lake",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    "__pycache__",
    "dist",
    "node_modules",
    "target",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}
EXCLUDED_FILE_SUFFIXES = {
    ".expected",
    ".ilean",
    ".o",
    ".olean",
    ".profraw",
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
    ".tmp",
    ".trace",
    ".zip",
}
NORMALIZED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class BundleError(RuntimeError):
    """Raised when the requested bundle cannot be produced safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the current Nightstream Lean working tree, split SuperNeo "
            "and HyperNova paper sections, and project-structure documentation."
        )
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path.cwd() / "nightstream-lean-review.zip",
        help="output ZIP path (default: ./nightstream-lean-review.zip)",
    )
    return parser.parse_args()


def excluded_file(path: Path) -> bool:
    return (
        path.name in EXCLUDED_FILE_NAMES
        or path.name.startswith(".tmp")
        or path.suffix.lower() in EXCLUDED_FILE_SUFFIXES
    )


def collect_tree(
    relative_root: Path,
    include: Callable[[Path], bool] = lambda _path: True,
) -> list[Path]:
    root = REPO_ROOT / relative_root
    if not root.is_dir():
        raise BundleError(f"required directory is missing: {relative_root}")

    collected: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            child = current_path / name
            if name in EXCLUDED_DIR_NAMES:
                continue
            if child.is_symlink():
                raise BundleError(f"unexpected source symlink: {child}")
            kept_dirs.append(name)
        dirs[:] = kept_dirs

        for name in sorted(files):
            source = current_path / name
            if excluded_file(source) or not include(source):
                continue
            if source.is_symlink():
                raise BundleError(f"unexpected source symlink: {source}")
            if not source.is_file():
                raise BundleError(f"unexpected non-file source: {source}")
            collected.append(source.relative_to(REPO_ROOT))
    return collected


def collect_sources() -> tuple[list[Path], dict[str, int]]:
    lean_files = collect_tree(LEAN_ROOT)
    paper_files: list[Path] = []
    paper_counts: dict[str, int] = {}
    for paper_root in PAPER_ROOTS:
        sections = collect_tree(
            paper_root,
            include=lambda path: path.suffix.lower() == ".md",
        )
        if not sections:
            raise BundleError(f"no Markdown paper sections found in {paper_root}")
        paper_files.extend(sections)
        paper_counts[paper_root.name] = len(sections)

    for relative in STRUCTURE_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise BundleError(f"required structure document is missing: {relative}")

    sources = sorted(set(lean_files + paper_files + list(STRUCTURE_FILES)))
    counts = {
        "nightstream_lean": len(lean_files),
        "paper_sections": len(paper_files),
        "structure_files": len(STRUCTURE_FILES),
        **paper_counts,
    }
    return sources, counts


def git_output(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "(clean)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_readme(counts: dict[str, int]) -> str:
    lean_toolchain = (REPO_ROOT / LEAN_ROOT / "lean-toolchain").read_text().strip()
    status = git_output(
        [
            "status",
            "--short",
            "--",
            str(LEAN_ROOT),
            *(str(root) for root in PAPER_ROOTS),
        ]
    )
    return f"""# Nightstream Lean review bundle

This is a snapshot of the current filesystem working tree, not an export of
Git HEAD. Modified and untracked files below `formal/nightstream-lean` are
included.

## Source provenance

- Git commit: `{git_output(["rev-parse", "HEAD"])}`
- Git branch: `{git_output(["branch", "--show-current"])}`
- Lean toolchain: `{lean_toolchain}`
- Nightstream Lean files: {counts["nightstream_lean"]}
- SuperNeo Markdown sections: {counts["superneo-paper"]}
- HyperNova Markdown sections: {counts["hypernova-paper"]}
- Additional structure documents: {counts["structure_files"]}

Relevant working-tree status at packaging time:

```text
{status}
```

## Layout

- `formal/nightstream-lean/`: complete active Lean package, tests, specs,
  assurance data, and validation scripts.
- `docs/superneo-paper/`: all split SuperNeo Markdown sections.
- `docs/hypernova-paper/`: all split HyperNova Markdown sections.
- `wiki/` and repository-root documents: curated architecture, protocol,
  testing, security, and contribution context.

The archive excludes `.lake`, `target`, Git/editor metadata, Python caches,
temporary files, compiled Lean objects, and pre-existing ZIP files.

## Compile

Install the toolchain named by `formal/nightstream-lean/lean-toolchain`, then:

```bash
cd formal/nightstream-lean
lake build
lake build tests.Axioms
```

The Lake project has no external package dependencies. Some executable
conformance and repository-static checks intentionally inspect Rust paths
outside the Lean package; those checks require the full Nightstream repository
and are not made standalone by this review bundle.
"""


def zip_info(name: PurePosixPath, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(str(name), date_time=NORMALIZED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((stat.S_IFREG | mode) & 0xFFFF) << 16
    return info


def write_bundle(output: Path, sources: list[Path], counts: dict[str, int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    readme = bundle_readme(counts)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            readme_name = BUNDLE_ROOT / "BUNDLE_README.md"
            archive.writestr(
                zip_info(readme_name, 0o644),
                readme.encode("utf-8"),
                compresslevel=9,
            )
            for relative in sources:
                source = REPO_ROOT / relative
                mode = stat.S_IMODE(source.stat().st_mode)
                member = BUNDLE_ROOT / PurePosixPath(relative.as_posix())
                with source.open("rb") as source_handle:
                    with archive.open(
                        zip_info(member, mode),
                        mode="w",
                        force_zip64=True,
                    ) as member_handle:
                        shutil.copyfileobj(source_handle, member_handle)

        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise BundleError("output path must end in .zip")
    sources, counts = collect_sources()
    write_bundle(output, sources, counts)
    print(f"created: {output}")
    print(f"files: {len(sources) + 1}")
    print(f"size: {output.stat().st_size} bytes")
    print(f"sha256: {sha256(output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
