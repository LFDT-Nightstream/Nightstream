#!/usr/bin/env python3
"""Create a cache-free Nightstream F′ Lean source bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path("formal/nightstream-fprime")
DEPENDENCY_ROOT = Path("formal/nightstream-lean/.lake/packages")
GOAL_FILE = Path("FPRIME_STAGE1_GOAL.md")
CONTRACT_FILES = (
    Path("AGENTS.md"),
    Path("FPRIME_LEAN_ARCHITECTURE_SPEC.md"),
    GOAL_FILE,
    Path("formal/nightstream-lean/AGENTS.md"),
)
BUNDLE_ROOT = PurePosixPath("nightstream-fprime-source")
MAX_ARCHIVE_BYTES = 500_000_000
PROOFWIDGETS_TRACES = {
    REPO_ROOT
    / DEPENDENCY_ROOT
    / "proofwidgets/widget/package-lock.json.trace",
    REPO_ROOT
    / DEPENDENCY_ROOT
    / "proofwidgets/widget/js/lake.trace",
}

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
    "artifacts",
    "build",
    "dist",
    "node_modules",
    "target",
}
EXCLUDED_FILE_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDED_FILE_SUFFIXES = {
    ".a",
    ".dylib",
    ".exe",
    ".ilean",
    ".o",
    ".olean",
    ".profraw",
    ".pyc",
    ".pyo",
    ".so",
    ".swp",
    ".swo",
    ".tmp",
    ".trace",
    ".zip",
}
NORMALIZED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class BundleError(RuntimeError):
    """Raised when the source bundle cannot be produced safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive formal/nightstream-fprime, dependency sources, and the "
            "governing contract files without build output or artifacts."
        )
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "nightstream-fprime-source.zip",
        help=(
            "output ZIP path "
            "(default: <repository>/nightstream-fprime-source.zip)"
        ),
    )
    return parser.parse_args()


def excluded_file(path: Path) -> bool:
    if path in PROOFWIDGETS_TRACES:
        return False
    return (
        path.name in EXCLUDED_FILE_NAMES
        or path.suffix.lower() in EXCLUDED_FILE_SUFFIXES
    )


def collect_tree(relative_root: Path, *, allow_symlinks: bool) -> list[Path]:
    root = REPO_ROOT / relative_root
    if not root.is_dir():
        raise BundleError(f"required directory is missing: {relative_root}")

    sources: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            child = current_path / name
            if name in EXCLUDED_DIR_NAMES:
                continue
            if child.is_symlink():
                if not allow_symlinks:
                    raise BundleError(f"unexpected source symlink: {child}")
                sources.append(child.relative_to(REPO_ROOT))
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs

        for name in sorted(files):
            source = current_path / name
            if excluded_file(source):
                continue
            if source.is_symlink() and not allow_symlinks:
                raise BundleError(f"unexpected source symlink: {source}")
            if not source.is_symlink() and not source.is_file():
                raise BundleError(f"unexpected non-file source: {source}")
            sources.append(source.relative_to(REPO_ROOT))

    return sources


def git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise BundleError(
            f"git command failed for {repository}: {' '.join(arguments)}"
        ) from error
    return result.stdout.strip()


def dependency_directories() -> list[Path]:
    root = REPO_ROOT / DEPENDENCY_ROOT
    if not root.is_dir():
        raise BundleError(f"required directory is missing: {DEPENDENCY_ROOT}")

    manifest_path = REPO_ROOT / PACKAGE_ROOT / "lake-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        revisions = {
            package["name"]: package["rev"] for package in manifest["packages"]
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid Lake manifest: {manifest_path}") from error

    dependencies = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    names = {path.name for path in dependencies}
    if names != set(revisions):
        raise BundleError("dependency checkout names do not match lake-manifest.json")

    for dependency in dependencies:
        actual = git_output(dependency, "rev-parse", "HEAD")
        expected = revisions[dependency.name]
        if actual != expected:
            raise BundleError(
                f"{dependency.name}: checkout {actual} does not match {expected}"
            )
    return dependencies


def collect_dependency_oleans(dependencies: list[Path]) -> list[Path]:
    outputs: list[Path] = []
    for dependency in dependencies:
        build_root = dependency / ".lake/build/lib/lean"
        if not build_root.is_dir():
            continue
        outputs.extend(
            path.relative_to(REPO_ROOT)
            for path in build_root.rglob("*.olean")
            if path.is_file()
        )
    if not outputs:
        raise BundleError("no dependency .olean files found")
    return sorted(outputs)


def collect_sources() -> tuple[list[Path], list[Path]]:
    dependencies = dependency_directories()
    sources = collect_tree(PACKAGE_ROOT, allow_symlinks=False)
    sources.extend(collect_tree(DEPENDENCY_ROOT, allow_symlinks=True))
    sources.extend(collect_dependency_oleans(dependencies))

    for relative in CONTRACT_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise BundleError(f"required contract file is missing: {relative}")
        sources.append(relative)

    return sorted(set(sources)), dependencies


def zip_info(name: PurePosixPath, mode: int, file_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(str(name), date_time=NORMALIZED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((file_type | mode) & 0xFFFF) << 16
    return info


def zip_directory_info(name: PurePosixPath) -> zipfile.ZipInfo:
    path = str(name)
    if not path.endswith("/"):
        path += "/"
    info = zipfile.ZipInfo(path, date_time=NORMALIZED_ZIP_TIME)
    info.create_system = 3
    info.external_attr = ((stat.S_IFDIR | 0o755) & 0xFFFF) << 16 | 0x10
    return info


def write_member(
    archive: zipfile.ZipFile,
    source: Path,
    relative: Path,
) -> None:
    member = BUNDLE_ROOT / PurePosixPath(relative.as_posix())
    source_mode = source.lstat().st_mode
    mode = stat.S_IMODE(source_mode)
    if source.is_symlink():
        archive.writestr(
            zip_info(member, mode, stat.S_IFLNK),
            os.readlink(source).encode("utf-8"),
            compresslevel=9,
        )
        return
    with source.open("rb") as source_handle:
        with archive.open(
            zip_info(member, mode, stat.S_IFREG),
            mode="w",
            force_zip64=True,
        ) as member_handle:
            shutil.copyfileobj(source_handle, member_handle)


def stage_git_metadata(stage_root: Path, dependency: Path) -> tuple[list[Path], Path]:
    relative_dependency = dependency.relative_to(REPO_ROOT)
    staged_git = stage_root / relative_dependency / ".git"
    pack_directory = staged_git / "objects/pack"
    refs_directory = staged_git / "refs"
    pack_directory.mkdir(parents=True)
    refs_directory.mkdir()

    source_git = dependency / ".git"
    for name in ("HEAD", "config", "index"):
        source = source_git / name
        if not source.is_file():
            raise BundleError(f"missing Git metadata: {source}")
        shutil.copy2(source, staged_git / name)

    head = git_output(dependency, "rev-parse", "HEAD")
    tree_objects = git_output(
        dependency,
        "rev-list",
        "--objects",
        "--no-object-names",
        "HEAD^{tree}",
    )
    object_ids = f"{head}\n{tree_objects}\n"
    pack_base = pack_directory / "pack"
    try:
        subprocess.run(
            ["git", "-C", str(dependency), "pack-objects", str(pack_base)],
            input=object_ids,
            text=True,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise BundleError(f"could not pack Git metadata for {dependency.name}") from error

    files = sorted(path for path in staged_git.rglob("*") if path.is_file())
    return files, refs_directory


def write_bundle(
    output: Path,
    sources: list[Path],
    dependencies: list[Path],
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    member_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

        with tempfile.TemporaryDirectory(
            prefix="nightstream-fprime-git-"
        ) as git_stage_name:
            git_stage = Path(git_stage_name)
            metadata: list[tuple[list[Path], Path]] = [
                stage_git_metadata(git_stage, dependency)
                for dependency in dependencies
            ]

            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                allowZip64=True,
            ) as archive:
                for relative in sources:
                    write_member(archive, REPO_ROOT / relative, relative)
                    member_count += 1
                for files, refs_directory in metadata:
                    refs_relative = refs_directory.relative_to(git_stage)
                    archive.writestr(
                        zip_directory_info(
                            BUNDLE_ROOT / PurePosixPath(refs_relative.as_posix())
                        ),
                        b"",
                    )
                    member_count += 1
                    for source in files:
                        relative = source.relative_to(git_stage)
                        write_member(archive, source, relative)
                        member_count += 1

        archive_size = temporary.stat().st_size
        if archive_size > MAX_ARCHIVE_BYTES:
            raise BundleError(
                f"archive is {archive_size} bytes; limit is {MAX_ARCHIVE_BYTES}"
            )

        os.replace(temporary, output)
        temporary = None
        return member_count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise BundleError("output path must end in .zip")

    sources, dependencies = collect_sources()
    member_count = write_bundle(output, sources, dependencies)
    print(f"created: {output}")
    print(f"members: {member_count}")
    print(f"size: {output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
