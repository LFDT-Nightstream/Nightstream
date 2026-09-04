#!/usr/bin/env python3
"""Write or check the deterministic F-prime Stage 1 review manifest.

Run this command only while other processes do not write to the scoped tree.
Two complete matching hash passes enforce that quiescent review window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "nightstream-fprime-stage1-review-manifest-v1"
DEFAULT_MANIFEST = Path("FPRIME_STAGE1_REVIEW_MANIFEST.json")

RECURSIVE_ROOTS = (
    Path("formal/nightstream-fprime"),
    Path("crates/nightstream-fprime"),
    Path("crates/neo-fold-clean"),
    Path("crates/neo-math"),
    Path("crates/neo-params"),
    Path("crates/neo-ccs"),
    Path("crates/neo-ajtai"),
    Path("crates/neo-transcript"),
    Path("crates/neo-reductions"),
    Path("crates/wip-spartan"),
    Path("decisions"),
    Path("docs/superneo-paper-v1_1"),
)

EXPLICIT_FILES = (
    Path("AGENTS.md"),
    Path("FPRIME_LEAN_ARCHITECTURE_SPEC.md"),
    Path("FPRIME_STAGE1_GOAL.md"),
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("rust-toolchain.toml"),
    Path(".cargo/config.toml"),
    Path(".gitattributes"),
    Path(".gitignore"),
    Path(".github/workflows/ci.yml"),
    Path("scripts/fprime_stage1_review_manifest.py"),
)

REQUIRED_ARTIFACTS = (
    Path(
        "formal/nightstream-fprime/artifacts/"
        "nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
    ),
    Path(
        "formal/nightstream-fprime/artifacts/"
        "nightstream-fprime-stage1-poseidon2-hash-chain-v1-expanded.json"
    ),
    Path(
        "formal/nightstream-fprime/artifacts/"
        "nightstream-fprime-stage1-poseidon2-hash-chain-v1-binding-v1.json"
    ),
    Path(
        "formal/nightstream-fprime/artifacts/"
        "nightstream-fprime-stage1-piccs-parity-v1.json"
    ),
    Path(
        "formal/nightstream-fprime/artifacts/"
        "nightstream-fprime-stage1-piccs-ownership-v1.json"
    ),
    Path(
        "formal/nightstream-fprime/artifacts/"
        "nightstream-fprime-ajtai-setup-v1-parity.json"
    ),
)

EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".lake",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}
EXCLUDED_FILE_NAMES = {".DS_Store", "Thumbs.db"}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"

OWNER_FILES = {
    Path("AGENTS.md"),
    Path("FPRIME_LEAN_ARCHITECTURE_SPEC.md"),
    Path("FPRIME_STAGE1_GOAL.md"),
}
CONFIG_FILES = {
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("rust-toolchain.toml"),
    Path(".cargo/config.toml"),
    Path(".gitattributes"),
    Path(".gitignore"),
    Path(".github/workflows/ci.yml"),
}


class ManifestError(RuntimeError):
    """Raised when an exact review manifest cannot be produced."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write or check the deterministic Nightstream F-prime Stage 1 "
            "review manifest."
        )
    )
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest path relative to the repository root",
    )
    return parser.parse_args()


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ManifestError("cannot read the repository Git HEAD") from error
    head = result.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ManifestError(f"invalid Git HEAD: {head!r}")
    return head


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def output_relative_path(output: Path) -> Path | None:
    try:
        return output.relative_to(REPO_ROOT)
    except ValueError:
        return None


def reject_symlink_components(relative: Path) -> None:
    current = REPO_ROOT
    if stat.S_ISLNK(current.lstat().st_mode):
        raise ManifestError(f"repository root is a symlink: {current}")
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ManifestError(f"required path has a symlink component: {relative}")


def tree_path_is_excluded(relative: Path, recursive_root: Path) -> bool:
    local = relative.relative_to(recursive_root)
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in local.parts[:-1]):
        return True
    if relative.name in EXCLUDED_FILE_NAMES:
        return True
    artifact_root = Path("formal/nightstream-fprime/artifacts")
    return is_relative_to(relative, artifact_root)


def selected_path(relative: Path, output_relative: Path | None) -> bool:
    if output_relative is not None and relative == output_relative:
        return False
    if relative in EXPLICIT_FILES or relative in REQUIRED_ARTIFACTS:
        return True
    for recursive_root in RECURSIVE_ROOTS:
        if is_relative_to(relative, recursive_root):
            return not tree_path_is_excluded(relative, recursive_root)
    return False


def collect_tree(
    recursive_root: Path,
    output_relative: Path | None,
) -> list[Path]:
    root = REPO_ROOT / recursive_root
    reject_symlink_components(recursive_root)
    if not root.is_dir():
        raise ManifestError(f"required directory is missing: {recursive_root}")

    paths: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            child = current_path / name
            relative = child.relative_to(REPO_ROOT)
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            if is_relative_to(
                relative,
                Path("formal/nightstream-fprime/artifacts"),
            ):
                continue
            if child.is_symlink():
                raise ManifestError(f"unexpected source symlink: {relative}")
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(files):
            source = current_path / name
            relative = source.relative_to(REPO_ROOT)
            if not selected_path(relative, output_relative):
                continue
            if source.is_symlink():
                raise ManifestError(f"unexpected source symlink: {relative}")
            if not source.is_file():
                raise ManifestError(f"unexpected non-file source: {relative}")
            paths.append(relative)
    return paths


def tracked_paths() -> list[Path]:
    pathspecs = [str(path) for path in (*RECURSIVE_ROOTS, *EXPLICIT_FILES, *REQUIRED_ARTIFACTS)]
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", *pathspecs],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ManifestError("cannot enumerate scoped tracked files") from error
    return [Path(os.fsdecode(raw)) for raw in result.stdout.split(b"\0") if raw]


def collect_paths(output_relative: Path | None) -> list[Path]:
    paths: list[Path] = []
    for recursive_root in RECURSIVE_ROOTS:
        paths.extend(collect_tree(recursive_root, output_relative))

    for relative in (*EXPLICIT_FILES, *REQUIRED_ARTIFACTS):
        if output_relative is not None and relative == output_relative:
            raise ManifestError("the output path cannot replace a required input")
        reject_symlink_components(relative)
        source = REPO_ROOT / relative
        if not source.exists():
            raise ManifestError(f"required file is missing: {relative}")
        if source.is_symlink():
            raise ManifestError(f"required file is a symlink: {relative}")
        if not source.is_file():
            raise ManifestError(f"required path is not a file: {relative}")
        paths.append(relative)

    missing_tracked = sorted(
        (
            relative
            for relative in tracked_paths()
            if selected_path(relative, output_relative)
            and not (REPO_ROOT / relative).exists()
        ),
        key=lambda path: path.as_posix().encode("utf-8"),
    )
    if missing_tracked:
        joined = ", ".join(path.as_posix() for path in missing_tracked)
        raise ManifestError(f"scoped tracked files are deleted: {joined}")

    return sorted(
        set(paths),
        key=lambda path: path.as_posix().encode("utf-8"),
    )


def entry_class(relative: Path) -> str:
    if relative in REQUIRED_ARTIFACTS:
        return "artifact"
    if relative in OWNER_FILES or is_relative_to(relative, Path("decisions")):
        return "owner"
    if is_relative_to(relative, Path("docs/superneo-paper-v1_1")):
        return "paper"
    if "tests" in relative.parts:
        return "test"
    if (
        relative in CONFIG_FILES
        or relative.suffix in {".lock", ".toml", ".yaml", ".yml"}
        or relative.name in {"AGENTS.md", "lean-toolchain", "lake-manifest.json"}
    ):
        return "config"
    return "source"


def stat_signature(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def hash_entry(
    relative: Path,
) -> tuple[dict[str, object], tuple[int, int, int, int, int, int]]:
    source = REPO_ROOT / relative
    path_before = source.lstat()
    if not stat.S_ISREG(path_before.st_mode):
        raise ManifestError(f"source changed to a non-file: {relative}")

    digest = hashlib.sha256()
    byte_count = 0
    prefix = b""
    with source.open("rb") as handle:
        descriptor_before = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            if len(prefix) < len(LFS_POINTER_PREFIX):
                prefix += chunk[: len(LFS_POINTER_PREFIX) - len(prefix)]
            digest.update(chunk)
            byte_count += len(chunk)
        descriptor_after = os.fstat(handle.fileno())

    path_after = source.lstat()
    signature = stat_signature(path_before)
    if (
        signature != stat_signature(descriptor_before)
        or signature != stat_signature(descriptor_after)
        or signature != stat_signature(path_after)
    ):
        raise ManifestError(f"source changed while it was hashed: {relative}")
    if relative in REQUIRED_ARTIFACTS and prefix == LFS_POINTER_PREFIX:
        raise ManifestError(f"required artifact is still a Git-LFS pointer: {relative}")

    mode = "100755" if path_before.st_mode & 0o111 else "100644"
    return (
        {
            "class": entry_class(relative),
            "path": relative.as_posix(),
            "mode": mode,
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
        },
        signature,
    )


def snapshot(
    output_relative: Path | None,
) -> tuple[
    str,
    list[dict[str, object]],
    dict[Path, tuple[int, int, int, int, int, int]],
]:
    head_before = git_head()
    paths = collect_paths(output_relative)
    entries: list[dict[str, object]] = []
    signatures: dict[Path, tuple[int, int, int, int, int, int]] = {}
    for relative in paths:
        entry, signature = hash_entry(relative)
        entries.append(entry)
        signatures[relative] = signature

    if collect_paths(output_relative) != paths:
        raise ManifestError("the scoped file set changed while it was hashed")
    for relative, signature in signatures.items():
        if stat_signature((REPO_ROOT / relative).lstat()) != signature:
            raise ManifestError(f"source changed while the cut was hashed: {relative}")
    head_after = git_head()
    if head_after != head_before:
        raise ManifestError("Git HEAD changed while the cut was hashed")
    return head_before, entries, signatures


def manifest_bytes(output: Path) -> bytes:
    output_relative = output_relative_path(output)
    first = snapshot(output_relative)
    second = snapshot(output_relative)
    if first != second:
        raise ManifestError("the reviewed cut did not remain stable for two hash passes")
    head, entries, _signatures = second

    manifest = {
        "schema": SCHEMA,
        "git_head": head,
        "entries": entries,
    }
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def display_path(path: Path) -> str:
    relative = output_relative_path(path)
    return relative.as_posix() if relative is not None else str(path)


def write_manifest(output: Path) -> None:
    if not output.parent.is_dir():
        raise ManifestError(f"manifest parent directory is missing: {output.parent}")
    content = manifest_bytes(output)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, output)
    except OSError as error:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise ManifestError(f"cannot write manifest: {output}") from error
    print(f"wrote {display_path(output)}")
    print(f"manifest_sha256 {hashlib.sha256(content).hexdigest()}")


def check_manifest(output: Path) -> None:
    try:
        stored = output.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read manifest: {output}") from error
    current = manifest_bytes(output)
    if stored != current:
        print(f"stored_sha256 {hashlib.sha256(stored).hexdigest()}", file=sys.stderr)
        print(f"current_sha256 {hashlib.sha256(current).hexdigest()}", file=sys.stderr)
        raise ManifestError("manifest does not match the current reviewed cut")
    print(f"checked {display_path(output)}")
    print(f"manifest_sha256 {hashlib.sha256(stored).hexdigest()}")


def main() -> int:
    arguments = parse_args()
    output = arguments.manifest
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    try:
        if arguments.action == "write":
            write_manifest(output)
        else:
            check_manifest(output)
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
