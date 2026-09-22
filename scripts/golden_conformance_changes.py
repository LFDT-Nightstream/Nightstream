#!/usr/bin/env python3
"""Select golden checks from changed repository paths.

``lean_reference`` selects an independent-source/input audit. Changed Lean
computation or inputs require regeneration. Fresh Lean verification is part
of ``native`` even when this flag is false. These flags do not prove success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
# Cargo path dependencies of nightstream, optional GPU adapters, and the
# legacy golden checker, including wip-spartan. Metal is checked
# against the CPU result; CUDA currently has only an unavailable boundary.
CRATES = {
    "nightstream", "nightstream-fprime", "neo-fold-legacy", "wip-spartan",
    "neo-math", "neo-params", "neo-ccs", "neo-ajtai", "neo-transcript",
    "neo-reductions", "neo-prover-metal", "neo-prover-cuda",
}
NATIVE = ("native", "metal")
ALL = ("native", "lean_reference", "metal")
BUILD_FILES = {"Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml", ".gitattributes"}
EMBEDDED_DOCS = {
    "crates/neo-fold-legacy/tests/preprocessing_read_only.md",
    "crates/neo-fold-legacy/tests/nebula_preprocessing_read_only.md",
}
# These records locate the original witnesses/reference archive or supply the
# replay coordinator's measured partitions. Other review reports are receipts.
REPLAY_INPUTS = {
    "INDEPENDENT_REPLAY_ASSET.json", "PIRLC_WITNESS_REPLAY_ASSET.json",
    "PICCS_FIRST_ROUND_REPLAY.json", "PIDEC_MATRIX_RANGES.json",
    "EVIDENCE_RELEASE.json", "TERMINAL_REPLAY_INPUTS.json", "NONZERO_EXECUTION_RELEASE.json",
}


def check_dependency_roots(root: Path) -> None:
    """Fail if a new Cargo path dependency has no checked-in trigger rule."""
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    pending, seen = [root / "crates/nightstream"], set()
    while pending:
        directory = pending.pop().resolve()
        if directory in seen:
            continue
        seen.add(directory)
        relative = directory.relative_to(root.resolve()).as_posix()
        if relative not in {f"crates/{name}" for name in CRATES}:
            raise ValueError(f"unregistered Cargo dependency: {relative}")
        manifest = tomllib.loads((directory / "Cargo.toml").read_text())
        sections = [manifest, *manifest.get("target", {}).values()]
        for section in sections:
            kinds = ["dependencies", "build-dependencies"]
            if relative == "crates/nightstream":
                kinds.append("dev-dependencies")
            for kind in kinds:
                for name, dependency in section.get(kind, {}).items():
                    if not isinstance(dependency, dict):
                        continue
                    base = directory
                    if dependency.get("workspace"):
                        dependency = workspace["workspace"]["dependencies"][name]
                        base = root
                    if isinstance(dependency, dict) and "path" in dependency:
                        pending.append(base / dependency["path"])


def checks_for_path(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError(f"expected a repository-relative path: {name!r}")
    if name in EMBEDDED_DOCS:
        return NATIVE
    if path.suffix == ".md":
        return ()
    if name in BUILD_FILES or name.startswith(".cargo/"):
        return NATIVE
    if name.startswith(("formal/nightstream-fprime/", "scripts/lean_graph/")):
        return ALL
    if (name.startswith(("scripts/golden_conformance", "scripts/tests/test_golden_conformance"))
            or name in {"scripts/bridge_first_second.py", "scripts/check_selected_replay.py",
                        "scripts/tests/test_bridge_first_second.py", "scripts/tests/test_check_selected_replay.py"}
            or name == "scripts/package_nightstream_fprime_bundle.py"
            or name == "scripts/fprime_stage1_review_manifest.py"
            or name.startswith(".github/workflows/")):
        return ALL
    if name.startswith("docs/reviews/nightstream-fprime-requirements/") and path.name in REPLAY_INPUTS:
        return ALL
    if name == "docs/reviews/nightstream-fprime-requirements/NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip":
        return ("native",)  # Input to the mutation-generator regression.
    if name == "crates/nightstream/tests/restore_golden_inputs.py":
        return ALL
    if name == "scripts/build_metal_shaders.sh":
        return ("metal",)
    if len(path.parts) < 3 or path.parts[0] != "crates" or path.parts[1] not in CRATES:
        return ()
    crate, part = path.parts[1:3]
    if crate == "neo-prover-metal":
        if part in {"src", "shaders", "tests", "Cargo.toml", "build.rs"}:
            return ("metal",)
    elif part in {"src", "Cargo.toml", "build.rs"}:
        return NATIVE
    if part == "artifacts":
        return ALL
    if part == "tests":
        if crate == "nightstream":
            if len(path.parts) > 3 and path.parts[3] == "evidence":
                return ()  # Historical receipts are not execution inputs.
            return ALL if "fixtures" in path.parts[3:4] else NATIVE
        if crate == "nightstream-fprime":
            return NATIVE
        if crate == "neo-fold-legacy" and len(path.parts) > 3 and path.parts[3] == "nifs":
            return ALL if "fixtures" in path.parts[4:5] else NATIVE
        return ()
    if part in {"LICENSE", "rustfmt.toml", "open-questions", "tests-paper-exact"}:
        return ()
    raise ValueError(f"unclassified file in a golden dependency: {name}")


def classify(paths: list[str]) -> dict:
    triggers = {check: [] for check in ALL}
    for path in sorted(set(paths)):
        for check in checks_for_path(path):
            triggers[check].append(path)
    return {**{check: bool(paths) for check, paths in triggers.items()}, "triggers": triggers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="changed repository-relative paths")
    parser.add_argument("--base", help="compare this commit with --head")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    if bool(args.paths) == bool(args.base):
        parser.error("provide changed paths or --base, but not both")
    try:
        check_dependency_roots(ROOT)
        paths = args.paths
        if args.base:
            revisions = [subprocess.check_output(
                ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
                cwd=ROOT, text=True).strip() for ref in (args.base, args.head)]
            # Disable rename detection so removal from a trigger root cannot
            # disappear when a file moves to a directory outside that root.
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", "--no-renames", "-z", *revisions, "--"], cwd=ROOT)
            paths = [name.decode() for name in changed.split(b"\0") if name]
        print(json.dumps(classify(paths), sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"golden conformance selection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
