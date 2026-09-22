#!/usr/bin/env python3
"""Restore the selected golden inputs from the recorded evidence archives.

Archive hashes identify retained files. Source openings and proof relations
must still be checked by the production and Lean verifiers.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
REVIEW = ROOT / "docs/reviews/nightstream-fprime-requirements"


def read(name):
    return json.loads((REVIEW / name).read_text())


def checked_asset(directory, name, size, digest):
    path = directory / name
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"missing archive or wrong size: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"archive differs from its retained source record: {path}")
    return path.resolve(strict=True), {"file": name, "bytes": size, "sha256": actual}


def restore_zip(archive, output):
    output.mkdir()
    with zipfile.ZipFile(archive) as source:
        names = set()
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or member.filename in names:
                raise ValueError(f"invalid or duplicate archive path: {member.filename}")
            names.add(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"unexpected archive link: {member.filename}")
        source.extractall(output)


def restore_independent(archive, output):
    output.mkdir()
    with subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as source:
            source.extractall(output, filter="data")
        if process.wait() != 0:
            raise ValueError("independent evidence archive decompression failed")


def expected_assets():
    """The same retained archive identities drive download and restoration."""
    release = read("EVIDENCE_RELEASE.json")
    envelope = next(item for item in release["assets"]
                    if item["source_path"].endswith("/NATIVE_ENVELOPE_EVIDENCE.zip"))
    children = read("TERMINAL_REPLAY_INPUTS.json")
    later = read("NONZERO_EXECUTION_RELEASE.json")
    independent = read("INDEPENDENT_REPLAY_ASSET.json")
    sources = independent["source_dependency"]
    records = [
        ("envelope", envelope["asset_name"], envelope["bytes"], envelope["sha256"],
         release["repository"], release["proposed_tag"]),
        ("children", children["asset_name"], children["archive_bytes"], children["archive_sha256"],
         children["repository"], children["proposed_release_tag"]),
        ("reference-later", later["asset"], later["bytes"], later["sha256"],
         later["repository"], later["release_tag"]),
        ("original-sources", sources["file"], sources["bytes"], sources["sha256"],
         independent["repository"], independent["release_tag"]),
        ("independent", independent["archive"]["file"], independent["archive"]["bytes"],
         independent["archive"]["sha256"], independent["repository"], independent["release_tag"]),
    ]
    return [dict(zip(("label", "name", "bytes", "sha256", "repository", "tag"), record))
            for record in records]


def restore(archives, directory):
    # Resolve and check every source before creating the new restoration.
    checked = [(asset["label"], *checked_asset(archives, asset["name"], asset["bytes"], asset["sha256"]))
               for asset in expected_assets()]
    directory.mkdir(parents=True, exist_ok=False)
    receipts = []
    for label, archive, record in checked:
        output = directory / label
        if label == "independent":
            restore_independent(archive, output)
        else:
            restore_zip(archive, output)
        receipts.append({**record, "restored_as": label})
    first = directory / "reference-first"
    first.mkdir()
    for name in ("envelope.json", "fresh-claim.json", "fresh-witness.json"):
        shutil.copyfile(directory / "envelope/output" / name, first / name)
    shutil.copytree(directory / "children/child-witnesses", first / "material")
    original = directory / "original-sources/inputs/original-sources"
    request = ROOT / "crates/neo-fold-legacy/tests/nifs/fixtures/stage1_recursive_states/nonzero-running.json"
    shutil.copyfile(request, original / "next-message-input.json")
    receipt = {
        "schema": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "archives": receipts,
        "scope": "Archive restoration only. The restored bytes are inputs to later semantic comparisons.",
        "protected_reproduction": "unchanged; no protected-checker acceptance is claimed",
    }
    (directory / "restoration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    receipt = restore(args.archives.resolve(), args.directory.resolve())
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
