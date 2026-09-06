"""Capture source and input bytes; exclude result records from source identity."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


class EvidenceError(Exception):
    """An evidence claim cannot be established."""


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=unique)
    except (OSError, ValueError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_relative(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise EvidenceError(f"expected a relative path without '..': {value}")
    return path


# Reuse the existing review manifest's file identity and exclusion rules.
_spec = importlib.util.spec_from_file_location(
    "stage1_review_manifest", Path(__file__).parents[1] / "fprime_stage1_review_manifest.py")
_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_review)


def signature(path):
    return _review.stat_signature(path.lstat())


def file_entry(path):
    path = Path(path)
    before = signature(path)
    if stat.S_ISLNK(before[2]):
        target = os.readlink(path)
        if signature(path) != before:
            raise EvidenceError(f"symlink changed during read: {path}")
        return {"link": target, "sha256": hashlib.sha256(os.fsencode(target)).hexdigest(),
                "bytes": len(os.fsencode(target)), "executable": False}
    if not stat.S_ISREG(before[2]):
        raise EvidenceError(f"expected a regular file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as handle:
        if _review.stat_signature(os.fstat(handle.fileno())) != before:
            raise EvidenceError(f"file changed before read: {path}")
        prefix = handle.read(len(_review.LFS_POINTER_PREFIX))
        if prefix == _review.LFS_POINTER_PREFIX:
            raise EvidenceError(f"Git LFS content is missing: {path}")
        value.update(prefix)
        for chunk in iter(lambda: handle.read(shutil.COPY_BUFSIZE), b""):
            value.update(chunk)
        if _review.stat_signature(os.fstat(handle.fileno())) != before:
            raise EvidenceError(f"file changed during read: {path}")
    if signature(path) != before:
        raise EvidenceError(f"file changed during read: {path}")
    return {"sha256": value.hexdigest(), "bytes": before[3],
            "executable": bool(before[2] & 0o111)}


def copy_file(source, destination):
    """Copy into owned staging space; preserve executable mode at creation.

    The installed library seed is several gigabytes. macOS clonefile retains
    copy-on-write isolation without duplicating those blocks for each gate.
    Other filesystems use the standard streaming copy.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        destination.unlink()
    if sys.platform == "darwin":
        import ctypes
        library = ctypes.CDLL(None, use_errno=True)
        clone = library.clonefile
        clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return str(destination)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         stat.S_IMODE(source.stat().st_mode))
    with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
        shutil.copyfileobj(reader, writer)
    return str(destination)


def no_symlinks(path, allow_leaf=False):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            if allow_leaf and component == path:
                continue
            raise EvidenceError(f"symlink is not a snapshot input: {component}")


def files(root, exclusions=(), skip_build=True):
    root = Path(root)
    no_symlinks(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise EvidenceError(f"missing snapshot input: {root}")
    selected = []
    for directory, directories, names in os.walk(root, followlinks=False):
        here = Path(directory)
        def excluded(path):
            relative = path.relative_to(root).as_posix()
            return any(fnmatch.fnmatch(relative, pattern) for pattern in exclusions)
        kept = []
        for name in sorted(directories):
            path = here / name
            if (skip_build and name in _review.EXCLUDED_DIRECTORY_NAMES) or excluded(path):
                continue
            no_symlinks(path)
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = here / name
            if name in _review.EXCLUDED_FILE_NAMES or excluded(path):
                continue
            no_symlinks(path, allow_leaf=True)
            selected.append(path)
    return sorted(selected)


def inspect(source, policy, inputs, scope=None):
    """Return identity and source paths. Missing external inputs remain explicit."""
    source = Path(source)
    scope = scope if scope is not None else {
        "sources": list(policy["sources"]), "inputs": list(policy["inputs"])}
    for kind in ("sources", "inputs"):
        if not set(scope[kind]) <= policy[kind].keys():
            raise EvidenceError(f"unknown captured {kind}")
    groups, origins = {}, {}
    for name, group in policy["sources"].items():
        if name not in scope["sources"]:
            continue
        entries = {}
        for relative in group["roots"]:
            root = source / safe_relative(relative)
            if relative == "formal/nightstream-fprime/.lake/packages" and "library_seed" in inputs:
                root = Path(inputs["library_seed"]).resolve()
            for path in files(root, group.get("exclude", [])):
                suffix = "" if root.is_file() else "/" + path.relative_to(root).as_posix()
                key = "source/" + relative + suffix
                entries[key] = file_entry(path)
                origins[key] = path
        if not entries:
            raise EvidenceError(f"source group is empty: {name}")
        groups[name] = entries
    captured_inputs = {}
    for name in policy["inputs"]:
        if name not in scope["inputs"]:
            continue
        if name not in inputs:
            captured_inputs[name] = None
            continue
        root = Path(inputs[name]).absolute()
        entries = {}
        for path in files(root, skip_build=False):
            suffix = "" if root.is_file() else "/" + path.relative_to(root).as_posix()
            key = "inputs/" + name + suffix
            entries[key] = file_entry(path)
            origins[key] = path
        if not entries:
            raise EvidenceError(f"input is empty: {name}")
        captured_inputs[name] = entries
    manifest = {"schema": 1, "sources": groups, "inputs": captured_inputs}
    validate_links(manifest)
    return manifest, origins


def entries(manifest):
    result = {}
    for group in (*manifest["sources"].values(), *manifest["inputs"].values()):
        if group:
            result.update(group)
    return result


def validate_links(manifest):
    selected = entries(manifest)
    for name, value in selected.items():
        seen = set()
        while "link" in value:
            if name in seen or Path(value["link"]).is_absolute():
                raise EvidenceError(f"cyclic or absolute source symlink: {name}")
            seen.add(name)
            name = os.path.normpath(str(Path(name).parent / value["link"]))
            if name not in selected:
                raise EvidenceError(f"symlink target is outside the captured file set: {name}")
            value = selected[name]


def verify(directory, manifest):
    validate_links(manifest)
    for name, expected in entries(manifest).items():
        path = Path(directory) / safe_relative(name)
        no_symlinks(path, allow_leaf="link" in expected)
        if file_entry(path) != expected:
            raise EvidenceError(f"snapshot content changed: {name}")


def capture(source, policy, inputs, store, scope=None):
    manifest, origins = inspect(source, policy, inputs, scope)
    identity = digest(manifest)
    destination = Path(store) / "snapshots" / identity
    if destination.exists():
        verify(destination, manifest)
        return identity, manifest, destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    signatures = {name: signature(path) for name, path in origins.items()}
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        for name, path in origins.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                os.symlink(os.readlink(path), target)
                continue
            copy_file(path, target)
        verify(temporary, manifest)
        after, _ = inspect(source, policy, inputs, scope)
        if after != manifest or any(signature(path) != signatures[name]
                                    for name, path in origins.items()):
            raise EvidenceError("source or input changed during snapshot capture")
        write_json(temporary / "manifest.json", manifest)
        # Rename the complete capture. The retained directory is never a build tree.
        os.rename(temporary, destination)
    return identity, manifest, destination


def dependency_keys(manifest, gate, policy):
    values = {"source:" + name: (digest(manifest["sources"][name])
                               if manifest["sources"].get(name) else None)
              for name in gate["sources"]}
    required = set(gate.get("inputs", []))
    if gate.get("identity_bound"):
        required.update(policy["identity_inputs"])
    for name in sorted(required):
        value = manifest["inputs"].get(name)
        values["input:" + name] = digest(value) if value else None
    return values
