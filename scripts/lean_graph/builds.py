"""Retain checked Lean build products outside candidate execution directories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .snapshot import (EvidenceError, copy_file, digest, file_entry, files,
                       no_symlinks, read_json, safe_relative, write_json)


def build_files(directory):
    return {path.relative_to(directory).as_posix(): file_entry(path)
            for path in files(directory, skip_build=False)}


def build_directory(work, context):
    return Path(work) / "source" / safe_relative(context["command"]["cwd"]) / ".lake/build"


def cache_root(store, authority):
    root = Path(authority.directory if authority else store) / "builds"
    no_symlinks(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def restore(context, work, store, authority):
    """A missing or invalid cache is a cold build, never passing evidence."""
    directory = cache_root(store, authority) / digest(context)
    if not directory.exists():
        return "miss"
    try:
        no_symlinks(directory)
        record = (authority.read(directory / "record.json") if authority else
                  read_json(directory / "record.json")["record"])
        if record["context"] != context:
            raise EvidenceError("build context changed")
        products = directory / "products"
        if build_files(products) != record["files"]:
            raise EvidenceError("cached build products changed")
        if any("link" in item for item in record["files"].values()):
            raise EvidenceError("cached build products contain symlinks")
        destination = build_directory(work, context)
        no_symlinks(destination)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(products, destination, copy_function=copy_file)
        if build_files(destination) != record["files"]:
            raise EvidenceError("restored build products changed")
        return "hit"
    except (EvidenceError, OSError, KeyError, TypeError, ValueError):
        destination = build_directory(work, context)
        no_symlinks(destination)
        if destination.exists():
            shutil.rmtree(destination)
        return "invalid"


def stage(context, work, directory):
    """Freeze products immediately after the build, before later commands run."""
    source = build_directory(work, context)
    if not source.exists():
        return None
    expected = build_files(source)
    if any("link" in item for item in expected.values()):
        return None
    directory = Path(directory)
    shutil.copytree(source, directory / "products", copy_function=copy_file)
    if build_files(directory / "products") != expected or build_files(source) != expected:
        raise EvidenceError("build products changed while being retained")
    return {"context": context, "files": expected}


def publish(staged, record, store, authority):
    root = cache_root(store, authority)
    destination = root / digest(record["context"])
    no_symlinks(destination)
    write_json(Path(staged) / "record.json",
               authority.sign(record) if authority else {"record": record})
    if destination.exists():
        shutil.rmtree(destination)
    os.rename(staged, destination)
