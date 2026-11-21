"""Compare six compressed memory proofs produced by stock, CPU, and Metal builds.

Run python3 scripts/test_akita_compat.py on an Apple Metal host. On other hosts,
pass --cpu-only to run just the CPU comparison. Both verifiers run on the CPU.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
STOCK = "cdc29e826bb167fa58d9a14425693503f149b892"
AKITA = "d50f700fb1a292395a717cef0e6f437c7ce2cdb1"
UPSTREAM = f"git+https://github.com/LayerZero-Labs/akita.git?rev={AKITA}#{AKITA}"


def stock_checkout(destination):
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", ROOT, destination], check=True)
    has_stock = subprocess.run(["git", "cat-file", "-e", STOCK + "^{commit}"], cwd=destination,
                               stderr=subprocess.DEVNULL)
    if has_stock.returncode:
        subprocess.run(["git", "fetch", "--quiet", "--depth", "1",
                        "https://github.com/LFDT-Nightstream/Nightstream.git", STOCK], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", STOCK], cwd=destination, check=True)
    examples = destination / "crates/nightstream/examples"
    examples.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "crates/nightstream/tests/akita_compat/driver.rs", examples / "akita_compat.rs")


def build(cargo, source, target, *, reference=False, metal=False):
    features = ["--features", "nightstream/metal"] if metal else []
    graph = json.loads(subprocess.check_output([cargo, "metadata", "--locked", "--format-version", "1", *features],
                                               cwd=source))
    packages = {p["id"]: p for p in graph["packages"]}
    if reference:
        akita_sources = {p["source"] for p in packages.values() if p["name"].startswith("akita-")}
        if akita_sources != {UPSTREAM}:
            raise ValueError(f"unexpected reference Akita sources: {akita_sources}")
    else:
        check_verifier_dependencies(graph, packages)
    lines = subprocess.check_output([cargo, "build", "--release", "--locked", "-p", "nightstream",
                                     "--example", "akita_compat", "--target-dir", target,
                                     "--message-format=json-render-diagnostics", *features], cwd=source)
    executable, = (m["executable"] for m in map(json.loads, lines.splitlines())
                   if m.get("executable") and m["target"]["name"] == "akita_compat")
    return executable


def check_verifier_dependencies(graph, packages):
    edges = {n["id"]: {d["name"]: d["pkg"] for d in n["deps"]
                      if any(k["kind"] is None for k in d["dep_kinds"])} for n in graph["resolve"]["nodes"]}
    api, = (p for p in graph["workspace_members"] if packages[p]["name"] == "nightstream")
    deps = edges[edges[api]["ns_compress"]]
    pending = []
    for name in ("verifier", "params", "config", "types", "serialization"):
        package = deps[f"akita_{name}"]
        if packages[package]["name"] != f"akita-{name}" or packages[package]["source"] != UPSTREAM:
            raise ValueError(f"unexpected verifier dependency for {name}: {package}")
        pending.append(package)
    seen = set()
    while pending:
        package = pending.pop()
        if package in seen:
            continue
        seen.add(package)
        dep = packages[package]
        if dep["name"].startswith("akita-") and dep["source"] != UPSTREAM:
            raise ValueError(f"unexpected verifier dependency: {package}")
        pending.extend(edges[package].values())
    prover = packages[deps["akita_pcs"]]
    if prover["name"] != "akita-pcs" or not (prover["source"] or "").startswith("git+https://github.com/nicarq/akita.git?rev="):
        raise ValueError(f"unexpected prover dependency: {prover['id']}")


def run(binary, command, directory):
    subprocess.run([binary, command, directory], check=True, timeout=300)


def compare(cargo, stock, target, temp, engines):
    receiver = build(cargo, stock, target / "reference", reference=True)
    original = temp / "original"
    original.mkdir()
    run(receiver, "cpu", original)
    for engine in engines:
        candidate = build(cargo, ROOT, target / "candidate", metal=(engine == "metal"))
        proofs = temp / engine
        proofs.mkdir()
        run(candidate, engine, proofs)
        for index in range(6):
            name = f"{index}.proof"
            if (original / name).read_bytes() != (proofs / name).read_bytes():
                raise ValueError(f"{engine} {name} differs from the stock encoding")
        for verifier in [receiver, candidate]:
            run(verifier, "verify", proofs)
        print(f"{engine}: six identical proofs, accepted by both verifiers. "
              "Both reject each changed opening and final state.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Reference Nightstream {STOCK}, upstream Akita {AKITA}.")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "target/akita-compat")
    parser.add_argument("--cargo", default="cargo", help="Cargo or a wrapper executable")
    args = parser.parse_args()
    if not args.cpu_only and sys.platform != "darwin":
        parser.error("Metal requires macOS. Use --cpu-only to check the CPU prover.")
    with TemporaryDirectory(prefix="nightstream-proof-") as directory:
        temp = Path(directory)
        stock = temp / "stock"
        stock_checkout(stock)
        compare(args.cargo, stock, args.target_dir.resolve(), temp, ["cpu"] if args.cpu_only else ["cpu", "metal"])
