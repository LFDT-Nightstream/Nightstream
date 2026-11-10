#!/usr/bin/env python3
"""Check SPEC.md's pin syntax and paths, and write or check its parameters table.

SPEC.md cites a Lean theorem as a pin, `` `Name` (`0123456789abcdef`) ``. Lean checks the
fingerprints (`#spec_census`); this script, which needs no Lean build, checks only that a
Lean name followed by a backticked word in parentheses has 16 lowercase hex digits there. A
word with a dot is a citation, not a pin. A backticked token with a slash that ends in a slash or a file
extension is a repository path, such as `crates/nightstream/` or `Lifecycle/XOut.lean`. It
must exist under one of BASES.

Between `<!-- params -->` and `<!-- /params -->` SPEC.md holds a table that this script
generates. Its values come from the integer equations of the Lean theorems in THEOREMS, from
the number definitions in DEFINITIONS, from the headers of the Lean-emitted bounded-depth packages
in PACKAGES and from the row counts of the Lean-emitted parity file PARITY. The header field
names come from the Lean format `Package.format` in `Export/BoundedDepth/StepPackage.lean`. Every
value needs a label in LABELS and every label a value, so the table never drops or invents a
row.

The repository has one protocol specification (`AGENTS.md`, Specs). So the script also fails when
a tracked file is in a `specs/` folder, when a Markdown file outside SPEC_FILES has "spec" in its
name, or when a Lean or Rust file cites `spec §`, the citation form of the deleted spec files.

  python3 scripts/check_spec.py            check pins, paths, the table and the spec files
  python3 scripts/check_spec.py --write    rewrite the table (pins are never edited)
"""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
FPRIME = ROOT / "formal/nightstream-fprime/NightstreamFPrime"
COMPRESSION = ROOT / "formal/compression-lean/CompressionLean"
BASES = [ROOT, FPRIME, COMPRESSION]
ARTIFACTS = ROOT / "formal/nightstream-fprime/artifacts"
# The bounded-depth packages of the reference application: the light and refresh step circuits and
# the finish circuit (`Export/BoundedDepth/EmitMain.lean`).
PACKAGES = [(name, ARTIFACTS / f"nightstream-fprime-bounded-depth-{name}-v1.json")
            for name in ("light", "refresh", "finish")]
PARITY = ARTIFACTS / "nightstream-fprime-bounded-depth-parity-v1.json"
FORMATS = FPRIME / "Export/BoundedDepth/StepPackage.lean"
PARITY_SOURCE = FPRIME / "Export/BoundedDepth/Parity.lean"
BEGIN, END = "<!-- params -->", "<!-- /params -->"
# The Markdown files whose names say "spec": the protocol specification and the scope of a
# profiling tool.
SPEC_FILES = {"SPEC.md", "scripts/gpuscope/SPEC.md"}
OLD_CITATION = re.compile(r"[Ss]pec §")

PIN = re.compile(r"`((?:Export|Layout|Lifecycle|Spec|CompressionLean)\.[^`\s]*)` \(`([^`\s.]*)`\)")
FINGERPRINT = re.compile(r"[0-9a-f]{16}")
TICKED = re.compile(r"`([^`\n]+)`")
PATH = re.compile(r"[^/\s`]\S*/(\S*\.[A-Za-z][A-Za-z0-9]*)?")
HEADING = re.compile(r"^#+ .*$", re.M)

THEOREMS = [
    (FPRIME / "Spec/Profile.lean", "production_parameter_values"),
    (FPRIME / "Spec/Profile.lean", "production_norm_stages"),
    (FPRIME / "Spec/Profile.lean", "production_msis_norm_bound"),
    (FPRIME / "Lifecycle/Nifs/SelectedTestNumerator.lean", "numerator_eq"),
    (FPRIME / "Spec/Folding/Light/ErrorBudget.lean", "production_testNumerator"),
    (FPRIME / "Spec/Folding/Light/ErrorBudget.lean", "production_forkCoordinates"),
    (FPRIME / "Spec/Folding/Schedule.lean", "production_msisChainBound"),
]

# Lean definitions whose value is a number or a power `base ^ exponent`.
DEFINITIONS = [
    (FPRIME / "Spec/Folding/Schedule.lean", "productionLightFoldCap"),
    (FPRIME / "Export/BoundedDepth/StepRelation.lean", "keyWidth"),
]

# Each value's symbol and meaning. A key is the left side of a theorem's equation or a field
# path of the package header. `over_q2` shows the value as a probability `value / q^2`.
LABELS = {
    "productionProfile.global.q": ("q", "The field modulus, the Goldilocks prime."),
    "productionProfile.global.b": ("b", "The decomposition base and the fresh norm bound."),
    "productionProfile.global.k": ("k_rho", "The decomposition digits, and the Π_DEC children."),
    "productionProfile.global.maxFresh": ("—", "The largest fresh count with `(maxFresh + k_rho)·T·(b − 1) < B` (SuperNeo v1.2 Definition 22)."),
    "productionProfile.global.expansionT": ("T", "The expansion factor of the challenge set."),
    "productionProfile.global.bigB": ("B", "The combined norm bound, `b^k_rho`."),
    "productionProfile.eta": ("η", "The cyclotomic index: `R_q = F_q[X]/Φ_η`."),
    "productionProfile.ringDegree": ("d", "The ring degree, `φ(η)`."),
    "productionProfile.commitmentWidth": ("κ", "The rows of the Ajtai key."),
    "productionProfile.extensionDegree": ("—", "The degree of `K` over `F_q`."),
    "productionProfile.challengeSetBitsFloor": ("—", "A floor of `log₂ |C|`."),
    "productionProfile.piRlcInputs": ("—", "The Π_RLC inputs of a refresh fold, the fresh claim and the 16 Π_DEC children."),
    "productionProfile.piDecChildren": ("—", "The Π_DEC children of the accumulator that a refresh fold splits."),
    "productionProfile.ccsMatrices": ("t", "The CCS matrices."),
    "NormStage.bound productionGlobalParams .fresh": ("—", "The norm bound of a fresh witness, `b`."),
    "NormStage.bound productionGlobalParams .combined": ("—", "The norm bound of a Π_RLC parent, `B`."),
    "NormStage.bound productionGlobalParams .ambient": ("—", "The ambient bound, `q/2 + 1`."),
    "productionGlobalParams.msisNormBound": ("—", "The MSIS norm `8TB` of SuperNeo v1.2 Section 8, the chain norm at cap 0."),
    "numerator": ("—", "The numerator of the refresh fold's Π_CCS test error, 17 sources."),
    "ε_test": ("ε_test", "The refresh fold's Π_CCS test error, `numerator / q²`.", "over_q2"),
    "testNumerator (shape 28) 8": ("—", "The numerator of the light fold's Π_CCS test error, 2 sources."),
    "ε_test light": ("—", "The light fold's Π_CCS test error, `testNumerator / q²`.", "over_q2"),
    "forkCoordinates": ("—", "The `ρ` coordinates of a light fork. A refresh fork has 17."),
    "msisChainBound productionGlobalParams productionLightFoldCap": ("—", "The chain MSIS norm `8T(B + cap·T·(b − 1))`."),
    "productionLightFoldCap": ("cap", "The depth at which the light circuit rejects a fold."),
    "keyWidth": ("—", "The logical width of both circuit keys, `2^26`."),
}
# The header fields of a step package (`StepPackage.Package.format`) and the row counts of the
# parity file, for each circuit.
HEADER_LABELS = {
    "circuit": "The circuit index: 0 light, 1 refresh, 2 finish.",
    "publicColumns": "The public input columns, five ring elements.",
    "width": "The source width. Every source column is below it.",
    "firstFresh": "The first fresh lowering column.",
    "sboxStart": "The first S-box output column.",
    "pieces": "The pieces, which are lowered rows, ring products and Poseidon2 blocks.",
    "witness": "Witness batches.",
    "slots": "Slot runs of the low-norm layout.",
    "application[0]": "The witness word count of the application region.",
    "application[1]": "The first local of the application region.",
    "application[2]": "The local count of the application region.",
    "application[3]": "The pieces of the application region, from the first piece.",
    "application[4]": "The fresh columns of the application region, from `firstFresh`.",
}
PARITY_LABELS = {
    "relationRows": "R1CS rows of the {} relation, from the pieces and the ring products.",
    "sboxRows": "S-box rows of the {} relation, 150 for each Poseidon2 block.",
}
for circuit in ("light", "refresh", "finish"):
    LABELS.update({f"{circuit}.{field}": ("—", meaning) for field, meaning in HEADER_LABELS.items()})
for circuit in ("light", "refresh"):
    LABELS.update({f"{circuit}.{field}": ("—", meaning.format(circuit))
                   for field, meaning in PARITY_LABELS.items()})


def rel(path):
    return path.relative_to(ROOT).as_posix()


def short(path):
    """`path` relative to the F′ Lean root when it is under it, else to the repository."""
    return path.relative_to(FPRIME).as_posix() if FPRIME in path.parents else rel(path)


def theorem_values(path, theorem):
    """The `left = number` conjuncts of `theorem`'s statement, in order."""
    text = path.read_text(encoding="utf-8")
    found = re.search(rf"^theorem {re.escape(theorem)}\s*:(.*?):=", text, re.S | re.M)
    if not found:
        sys.exit(f"spec: {rel(path)} has no theorem {theorem}")
    values = []
    for part in found.group(1).split("∧"):
        equation = re.fullmatch(r"\s*(.+?)\s*=\s*([0-9]+)\s*", part, re.S)
        if not equation:
            sys.exit(f"spec: {theorem} has a conjunct that is not `name = number`: {part.strip()!r}")
        values.append((" ".join(equation.group(1).split()), int(equation.group(2))))
    return values


def definition_value(path, name):
    """The value of `def name : Nat := number` or `def name : Nat := base ^ exponent`."""
    text = path.read_text(encoding="utf-8")
    found = re.search(rf"^def {re.escape(name)} : Nat := ([0-9]+)(?: \^ ([0-9]+))?\s*$", text, re.M)
    if not found:
        sys.exit(f"spec: {rel(path)} has no number definition {name}")
    base, exponent = found.groups()
    return int(base) ** int(exponent) if exponent else int(base)


def format_fields(structure):
    """The fields of `structure` in the order its Lean format encodes them."""
    text = FORMATS.read_text(encoding="utf-8")
    found = re.search(rf"^def {structure}\.format : Format {structure} where\s*encode := fun value =>"
                      r"\s*\.array \[(.*?)\]\s*decode", text, re.S | re.M)
    if not found:
        sys.exit(f"spec: {rel(FORMATS)} has no array encoding for {structure}")
    return re.findall(r"value\.(\w+)", found.group(1))


def nested_pair(value, count):
    """The `count` numbers of a value that `pair nat (pair nat ...)` encodes."""
    items = []
    while len(items) < count - 1:
        if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], int)):
            sys.exit("spec: an application region is not a nested pair of numbers")
        items.append(value[0])
        value = value[1]
    if not isinstance(value, int):
        sys.exit("spec: an application region does not end in a number")
    return items + [value]


def package_rows(name, path):
    """The header of a step package: its numbers, the lengths of its lists, and its application
    region (`StepPackage.Package.format`)."""
    package = json.loads(path.read_text(encoding="utf-8"))
    names = format_fields("Package")
    if not isinstance(package, list) or len(names) != len(package):
        sys.exit(f"spec: {rel(path)} has {len(package)} fields; Package encodes {len(names)}")
    found = []
    for field, value in zip(names, package):
        if field == "application":
            for index, item in enumerate(nested_pair(value, 5)):
                found.append((f"`{rel(path)}`", f"{name}.application[{index}]", item))
        elif isinstance(value, int):
            found.append((f"`{rel(path)}`", f"{name}.{field}", value))
        elif isinstance(value, list):
            found.append((f"`{rel(path)}`", f"{name}.{field}", len(value)))
        else:
            sys.exit(f"spec: {rel(path)} field {field} is not a number or a list")
    return found


def parity_rows():
    """The relation rows and the S-box rows of the light and the refresh circuit (`Parity.summary`:
    its first two entries)."""
    text = PARITY_SOURCE.read_text(encoding="utf-8")
    if not re.search(r"\.array \[\.atom rows\.length, \.atom \(150 \* blocks\.length\)", text):
        sys.exit(f"spec: {rel(PARITY_SOURCE)} no longer starts a summary with the relation rows and "
                 "the S-box rows")
    if not re.search(r"\.array \[\.atom 3, summary width light\.1 .*?,\s*summary width refresh\.1",
                     text, re.S):
        sys.exit(f"spec: {rel(PARITY_SOURCE)} no longer writes version 3, then the light and the "
                 "refresh summary")
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if not (isinstance(parity, list) and parity[0] == 3):
        sys.exit(f"spec: {rel(PARITY)} is not a version 3 parity file")
    found = []
    for name, summary in (("light", parity[1]), ("refresh", parity[2])):
        found.append((f"`{rel(PARITY)}`", f"{name}.relationRows", summary[0]))
        found.append((f"`{rel(PARITY)}`", f"{name}.sboxRows", summary[1]))
    return found


def neg_log2_floor(numerator, denominator):
    """`−log₂(numerator/denominator)` rounded down to two decimals, exactly."""
    j = max(0, math.floor(100 * (math.log2(denominator) - math.log2(numerator))))
    power_n, power_d = numerator ** 100, denominator ** 100
    while (power_n << j) > power_d:
        j -= 1
    while (power_n << (j + 1)) <= power_d:
        j += 1
    return f"{j // 100}.{j % 100:02d}"


def rows():
    """(source, key, shown value) for every value of the table, in order."""
    found = []
    for path, theorem in THEOREMS:
        for key, value in theorem_values(path, theorem):
            found.append((f"`{short(path)}`, `{theorem}`", key, value))
    for path, name in DEFINITIONS:
        found.append((f"`{short(path)}`, `{name}`", name, definition_value(path, name)))
    lean = {key: value for _, key, value in found}
    modulus = lean["productionProfile.global.q"]
    found.append(("computed", "ε_test", (lean["numerator"], modulus ** 2)))
    found.append(("computed", "ε_test light", (lean["testNumerator (shape 28) 8"], modulus ** 2)))
    for name, path in PACKAGES:
        found.extend(package_rows(name, path))
    found.extend(parity_rows())
    return found


def table():
    found = rows()
    keys = [key for _, key, _ in found]
    if sorted(set(keys)) != sorted(keys):
        sys.exit("spec: a table key appears twice")
    if set(keys) != set(LABELS):
        sys.exit(f"spec: values and LABELS disagree: unlabelled {sorted(set(keys) - set(LABELS))}, "
                 f"labelled but absent {sorted(set(LABELS) - set(keys))}")
    lines = ["| Source | Lean | Symbol | Value | Meaning |", "|---|---|---|---|---|"]
    for source, key, value in found:
        symbol, meaning, *kind = LABELS[key]
        if kind == ["over_q2"]:
            numerator, denominator = value
            shown = f"{numerator}/q² ≈ 2^−{neg_log2_floor(numerator, denominator)}"
        else:
            shown = str(value)
        symbol = symbol if symbol == "—" else f"`{symbol}`"
        lines.append(f"| {source} | `{key}` | {symbol} | {shown} | {meaning} |")
    return "\n".join([
        "Generated by `scripts/check_spec.py`. Do not edit by hand: run "
        "`python3 scripts/check_spec.py --write`.",
        "",
        *lines,
        "",
        "A probability is shown as `2^−x`, where `x` is `−log₂` of the exact fraction rounded down "
        "to two decimals, so the shown bound is never below the true value.",
    ])


def section(text, position):
    heads = [heading for heading in HEADING.finditer(text) if heading.start() <= position]
    line = text.count("\n", 0, position) + 1
    return f"{heads[-1].group(0)!r} (line {line})" if heads else f"line {line}"


def pin_errors(text):
    for match in PIN.finditer(text):
        name, fingerprint = match.groups()
        if not FINGERPRINT.fullmatch(fingerprint):
            yield f"`{name}` at {section(text, match.start())} has `{fingerprint}`, not 16 hex digits"


def path_errors(text):
    for match in TICKED.finditer(text):
        token = match.group(1)
        if not PATH.fullmatch(token):
            continue
        if not any((base / token).is_dir() if token.endswith("/") else (base / token).exists()
                   for base in BASES):
            yield (f"`{token}` at {section(text, match.start())} is not a path under "
                   f"{', '.join(rel(base) or '.' for base in BASES)}")


def tracked_files():
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [path for path in listing.stdout.decode("utf-8").split("\0") if path]


def layout_errors(paths):
    for path in paths:
        parts = Path(path).parts
        if "specs" in parts[:-1]:
            yield f"`{path}` is in a `specs/` folder; protocol text goes in SPEC.md (AGENTS.md, Specs)"
        elif parts[-1].lower().endswith(".md") and "spec" in parts[-1].lower() and path not in SPEC_FILES:
            yield f"`{path}` is a separate spec file; protocol text goes in SPEC.md (AGENTS.md, Specs)"


def citation_errors(files):
    for path, text in files:
        for match in OLD_CITATION.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            yield f"{path}:{line} cites `spec §`; cite a SPEC.md section or algorithm, or a Lean name"


def repository_errors():
    paths = tracked_files()
    sources = [(path, (ROOT / path).read_text(encoding="utf-8", errors="replace")) for path in paths
               if path.endswith((".lean", ".rs")) and (ROOT / path).is_file()]
    return [*layout_errors(paths), *citation_errors(sources)]


def main():
    parser = argparse.ArgumentParser(description="Check SPEC.md's pins, paths and parameters table.")
    parser.add_argument("--write", action="store_true", help="rewrite the parameters table")
    parser.add_argument("--spec", type=Path, default=SPEC, help="the spec to read (default: SPEC.md)")
    args = parser.parse_args()
    text = args.spec.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        sys.exit(f"spec: {args.spec.name} needs one {BEGIN} … {END} block")
    head, rest = text.split(BEGIN)
    _, tail = rest.split(END)
    new = f"{head}{BEGIN}\n{table()}\n{END}{tail}"
    if args.write:
        args.spec.write_text(new, encoding="utf-8")
        print(f"spec: wrote the parameters table of {args.spec.name}")
        return
    errors = [*pin_errors(text), *path_errors(text)]
    if new != text:
        errors.append("the parameters table is stale; run python3 scripts/check_spec.py --write")
    for error in errors:
        print(f"{args.spec.name}: {error}")
    repository = repository_errors()
    for error in repository:
        print(f"repository: {error}")
    if errors or repository:
        sys.exit(f"spec: {len(errors) + len(repository)} errors")
    print(f"spec: {args.spec.name} has well-formed pins, existing paths and a current parameters table, "
          "and no other spec file or `spec §` citation exists")


if __name__ == "__main__":
    main()
