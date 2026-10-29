#!/usr/bin/env python3
"""Check SPEC.md's pin syntax and paths, and write or check its parameters table.

SPEC.md cites a Lean theorem as a pin, `` `Name` (`0123456789abcdef`) ``. Lean checks the
fingerprints (`#spec_census`); this script, which needs no Lean build, checks only that a
Lean name followed by a backticked word in parentheses has 16 lowercase hex digits there. A
word with a dot is a citation, not a pin. A backticked token with a slash that ends in a slash or a file
extension is a repository path, such as `crates/nightstream/` or `Lifecycle/XOut.lean`. It
must exist under one of BASES.

Between `<!-- params -->` and `<!-- /params -->` SPEC.md holds a table that this script
generates. Its values come from the integer equations of the Lean theorems in THEOREMS and
from the header of the Lean-emitted package; the field names come from the Lean formats in
`Export/Package.lean`. Every value needs a label in LABELS and every label a value, so the
table never drops or invents a row.

  python3 scripts/check_spec.py            check pins, paths and the table
  python3 scripts/check_spec.py --write    rewrite the table (pins are never edited)
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
FPRIME = ROOT / "formal/nightstream-fprime/NightstreamFPrime"
COMPRESSION = ROOT / "formal/compression-lean/CompressionLean"
BASES = [ROOT, FPRIME, COMPRESSION]
PACKAGE = ROOT / "formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
FORMATS = FPRIME / "Export/Package.lean"
SEALED = FPRIME / "Export/Stage1/PerApplicationCanonicalPackage.lean"
BEGIN, END = "<!-- params -->", "<!-- /params -->"

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
    "productionProfile.piRlcInputs": ("—", "The Π_RLC inputs: one fresh claim and the running claims."),
    "productionProfile.piDecChildren": ("—", "The Π_DEC children, which are the next running claims."),
    "productionProfile.ccsMatrices": ("t", "The CCS matrices."),
    "NormStage.bound productionGlobalParams .fresh": ("—", "The norm bound of a fresh witness, `b`."),
    "NormStage.bound productionGlobalParams .combined": ("—", "The norm bound of a Π_RLC parent, `B`."),
    "NormStage.bound productionGlobalParams .ambient": ("—", "The ambient bound, `q/2 + 1`."),
    "productionGlobalParams.msisNormBound": ("—", "The MSIS norm of the binding reduction, `8TB`."),
    "numerator": ("—", "The numerator of the Π_CCS test error `ε_test`."),
    "ε_test": ("ε_test", "The Π_CCS test error, `numerator / q²` (`test_error_eq`).", "over_q2"),
    "sealed.schema": ("—", "The schema of the sealed package file."),
    "sealed.logicalPublicInputCount": ("—", "The public input of a claim, in field elements."),
    "package.schemaVersion": ("—", "The schema of the circuit package."),
    "package.poseidon.width": ("—", "Poseidon2 state width."),
    "package.poseidon.rate": ("—", "Poseidon2 rate."),
    "package.poseidon.digestLength": ("—", "Poseidon2 digest length."),
    "package.poseidon.initialFullRounds": ("—", "Poseidon2 full rounds before the partial rounds."),
    "package.poseidon.partialRounds": ("—", "Poseidon2 partial rounds."),
    "package.poseidon.terminalFullRounds": ("—", "Poseidon2 full rounds after the partial rounds."),
    "package.poseidon.recipesPerPermutation": ("—", "Witness recipes of the permutation template."),
    "package.poseidon.outputLocalStart": ("—", "The first output local of the permutation template."),
    "package.layout.rowCount": ("—", "Rows of the physical layout."),
    "package.layout.privateColumnCount": ("—", "Private columns of the physical layout."),
    "package.layout.constantColumn": ("—", "The constant column of the physical layout."),
    "package.layout.publicColumnCount": ("—", "Public columns of the physical layout."),
    "package.layout.totalColumnCount": ("—", "All columns of the physical layout."),
    "package.relation.rowCount": ("n", "Rows of the CCS relation."),
    "package.relation.columnCount": ("m", "Columns of the CCS relation, the logical width."),
    "package.relation.cubeVariables": ("ℓ", "Variables of the row cube: `n ≤ 2^ℓ`, and Π_CCS has `ℓ` rounds."),
    "package.relation.degreeBound": ("—", "The degree bound of a Π_CCS round polynomial."),
    "package.relation.terms": ("f", "The CCS polynomial over the matrix images `M_j z`."),
    "package.terminal.rowStart": ("—", "The first row of the terminal layout."),
    "package.terminal.rowCount": ("—", "Rows of the terminal layout."),
    "package.terminal.runningClaims": ("—", "Running claims of the terminal layout."),
    "package.terminal.freshClaims": ("—", "Fresh claims of the terminal layout."),
}
HEADER_STRUCTURES = {"poseidon": "PoseidonSchedule", "layout": "PhysicalLayout",
                     "relation": "CcsRelation", "terminal": "TerminalLayout"}


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


def format_fields(structure):
    """The fields of `structure` in the order its Lean format encodes them."""
    text = FORMATS.read_text(encoding="utf-8")
    found = re.search(rf"^def {structure}\.format : Format {structure} where\s*encode := fun value =>"
                      r"\s*\.array \[(.*?)\]\s*decode", text, re.S | re.M)
    if not found:
        sys.exit(f"spec: {rel(FORMATS)} has no array encoding for {structure}")
    return re.findall(r"value\.(\w+)", found.group(1))


def sealed_header():
    """The sealed package's schema, its circuit package and its public-input count."""
    text = SEALED.read_text(encoding="utf-8")
    if not re.search(r"\.array \[\s*\.atom sealedPackageSchema,\s*CircuitPackage\.format\.encode"
                     r".*?\.atom logicalPublicInputCount\]", text, re.S):
        sys.exit(f"spec: {rel(SEALED)} no longer encodes the schema, the circuit package, ..., "
                 f"the public-input count")
    sealed = json.loads(PACKAGE.read_text(encoding="utf-8"))
    if not (isinstance(sealed, list) and isinstance(sealed[0], int) and isinstance(sealed[-1], int)):
        sys.exit(f"spec: {rel(PACKAGE)} is not a sealed package")
    return sealed[0], sealed[1], sealed[-1]


def polynomial(terms, modulus):
    rendered = []
    for coefficient, exponents in terms:
        factors = "·".join(f"M_{j}" + (f"^{e}" if e > 1 else "") for j, e in enumerate(exponents) if e)
        sign = "−" if coefficient > modulus // 2 else "+"
        magnitude = modulus - coefficient if sign == "−" else coefficient
        rendered.append((sign, factors if magnitude == 1 else f"{magnitude}·{factors}"))
    text = " ".join(f"{sign} {term}" for sign, term in rendered)
    return text[2:] if text.startswith("+ ") else text


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
    lean = {key: value for _, key, value in found}
    modulus = lean["productionProfile.global.q"]
    schema, package, public_count = sealed_header()
    found.append((f"`{rel(PACKAGE)}`", "sealed.schema", schema))
    names = format_fields("CircuitPackage")
    if len(names) != len(package):
        sys.exit(f"spec: the package header has {len(package)} fields; CircuitPackage encodes {len(names)}")
    header = dict(zip(names, package))
    found.append((f"`{rel(PACKAGE)}`", "package.schemaVersion", header["schemaVersion"]))
    for field, structure in HEADER_STRUCTURES.items():
        value = header[field]
        if structure == "TerminalLayout":
            if value[0] != 1:
                sys.exit("spec: the package has no terminal layout")
            value = value[1]
        fields = format_fields(structure)
        if len(fields) != len(value):
            sys.exit(f"spec: {structure} encodes {len(fields)} fields; the package has {len(value)}")
        for name, item in zip(fields, value):
            key = f"package.{field}.{name}"
            if isinstance(item, int):
                found.append((f"`{rel(PACKAGE)}`", key, item))
            elif key == "package.relation.terms":
                found.append((f"`{rel(PACKAGE)}`", key, polynomial(item, modulus)))
    found.append((f"`{rel(PACKAGE)}`", "sealed.logicalPublicInputCount", public_count))
    found.append(("computed", "ε_test", (lean["numerator"], modulus ** 2)))
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
    if errors:
        sys.exit(f"spec: {len(errors)} errors")
    print(f"spec: {args.spec.name} has well-formed pins, existing paths and a current parameters table")


if __name__ == "__main__":
    main()
