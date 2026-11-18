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

SPEC.md also holds the table of verifier entry points between `<!-- entry-points -->` and
`<!-- /entry-points -->`. Each public `verify…` function of `crates/nightstream/src/` needs a row,
and each row cites exactly one pinned theorem of the Lean-emitted premise list PREMISES, or it
starts with `Open:`. Each claim of PREMISES needs a pin in SPEC.md.

Between `<!-- papers -->` and `<!-- /papers -->` SPEC.md lists each cited paper number. A citation
in PAPER_DOCUMENTS names a protocol paper, a paper from the table or an ePrint number, then in the same sentence a kind
of ITEM and its number. Each citation needs a row, and each row a citation.

The prose marks a derived number "(computed)". `computed_claims` recomputes these numbers, and some other
derived numbers, from the Lean values of the table, Lean definitions, the Lean-emitted artifacts
and the field constant `Q` of `crates/neo-params`.
Each claim is a template of SPEC.md text with `{}` for each number. The text must hold each
template, and each number must be its value: an integer exactly, and a decimal within one unit of
its last digit. Each "(computed)" marker must be in the text of a claim or of an UNCHECKED entry,
whose reason the script prints.

  python3 scripts/check_spec.py            check pins, paths, the table, the computed values and
                                           the spec files
  python3 scripts/check_spec.py --write    rewrite the table (pins are never edited)
"""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
FPRIME = ROOT / "formal/nightstream-fprime/NightstreamFPrime"
BASES = [ROOT, FPRIME]
ARTIFACTS = ROOT / "formal/nightstream-fprime/artifacts"
# `Export/BoundedDepth/EmitMain.lean`
PACKAGES = [(name, ARTIFACTS / f"nightstream-fprime-bounded-depth-{name}-v1.json")
            for name in ("light", "refresh")]
PARITY = ARTIFACTS / "nightstream-fprime-bounded-depth-parity-v1.json"
FORMATS = FPRIME / "Export/BoundedDepth/StepPackage.lean"
PARITY_SOURCE = FPRIME / "Export/BoundedDepth/Parity.lean"
BEGIN, END = "<!-- params -->", "<!-- /params -->"
ENTRY_BEGIN, ENTRY_END = "<!-- entry-points -->", "<!-- /entry-points -->"
PREMISES = ARTIFACTS / "nightstream-fprime-premises-v1.tsv"
ENTRY_FUNCTION = re.compile(r"^\s*pub fn (verify\w*)\(", re.M)
SPEC_FILES = {"SPEC.md", "scripts/gpuscope/SPEC.md"}
OLD_CITATION = re.compile(r"[Ss]pec §")
PAPERS_BEGIN, PAPERS_END = "<!-- papers -->", "<!-- /papers -->"
PAPER_DOCUMENTS = [ROOT / "formal/nightstream-fprime/SECURITY_MODEL.md",
                   *sorted((ROOT / "decisions").glob("*.md")), *sorted((FPRIME / "Claims").glob("*.lean"))]
PROTOCOL_PAPERS = {"SuperNeo", "HyperNova", "Nebula", "Cyclo", "Akita"}
NUMBER = r"(?:[0-9]+|[A-Z])(?:\.[0-9]+)*"
ITEM = (r"\b(Theorem|Lemma|Definition|Construction|Section|Assumption|Corollary|Proposition|Remark|Figure"
        rf"|Appendix)s? ({NUMBER}(?:(?:,? and | to |, |/){NUMBER})*)\b")
DOCSTRING = re.compile(r"/-[-!](.*?)-/", re.S)
FOLD_GATE = ARTIFACTS / "nightstream-fprime-fold-gate-parity-v1.json"
NEBULA_GATE = ARTIFACTS / "nightstream-fprime-nebula-gate-parity-v1.json"
COMPRESSION = ARTIFACTS / "nightstream-fprime-compression-parity-v1.json"
PLAN = FPRIME / "Spec/Nebula/Plan.lean"
PRODUCTION = FPRIME / "Spec/Compression/Production.lean"
PARAMS = ROOT / "crates/neo-params/src/lib.rs"
POSEIDON2 = FPRIME / "Spec/Poseidon2.lean"
COMPUTED = re.compile(r"\((?:both )?computed\b|That fact is computed")
COMPUTED_NUMBER = r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)"
UNCHECKED = [
    ("probability below `1/5^64 + 2^−173` (computed)",
     "a sum of `1/|C|` and the bound of `PiRlcSampler.distance_lt`, which no artifact holds"),
    ("`−log₂` of the sum in `f64` is 119 (computed)",
     "an `f64` value of the claims of a Rust test, which that test checks"),
]

PIN = re.compile(r"`((?:Claims|Export|Layout|Lifecycle|Spec)\.[^`\s]*)` \(`([^`\s.]*)`\)")
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

DEFINITIONS = [
    (FPRIME / "Spec/Folding/Schedule.lean", "productionLightFoldCap"),
    (FPRIME / "Export/BoundedDepth/StepRelation.lean", "keyWidth"),
    (FPRIME / "Spec/Nebula/SegmentGate.lean", "secureTuples"),
    (FPRIME / "Spec/Nebula/SegmentGate.lean", "gateBits"),
]

LABELS = {
    "productionProfile.global.q": ("q", "The field modulus, `2^64 − 59`."),
    "productionProfile.global.b": ("b", "The decomposition base and the fresh norm bound."),
    "productionProfile.global.k": ("k_rho", "The decomposition digits, and the Π_DEC children."),
    "productionProfile.global.maxFresh": ("—", "The largest fresh count with `(maxFresh + k_rho)·T·(b − 1) < B` (SuperNeo v1.2 Definition 22)."),
    "productionProfile.global.expansionT": ("T", "The expansion factor of the challenge set."),
    "productionProfile.global.bigB": ("B", "The combined norm bound, `b^k_rho`."),
    "productionProfile.eta": ("η", "The cyclotomic index: `R_q = F_q[X]/Φ_η`, and `Φ_128 = X^64 + 1`."),
    "productionProfile.ringDegree": ("d", "The ring degree, `φ(η)`."),
    "productionProfile.commitmentWidth": ("κ", "The rows of the Ajtai key."),
    "productionProfile.extensionDegree": ("—", "The degree of `K` over `F_q`."),
    "productionProfile.challengeSetBitsFloor": ("—", "A floor of `log₂ |C|`."),
    "productionProfile.piRlcInputs": ("—", "The Π_RLC inputs of a refresh fold, the fresh claim and the `k_rho` Π_DEC children."),
    "productionProfile.piDecChildren": ("—", "The Π_DEC children of the accumulator that a refresh fold splits."),
    "productionProfile.ccsMatrices": ("t", "The CCS matrices."),
    "NormStage.bound productionGlobalParams .fresh": ("—", "The norm bound of a fresh witness, `b`."),
    "NormStage.bound productionGlobalParams .combined": ("—", "The norm bound of a Π_RLC parent, `B`."),
    "NormStage.bound productionGlobalParams .ambient": ("—", "The ambient bound, `q/2 + 1`."),
    "productionGlobalParams.msisNormBound": ("—", "The MSIS norm `8TB` of SuperNeo v1.2 Section 8, the chain norm at cap 0."),
    "numerator": ("—", "The numerator of the refresh fold's Π_CCS test error, `1 + k_rho` sources."),
    "ε_test": ("ε_test", "The refresh fold's Π_CCS test error, `numerator / q²`."),
    "testNumerator (shape 28) 8": ("—", "The numerator of the light fold's Π_CCS test error, 2 sources."),
    "ε_test light": ("—", "The light fold's Π_CCS test error, `testNumerator / q²`."),
    "forkCoordinates": ("—", "The `ρ` coordinates of a light fork. A refresh fork has `1 + k_rho`."),
    "msisChainBound productionGlobalParams productionLightFoldCap": ("—", "The chain MSIS norm `8T(B + cap·T·(b − 1))`."),
    "productionLightFoldCap": ("cap", "The depth at which the light circuit rejects a fold."),
    "keyWidth": ("—", "The logical width of both circuit keys, `2^26`."),
    "secureTuples": ("—", "The bound of section 7.1, rule 6 on `m_mem = R + M + N · B_ops`."),
    "gateBits": ("—", "The bits that the memory gate certifies for all segments of each fixed WASM size (section 7.1)."),
}
PROBABILITY_OVER_Q2 = {"ε_test", "ε_test light"}
HEADER_LABELS = {
    "circuit": "The circuit index: 0 light, 1 refresh.",
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
for circuit in ("light", "refresh"):
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
    if not re.search(r"\.array \[\.atom 4, summary width light\.1 .*?,\s*summary width refresh\.1",
                     text, re.S):
        sys.exit(f"spec: {rel(PARITY_SOURCE)} no longer writes version 4, then the light and the "
                 "refresh summary")
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if not (isinstance(parity, list) and parity[0] == 4):
        sys.exit(f"spec: {rel(PARITY)} is not a version 4 parity file")
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


def table(found):
    keys = [key for _, key, _ in found]
    if sorted(set(keys)) != sorted(keys):
        sys.exit("spec: a table key appears twice")
    if set(keys) != set(LABELS):
        sys.exit(f"spec: values and LABELS disagree: unlabelled {sorted(set(keys) - set(LABELS))}, "
                 f"labelled but absent {sorted(set(LABELS) - set(keys))}")
    lines = ["| Source | Lean | Symbol | Value | Meaning |", "|---|---|---|---|---|"]
    for source, key, value in found:
        symbol, meaning = LABELS[key]
        if key in PROBABILITY_OVER_Q2:
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


def artifact(path, schema):
    """A Lean-emitted JSON file of schema `schema`: a list that starts with it, or an object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    found = value["schema"] if isinstance(value, dict) else value[0]
    if found != schema:
        sys.exit(f"spec: {rel(path)} has schema {found}, not {schema}")
    return value


def lean_number(path, pattern):
    found = re.search(pattern, path.read_text(encoding="utf-8"), re.M)
    if not found:
        sys.exit(f"spec: {rel(path)} has no line that matches {pattern!r}")
    return int(found.group(1))


def only(values, what):
    found = set(values)
    if len(found) != 1:
        sys.exit(f"spec: {what} has the values {sorted(found)}, not one value")
    return found.pop()


def neg_log2(value):
    return math.log2(value.denominator) - math.log2(value.numerator)


def little_endian(words):
    return sum(word << (64 * index) for index, word in enumerate(words))


def computed_claims(lean):
    """(template, values) for each derived number of SPEC.md prose, in the order of the text."""
    q, b, k, T, B, max_fresh = (lean[f"productionProfile.global.{name}"]
                                for name in ("q", "b", "k", "expansionT", "bigB", "maxFresh"))
    d, kappa, t = (lean[f"productionProfile.{name}"]
                   for name in ("ringDegree", "commitmentWidth", "ccsMatrices"))
    cap, public = lean["productionLightFoldCap"], lean["light.publicColumns"]
    refresh, light = lean["numerator"], lean["testNumerator (shape 28) 8"]
    chain = lean["msisChainBound productionGlobalParams productionLightFoldCap"]
    _, ell, degree, _, census, bits, count, denominator = artifact(FOLD_GATE, 1)
    _, tuples, _, sizes, _ = artifact(NEBULA_GATE, 1)
    compression = artifact(COMPRESSION, 1)
    production = compression["production"]
    wts = lean_number(PLAN, r"^  fieldEncoding : p\.wTs ≤ ([0-9]+)$")
    rate = definition_value(POSEIDON2, "rate")
    if 2 ** 64 - lean_number(PARAMS, r"^    pub const Q: u64 = 0u64\.wrapping_sub\(([0-9]+)\);$") != q:
        sys.exit(f"spec: `Q` of {rel(PARAMS)} is not Lean's q")

    def accumulator(depth):  # `Spec.GlobalParams.accumulatorBound`
        return (1 + depth + k) * T * (b - 1)

    if compression["bounds"] != [accumulator(depth) for depth in range(cap + 1)]:
        sys.exit(f"spec: the bounds of {rel(COMPRESSION)} are not (1 + j + k_rho)·T·(b − 1)")
    recovered = B + cap * T * (b - 1)
    if 8 * T * recovered != chain:
        sys.exit("spec: Lean's msisChainBound is not 8·T·(B + cap·T·(b − 1))")
    table_max = lean_number(PRODUCTION, r"^    2 \* productionGlobalParams\.accumulatorBound depth \+ 1 "
                                        r"≤ ([0-9]+) :=")
    if table_max < 2 * accumulator(cap) + 1:
        sys.exit("spec: the bound of `Spec.Compression.Production.slot_table_le` is below "
                 "2·accumulatorBound(cap) + 1")

    n_light, n_refresh = (only((entry["layout"]["witness_vars"] for entry in production
                                if entry["pc"] == pc), f"`n` at pc {pc}") for pc in (1, 2))
    inverse_vars = only((entry["layout"]["inverse_vars"] for entry in production), "the variables of V")
    key_vars = compression["key_blocks"].bit_length() - 1
    lanes, lookups = compression["lane_vars"], 2 ** compression["lookup_vars"]
    error_bits = only((entry["error"]["bits"] for entry in production), "error_bits")
    counts = [entry["error"]["count"] for entry in production]
    sums = [Fraction(int(entry["error"]["numerator"]), int(entry["error"]["denominator"]))
            for entry in production]
    worst = [cell for cell in production[sums.index(max(sums))]["layout"]["cells"] if cell["table"]]
    lookup_error = sum(Fraction(2 ** cell["witness"][1] + 2 * cell["table"][0] + 1, q) ** lookups
                       for cell in worst)

    def batch(claims):  # SPEC.md section 3, the batch of the equations
        return kappa + 1 + t + max(d - 1, compression["public_cells"] - 1) + claims - 1

    rest = batch(3) + 4 + 2 * (lanes + key_vars)
    other = rest + 4 * n_refresh
    lean_other = batch(3) + 2 + n_refresh + 3 + 2 + 2

    family = 2 * (1 + t) * d
    slot_words = kappa * d + family + 2 * ell + public + 1
    slot_bytes = 8 + 8 * (kappa * d + public + 2 * ell + family)
    fresh_bytes = 8 * (kappa * d + 2 * ell + family) + 32
    round_bytes = 8 * 2 * (degree + 1) * ell
    after = [8 + 16 * 4 * n_refresh, 16 + 16 * lookups + 16, 8 + 16 * 3 * (lanes + key_vars)]

    wasm_bits = only([size[8] for size in sizes] + [lean["gateBits"]], "the bits of the WASM sizes")
    micro, *larger = (size[5] * size[6] for size in sizes)
    q2 = q * q
    return [
        ("`q = 2^64 − {} = {}`, a prime", [2 ** 64 - q, q]),
        ("`(Q + 1) · {}/2^128` for light and `(Q + 1) · {}/2^128` for refresh", [light + 1, refresh + 1]),
        ("with norm below the chain norm {} (", [chain]),
        ("has probability at most `(Q_η + 1) · 2^−{}`", [neg_log2(Fraction(2 * tuples, q2))]),
        ("It has `ℓ = {}` rounds", [ell]),
        ("The light relation has `{} + {} = {}` rows and the refresh relation has `{} + {} = {}` rows "
         "(computed", [lean["light.relationRows"], lean["light.sboxRows"],
                       lean["light.relationRows"] + lean["light.sboxRows"], lean["refresh.relationRows"],
                       lean["refresh.sboxRows"], lean["refresh.relationRows"] + lean["refresh.sboxRows"]]),
        ("This bound is below `B` for each `j ≤ {}`",
         [max(depth for depth in range(B) if accumulator(depth) < B)]),
        ("So each recovered accumulator is at most `B + cap·T·(b − 1) = {}` (computed", [recovered]),
        ("`msisChainBound = 8·T·(B + cap·T·(b − 1)) = {}`, about `2^{}` (computed), in place of "
         "`8TB = {}`", [chain, math.log2(chain), lean["productionGlobalParams.msisNormBound"]]),
        ("`ε_test = (ℓ · 8 + {})/q² = {}/q²`", [refresh - 8 * ell, refresh]),
        ("where {} is `{} − 1 + ℓ`", [refresh - 8 * ell, refresh - 8 * ell + 1 - ell]),
        ("so its numerator is `ℓ · 8 + {} − 1 + ℓ = {}`", [light - 9 * ell + 1, light]),
        ("and a refresh fork has {} (", [1 + k]),
        ("The census counts the {} fresh sources that the Π_RLC guard `(K + k_rho) · T · (b − 1) < B` "
         "admits, so `N = ℓ · 8 + ({} + 127 + 4480 − 1 + ℓ) = {}`", [max_fresh, max_fresh, census]),
        ("This is not below Lean's {}.", [refresh]),
        ("The bits are {} at the counts of each step relation", [bits]),
        ("so the statistical term of each fold is at most `(Q + 1)/2^{}`", [bits]),
        ("The error is about `2^−{}`, and the refresh term at `Q = 0` is about `2^−{}` (computed). The "
         "gate refuses a caller minimum above these bits or above the census bits, which are also {} "
         "here.", [neg_log2(Fraction(little_endian(count), little_endian(denominator))),
                   neg_log2(Fraction(refresh + 1, 2 ** 128)), bits]),
        ("pass {} bits and refuse {}", [bits, bits + 1]),
        ("| `n` | {} or {} | The variables of `W` for the reference application: {} when the light "
         "circuit ran last, and {} when the refresh circuit ran last (computed",
         [n_light, n_refresh, n_light, n_refresh]),
        ("| variables of `V` | {} | The value for the reference application (computed", [inverse_vars]),
        ("| `T_s` | `2 · accumulatorBound(depth_s) + 1`, at most {} |", [table_max]),
        ("`(κ + 1 + t + max(63, m − 1) + s − 1)/q²`, which is `{}/q²`, or `{}/q²` in the light mode",
         [batch(3), batch(2)]),
        ("For the reference application, `N_0 = 2^{}`, `N_1 = 2^{}` and `n_A = {}`. At `L = {}` and "
         "`n = {}`, the lookups give about `2^−{}`, the other terms give `({} + 4n)/q² = {}/q²`, about "
         "`2^−{}`, and the sum is about `2^−{}` (computed",
         [worst[0]["witness"][1], worst[1]["witness"][1], key_vars, lookups, n_refresh,
          neg_log2(lookup_error), rest, other, neg_log2(Fraction(other, q2)), neg_log2(max(sums))]),
        ("`count` is between {} and {} and `error_bits` is {} (computed", [min(counts), max(counts), error_bits]),
        ("The gate of the folds already holds that minimum to {} bits.", [bits]),
        ("for each of the {} claim sets of the reference application", [len(production)]),
        ("On these sets the exact sum is between `2^−{}` and `2^−{}` and `error_bits` is {}, so the gate "
         "loses no integer bit there (computed", [neg_log2(min(sums)), neg_log2(max(sums)), error_bits]),
        ("Lean emits the same values for the {} claim sets of the light mode", [len(compression["light"])]),
        ("At the production shapes, `span ≤ {}`, and the terms other than the lookups are at most "
         "`{}/q²`, about `2^−{}` (computed", [n_refresh, lean_other, neg_log2(Fraction(lean_other, q2))]),
        ("Without the domain, a slot is {} words (", [slot_words]),
        ("With the domain, a slot digest absorbs {} words in {} blocks (computed",
         [slot_words + rate, (slot_words + rate) // rate + 1]),
        ("The parts before the Akita commitments have `96 + 2 · {} + {} + {} = {}` bytes, or `{}` in the "
         "light mode. For the reference application at `n = {}` and `n_A = {}`, the parts after the "
         "commitments, without the opening, have `{} + {} + {} + 16 + 8 = {}` bytes (both computed",
         [slot_bytes, fresh_bytes, round_bytes, 96 + 2 * slot_bytes + fresh_bytes + round_bytes,
          96 + slot_bytes + fresh_bytes + round_bytes, n_refresh, key_vars, *after, sum(after) + 24]),
        ("about `2^−{}` with both slots (computed", [neg_log2(Fraction(lookups * 2 * table_max, q))]),
        ("`W_ts ≤ {}`", [wts]),
        ("A value has 32 bits and a timestamp has at most {} bits, so `v + 2^32 · t` is below `2^{}`",
         [wts, 32 + wts]),
        ("**Security:** `R + M + N · B_ops ≤ {}`. This holds exactly when the fingerprint error "
         "`2(R + M + N · B_ops)/q²` is at most `2^−{}`", [tuples, neg_log2(Fraction(2 * tuples, q2))]),
        ("certifies exactly {} bits (", [wasm_bits]),
        ("which is {} for Micro and {} for Small, Medium and Large", [micro, only(larger, "the steps of a size")]),
        ("Extraction recovers accumulators up to `B + {}·T·(b − 1)`", [cap]),
        ("binding needs MSIS at the chain norm {} (section 3)", [chain]),
    ]


def agrees(shown, value):
    if isinstance(value, int):
        return "." not in shown and int(shown.replace(",", "")) == value
    return abs(float(shown) - value) < 10 ** -len(shown.partition(".")[2])


def formatted(value, shown):
    return f"{value:,}" if isinstance(value, int) else f"{value:.{len(shown.partition('.')[2])}f}"


def computed_errors(text, checked):
    """Each claim's text is in SPEC.md with its values, and each "(computed)" marker is in the text
    of a claim or of an UNCHECKED entry."""
    covered = []
    for template, values in checked:
        if template.count("{}") != len(values):
            sys.exit(f"spec: the claim {template!r} has {len(values)} values for {template.count('{}')} numbers")
        pattern = re.escape(template).replace(r"\{\}", COMPUTED_NUMBER)
        matches = list(re.finditer(pattern, text))
        if not matches:
            yield f"no text matches the computed claim {template!r} of scripts/check_spec.py"
        for match in matches:
            covered.append(match.span())
            for shown, value in zip(match.groups(), values):
                if not agrees(shown, value):
                    yield (f"{section(text, match.start())} has {shown} where the sources give "
                           f"{formatted(value, shown)}, in {template!r}")
    for template, _ in UNCHECKED:
        matches = [match.span() for match in re.finditer(re.escape(template), text)]
        if not matches:
            yield f"no text matches the unchecked entry {template!r} of scripts/check_spec.py"
        covered.extend(matches)
    for marker in COMPUTED.finditer(text):
        if not any(start <= marker.start() < end for start, end in covered):
            yield (f"the computed value at {section(text, marker.start())} is not recomputed: add it to "
                   "`computed_claims` or UNCHECKED in scripts/check_spec.py")


def section(text, position):
    heads = [heading for heading in HEADING.finditer(text) if heading.start() <= position]
    line = text.count("\n", 0, position) + 1
    return f"{heads[-1].group(0)!r} (line {line})" if heads else f"line {line}"


def pin_errors(text):
    for match in PIN.finditer(text):
        name, fingerprint = match.groups()
        if not FINGERPRINT.fullmatch(fingerprint):
            yield f"`{name}` at {section(text, match.start())} has `{fingerprint}`, not 16 hex digits"


def claims():
    return {line.split("\t")[0] for line in PREMISES.read_text(encoding="utf-8").splitlines()}


def claim_pin_errors(text):
    """Each claim has a pin, so the census sees each change of its statement."""
    pinned = {name for name, _ in PIN.findall(text)}
    for claim in sorted(claims() - pinned):
        yield (f"claim `{claim}` has no pin: cite it as `{claim}` (`0000000000000000`) where SPEC.md "
               "restates it, then run formal/nightstream-fprime/scripts/validate.sh repin")


def citations(text, citation_pattern):
    """(paper, item) for each paper number of `text`. A sentence cites the last paper that it names
    before the item."""
    text = TICKED.sub("`…`", text)
    for block in re.split(r"\n\s*\n|\n\s*(?=[-*] |\d+\. )|\|", text):
        for sentence in re.split(r"(?<=[.;!?])\s+(?=[A-Z(*`])", " ".join(block.split())):
            paper = None
            for match in citation_pattern.finditer(sentence):
                name, kind, numbers = match.groups()
                if name is not None:
                    paper = name
                elif paper:
                    yield from ((paper, f"{kind} {number}") for number in re.findall(NUMBER, numbers))


def paper_errors(text):
    """Each cited paper number is a row of the papers table, and each row is cited."""
    if text.count(PAPERS_BEGIN) != 1 or text.count(PAPERS_END) != 1:
        yield f"SPEC.md needs one {PAPERS_BEGIN} … {PAPERS_END} table"
        return
    head, rest = text.split(PAPERS_BEGIN)
    table, tail = rest.split(PAPERS_END)
    rows = {}
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"\d{4}/\d+", cells[1]):
            rows[(cells[1], cells[3])] = cells[0]
    eprints = {paper: eprint for (eprint, _), paper in rows.items()}
    for paper, eprint in list(eprints.items()):
        shorthand = re.sub(r" v\d+(?:\.\d+)+$", "", paper)
        eprints.setdefault(shorthand, eprint)
    papers = [r"\bePrint \d{4}/\d+\b", *(rf"(?<!\w){re.escape(name)}(?!\w)"
              for name in sorted(PROTOCOL_PAPERS | eprints.keys(), key=len, reverse=True))]
    citation_pattern = re.compile(rf"({'|'.join(papers)})|{ITEM}")
    documents = [("SPEC.md", head + tail)]
    for path in PAPER_DOCUMENTS:
        document = path.read_text(encoding="utf-8")
        if path.suffix == ".lean":
            document = "\n\n".join(DOCSTRING.findall(document))
        documents.append((rel(path), document))
    cited = set()
    for name, document in documents:
        for paper, item in citations(document, citation_pattern):
            key = (eprints.get(paper, paper.removeprefix("ePrint ")), item)
            if key not in rows and key not in cited:
                yield f"{name} cites {paper} {item}, which is not a row of the papers table"
            cited.add(key)
    for eprint, item in sorted(rows.keys() - cited):
        yield f"the papers table lists {rows[(eprint, item)]} {item}, which no document cites"


def path_errors(text):
    for match in TICKED.finditer(text):
        token = match.group(1)
        if not PATH.fullmatch(token):
            continue
        if not any((base / token).is_dir() if token.endswith("/") else (base / token).exists()
                   for base in BASES):
            yield (f"`{token}` at {section(text, match.start())} is not a path under "
                   f"{', '.join(rel(base) or '.' for base in BASES)}")


def entry_point_errors(text):
    """Each public verifier of the `nightstream` crate has one row, and each row has one pinned
    theorem of the premise list or starts with `Open:`."""
    if text.count(ENTRY_BEGIN) != 1 or text.count(ENTRY_END) != 1:
        yield f"SPEC.md needs one {ENTRY_BEGIN} … {ENTRY_END} table"
        return
    rows = {}
    for line in text.split(ENTRY_BEGIN)[1].split(ENTRY_END)[0].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        found = re.match(r"`(crates/[^`]+)`, (?:`(\w+)`)?", cells[0])
        if len(cells) == 3 and found:
            rows[(found.group(1), found.group(2))] = cells[2]
    listed = claims()
    for (path, name), theorem in rows.items():
        pins = PIN.findall(theorem)
        if theorem.startswith("Open:"):
            continue
        if len(pins) != 1:
            yield f"entry point `{path}` `{name}` needs exactly one pinned theorem, or `Open:`"
        elif pins[0][0] not in listed:
            yield (f"`{pins[0][0]}` of entry point `{path}` `{name}` is not in {rel(PREMISES)}: "
                   "add it to tests/PremiseCensus.lean")
    for source in sorted((ROOT / "crates/nightstream/src").rglob("*.rs")):
        for name in ENTRY_FUNCTION.findall(source.read_text(encoding="utf-8")):
            if (rel(source), name) not in rows:
                yield f"`{rel(source)}` `{name}` is a public verifier without an entry-point row"


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
    found = rows()
    new = f"{head}{BEGIN}\n{table(found)}\n{END}{tail}"
    if args.write:
        args.spec.write_text(new, encoding="utf-8")
        print(f"spec: wrote the parameters table of {args.spec.name}")
        return
    checked = computed_claims({key: value for _, key, value in found})
    errors = [*pin_errors(text), *claim_pin_errors(text), *path_errors(text), *entry_point_errors(text),
              *paper_errors(text), *computed_errors(text, checked)]
    if new != text:
        errors.append("the parameters table is stale; run python3 scripts/check_spec.py --write")
    for error in errors:
        print(f"{args.spec.name}: {error}")
    repository = repository_errors()
    for error in repository:
        print(f"repository: {error}")
    if errors or repository:
        sys.exit(f"spec: {len(errors) + len(repository)} errors")
    for template, reason in UNCHECKED:
        print(f"spec: not recomputed: {template}: {reason}")
    print(f"spec: {args.spec.name} has well-formed pins, a pin for each of the {len(claims())} claims, "
          "existing paths, a current parameters table, a row for each cited paper number and "
          f"the recomputed values of {len(checked)} claims, and no other spec file or `spec §` "
          "citation exists")


if __name__ == "__main__":
    main()
