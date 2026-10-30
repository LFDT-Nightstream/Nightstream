#!/usr/bin/env python3
"""Chart the logical CCS rows and committed coordinates of one F′ step.

The shared verifier manifest (`crates/nightstream/artifacts/shared-verifier-v1.json`,
emitted by Lean `Export/SharedVerifier.lean`) gives the rows of each verifier
component and the coordinate slots of each assignment block as affine forms of
three application numbers: witness words, local words, and rows. The finer
leaves inside each component come from the literal counts that Lean theorems
and definitions state (`LEAN_FACTS`), and the leaves of each component must sum
to the manifest's value for it. The script prints a table and writes a two-tab
HTML page, and with `--png` a PNG image of each tab.

Without `--application` it charts the manifest's reference application
(Poseidon2HashChainV1). A Rust program reads its three numbers from
`nightstream::Circuit::application_size`, and a Nebula program its parts from
`nightstream::nebula::size_parts`. A part named `Section / Leaf` goes under its
section.
"""

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "crates/nightstream/artifacts/shared-verifier-v1.json"
LEAN = ROOT / "formal/nightstream-fprime/NightstreamFPrime"
PARAMETERS = ["witness_words", "local_words", "application_rows"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Lean `PerApplicationAssignmentPlan.canonicalKinds`, in opcode order.
BLOCK_KINDS = [
    "priorPoseidon", "outputPoseidon", "laterPoseidon", "productGroup", "challengeWords",
    "productOutput", "priorPoseidonInput", "outputPoseidonInput", "runningPiDec", "runningFresh",
    "piCcsFreshPublicInput", "piCcsPriorLast", "piCcsOutputLast", "piCcsExpectedContext",
    "piCcsProofLogical", "piCcsOutputEndpoint", "piCcsFresh", "pilotCanonicalLocal",
    "pilotCanonicalFresh", "pilotOutputDigest", "piDecLogical", "piDecFresh", "samplerLogical",
    "samplerFresh", "applicationWitness", "applicationLocal",
]

CHILDREN = {
    "pilot_poseidon", "pi_ccs_poseidon", "pi_ccs_ordinary", "pilot_ordinary", "pilot_digest_binding",
    "pi_ccs_endpoint", "pi_rlc_sampler_poseidon", "pi_rlc_sampler_ordinary", "pi_rlc_combination",
    "pi_dec", "running_transition", "application", "next_preimage", "recursive_public_output",
}

# Each fact: (file under LEAN, kind, name). `theorem` reads the first `= N :=`
# of the theorem, `def` reads `def|abbrev name : Nat := N`.
LEAN_FACTS = {
    "permutation_rows": ("Layout/ProductionRelation/PoseidonRetainedSlots.lean", "theorem", "rows_length"),
    "permutation_slots": ("Layout/ProductionRelation/PoseidonRetainedSlots.lean", "theorem", "slots_length"),
    "prior_permutations": ("Export/Stage1/PoseidonRetainedBlock.lean", "theorem", "priorInvocationCount_eq"),
    "output_permutations": ("Export/Stage1/PoseidonRetainedBlock.lean", "theorem", "outputInvocationCount_eq"),
    "statement_permutations": ("Export/Stage1/PiCCSInvocations.lean", "theorem", "statementInvocationCount_eq"),
    "challenge_permutations": ("Export/Stage1/PiCCSInvocations.lean", "theorem", "challengeInvocationCount_eq"),
    "round_permutations": ("Export/Stage1/PiCCSInvocations.lean", "theorem", "roundInvocationCount_eq"),
    "output_binding_permutations": ("Export/Stage1/PiCCSInvocations.lean", "theorem", "outputInvocationCount_eq"),
    "sampler_permutations": ("Export/Stage1/PiRLCSamplerPoseidonPlan.lean", "theorem", "invocationCount_eq"),
    "statement_binding_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "statementBindingRows_length"),
    "initial_claim_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "initialClaimRows_length"),
    "sumcheck_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "sumcheckRows_length"),
    "eval_k_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "evalKRows_length"),
    "eval_a_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "evalARows_length"),
    "ccs_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "ccsRows_length"),
    "norm_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "normRows_length"),
    "final_identity_rows": ("Export/Stage1/PiCCSArithmetic.lean", "theorem", "finalIdentityRows_length"),
    "commitment_products": ("Export/Stage1/PiRLCProductSchedule.lean", "theorem", "commitment_invocationCount"),
    "public_input_products": ("Export/Stage1/PiRLCProductSchedule.lean", "theorem", "publicInput_invocationCount"),
    "eval_k_products": ("Export/Stage1/PiRLCProductSchedule.lean", "theorem", "evalK_invocationCount"),
    "eval_a_products": ("Export/Stage1/PiRLCProductSchedule.lean", "theorem", "evalA_invocationCount"),
    "sampler_sources": ("Export/Stage1/PiRLCSamplerInvocations.lean", "def", "sourceCount"),
    "sampler_range_rows": ("Export/Stage1/PiRLCSamplerOrdinaryRows.lean", "theorem", "rangeRows_length"),
    "sampler_word_rows": ("Export/Stage1/PiRLCSamplerOrdinaryRows.lean", "theorem", "wordRows_length"),
    "wide_fields": ("Gadgets/Sampling/WideReduction.lean", "def", "fieldCount"),
    "canonical_rows": ("Gadgets/Range/CanonicalU64.lean", "def", "exactRowCount"),
    "canonical_fresh": ("Layout/Range/CanonicalU64.lean", "theorem", "totalFreshCount_eq"),
    "wide_new_bits": ("Gadgets/Sampling/WideReduction/Footprint.lean", "theorem", "newBitCount_eq"),
    "wide_digits": ("Gadgets/Sampling/WideReduction.lean", "def", "digitCount"),
    "wide_checks": ("Gadgets/Sampling/WideReduction.lean", "def", "checkCount"),
    "running_mux_words": ("Lifecycle/Stage1/RunningTransition.lean", "def", "exactWordCount"),
    "running_state_words": ("Lifecycle/Stage1/RunningTransition.lean", "def", "stateWordCount"),
    "running_rows": ("Lifecycle/Stage1/RunningTransition.lean", "def", "exactRowCount"),
}
# Lean `RunningTransition.flagConstraint` and `bindingConstraint`: rows, not literals.
RUNNING_FLAG_AND_BINDING_ROWS = 2
# Lean `DirectPiRLCProductFootprint.rowCount` and `PiDEC.v1_2.exactRowDeltas`.
PRODUCT_ROWS = ("Export/Stage1/DirectPiRLCProductFootprint.lean", r"def rowCount : Nat := invocationCount \* (\d+)")
PI_DEC_DELTAS = ("Layout/PiDEC/v1_2/Values.lean", r"exactRowDeltas = \[([\d, ]+)\] := by rfl")

CATEGORIES = {
    "hash": ("Poseidon2 hashing", "#378ADD"),
    "ring": ("Ring products", "#D85A30"),
    "piccs": ("PiCCS arithmetic", "#1D9E75"),
    "samp": ("Sampling and range", "#EF9F27"),
    "app": ("Application", "#7F77DD"),
    "other": ("Other", "#888780"),
}

GLUE = "IVC state glue: bind, select, re-hash"
VERIFIER = "Recursive verifier: NIFS.V of the previous step"
APPLICATION = "Application F: the actual step"


@dataclass(frozen=True)
class Size:
    """The three numbers of an application that the manifest's forms take."""

    witness_words: int
    local_words: int
    rows: int


@dataclass(frozen=True)
class Application:
    name: str
    size: Size
    parts: tuple  # (name, Size) pairs that sum to `size`, or empty


class ManifestError(Exception):
    pass


def evaluate(form, size):
    constant, witness, local, rows = form
    return constant + witness * size.witness_words + local * size.local_words + rows * size.rows


def check(condition, message):
    if not condition:
        raise ManifestError(f"{message}; update scripts/fprime_breakdown.py to the new manifest or Lean")


def lean_facts(root=LEAN):
    """Every `LEAN_FACTS` count, the PiRLC product rows per invocation, and the
    PiDEC row deltas, read from the Lean sources under `root`."""
    texts = {}

    def text(path):
        if path not in texts:
            texts[path] = (root / path).read_text()
        return texts[path]

    def find(path, regex, name):
        match = re.search(regex, text(path), re.MULTILINE)
        check(match is not None, f"Lean {path} has no {name}")
        return match.group(1)

    facts = {}
    for key, (path, kind, name) in LEAN_FACTS.items():
        regex = (rf"theorem {re.escape(name)}\b[\s\S]*?=\s*(\d+)\s*:=" if kind == "theorem"
                 else rf"^(?:def|abbrev) {re.escape(name)} : Nat := (\d+)\s*$")
        facts[key] = int(find(path, regex, name))
    facts["product_rows_per_invocation"] = int(find(*PRODUCT_ROWS, "rowCount"))
    facts["pi_dec_deltas"] = [int(value) for value in find(*PI_DEC_DELTAS, "exactRowDeltas").split(",")]
    return facts


def parse_size(text):
    try:
        witness, local, rows = (int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected WITNESS_WORDS,LOCAL_WORDS,ROWS") from error
    return Size(witness, local, rows)


def parse_part(text):
    name, separator, numbers = text.rpartition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("expected NAME=WITNESS_WORDS,LOCAL_WORDS,ROWS")
    return name, parse_size(numbers)


def breakdown(manifest, application, facts):
    """The row and coordinate groups of `application`: (rows, coordinates), each a
    list of (group, [(section, [(leaf, value, category)])]), leaves largest first."""
    check(manifest.get("parameters") == PARAMETERS, "manifest parameters changed")
    size = application.size
    if application.parts:
        total = Size(*(sum(getattr(part, field) for _, part in application.parts)
                       for field in ("witness_words", "local_words", "rows")))
        if total != size:
            raise ValueError(f"the parts sum to {total}, not to the application {size}")

    children = {child["id"]: child for child in manifest["children"]}
    check(set(children) == CHILDREN, "manifest children changed")
    rows = {name: evaluate(child["row_count"], size) for name, child in children.items()}
    blocks = manifest["assignment_blocks"]
    check(len(blocks) == len(BLOCK_KINDS) and all(block["opcode"] == index for index, block in enumerate(blocks)),
          "manifest assignment blocks changed")
    check(manifest["application_local_index"] == BLOCK_KINDS.index("applicationLocal"),
          "application local block moved")
    form = {kind: block["slot_count"] for kind, block in zip(BLOCK_KINDS, blocks)}
    check(form["applicationWitness"] == [0, 1, 0, 0] and form["applicationLocal"] == [0, 0, 1, 0],
          "application blocks changed")
    slots = {kind: evaluate(slot_form, size) for kind, slot_form in form.items()}

    f = facts
    per_row, per_slot = f["permutation_rows"], f["permutation_slots"]
    state_hash = rows["pilot_poseidon"] // 2
    check(children["pilot_poseidon"]["block_count"] == 2
          and f["prior_permutations"] * per_row == state_hash == f["output_permutations"] * per_row
          and rows["pilot_poseidon"] == 2 * state_hash
          and slots["priorPoseidon"] == f["prior_permutations"] * per_slot
          and slots["outputPoseidon"] == f["output_permutations"] * per_slot, "state hashes changed")
    families = [("Output binding: absorb y′", f["output_binding_permutations"]),
                ("Statement absorption", f["statement_permutations"]),
                ("Round transcript", f["round_permutations"]),
                ("Challenge derivation", f["challenge_permutations"])]
    piccs_permutations = sum(count for _, count in families)
    check(piccs_permutations * per_row == rows["pi_ccs_poseidon"], "PiCCS hash families changed")
    check(f["sampler_permutations"] * per_row == rows["pi_rlc_sampler_poseidon"], "sampler hash changed")
    check(slots["laterPoseidon"] == (piccs_permutations + f["sampler_permutations"]) * per_slot,
          "later Poseidon2 blocks changed")
    arithmetic = [("Initial claim", f["initial_claim_rows"]), ("Eval_A terminal", f["eval_a_rows"]),
                  ("Statement binding", f["statement_binding_rows"]),
                  ("Final identity v = Q(r′)", f["final_identity_rows"]), ("Eval_K terminal", f["eval_k_rows"]),
                  ("Norm terminal", f["norm_rows"]), ("Sum-check chain", f["sumcheck_rows"]),
                  ("CCS terminal", f["ccs_rows"])]
    check(sum(value for _, value in arithmetic) == rows["pi_ccs_ordinary"], "PiCCS arithmetic changed")
    products = [("commitment", f["commitment_products"]), ("public input", f["public_input_products"]),
                ("Eval_A", f["eval_a_products"]), ("Eval_K", f["eval_k_products"])]
    product_count = sum(count for _, count in products)
    check(product_count * f["product_rows_per_invocation"] == rows["pi_rlc_combination"]
          and product_count == slots["productGroup"] == slots["productOutput"], "PiRLC product families changed")
    sources, fields = f["sampler_sources"], f["wide_fields"]
    wide = [(f"Wide sampler: canonical u64 checks ({sources} × {fields} × {f['canonical_rows']})",
             sources * fields * f["canonical_rows"]),
            (f"Wide sampler: new-bit booleanity ({sources} × {f['wide_new_bits']})", sources * f["wide_new_bits"]),
            (f"Wide sampler: canonical u64 lowering ({sources} × {fields} × {f['canonical_fresh']})",
             sources * fields * f["canonical_fresh"]),
            (f"Wide sampler: digit range ({sources} × {f['wide_digits']})", sources * f["wide_digits"]),
            (f"Wide sampler: challenge-word packing ({sources} × {f['sampler_word_rows']})",
             sources * f["sampler_word_rows"]),
            (f"Wide sampler: modular checks ({sources} × {f['wide_checks']})", sources * f["wide_checks"])]
    check(f["sampler_range_rows"] == fields * (f["canonical_rows"] + f["canonical_fresh"]) + f["wide_new_bits"]
          + f["wide_digits"] + f["wide_checks"]
          and sum(value for _, value in wide) == rows["pi_rlc_sampler_ordinary"], "wide sampler changed")
    deltas = [delta for delta in f["pi_dec_deltas"] if delta]
    check(len(deltas) == 4 and sum(deltas) == rows["pi_dec"], "PiDEC phases changed")
    running = rows["running_transition"]
    check(f["running_rows"] == running == f["running_mux_words"] + f["running_state_words"]
          + RUNNING_FLAG_AND_BINDING_ROWS, "running transition changed")

    geometry = manifest["geometry"]
    field = geometry["field_slot_width"]
    width = evaluate(geometry["logical_width"], size)
    carrier = -(-width // geometry["ring_degree"]) * geometry["ring_degree"]

    def application_sections(label, value_of):
        if not application.parts:
            return [(application.name, [(label, value_of(size), "app")])]
        sections = {}
        for name, part in application.parts:
            section, separator, leaf = name.partition(" / ")
            if not separator:
                section, leaf = application.name, name
            sections.setdefault(section, []).append((leaf, value_of(part), "app"))
        return list(sections.items())

    row_groups = [
        (GLUE, [
            ("Prior-state hash: check input digest", [
                (f"Hash of carried state ({f['prior_permutations']:,} perms)", state_hash, "hash"),
                ("State framing and digest encoding", rows["pilot_ordinary"], "other"),
                ("Digest equality pins", rows["pilot_digest_binding"], "other")]),
            ("Output-state hash: make next digest", [
                (f"Hash of next state ({f['output_permutations']:,} perms)", state_hash, "hash")]),
            ("Running transition: base or recursive", [
                (f"Base/running mux ({f['running_mux_words']:,} words)", f["running_mux_words"], "other"),
                (f"Base state z0 = zi ({f['running_state_words']} words)", f["running_state_words"], "other"),
                ("Flag and binding rows", RUNNING_FLAG_AND_BINDING_ROWS, "other")]),
            ("Final input/output binding", [
                ("Binding rows", rows["next_preimage"] + rows["recursive_public_output"], "other")])]),
        (VERIFIER, [
            ("PiCCS.V: check the sum-check proof",
             [(f"{label} ({count:,} perms)", count * per_row, "hash") for label, count in families]
             + [(label, value, "piccs") for label, value in arithmetic]
             + [("Endpoint pins", rows["pi_ccs_endpoint"], "piccs")]),
            ("PiRLC.V: sample ρ, combine 17",
             [(f"Ring products: {label}", count * f["product_rows_per_invocation"], "ring")
              for label, count in products]
             + [(label, value, "samp") for label, value in wide]
             + [(f"Sampler hash ({f['sampler_permutations']} perms)", rows["pi_rlc_sampler_poseidon"], "hash")]),
            ("PiDEC.V: check the 16 children", [
                ("Public-input split and range", deltas[0], "other"),
                ("Commitment recomposition", deltas[1], "other"),
                ("Eval_K recomposition", deltas[2], "other"),
                ("Eval_A recomposition", deltas[3], "other")])]),
        (APPLICATION, application_sections("Application rows", lambda part: part.rows)),
    ]

    def fields_of(*kinds):
        return field * sum(slots[kind] for kind in kinds)

    coordinate_groups = [
        (GLUE, [
            ("Prior-state hash: check input digest", [
                ("S-boxes of carried-state hash", fields_of("priorPoseidon"), "hash"),
                (f"Prior state preimage ({slots['priorPoseidonInput']:,} words)", fields_of("priorPoseidonInput"),
                 "other"),
                ("Framing and digest fields", fields_of("pilotCanonicalLocal", "pilotCanonicalFresh",
                                                        "pilotOutputDigest"), "other")]),
            ("Output-state hash: make next digest", [
                ("S-boxes of next-state hash", fields_of("outputPoseidon"), "hash"),
                (f"Next state preimage ({slots['outputPoseidonInput']:,} words)", fields_of("outputPoseidonInput"),
                 "other")]),
            ("Running transition: base or recursive", [
                (f"Carried running words ({slots['runningPiDec']:,})", fields_of("runningPiDec"), "other"),
                (f"Flag fields ({slots['runningFresh']})", fields_of("runningFresh"), "other")]),
            ("Public prefix and alignment", [
                (f"Public prefix {geometry['logical_public']} + alignment {carrier - width}",
                 geometry["logical_public"] + carrier - width, "other")])]),
        (VERIFIER, [
            ("PiCCS.V: check the sum-check proof",
             [(f"{label} S-boxes", field * count * per_slot, "hash") for label, count in families]
             + [(f"Proof values ({slots['piCcsProofLogical']:,} fields)", fields_of("piCcsProofLogical"), "piccs"),
                (f"Arithmetic scratch ({slots['piCcsFresh']:,} fields)", fields_of("piCcsFresh"), "piccs"),
                ("Claim boundary fields", fields_of("piCcsPriorLast", "piCcsOutputLast"), "piccs"),
                ("Fresh public input", fields_of("piCcsFreshPublicInput"), "piccs"),
                ("Context and endpoint fields", fields_of("piCcsExpectedContext", "piCcsOutputEndpoint"), "piccs")]),
            ("PiRLC.V: sample ρ, combine 17",
             [(f"Ring products: {label}", field * 2 * count, "ring") for label, count in products]
             + [(f"Sampler core fields ({slots['samplerLogical']:,})", fields_of("samplerLogical"), "samp"),
                (f"Sampler range fields ({slots['samplerFresh']:,})", fields_of("samplerFresh"), "samp"),
                (f"Challenge coefficients ρ ({slots['challengeWords']} fields)", fields_of("challengeWords"), "samp"),
                ("Sampler hash S-boxes", field * f["sampler_permutations"] * per_slot, "hash")]),
            ("PiDEC.V: check the 16 children", [
                ("Public-input fields", fields_of("piDecLogical"), "other"),
                ("Fresh fields", fields_of("piDecFresh"), "other")])]),
        (APPLICATION, application_sections(
            "Witness and local fields", lambda part: field * (part.witness_words + part.local_words))),
    ]

    def total(groups):
        return sum(value for _, sections in groups for _, leaves in sections for _, value, _ in leaves)

    check(total(row_groups) == evaluate(geometry["logical_rows"], size), "row total differs from the manifest")
    check(total(coordinate_groups) == carrier, "coordinate total differs from the manifest")

    def shown(groups):
        return [(group, [(section, sorted((leaf for leaf in leaves if leaf[1]), key=lambda leaf: -leaf[1]))
                         for section, leaves in sections if any(leaf[1] for leaf in leaves)])
                for group, sections in groups]

    return shown(row_groups), shown(coordinate_groups)


def table(title, groups):
    total = sum(value for _, sections in groups for _, leaves in sections for _, value, _ in leaves)
    lines = [f"{title}: {total:,}"]
    for group, sections in groups:
        lines.append(f"  {group}: {sum(v for _, leaves in sections for _, v, _ in leaves):,}")
        for section, leaves in sections:
            lines.append(f"    {section}: {sum(v for _, v, _ in leaves):,}")
            lines.extend(f"      {leaf}: {value:,} ({100 * value / total:.1f}%)" for leaf, value, _ in leaves)
    return "\n".join(lines)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>F′ step breakdown</title>
<style>
:root{--bg:#ffffff;--text:#1f1e1d;--muted:#6b6a68;--border:#e5e4e2;--strong:#c9c8c5;--surface:#f3f2f0;--total:#b4b2a9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1f1e1d;--text:#f3f2f0;--muted:#a3a29f;--border:#3a3937;--strong:#55534f;--surface:#2b2a28;--total:#6b6a68}}
:root[data-theme="dark"]{--bg:#1f1e1d;--text:#f3f2f0;--muted:#a3a29f;--border:#3a3937;--strong:#55534f;--surface:#2b2a28;--total:#6b6a68}
body{background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,sans-serif;margin:0;padding:16px}
main{max-width:1100px;margin:0 auto}
h1{font-size:18px;font-weight:600;margin:0 0 4px}
p{color:var(--muted);margin:0 0 14px}
.fb-btn{font:inherit;padding:8px 14px;border:1px solid var(--strong);border-radius:8px;background:transparent;color:var(--text);cursor:pointer;margin:0 8px 8px 0}
.fb-btn.on{background:var(--surface);font-weight:600}
.fb-row{display:flex;align-items:center;gap:12px;padding:3px 0;font-size:13px}
.fb-name{flex:0 0 40%;box-sizing:border-box;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fb-bar{flex:1;display:flex;gap:2px;height:12px;min-width:0}
.fb-seg{height:100%;border-radius:3px;min-width:2px}
.fb-val{flex:0 0 170px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.fb-g{font-size:14px;font-weight:600;padding-top:10px}
.fb-s{font-weight:600}.fb-s .fb-name{padding-left:14px}
.fb-l{color:var(--muted)}.fb-l .fb-name{padding-left:30px}
.fb-sep{border-top:1px solid var(--border);margin-top:10px}
.fb-leg{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;padding:10px 0 6px}
.fb-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
@media (max-width:640px){.fb-name{flex-basis:46%}.fb-val{flex-basis:120px;font-size:12px}.fb-bar{display:none}}
</style></head>
<body><main>
<h1>__TITLE__</h1>
<p>__SUBTITLE__</p>
<div><button class="fb-btn on" id="fb-b-rows">Logical CCS rows</button><button class="fb-btn" id="fb-b-coords">Committed coordinates</button></div>
<div id="fb-out"></div>
</main>
<script>
(function(){
var C=__CATEGORIES__;
var D=__DATA__;
var q=new URLSearchParams(location.search);
if(q.get("theme"))document.documentElement.setAttribute("data-theme",q.get("theme"));
function fmt(n){return n.toLocaleString("en-US")}
function pct(v,t){var p=100*v/t;if(p<0.001)return "<0.001%";if(p<0.1)return p.toFixed(3)+"%";return p.toFixed(1)+"%"}
function sum(a){var s=0;for(var i=0;i<a.length;i++)s+=a[i];return s}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;")}
function segs(leaves,t){return leaves.map(function(l){return '<span class="fb-seg" style="flex:0 0 '+(100*l[1]/t)+'%;background:'+C[l[2]][1]+'"></span>'}).join("")}
function row(cls,name,bar,v,t){return '<div class="fb-row '+cls+'"><span class="fb-name" title="'+esc(name)+'">'+esc(name)+'</span><span class="fb-bar">'+bar+'</span><span class="fb-val">'+fmt(v)+' · '+pct(v,t)+'</span></div>'}
function render(key){
var d=D[key],all=[];
d.groups.forEach(function(g){g[1].forEach(function(s){s[1].forEach(function(l){all.push(l)})})});
var t=sum(all.map(function(l){return l[1]}));
var h='<div class="fb-leg">';
Object.keys(C).forEach(function(k){var v=sum(all.filter(function(l){return l[2]===k}).map(function(l){return l[1]}));if(v)h+='<span><span class="fb-dot" style="background:'+C[k][1]+'"></span>'+C[k][0]+' '+pct(v,t)+'</span>'});
h+='</div>'+row("fb-g",d.total,'<span class="fb-seg" style="flex:0 0 100%;background:var(--total)"></span>',t,t);
d.groups.forEach(function(g){
var gl=[];g[1].forEach(function(s){gl=gl.concat(s[1])});
h+='<div class="fb-sep"></div>'+row("fb-g",g[0],segs(gl,t),sum(gl.map(function(l){return l[1]})),t);
g[1].forEach(function(s){
h+=row("fb-s","▾ "+s[0],segs(s[1],t),sum(s[1].map(function(l){return l[1]})),t);
s[1].forEach(function(l){h+=row("fb-l",l[0],segs([l],t),l[1],t)});
});
});
document.getElementById("fb-out").innerHTML=h;
document.getElementById("fb-b-rows").classList.toggle("on",key==="rows");
document.getElementById("fb-b-coords").classList.toggle("on",key==="coords");
}
document.getElementById("fb-b-rows").addEventListener("click",function(){render("rows")});
document.getElementById("fb-b-coords").addEventListener("click",function(){render("coords")});
render(q.get("tab")==="coords"?"coords":"rows");
})();
</script>
</body></html>
"""


def page(application, row_groups, coordinate_groups, manifest_path):
    size = application.size
    subtitle = (f"{application.name}: {size.witness_words:,} witness words, {size.local_words:,} local words, "
                f"{size.rows:,} rows. From {manifest_path.name} and the Lean counts it splits.")
    data = {"rows": {"total": "Total logical CCS rows", "groups": row_groups},
            "coords": {"total": "Total committed coordinates", "groups": coordinate_groups}}
    replacements = {
        "__TITLE__": html.escape(f"F′ step: {application.name}"),
        "__SUBTITLE__": html.escape(subtitle),
        "__CATEGORIES__": json.dumps({key: list(value) for key, value in CATEGORIES.items()}),
        "__DATA__": json.dumps(data, ensure_ascii=False).replace("</", "<\\/"),
    }
    text = PAGE
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def write_png(page_path, png_path, tab, groups, theme, chrome):
    """Render one tab of the page with headless Chrome. The window is tall
    enough for every line; Pillow, when present, trims the empty bottom."""
    lines = 1 + sum(1 + len(sections) + sum(len(leaves) for _, leaves in sections) for _, sections in groups)
    height = 190 + 24 * lines + 22 * len(groups)
    url = f"{page_path.resolve().as_uri()}?tab={tab}&theme={theme}"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=1200,{height}", f"--screenshot={png_path}", url],
                   check=True, capture_output=True, timeout=120)
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return
    image = Image.open(png_path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((2, image.height - 2)))
    box = ImageChops.difference(image, background).getbbox()
    if box:
        image.crop((0, 0, image.width, min(image.height, box[3] + 32))).save(png_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--name", help="the application's name on the chart")
    parser.add_argument("--application", type=parse_size, metavar="WITNESS,LOCAL,ROWS",
                        help="the application's size (default: the manifest's reference application)")
    parser.add_argument("--part", type=parse_part, action="append", default=[], metavar="NAME=WITNESS,LOCAL,ROWS",
                        help="one part of the application; the parts must sum to --application")
    parser.add_argument("--out", type=Path, help="write the HTML page here")
    parser.add_argument("--png", type=Path, metavar="PREFIX",
                        help="write PREFIX-rows.png and PREFIX-coordinates.png with headless Chrome")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark", help="the theme of the PNG images")
    parser.add_argument("--chrome", default=shutil.which("google-chrome") or CHROME, help="the Chrome binary")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    size = args.application or Size(*manifest["reference"])
    name = args.name or ("Poseidon2HashChainV1" if args.application is None else "Application")
    application = Application(name, size, tuple(args.part))
    try:
        row_groups, coordinate_groups = breakdown(manifest, application, lean_facts())
    except (ManifestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(table("Logical CCS rows", row_groups))
    print(table("Committed coordinates", coordinate_groups))
    text = page(application, row_groups, coordinate_groups, args.manifest)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    if args.png:
        with tempfile.TemporaryDirectory() as scratch:
            page_path = Path(scratch) / "page.html"
            page_path.write_text(text)
            for tab, groups, suffix in (("rows", row_groups, "rows"), ("coords", coordinate_groups, "coordinates")):
                png_path = Path(f"{args.png}-{suffix}.png").resolve()
                write_png(page_path, png_path, tab, groups, args.theme, args.chrome)
                print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
