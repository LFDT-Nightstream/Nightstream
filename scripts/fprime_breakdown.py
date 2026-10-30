#!/usr/bin/env python3
"""Chart the logical CCS rows and committed coordinates of one F′ step.

The shared verifier manifest (`crates/nightstream/artifacts/shared-verifier-v1.json`,
emitted by Lean `Export/SharedVerifier.lean`) gives the rows of each verifier
component and the coordinate slots of each assignment block as affine forms of
three application numbers: witness words, local words, and rows. This script
evaluates them for one application, checks the totals against the manifest's
own formulas, prints a table, and writes a two-tab HTML page. Without
`--application` it charts the manifest's reference application
(Poseidon2HashChainV1). A Rust program reads its three numbers from
`nightstream::Circuit::application_size`.
"""

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "crates/nightstream/artifacts/shared-verifier-v1.json"
PARAMETERS = ["witness_words", "local_words", "application_rows"]

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

    def __sub__(self, other):
        return Size(self.witness_words - other.witness_words, self.local_words - other.local_words,
                    self.rows - other.rows)


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
        raise ManifestError(f"{message}; update scripts/fprime_breakdown.py to the new manifest")


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


def breakdown(manifest, application):
    """The row and coordinate groups of `application`: (rows, coordinates), each a
    list of (group, [(section, [(leaf, value, category)])])."""
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
    state_hash = rows["pilot_poseidon"] // 2
    check(children["pilot_poseidon"]["block_count"] == 2 and rows["pilot_poseidon"] == 2 * state_hash
          and slots["priorPoseidon"] == state_hash == slots["outputPoseidon"], "state hash blocks changed")
    check(slots["laterPoseidon"] == rows["pi_ccs_poseidon"] + rows["pi_rlc_sampler_poseidon"],
          "later Poseidon2 blocks changed")

    geometry = manifest["geometry"]
    field = geometry["field_slot_width"]
    width = evaluate(geometry["logical_width"], size)
    carrier = -(-width // geometry["ring_degree"]) * geometry["ring_degree"]

    def application_leaves(label, value_of):
        if application.parts:
            return [(name, value_of(part), "app") for name, part in application.parts]
        return [(label, value_of(size), "app")]

    row_groups = [
        (GLUE, [
            ("Prior-state hash: check input digest", [
                ("Hash of carried state", state_hash, "hash"),
                ("State framing and digest encoding", rows["pilot_ordinary"], "other"),
                ("Digest equality pins", rows["pilot_digest_binding"], "other")]),
            ("Output-state hash: make next digest", [("Hash of next state", state_hash, "hash")]),
            ("Running transition: base or recursive", [("Base/running mux", rows["running_transition"], "other")]),
            ("Final input/output binding", [
                ("Binding rows", rows["next_preimage"] + rows["recursive_public_output"], "other")])]),
        (VERIFIER, [
            ("PiCCS.V: check the sum-check proof", [
                ("Transcript and output hashing", rows["pi_ccs_poseidon"], "hash"),
                ("Sum-check and final-identity arithmetic", rows["pi_ccs_ordinary"], "piccs"),
                ("Endpoint pins", rows["pi_ccs_endpoint"], "piccs")]),
            ("PiRLC.V: sample ρ, combine 17", [
                ("Ring products: combine 17 claims", rows["pi_rlc_combination"], "ring"),
                ("Wide sampler: division and range", rows["pi_rlc_sampler_ordinary"], "samp"),
                ("Sampler hash", rows["pi_rlc_sampler_poseidon"], "hash")]),
            ("PiDEC.V: check the 16 children", [("Split and recomposition checks", rows["pi_dec"], "other")])]),
        (APPLICATION, [(application.name, application_leaves("Application rows", lambda part: part.rows))]),
    ]

    def fields(*kinds):
        return field * sum(slots[kind] for kind in kinds)

    coordinate_groups = [
        (GLUE, [
            ("Prior-state hash: check input digest", [
                ("S-boxes of carried-state hash", fields("priorPoseidon"), "hash"),
                (f"Prior state preimage ({slots['priorPoseidonInput']:,} words)", fields("priorPoseidonInput"), "other"),
                ("Framing and digest fields", fields("pilotCanonicalLocal", "pilotCanonicalFresh", "pilotOutputDigest"),
                 "other")]),
            ("Output-state hash: make next digest", [
                ("S-boxes of next-state hash", fields("outputPoseidon"), "hash"),
                (f"Next state preimage ({slots['outputPoseidonInput']:,} words)", fields("outputPoseidonInput"), "other")]),
            ("Running transition: base or recursive", [
                (f"Carried running words ({slots['runningPiDec']:,})", fields("runningPiDec"), "other"),
                (f"Flag fields ({slots['runningFresh']})", fields("runningFresh"), "other")]),
            ("Public prefix and alignment", [
                (f"Public prefix {geometry['logical_public']} + alignment {carrier - width}",
                 geometry["logical_public"] + carrier - width, "other")])]),
        (VERIFIER, [
            ("PiCCS.V: check the sum-check proof", [
                ("Transcript and output S-boxes", field * rows["pi_ccs_poseidon"], "hash"),
                (f"Proof values ({slots['piCcsProofLogical']:,} fields)", fields("piCcsProofLogical"), "piccs"),
                (f"Arithmetic scratch ({slots['piCcsFresh']:,} fields)", fields("piCcsFresh"), "piccs"),
                ("Claim boundary fields", fields("piCcsPriorLast", "piCcsOutputLast"), "piccs"),
                ("Fresh public input", fields("piCcsFreshPublicInput"), "piccs"),
                ("Context and endpoint fields", fields("piCcsExpectedContext", "piCcsOutputEndpoint"), "piccs")]),
            ("PiRLC.V: sample ρ, combine 17", [
                ("Ring product groups", fields("productGroup"), "ring"),
                ("Ring product outputs", fields("productOutput"), "ring"),
                (f"Sampler core fields ({slots['samplerLogical']:,})", fields("samplerLogical"), "samp"),
                (f"Sampler range fields ({slots['samplerFresh']:,})", fields("samplerFresh"), "samp"),
                (f"Challenge coefficients ρ ({slots['challengeWords']} fields)", fields("challengeWords"), "samp"),
                ("Sampler hash S-boxes", field * rows["pi_rlc_sampler_poseidon"], "hash")]),
            ("PiDEC.V: check the 16 children", [
                ("Public-input fields", fields("piDecLogical"), "other"),
                ("Fresh fields", fields("piDecFresh"), "other")])]),
        (APPLICATION, [(application.name, application_leaves(
            "Witness and local fields", lambda part: field * (part.witness_words + part.local_words)))]),
    ]

    def total(groups):
        return sum(value for _, sections in groups for _, leaves in sections for _, value, _ in leaves)

    check(total(row_groups) == evaluate(geometry["logical_rows"], size), "row total differs from the manifest")
    check(total(coordinate_groups) == carrier, "coordinate total differs from the manifest")

    def nonzero(groups):
        return [(group, [(section, [leaf for leaf in leaves if leaf[1]]) for section, leaves in sections])
                for group, sections in groups]

    return nonzero(row_groups), nonzero(coordinate_groups)


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
.fb-name{flex:0 0 40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
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
render("rows");
})();
</script>
</body></html>
"""


def page(application, row_groups, coordinate_groups, manifest_path):
    size = application.size
    subtitle = (f"{application.name}: {size.witness_words:,} witness words, {size.local_words:,} local words, "
                f"{size.rows:,} rows. From {manifest_path.name} (Lean Export/SharedVerifier.lean).")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--name", help="the application's name on the chart")
    parser.add_argument("--application", type=parse_size, metavar="WITNESS,LOCAL,ROWS",
                        help="the application's size (default: the manifest's reference application)")
    parser.add_argument("--part", type=parse_part, action="append", default=[], metavar="NAME=WITNESS,LOCAL,ROWS",
                        help="one part of the application; the parts must sum to --application")
    parser.add_argument("--out", type=Path, help="write the HTML page here")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    size = args.application or Size(*manifest["reference"])
    name = args.name or ("Poseidon2HashChainV1" if args.application is None else "Application")
    application = Application(name, size, tuple(args.part))
    try:
        row_groups, coordinate_groups = breakdown(manifest, application)
    except (ManifestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(table("Logical CCS rows", row_groups))
    print(table("Committed coordinates", coordinate_groups))
    if args.out:
        args.out.write_text(page(application, row_groups, coordinate_groups, args.manifest))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
