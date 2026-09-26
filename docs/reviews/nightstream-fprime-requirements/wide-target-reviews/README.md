# Wide-key target reviews

Status: **pending independent review and signed acceptance.** The requests below
cover the completed PR #123 source. Historical passing responses do not approve
this source. No controlled checker or signing process has been supplied to this
task, and the implementation author has not created acceptance records.

## Current source and requests

Captured source commit: `65d14abb27ff20a1c87ddade4b77982cc9864995`.

The source includes the single application and witness paths, actual application
row custody, the current wide package and manifest, fresh two-fold fixtures,
legacy-consumer removal, and the completed validation fixes. The source capture
excludes Python bytecode caches and this review directory. This follow-up commit
changes review metadata only; the captured implementation and artifacts remain
those of the source commit above.

| Obligation | Snapshot | Request | Review |
|---|---|---|---|
| `stage1-terminal-assignment` | `2bfc7b5ff9fe4c03976fcf70ffbdcd154890f97f576352779a040a12c0595030` | `e5317552cbc45b2f1a177c7cd33a5be0e5869310e80da8345f036dff4129864d` | pending |
| `stage1-terminal-parent` | `2bfc7b5ff9fe4c03976fcf70ffbdcd154890f97f576352779a040a12c0595030` | `15d16aa09826ae41f7ecc4c91f56a7d7168d8a4ce22f114c963cdf88e067c3a5` | pending |
| `hypernova-linear-security` | `f972cc527645a739d247c1691c9a1956e764b20eb5b4e83f30fe8ede1b7c6b6d` | `d6c3c382e0529f6385371a4cd41a4830ec88324265bfa018270f6c0778496fee` | pending |
| `hypernova-terminal-false-acceptance` | `f972cc527645a739d247c1691c9a1956e764b20eb5b4e83f30fe8ede1b7c6b6d` | `bbc525c824a7b1ca8b5aadd36c5c4c293be734b4f4e444d3c3373005a5814ef7` | pending |

`current/<obligation>.proposal.json` contains the proposed argument.
`current/<obligation>.request.json` contains the exact source binding and blank
response template. Every response remains pending. The local snapshot store
also freezes the pinned Lean library seed; it is not a build tree.

## Changes since the previous requests

The preceding requests covered `00226950e`. This source consolidates the default
emitter and application assembly, removes the alternate witness executor and
legacy Rust consumers, proves the general application's physical-row custody,
and uses matching version-3 manifests and selected artifacts. It also removes
unused sampler support, stale evidence consumers and unreachable Metal code.

Fresh checks cover two native folds, Lean verifier acceptance, complete proof
bytes and caller values, terminal rejection of balanced false K/A openings,
PiCCS and PiDEC mutations, 17-source child handoffs, and independent recursive
row/assignment evaluation. The final production and axiom builds, identity
checks, Rust release suites and Metal public flows pass. See
[the completion record](../../../../PR123_FIX_HANDOFF.md) for exact scope and
remaining assurance limits. These tests do not replace independent assessment
of the four mathematical targets.

The proposals retain the target meanings and explicit cryptographic assumptions.
They have new source bindings. The false-acceptance proposal continues to name
`HyperNovaFirstFailure.accepted_failure_exists_first`.

## Handoff to the controlled review process

1. Use a checkout containing these request files and the captured source commit.
2. Give each request, proposal and captured source to an independent reviewer.
   Assess the target meaning, premises, argument, correspondence and parent use
   through the required decomposition questions. Do not relabel an older
   response as a review of the current snapshot.
3. Use the independently provisioned checker, policy and protected library seed.
   If those produce different request identifiers, create matching requests and
   review their exact source.
4. Import the signed response with
   `python3 -B scripts/lean_graph/evidence.py --authority APPROVED_CHECKER --store STORE record-review REQUEST SIGNED_ENVELOPE`.
5. Use that authority, store and source for `explain OBLIGATION`. Accepted closure
   also requires the proof gates and every review named by the policy.

A local import without `--authority` is diagnostic. The implementation author
cannot create independent acceptance by signing their own findings.

## Historical records

`final/` retains the reviews of `eb3f78599`; `initial/` retains those of
`9e9cd23d1`. Both remain historical records. Their response files and the
protected owner files were not changed by this completion.
