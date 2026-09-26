# Wide-key target reviews

Status: **pending independent review and signed acceptance.** Current requests
are captured below. The older passing responses do not approve this source.
No controlled checker or signing process has been supplied to this task.

## Current source and requests

Captured source commit: `ce0159538534579f3459adacecfca325b5b9692b`.
The requests remain valid at `31148bcce`: its CI workflow, local check scripts
and saved primitive archive are outside these four declared review scopes.
Both terminal request IDs were recomputed and are unchanged; the security
scopes add only the unchanged security reference documents.
Later changes only to this review directory do not change the captured source.
The source capture excludes Python bytecode caches.

| Obligation | Snapshot | Request | Review |
|---|---|---|---|
| `stage1-terminal-assignment` | `f1218dc180ec00b52faf5c520dd8fa1f4496159ecbdda3bda9fa4be926616c86` | `4fe5a5fc3ead7e00d687c2173059e448c8811dee36e473ce460a72fe9cf389d4` | pending |
| `stage1-terminal-parent` | `f1218dc180ec00b52faf5c520dd8fa1f4496159ecbdda3bda9fa4be926616c86` | `95fe9e5407111d7a0ec69d5c45b65bf8ba96f78601da5189384ab95545728b9d` | pending |
| `hypernova-linear-security` | `531d27f5cfa99efc945073ec744fbcbd6f4aa544a1749b421d0a996cbc5c74f3` | `b06476363a187ae6ce029461cf7bafd04ea4e1963d5ab813ce23391459549c79` | pending |
| `hypernova-terminal-false-acceptance` | `531d27f5cfa99efc945073ec744fbcbd6f4aa544a1749b421d0a996cbc5c74f3` | `fb6cb1592b282bdfc424dd78cf1b2a6c8a3f205393ee122856b7073f37686911` | pending |

`current/<obligation>.proposal.json` is the proposal.
`current/<obligation>.request.json` contains the exact request, its binding,
and the blank response template. No template is a passing review record.
The local captured store is `/tmp/nightstream-review-ce0159538.cHjmiI`;
retain or transfer that store when arranging the controlled review.

## Changes since the previous reviews

The only changed Lean source since `eb3f78599` is
`Export/FoundationParityMain.lean`, which emits the new arithmetic test
vectors. The four target definitions and their security proofs are unchanged.
The Rust and Python changes repair golden checks, retain the test inputs,
and add parity tests. Local static and axiom gates pass.

The current false-acceptance proposal names
`HyperNovaFirstFailure.accepted_failure_exists_first`, correcting the old
proposal's dependency list. That correction deliberately changes its request.

## Handoff to the controlled review process

1. Use a checkout containing these `current/` files and the captured source
   above. Do not check out the earlier commit that predates the review files.
2. Give the captured source, proposal and request to the independent reviewer.
   The reviewer must fill the response template, including all five
   assessments and the target-meaning assessment. Do not relabel an earlier
   response as a review of the current snapshot.
3. The authorized process must use its independently provisioned checker,
   policy and library seed. If those produce different request identifiers,
   prepare new requests under that checker and review their matching source.
4. Import the signed response with
   `python3 -B scripts/lean_graph/evidence.py --authority APPROVED_CHECKER --store STORE record-review REQUEST SIGNED_ENVELOPE`.
5. Use the same `--authority`, store and source when running
   `explain OBLIGATION`. Accepted closure also requires the current proof gates
   and the other reviews required by the policy.

A local import without `--authority` remains diagnostic. This task cannot
create an accepted record by signing its own findings.

## Previous review records

- `final/` retains the proposals and passing responses for source
  `eb3f78599`. They are historical context, not current source approval.
- `initial/` retains the earlier reviews of `9e9cd23d1`.
- The stale headers and the nonexistent theorem reference reported in the
  initial reviews were fixed before `eb3f78599`.
- The source-capture exclusions for the four Lean-artifact symlinks remain
  unchanged. Artifact inputs are separate from captured Rust sources.
- The existing cryptographic assumptions and unproved compiler-success and
  recursive-terminal existence claims are unchanged by these test-tool fixes.
