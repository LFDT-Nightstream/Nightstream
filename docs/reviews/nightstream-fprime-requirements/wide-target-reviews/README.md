# Wide-key target reviews

Status: **pending independent review and signed acceptance.** Current requests
are captured below. The older passing responses do not approve this source.
No controlled checker or signing process has been supplied to this task.

## Current source and requests

Captured source commit: `00226950e51f389e09ecb4cfdadb845d14b36908`.
The capture equals that committed tree: the working tree had no other
changes, and the source capture excludes Python bytecode caches. Later
changes only to this review directory do not change the captured source.

These requests replace the ones captured at `ce0159538`. Two changes since
then are inside the review scopes. The `AGENTS.md` rule for the removed Lean
project changed (`owner` group, all four requests). The security requests
also capture `crates/neo-math/tests/fixtures/lean-foundation.zip` (`rust`
group); the earlier security capture held an uncommitted copy of that file,
so no commit reproduced it. No review existed for the replaced requests.

| Obligation | Snapshot | Request | Review |
|---|---|---|---|
| `stage1-terminal-assignment` | `ec3c2a7ab788ba5b6e5f2bfc36d2b68b54cc86883af784695ba5c6887f84b62e` | `c7b046303e0497de7248356768e3111e8479bfba52cdb9f43bf26691bcd8995b` | pending |
| `stage1-terminal-parent` | `ec3c2a7ab788ba5b6e5f2bfc36d2b68b54cc86883af784695ba5c6887f84b62e` | `b2b1b762a16cc806c4c6c91a3b67e6dec6aea1d61c80ec0b617e679d96d93224` | pending |
| `hypernova-linear-security` | `b6c02cf5a635fcd85769b95245fe22776f962a7c3e070d44c9dc8d75125a7611` | `d20b51575af8cd33add756ddceb28140906ab7cfcd7541ea1084535a9129e4cf` | pending |
| `hypernova-terminal-false-acceptance` | `b6c02cf5a635fcd85769b95245fe22776f962a7c3e070d44c9dc8d75125a7611` | `e48c6d7063f9ed66a223df0fe9963c45df91ad865bb7625798d1309495293d58` | pending |

`current/<obligation>.proposal.json` is the proposal.
`current/<obligation>.request.json` contains the exact request, its binding,
and the blank response template. No template is a passing review record.
The requests were captured in a temporary local store. The controlled
review process recaptures them from the source commit above with its own checker.

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
