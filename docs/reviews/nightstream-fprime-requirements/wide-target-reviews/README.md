# Wide-key target reviews

Status: **pending acceptance.** The four reviews below pass. They are local
diagnostic records. The review gate stays open until the controlled review
process accepts them.

## Final snapshot

Commit `eb3f78599f968f34ae363272d7306c89d7889019`. The lean-graph snapshot
covers only the captured source roots, so a later commit that changes no
captured file (for example this directory) keeps it valid.

| Obligation | Snapshot | Request | Diagnostic record |
|---|---|---|---|
| `stage1-terminal-assignment` | `33e467917a6d313680371677b2efaf0b78838bd930fb2291653574e79ad2b1b0` | `d28bd13df6c11754ed85765e63ce051894fe3245680dc80d4348c2112e495113` | `87ab939abad3f0341870b1ed4f006196daaf59b73a54681b2ed4f3d48c6af791` |
| `stage1-terminal-parent` | `33e467917a6d313680371677b2efaf0b78838bd930fb2291653574e79ad2b1b0` | `0e20de364ec49d81f1c3dbacc288a54f3552420d7f92f70bb91cd5153d6e5fa7` | `07fd87792f83f09a7ad4965ec829f5c2c84732de2ab28e72808655780fe795a0` |
| `hypernova-linear-security` | `cf1d93d89a4533c72848742960353bd657915c34680de775836cb5798db9f1a9` | `cec62fc392f8fca131c5b6a37f4ff08428caa7851474d93a5bcda643d9340182` | `c2ec5d89aa5d33a3c94455b3c474ea945ceaab47b00a2547f9ff3dee79525d0f` |
| `hypernova-terminal-false-acceptance` | `cf1d93d89a4533c72848742960353bd657915c34680de775836cb5798db9f1a9` | `5bc442b34b72b198fbd6ca6b98773a9deac1b6a6374d1051c823c3ffe98e412f` | `cde65f3831413b6d9c28258cf3eb6fa21857d06622d34e0ebe7a0d158a6b8336` |

Each `final/<obligation>.proposal.json` is the author's proposal, and each
`final/<obligation>.json` is the reviewer's response. Every response passes
target meaning and all five decomposition checks (substantiveness, premises,
argument, correspondence, parent use). The reviewers were independent review
agents (Claude) with fresh context and read-only access.

## Required action for the controlled review process

1. Check out commit `eb3f78599`. Below, `DIR` is this directory.
2. For each obligation, run
   `python3 scripts/lean_graph/evidence.py --store STORE review-request OBLIGATION --proposal DIR/final/OBLIGATION.proposal.json`
   and confirm that the request identifier equals the table value.
3. Put `DIR/final/OBLIGATION.json` in an envelope signed by the controlled review
   process.
4. Run
   `python3 scripts/lean_graph/evidence.py --authority APPROVED_CHECKER --store STORE record-review REQUEST SIGNED_ENVELOPE`.
5. Run `explain OBLIGATION` to confirm the accepted record. The tool imports
   only decomposition reviews; each response also carries its
   `target_meaning` assessment for the process to accept.

## Findings and dispositions

- **Minor, final review:** the proposal for
  `hypernova-terminal-false-acceptance` lists
  `HyperNovaFirstFailure.accepted_probability_le_first_failures`; the proof
  uses `accepted_failure_exists_first`. The proposal stays as reviewed, so its
  request identifier does not change.
- **Minor, initial review at `9e9cd23d1`:** stale "candidate" headers in four
  wide Lean files and nonexistent declarations in
  `HYPERNOVA_LINEAR_SECURITY.md`. Fixed before the final snapshot.
- **Capture defect:** four symlinks under `crates/nightstream` point to Lean
  artifacts that the `lean` group excludes, so no gate that captures `rust`
  could run. Fixed in `eb3f78599` by excluding them from the `rust` group.
- **Notes, no change:**
  - `Wide.selected` has no `AuthorityStream.prepare compiled = .ok parts`
    premise, and no theorem uses it. Each target covers every wide target.
  - No Lean theorem proves that `RangePlan.compile?` succeeds or that a
    recursive `target.Holds` can hold; emitter and test runs show both.
  - `Terminal.HoldsFor` builds the narrow `Lifecycle.setup`; the terminal
    transition reads only its verifier keys.
  - The interactive challenge instances name `ProductionKey.key`; the wide key
    changes only `piRlcResponse`.
  - `LowNormInvertibility` stays an explicit premise, as before.
  - Some proof-path docstrings still use the word "candidate".
  - `obligations.json` (the `hypernova-linear-security` argument),
    `HYPERNOVA_LINEAR_SECURITY.md` line 4, `STAGE1_BASELINE.md` lines 500-501,
    `STAGE1_BASELINE_STATUS.md` and `FALSE_ACCEPTANCE_REVIEW.json` still hold
    older text.

The `initial/` directory holds the earlier reviews of commit `9e9cd23d1`.
