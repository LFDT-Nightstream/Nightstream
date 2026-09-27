# Replay coverage after the sampler replacement

The six full-replay registrations retired by PR #124 combined mathematical
kernel claims with execution by the old coordinator. Their kernel theorems
remain. The map now registers those claims separately from the maintained
native/Lean workflow and the missing complete independent-generation run.

Registry schema 2 records a `tier`: `Compiler`, `Conformance`, or `Production`.
It cannot declare a status. Reports derive `status: Open` whenever a required
gate, review, or other requirement is missing. Only accepted closure yields the
owner-defined `Compiler-closed`, `Conformance-closed`, or `Production-closed`
status. Schema 1 and authored status claims are rejected.

| Retired obligation | Retained kernel registration and target | Current execution coverage |
|---|---|---|
| `pirlc-witness-replay` | `pirlc-witness-kernel`: `PiRLCWitnessReplay` | Golden checks native PiRLC through Lean verification and subsequent assignment comparisons; it does not regenerate the complete parent witness independently in Lean. |
| `pidec-witness-replay` | `pidec-witness-kernel`: `PiDECWitnessReplay` | Native split, verifier checks, terminal opening checks and fresh physical-witness comparison; no complete independent Lean child-witness generation run. |
| `pidec-commitment-replay` | `pidec-commitment-kernel`: `PiDECCommitmentReplay` | Native commitments and maintained opening/child-handoff checks; no complete independent Lean generation of all child commitment coefficients. |
| `pidec-evaluation-replay` | `pidec-evaluation-kernel`: `PiDECChildEvaluationReplay` | Separate `Eval_K` and `Eval_A` terminal checks and current matrix comparisons; no complete independent Lean generation of every child evaluation family. |
| `fresh-witness-replay` | `fresh-witness-kernel`: `FreshWitnessKernels` | Golden independently reconstructs and compares every physical witness value from the checked caller. That caller uses native PiCCS proof messages. |
| `fresh-recursive-loop` | `recursive-loop-kernel`: `CheckedRecursiveReplay` | Two fresh native folds and successors, Lean verification and exact handoffs, and terminal acceptance/rejections. It does not generate the complete proof independently in Lean. |

Each kernel registration retains its existing statement, axiom audit, checked
closure and declaration-export command. `FreshWitnessKernels` and
`CheckedRecursiveReplay` are defined in their own evidence files. Their explicit
geometry/custody premises remain visible; registration does not discharge them
through an execution claim. No theorem or production code changes in this repair.

## Maintained golden workflow

`golden-conformance` registers the workflow described in
[the execution instructions](../GOLDEN_CONFORMANCE.md):

```sh
python3.12 -B scripts/golden_conformance_ci.py --directory NEW_DIRECTORY
```

Run this coordinator outside the graph lock. Each native/Python test child
retains its 300-second cap, and each Lean child uses `validate.sh` under its
1,500-second cap. The coordinator produces both native folds and verifies their
native proof messages with Lean, compares all proof bytes and caller words,
independently reconstructs every physical witness value, and checks terminal
acceptance and the required rejection cases.

The graph's `golden-coordinator-contract` gate runs the existing coordinator
regressions. It binds their source files and checks failure/receipt/byte-handoff
behavior. It does **not** run the complete fold workflow. The obligation therefore
keeps full source-bound execution and its independent review as an explicit
open requirement. Passing those unit tests cannot mark the full workflow closed.
The completed author-run execution at `0846e4df2` is recorded in
[VALIDATION.json](../../docs/reviews/pirlc-sampler-replacement/VALIDATION.json).
It is local evidence, not protected-checker acceptance.

## Coverage still missing

`independent-generation` remains open with no implemented closing gate. Its
scope is a complete current-package run that generates the PiCCS proof from the
original source witnesses, the PiRLC parent witness, and all PiDEC child
witnesses, commitments and evaluation families in Lean. That run must compose
complete proof encoding, the fresh caller/witness/commitment and both recursive
successors, with exact native comparisons and rejection checks.

The remaining phase-specific producer gates and all six kernel proofs are
useful components. Historical old-package receipts and the current golden
workflow do not establish this complete execution. Restoring the registrations
records this difference; it does not restore the missing generation coverage.
