# Replay coverage after the sampler replacement

The six full-replay registrations retired by PR #124 combined mathematical
kernel claims with execution by the old coordinator. Their kernel theorems
remain. The map now registers those claims separately from the maintained
native/Lean workflow and the completed author-run independent generation.

Registry schema 2 records a `tier`: `Compiler`, `Conformance`, or `Production`.
It cannot declare a status. Reports derive `status: Open` whenever a required
gate, review, or other requirement is missing. Only accepted closure yields the
owner-defined `Compiler-closed`, `Conformance-closed`, or `Production-closed`
status. Schema 1 and authored status claims are rejected.

| Retired obligation | Retained kernel registration and target | Current execution coverage |
|---|---|---|
| `pirlc-witness-replay` | `pirlc-witness-kernel`: `PiRLCWitnessReplay` | Both independent folds match every native parent coefficient, including full carrier tails and changed-tail rejection. |
| `pidec-witness-replay` | `pidec-witness-kernel`: `PiDECWitnessReplay` | All 16 child witnesses match in both folds through complete validated signed masks, including implicit zeros and tail rejection. |
| `pidec-commitment-replay` | `pidec-commitment-kernel`: `PiDECCommitmentReplay` | Every child commitment coefficient matches in both folds; complete NIFS values and bytes also match. |
| `pidec-evaluation-replay` | `pidec-evaluation-kernel`: `PiDECChildEvaluationReplay` | Both folds match separate Pad values and all 14 matrix families. Iteration-four balanced opening tests reach the exact late Eval_K/Eval_A errors. |
| `fresh-witness-replay` | `fresh-witness-kernel`: `FreshWitnessKernels` | Both independent proofs produce matching callers, full physical/logical witnesses, every active canonical row, fresh commitments and public claims. Required rejection checks pass. |
| `fresh-recursive-loop` | `recursive-loop-kernel`: `CheckedRecursiveReplay` | The exact first Lean successor supplies the second fold; both complete proofs/successors and iteration-four terminal validation pass. |

Each kernel registration retains its existing statement, axiom audit, checked
closure and declaration-export command. `FreshWitnessKernels` and
`CheckedRecursiveReplay` are defined in their own evidence files. Their explicit
geometry/custody premises remain visible; registration does not discharge them
through an execution claim.

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

## Independent-generation status

`independent-generation` remains open. The maintained
[independent coordinator](../INDEPENDENT_GENERATION.md) generates the PiCCS proof from the
original source witnesses, the PiRLC parent witness, and all PiDEC child
witnesses, commitments and evaluation families in Lean. That run must compose
complete proof encoding, the fresh caller/witness/commitment and both recursive
successors, with exact native comparisons and rejection checks.

The complete author-run sequence now passes: both independently generated
PiCCS/PiRLC/PiDEC proofs, every parent/child coefficient, all commitments, separate
Pad and all 14 matrix families, exact NIFS bytes, both fresh successors and
iteration-four terminal acceptance/rejections. The exact first Lean successor
supplies the second fold. The [execution index](../../docs/reviews/pirlc-sampler-replacement/INDEPENDENT_EXECUTION.json)
binds the command evidence and handoffs; the [report](../../docs/reviews/pirlc-sampler-replacement/REPORT.md)
states scope and limitations. Independent review of this evidence remains open.

The `independent-coordinator-contract` gate checks checkpoint integrity and
requires the second source projection to use the exact first Lean successor.
It binds the Lean producers, native comparison code and coordinator sources.
These tests cannot close the required complete two-fold execution or its
independent review. The retained norm-prefix gate now uses the current carrier:
43,484,783 values per source after two folds, for 739,241,311 encoded values
across all 17 sources. Decoding yields 11,827,860,976 bytes.

The remaining phase-specific producer gates and all six kernel proofs are
useful components. Historical old-package receipts and the current golden
workflow do not establish this complete execution. Kernel compilation, executed
conformance and independent review remain separate requirements.
