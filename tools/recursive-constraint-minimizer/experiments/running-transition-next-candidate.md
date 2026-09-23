# Shared running-transition flag

The shared-flag layout is selected in Lean. Its complete assignment transport,
soundness, constructive witness construction, norm bound, and matrix-program
proofs passed the full 4,224-job library and axiom check. The selected carrier
has **172,217,934 committed coordinates**. With Poseidon pin removal, it has
**4,147,335 logical rows**. The package and parity fixtures have been
regenerated. The independent Rust comparison checked all fourteen logical
matrices and confirmed **2,968,490,185 total nonzeros**. Assignment, regenerated recursive-fixture,
and selected consumer checks passed. No prover benchmark ran.

The original scan used the checkpoint artifact
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`,
SHA-256 `6216d1f62250a58d073ecf0a908bd074d3834957ec5be3361620bbdeb5a97642`.
The hash identifies the inspected bytes; it is not a soundness proof.

## Candidate and costs

Keep the existing inverse field `inv`. Store the recursive flag `g = t * inv`
once as one Boolean coordinate. Replace the transition's multiplication
intermediates with these equations:

```text
t * inv = g
t * (1 - g) = 0
g * (recursive - default) = output - default  [49,353 words]
(1 - g) * (initial - current) = 0             [4 state words]
```

Each equation fits the existing ordinary CCS ports `selector * (A * B - C)`.
No new relation term, matrix, hash, parameter, or security assumption is
needed. The first two equations imply `g = 0` or `g = 1`, including when
`t = 0`. This permits the existing one-coordinate `.bit` encoding. Keep the
inverse's 41-trit field encoding. Projection removes `g`; constructive
extension sets `g = t * inv`. The old R1CS lowering can reconstruct its
scratch values from the resulting logical witness.

| Measure | Quotient checkpoint | Shared-flag change alone |
|---|---:|---:|
| Transition scratch field slots | 296,137 | 0 |
| New Boolean coordinates | 0 | 1 |
| Transition rows | 345,495 | 49,359 |
| Total logical coordinates | 184,359,519 | 172,217,903 |
| Total committed coordinates | 184,359,564 | 172,217,934 |
| Total logical rows | 4,703,127 | 4,406,991 |
| Transition matrix nonzeros | 36,979,978 | 4,158,174, analytical |
| Total matrix nonzeros | 3,001,571,645 | 2,968,749,841, analytical |

The logical saving is `296137 * 41 - 1 = 12141616`. Alignment to 54 gives
31 padding coordinates, so the committed saving is **12,141,630**. The row
saving is **296,136**. These are the proved selected geometry changes. This layout
does not meet the additional target of 92,179,782 committed coordinates.

## Source and dependency evidence

The source owners are:

- [RunningTransition.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/RunningTransition.lean):
  `exactWordCount`, `exactPrivateCount`, `exactRowCount`, `bindingConstraint`,
  `muxConstraint`, `baseStateConstraint`, `soundness`, and `completeness`.
- [RunningTransitionCost.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/RunningTransitionCost.lean):
  `exactFreshCount = 3 + 49353 * 6 + 4 * 4 = 296137` and affine source facts.
- [RunningTransitionRetainedBlocks.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/RunningTransitionRetainedBlocks.lean):
  `freshBlock` retains 296,138 fields: one inverse and 296,137 scratch values.
- [RunningTransitionValues.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/RunningTransitionValues.lean):
  the source scratch interval is `[29040587, 29336724)`.
- [RunningTransitionMatrixProgram.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/RunningTransitionMatrixProgram.lean):
  the current row schedule and source substitutions.

After the Spartan column permutation, the scratch interval is
`[29040309, 29336446)`. The package's transition source rows are
`[28872529, 29218024)`: 296,137 multiplication instructions and 49,358
assertions.

The read-only [scan](running_transition_inspect.py) found **zero** references
to this scratch interval in other ordinary rows and multiplication
instructions, all witness recipe and hint expressions, all explicit
permutation inputs, and all compact-invocation input mappings. It counts
input references even when their coefficient is zero. It does not establish
a universal theorem for all source constructors, implicit hash-chain reads,
the complete retained-assignment schedule, or a new production executor.
Those checks must become Lean support and witness-map proofs before the
scratch allocation is removed. The saved script repeats the inspected
calculation and rejects a different artifact hash.

The subsequent focused Lean custody check passed eighteen axiom audits. It
proves support exclusion for the actual unchanged ordinary prefix and
application rows, the original transition logical constraints, and every
retained block except `runningFresh`. Changing the transition scratch thus
preserves those rows and retained values. It also proves that the existing
full physical witness already stores the Boolean value `t * inv` in the
first scratch cell and satisfies the reduced rows without modification.
The current full witness producer can supply that value; this result makes
no claim about a producer that skips scratch computation.

## Matrix count derivation

The current block contributes these canonical nonzeros:

| Port | Current nonzeros |
|---|---:|
| Selector | 345,495 |
| A | 14,312,998 |
| B | 10,179,868 |
| C | 12,141,617 |
| Total | 36,979,978 |

The scan merges equal source-column coefficients modulo Goldilocks and
counts 41 coordinates per field. The point grids use eight retained S-box
outputs, or 328 nonzeros per point component. These families have disjoint
retained coordinates. The Poseidon external-layer coefficients are nonzero.

For the proposed forms, the serialized running value has 49,248 PiDEC field
words, 56 point components, and 49 length headers. The existing
`Lifecycle/XOut.lean` `defaultRunning` has zero data words. The three running
serializations have equal length headers. Retain even those 49 header rows
as `g * 0 = 0` for this cost calculation.

| New row family | Count | Nonzeros per row |
|---|---:|---:|
| `t * inv = g` | 1 | 84 |
| `t * (1 - g) = 0` | 1 | 44 |
| PiDEC field mux | 49,248 | 84 |
| Point component mux | 56 | 371 |
| Equal length headers | 49 | 2 |
| Base-state equality | 4 | 85 |

Thus the candidate block has `84 + 44 + 49248*84 + 56*371 + 49*2 + 4*85 =
4158174` predicted nonzeros, a saving of **32,821,804**. This is an analytical
count for the specified forms. The full emitted matrix measurement, including
Poseidon pin removal, confirms 2,968,490,185 total nonzeros. This remains above the original pre-quotient baseline of
2,335,822,475 nonzeros.

## Proof status

The [cvc5 script](running_transition_reduction.py) and
[results](running_transition_reduction.json) cover local projection,
constructive extension, Boolean flag validity, and missing-check
counterexamples. They do not replace Lean proofs.

The proved implementation is
[RunningTransitionReducedRows.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/RunningTransitionReducedRows.lean).
The focused [audit wrapper](../../../formal/nightstream-fprime/tests/RunningTransitionReductionResearch.lean)
passed eleven audits: exact equivalence to the selected logical constraints,
specification soundness, constructive completeness, executable witness maps
in both directions, agreement outside the changed intervals, Boolean and
low-norm flag, row count, and local geometry. Only permitted axioms occur.
Validation: `timeout --signal=KILL 1500 bash scripts/validate.sh file
tests/RunningTransitionReductionResearch.lean` from the formal project.

The witness proofs use named constraint equalities and a generic lowering
lemma. They do not expand the production-sized constraint list. These are
maps for the actual transition relation; full selected-assignment transport
is a separate obligation.

The compact matrix proof is now complete. Its six block counts are
`[1, 1, 1, 56, 49296, 4]`.
[RunningTransitionReducedMatrixComplete.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/RunningTransitionReducedMatrixComplete.lean)
proves that actual decoding and the fixed production polynomial accept
exactly the reduced rows. There is no caller-supplied row arithmetic premise.

[RunningTransitionReducedEncoding.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/RunningTransitionReducedEncoding.lean)
connects the shared input blocks, inverse field, and flag bit to the original
source packet. It proves row equivalence, specification soundness, and
acceptance from an existing physical witness. It does not require encoding
the removed scratch fields. The final selected soundness interface now uses
the existing transition specification, rather than requiring obsolete
scratch values to satisfy the old physical rows.

[RunningTransitionReducedPlan.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/RunningTransitionReducedPlan.lean)
provides the composable production plan and proves that every declared row
decodes to its exact sparse forms. Its 49,359-row count and acceptance
equivalence passed the focused axiom audit. The plan uses no source R1CS row
accessor. These proofs reuse the existing matrix vocabulary and polynomial.

Full selected retained-assignment transport is complete. The inverse and flag
blocks replace the old fresh-field block; subsequent blocks shift by
12,141,616 coordinates. The canonical assignment uses the same source values.
The complete norm proof derives flag validity from the actual transition rows,
and the existing full physical witness supplies those rows. The final soundness
proof keeps the unchanged specification. No new final caller assumption is added.
The 27.13% checkpoint remains preserved in Git and in the integration staging
backup. The new artifact's selected Rust conformance passed.

Validation for this source batch: the full Lean library build passed 4,068
jobs; the full axiom/test build passed 4,224 jobs, including the new source
correspondence and production-plan audits. Only `propext`, `Classical.choice`,
and `Quot.sound` occur in the new audited results. Static boundary checks also
passed. The logs are
`/tmp/nightstream-constraint-source-batch-build.log` and
`/tmp/nightstream-constraint-source-batch-axioms.log`. No package or fixture
regeneration, Rust test, or prover benchmark ran in this batch.

The selected integration batch passed `validate.sh axioms` (4,224 jobs), which
also builds the production root, and static checks. The logs are
`/tmp/running-transition-selected-codec-batch.log` and
`/tmp/running-transition-selected-static-final.log`. Earlier source-batch
records above describe the local proof checkpoint before selection.
