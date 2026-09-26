# Poseidon coordinate and row candidates

The selected layout now combines the proved 86-row Poseidon template with
the shared running-transition flag. It has 172,217,934 committed coordinates,
4,147,335 logical rows, and 2,968,490,185 measured matrix nonzeros. The full
Lean proof batch and independent Rust matrix comparison passed. Packages and
parity fixtures have been regenerated; recursive-fixture and selected consumer conformance
passed. Prover benchmarks remain paused.

The research calculations below isolate Poseidon changes relative to the
preserved `961ba3d2e` checkpoint: 184,359,564 committed coordinates,
4,703,127 logical rows, and 3,001,571,645 matrix nonzeros.

## Remove the eight final pin rows of each retained invocation

This is a concrete **row reduction**, with no coordinate reduction.
The retained adapter defines every output as the linear layer of its last
eight S-box outputs. The generic trace computes the same form. Existing Lean
theorems prove this equality for every assignment, without a hash-collision
assumption:

- [PoseidonRetainedFamily.lean:126](../../../formal/nightstream-fprime/NightstreamFPrime/Layout/ProductionRelation/PoseidonRetainedFamily.lean#L126),
  `trace_state_eq_outputState`;
- [PoseidonRetainedFamily.lean:152](../../../formal/nightstream-fprime/NightstreamFPrime/Layout/ProductionRelation/PoseidonRetainedFamily.lean#L152),
  `outputEquations`.

At the checkpoint, `PoseidonSboxFamilyPlan.plan` emitted 94 rows per invocation:
86 S-box rows and eight output pins. The pins test a form against itself.
Removing them preserves the assignment exactly. The witness map and its
inverse are the identity; no output check owned by a separate caller may be
removed on this basis.

| Measure | Current | Candidate | Reduction |
|---|---:|---:|---:|
| Committed coordinates | 184,359,564 | 184,359,564 | 0 |
| Logical rows | 4,703,127 | 4,443,471 | 259,656 |
| Normalized matrix entries | 3,001,571,645 | 3,001,311,989 | 259,656 |

The counts use `32,457 * 8`. Exact coefficient addition modulo Goldilocks
checked all eight final pin templates in the selected saved formula library.
Each value form cancels to zero. Only one selector entry remains in each
row. The scalar matrix expansion removes cancelling entries in
[form.rs:125](../../../crates/nightstream-fprime/src/package/matrix_program/form.rs#L125).
Thus the matrix count above is derived from the actual templates, rather
than from an estimate of their expression size.

[poseidon-pin-controls.json](poseidon-pin-controls.json) binds that inspection
to the formula-library SHA-256 value. cvc5 1.3.4 with `--ff-solver=gb` returned
`unsat` for all eight nonzero-pin queries. A separate caller-owned output
control returned `sat`, with actual value zero and expected value one. These
are search controls; Lean is the proof authority.

[PoseidonPinReductionResearch.lean](../../../formal/nightstream-fprime/tests/PoseidonPinReductionResearch.lean)
proves the exact full-row/S-box-row equivalence and row counts. Its focused
Lean check passed with four axiom audits; only permitted axioms occur.
Validation: `timeout --signal=KILL 1500 bash scripts/validate.sh file
tests/PoseidonPinReductionResearch.lean` from the formal project directory.
Lean now selects the 86-row retained-family plan. Its interface derives the
output from the retained S-box forms; a caller cannot supply a different
output. The generic `PoseidonSboxPlan` still checks all 94 rows for callers
with independent output forms.

The full selected Lean library, production matrix axiom audit, shared-formula
expansion checks, and affected replay modules passed. The selected package
theorem proves 4,443,471 logical rows. Matrix decoding and numeric evaluators
use the same 86-row indexing. The emitted package and Rust row decoder now use this template. The complete
independent comparison of the combined selected layout passed for every
logical row and all fourteen matrices. Final fixture and selected consumer checks passed.

## Constant and shared sponge prefixes

The actual pilot schedule starts from zero and absorbs four words per
permutation. [StateBinding.lean:25](../../../formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/StateBinding.lean#L25)
fixes preimage words 0 through 22 to the state domain tag and word 23 to four.
The first six permutations of both pilot chains are therefore constant.
The same module binds words 24 through 27 to the same verifier-owned context.
Consequently, the seventh permutation has the same complete input in both
chains. This conclusion uses the whole prior sponge state, not only equal
message words. The chain wiring is in
[PilotPoseidonPlan.lean:54](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PilotPoseidonPlan.lean#L54).

Replace the twelve constant traces by their computed constants and use one
shared trace for the seventh permutation. This removes thirteen traces:

- logical coordinates: `13 * 86 * 41 = 45,838`;
- logical rows with the current 94-row template: `13 * 94 = 1,222`;
- normalized matrix entries: not yet counted for the replacement layout.

The next absorbed block starts at word 28, the iteration counter. The output
counter is the prior counter plus one, so equal context words do not justify
sharing the remaining chains. A prefix proof must retain the state-framing
rows that establish the constants and context equality.

There is another constant prefix in PiCCS. Its zero initial state is defined
in [PiCCSPoseidonPlan/Retained.lean:131](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiCCSPoseidonPlan/Retained.lean#L131).
The first absorb action contains the 43-word digest domain tag. Padding gives
44 constant words, hence eleven complete constant invocations. The action
comes from [StatementAbsorption.lean:824](../../../formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/StatementAbsorption.lean#L824).
The selected package's PiCCS payload forms have exactly 45 initial constant
words; the first variable payload is the prior public digest. The additional
constant length word starts the next invocation and cannot remove it.

The eleven PiCCS traces add a possible 38,786 logical-coordinate and
1,034-row reduction. Together, the two prefix proposals remove 24 traces:
84,624 logical coordinates, giving 184,274,895 logical coordinates and
184,274,946 committed coordinates after ring alignment. The committed saving
is 84,618, about 0.046% of the current carrier. The current-template row
saving is 2,256. These are candidate layout counts; complete witness maps,
source substitution, and Lean proofs are not yet established.

If combined with pin removal, count only 86 removed rows per eliminated
trace. The eight pin rows must not be counted twice. No candidate here comes
close to the requested additional 50% coordinate reduction.

## Sources and scope

[CLAP section VI-C](https://arxiv.org/html/2405.12115v2#S6.SS3) motivates
common-expression elimination and section VI-D states the need for witness
correspondence. The local prefix proposal applies that technique to proved
equal full sponge inputs. CLAP's reported row counts do not transfer directly
to this CCS, and its optimizer is not a proof authority for this code.

The four PiCCS action families form one continuous sponge chain. The sampler
also continues the final PiCCS state and then continues across its 17 source
entries; it does not reset to zero at each source. Repeated tags or equal
source words therefore do not provide further equal-input traces. Inspection
did not establish a large repeated region or a new multivariate degree-below-
nine encoding. This is an open candidate search, not a global lower bound.
