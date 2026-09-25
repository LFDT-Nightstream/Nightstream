# Wide sampler integration checkpoint

Base: `be5b7fcdc`. Worktree branch: `nico/pirlc-wide-sampler-integration`.
The wide package is selected in production since `c65a8605a`. The sections
below are dated checkpoints; later sections supersede earlier limits.

The candidate implements the merged whole-vector sampler with the existing
hint vocabulary. Its deterministic schedule enters `[4, i]`, reads one joint
four-field block, and advances Poseidon2 once for each of the 17 scalars.
The map is the specified reduction modulo `5^54`, followed by centered digits.
There is no rejection, retry, or shortfall.

## Implemented and proved

- The exact helper program, including integer bounds before field operations,
  limb regrouping, division by five, and the biased CRT check quotient.
- Soundness and constructive completeness of the executable scalar gadget and
  its 17-scalar lifecycle and the complete candidate PiRLC phase.
  Temporary digit words preserve the existing R1CS ring-recipe shape;
  compact CCS ring rows consume the checked digit bits directly.
- The compact sampler matrix plan: 34 Poseidon permutations and 17 range
  gadgets. Accepted rows imply the exact transcript schedule and scalar map.
- A constructive assignment for that plan, preservation of caller columns,
  exact retained-block encodings, and the unchanged strict norm bound `b=2`.
- The challenge forms used by ring products are the checked digit bits minus
  two. They allocate no separate field witness.
- Temporary helper replacement and reconstruction preserve all checked rows.
  The compact assignment omits those helpers.
- The candidate Stage 1 physical prefix through PiDEC and the running
  transition has exact source endpoints. The PiCCS-to-PiRLC and PiRLC-to-PiDEC
  input interfaces use the actual canonical source columns.
- The sampler and all 969 quotient products compose into a 119,153-row
  candidate plan. Its soundness theorem uses the exact sampled challenges.
  Its direct witness constructor computes the outputs and quotient coefficients
  from the checked challenges and the actual Stage 1 right-operand forms.
  It allocates no R1CS ring-multiplication scratch.

The proof entrypoints are in
`NightstreamFPrime/Layout/PiRlcWideSampler/{Completeness,StateSemantics,Norm,Challenges}.lean`.
The executable hint program is
`NightstreamFPrime/Gadgets/Sampling/WideReduction/Program.lean`.

## Component costs

These are sampler costs, not selected F′ totals.

| Measure | One range gadget | Complete 17-scalar sampler |
|---|---:|---:|
| Logical CCS rows | 681 | 14,501 |
| Retained coordinates | 937 | 135,813 |
| Temporary hint helpers | 1,404 | 23,868 |
| Temporary digit words for R1CS recipes | — | 918 |
| Normalized matrix nonzeros | 8,364 with the Poseidon endpoint input | 1,794,350 |

Rows and retained-coordinate counts are proved in Lean. Matrix nonzeros are
measured by exact field arithmetic from Lean-emitted forms and round constants;
the numeric nonzero total is not a Lean theorem.

Each scalar retains 609 bit values and eight field auxiliaries. Each field
auxiliary uses the existing 41-coordinate encoding. The helpers allocate no
rows or matrix entries. Their execution still does work: 256 input-bit hints,
16 accumulator hints and 592 limb/carry-bit hints, and 540 division hints per
scalar. The unchanged checked gadget also allocates its canonical children
and 353 result bits.

The matrix count uses the previous Poseidon endpoint as input, including its
actual external linear layer. It includes coefficient merging and cancellation.
The sampler total excludes the ring-product matrices. A separate measurement
of their left-operand port gives 229,698,543 baseline entries and 16,904,205
candidate entries across 969 products: 212,794,338 fewer entries. These values
come from normalized Lean-emitted forms at all 108 evaluation points. The
other product ports and the complete selected package are not part of that
measurement. The earlier standalone range count of 7,216 used four separate
field inputs; that is a different input geometry.

## Security scope

Let `p` be Goldilocks, `M=p^4`, `N=5^54`, and `r=M mod N`. The proved per-scalar
statistical distance is `δ=r(N-r)/(N*M) < 2^-132`.

`OracleModel.lean` defines a classical ideal block oracle. Its keys are complete
normalized histories of rate additions, including zero additions for digest
advances. A new key receives one joint uniform four-field block. The cache
returns the same raw block for repeated keys. The adaptive comparison counts
all oracle calls, including adversarial calls and repeats, and bounds the
change in any bounded test by `q*δ`.

The comparison keeps raw blocks consistent with their scalars: it samples a
uniform scalar and a raw preimage conditioned on that scalar. It does not put
an unrelated uniform scalar beside an unchanged raw block.

`ScheduleLaw.lean` proves that the 17 scheduled keys are distinct. Its batch law
requires that those keys are absent from the incoming cache. Under that fresh
batch experiment, it gives the exact law used by V6. The `17*L*δ` bound applies
to `L` independent fresh batches. It is not a bound for an arbitrary adaptive
Fiat–Shamir adversary; that experiment uses its full query budget.

V6 still refers to the existing uniform-challenge interactive extractor. Its
`n/|C|` extraction loss remains separate. These modules do not prove a concrete
Poseidon2 ideal-oracle theorem, a Fiat–Shamir extractor, or a quantum-query
bound. Any application of an established cryptographic reduction must state
its model and applicability conditions. The concrete-to-ideal error remains
an explicit, separate premise in `under_block_oracle_assumption`.

`TranscriptHistory.lean` proves that replaying a normalized history from any
seed with additive Poseidon2 gives exactly `Transcript.drawAt`. The wide key's
`response_history` theorem states the same fact for the key's response. No
theorem identifies the verifier's PiCCS output state with the replay of the
complete transcript history, so the block-oracle bound is not consumed by
the extraction chain; see the FS boundary note.
`WideSamplerSecurity.no_sampler_abort` holds for every concrete initial state.
`any_test_le` unions only the PiCCS test events; it has no old shortfall term.

`Lifecycle/Nifs/WideFiatShamir.lean` applies the existing owner-approved
parametric transfer boundary to the wide verifier. Its real event requires
acceptance and openings for the verifier's exact sixteen returned children.
The proved source bound retains `g Q p_real - deltaFS Q - weakLoss - testError
- 17 * adaptiveMsisSuccess`. It reuses the existing interactive extractor and
does not supply a numerical Fiat–Shamir model.

`WideSamplerSecurity.adaptive_bias_bound` uses that same verifier event as a
test of a complete adaptive block-oracle trace. `concrete_bias_bound` gives
`|p_real - p_balanced| <= modelError + q*δ` under an explicit concrete-to-ideal
approximation premise. The balanced raw-block experiment is not identified
with the interactive extractor. A general `g` need not preserve additive
errors: these results do **not** replace `g Q p_real` by `g Q p_balanced - q*δ`.
An external game transfer must establish any further composition it uses.

`q` counts all block calls, including adversarial calls and repeats. `Q` counts
permutation calls in the Fiat–Shamir experiment, including preprocessing and
replays. No equality or query inflation between them is assumed. For independent
fresh batches only, the separate V6 bound uses `17*L*δ` and `17*L/|C|`.
Numerically, `log2(δ) = -132.979646922085`, `log2(17*δ) = -128.892184080835`,
and `log2(|C|) = 125.384117123918`, from the exact formula above. These are
component quantities, not a deployed security level.

The owner approved the changed map and schedule on 2026-09-24. Poseidon2,
the challenge set, and the Nightstream Goldilocks profile remain unchanged.
The [duplex-sponge paper](https://eprint.iacr.org/2025/536) motivates keeping
its precise model separate from an ideal block experiment; this work does
not claim to instantiate its knowledge-soundness theorem.

## Stage 1 candidate connection

`Wide/Stage1Plan.lean` assembles the existing pilot, PiCCS, PiDEC, running,
application, and final binding plans with the wide PiRLC plan. A partial
coordinate map removes the old sampler and First-54 intervals. Common retained
coordinates keep their internal order. The complete PiRLC allocation follows
them, so its direct witness cannot overwrite its inputs.

Lean proves that the retained map is bounded and injective on its domain.
The candidate now keeps the reference family/source/block/lane/cell order
for both outputs and quotient coefficients, as requested by the owner on
2026-09-24. Each product region moves by a constant shift. The former
ring/lane permutation module has been removed.
`InputSupport.lean` proves that the actual initial transcript forms and every
right operand read common coordinates. `Stage1Witness.lean` supplies the direct
PiRLC completion on these forms, preserves common values, and identifies the
final ordered combinations. `PiDECOutput.lean` proves that all four PiDEC
parent views read those same retained outputs, including both extension-field
cells in the reference order. `SelectedAssignmentCompleteness.complete` now
constructs the complete accepting candidate assignment.

`HashChainCounts.lean` proves these counts of the **assembled, unselected
candidate**, using the current application:

| Measure | Selected baseline | Candidate plan |
|---|---:|---:|
| Logical CCS rows | 3,588,191 | 3,248,956 |
| Logical coordinates | 149,292,999 | 137,341,846 |
| Alignment coordinates | 45 | 26 |
| Committed coordinates | 149,293,044 | 137,341,872 |
| Normalized matrix nonzeros | 2,857,409,270 | 2,607,606,765 |

The candidate dimensions are derived from the assembled row plan and retained
intervals. They are not counts from a regenerated production artifact, and are
not an achieved production or overall performance reduction. Whole-package
witness construction, step soundness, and whole-program matrix
correspondence are proved. The exact-arithmetic matrix count is recorded below.

The old sampler's ordinary retained blocks contain 2,230,400 + 6,758,112 =
8,988,512 coordinates. The external funnel's 8,998,512 entry is 10,000 too high;
the selected package's total has not changed.

The candidate NIFS key in `Lifecycle/PiRLC/Wide/Key.lean` uses the exact new
response map. Lean connects that response to accepted candidate phase values
and proves that the profile and challenge-set cardinality remain unchanged.
The candidate security consumers now use this key; production selection is
still separate.
`Wide/FixedPoint.lean` derives the recursive relation at the candidate width
from its own matrices and proves that reassembly against that relation gives
the same Stage 1 plan. The old zero-matrix seed supplies no semantic authority.

## Checked coordinate remapping

The retained-coordinate map now requires a proof that each source coordinate
has an image. Removed coordinates cannot be passed to `RetainedLayout.column`.
`renameForm` maps every stored entry with that proof; it neither substitutes a
constant nor drops an entry. The map is injective, and the stored entry count
is preserved.

`Stage1Plan.rename` requires support for every row and sparse port. The six
reused phases supply structural Lean proofs: the complete PiCCS prefix, PiDEC,
the running transition, the application, next-preimage binding, and public-output
binding. The PiCCS prefix proof includes both pilot hashes, PiCCS hashing and
arithmetic, framing, and digest pins. The application proof covers both supported
application layouts. The proofs follow sparse constructors and retained blocks;
they do not enumerate the row or coordinate domains.

The direct PiRLC witness and PiDEC output-form proofs use this checked map.
PiDEC still reads the same four product families in the proved lane/cell order.
The fixed-point proof and dimension theorems remain valid. No new reduction is
claimed from this correctness repair.

`tests/WideRetainedSupport.lean` rejects all three removed coordinate intervals,
including a stored entry with coefficient zero. It also checks that a column
cannot be mapped without a support proof, and that public column zero retains
its own identity. The complete candidate witness construction now uses this support result;
the old PiDEC witness builder is not a candidate proof premise.

## Retained witness projection

`CoordinateRecovery.source?` is an executable inverse of the retained map.
Lean proves that every result is a live, bounded reference coordinate and that
mapping a retained coordinate forward and back recovers the same coordinate.
Output and quotient coordinates recover their reference indices by subtracting
the corresponding block start; no cell/lane permutation is needed.

`AssignmentProjection.project` preserves the evaluation of every certified
sparse form and the row equations of each reused plan. This is an equality for
arbitrary values, before assuming row acceptance. It does not claim that an
arbitrary reference witness satisfies the new relation.

The final constructor copies only the common coordinate prefix, then uses the
proved direct wide PiRLC constructor. Lean proves exact equality with the result
of projecting all retained reference values first. Old sampler coordinates,
ring outputs, and quotient coordinates are not queried by this constructor.
The new PiRLC rows and the strict `b = 2` norm bound are proved. This does not yet
remove the corresponding work from the selected Rust witness program.

`PiDECSource` gives the new physical address for each of the seven existing
PiDEC location kinds. The four parent-product families move by 208,165 source
columns; proof inputs, split values, and scratch fields move by 925,480. Lean
connects the parent reads to the actual new PiDEC interface, including both
extension-field cells. The cvc5 controls in `wide_source_controls.py` give a
counterexample to one global shift, and find no counterexample to the two
shifts, source disjointness, or the running-transition boundary.

`SourceAssignment` now constructs the common values from the wide physical
source and the application values. It uses a source-address view and the proved
field encoders. It does not execute the old sampler. Removed sampler and
ring-scratch intervals have no source image. Lean proves the source bounds,
all seven PiDEC field reads, and acceptance of the direct PiRLC rows.

`ConstraintRenaming` proves exact ordered-row equality for the existing
optimized R1CS compiler under column renaming, including direct recipe rows
and scratch allocation. `PiDECLeafRenaming` composes this through the scalar
split and radix leaves. `PiDECSourceRenaming` connects all four families to the
actual old and new PiDEC interfaces. These proofs are structural; they do not
scan the 25,488 physical rows.

`PiDECSourceWitness.rowsHold` proves that an accepted wide physical PiDEC
witness accepts the reference source view under the existing PiDEC input
assumptions. `PiDECWitnessInputs` also proves that the final compact witness
preserves PiDEC proof, split, and scratch values, and that its parent forms read
the direct ordered fold.

`PiRLCSourceInputs` now proves that the direct constructor reads the checked
physical PiCCS final state and the same ring operands. The PiCCS scope proof
preserves that phase when reading the wide source view. The existing final-layer
proof accepts any canonical raw packet with the copied base, so it does not
require construction of old sampler-derived values.

`PiRLCSourceOutput` connects all four direct ordered sums to the physical PiDEC
parents. `Wide.PiDECCompletedAssignment.rowsZero` then proves acceptance of all
25,488 compact PiDEC rows on `SourceAssignment.assignment`, from accepted
physical C, wide R, and D phases under their existing input assumptions. The
proof uses every PiDEC source location and the exact ordinary-row compiler;
it does not add a row-validity callback or an old sampler-success premise.

`Wide.PrefixCompletedAssignment.rowsZero` now proves acceptance of all
3,054,685 pilot and PiCCS prefix rows on the same `SourceAssignment.assignment`.
The source view preserves the accepted physical prefix. Each of the six compact
components reads only common retained values, and the direct PiRLC constructor
preserves those values. The existing pilot and PiCCS completeness proofs now
accept any packet with the copied base. No old sampler execution, old PiRLC row,
or sampler-success premise is used.

`Wide.RunningCompletedAssignment.rowsZero` connects all 49,359 compact
transition rows to the accepted wide physical transition. The proof transports
its input values and logical equations, including the first scratch value used
as the shared flag. It does not enumerate the physical rows.

`Wide.ApplicationCompletedAssignment` constructs the three application
permutations from the same source state and preserves the exact four advice
words. Its support certificate includes only copied coordinates. The five
next-preimage rows and four public-output rows also accept the same assignment.

`Wide.CompletedRows.rowsZero` composes all seven candidate plan components:
all 3,248,956 rows accept one `SourceAssignment.assignment`. Its premises are
an accepted wide physical prefix, the existing pilot/C/R/D input assumptions,
the next-preimage specification, and the application step. It has no premise
that the compact rows hold and no old sampler-success premise.

`Wide.AssignmentNorm` now proves the strict magnitude bound below two for
every logical coordinate. It checks only the copied hash prefix and common
suffix; the removed First-54 bit blocks need no validity premise. Field slots
use the existing total encoder. Accepted wide running-transition rows prove
that the single copied branch flag is a bit. The direct PiRLC constructor
supplies the bounds for its own allocation.

`Wide.CarrierAssignment.values` extends that same logical assignment to the
candidate carrier with zero padding. Its theorems prove preservation of every
logical value, zero in every padding position, the full carrier norm, and the
exact public encoding of the physical pilot output digest. This closes the
norm and public-coordinate transport obligations from the physical source.

`Lifecycle.Stage1.Wide.Relation.StepHoldsFor` instantiates the existing HyperNova
transition with the candidate wide-sampler key. Its prior and next state-hash
preimages equal the baseline preimages for the same state data and `vk` argument.
This does not equate the old and new production digests: package selection must
bind the new verifier identity.

## Complete witness from a semantic step

`Layout.Stage1.Wide.StepPhysicalCompleteness.complete` now constructs the
physical source from a candidate semantic step and accepted wide-key NIFS
advice. It completes the pilot, PiCCS, wide PiRLC, PiDEC, and running transition
in order. Each stage preserves all earlier source values and physical rows.
The PiDEC input loader reads the new PiRLC parent and the exact proof messages.
Its sixteen-child result equals the new verifier's result.

`Export.Stage1.Wide.SelectedAssignmentCompleteness.complete` uses that source
to construct the complete compact CCS assignment for `Poseidon2HashChainV1`.
One theorem proves all
3,248,956 rows, the strict magnitude bound below two on all carrier coordinates,
the exact public digest, and the unchanged four-word application advice. There
are no physical-row, sampler-success, or witness-transport premises. As in the
baseline theorem, the inputs include well-formed state encodings, the fresh
public-input binding, accepted NIFS advice, its recursive output equality, and
the application's witness width. No stronger security assumption is introduced.

The physical source is a proof construction. It does not require the production
Rust witness generator to compute discarded R1CS ring scratch. The compact
PiRLC witness still constructs its own products and quotient coefficients
directly. Rust transport remains to be connected to this construction.

Whole-program matrix correspondence and the full normalized nonzero count
are now complete. The selected security and Rust connections remain open.
The candidate remains at 137,341,872 committed coordinates.

## Whole-step soundness for arbitrary assignments

`Wide.FixedPointSoundness.rowsZero_implies_stepHoldsFor` proves that every
accepted candidate assignment with constant column one satisfies the complete
HyperNova augmented step. It selects `Wide.FixedPoint.relation` and the wide
NIFS key. It decodes the input, output and application advice from the same
assignment. No canonical witness, physical-row satisfaction, sampler success,
or retained-block encoding premise is required.

`Wide.PublicBinding.step` now derives the constant-one premise from the actual
carrier public-input projection. It also proves that the output digest decoded
from any accepted compact assignment equals the verifier-supplied digest. The
proof uses pointwise public-coordinate equalities; it does not normalize the
complete concrete layout. There is no canonical-witness premise.

The context digest is decoded from the constrained state; its identification
with the verifier-owned package context is closed later by
`ContextBinding.step_or_collision` and `SetupBinding`.

`Wide.DecodedPrefix` recovers the pilot and PiCCS contracts. The direct PiRLC
inputs use the checked PiCCS transcript endpoint and values. `DecodedPiDEC`
recovers the retained child fields and their parent values. `DecodedAccumulator`
composes these contracts into acceptance by the new NIFS key and proves that
auxiliary physical proof values do not change the decoded input, proof or
output. The running transition and application use those same values. The
final composition proves both HyperNova branches and the exact next-state
hash serialization, including field reduction of the iteration word.

The complete witness theorem in the preceding section supplies the other
direction. This proof batch changes no row, coordinate or matrix count and
does not select the candidate for production.

## Compact matrix decoder batch

`Wide.ReusedMatrixPrograms` proves exact sparse-form equality for all six
reused phases: the pilot/PiCCS prefix, PiDEC, the running transition, the
application, next-preimage binding, and public-output binding. The row
accessor remains explicit. `source_custody` constructs its contract from the
actual Lean-authored package for the candidate relation. There is no
old-sampler row in that contract.

`MatrixProjection.column_eq` proves that the serialized interval map is the
checked retained-column map. The matrix interpreter rejects missing and
out-of-range coordinates, including zero-coefficient entries. The generic
`Exact.mapColumns` theorem connects that interpreter to the supported Lean
plan. It does not scan the full matrices.

`PiRlcWideSampler.BatchMatrix.exact` proves exact interpretation of the
complete 14,501-row sampler. Its 34 Poseidon permutations use the specified
[4,i] entry and digest-advance schedule. The eight initial forms come directly
from the semantic plan. The 17 range blocks contain a certified 681-row
template and use checked source substitutions. This template is matrix data;
it does not add physical R1CS constraints, hints, or committed coordinates.
The 1,404 temporary helper columns remain unmapped.

The codec preserves the five selected block tags and their encoding. The
candidate adds checked-column projection and an ordinary-row template, plus
a small sparse table for the initial Poseidon state. The recursive codec has
private encode/decode helpers. The source-boundary check now accepts those
codec signatures; its regression tests still reject physical declarations,
non-private helpers and non-codec return types. The Format round-trip proofs
are audited as well as the row proofs.

These new forms still need Rust decoder support before selection. No selected
artifact has changed. The candidate's generic mapped-row evaluation is not a
proving-time improvement; its runtime and cache behavior must be checked at
consumer integration. The selected evaluator retains its current paths.

The actual range compiler passed its new active-source check. The focused
sampler cost run and the independent Python normalization returned the same
complete JSON cost record as `wide-sampler-matrix-cost.json`: 14,501 rows,
135,813 coordinates, and 1,794,350 normalized nonzeros. The complete product
program and whole-program composition are now proved as described below.

## Whole-program matrix correspondence and count

`Wide.MatrixProgram.fixedPoint_exact` connects every decoded row and all
13 meaningful ports to the candidate fixed-point plan. The fourteenth matrix
is zero by the existing plan convention. The theorem constructs source custody
from the canonical package; it requires no caller-supplied rows or sampler
success premise.

`Wide.ProductMatrix.interface_exact` connects all 969 products to the checked
three-bit challenges, the relocated inputs, and the original output/quotient
order. The six input maps retain their source-key ranges and strides. Their
three retained blocks move by the proved shared-region shift. The reference
product codec keeps its existing encoding; a direct sparse challenge has its
own wire case. The existing optimized numeric interpreter is proved equal
to that same decoder.

The full counter reads the emitted compact program and the already validated
95,161,747-byte source archive. It normalizes coefficients modulo the exact
Goldilocks prime. Field forms remain 41-coordinate atoms during linear
arithmetic; overlapping atoms are expanded before counting. One process pool
uses the available logical CPUs for independent blocks. It first reproduces
the full baseline vector, including every one of the 14 matrix totals.
Reused byte-equal blocks preserve their counts under the checked injective
column map and the Lean read-support proofs. The changed sampler and product
blocks are counted directly. The input archive hash matches the earlier
validated package record.

| Measure | Selected baseline | Wide candidate | Reduction |
| --- | ---: | ---: | ---: |
| Logical rows | 3,588,191 | 3,248,956 | 339,235 (9.45%) |
| Committed coordinates | 149,293,044 | 137,341,872 | 11,951,172 (8.01%) |
| Normalized matrix nonzeros | 2,857,409,270 | 2,607,606,765 | 249,802,505 (8.74%) |

The numeric nonzero count is an independent exact-arithmetic measurement,
not a Lean numeric theorem. The matrix-program correspondence, slot mapping,
row counts and coordinate counts are Lean theorems. No overall proving-time
or peak-memory improvement is claimed. Production remains unselected.

The complete vectors, block counts and input hashes are saved in
[wide-matrix-cost.json](wide-matrix-cost.json). Reproduce from the formal
project and then the repository root:

```sh
timeout --signal=KILL 1500 scripts/validate.sh build emitWideMatrixCost
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/emitWideMatrixCost /tmp/nightstream-wide-matrix-operands.json
```

```sh
timeout --signal=KILL 300 python3 -B tools/recursive-constraint-minimizer/experiments/wide_matrix_cost.py /tmp/nightstream-wide-matrix-operands.json formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json /tmp/nightstream-wide-full-matrix-cost.json
```

The counter completed within the 300-second test cap. It does not generate a
production package, fixture, or witness. The source/constraint distinction
follows the semantics-preserving compilation approach in
[CLAP](https://arxiv.org/abs/2405.12115). No compiler dependency was added.

## Validation at this checkpoint

The full production library and test gate passed, including
all 999 sampler, integration, security and codec audits, plus four retained-support regression
audits. Every audited declaration uses only the three allowed axioms. Static
boundary checks pass. The complete `NightstreamFPrime NightstreamFPrimeTests`
build covers witness construction, full-step soundness, matrix
correspondence, and the candidate security consumers. The commands were:

```sh
timeout --signal=KILL 1500 scripts/validate.sh static
timeout --signal=KILL 1500 scripts/validate.sh build NightstreamFPrime NightstreamFPrimeTests
timeout --signal=KILL 1500 scripts/validate.sh axioms
timeout --signal=KILL 1500 scripts/validate.sh identity
```

The identity check passed: the freshly computed canonical binding, structural
identifier, package identity and verifier-key pins all match the selected
fixture. This check writes only a temporary binding file. Selected packages
and fixtures were not regenerated.

These checks do not select the candidate. The 12 source-codec boundary tests
also pass. The full gate exposed missing new-block cases in `FreshRowsCheck`;
those cases now use the existing fail-closed sparse-row checker. The static
gate rejected private recursive codec helpers before the codec-signature rule
and its rejection tests were added.

The carrier proof uses an abstract assignment while splitting the optional
logical-column lookup. Splitting the concrete witness expression caused its
code to unfold during proof elaboration. The affected module build changed
from 94 seconds to 1.5 seconds after this proof-only change, as recorded by
`validate.sh build NightstreamFPrime.Export.Stage1.Wide.CarrierAssignment`.
No runtime witness constructor or circuit count changed for this repair.

The standalone Rust decoder passes both parity tests: 11 modular boundary
inputs and three 17-scalar transcript traces. The saved fixture is byte-for-byte
equal to a fresh Lean emission. Production Rust rho derivation still calls the
old sampler. There is no new hint kind, generic hint-interpreter change, crate
dependency, or selected production artifact.

The six controls in `wide_source_controls.py` return the expected SAT/UNSAT
results. They reject the former ring-major indexing for a reference-order
block, prove recovery of lane/cell/digit indices after a shift, and check the
separate PiDEC source offsets and their bounds. The obsolete permutation
control script has been removed.
These solver controls do not replace the universal Lean coordinate proofs.
The experiment uses the semantics-preserving compilation discipline described
in [CLAP](https://arxiv.org/abs/2405.12115), with the applicable
[cvc5 arithmetic theories](https://cvc5.github.io/tutorials/beginners/theories.html).
The 2026 [CLAP implementation report](https://www.nethermind.io/blog/clap-correctly-compiling-the-leanest-possible-circuits)
also separates constraint soundness from completeness of the constraint system
with its witness generator. We keep that same proof distinction here; this
integration does not add a compiler dependency. The [cvc5 field/integer discussion](https://github.com/cvc5/cvc5/discussions/11911)
is relevant to modular arithmetic experiments, but the checked integer bounds
and field conversions remain Lean obligations.

The source sampler allocates 55,403 private DSL values and has 32,623 DSL
rows, including the 918 temporary digit bindings. Structural lowering gives
58,939 R1CS rows and 26,316 R1CS scratch values. The original sampler uses
1,008,848 R1CS rows. These physical figures describe the reference witness
construction; the compact sampler remains 14,501 CCS rows and 135,813
committed coordinates.

The candidate physical prefix through the running transition has 27,716,409
rows and 27,859,538 source columns. These are source-layout counts, not the
committed width. The application and final public layout are not selected
from this candidate yet.

## Reproduce the focused checks

Run from `formal/nightstream-fprime`, through the required validation wrapper:

```sh
timeout --signal=KILL 1500 scripts/validate.sh build NightstreamFPrime NightstreamFPrimeTests measureWideSamplerCost checkWideSamplerHints
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/checkWideSamplerHints
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/measureWideSamplerCost > /tmp/nightstream-wide-cost-input.jsonl
```

From the repository root:

```sh
timeout --signal=KILL 300 python3 -B tools/recursive-constraint-minimizer/experiments/wide_sampler_matrix_cost.py /tmp/nightstream-wide-cost-input.jsonl
```

The controls execute the exact hint program on 11 boundary inputs, including
`0`, `N-1`, `N`, `N+1`, `p-1`, `p`, `p+1`, and the upper end of the draw domain.
All 681 rows and all 54 digits are checked for every case.

## Rust matrix reader and witness parity

The reader accepts all block types used by the emitted candidate: checked
coordinate maps, embedded ordinary-row templates, sparse Poseidon inputs,
and seven-field Phi81 blocks with directly centered challenges.

Checked maps are applied to operands once during decoding. Source-row indices
remain in their original space. The reader checks every stored sparse entry
before normalization, including zero coefficients and cancelling terms.
Retained operands must stay contiguous under the map. The check splits each
operand at every projection boundary, so an interior gap or overlap cannot
pass merely because the first and last columns are valid. All 104 retained
operands in the actual candidate meet this condition. The complete emitted
program loads with 41 blocks, 3,248,956 rows and 137,341,846 logical coordinates.
No per-entry map is added to the matrix traversal.

The saved Lean Phi81 template takes an uncentered digit and subtracts two.
For an already-centered wire form, the reader adds two only at that template
input. The offsets cancel; there is no new witness value or matrix row.
Tests compare every port at all 108 points for two sources against the retained
digit form. Mapping tests compare all ports with the original interpreter,
including nested maps, and reject missing, overlapping and non-contiguous
ranges. All 21 focused matrix tests pass.

```sh
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib matrix_program
tools/recursive-constraint-minimizer/experiments/check_wide_matrix_reader.sh /tmp/nightstream-wide-matrix-operands.json
```

The second command has the project-required 300-second timeout internally.
It tests the actual compact operands emitted by `emitWideMatrixCost`; it does
not replace complete matrix-entry conformance after production selection.

The unchanged Rust witness interpreter also matches the complete scalar hint
program on all 11 Lean boundary cases: all 2,021 private values and all 937
retained coordinates. The 1,404 helpers have no retained slot. This is a regular
unit test backed by the committed Lean-generated fixture
`crates/nightstream-fprime/tests/fixtures/pi-rlc-wide-witness-v1.json`, not a test
that requires unrecorded stdin. The interpreter and its hint kinds are unchanged.

This check first failed because the raw witness expressions produced JSON at
depth 303; the installed serde_json parser has a depth limit of 128.
`Export/WitnessEncoding.lean` now balances sum trees and proves preservation of
expression values, variable support, hint results, batch sizes, and complete
sequential batch execution. The emitted depth is 27. The boundary outputs and
retained map are unchanged. No compiler limit, Rust feature, dependency, or
constraint changed. This follows the explicit witness/constraint separation
in [CLAP](https://arxiv.org/abs/2405.12115); Lean supplies the local proofs.

Regenerate the parity fixture from `formal/nightstream-fprime`:

```sh
timeout --signal=KILL 1500 scripts/validate.sh build emitWideWitnessParity
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/emitWideWitnessParity ../../crates/nightstream-fprime/tests/fixtures/pi-rlc-wide-witness-v1.json
```

Run its regular native test from the repository root:

```sh
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib lean_wide_hints_match_native_execution
```

All 44 regular `nightstream-fprime` library tests pass; five tests are ignored
by default. The ignored actual-candidate operand check above also passes with
its recorded driver. `cargo fmt --all` passes. Native rho selection, whole-package witness transport,
and the production package switch are still open. The selected packages and
fixtures were not regenerated, and no Rust proving benchmark was run.

## Remaining coordinate budget for the research target

The actual candidate matrix operands contain 32,341 Poseidon2 permutations.
Their retained S-box blocks are disjoint after the checked coordinate map:
2,781,326 fields at 41 coordinates each, or 114,034,366 coordinates. This count
includes the three application permutations and the 34 wide-sampler
permutations. It is obtained from the complete `emitWideMatrixCost` program,
not from the older funnel's category estimates.

| Budget | Coordinates |
|---|---:|
| Complete candidate, including alignment | 137,341,872 |
| Current Poseidon2 S-box encoding | 114,034,366 |
| All remaining coordinates, including alignment | 23,307,506 |
| Research target | 92,179,782 |
| Reduction still needed | 45,162,090 |

Even removing every other coordinate would leave the current S-box allocation
21,854,584 above the target. Thus at least 19.16% of that allocation must go,
and the necessary reduction is larger when the remaining work is retained.
This is a budget obstruction for the current encoding, not an impossibility
proof for other sound Poseidon2 encodings. The additional 50% target is not
achieved, and no proving-time or peak-memory result is claimed.

### Further research: fewer matrix evaluations in the carried state

A source review on 2026-09-25 identified a different possible route. The
selected matrix program uses ordinary R1CS rows, linear pins, Poseidon2
`x^7 = y` rows, and single-product Phi81 checks. The Phi81 output form already
includes the prior output and the fixed evaluation-point multiple of the
quotient. It therefore has only one variable product, despite the generic
five-product interface that represents it.

These selected families appear compatible with four matrix ports and
`A*B + D^7 - C = 0`, with the existing canonical constant coordinate equal
to one. This is a candidate, not a Lean-proved replacement. The generic
five-product, centered-unit, and borrow-row interfaces are outside this
claim. Every emitted selected row still needs a structural classification
and an equivalence proof; matrix entry counts need their own measurement.

The possible coordinate benefit comes from a smaller recursive state.
Reducing `Eval_A` from 14 matrices to four would remove 1,080 words from each
running claim. Applied only to the two state hashes and the PiCCS output
binding, the current schedules predict 13,230 fewer Poseidon2 permutations,
or 46,648,980 fewer S-box coordinates. Subtracting that estimate alone from
the current total gives 90,692,892 coordinates. This is **not** a complete
candidate count: the recursive layout and witness budget must be rebuilt
and proved before counting any saving.

The [HyperNova author paper](https://www.andrew.cmu.edu/user/bparno/papers/hypernova.pdf),
Theorems 1 and 3, explicitly makes prover work and recursive hash cost depend
on the number of matrices. It supports studying this cost source; it does
not prove the proposed local encoding. The change would also change claims,
state serialization, transcript lengths, the polynomial degree and package
binding. Existing 14-matrix evaluation claims cannot in general be converted
by a fixed linear map because the proposed routing depends on the row family.
This requires a concrete reviewed protocol/specification change, structural
Lean proofs, cvc5 controls and updated security accounting before selection.
The approved profile and Poseidon2 must remain fixed.

The current format also reserves a final zero matrix. Keeping that convention
would use five matrices: four useful ports and one zero port. With nine fewer
`Eval_A` families, the Poseidon-only estimate is 95,357,790 coordinates.
The existing retained-block owners give these further disjoint reductions:

| Allocation | Formula | Coordinates saved |
|---|---|---:|
| Two state preimages | `2 * 16 * 9 * 108 * 41` | 1,275,264 |
| Carried child fields | `16 * 9 * 108 * 41` | 637,632 |
| PiCCS proof outputs | `17 * 9 * 108 * 41` | 677,484 |
| PiRLC outputs and quotients | `2 * 17 * 9 * 108 * 41` | 1,354,968 |
| Total additional reduction | | 3,945,348 |

The owners are `Lifecycle/XOut.lean`, `Layout/Stage1/Wide/PiDECInputs.lean`,
`Layout/Stage1/PiCCSInputs.lean`, and `Wide/PiRLCGeometry.lean`. PiDEC's proof
view reuses the carried-child block; its parent views reuse PiRLC outputs;
PiRLC inputs reuse PiCCS proof values. These aliases are not counted again.

This gives a conditional carrier budget of **91,412,442**, or 767,340 below
the research target. It keeps 28 rounds, all five evaluation slots, the
current field encoding, and the current budget for every uncounted block.
The logical width would be 91,412,416 plus 26 alignment coordinates. This
establishes a concrete budget worth testing, not feasibility of the complete
new circuit: row classification, constructive witness maps, the new recursive
layout, matrix costs, binding and security still need proofs and checks.

## Physical package and matrix-source cutover

The wide physical prefix and complete application suffix now emit as one
physical source archive. The exporter rejects unmapped old source columns
and rows. It includes the new range gadget and its exact existing-hint
program, the 34 new sampler permutations, the temporary digit bridge, the
ring families, PiDEC, running transition, application and next-preimage rows.
It does not emit First-54 templates or the old sampler witness batches.

The physical archive has 27,724,114 R1CS rows and 27,867,239 source columns.
These are witness/source dimensions, not committed CCS dimensions. Its
companion matrix program has 3,248,956 logical rows and 137,341,846 logical
coordinates; the carrier theorem adds 26 alignment coordinates. A complete
Rust matrix traversal over this new source archive reproduces all 14
nonzero counts from the candidate ledger, totalling 2,607,606,765.

Lean proves the new range-row lowering, expression and hint evaluation under
the source permutation, affine source-projection composition, and unique
ownership of the inverse physical ranges. The existing whole-plan exactness
theorem still uses the reference source archive. Connecting the new emitted
archive to that theorem and the complete committed-witness transport remains
part of the production cutover. The native sampler and selected identity pins
are unchanged.

The first physical export rebuilt common data through the reference path.
That path also traversed the full running-transition expressions to collect
one inverse hint. The exporter now shares each prepared arithmetic packet,
uses the existing parallel preparation, and calls the proved direct
running-transition hint exporter. The completed physical-only export took
10.3 seconds; the physical archive plus matrix program took 12.7 seconds.
The complete Rust source, ownership and matrix checks took 2.51 seconds after
compilation. These are experiment costs, not proving-performance claims.

Reproduction (the outputs remain temporary until the full switch is ready):

```sh
# From formal/nightstream-fprime:
timeout --signal=KILL 1500 scripts/validate.sh build emitWidePhysicalPackage
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/emitWidePhysicalPackage /tmp/nightstream-wide-physical-package.json
# From the repository root; the driver applies the 300-second test cap:
tools/recursive-constraint-minimizer/experiments/check_wide_physical_package.sh /tmp/nightstream-wide-physical-package.json
```

Validation for this batch: the full Lean library and test build passes
(4,427 jobs), including the eight new theorem audits. Static and axiom gates
pass. All 44 regular Rust library tests pass; the new ignored conformance
test passes through its committed driver. The selected-identity check also
passes: the canonical binding and all existing identity pins match. Production
selection is unchanged.

The driver reads the physical archive and its `.matrix.json` companion. It
uses the production package validator and checks every matrix row against
that archive, including source bounds, witness coverage, ownership, column
bounds, sparse normalization and the complete nonzero vector. No proving
benchmark runs in this check.

## Sealed candidate and complete native base witness

The candidate emitter now writes a schema-6 sealed package with the new
physical archive, matrix program, application metadata and schema-4 retained
assignment transport. Its 76 retained blocks keep the common values, sampler
S-box outputs, checked range values, ring outputs and quotient coefficients.
The quotient recipe reads the checked three-bit digits directly. It does not
read First-54 products or the 918 temporary digit words. The existing hint
interpreter and dependency list are unchanged.

The Lean base fixture uses the exact wide transcript. `batch_challenges`
proves that its cached array returns the specified challenge at every source
and lane. The array is stored once in the batch: returning a function that
built the array on each read made the first fixture attempts too slow. The
corrected full candidate export with its base fixture completed in 57.6 seconds.
This is a development-loop measurement, not a proving-performance result.

The native check uses the real sealed decoder, physical witness engine,
retained-value transport and matrix reader. It verifies every one of the
3,248,956 CCS rows and all 270 public coordinates against the Lean fixture.
The logical assignment has 137,341,846 coordinates; its committed carrier has
137,341,872 after 26 alignment coordinates. Replacing all 23,868 temporary
helpers and 918 digit words leaves the assignment byte-identical. Flipping a
checked sampler quotient bit produces an encodable assignment that the CCS
rows reject. The fresh-export check passed in 12.12 seconds after compilation.

The source relocation has additional structural Lean proofs. Successful
expression and hint relocation preserves evaluation; successful affine-run
relocation preserves every source index and its order. The common blocks
select the reference block's sources after relocation, including the shortened
PiCCS Poseidon prefix. The three retained range blocks select exactly the
physical cells used by the proved hint program. These statements apply to
arbitrary values, not just the fixture.

The native check deliberately uses a supplied fixture context and a test
identity. It does not establish the selected production identity, a recursive
fold, universal Rust semantics, or the complete emitted-transport equality to
`SourceAssignment.assignment`. That final equality and source-archive custody
are still required. The already proved whole-plan matrix theorem still uses
the reference archive; this batch does not claim otherwise.

Reproduction, from the Lean project and then the repository root:

```sh
timeout --signal=KILL 1500 scripts/validate.sh build emitWidePhysicalPackage
# The four context words are artifacts/nightstream-fprime-stage1-base-step-fixture-v1.json[1].
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/emitWidePhysicalPackage /tmp/nightstream-wide-physical-package.json <context0> <context1> <context2> <context3>
tools/recursive-constraint-minimizer/experiments/check_wide_assignment.sh /tmp/nightstream-wide-physical-package.json
```

The driver supplies the `.sealed.json` and `.base.json` companions and applies
the required 300-second cap. The new exporter outputs stay in `/tmp`; selected
packages and official fixtures are not regenerated in this batch. The checked
implementation is this checkpoint, based on `965351823`.

The full library, tests and emitter build, `static`, `axioms`, and `identity`
all pass. The 16 new exported theorems use only the allowed standard axioms.
All 44 regular Rust library tests pass; seven large or driven tests remain
ignored by default, including the new test run through its committed driver.
The original selected base fixture is byte-identical to a fresh Lean emission,
and the existing structural, package and verifier-key identity pins match.
The source-order cvc5 controls still produce the expected SAT counterexamples
and UNSAT results. No proving benchmark ran.

One export iteration took 75.3 seconds because the new `commonLimit` accessor
constructed a complete source-run plan again just to read its slot count.
Reading that count from the block geometry removed the duplicate work; the
next run took 57.6 seconds and all four output files were byte-identical.

## Complete archive and emitted witness proofs

The candidate now has the two full package connections that were open at
`daf5575c4`:

- `PackageAuthority.matrix_exact` derives the complete matrix correspondence
  from `AuthorityStream.prepare = .ok parts`. The actual stored physical rows
  supply source custody; there is no caller-supplied row-equality certificate.
- `AssignmentTransportCorrectness.canonical_execute_eq_assignment` proves
  that the exact emitted schema-4 plan returns the direct logical assignment.
  `canonical_carrier_eq` includes all padding coordinates.
- `PackageCompleteness.complete` applies that execution result to the same
  constructive completion used by the full candidate relation. It proves
  successful transport, row satisfaction, strict norm below 2, the exact public
  output, and unchanged application advice. The semantic caller supplies no
  physical-row, sampler-success, or witness-transport premise.

The common source map now rejects removed columns before emission. Its source
bounds are checked on compact affine-run endpoints. The proofs then follow
the block structure, without expanding retained coordinates. Exact range
readback is preserved from the canonical input children through the scalar,
batch, PiRLC phase and complete physical witness. The product output and
quotient values follow the proved direct ring construction.

`SetupBinding` derives authority from the same sealed children as the emitter.
The application component hashes the emitted relocated application plan.
The setup keeps the approved seed and uses 22 rows and 2,543,368 message
columns. Both the NIFS key component and the outer context bind the wide
sampler schedule. The native structural identity stream has a proof that it
is the canonical Poseidon2 hash of the sealed envelope.

The emitter shares `AuthorityStream.ofChildren` with the proved constructor.
It also emits a binding fixture, computing the structural stream once. With
no fixture-context argument, it derives the base fixture context from this
binding. The baseline identity serializers and selected pins are unchanged
in this proof checkpoint.

The focused highest-target check passed:

```text
timeout --signal=KILL 1500 scripts/validate.sh build NightstreamFPrime.Export.Stage1.Wide.PackageCompleteness
```

Before adding the binding sidecar, a fresh export of the four package and
base-fixture files was byte-identical to the `daf5575c4` checkpoint. The full
Rust assignment check passed all 3,248,956 rows, temporary-value independence,
and checked-bit mutation rejection. Export took 60.6 seconds; the Rust test
took 12.6 seconds. These are conformance runs, not proving benchmarks.

This batch does not change the candidate's 3,248,956 rows, 137,341,872 committed
coordinates, or 2,607,606,765 normalized matrix entries. Production selection
remains open.

Checkpoint validation, based on `daf5575c4`:

```text
timeout --signal=KILL 1500 scripts/validate.sh static
timeout --signal=KILL 1500 scripts/validate.sh build NightstreamFPrime NightstreamFPrimeTests emitWidePhysicalPackage
timeout --signal=KILL 1500 scripts/validate.sh axioms
timeout --signal=KILL 1500 scripts/validate.sh identity
timeout --signal=KILL 1500 scripts/validate.sh lean-executable .lake/build/bin/emitWidePhysicalPackage /tmp/nightstream-wide-proved-package.json
tools/recursive-constraint-minimizer/experiments/check_wide_assignment.sh /tmp/nightstream-wide-proved-package.json
```

All checks passed. The full gate covered 8,360 jobs in 102 seconds; all 191
added axiom entries pass. The selected baseline binding and three identity
pins are unchanged. The fresh exporter, including the binding sidecar and
base fixture with its derived context, took 92.9 seconds. The physical,
matrix and sealed package bytes are unchanged. The full Rust assignment and
mutation check on this new-context fixture passed in 12.4 seconds. This check
does not establish native parity for the new binding serializer; that remains
part of the production switch.

## Native selection and conformance in progress

The local production entrypoint and Rust sampler now select the proved wide
package. The saved proof checkpoint is `a0f4b5b41`; this native integration
batch is committed in `c65a8605a`. The fixed profile and approved indexed setup seed
are unchanged. The selected key uses 2,543,368 ring columns and 22 rows.

| Measure | Previous selected package | Wide selection |
|---|---:|---:|
| Logical rows | 3,588,191 | 3,248,956 |
| Committed coordinates | 149,293,044 | 137,341,872 |
| Normalized matrix entries | 2,857,409,270 | 2,607,606,765 |

The complete independent Rust comparison checked every entry of all 14
matrices in 27.26 seconds. The per-matrix counts are
`[16904205, 3144304, 264072386, 25752159, 800725610, 1496903449, 0, 104652, 0, 0, 0, 0, 0, 0]`.
Changed block order, a changed column, and a changed nonzero coefficient all
decode independently and then fail the selected structural identity.

The following current-package checks pass:

- Eight package-loader checks, three independent binding checks, and complete
  physical-matrix comparison against the separate Lean expansion.
- Direct CCS assignment equals full physical assignment at every logical
  coordinate; both paths reject the checked invalid caller inputs.
- Native wide sampling matches the Lean boundary and transcript fixtures,
  including both existing Rust engines and the production sampler entrypoint.
- Setup vectors and the sparse commitment match Lean, including the final
  coordinate of the smaller carrier.
- The independent assignment interpreter checks all 3,248,956 rows and the
  retained-value mutations. The complete check passes in 62.93 seconds.
- All 44 regular F′ library tests pass. Seven large driver checks remain
  ignored by default. The loader now rejects the retired transport schema 3;
  the selected schema 4 is the only witness-transport decoder.
- Native assembly passes for the selected hash-chain application and a second
  application. The same-seed setup-prefix test derives its widths from the
  selected Lean manifest and passes.

Logs are `/tmp/nightstream-wide-selected-{loader,binding,physical-matrices,
direct-assignment,logical-matrices,matrix-mutations,sampler-engines,setup,
primitive}.log`. Each native test used the project 300-second cap. These
check durations are validation evidence, not proving benchmarks.

The first direct-assignment check exposed an application fixture that still
used the old sampler. Its generator now uses the wide sampler and derives
terminal metadata from the wide structural plan. The corrected fixture
passes the direct/full witness comparison. The independent matrix decoder
now supports the emitted checked column maps, local affine row templates,
direct challenge forms, and sparse Poseidon inputs; expected arithmetic
still runs in the independent reference interpreter.

Only the unused old sampler parity emitter, its wrapper, and its artifact
have been removed. Baseline layout and witness proofs still use old sampler
owner modules for the common regions they preserve. Their presence does not
select old sampler rows or change the new committed width.

Recipe rejection, fresh recursive fixtures, final consumer checks, the full
Lean gate and identity check remain open. No proving-time or peak-memory
improvement is claimed.

Fresh native fixture construction uses
`/tmp/nightstream-wide-selection-FAJ41z`. The normal optimized C prover and
verifier pass on the selected base witness. Source loading ends at 42.514 s,
proof construction ends at 118.459 s, and verification ends at 118.476 s.
The full invocation takes 118.76 s and has peak RSS 5,522,316 KiB. The proof
phase is 75.945 s; the RSS includes preparation. These are absolute stage
measurements, not a before/after performance comparison.

R replays the saved C proof against the original sources and constructs the
new parent. Its invocation passes in 46.65 s with peak RSS 3,596,020 KiB.
The separate parent check recomputes the commitment and canonical split;
it passes in 108.02 s with peak RSS 3,596,792 KiB. Children 0 through 5 are
active. The six openings pass in one 201.05 s invocation with peak RSS 5,538,624 KiB.
D assembly, normal NIFS verification and all 43 mutation controls pass in
114.92 s with peak RSS 3,596,024 KiB. The independent Lean C/R/D result and
recursive caller fixture are freshly emitted from the C input and child
claims. Native comparison passes on every result field and the complete
945,983-byte proof; all 55 PiDEC mutation cases reject. The new proof and
independent fixtures are installed in their published paths.

## Production-switch obligations at the proof checkpoint

- Select the candidate authority, archive and transport together. The candidate
  now has full matrix correspondence, constructive emitted-witness transport,
  and `SetupBinding.step_or_collision` for the verifier-owned context. The
  production selector and native binding must use those same children.
- Use the wide-key security consumers when selecting the production package.
  Their key response, adaptive query accounting, and explicit Fiat–Shamir
  boundary are recorded above; the complete-transcript link to the block
  oracle is not proved, and the wide-versus-uniform challenge difference
  stays inside `deltaFS`. Numerical cryptographic
  advantages and any external adversary translation remain external, as for
  the selected baseline.
- Switch native Rust rho derivation and assignment transport to the checked
  decoder after the full layout and security gates pass, then select and
  regenerate the package, identities, and fixtures. The owner confirmed approval
  for this protocol change on 2026-09-24; no additional approval is pending.
- Run the required conformance checks and remove the old sampler dependencies.

At `a0f4b5b41`, the selected baseline remained 3,588,191 rows, 149,293,044
committed coordinates, and 2,857,409,270 normalized matrix nonzeros. The
native-selection section above records the subsequent working-tree changes.

## Lean 4.32.2 base and review fixes

The branch now merges `nico/f-prime-constraints-cuda-formal` (Lean 4.32.2 and
its Mathlib). Lean conflicts kept this branch's layouts; the upgrade repairs
follow the base commit's patterns (`using!`, explicit `unfold`/`rw` for large
layout constants, removed no-progress `dsimp`). Rust conflicts combine the
base's streamed row visitors and application records with the wide blocks.

Review fixes in the same merge:

- `Wide.Emitter` writes the parts of the proved pure `AuthorityStream.prepare`.
  A fresh selected emission is byte-identical to the committed package, and the
  identity pins match.
- `Wide.OpeningBinding` adds the verifier-side links: rows read from the sealed
  matrix program imply the step (`step_or_collision_of_matrix`), a fresh CCS
  opening gives the rows and public input, and an accepted recursive terminal
  gives the step or the named state-hash collision
  (`terminal_implies_stepOrCollision`).
- The FS boundary note, assurance surface and security docstring no longer
  claim a complete-transcript link or a consumed `q*δ` term. The owner's
  2026-09-25 confirmation of the Fiat–Shamir extension is recorded with a hash.
- The Rust wide transport pins the exact Phi81 family order and has regular
  unit tests for bit, digit, profile and quotient handling. The detached
  application regression now finds the mapped application block and rejects
  the detached output at row 3,248,943. The pilot parity fixture is
  regenerated for the wide verifier context.
- The selected key permits ordinary applications with at most 262 private
  words; the assembly node test derives this capacity from the manifest.
  Widening the key is an owner decision.

The wide-key security port of the baseline NIFS and HyperNova history chain
is recorded in the next section.

## Wide-key security port

The NIFS extraction and HyperNova history chain now proves its statements
for the wide package. Nothing in the chain is pinned to the baseline
Poseidon2 package.

- `SecurityInstance` holds the width, relation and Ajtai key that one
  extraction statement is about. The NIFS modules (`PiCCSStoredWitnessCheck`
  through `NifsProviderLaw`) take it as their first argument.
  `PiCCSInputCheck.runningAt` and `freshAt` read the checker input at any
  width. The input arrays fix every size, so no proof compares two concrete
  widths.
- `Wide.Target` fixes the application program, compiled sampler plan, carrier
  fit, Ajtai key and verifier context digest. `Wide.TerminalSecurity` proves,
  for each target, the preimage match, the matching step, the wide-key NIFS
  output, the PiDEC parent and both predecessor cases, or the named
  state-hash collision. The history modules (`HyperNovaHistory` through
  `HyperNovaSourceWork`) take a target.
- `WideFiatShamir` stays the owner of the approved wide real event and
  model. `FiatShamirTransfer` keeps only the key-independent composition,
  with the transfer inequality as a premise. The retired `ProductionKey`
  event and model were deleted; the approved wide text is unchanged.
- `ActualContextSecurity`, `ActualTerminalSecurity` and `HyperNovaPredecessor`
  were deleted. The baseline replay modules keep the baseline payload type
  from `PerApplicationTerminal`.
- The four evidence targets (terminal assignment, terminal parent, linear
  history security and terminal false acceptance) now quantify over a wide
  target. Their meaning changed, so their target-meaning reviews must be
  done again.

`Wide.selected compiled parts` is the selected package's target: the
Poseidon2 hash-chain application, its compiled sampler plan and fit, the
production Ajtai key and the context digest of the prepared descriptor. The
statements stay general over the target; no numerical security level is
claimed.

## Key capacity and generic hash-chain vector

On 2026-09-25 the owner approved 22 × 4,708,530 ring columns, the approved
public-seed MSIS matrix, as the supported key capacity. Each package still
binds its exact key prefix, so the selected package keeps 2,543,368 columns
and all its identities.

- `neo_ajtai::nightstream_fprime_setup::MAX_MESSAGE_COLUMNS` is the upper
  bound for the prefix commitment, PiDEC, fresh-carrier and key-prefix
  checks. Ordinary applications may use 2,851,939 witness and local words,
  up from 262. The largest fixed envelope has 26,540,836 nodes, below the
  loader's 32,045,229-node bound.
- `poseidon2_hash_chain(links)` generalizes the hash chain;
  `poseidon2_hash_chain_v1` is its one-link case and stays byte-identical.
- The generic vector is `poseidon2_hash_chain(2)`: 15,392 private words on the
  ordinary route with its own wider key prefix. Its structural identifier,
  package identity and verification-key digest are pinned. Its base proof
  verifies; a changed endpoint, initial state or counter, a changed witness
  with the old or a recomputed commitment, and a changed commitment are each
  rejected for their exact reason (219 s of test time).
