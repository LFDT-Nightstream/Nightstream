# Wide sampler integration checkpoint

Base: `be5b7fcdc`. Worktree branch: `nico/pirlc-wide-sampler-integration`.
Production selection and generated packages remain unchanged at this checkpoint.

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
cells in the reference order. This does not yet construct a complete accepting Stage 1 assignment.

`HashChainCounts.lean` proves these counts of the **assembled, unselected
candidate**, using the current application:

| Measure | Selected baseline | Candidate plan |
|---|---:|---:|
| Logical CCS rows | 3,588,191 | 3,248,956 |
| Logical coordinates | 149,292,999 | 137,341,846 |
| Alignment coordinates | 45 | 26 |
| Committed coordinates | 149,293,044 | 137,341,872 |
| Normalized matrix nonzeros | 2,857,409,270 | Not measured for the complete plan |

The candidate dimensions are derived from the assembled row plan and retained
intervals. They are not counts from a regenerated production artifact, and are
not an achieved production or overall performance reduction. Whole-package
source preservation and matrix-program correspondence remain open.

The old sampler's ordinary retained blocks contain 2,230,400 + 6,758,112 =
8,988,512 coordinates. The external funnel's 8,998,512 entry is 10,000 too high;
the selected package's total has not changed.

The candidate NIFS key in `Lifecycle/PiRLC/Wide/Key.lean` uses the exact new
response map. Lean connects that response to accepted candidate phase values
and proves that the profile and challenge-set cardinality remain unchanged.
This closes a deterministic verifier connection, not the production FS model.
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
its own identity. Physical source-environment transport is still open; this
support result concerns the CCS matrices, not the old PiDEC witness builder.

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
preimages are proved equal to the baseline preimages. This definition provides
the target for semantic-step construction and whole-plan soundness; it is not
yet a proof of either result.

The accepted physical source still needs construction from that semantic step.
Whole-plan soundness for the new key, emitted-matrix correspondence, the full
normalized nonzero count, and the selected security and Rust connections remain
open. No selected circuit count changes in this batch.

## Validation at this checkpoint

The full production library and test gate passed, including
all 845 sampler and integration audits, plus four retained-support regression
audits. Every audited declaration uses only the three allowed axioms. Static
boundary checks pass. These checks do not select the candidate.

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

## Still required for the production switch

- Complete the candidate witness construction from a semantic HyperNova step.
  All compact phase rows now accept one assignment built from the accepted wide
  physical prefix. Its carrier norm and exact public-digest transport are proved.
  Construct the physical source from the candidate semantic relation and connect
  its state fields and advice to that step. Prove whole-plan soundness with the
  new key. Then prove that the emitted matrix program denotes the same plan and
  measure all normalized matrix entries. Acceptance from an existing physical
  witness does not by itself close these obligations.
- Apply the security model to the selected production transcript and record the
  applicable Fiat–Shamir assumption and query accounting. The adaptive block
  budget `q` in `q * distance` must not be identified with a fold count or the
  extractor index count without a proved experiment connection. No fixed
  comparison with the extraction loss is claimed here.
- Switch native Rust rho derivation and assignment transport to the checked
  decoder after the full layout and security gates pass, then select and
  regenerate the package, identities, and fixtures. The owner confirmed approval
  for this protocol change on 2026-09-24; no additional approval is pending.
- Run the required conformance checks and remove the old sampler dependencies.

The selected baseline remains 3,588,191 rows, 149,293,044 committed coordinates,
and 2,857,409,270 normalized matrix nonzeros. No Rust benchmark has been run.
