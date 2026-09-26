# Remaining witness budget

Research checkpoint: `8b7c07d8`, checked on 2026-09-22. The new owner target is
**92,179,782 committed coordinates**, exactly half of 184,359,564 and exactly
1,707,033 Phi81 blocks. No candidate that meets this target is established.
Keep Goldilocks, Poseidon2, degree strictly below nine, `b = 2`, `k_rho = 16`,
`B = 65536`, the existing relation, and the existing security assumptions.

The selected Lean shared-flag layout and Poseidon pin removal now give:

| Measure | Quotient checkpoint | Selected layout |
|---|---:|---:|
| Committed coordinates | 184,359,564 | 172,217,934 |
| Logical rows | 4,703,127 | 4,147,335 |
| Matrix nonzeros | 3,001,571,645 | 2,968,490,185 |

The complete Lean soundness, constructive witness maps, norm bound, and
matrix proofs passed. Independent Rust comparison checked every logical
matrix row and confirmed the nonzero count. Selected Rust fixture and consumer checks
passed. [Current count evidence](running-transition-metrics.json)
is separate from the preserved checkpoint below.

## Exact costs and the main tradeoff

The quotient change saves **27.1339% of committed coordinates**, but increases
logical matrix nonzeros by **28.5017%**. It does not establish a runtime or
memory improvement. [The saved matrix evidence](checkpoint-metrics.json)
binds the baseline to the selected artifact, including its SHA-256 value.

| Measure | Baseline `469d12e2` | Quotient checkpoint |
|---|---:|---:|
| Logical coordinates | 253,011,231 | 184,359,519 |
| Alignment coordinates | 45 | 45 |
| Committed coordinates | 253,011,276 | 184,359,564 |
| Logical rows | 6,377,559 | 4,703,127 |
| Physical R1CS rows | 29,225,729 | 29,225,729 |
| Logical matrix nonzeros | 2,335,822,475 | 3,001,571,645 |
| Fixed domain | `2^28` | `2^28` |

Logical rows fall by 26.2551%. Physical A/B/C nonzeros are
`[93701820, 39358148, 28868018]`. Coordinates, rows, sparse-matrix work, and
execution cost are separate measures, as required by the CCS cost model in
[HyperNova](https://eprint.iacr.org/2023/573). Dense quotient evaluation forms
make this distinction material here.

The quotient checkpoint has the following exact coordinate partition. Counts use
retained allocation owners; shared source views are counted once.

| Allocation | Coordinates | Source formula |
|---|---:|---|
| Prior-hash S-box outputs | 43,546,100 | `12350 * 86 * 41` |
| Output-hash S-box outputs | 43,546,100 | `12350 * 86 * 41` |
| PiCCS and sampler S-box outputs | 27,351,182 | `(7604 + 153) * 86 * 41` |
| **All retained S-box outputs** | **114,443,382** | `32457 * 86 * 41` |
| Public prefix | 270 | `ProductionAssignment.publicWidth` |
| Product quotient slots | 2,145,366 | `52326 * 41` |
| Product output slots | 2,145,366 | `52326 * 41` |
| First54 retained blocks | 2,558,976 | `1088 + 44608 + 59840 + 2408832 + 44608` |
| Two pilot preimages | 4,050,226 | `2 * 49393 * 41` |
| Running-transition allocations | 14,160,826 | `RunningTransitionRetainedBlocks.retainedCoordinateCount` |
| PiCCS ordinary allocations | 34,765,909 | `PiCCSOrdinaryRetainedBlocks.retainedCoordinateCount` |
| Pilot ordinary allocations | 43,296 | `PilotOrdinaryRetainedBlocks.retainedCoordinateCount` |
| PiDEC ordinary allocations | 741,690 | `(270 + 17820) * 41` |
| Sampler ordinary allocations | 8,988,512 | `17 * 8 * 4 * (100 + 303) * 41` |
| Application witness and local allocations | 315,700 | `(4 + 7696) * 41` |
| **All other logical coordinates** | **69,916,137** | Sum of the eleven rows above |
| Current alignment | 45 | `184359564 - 184359519` |

The count owners are [PoseidonRetainedBlock](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PoseidonRetainedBlock.lean),
[PiRLCRetainedGeometry](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiRLCRetainedGeometry.lean),
and the named phase allocation modules. The final allocation sequence is
`174313617 + 741690 = 175055307` after PiDEC,
`175055307 + 8988512 = 184043819` after sampler ordinary rows, and
`184043819 + 315700 = 184359519` after the application. The sampler portion
contains 2,230,400 logical and 6,758,112 fresh coordinates; PiDEC contains
270 logical and 17,820 fresh field slots. These follow
[PiDECRetainedBlocks](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiDECRetainedBlocks.lean),
[PiRLCSamplerOrdinaryRetainedGeometry](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiRLCSamplerOrdinaryRetainedGeometry.lean),
and [ApplicationRetainedGeometry](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ApplicationRetainedGeometry.lean).
The product schedule has
`17 * (22 + 5 + 2 + 14*2) = 969` base-field ring products and
`969 * 54 = 52326` coefficient lanes. The two extension-valued families
account for their two components in this formula.

S-box outputs alone are 62.0762% of the carrier and exceed the target.
If the other logical blocks stay fixed, at most 22,263,645 coordinates remain
for S-box outputs before alignment. Reserving the present 45 padding
coordinates gives 22,263,600: an 80.5462% reduction of that block. Padding must
be recomputed for any new layout. Even deleting both pilot S-box blocks
entirely would leave 97,267,364 coordinates, still above the target. These
are budget tests, not valid deletion proposals.

The selected [shared-flag layout](running-transition-next-candidate.md) closes
the full retained-assignment transport. It removes 12,141,616 logical
coordinates from the transition allocation, leaving 2,019,210 there and
57,774,521 in all non-S-box allocations. The selected carrier has 31 padding
coordinates. Its gap to the research target is **80,038,152 coordinates**.
If that whole gap came from the unchanged S-box block, about **69.9369%**
of that block would have to be removed. Padding must be recomputed for a new
layout. Poseidon output-pin removal saves rows, with no S-box-coordinate
saving. The original checkpoint remains preserved in Git.

## What the local experiments establish

[The cvc5 record](poseidon_reduction.json) has seven passing controls with
cvc5 1.3.4. Prime-field queries explicitly use `--ff-solver=gb`; the encoding
query uses exact integers with an unrestricted modulus quotient. There are
no bitvector overflow assumptions. The ten research theorems in
[ConstraintReductionResearch.lean](../../../formal/nightstream-fprime/tests/ConstraintReductionResearch.lean)
passed their focused Lean check and axiom audit in 3 seconds.

**Independent field encoding.** Forty signed unit coordinates have at most
`3^40 = 12157665459056928801` encodings, fewer than
`p = 18446744069414584321`. Thus no decoder from at most 40 ternary coordinates
can cover all Goldilocks values, regardless of its weights. For the current
radix-three decoder, the exact missing value is `6078832729528464401`.
It is the seventh power of `3194645001229403778`. Its 41-trit witness is
forty `-1` digits followed by `1`. Lean proves the field recomposition bound
and this missing S-box output. This does not assert that this particular
value occurs at each slot of every complete valid protocol execution.

Joint packing of arbitrary independent fields also cannot explain a factor
of two: encoding `k` arbitrary fields requires `3^m >= p^k`, hence
`m >= ceil(k * log_3(p))`, with `log_3(p)` approximately 40.37950423.
The asymptotic saving over 41 coordinates is at most about 1.5134% in that
class. A correlated trace encoding is outside this counting argument.

**Two adjacent scalar S-boxes.** Eliminating `z` from `z = x^7; y = z^7`
gives `y = x^49`. No nonzero bivariate polynomial of total degree at most
eight vanishes on this graph over Goldilocks. The 45 monomials map to
distinct exponents `i + 49*j <= 392`; universal vanishing forces every
coefficient to zero. Lean proves this root-bound and coefficient argument.
The cvc5 coefficient query is `unsat`; `y - x^49` is a positive control at
degree 49. Replacing the pair by `y = x^7`, or dropping the first link,
has explicit counterexamples. Degree 49 exceeds the fixed profile.
This excludes only the scalar bivariate class with no additional witnesses,
not full Poseidon rounds, affine mixing, or multivariate trace encodings.

The earlier Phi81 controls remain separate: 108 nodes reject every nonzero
residual of degree at most 107 (`unsat`, 98.453 s); 107 nodes admit a false
product (`sat`, 2.086 s), with exact replay. These solver results are search
evidence. Lean proofs remain the authority.

## Primary-source candidates and their limits

- **Affine substitution and S-box flattening are already used.**
  [CLAP, section VI](https://arxiv.org/html/2405.12115v2#S6) removes affine
  intermediates and common expressions, and flattens gates up to a degree
  bound. [Plonky3's Poseidon2 AIR](https://github.com/Plonky3/Plonky3/blob/main/poseidon2-air/src/air.rs)
  offers degree-seven S-boxes without an extra S-box register, or degree-three
  constraints with a cube register. The local retained template already has
  one field slot per S-box output and no retained linear trace values.
  Neither source gives another factor-of-two saving for this layout.

- **Linear skip can address affine evaluation or matrix density.**
  [Ambrona et al., section 4.2](https://eprint.iacr.org/2022/462.pdf) compose
  partial-round linear layers while retaining the nonlinear values. Our
  direct template already removes those linear witness columns. A different
  affine evaluation order could still reduce arithmetic or nonzeros, but
  needs exact form equality and a measured cost change. The paper's
  Turbo-PlonK constraint counts do not transfer directly to CCS coordinates.

- **Exact sharing across invocations remains a concrete search class.**
  CLAP's common-expression pass suggests checking equal complete permutation
  inputs or equal sponge prefixes across the actual schedules. A shared
  witness needs a proved input equality and a deterministic reconstruction
  map. Equal message words or a digest match alone do not establish equal
  sponge states. The code already shares pilot preimages and PiDEC parent
  views. No additional large repeated region has been established; savings
  from one whole pilot chain would still be only 43,546,100 coordinates.

- **Use dependencies beyond a scalar pair, if they exist.** A joint encoding
  of correlated states, or elimination involving retained adjacent-state
  coordinates, is not excluded by the local tests. It must yield equations
  of degree below nine, signed unit coordinates, and efficient witness maps
  in both directions. No such construction or size estimate is established.
  A smaller concrete opportunity is the known zero final quotient
  coefficient: 53 quotient slots would permit 107 exact evaluation checks,
  saving at most `969 * 41 = 39729` logical coordinates and 969 rows before
  alignment. It cannot address the half-coordinate target by itself.

- **GKR and lookup arguments require a different security composition.**
  [gnark exposes GKR Poseidon2](https://pkg.go.dev/github.com/consensys/gnark/std/hash/poseidon2/gkr-poseidon2),
  and [LogUp-GKR](https://eprint.iacr.org/2023/1284) reduces committed lookup
  columns through a separate argument. These are potential ways to avoid
  committing a complete trace, but they do not provide deterministic
  equivalence to the current CCS. Transcript binding, error accumulation,
  efficient extraction, and the complete recursive verifier cost would all
  need proof under the unchanged security contract. They are not accepted
  replacements here. Changing the field, hash, rounds, `b`, or decomposition
  parameters is outside this research contract.

The [reported cvc5 split-solver bug](https://hackmd.io/@tbtl/BJ8ak2W9bl)
affected a different finite-field backend; the author reports its fix was
merged on 2026-02-27. The saved queries explicitly select `gb`, so this report
does not establish an affected result here. It gives no reason to expand
solver trust: accepted claims still require Lean reconstruction.

## Direct witness work and measured scope

The formal direct combination executor computes required outputs without
materializing the old multiplication scratch interval. Geometry proves the
interval `[21124070, 28972970)` and input/output separation. The interval has
7,848,900 source scratch columns; these are not an additional committed-width
saving. Full/direct witness correspondence and custody concern computation
and proof transport. They leave the checkpoint CCS coordinates, rows, and
matrix nonzeros unchanged. No Rust timing or peak-memory result exists for
this construction.

The complete selected Lean producer connection now passes.
[CanonicalDirectPhysicalExecution](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/CanonicalDirectPhysicalExecution.lean)
derives support for every event of the actual sorted canonical plan and
executes from the caller-seeded array. It needs no completed full witness or
caller-supplied support premise. `execute_agree` proves identical errors or
successful arrays that agree outside scratch. `successful_assignment_eq`
proves equality of the complete retained CCS assignment and public digest
for the selected Poseidon application. The proof includes actual recipe and
hint reads, permutation inputs, ordinary instructions, and First54 compact
recipes and A/B/C checks. The full physical executor remains the reference.
The canonical package and Rust direct producer are integrated in checkpoint
`f42a6d53d`; the selected assignment and lifecycle checks passed.

Checkpoint validation passed: full Lean library, 4,018 jobs in 79 s; full
axiom/test build, 4,177 jobs in 26 s. New focused checks include research
bounds (3 s), scratch custody (25 s), and scratch geometry (4 s). Research
checkpoint `a650fa77` passed static checks and the full 4,019-job Lean library
build in 38 s. Its full axiom/test gate passed all 4,182 jobs in 7 s.
The earlier producer-support batch passed static checks, the full library
(4,020 jobs, 203 s), and the full axiom/test gate (4,189 jobs, 6 s).
The final complete-producer proof passed the full library with **4,020 jobs
in 130 s** and the full axiom/test gate with **4,207 jobs in 100 s**, including
**71 new public theorem audits**. These are build/check times, not prover
benchmarks. Candidate checks measured 42.43 s
for all logical matrices, 26.67 s for physical matrices, 75.81 s for identity
rejections, and 70.88 s for the complete base assignment. The earlier combined
assignment regression reached its 300 s cap. The split output-digest recipe
check subsequently passed, closing that rejection case. Detailed completed
coverage is in [PLAN.md](PLAN.md).

The exact old-layout CPU baseline used the same reference application and
Nightstream `k_rho = 16` profile on the AMD host:

| Run | Preparation | Base proof | Active fold | Verification | Total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| One step, verified | 26.210 s | 21.826 s | — | 32.254 s | 80.290 s | 5,984,964,608 bytes |
| Two steps, verified | 21.399 s | 17.132 s | 226.013 s | 152.593 s | 417.137 s | 8,500,621,312 bytes |

[The two-step record](baseline-cpu.json) includes the source, command, identity,
and final state. Its 1,200 s cap had specific owner approval. The earlier
300 s attempt timed out during verification and remains a failed attempt;
it completed an active fold in 224.995 s. No candidate lifecycle benchmark
has run. Production selection, saved recursive fixtures, and the final Rust
checks are complete. The owner stopped timing comparisons and directed work
to constraint reduction. The next Lean batch is now stable; its required artifact and fixture
integration passed. Timing comparisons remain paused.

A viable half-coordinate candidate remains unknown. Any proposed replacement
must prove the unchanged relation, constructive completeness, efficient
witness recovery for extraction, degree and norm bounds, and complete selected
path preservation. This is the same obligation used for the quotient change
and the [SuperNeo](https://eprint.iacr.org/2026/242) composition; fewer variables
or a cvc5 `unsat` result alone do not discharge it.

## Later matrix-cost candidate

A source-level candidate keeps the same 54 quotient field slots and 108
checks, but stores `Q(0), ..., Q(53)` instead of monomial coefficients. Fixed
Lagrange interpolation recovers a polynomial of degree at most 53. The first
54 quotient forms are singletons; the remaining 54 each use all 54 values.
This gives `54 + 54*54 = 2970` weights per ring, versus
`1 + 107*54 = 5779` now. The
[exact basis experiment](phi81_quotient_basis.md) checked both field witness
maps, all 108 saved quotient forms, the lane/cell slot map, and the 41-trit
expansion. It confirms the predicted saving of
`2809*41*969 = 111598761` normalized entries. Total nonzeros would be
2,889,972,884, still above the original baseline.

This candidate changes neither coordinate count nor logical rows. It still
needs Lean soundness, constructive encoding and extraction maps, matrix
placement proofs, selected consumer conformance, and runtime measurements.
The exact Python checks and cvc5 basis controls passed in 4.090 seconds; they
are not Lean proof authority. The candidate has not been implemented or
selected. The whole-producer proof does not change the checkpoint totals or
establish the additional 50% coordinate reduction.
