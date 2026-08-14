# Nebula F′ production-readiness report

Date: 2026-08-14

## Verdict

Nebula F′ is not ready for production.

The simple WASM proof path works end to end on Metal. The current branch also
removes large implementation costs and fixes two radix-four correctness and
lifecycle faults:

- Bootstrap recursion and steady recursion now share one physical recursive
  relation arm.
- Geometric matrix runs stay compact on the CPU and Metal paths.
- ROM geometry is fixed by the profile, not by each program.
- Program-specific values are witnesses under one verifier-owned Poseidon2
  binding. They are no longer matrix coefficients.
- One radix-aware accumulator codec now serves radix two and radix four. The
  previous native path hard-coded radix two and produced an invalid
  radix-four recursive assignment.
- A reduced-memory radix-four test now completes base, bootstrap recursion,
  steady recursion, terminal finalization, and verification on Metal.
- The exact production-width radix-four evaluator artifact is 460.629 MiB.
  Its encoder artifact is 63.920 MiB. They share the exact relation shape and
  matrix digest.
- Artifact-backed setup no longer emits or constructs a second raw CSC
  relation. It uses an explicit shape-only relation header and fails closed if
  code requests raw matrix entries.
- Parallel checked radix-four artifact load plus complete profile restoration
  takes 3.843 seconds. A later program bind takes 60.5 ms. Cold load has
  1.157 seconds of measured margin below the five-second gate.
- A complete radix-four candidate census now has 8,102,331 rows and
  12,288,726 columns. It selects `ell = 24`, with 4,488,490 columns of margin.
- Pairing centered-domain checks removes 2,482,376 final rows, or 23.45%,
  against the prior candidate. The fixed-point feedback adds 65,340 columns,
  but both axes remain below `2^24`.
- The previous 5,624,622-row result is superseded. It used a base-two
  signed-unit encoding under base-four parameters. The corrected compiler uses
  septenary words and explicitly constrains coordinates that must remain in
  `{-1, 0, 1}`.
- The PiRLC first-accepted selection compiler now replaces each 36-row source
  block with nine product-sum rows. Across 432 blocks, this removes 11,664
  final rows.
- Rust generates all 6,480 canonical-X rows for that candidate. An independent
  Lean checker accepts every row. Lean also proves that satisfaction of these
  exact generated rows forces the unique canonical seven-child split.
- The generated compact Poseidon2 trace now covers all 86 S-box input forms
  and eight final output forms. Lean proves that its physical equations force
  the same selected reference permutation on the same canonical assignment.
- The selected PiCCS execution receipt now binds the exact 74-term polynomial.
  The Rust drift test and the independent Lean receipt checker both pass.
- A bounded six-row grouped-product fixture now has an executable Lean
  polynomial certificate. Satisfaction of its 33 exact source rows implies all
  six decoded recurrence steps on the same source assignment and three explicit
  carry values.
- Property `FPRIME-R4-SOURCE-STAGE-COVERAGE` now gives the production-width
  recursive arm a 14-record outer certificate. Rust checks every nonempty
  physical stage and every branch-private source-field disposition. Lean checks
  the exact aggregate partition without axioms.
- The evaluator cache now stores every row reference in four bytes. Signed
  singletons encode the block, local coordinate, and sign directly. Dense
  blocks use a checked side table. This closes the 512 MiB cache gate.
- The specialized radix-four Metal evaluator now includes the selector
  correction polynomial used by Rust. The complete Metal adapter suite passes:
  13 tests pass, zero fail, and one remains ignored.

These changes make the relation much smaller and cold setup much cheaper than
the initial implementation. The radix-four result closes the numerical
`ell = 24` width target for a candidate relation. The candidate now also has a
corrected reduced-memory all-branch proof and exact production-width loadable
artifacts.
It is not the selected production relation. It still needs a production-width
and 1,088-fold endurance proof, complete recursive and terminal conformance,
a frozen artifact manifest, and hostile-input hardening. The selected
production profile remains radix two.

## Scope and assumptions

This report covers:

- the SuperNeo, HyperNova, and Nebula protocol documents in `docs/`;
- the active Nightstream Lean project in `formal/nightstream-lean`;
- the Rust implementation in `neo-fold-clean`, `neo-reductions`, `neo-wasm`,
  and `neo-prover-metal`;
- the WASM Nebula benchmark named `wasm_nebula_pipeline_profile`.

The timing benchmark uses an Apple M5 Max and the Metal NIFS backend. It proves
a simple ten-instruction WASM program. It uses production SuperNeo core
parameters, a reduced `R = M = 2,048` memory profile, one base fold, one
bootstrap-recursive fold, and the required terminal fold. It does not execute
steady recursion in the same measured proof.

The production-width census is a separate run. It uses `R = 4,096`,
`M = 65,536`, `N = 1,088`, `B_ops = 63`, `B_scan = 64`, and application batch
size 3. The run completed exact relation generation and the full width audit.
It was stopped before the three proof folds. Therefore, its rows and columns
are exact for the generated relation, but it is not a proof-time benchmark or
a release receipt.

The radix-four candidate artifact run uses the same production WASM limits. It
uses seven running children, `kappa = 18`, and `lambda = 114`. It completed
exact relation generation and cache construction in 16.879 seconds. It wrote
and checked both loadable artifacts. It did not run the production-width proof
or the 1,088-fold endurance schedule. Its dimensions and artifacts are exact,
but it is not a production proof receipt.

The first all-branch Metal attempt exposed a signed-unit-only path when the
second fold consumed radix-four running digits. The compact Metal digit-mask
path now accepts `0`, `±1`, `±2`, and `±3`. A focused test constructs running
witnesses that contain magnitude-two and magnitude-three digits. It checks the
complete Metal NIFS output against the canonical CPU prover and verifies the
result. A separate enlarged fixture passes the complete base,
bootstrap-recursive, steady-recursive, and terminal lifecycle. The corrected
run took 55.341 seconds. This is a backend and lifecycle check. It is not the
production-width authority.

Enzo's memory router is unchanged. This branch has no diff in
`crates/neo-wasm/src/memory_routing.rs`. The measured relation still has 760
logical ports and 210 physical slots for the ten-instruction batch.

## Critical path and ownership

| Stage | Owner | Production responsibility |
|---|---|---|
| WASM prepared profile | `crates/neo-wasm/src/nebula.rs:266` | Own one process-local compiled profile and bind exact program values |
| WASM profile and program preprocessing | `crates/neo-wasm/src/nebula.rs:362` | Validate limits, build fixed memory geometry, and construct the exact program plan |
| Fixed F′ relation | `crates/neo-fold-clean/src/frontends/nebula/f_prime.rs:161` | Compile the base and shared recursive arms into one selective CCS relation |
| Nebula prepared profile | `crates/neo-fold-clean/src/frontends/nebula/f_prime/chain.rs:143` | Share the exact relation, encoder maps, verifier setup, and evaluator cache across program bindings |
| Encoder artifact | `crates/neo-fold-clean/src/frontends/r1cs_f_prime/lowering/encoder_artifact.rs:1` | Persist and bounded-load the exact assignment encoder without matrix content |
| Nebula artifact pairing | `crates/neo-fold-clean/src/frontends/nebula/f_prime/encoder_artifact.rs:1` | Bind encoder arm metadata to the evaluator shape and matrix digest |
| Optimized CPU cache | `crates/neo-reductions/src/engines/optimized_engine/mod.rs:91` | Own the matrix digest and compact SuperNeo evaluation state |
| Metal adapter | `crates/neo-prover-metal/src/adapter.rs:36` | Prepare static GPU state and accelerate NIFS and terminal openings |
| WASM proof | `crates/neo-wasm/src/nebula.rs:781` | Build application witnesses and append Nebula F′ steps |
| WASM verification | `crates/neo-wasm/src/nebula.rs:847` | Check the final WASM state and verifier-owned proof bindings |

The intended deployment flow is:

```text
profile limits -> one reusable F′ relation and static cache
program ROM/RAM -> exact plan and Poseidon2 program binding
trace + both inputs -> Metal proof -> verifier recomputes the program binding
```

The first arrow can now load the evaluator and exact encoder artifacts without
relation synthesis. Repeated programs reuse the loaded encoder maps, Ajtai
setup, and evaluator cache. The process still reconstructs the checked WASM
profile template and verifier policy. Raw CSC matrices do not exist on this
path.

## Measured result

The historical optimization series used the old `R = M = 1,024` benchmark
fixture. Its final point was 6,675,851 rows, 18,442,350 columns, a 0.48 GiB
evaluation cache, 15.696 seconds of preprocessing, and 20.263 seconds total.
That fixture cannot represent the new fixed ROM ranges and is not a valid
cross-program reuse profile.

The live-construction reusable-profile measurement is:

| Run | Rows | Columns | Eval cache | Fixed preprocessing | Metal setup | Proof | Verify | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed ROM profile | 6,676,895 | 18,463,842 | 493.35 MiB | 16.441 s | 0.438 s | 8.900 s | 1.992 s | 27.929 s |

A separate all-branch Metal test uses four application folds with
`B_scan = 1,024`. It executes base, bootstrap recursion, steady recursion, and
terminal finalization in one proof. Its corrected reduced-memory fixture has
11,414,566 rows and 13,056,714 columns, so it selects `ell = 24`. The run took
55.341 seconds in total. Fixed preprocessing took 18.329 seconds, Metal setup
took 0.569 seconds, proof took 32.112 seconds, and verification took 4.210
seconds. These dimensions do not replace the production census. The fixture
uses different memory and WASM limits. The production-width authority remains
the separate 8,102,331-row, 12,288,726-column census above. The test is at
`crates/neo-wasm/tests/wasm_nebula_pipeline_profile/radix_four.rs`.

The process-local prepared-profile test now measures a different lifecycle:

| Operation | Time |
|---|---:|
| Compile relation and evaluator cache once | 14.825 s |
| Bind first program | 65.9 ms |
| Bind second program | 75.3 ms |

Both bound programs share the exact compiled relation and evaluator storage.
They have different initial semantic-state, plan, `D_init`, and program-binding
digests. A proof produced under the first program fails verification under the
second program. The release gate requires each bind to stay below 500 ms. See
`crates/neo-wasm/tests/wasm_nebula_pipeline_profile.rs:342`.

The reduced timing relation has 92,535,056 reported explicit matrix entries,
36 seeded Phi81 blocks, and 2,384,313 compact geometric runs. The runs
represent 97,757,569 scalar slots without materializing those slots. The
relation still selects a `2^25` joint domain.

The old and new timings are not a strict performance comparison because the
memory profile changed. The measured 8.78–9.39 second proof range is still a
real regression from the prior 2.8 second measurement. Its cause is not
isolated.

The persistent-profile benchmark measures the same 6,676,895-row relation:

| Artifact operation | Result |
|---|---:|
| Evaluator artifact | 531,441,005 bytes / 506.822 MiB |
| Encoder artifact | 52,245,432 bytes / 49.825 MiB |
| Evaluator write, including Poseidon2 receipt | 2.688–2.727 s |
| Encoder write, including Poseidon2 receipt | 1.069–1.079 s |
| Evaluator checked load | 3.001–3.578 s |
| Encoder checked load | 1.142–1.145 s |
| Parallel checked load | 3.001–3.579 s |
| Restore the complete prepared profile | 1.095–1.099 s |
| Parallel load plus profile restoration | 4.101–4.674 s |
| Bind first / second program | 152–157 ms |
| Metal setup | 0.463–0.484 s |
| Uncontended proof | 9.246–9.392 s |
| Verification | 2.044–2.084 s |

The artifact-backed proof and verification pass. The loader checks exact byte,
row, column, matrix, arm, field, and derived-value limits before run expansion.
Changed bytes, a different matrix identity, and an oversized expanded encoder
fail before use. The loaded profile binds two different programs while sharing
the exact relation and cache. A proof for the first program fails under the
second program. Recursive-arm synthesis, selective layout, row emission, and
raw matrix construction are absent from this path. The measured cold-profile
budget is now met.

The exact production-width radix-four artifact benchmark reports:

| Artifact operation | Result |
|---|---:|
| Evaluator artifact | 483,004,415 bytes / 460.629 MiB |
| Encoder artifact | 67,025,162 bytes / 63.920 MiB |
| Live relation and cache construction | 16.879 s |
| Evaluator write, including Poseidon2 receipt | 2.881 s |
| Encoder write, including Poseidon2 receipt | 1.330 s |
| Parallel checked load | 3.159 s |
| Restore the complete prepared profile | 0.685 s |
| Parallel load plus profile restoration | 3.843 s |
| Bind one later program | 60.5 ms |
| Peak descendant RSS for the complete artifact run | 10,906,304 KiB |

The public restore path requires the verifier-owned Ajtai public parameters
for the exact `(D, m / D)` width before restore starts. The benchmark installs
a seeded global test setup. Deployment must install and bind the reviewed
production setup through the profile manifest. Artifact bytes do not provide
that authority.

### Production-profile geometry

The exact selected radix-two production-width census reports:

| Measure | Value |
|---|---:|
| Final selective CCS rows | 9,304,520 |
| Final selective CCS coefficient coordinates (`m`) | 25,870,482 |
| Packed witness shape | `54 × 479,083` ring coordinates |
| Public columns | 2,430 |
| Committed coordinates after the constant | 25,870,481 |
| Matrices / maximum degree | 13 / 8 |
| Selected joint domain | `2^25` |
| Row padding to `2^25` | 24,249,912 |
| Column padding to `2^25` | 7,683,950 |
| Reported explicit nonzero entries | 106,554,114 |

The complete radix-four candidate census reports:

| Measure | Radix two | Radix four | Change |
|---|---:|---:|---:|
| Final selective CCS rows | 9,304,520 | 8,102,331 | -1,202,189 / -12.9% |
| Final selective CCS columns | 25,870,482 | 12,288,726 | -13,581,756 / -52.5% |
| Selected joint domain | `2^25` | `2^24` | one exponent lower |
| Limiting margin below `2^24` | not applicable | 4,488,490 | 26.75% |
| Full relation preprocessing | not measured in this census | 16.879–17.122 s | artifact and census runs |

The candidate cache has 41,286,880 row blocks and 2,056,688 geometric runs.
Of the row blocks, 39,911,332 are signed singletons and 1,375,548 use a dense
side table. Its compact storage is 444.18 MiB: 202.47 MiB of row offsets,
157.50 MiB of four-byte row references, 10.49 MiB of dense side references,
3.47 MiB of interned dense coefficients, 23.18 MiB of geometric offsets, and
47.07 MiB of geometric runs. The persisted evaluator form is 460.629 MiB.
This cache is reusable profile data. The 16.9–17.1-second live construction is
offline profile work. A later program bind does not repeat it.

### Centered-domain pairing

The previous candidate used one cubic centered-unit row for each of 4,970,473
coordinates. The final compiler now pairs two residuals in one degree-six row:

```text
(left^3 - left)^2 - 7 * (right^3 - right)^2 = 0
```

Seven is a proved projective nonresidue in the production Goldilocks field.
Therefore, an active pair row is zero exactly when both centered-unit
residuals are zero. An odd final coordinate uses the same row with the right
residual fixed to zero. The 4,970,473 coordinates now use 2,485,236 pair rows
and one tail row, for 2,485,237 rows in total.

The complete relation changes from 10,584,707 rows and 12,223,386 columns to
8,102,331 rows and 12,288,726 columns. This is a net removal of 2,482,376 rows,
or 23.45%, with 65,340 added columns from exact fixed-point feedback.

Lean proves the pair and tail equivalences at the security-reduced model tier.
A generated artifact also checks one exact production pair row and the odd
tail row against their final matrix coefficients. This is not yet a
family-wide Rust-conformance theorem. Before Lean can authorize later changes
to this family, Rust must generate a compact coverage certificate for every
pair and tail interval, and Lean must prove that the intervals cover the full
4,970,473-coordinate census without overlap or remainder.

For the selected radix-two relation, reaching `2^24` would require removal of
9,093,266 columns, or 35.15% of its total. Radix four closes that numerical gap
through a parameter and circuit change. The selected radix-two recursive arm
accounts for 25,814,422 final columns. Its source relation has 27,164,726
columns. Selective lowering removes 13,833,463 source columns, keeps 12,855,060
unit columns and 475,710 balanced columns, creates 2,033,600 derived-product
columns, and allocates 23,780,822 branch columns after aliases.

The family census is inclusive: parent ranges contain child ranges. Its values
must not be added together. The physical-stage audit now assigns every
recursive branch coordinate to exactly one allocation interval. Its largest
exclusive stages are:

| Exclusive physical stage | Allocated coordinates |
|---|---:|
| PiCCS claim allocation | 6,452,816 |
| PiRLC parent and child allocation | 3,452,220 |
| PiCCS claim and accumulator binding | 2,546,196 |
| PiCCS challenge computation | 1,832,208 |
| Outgoing accumulator child-digest authority | 1,652,703 |
| PiCCS transcript prefix | 962,598 |
| PiCCS terminal check | 904,050 |
| PiCCS sum-check | 653,950 |
| PiCCS output SIS digest | 529,404 |
| Delayed Nebula transition | 421,942 |

The five largest stages use 15,936,143 coordinates, or 67.0% of the complete
recursive branch. This confirms that the width is mainly the materialized
NIFS verifier and its state authority. It is not mainly alignment or final
domain padding.

The 25,870,482 value is a flattened coefficient-coordinate count. It is not
the number of independent PLONK-style advice columns. The committed witness
packs those coordinates into 479,083 ring columns with 54 coefficients per
column. The performance problem is still real: the current one-SumCheck
domain is selected from the flattened row and coordinate bounds, so this
representation fixes `ell = 25`.
The production census and printout are owned by
`crates/neo-wasm/tests/wasm_nebula_pipeline_profile.rs:1061`. The exclusive
stage accounting is implemented at
`crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective_audit.rs:783` and
connected to lowering at
`crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective.rs:675`.

### Width is an architecture problem

The current compiler builds one flat R1CS/CCS assignment. A new field value or
intermediate result normally gets a new global assignment coordinate. This is
a compiler and relation design choice. Nebula soundness does not require every
temporary value to remain a separate global coordinate.

The `2^24` result is a release ceiling, not the architectural end state. A
12.22-million-coordinate one-shot verifier is still too large. The long-term
target is a narrow verifier state machine: each transition performs a bounded
amount of verifier work, updates verifier-owned authenticated state, and makes
terminal acceptance possible only after every scheduled check is complete.
This design is useful only if its queue cannot skip, reorder, replay, or leave
new verification work growing faster than it is discharged. Lean must prove
those lifecycle properties before Rust adopts this architecture.

The requirements that cannot be removed are:

- verifier-derived transcript challenges;
- exact verification of the NIFS subprotocols;
- authoritative prior-state and successor-state links;
- exact application and memory semantics;
- a verifier-controlled lifecycle that cannot skip a required check.

The current one-step, one-flat-assignment form is not itself a security
requirement. There are three possible reduction levels:

1. Exact aliases and algebraic substitutions inside the current relation. This
   is the lowest-risk option, but the measured 9,093,266-column gap is too
   large to assume that local changes are sufficient.
2. A deferred-verification F′ pipeline. Each invocation checks one scheduled
   part of an earlier NIFS verification and carries the exact continuation
   state. This is not the same as putting PiCCS, PiRLC, and PiDEC into three
   ordinary recursive steps: each such step would still have to verify its
   own complete fold and would reproduce the large verifier. A useful pipeline
   needs a new soundness argument for the delayed checks.
3. A different trace arithmetization or proof composition. This can change the
   width model, but it is a larger backend and security change.

The first architecture target is a narrower SuperNeo decomposition. The active
profile uses radix two and 14 running sources. A radix-four candidate uses
seven running sources while it keeps `B = 16,384`, `T = 216`, the degree-eight
CCS relation, and joint verifier degree nine. This attacks the large carried
state before compiler-level cleanup. A direct degree-eight CCS verifier is the
second target. It must not materialize duplicate claim views or R1CS-only
intermediate values. A deferred-verification pipeline is the fallback if the
parameter change, direct CCS, and exact aliases cannot close the remaining
gap.

The radix-four candidate has passed two model-level checks:

- Lean proves the Definition 14 RLC bound, the unchanged Module-SIS norm
  parameter, the unchanged degree-eight verifier degree, and the reduction
  from 14 to seven running sources.
- Lean also proves an exact seven-child canonical split of every bounded
  Phi81 assignment, both norm directions, and equality between Horner
  recomposition and the verifier's `4^i` weighted recomposition.
- The Rust security census passes against the current production relation
  shape. Its extraction fork factor is 26 instead of 76.

It has also passed one Rust-conformant leaf check:

- Rust emits an exact radix-four canonical-X circuit and a matching selective
  reconstruction rule.
- The generated artifact contains all 6,480 physical rows for 270 logical
  coordinates and 6,481 columns.
- An independent Lean row compiler checks all 29 shards. The geometry link and
  the axiom guard pass.
- A second Lean refinement proves that satisfaction of those exact generated
  rows forces the verifier-computed canonical radix-four split. Alternate
  limb assignments and alternate child decompositions cannot pass this leaf.

It now also has complete Rust-conformant selector-port coverage under property
`FPRIME-R4-SELECTOR-COVERAGE`:

- Rust reads the final general and evaluation selector CSC matrices and
  reconciles them coefficient-for-coefficient with all 185,526 compiler owner
  runs. There are 180,665 nonempty source runs.
- Adjacent runs merge only when their owner family, arm, selector port,
  selector column, and coefficient are equal. This reduces the formal wire
  image to 14 maximal intervals without changing the covered rows.
- The generated artifact fixes all 8,102,331 rows, 12,288,726 columns,
  selector columns `[2430, 2431]`, and all 74 ordered
  polynomial terms.
- The latest complete drift-owner rebuild takes 17.85 seconds. Relation and
  evaluator-cache construction dominate this time; the 14-run export is small.
- Lean checks the profile, dimensions, interval partitions, owner-to-gate map,
  unit coefficients, and polynomial syntax. It proves that every candidate
  row belongs to one owner interval with its expected general or evaluation
  selector gate.

This result owns only selector activation. It does not identify the arithmetic
equation in each row or prove the other eleven matrix ports.

The fixed running-claim carrier changes from
`14 × (3,888 + 540 + 1,512) = 83,160` fields to
`7 × (3,888 + 540 + 1,512) = 41,580` fields. This is an exact carrier-width
reduction. It is not an exact complete-circuit saving because radix-four
child values require a wider small-alphabet encoding than binary child values.
The complete generated census remains the decision authority.

This is still not the selected production circuit. The exact Rust circuit and
the complete relation census now exist, but the formal result is local to the
canonical-X leaf. The complete recursive and terminal relations do not yet
have same-assignment conformance artifacts. No accepted production relation
has changed.

Model-level owners:

- `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Parameters.lean`;
- `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/PiDECAlgebra/Radix4.lean`;
- fail-closed reports in
  `formal/nightstream-lean/tests/Axioms/SuperNeoRadix4Candidate.lean`.

Rust-conformant leaf owners:

- `crates/neo-fold-clean/src/engine/r1cs_circuit/pi_dec_canonical_x_program.rs`;
- `crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective_definitions.rs`;
- `crates/neo-fold-clean/tests/f_prime/pi_dec_canonical_x_lean_artifact.rs`;
- `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Gadgets/PiDecStrictProductionCompiler/RadixFourCanonicalX.lean`;
- `formal/nightstream-lean/tests/Axioms/PiDecRadixFourCanonicalXRustConformance.lean`.

Rust-conformant selector-coverage owners:

- `crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective_selector_coverage.rs:198`;
- `crates/neo-wasm/tests/wasm_nebula_pipeline_profile/radix_four.rs:165`;
- `formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRadixFourSelectorCoverage.lean`;
- `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/SelectiveCcs/SelectorComposition/RadixFourSelectorCoverageArtifact.lean:71`.

A deferred pipeline is sound only if the phase tag, profile identity,
transcript state, required inputs, and required outputs are constrained and
carried by the verifier-owned lifecycle. A digest can compress this
continuation, but it cannot be its authority. Terminal acceptance must prove
that every required phase ran in the required order and that every delayed
check was discharged.

Decision rule: keep radix two selected until the radix-four production-width
and 1,088-fold endurance run passes, the complete recursive and terminal Lean
bridges pass, and the reviewed manifest owns the artifacts and Ajtai setup.
The reduced all-branch lifecycle and production-width artifact restore now
pass. Use a direct degree-eight CCS prototype only if the candidate's proof
time or margin is not sufficient. Specify a deferred-verification state
machine in Lean before any such Rust architecture change.

## What changed

### One radix-aware accumulator codec

The native accumulator digest selected its compact family codec with a
hard-coded radix-two split. The radix-four circuit used the compact
radix-four digest, but the native handle fell back to the conservative digest.
This made the frontend assignment invalid. The first exact failure was row
3,730,978 in
`fprime.recursive.step.accumulator.output_authority.aggregate`.

There is now one native `strict_radix_accumulator_family_digest` and one
in-circuit mirror. Both receive the verifier-owned `Params::b()` value. All
production transition and finalization paths pass that value explicitly. The
digest domain and preimage bytes did not change. No second radix-four protocol
implementation was added.

A regression uses a parent `X` coefficient of two. It proves that radix four
selects the compact digest, that this digest differs from the conservative
fallback, and that radix two rejects the radix-four-only decomposition. The
complete reduced-memory radix-four lifecycle passes after this fix.

### Exact radix-four Metal digit masks

The Metal running-digit path now accepts the exact radix-four alphabet `0`,
`±1`, `±2`, and `±3`. Tests pin the plane order to
`[+1, -1, +2, -2, +3, -3]`. They reject magnitude four for radix four and
magnitude two for radix two. Focused host/Metal tests cover geometric running
evaluation and complete NIFS output parity.

The specialized radix-four selective shader is enabled. A complete NIFS
regression first found a mismatch at round seven. The shader had omitted the
current general-selector and evaluation-selector correction terms. The fixed
base-field and extension-field kernels now use the same factored correction
polynomials as Rust. Packed base-application tables, folded application tables,
assignment tables, and the general polynomial kernel match the CPU path. The
complete Metal adapter suite passes with 13 tests passing, zero failing, and
one ignored.

Fresh-instance construction had a separate signed-unit fast path. It rejected
the first valid radix-four running digit with value two. Radix two still uses
that packed fast path. Radix four now uses the existing dense radix-aware
encoder and builds the fresh CCS instance from that exact assignment when the
packed adapter declines it. This removes the correctness fault without a new
protocol path. It costs about 0.45–0.56 seconds for each recursive instance.
The next backend change is one canonical packed magnitude-mask assignment API
for radix four, followed by a differential test against the dense assignment.

### Aggregated first-accepted selection

The PiRLC rejection sampler previously materialized three products for each of
11 candidates. The compiler now enforces the same three aggregate identities:
accepted sum equals one, prefix-weighted sum equals the output position, and
symbol-weighted sum equals the selected output. The production census reports
432 blocks:

| Measure | Total | Per block |
|---|---:|---:|
| Exact source rows | 15,552 | 36 |
| Final emitted rows | 3,888 | 9 |
| Removed final rows | 11,664 | 27 |
| Source fields | 14,688 | 34 |
| Trace-eliminated fields | 14,256 | 33 |
| Allocated final coordinates | 9,936 | 23 |

The focused Rust compiler test passes. The generated schedule fixes all eight
samplers and expands them into 432 non-overlapping source and emitted
intervals. Lean checks that exact schedule and applies the model-level
`currentAt_iff_aggregateAt` substitution at every output position. The
remaining gap is the final nine-row low-norm gate semantics and its join to
sampler one-hotness and complete PiRLC semantics. Therefore, this is not yet
whole-relation Rust conformance.

### Reusable initial-memory binding

`NebulaPlan` now separates immutable profile state from program memory. A
prepared profile reuses the same `S_mem` circuit and lane commitment matrices.
`bind_initial_memory` recomputes only `D_init` and the plan digest. Exact tests
show that the bound plan has the same `D_init`, plan digest, rows, and columns
as a fresh plan.

Initial binary lane data now uses exact compact column masks. An all-zero lane
returns the exact zero Ajtai commitment. The production initial image has
1,081 zero scan lanes and seven nonzero scan lanes. The zero commitment has
one fixed memory-leaf digest, because that leaf has no step index. `D_init`
therefore computes this leaf once and reuses it. It still computes every chain
link in order, because each link has a different prior digest.

The zero shortcut checks the exact `lane_columns * D` input width before it
returns. A wrong-width all-zero slice is rejected with `WitnessWidth`, just as
the dense path rejects it. This keeps the fast path fail-closed without adding
matrix packing or a second bit scan to valid zero lanes.

| `D_init` phase | Before | After |
|---|---:|---:|
| Encode 1,088 lanes | 3 ms | 3 ms |
| Commit lanes | 26 ms | 25 ms |
| Compute memory leaves | 592 ms | 11 ms |
| Compute sequential chain links | 3 ms | 3 ms |
| Complete `D_init` | 625 ms | 42 ms |
| Plan digest | 9 ms | 9 ms |
| Complete later program bind | 641 ms | 60.5 ms |

The optimization is exact common-subexpression reuse. It does not change a
field, tag, hash family, absorb order, or chain link. An external test compares
the cached result with the original uncached formula over repeated zero lanes.

### One physical recursive arm

`NebulaFPrimeBranch` still has three lifecycle states: base, bootstrap
recursion, and steady recursion. Both recursive states call the same recursive
R1CS circuit. Preprocessing previously cloned that R1CS and emitted it under two
selectors.

The compiler now emits two physical arms:

- arm 0: base;
- arm 1: recursive, used by bootstrap and steady recursion.

This is an exact language-preserving reduction. The union
`base ∪ recursive ∪ recursive` equals `base ∪ recursive`.

Rust generates the mapping `[0, 1, 1]` into a Lean artifact. Lean proves the
accepted-language equivalence. The assurance tier is artifact-checked for the
Rust mapping and model-level for the set equivalence. This result does not
prove the semantics of every recursive-circuit row.

Relevant owners:

- `crates/neo-fold-clean/src/frontends/nebula/f_prime.rs`
- `crates/neo-fold-clean/src/frontends/nebula/f_prime/shape.rs`
- `crates/neo-fold-clean/tests/gadgets/nebula_recursive_arm_lean_artifact.rs`
- `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/SelectiveCcs/NebulaRecursiveArmSharing.lean`

### Compact geometric matrix runs

The evaluation cache previously expanded geometric matrix runs into scalar row
blocks. CPU evaluation, weighted evaluation, ring-form evaluation, and Metal
opening evaluation now consume the compact descriptors directly.

Relevant owners:

- `crates/neo-reductions/src/superneo_eval/geometric.rs`
- `crates/neo-reductions/src/superneo_eval/cache.rs`
- `crates/neo-prover-metal/shaders/joint.metal`
- `crates/neo-prover-metal/shaders/dec_forms.metal`
- `crates/neo-prover-metal/src/session/joint/opening.rs`

### Exact evaluator-cache interning

The relation refers to 2,661,412 dense 54-coefficient blocks, but it contains
only 23,778 distinct coefficient patterns. The cache now stores each distinct
pattern once and remaps each row block to it. Two fingerprints group possible
matches. Exact field-array equality decides every merge, so a fingerprint
collision cannot change evaluation.

Row offsets now use one 32-bit base for each 256-row chunk and exact 16-bit
local offsets when each chunk fits. They fall back to the prior 24-bit or
32-bit form when required. Metal expands these host offsets once during static
upload. The shaders and relation do not change because of this offset format.

The pre-packed 493.35 MiB host cache contained:

- 166.85 MiB of row offsets;
- 248.04 MiB of row blocks;
- 4.79 MiB of interned dense coefficients;
- 19.10 MiB of geometric-row offsets;
- 54.57 MiB of geometric-run descriptors.

All 24 CPU evaluator-equivalence tests and the focused Metal compact-row test
pass. The full Metal adapter suite also passes. This intermediate form already
met the 512 MiB evaluator-cache gate. It did not reduce GPU upload size because
Metal expands chunked row offsets to 32 bits during setup.

### Four-byte production row references

The production cache previously stored each row block as a wider tagged
record. It now stores one four-byte reference per block. A signed singleton
packs a 24-bit block index, a six-bit local coordinate, and its sign. A dense
reference packs a checked 31-bit side-table index. Each dense side entry stores
the exact block and interned pattern in eight bytes.

The artifact loader checks every packed block, local coordinate, dense index,
and side-table entry before restoration. The CPU and Metal paths decode the
same format. The production artifact schema and Poseidon2 digest domains were
advanced together, so an old or mixed artifact fails closed.

This changes storage only. It does not change a matrix value or proof
relation. The direct cache storage falls from about 591 MiB to 444.18 MiB, a
147 MiB or 24.9% reduction. The persisted evaluator artifact is 460.629 MiB.
Cold checked load plus complete restoration falls from 4.966 seconds to 3.843
seconds. The production cache gate is now closed with 51.371 MiB of file-size
margin.

### Smaller fixed-point discovery

The compiler no longer tests five optional shared-private prefixes in every
round. It shares only the `S_mem` private bit prefix that the lane commitment
requires. Mutually exclusive branches already reuse one branch arena, so more
sharing did not reduce the committed width.

Fixed-point discovery also starts in the selective relation family: 13
matrices, degree 8, and the verifier's row and packed-assignment domains. It no
longer performs a full first transition from the unrelated 15-matrix,
degree-4 `S_mem` relation. Exact `(rows, columns, arity, degree)` equality is
still required before preprocessing returns.

### Reused final selective layout

Each fixed-point round prepares an exact selective layout to obtain the next
relation shape. The final round previously discarded that layout and prepared
the same plans, aliases, slots, and row program again before matrix emission.

The final prepared layout now owns its source arms and is consumed once by the
emitter. This ownership prevents the shape check and emitter from receiving
different arms. The final repeated `selective-layout-core` phase, measured at
about 0.49 seconds in this run, is gone. A direct unit test checks that the
prepared shape, emitted relation, public width, and compiler audit agree.

### Profile-level relation and explicit program binding

The WASM memory layout previously derived ROM widths from the program's loaded
addresses. This made relation geometry program-dependent. The profile now owns
explicit limits for program counters, functions, types, control choices, and
grammar tables. See `crates/neo-wasm/src/nebula.rs:43` and
`crates/neo-wasm/src/nebula.rs:644`.

The exact initial semantic digest, plan digest, and initial-memory handle stay
in verifier-owned preprocessing. They are compressed with Poseidon2 into one
four-field program binding. The base circuit recomputes this binding from the
exact witness values, binds the semantic input and `D_init`, and carries the
binding in every Nebula lane. The terminal verifier recomputes it from its own
preprocessing and rejects a mismatch. See:

- `crates/neo-fold-clean/src/paper/construction2/nebula_lane.rs:314`;
- `crates/neo-fold-clean/src/paper/f_prime/nebula_lane_circuit.rs:187`;
- `crates/neo-fold-clean/src/lifecycle/verify.rs:1482`.

The following checks pass:

- two different binding values produce identical base-circuit matrix rows;
- two different WASM programs under one profile have the same relation digest,
  PiCCS header bundle, and Ajtai setup digest;
- the two programs have different plan, initial-memory, and carried binding
  digests;
- the terminal verifier rejects a substituted program binding.

The relevant regression tests are
`crates/neo-fold-clean/tests/f_prime/nebula_lane_circuit.rs:412`,
`crates/neo-wasm/tests/wasm_nebula_pipeline_profile.rs:358`, and
`crates/neo-fold-clean/tests/nebula/lifecycle.rs:184`.

This fixes the relation-authority boundary. The compact evaluator cache and the
exact encoder plan now load across processes. Their receipts must agree on the
relation shape and matrix digest. The prepared profile restores from those
artifacts without compiling an F′ arm or rebuilding raw matrices. Every later
program in that process reuses the result.

The terminal latest-witness check now evaluates the exact CCS polynomial over
the compact matrix authority selected by preprocessing. It no longer reads the
raw CSC relation. A differential test shows that the raw and compact checks
accept the same valid assignment and report the same first failing row after a
mutation. This closes one duplicate-authority path. The terminal decider
circuit compiler still reads materialized matrices and cannot yet use the
shape-only artifact header.

### Process-local prepared profile

`NebulaFPrimePreparedProfile` owns one compiled relation and one preprocessing
cache. Program binding clones only shared handles, checks exact Nebula
parameters, commitment seeds, memory routing, application R1CS, and recursive
plan shape, then installs the program-specific verifier policy. Only the
initial semantic-state anchor is excluded from the relation-profile comparison;
Nebula authenticates that value through the carried Poseidon2 program binding.

`WasmNebulaPreparedProfile` also retains the program-independent canonical WASM
application relation. A later bind does not regenerate its 51,329 rows. It
builds the exact ROM/RAM plan, updates the semantic anchor, and passes the exact
profile check. See:

- `crates/neo-fold-clean/src/frontends/nebula/f_prime/chain.rs:135`;
- `crates/neo-fold-clean/src/frontends/nebula/f_prime.rs:383`;
- `crates/neo-fold-clean/src/frontends/nebula/application.rs:442`;
- `crates/neo-wasm/src/nebula.rs:265`.

The in-memory sharing still ends when the process exits. The two persistent
artifacts now restore the same online state in a later process.

## Lean status

The important distinction is the assurance tier.

### Established

- Model-level SuperNeo, HyperNova, and Nebula definitions and local refinement
  results exist.
- The shifted-ternary selective rewrite has an exact generated local artifact.
  For the active `b = 2` profile, one canonical field opening uses 21 selective
  rows instead of retaining the 124 source R1CS rows.
- Rust already uses this 21-row rewrite in the selective compiler.
- The binary canonical-X checker covers all 4,590 exact Rust rows in 21
  bounded shards.
- The radix-four candidate checker covers all 6,480 exact Rust rows in 29
  bounded shards. It checks the seven-child geometry and the exact physical
  row schedule. Lean proves that satisfaction of the generated rows forces the
  unique canonical split. The axiom guards pass with only the expected
  proposition, quotient, choice, and `native_decide` trust dependencies.
- Property `FPRIME-R4-SELECTOR-COVERAGE` covers every production-width
  candidate row. The final Rust selector ports compress from 180,665 nonempty
  owner/gate runs to 14 exact semantic intervals. Lean checks these intervals
  and the exact 74-term polynomial, then proves every row has its expected
  selector gate.
- The generated compact Poseidon2 trace certificate covers all 86 scheduled
  S-box inputs and eight final forms. Compact coefficient tables replace the
  recursive partial-round expressions. `trace_refines_compact` proves
  same-assignment refinement, and `trace_computes_reference` proves the exact
  reference permutation result. Their axiom snapshot records only `propext`,
  `Classical.choice`, `Lean.trustCompiler`, and `Quot.sound`.
- The selected PiCCS receipt drift gate is consistent with the 74-term
  polynomial. A stale 66-term count existed in both the Rust generator and the
  Lean checker. Both counts and the generated transcript receipt are fixed;
  the independent checker accepts the receipt and rejects its mutation set.
- Lean proves the production Goldilocks centered-pair and odd-tail semantics.
  A generated artifact checks one exact final pair row and the exact odd-tail
  row at 8,102,331 rows and 12,288,726 columns. This is artifact-checked for
  those rows, not for every centered-domain interval.
- The new Nebula branch map is generated by Rust and checked by Lean.
- The first-accepted selection schedule has an exact generated artifact. Its
  eight compact sampler entries expand to 432 non-overlapping production
  intervals. Lean checks this exact Rust schedule and proves the model-level
  substitution. The artifact does not prove the final nine-row low-norm gate,
  one-hotness, complete PiRLC semantics, or the whole relation.
- The exact 3,682-row program-binding artifact is generated by Rust. Lean
  checks the 3,628 Poseidon2 row definitions and 54 equality checks, and proves
  that the carried binding, semantic state, and initial-memory handle have the
  required values.
- One bounded six-row product-sum rewrite from a deterministic Rust compiler
  fixture now has exact executable source provenance and exact final matrix
  rows. The fixture contains 27 used source slots, one affine source definition
  (`column 15 = column 3 + 3`), three derived slots, 33 exact source rows, six
  final rows, and 1,458 final assignment coordinates. Lean decodes the exact
  source interval `[4, 37)`, checks the row-owner join, and checks every source
  and derived port against its expanded final low-norm image.
- A proof-free executable degree-two normalizer checks the exact `q`, `p`, and
  `r` polynomial identities for those 33 rows. Its semantic theorem proves that
  satisfaction of all exact source rows implies all six decoded recurrence
  steps on the same source assignment and three explicit carry values. This is
  an artifact-checked forward refinement for the fixture. The guarded theorem
  reports `propext`, `Classical.choice`, `Lean.trustCompiler`, and `Quot.sound`,
  as expected for its `native_decide` certificate. It is not production
  coverage.
- Property `FPRIME-R4-SOURCE-STAGE-COVERAGE` is Rust-conformant for the outer
  recursive source-stage partition. The live compiler has 6,578 physical stage
  markers and 16,181,176 branch-private source fields. Exactly 6,575 stages
  match one of 14 reviewed top-level path prefixes; the other three markers are
  empty. The source fields partition into 2,838,233 direct fields, 5,006,998
  decomposition aliases, 445 equality aliases, 86,880 affine definitions, and
  8,248,620 trace-eliminated fields. The generated artifact is 78 lines. Lean
  checks the owner order, every owner total, all five global totals, and the
  stage and coordinate sums without axioms.
- This stage property is structural only. Physical path labels are caller
  assertions, and aggregate counts do not prove any arithmetic-family theorem.
  PiRLC alone contains 6,420 stage markers, so one theorem per stage is not a
  production-scale formal representation.
- The active Lean tree passes the forbidden-hole audit. The changed formal
  target builds without `sorry`, new axioms, `admit`, or `unsafe`. The complete
  validation script also passes its Rust-origin and relation-evidence checks.

### Not established

- Lean does not prove that the complete production recursive circuit should be
  much smaller than the current Rust relation.
- `WasmBenchmark42x6/CostArithmetic.lean` is a model-level cost model. It does
  not own production rows, columns, compiler recipes, setup, or Rust behavior.
- The V2 known-core result is a lower bound for a different model. It is not a
  generated production relation.
- The radix-four canonical-X result is Rust-conformant only for that leaf. It
  does not prove the complete recursive or terminal relation around the leaf.
- The compact Poseidon2 result is also a leaf theorem. It does not prove
  selector authority, call-site column renaming, or inclusion of those rows in
  the complete recursive and terminal relations.
- The complete selector-port result proves row activation only. It does not
  prove arithmetic-family identity, source-to-final assignment refinement, or
  any of the other eleven matrix ports.
- The complete radix-four recursive and terminal relations have no generated
  same-assignment refinement theorem.
- The first-accepted selection schedule is Rust-conformant for all 432 source
  blocks. It does not yet connect the final nine-row low-norm gate, one-hotness,
  and complete PiRLC semantics to a whole-relation refinement theorem.
- The radix-four candidate has completed a reduced-memory all-branch Metal
  proof and verification run. This execution evidence is not a Lean theorem
  and is not a production-width or 1,088-fold endurance receipt.
- There is no complete generated artifact for the production recursive and
  terminal relations with a same-assignment refinement theorem.
- The program-binding artifact proves its local row contract. It does not prove
  the complete base, recursive, or terminal relation around those rows.
- The grouped-product same-assignment theorem covers only the six generated
  fixture rows. It does not cover every production product-sum and
  polynomial-evaluation run. Complete coverage is required before this rewrite
  can authorize a production row or coordinate reduction.
- The bounded fixture forward direction passes. The separate reverse draft does
  not. Its final-rows-to-source theorem fails when it tries to rewrite the first
  carry equation into a goal where that carry term is already absent. Thus, the
  accepted final rows do not yet reconstruct a witness for all 33 exact source
  rows.
- There is no end-to-end security reduction from the exact production Rust
  verifier, transcript, parameter profile, and generated relation to the paper
  theorem.

Therefore, Lean can approve local rewrites after their artifact checks. It
cannot yet act as a complete oracle for arbitrary aggressive Rust changes.

### Sound reduction gate

No row or column class can be removed only because a census marks it as large,
repeated, or derived. Each reduction must have all of this evidence:

1. Rust generates an exact before-and-after artifact from the same emitter that
   builds the verifier relation.
2. Lean proves soundness: every assignment accepted by the reduced rows maps to
   an assignment accepted by the original F′ rows.
3. Lean proves completeness on the same witness data: each valid original
   assignment has the reduced assignment that the Rust compiler emits.
4. A removed coordinate is either the same authoritative physical column, is
   uniquely reconstructed from constrained columns, or is proved to be an
   algebraic consequence of retained rows.
5. A Rust drift test checks the exact artifact, layout, selectors, and public
   columns.

A digest can identify an artifact. It cannot prove row authority or replace
these conditions. A local artifact-checked theorem is also not a complete
Rust-conformance theorem until the recursive and terminal bridges cover every
generated row.

## Remaining production blockers

### P0-1: Complete recursive and terminal relation conformance

Problem:

The current formal artifacts cover local gadgets, manifests, row projections,
the new branch map, the exact binary and radix-four canonical-X leaves, and
complete selector-port coverage for the radix-four candidate. They do not
cover the arithmetic content of the complete generated production recursive
relation or the terminal relation.

The complete source relation cannot become a literal Lean list. The current
generated schedule fixes the recursive source relation at 16,407,566 rows and
16,237,141 columns. These are exact geometry values, not an arithmetic
refinement. The current source-stage certificate below is the structural
authority for the branch-private field suffix.

The new source-stage certificate covers the branch-private suffix after the
public and shared prefixes. This is why its 16,181,176 fields are fewer than
the complete recursive source relation's 16,237,141 columns. The public and
shared prefixes retain separate compiler owners; neither number replaces the
other.

The exact affine-definition census is smaller but is still too large for one
proof item per term. The recursive source-stage artifact has 86,880 linear
definitions. A production certificate must group these definitions by compiler
family and prove each family with a generic polynomial theorem.

Required fix:

1. Keep the new 14-owner source-stage artifact as the outer coverage manifest.
   It must not be treated as arithmetic authority.
2. Split PiRLC's 6,420 physical stages into exact repeated compiler templates.
   Each template certificate must include its exact stage count, parameters,
   source interval formula, decoder dispositions, row formula, and final port
   formula.
3. Define the same template certificates for PiCCS and the remaining twelve
   owners. Do not emit one Lean record per source field, decoder run, or
   physical stage.
4. Generate each certificate from the same Rust emitter that builds the verifier
   relation.
5. Give Lean one executable degree-bounded polynomial kernel and one semantic
   theorem for each compiler family.
6. Complete both directions for the bounded grouped-product fixture: exact
   source rows to final rows, and accepted final rows to reconstructed exact
   source rows.
7. Prove whole-relation same-assignment soundness and completeness against the
   Lean F′ model, with exact family coverage and no overlap or remainder.
8. Generate and check the terminal relation in the same way.
9. Add a drift test that fails on any unreviewed Rust relation change.

Acceptance test:

- The generated recursive and terminal artifacts rebuild in CI.
- Lean proves both refinements without trusted holes.
- Rust and Lean agree on the exact public layout, selectors, rows, columns,
  polynomial, and program binding.

### P0-2: Exercise all lifecycle branches at production scale

Status:

The new ignored Metal test executes base, bootstrap recursion, steady
recursion, and terminal finalization in one proof. It passes locally on the
Apple M5 Max inside the five-minute cap. It uses an enlarged test fixture at
`ell = 24`. Its latest proof takes 32.112 seconds and its verification takes
4.210 seconds. The separate exact production census is the authority for the
`ell = 24` production-width size claim.

Remaining problem:

The test is not a CI release gate. It does not use the production memory width
or the 1,088-fold endurance schedule. It does not yet run a CPU differential
oracle, and it does not include the required branch and delayed-lane mutation
corpus. The current non-ignored `nebula_segment` target also has one independent
fixture failure: it requests 116 statistical-security bits, while its current
field and coordinate-fork census supplies 114 bits.

Required fix:

1. Run the frozen production-width candidate through the required 1,088-fold
   endurance schedule.
2. Promote the reduced all-branch test to an Apple CI release lane with an
   explicit time budget.
3. Add negative tests for the wrong recursive selector, wrong prior public
   digest, wrong accumulator digest, and wrong delayed Nebula lane.
4. Run the CPU path as a differential oracle on a smaller profile.
5. Align the segment fixture with an approved security profile or increase its
   coordinate-fork census. Do not lower the target without a security review.

Acceptance test:

- Both backends produce proofs accepted by the same verifier.
- Every listed mutation is rejected.
- The production-width endurance receipt fixes the profile, fold count, device,
  rows, columns, proof time, verification time, and peak RSS.
- The test finishes inside the five-minute non-Lean limit.

### P0-3: Finish the loadable artifact lifecycle

Status:

The relation is per profile. Program-specific values do not change its matrix
coefficients. The verifier recomputes the carried Poseidon2 binding from exact
authoritative inputs. The evaluator and encoder artifacts restore the complete
online prepared profile without relation synthesis. Parallel load plus profile
restoration takes 4.10–4.67 seconds. Two later program binds take 152–157 ms,
share the same relation and cache, and reject cross-program proof reuse.

The production-width radix-four candidate now has the same artifact lifecycle.
Its checked parallel load and restore take 3.843 seconds, and a later program
bind takes 60.5 ms. This is 1.157 seconds below the five-second gate. The restore
path requires the verifier-owned Ajtai setup to be installed first. The
benchmark uses seeded test setup. Production setup is not frozen yet.

Remaining problem:

The core online lifecycle is implemented, but deployment still lacks one
frozen profile manifest. The current receipts bind artifact bytes, schema,
relation shape, matrix digest, encoder dimensions, and field-native arm shapes.
The loaders apply explicit expansion limits. The verifier key separately binds
the polynomial-derived structure digest and Ajtai setup digest. These pieces
must be stored under one reviewed manifest that also fixes parameters, profile
limits, compiler identity, and public layout. Artifact-backed Spartan
finalization is still unavailable because the terminal decider compiler reads
materialized matrices.

Required fix:

1. Define one profile manifest that binds parameters, compiler identity,
   profile limits, polynomial, public layout, both artifact receipts, and the
   exact Ajtai setup receipt.
2. Install the manifest-owned Ajtai setup before public artifact restoration.
3. Add cross-profile and cross-compiler substitution tests for that manifest.
4. Extend the compact matrix authority to the terminal decider compiler before
   enabling artifact-backed Spartan finalization.
5. Treat digests only as integrity checks; the configured manifest and exact
   verifier inputs remain the authority.

Acceptance test:

- The stored artifact pair loads, then binds two different programs under the
  same profile through the existing 500 ms path. This now passes locally.
- Relation, PiCCS header, and Ajtai setup receipts match live construction.
- Program plan, `D_init`, and binding receipts differ as expected.
- Cross-profile, cross-compiler, corrupt, truncated, and binding-substitution
  artifacts are rejected.
- Total parallel artifact load plus prepared-profile restoration is below five
  seconds, and per-program binding is below 500 ms. This now passes locally.

### P0-4: Production verifier and parser hardening

Problem:

The benchmark proves an honest in-process proof object. The artifact readers
now reject corruption, a different matrix identity, excessive byte size, and
excessive expanded encoder dimensions before allocation. Proof, transcript,
and public-input parsers still need the full hostile corpus.

Required fix:

1. Freeze and version proof and verifier-key encodings.
2. Add explicit length and allocation bounds before every remaining proof and
   verifier decode allocation.
3. Fuzz proof, artifact, transcript, and opening decoders.
4. Add truncation, oversized-length, duplicate-field, non-canonical-field, and
   cross-profile test corpora.
5. Confirm that verifier failure returns an error and does not panic.

Acceptance test:

- The fuzz corpus runs under memory and time limits.
- No malformed input panics, allocates outside the limit, or reaches a verifier
  path with unchecked dimensions.

## Performance blockers

### P1-1: Select and validate the radix-four `2^24` relation

The selected radix-two profile has 9,304,520 rows and 25,870,482 columns, so it
still selects `2^25`. The corrected complete radix-four candidate has
8,102,331 rows and 12,288,726 columns. It selects `2^24` with 4,488,490
columns of margin. The numerical width target is met for the candidate, but
production selection is not complete. The production census fails closed if
rows or columns exceed `2^24`, or if the joint domain changes.

The reduced-memory all-branch Metal proof now passes. The exact
production-width evaluator and encoder artifacts also load and restore inside
the release time gates. These are separate receipts. There is still no
production-width, 1,088-fold proof receipt. The specialized radix-four shader
now passes the complete Metal adapter suite after its missing selector
correction terms were restored.

The exact radix-four recursive width owners are:

| Owner | Allocated coordinates |
|---|---:|
| PiCCS | 6,687,893 |
| PiRLC | 3,275,198 |
| Accumulator authority | 810,186 |
| Nebula | 246,100 |
| Application | 169,110 |
| Prelude and transcript | 141,057 |
| Output and prior link | 99,768 |
| Both PiDEC checks | 72,900 |
| Counters | 176 |
| Exclusive recursive allocation | 11,502,388 |
| Derived-product encodings | 730,296 |
| Complete recursive branch | 12,232,684 |

PiCCS and PiRLC own 9,963,091 coordinates, or 86.6% of the exclusive
recursive allocation. The application owns 1.5%. This is evidence that the
main width is verifier representation and lowering cost, not an application
or security lower bound.

The production candidate also passes an exact physical row-coverage check.
The final matrices contain 185,526 compiler owner runs, of which 180,665 are
nonempty. Every nonempty owner run has exactly one matching selector-gate run,
and the final sparse polynomial has the expected 74 terms. This check rejects
row gaps, overlaps, wrong selector intervals, wrong selector columns, and
non-unit selector coefficients. It proves physical coverage only. It does not
prove that a row family implements its intended Nebula equation, and it does
not justify row removal.

The reduction order must preserve the protocol boundary. First, replace exact
copies with aliases and fuse R1CS intermediates into direct CCS equations while
keeping one F′ transition and the same transcript. Prove a generic Lean
refinement for each compiler rewrite, then prove the generated Rust relation is
an instance of that compiler. Only if this remains too wide should the protocol
split verification across recursive micro-steps. Such staging needs a new Lean
lifecycle invariant and a security reduction; a carried digest alone is not
authority.

The selected radix-two recursive branch owns:

- 27,164,726 source columns;
- 13,833,463 eliminated source columns;
- 12,855,060 retained unit columns;
- 475,710 retained balanced fields;
- 8,572,100 decomposition or equality aliases;
- 2,033,600 derived-product columns;
- 23,780,822 total recursive branch columns.

Removing every derived-product column would close only 22.4% of the required
gap. At least 7,059,666 more columns must still be removed or shared. Thus,
derived products alone could not make the radix-two circuit fit `2^24`. The
radix-four parameter change closes the full gap in the generated candidate.

Required fix:

1. Run base, bootstrap recursion, steady recursion, terminal finalization, and
   the 1,088-fold endurance schedule at the frozen production width on Metal.
2. Add candidate mutation tests for non-unit limbs, wrong reconstruction,
   transcript changes, and cross-program substitution.
3. Measure peak RSS for the persisted candidate artifacts and the production
   proof. Artifact size, cold restore, and program binding are now measured.
4. Generate complete recursive and terminal row artifacts and prove
   same-assignment soundness and completeness in Lean.
5. Select and freeze radix four only after those gates pass. If proof time or
   the 4,488,490-column margin is not sufficient, measure a direct degree-eight
   CCS form before a deferred pipeline.

Batching does not reduce the circuit domain. It changes how often the circuit
is invoked. A full Poseidon2 pass over the running state is also not a proved
lower bound: a selective binding or a proved physical alias can bind the same
authority with a different cost.

Do not replace the selective SIS output binding with direct Poseidon2. For
23,033 fields, direct Poseidon2 would cost about 2.05 million rows. The active
21-row selective opening path costs about 486 thousand rows for the same field
count.

Acceptance test:

- Both axes are at most 16,777,216.
- The benchmark reports `ell = 24`.
- The production-width endurance proof and verification pass on Metal.
- The exact generated leaf artifact, full recursive artifact, terminal
  artifact, and Lean refinements pass.
- Cold restore is below five seconds and repeated program binding is below
  500 ms for the selected candidate.

### P1-2: Evaluator cache gate is closed

The production evaluator artifact is 460.629 MiB, which is 51.371 MiB below
the release gate. It contains 41,286,880 row blocks. Signed singletons account
for 39,911,332 of them, or 96.7%, and each now uses one four-byte reference.
The remaining 1,375,548 occurrences use checked dense side references.

The required regression gate is now exact: the 24 CPU evaluator-equivalence
tests, four artifact tests, targeted Metal parity regression, and complete
Metal adapter suite must pass after any cache schema or decoder change. A
production artifact at or above 512 MiB must fail the release gate.

### P1-3: Cold restore gate is closed; manifest work remains

Live construction takes 15.7–16.9 seconds. Its measured work is:

- WASM program binding: 1.056 seconds, including 0.909 seconds for the
  canonical application shape and 0.124 seconds for the memory plan;
- two fixed-point synthesis and selective-layout rounds: 6.503 seconds;
- final selective audit, row emission, and matrix construction: about 3.280
  seconds;
- optimized evaluator-cache construction: 5.86 seconds;
- matrix Poseidon2 digest: 5.78 seconds, run in parallel with cache construction.

The evaluator artifact is 506.822 MiB. The exact encoder artifact uses affine
field runs and compact indexes and is 49.825 MiB. The recursive encoder has
13,470,669 source fields represented by 4,028,499 runs; its derived-product
section has 26,480 values. Both readers check Poseidon2 receipts and explicit
expansion limits. Parallel load takes 3.00–3.58 seconds. The remaining WASM
profile and verifier-policy restoration takes 1.10 seconds. The complete cold
profile therefore takes 4.10–4.67 seconds.

The corrected radix-four production candidate takes 16.879–17.122 seconds for
live relation and cache construction. This work now has an offline, reusable
artifact path. The evaluator artifact is 460.629 MiB, and the encoder artifact
is 63.920 MiB. Parallel checked load takes 3.159 seconds. Complete restore
takes another 0.685 seconds. The combined 3.843 seconds has 1.157 seconds of
margin. A later program bind takes 60.5 ms. The complete artifact run peaks at
10,906,304 KiB of descendant RSS. The remaining deployment work is the frozen
manifest, production Ajtai setup ownership, and explicit release RSS limits.

The live-build phase split explains the delay:

- three fixed-point synthesis and lowering rounds take about 9.84 seconds;
- the final audit and emission take about 2.12 seconds;
- evaluator-cache construction and its matrix receipt take about 3.8 seconds;
- profile and program preparation use the remaining time.

The fixed-point loop builds the large recursive compiler three times until its
shape is stable. This is valid work, but it is not needed for each program or
deployment. Artifact restore skips it. For live compiler work, the next fix is
one generated fixed-point shape certificate that lets an unchanged compiler
reuse the frozen shape and fails closed on drift.

Required fix:

1. Freeze the P0-3 profile manifest and keep both artifact loads parallel.
2. Preserve the shape-only matrix header. Do not restore raw CSC beside the
   loaded cache.
3. Measure load-only and proof-only peak RSS separately. The current
   10,906,304 KiB receipt includes live construction and artifact operations;
   it is not a proof-execution budget.
4. Keep the exact radix-four artifact receipt in the release benchmark and add
   peak-RSS limits.

Acceptance test:

- Parallel artifact load plus complete prepared-profile restoration is below
  five seconds. This now passes locally.
- Per-program binding is below 500 milliseconds. This now passes locally.
- Artifact size, peak RSS, and load time have explicit release budgets.

### P1-4: The current Metal proof regressed to about 9 seconds

Problem:

Two consecutive runs took 8.917 and 9.193 seconds. The earlier narrow-profile
measurement took 2.784 seconds. The relation dimensions changed by only 1,044
rows and 21,492 columns, so the dimension change alone does not explain a
three-times proof-time increase. The memory geometry also changed, so the old
and new runs are not a controlled A/B test.

Artifact-backed proof runs now span 8.78–9.39 seconds. The latest phase split
is:

- base and recursive F′ witness synthesis take 0.215 and 1.209 seconds;
- the first NIFS takes about 2.67 seconds: 1.57 PiCCS, 0.07 PiRLC, and 1.03
  PiDEC;
- terminal NIFS takes about 5.08 seconds: 2.81 PiCCS, 0.31 PiRLC, and 1.96
  PiDEC;
- terminal PiDEC alone spends 0.30 seconds splitting, 1.04 seconds on child
  commitments, and 0.58 seconds on child openings;
- verification takes 2.092 seconds. Its final witness-authority forms and
  commitments take about 0.57 and 1.00 seconds.

This isolates the current online cost. Terminal PiCCS and PiDEC are the first
targets. It does not yet explain the historical regression because the old
profile has not been run under the same code and device conditions.

Required fix:

1. Run the old and fixed-ROM geometries under the same binary, parameters,
   device, and fold schedule.
2. Profile terminal PiCCS oracle construction and terminal PiDEC child
   commitments and openings first.
3. Confirm that no compact data is expanded during an online fold.
4. Add a benchmark receipt for each fold type and fail CI on a reviewed
   regression budget.

Acceptance test:

- The cause of the regression is isolated with per-phase evidence.
- The selected production profile has an explicit proof-time budget.
- Repeated runs on the same Apple CI host stay inside that budget.

### P1-5: Metal is in the verifier trusted computing base

`verify_with_witness_opening_backend` treats the opening backend as trusted.
The new compact-run kernels have focused CPU parity tests, but production needs
broader backend assurance.

Required fix:

1. Differentially compare CPU and Metal forms for randomized valid matrices,
   compact runs, seeded blocks, and malformed geometry.
2. Add bounds tests for every buffer, offset, segment index, and dispatch size.
3. Run sanitizers on the host-side buffer construction.
4. Keep a verifier mode that recomputes a sampled set of Metal results on the
   CPU for deployment diagnostics.

Acceptance test:

- Randomized differential tests pass on every supported Apple GPU family.
- Malformed geometry fails before dispatch.
- No shader reads or writes outside a declared buffer.

## Validation performed

Latest checks on this tree:

| Check | Result |
|---|---|
| Compile every `neo-fold-clean` test target in release mode | pass |
| Radix-four accumulator compact-codec regression | pass |
| Radix-four digit-mask alphabet and plane-order regressions | pass |
| Production-width radix-four selector coverage: Rust drift owner and focused Lean refinement | pass |
| Production-width radix-four source-stage coverage: Rust drift owner and no-axiom Lean aggregate checker | pass: 18.23 s live census, 78-line artifact |
| Bounded grouped-product Rust source/final row fixture | pass |
| Bounded grouped-product Lean polynomial identities and forward same-assignment refinement | pass: 3.10 s focused test and axiom gate |
| Curated `Nightstream.Implementation` build with the forward refinement | pass: 7.26 s, 3,856,320 KiB peak descendant RSS |
| Compact Poseidon Rust artifact target | pass: 3 tests, 46.38 s including release compilation |
| Compact Poseidon2 same-assignment refinement and focused axiom target | pass: 0.78 s, 752,064 KiB peak descendant RSS |
| Final `Nightstream.Implementation` build | pass: 51.69 s, 3,866,080 KiB peak descendant RSS |
| Draft grouped-product reverse refinement | fail: carry rewrite does not match after decoded-step simplification; 272.26 s, 4,863,728 KiB peak descendant RSS |
| Exact-width rejection before the zero initial-memory shortcut | pass |
| Cached zero-leaf `D_init` against the uncached formula | pass |
| Complete non-ignored `nebula_segment` target | fail: fixture supplies 114 bits against a 116-bit target |
| Exact production-width radix-four artifact write, load, restore, and bind | pass |
| Reduced base, bootstrap, steady, terminal, and verify Metal lifecycle | pass |
| Selected 74-term PiCCS Rust receipt drift test | pass: 1 test; independent Lean acceptance and mutation rejection pass |
| Final complete Lean build | pass: 5,610 jobs, 25.09 s, 4,040,544 KiB peak descendant RSS |
| Final Lean axiom gate | pass: 4,631 jobs, 1.82 s, 694,816 KiB peak descendant RSS |
| Final Lean executable checker | pass: 5.18 s, 1,426,512 KiB peak descendant RSS |
| Final Lean static gate | pass: imports, generated layout, ownership, forbidden holes, assurance data, size, and `.expected` checks |
| Final Rust-origin gate | pass: step, terminal, relation artifact, mutation corpora, and Lean replays |
| `git diff --check` | pass |
| Diff in Enzo's `memory_routing.rs` | none |

The named phases now pass when run separately. The combined
`./scripts/validate.sh all` command is still not a clean release receipt from
this shared workspace because earlier runs ended after an external signal.
Clean CI must run the combined gate on the frozen change set.

The Lean radix-four canonical-X result is Rust-conformant for its 6,480-row
leaf artifact. `FPRIME-R4-SELECTOR-COVERAGE` is Rust-conformant for selector
activation across all candidate rows. Neither property proves the complete
recursive or terminal relation, so the same-assignment artifacts are still
absent.

## Release gates

A production release should require all of these gates:

1. Complete recursive and terminal Rust-to-Lean relation refinement.
2. The passing all-branch Metal proof promoted to a repeatable Apple CI gate.
3. Negative lifecycle, transcript, program-binding, and artifact-substitution
   tests.
4. A versioned verifier-owned, loadable per-profile artifact with exact
   per-program binding.
5. Select the measured radix-four candidate only after its production-width
   endurance proof and conformance gates pass. Then confirm `ell <= 24` on the
   frozen profile.
6. Cold artifact load and complete prepared-profile setup below 5 seconds,
   repeated per-program binding below 500 ms, and a reviewed proof-time budget.
7. Evaluator cache below 512 MiB. The corrected radix-four artifact is
   460.629 MiB, so this gate is met. Also require documented load-only and
   proof-only peak-RSS budgets.
8. Proof and verifier serialization fuzzing with explicit allocation limits.
9. CPU/Metal differential tests on supported hardware.
10. Reproducible benchmark receipts in CI with fixed parameters, profile,
    device, row count, column count, cache size, proof time, and verify time.

## Recommended order

1. Run the radix-four production-width and 1,088-fold endurance schedule on
   Metal. The reduced all-branch lifecycle already passes. Add wrong-limb,
   wrong-reconstruction, transcript, and cross-program mutation tests.
2. Freeze the candidate artifact manifest and production Ajtai setup. Artifact
   size, cold restore, and program binding are measured. Add peak-RSS limits.
3. Freeze complete recursive and terminal artifacts, then prove whole-relation
   same-assignment conformance.
4. Select and freeze radix four if these gates pass. The exact generated census
   already confirms `ell = 24`.
5. If proof time or the 4,488,490-column margin is not sufficient, build a direct
   degree-eight CCS prototype for the largest PiCCS paths.
6. Define a deferred-verification pipeline in Lean only if the direct form is
   still not sufficient. Do not use a simple multi-step phase split.
7. Promote the all-branch Metal test and its mutation corpus to Apple CI.
8. Freeze the combined profile manifest and extend compact matrix authority to
   the terminal decider compiler.
9. Harden serialization, resource limits, and the Metal verifier backend.

This order keeps soundness work ahead of the final deployment manifest. The
artifact schemas can still change while the relation layout is under active
reduction.
