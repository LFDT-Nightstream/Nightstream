# Constraint reduction

## Active goal

Owner update, 2026-09-22: **substantially decrease the number of constraints,
and run the experiments as fast as possible.** Experiment speed is a
requirement of this work. A proving-time comparison is not the current goal.

On 2026-09-23, the owner explicitly authorized committing and pushing this
tested change, overriding the historical no-commit instruction for this task.

Preserve the proved 27.13% checkpoint. Investigate the additional 50%
committed-coordinate target: **184,359,564 to at most 92,179,782**. This is a
research target, not an assumed feasible result. Track committed coordinates,
logical rows, and matrix nonzero entries separately. Moving cost between
these measures does not establish an overall improvement.

Keep the specification, soundness, constructive completeness, witness maps,
security assumptions, Goldilocks, Poseidon2, and the exact production profile
`b = 2`, `k_rho = 16`, `B = 65536` unchanged.

For each experiment batch:

- Choose the fastest sound way to resolve its specific open claim. Reuse
  prior results when the relevant source and assumptions are unchanged.
- Use exact cost calculations and small cvc5 queries to reject bad candidates
  before large proof or integration work. Check counterexamples independently.
- Check emitter input roles, changed block indices, and alignment counts
  before full fixture runs. Derive the verifier-context digest from its Lean
  descriptor; it is distinct from the verification-key digest.
- Run the affected Lean targets and axiom audits. Batch dependent edits before
  full library and audit checks; repeat broad checks only after a stable batch
  or when a concrete failure requires them. `validate.sh axioms` builds
  `tests.Axioms`, which imports the production `NightstreamFPrime` root; one
  successful run covers that library and the audit targets. Do not precede it
  with a duplicate full-library build on the same source.
- Keep incremental build products and reuse common preparation. Use the
  existing parallel build while keeping one Lean or Rust build process.
- Search papers, primary documentation, blogs, and source repositories for
  applicable ideas. Use focused dependency queries instead of broad graph
  exports. New tools must address an actual open constraint or proof problem.
- Remove demonstrated repeated work from the experiment loop. Do not start
  a separate timing-comparison task in place of constraint reduction.

Continue authorized Lean work after a successful batch. Paused benchmarks
are not a blocker for that work. Keep package and fixture regeneration and
Rust benchmarks paused until the chosen layout and proofs are stable. Then
complete the required production integration and its checks.

## Preserved checkpoint and current candidate

Status: the 27.1339% Lean carrier reduction and complete canonical direct
producer proof are saved in signed checkpoint `6bcdbb7c`. The latest full
library build
passed 4,020 jobs in 130 seconds; the full axiom/test build passed 4,207 jobs
in 100 seconds, including 71 new public theorem audits. The reduced geometry
and carrier inequality also passed their earlier axiom audit.
The quotient package and its supporting artifacts are now installed with new
Rust identity pins. The selected Rust core and lifecycle use the direct
producer; changed packages keep the full fallback. Independent assignment,
rejection, commitment, pilot, binding, and identity checks passed. Actual NIFS
regeneration and the complete independent comparison passed; native and Lean
base/recursive fixtures are published. Final Nightstream tests passed.
All correctness and integration checks for this quotient checkpoint are
complete. The owner stopped timing comparisons and directed further work to
constraint reduction. The candidate lifecycle benchmark was not started and
is not a blocker for this research. No runtime improvement is claimed.

The selected Lean layout now includes the shared running-transition flag and
Poseidon output-pin removal. It proves **172,217,934 committed coordinates**,
**172,217,903 logical coordinates**, and **4,147,335 logical rows**. The full
retained-assignment transport, arbitrary-assignment soundness, constructive
witness construction, strict norm bound, and exact matrix-program comparison
passed the combined production-library and axiom target (4,224 jobs). Static
checks passed. The fixed profile and semantic specification are unchanged.

The transition uses 42 local coordinates and 49,359 rows. Its assignment
schedule contains 31 blocks. Block opcodes follow that canonical order;
the shared flag is opcode 14, and the output-digest block is opcode 24.
The compact transition no longer requires the old R1CS row accessor.

After this stable Lean batch, affected packages and parity fixtures were
regenerated from Lean. Selected Rust conformance passed. The new package has
SHA-256 `fc3d8a8e798fde5ebabd388c64d5caa3d29116cfac1bb38a7142e888cdd66c6a`;
this identifies bytes and is not proof authority. Staging records and the
replaced checkpoint artifacts are in `/tmp/nightstream-running-flag-sGbJD2`.
The signed quotient checkpoint remains in Git. Prover benchmarks remain paused.

The measured matrix count is **2,968,490,185**. The independent Rust
comparison checked all 4,147,335 logical rows in all fourteen matrices and
matched the Lean plan exactly. All three measures are below the quotient
checkpoint. Matrix nonzeros remain above the original baseline; no overall
performance improvement is claimed. The additional 50% coordinate target is
not established. The
Poseidon S-box block alone still contains 114,443,382 coordinates, and the
tested encoding classes do not provide the required reduction. This is not
a proof of global impossibility. See [the remaining budget](remaining-witness-budget.md)
and [the shared-flag proof record](running-transition-next-candidate.md).

## Selected shared-flag integration evidence

The selected Rust path passed the complete independent logical-matrix and
physical-matrix comparisons, every retained-coordinate comparison, all
logical rows, 305 assignment mutation controls, the three derived-recipe
rejections, and the matrix order/column/coefficient rejection controls.
Direct CCS and full witness construction agree. Changed packages retain the
full validation path. Setup, sparse commitment, pilot, binding, and loader
checks passed.

The new actual C → R → D fixture passed the normal verifier. Its six active
child openings use three checked batches. Independent Lean outputs match
all phase values, children, transcript, and the 945,983-byte proof. All 43
NIFS and 55 PiDEC mutation cases reject. The four native/Lean NIFS and
recursive artifacts are published. The
[fixture record](../../../crates/neo-fold-clean/tests/nifs/fixtures/stage1_actual_nifs/README.md)
contains their hashes and exact scope.

Nightstream's complete package assembly, saved proof/transcript, recursive
successor assignment, detached-input rejection, and key-prefix authority
checks passed. The key-prefix test first used the old width; the five other
active lifecycle tests passed, and the corrected key-prefix test then passed
its focused rerun. The three existing full lifecycle tests stayed ignored;
no lifecycle benchmark or new nonzero-prior NIFS execution is claimed.

Initial integration failures exposed a wrong fixture context argument and
stale padding/output-block/key-prefix test constants. These were corrected;
the affected checks passed. Their failed logs remain in the staging record.
No Lean specification, profile, security assumption, or rejection requirement
was weakened. Formatting and the final diff check passed. The external
review files were absent at the required 00:20 UTC check on 2026-09-23.

## Contract

The initial, achieved target was a reduction of at least 25% from the selected
253,011,276-coordinate committed carrier. Whole Phi81 blocks have 54
coordinates, so the largest aligned carrier that meets this request is
189,758,430 coordinates.

Keep the semantic specification, soundness, completeness, efficient witness
extraction, and existing security assumptions. Keep Goldilocks, Poseidon2,
`b = 2`, `k_rho = 16`, and `B = 65536`. Final production integration must
connect the chosen layout to the canonical package and Rust consumer.
The completed proof checkpoint has passed its selected Rust integration
checks. The separate
additional 50% coordinate target below remains a research target.

Baseline source: `469d12e2dc01a7cd236aaef86950f0708260041f`.
`Poseidon2HashChainV1Package.logicalWidth` gives 253,011,231 logical
coordinates and `structuralRowCount` gives 6,377,559 rows. The selected
carrier has 45 alignment coordinates. The reference application has 7,700
rows. The row cube remains `2^28`.

## Selected reduction: exact quotient checks for Phi81 products

For each existing product over `F[X]/(X^54 + X^27 + 1)`, let `A` and `B`
be its coefficient polynomials and let `H` be output minus prior accumulator.
Retain a 54-coefficient quotient witness `Q`. For each integer point from
0 through 107, constrain

```text
A(t) * B(t) = H(t) + (t^54 + t^27 + 1) * Q(t).
```

The difference polynomial has degree at most 107. These 108 points are
distinct in Goldilocks. Thus all checks imply the polynomial identity and
the exact existing ring product. This is a deterministic argument. It uses
no new transcript challenge or probability bound.

Completeness uses the monic quotient of `A*B`; it has degree at most 52.
Its 54th coefficient is zero. Lean proves that the executable quotient recipe
equals that mathematical quotient. All quotient coefficients use the
existing 41-coordinate field encoding, so the committed norm remains below
two. Existing output values and the two base-field components of extension
values keep their exact meanings.

The schedule has 52,326 output lanes, or 969 full ring products. The baseline
retained 33 field values per lane. The reduced layout retains one per lane
and checks two points per lane. Lean proves these geometry values:

| Measure | Baseline | Reduced Lean layout |
|---|---:|---:|
| Product auxiliary field values | 1,726,758 | 52,326 |
| Product rows | 1,779,084 | 104,652 |
| Complete logical rows | 6,377,559 | 4,703,127 |
| Physical R1CS rows | 29,225,729 | 29,225,729 |
| Complete logical coordinates | 253,011,231 | 184,359,519 |
| Aligned carrier coordinates | 253,011,276 | 184,359,564 |
| Fixed domain | `2^28` | `2^28` |

The carrier reduction is 68,651,712 coordinates, or 27.1339%. The added
carrier inequality theorem establishes the requested reduction of at least
25%. Its axiom audit passed. Candidate decoding and complete independent
matrix comparison confirmed these counts. The same package bytes are now
installed for the current integration checks.

Complete logical rows decrease by 26.2551%. Physical R1CS rows and the
fixed domain remain unchanged.

Evaluation rows have denser affine forms than the baseline grouped products.
The smaller witness does not establish lower execution cost. The complete
candidate comparison counted 3,001,571,645 logical matrix nonzero entries:
`[229698543, 4598475, 336477167, 83572300, 844999263, 1502121245, 0,
104652, 0, 0, 0, 0, 0, 0]`. Physical A/B/C nonzeros are
`[93701820, 39358148, 28868018]`. The saved baseline conformance log
counts 2,335,822,475 logical matrix nonzeros. Its input file has the exact
selected baseline artifact hash `043bce25083eb15c733a903f4df2958acbc31a59dbe17420cd084c8242bd4d1f`
and matching structural/package identities. Thus logical nonzeros increase
by 665,749,170, or 28.5017%. `checkpoint-metrics.json` preserves both vectors.
No before/after runtime comparison is complete; this is not an overall
performance improvement claim.

## Research basis

- [SuperNeo, September 4, 2026](https://eprint.iacr.org/2026/242),
  sections 5 and 7: preserve coefficient embeddings, ring action, low-norm
  witnesses, and the complete composed reduction. The quotient circuit changes
  only the implementation of an existing ring multiplication.
- [HyperNova](https://eprint.iacr.org/2023/573), Theorem 1 and section 6:
  account for witness width, sparse matrix work, polynomial degree, and the
  complete augmented recursive circuit. Preserve efficient witness maps.
- [Satisfiability Modulo Finite Fields](https://eprint.iacr.org/2023/091):
  use exact prime-field equations for translation checks. Other encodings
  require a proved translation to the same field semantics.
- [Split Groebner Bases](https://eprint.iacr.org/2024/572): separate simple
  algebraic subsystems when possible. Solver cost depends on the equations,
  not just their count; `unknown` and timeout establish no equivalence.
- [Bounded Verification for Finite-Field-Blasting](https://theory.stanford.edu/~barrett/pubs/OWB%2B25.pdf):
  section 5.1 requires efficient witness maps in both directions. Its
  composition theorem connects local compiler steps, and its ZKP transfer
  argument uses the inverse witness map for knowledge soundness. Bounded
  solver checks do not establish a universal family theorem.
- [cvc5 finite-field implementation blog](https://cvc5.github.io/2024/01/26/adding-a-new-theory-to-cvc5.html):
  finite-field queries use a fixed prime, field operations, and equalities.
  A model or unsatisfiable core is search evidence, not a Lean proof.
- [CompPoly](https://github.com/Verified-zkEVM/CompPoly), at the repository's
  pinned revision `050f0bc7e9780703beb8d178ec533e52bd87d649`, already has
  proved NTT multiplication. An NTT circuit was considered; quotient checks
  need fewer retained values and no transform or interpolation weights.

## Evidence needed

1. Lean soundness and constructive completeness for the quotient checks,
   including the executable recipe and existing ring-product correspondence.
2. cvc5 positive and negative controls over Goldilocks, with independent
   replay. A 107-point negative control must expose the unchecked degree-107
   residual allowed by a 54-coefficient quotient.
3. Exact family wiring, low-norm assignment transport, full selected-step
   preservation, recursive dimensions, and the existing fixed-key prefix
   reduction to the approved MSIS instance.
4. Canonical emission, Rust matrix and assignment parity, rejection checks,
   identity checks, and measured execution on the same candidate.

Lean commands use the project's 1,500-second cap. Rust checks use the
300-second cap. The owner approved 1,200 seconds for the current two-step
baseline; it finished before the pause in further benchmarks. Preserve the
earlier timed-out attempt. No numerical security claim or global minimum
follows from this optimization.

## Focused checks completed

- `NightstreamFPrime.Spec.Phi81Relation.QuotientProduct`: exact soundness,
  completeness, accumulator form, and executable quotient correspondence.
- `NightstreamFPrime.Export.Stage1.PiRLCProductPlan`: unchanged source
  constraint soundness and canonical witness completeness.
- `NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramAllRows`: literal
  sparse-form equality for every port of all 104,652 product rows; 6 seconds.
- `NightstreamFPrime.Export.Stage1.CachedAssignmentProducts`: canonical and
  cached executable witness transport; 28 seconds including prerequisites.
- The cvc5 experiment and independent positive/negative replay passed; see
  `README.md` and `phi81_quotient.json` in this directory.
- `bash scripts/validate.sh static`: boundary, codec, and interface checks passed.
- `bash scripts/validate.sh build`: 4,018 jobs passed in 79 seconds, including
  the exact carrier inequality theorem.
- `bash scripts/validate.sh axioms`: 4,177 jobs passed in 26 seconds on the
  final incremental run. This includes shared-formula execution checks and
  the unchanged selected-step and assignment-completeness theorem audits.
  Only the existing allowed axioms occur: `propext`, `Classical.choice`, and
  `Quot.sound`.

The first full library attempts exposed old row readers, the product-row
soundness index, and the setup coefficient count. The first full axiom build
also exposed the old last-carrier address and commitment-row offset. These
were corrected; the final gates above passed. No proof assumption was added.

The four focused Phi81 tests and the transport ordering test passed. The
candidate's complete 14-matrix comparison passed in 42.43 seconds; physical
matrix comparison passed in 26.67 seconds; block-order, column, and coefficient
identity rejection checks passed in 75.81 seconds. The base fixture's entire
physical and logical assignment passed in 70.88 seconds. The final combined
assignment check reached its 300-second cap. Before that cap it passed all
4,703,127 rows, all 184,359,519 logical coordinates, 30 assignment-block
mutations, all 13 matrix-slot probes, the zero-matrix insertion check, 256
public-digest bit mutations, four digest-word mutations, and the Phi81 and
First54 recipe identity/equation rejection cases. The output-digest recipe
case was unverified at that checkpoint. The original combined command remains
a failed run; the separate current check below now closes the missing case.

Two old probes were repaired without reducing coverage: a centered-unit
value of zero changed by +1 is still valid, so probes now use each decoded
polynomial degree; the Phi81 recipe probe now targets the final sampler
state actually read by the quotient recipe. Mutation row traversal now
checks parallel ranges in canonical order and keeps the first failure.

The completed old-layout CPU baseline used two application steps, including
one active fold: preparation 21.399 s, base proving 17.132 s, active fold
226.013 s, verification 152.593 s, total 417.137 s. Peak RSS was
8,500,621,312 bytes. It verified the final state. No matched candidate
lifecycle benchmark has been recorded. `baseline-cpu.json` retains the
measured result.

The saved NIFS and recursive caller fixtures are published, and the final
Nightstream correctness and integration checks passed. The before/after
runtime comparison remains pending.

## Quotient checkpoint integration checks

The current emitter passed in 58 seconds. It produced exactly the same
128,098,903-byte package as the `8b7c07d8` candidate, with SHA-256
`6216d1f62250a58d073ecf0a908bd074d3834957ec5be3361620bbdeb5a97642`.
This hash records byte equality; it is not protocol authority. The package,
expanded reference, binding, setup, PiCCS/PiDEC/PiRLC/application parity, and
base fixture are installed. The three Rust identity pins now select this
package. Matrix counts remain unchanged.

The direct Rust producer is connected to the core and lifecycle. Its fast
path requires the complete selected structural identity. Other packages use
the full physical path. Current checks passed as follows:

| Check | Result and scope | Time |
|---|---|---:|
| Fresh pilot parity | New verifier context emitted | 16.077 s |
| Fresh PiCCS ownership sidecar | New structural identity emitted | 9.968 s |
| Fresh sparse commitment parity | New final carrier coordinate emitted | 8.183 s |
| Canonical identity | Emitted binding, saved binding, and Rust pins agree | 119.214 s |
| Scratch and First54 unit checks | Passed | 177 s compilation; 0 s tests |
| Full/direct agreement and changed-package fallback | Two tests passed, including error agreement | 72.05 s |
| Independent assignment and rejection checks | All 4,703,127 rows, 304 ordinary mutation controls, and final output-digest recipe case passed | 178.363 s |
| Sparse commitment | All 1,188 coefficients for three nonzero coordinates agree | 0.121 s |
| Pilot parity tests | Five passed; the existing external-input test remains ignored | 26.02 s |
| Production binding tests | Three passed | 70.94 s |
| Complete Nightstream assembly equality and schema-2 rejection | Passed | 5.11 s tests; 112 s compilation |
| Nightstream saved proof and transcript | Passed | 42.61 s |
| Nightstream recursive caller fields and detached-input rejection | Exact fields and rejection checks passed | 64.22 s |

The output-digest recipe mutation was rejected at row 3,864,779. The earlier
Phi81 and First54 recipe checks remain valid for their unchanged source.
The timed-out combined run above remains recorded as failed; it is not
relabeled as a pass. Its missing output-digest case now has a completed
separate check.

Actual NIFS regeneration completed under
`/tmp/nightstream-quotient-nifs-6bcdbb7c`. PiCCS proving and verification
passed in 195.254 seconds, the PiRLC parent in 76.306 seconds, and parent
replay/commitment/split checks in 166.380 seconds. Active children 0 through 5
had separately computed openings in 221.122, 222.475, 221.158, 203.917,
208.760, and 169.425 seconds. Complete assembly and verification passed in
213.997 seconds and rejected all 43 NIFS mutations.

The independent Lean base-NIFS result passed in 10.797 seconds; the recursive
caller fixture passed in 11.906 seconds. `check-owned-nifs` passed the complete
phase, child, transcript, and 945,983-byte proof comparison and rejected all
55 PiDEC mutation cases. The native result/proof and both Lean fixtures are
published. Their authoritative file hashes and source scope are in
[the fixture record](../../../crates/neo-fold-clean/tests/nifs/fixtures/stage1_actual_nifs/README.md).
The older evidence archive retains the old fixture and is not the source of
these outputs. Rust integration is still uncommitted at this update.

Final Nightstream tests passed, closing the required correctness and
integration checks for this checkpoint. The candidate lifecycle benchmark
remains pending. These stage timings do not establish an overall speed or
memory improvement.

## Shared child-opening preparation

The separate workflow change replays the authoritative sources, computes the
canonical split, and builds the matrix cache once for a requested child batch.
Each saved digit still requires exact split equality, its recomputed activity
flag, and its recomputed commitment. The public command is:

```text
generate_pi_ccs_fixture open-owned-children PKG BASE PARENT MATERIAL 0,5 FRESH_OUTPUT
```

The measured batch contains only children `[0, 5]`. Their separate producer
times were 221.121931749 s and 169.425220528 s, or 390.547152277 s combined.
The shared producer finished in 223.31805613 s: 167.229096147 s (42.8192%)
less for this pair. GNU wall time was 226.75 s and peak RSS was 7,251,272 KiB
(6.9154 GiB). No corresponding before-RSS measurement is available.
Both complete output files match their existing references byte for byte:
203,892 bytes for child 0 and 199,679 bytes for child 5. The three focused
tests passed in 0.03 s after 24.57 s test compilation; the generator build
took 34.22 s. The native run used the project's 300-second cap.

[workflow-batching.json](workflow-batching.json) records exact paths, timings,
and output checks. This observation does not establish a result for all six
children or other batches, and it sets no default batch size. The circuit,
proof, and published artifacts are unchanged. The quotient lifecycle
benchmark remains pending.

## Next research contract

The owner requested another 50% reduction from 184,359,564 to at most
92,179,782 committed coordinates. This is a research target, not an established
result. The first requirement was a direct CCS witness construction for the
optimized ring products with proved witness mappings before removing the
related R1CS work. The complete selected Lean producer now meets that proof
requirement. Its Rust core and lifecycle port is in place, and the integration
checks above passed. A complete compiler rewrite is outside
this task.

Use the actual remaining witness budget, starting with the largest costs and
Poseidon blocks. Search current papers, arXiv, blogs, webpages, and GitHub.
Use cvc5 for candidates and counterexamples; use Lean for soundness,
constructive completeness, and witness mappings under the unchanged
specification, security assumptions, and profile. Track coordinates, rows,
and matrix nonzeros separately. Moving cost between these measures is not an
overall performance improvement.

The bounded Lean research batch is complete. Complete the affected artifacts,
Rust integration, and proving-time/peak-memory measurements for this proved
checkpoint. If the extra 50% target appears infeasible, report the achieved reduction and concrete
obstacles without weakening these requirements.

## Checked research batch

The direct PiRLC constructor evaluates only the required output recipes for
all 52,326 canonical invocations. Lean proves successful reconstruction of
the old scratch rows and equality outside the 7,848,900-column scratch
interval. The custody proof covers all retained allocation blocks and proves
equal complete CCS assignments and public digests at the phase boundary.
No caller premise supplies scratch equality or successful full execution.

The later producer proof below extends this phase result to the actual sorted
canonical event loop. It derives event provenance and actual recipe/hint read
support, including PiDEC sign hints and sampler canonical-u64 batches, and
uses the existing array dispatcher. Constraint support alone was not used as
a substitute for witness-read support. No stronger caller assumption was added.

The cvc5 research rejects two restricted candidates: 40 independent ternary
coordinates per arbitrary field value, and elimination of an adjacent scalar
S-box pair through a nonzero bivariate equation of total degree below nine.
Ten Lean research theorems establish the count bounds, exact counterexample,
and coefficient/root argument without solver trust. These are local bounds;
they do not prove a global lower bound for Poseidon circuits.

At this earlier research checkpoint, static checks passed. The full Lean
library passed with 4,019 jobs in 38 s. The full axiom/test build passed with
4,182 jobs in 7 s after registering the new research module in the explicit
test roots. All 38 new production theorems and ten research theorems passed
their axiom audits.

The achieved committed-coordinate reduction remains 27.1339%; no further
coordinate reduction or proving-cost improvement is established.
`remaining-witness-budget.md` records the actual costs and online sources.
The owner paused package and fixture regeneration, Rust integration, and
benchmarks at this earlier research checkpoint. Integration resumed after
the complete producer proof passed.

## Complete canonical producer proof

The checked producer path skips PiRLC product scratch construction in Lean
without changing the relation or package data:

- `StoredPhysicalPlan.EventSources` covers all eleven canonical source
  families. `Plan.rowEvents_induction` applies their properties to the
  unsorted event array. Its erased proof field is derived by `ofSources`.
  `ArraySortMembership` and `Plan.events_induction` now extend this property
  to the actual `Array.qsort` execution array.
- `PiRLCCombinationWitnessReadSupport` proves actual arithmetic-recipe and
  hint-source support for all canonical sampler/PiDEC/running witness
  batches, their application shift, and the selected Poseidon application.
  The generic application record receives no stronger assumption.
  The PiCCS, pilot, ordinary-instruction, hash/permutation, and First54 support
  owners close the remaining canonical event families, including every
  compact A/B/C check and output recipe.
- `StoredExecutionSupport` proves equal retained values and matching compact
  success/rejection for the existing array interpreters under exact read
  support and equal storage sizes. It covers writes, recipes, hints, ordinary
  instructions, permutations, and compact rows.
- `PiRLCCombinationInvocationOrigin` identifies canonical product templates
  with their exact descriptors and preserves scratch bounds and input
  separation through the application shift.
- The stored compact executor now proves size preservation and agreement
  outside local scratch after its initial output write. These theorems need
  only the existing successful execution premise.

[CanonicalDirectPhysicalExecution.execute](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/CanonicalDirectPhysicalExecution.lean)
starts from the caller-seeded array and executes the complete sorted plan.
It does not require a completed full physical witness. `execute_agree`
proves identical error results or successful arrays of equal size that agree
outside product scratch. All support facts come from the canonical schedule;
the theorem has no caller-supplied support or scratch-equality premise.
`successful_assignment_eq` then proves equality of the complete retained CCS
assignment and public digest for the selected Poseidon application.

The full physical executor remains the reference path. The stored replay
entry point uses the extracted full dispatcher and loop. The Rust core and
lifecycle now select the direct constructor for the exact selected identity,
with the full path as the fallback. The theorem closes the whole Lean producer
connection. The selected Rust integration checks also passed; the benchmark
remains separate and pending.

The earlier sorted-membership and dispatcher attempts stopped under the
three-round rule. The fresh continuation resolved both: it proves membership
for the existing sort and uses the actual dispatcher. Those earlier failures
are historical and are not current proof gaps.

Earlier focused checks passed for array support (55 s including dependencies),
event sources (385 s including the leaf dependency rebuild; edited module
3.7 s), witness read support (5 s), compact invariants (3 s), and invocation
origin (3 s). That support batch passed static checks and all 4,020 library jobs
in 203 s, and the full axiom/test gate passed all 4,189 jobs in 6 s after
adding the direct plan import to the audit file. All forty new or newly
public theorem audits passed. Logs are `/tmp/nightstream-producer-support-`
with suffixes `static.log`, `build.log`, and `axioms.log`.
The final complete-producer gates passed **4,020 library jobs in 130 s** and
**4,207 axiom/test jobs in 100 s**, with **71 new public theorem audits**.
These are Lean build/check times, not prover benchmarks.

The [quotient-value basis experiment](phi81_quotient_basis.md) verifies exact
forward/inverse field maps and predicts 111,598,761 fewer matrix nonzeros.
It saves no coordinates or rows. It is not Lean-proved, implemented, or
selected. The checkpoint remains 184,359,564 committed coordinates,
4,703,127 logical rows, and 3,001,571,645 matrix nonzeros. The additional 50%
coordinate target remains unproved; the local Poseidon bounds do not establish
global impossibility.
