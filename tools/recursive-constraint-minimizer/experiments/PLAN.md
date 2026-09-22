# Committed witness reduction

Status: the Lean reduction target is proved. Static checks, the full library
build (4,018 jobs), and the full axiom/test build (4,177 jobs) passed.
The reduced geometry and carrier inequality passed the axiom audit.
A candidate package and its binding were emitted and checked independently.
The candidate has not been selected: production identity pins and saved
recursive fixtures remain unchanged. The owner requested a checkpoint here,
then a pause in package/fixture generation and further Rust benchmarks while
the next Lean experiments are evaluated. No runtime improvement is claimed.

## Contract

The owner requested a reduction of at least 25% from the selected
253,011,276-coordinate committed carrier. Whole Phi81 blocks have 54
coordinates, so the largest aligned carrier that meets this request is
189,758,430 coordinates.

Keep the semantic specification, soundness, completeness, efficient witness
extraction, and existing security assumptions. Keep Goldilocks, Poseidon2,
`b = 2`, `k_rho = 16`, and `B = 65536`. Final production integration must
connect the chosen layout to the canonical package and Rust consumer.
Complete the research described below before selecting a new production
artifact and refreshing recursive fixtures.

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
matrix comparison confirmed these counts. The candidate is not selected.

Complete logical rows decrease by 26.2551%. Physical R1CS rows and the
fixed domain remain unchanged.

Evaluation rows have denser affine forms than the baseline grouped products.
The smaller witness does not establish lower execution cost. The complete
candidate comparison counted 3,001,571,645 logical matrix nonzero entries:
`[229698543, 4598475, 336477167, 83572300, 844999263, 1502121245, 0,
104652, 0, 0, 0, 0, 0, 0]`. Physical A/B/C nonzeros are
`[93701820, 39358148, 28868018]`. No before/after runtime comparison is complete.

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
case remains unverified; the combined command is a failed gate at this
checkpoint. No further Rust check will run before the research phase ends.

Two old probes were repaired without reducing coverage: a centered-unit
value of zero changed by +1 is still valid, so probes now use each decoded
polynomial degree; the Phi81 recipe probe now targets the final sampler
state actually read by the quotient recipe. Mutation row traversal now
checks parallel ranges in canonical order and keeps the first failure.

The completed old-layout CPU baseline used two application steps, including
one active fold: preparation 21.399 s, base proving 17.132 s, active fold
226.013 s, verification 152.593 s, total 417.137 s. Peak RSS was
8,500,621,312 bytes. It verified the final state. No candidate runtime result
has been recorded. `baseline-cpu.json` retains the measured result.

Production identity pins, saved recursive fixtures, final Rust integration,
and before/after runtime comparison remain pending. The shared formula and
verifier exports in this checkpoint are candidate data; do not treat this
checkpoint as a selected production release.

## Next research contract

The owner requested another 50% reduction from 184,359,564 to at most
92,179,782 committed coordinates. This is a research target, not an established
result. First provide a direct CCS witness construction for the optimized
ring products, prove its required witness mappings, and remove the related
R1CS work only after resolving its remaining dependencies. A complete compiler
rewrite is outside this task.

Use the actual remaining witness budget, starting with the largest costs and
Poseidon blocks. Search current papers, arXiv, blogs, webpages, and GitHub.
Use cvc5 for candidates and counterexamples; use Lean for soundness,
constructive completeness, and witness mappings under the unchanged
specification, security assumptions, and profile. Track coordinates, rows,
and matrix nonzeros separately. Moving cost between these measures is not an
overall performance improvement.

Batch Lean experiments before further package or fixture generation. Once
the chosen layout and proofs are stable, complete the affected artifacts,
Rust integration, and proving-time/peak-memory measurements. If the extra
50% target appears infeasible, report the achieved reduction and concrete
obstacles without weakening these requirements.
