# Centered private-field encoding

This component isolates the representation layer that currently turns private
Goldilocks values into most of the fixed F-prime width and row count. The goal
is the smallest relation that is justified by the protocol, with every cost
owned by a named mathematical obligation.

The ordinary 41-coordinate representation is now used by the Rust compiler.
The Lean results below separate its model-level mathematics and checked role
census from the still-open exact placement and row-removal refinements.

## Decision summary

| Surface | Coordinates | Local rows/gates | Assurance status |
|---|---:|---:|---|
| Historical ordinary canonical-binary field | 95 | 127 | pre-ordinary-lowering baseline |
| Production ordinary exact 41-trit word with local centered gates | 41 | stage-local pair/tail schedule | Rust-conformant role/cost census; exact placement artifact open |
| Ordinary exact 41-trit word under authoritative outer `b = 2` norm | 41 | 0 | kernel model; production bridge open |
| Canonical shifted opening, existing reduced core | 122 | 123 | kernel model |
| Canonical shifted opening under outer norm, conservative | 122 | 82 | kernel model; production bridge open |
| Canonical shifted opening with derived negatives | 81 | 41 substituted borrow obligations | kernel semantic and degree model; emitted-gate refinement open |

The semantic minimum currently proved is therefore:

- ordinary private field: 41 committed coordinates and no local alphabet row;
- canonical shifted opening: 41 digits, 40 borrow coordinates, and 41
  substituted borrow obligations, if a concrete gate lowering is proved;
- otherwise retain the conservative 82-row shifted-opening core.

The last line is not yet a production-emission claim. Lean now defines the
exact polynomial AST obtained by substituting `n = d(d - 1) / 2`, proves that
all 41 borrow equations have total degree at most three, proves that the bound
is attained, and proves equivalence with `DerivedAccepts`. The production gate
expression, Rust emitter, generated artifact, and row-by-row emitted equality
remain open. A strictly bilinear R1CS backend would need replacement product
wires and a refinement proof for them.

## Protocol-to-leaf ownership tree

The code should expose three review levels: protocol, phase, and constraint
family. A leaf owns a real equation family; there should not be one file per
individual equation.

```text
fprime
└── witness_encoding
    ├── ordinary_private_field
    │   ├── exact_word
    │   ├── decode_substitution
    │   └── outer_norm_discharge
    ├── canonical_shifted_opening
    │   ├── outer_norm_discharge
    │   ├── negative_definition
    │   ├── borrow_transition
    │   └── derived_negative_candidate
    └── selector
        ├── inactive_ordinary
        ├── inactive_private_bits
        └── authority_linked_zeroing
```

The existing global protocol tree remains owned by
`crates/neo-fold-clean/src/paper/f_prime/stage.rs`. These encoding families
must appear beneath the semantic stage that allocated the source value; they
must not be collapsed into one global `value_encoding` bucket.

### Lean owners

| Family | Current or future module | Owns | Emits constraints? |
|---|---|---|---:|
| exact centered word and HyperNova inverse | `Correspondence/FieldEncoding/CenteredTernary.lean` | alphabet, decode, finite tuple, chosen-word parser/emitter | no |
| outer norm discharge | `Correspondence/FieldEncoding/NormDischarged.lean` | `normBounded 2` iff centered alphabet; conservative 82-row shifted core | no |
| derived negative | `Correspondence/FieldEncoding/DerivedNegative.lean` | quadratic indicator and 82-to-41 semantic equivalence | no |
| generic linear compiler | `Correspondence/FieldEncoding/LinearCompiler.lean` | finite slot layout, exact parser, arbitrary-row transport, honest completeness | no |
| fixed F-prime eligibility manifest | `Correspondence/FieldEncoding/LayoutManifest.lean` | one concrete branch's independent source and coordinate owner runs, exact encoded/CE partitions, exhaustive roles, indexed compiler binding, and conditional CE authority theorem | no |
| fixed F-prime layout width floor | `Correspondence/FieldEncoding/LayoutWidthFloor.lean` | exact run-length reconciliation and generated-census-conditioned `eligible fields × 41` encoded/CE lower bounds | no |
| inactive-slot noninterference | `Correspondence/FieldEncoding/InactiveNoninterference.lean` | branch-relative selector, acceptance, and authority-output invariance from explicit read supports | no |
| ordinary emitted artifact | future `Correspondence/FieldEncoding/Refinement/Ordinary.lean` | exact Rust columns and substituted rows | no |
| derived-borrow polynomial refinement | `Correspondence/FieldEncoding/Refinement/DerivedBorrow.lean` | explicit polynomial, exact 41-equation schedule, degree census, and semantic equality with the Lean candidate | no |
| derived-borrow emitted artifact | future generated refinement module | each production emitted gate equals the proved polynomial candidate | no |

### Rust owners if implementation is approved

| Family | Exact owner | Required change |
|---|---|---|
| protocol/phase/family names | `src/paper/f_prime/stage.rs` | add stable encoding children under allocating stages |
| exact per-stage accounting | `src/frontends/f_prime/gadget_native/profile.rs` | split `value_encoding` by slot family and reconcile every parent |
| ordinary 41-word allocation and decode terms | `src/frontends/f_prime/gadget_native/slots.rs` | add the augmented centered-word slot; no authority logic |
| conservative shifted-opening validation | `src/frontends/f_prime/gadget_native/balanced_ternary.rs` | retain trace validation and the 82-row fallback path |
| derived-negative gate | future `src/frontends/f_prime/gadget_native/balanced_ternary/derived_negative.rs` | own the explicit polynomial, gate emission, and materialization only |
| inactive ordinary/bit zero policy | `src/frontends/f_prime/gadget_native/selector_gated.rs` | remove rows only after the noninterference artifact is checked |
| exact source layout trace | `src/engine/r1cs_circuit/encoding_trace.rs` | export verifier-checked slot roles; no cost policy |

The existing two axes remain intentional:

- `paper/f_prime/stage.rs` and `gadget_native/profile.rs` own execution/cost
  placement;
- `slots.rs` and `balanced_ternary.rs` own mathematical representation and
  lowering.

Parent tables map between the axes. Forcing both into one directory would blur
ownership.

## Assurance tiers

### Kernel-checked model facts

| Theorem | Guarantee |
|---|---|
| `centeredUnitGate_iff` | the cubic gate has exactly the three centered roots |
| `decode_encodeDigit` | the old deterministic encoder is a left inverse |
| `represents_zero_unique` | a decoded-zero accepted word is coordinate-wise zero |
| `width_floor` | 40 centered coordinates are insufficient; 41 suffice |
| `duplicate_words_differ` and `duplicate_words_decode_same` | the old semantic field does not determine one accepted word |
| `encodeChosenPrivate_decodeChosenPrivate` | an exact finite tuple of chosen words parses and re-emits identically |
| `augmented_private_exists_iff_semantic_exists` | augmentation preserves the existential language |
| `normBoundTwo_iff_centeredResidue` | canonical strict norm `< 2` is exactly `{-1,0,1}` |
| `concrete_normBounded_two_implies_centered` | the concrete SuperNeo list norm implies the same alphabet |
| `accepts_iff_canonicalRows` | outer norm plus 82 retained shifted rows is equivalent to all old canonical rows |
| `derivedNegative_eq_indicator` | `d(d-1)/2` is the exact negative indicator on centered digits |
| `conservative_iff_derived_and_materialized` | the 82-row predicate fixes the negative witnesses uniquely and agrees with the substituted obligations |
| `materialized_accepts_iff_derived` | conservative acceptance of the reconstructed old-layout extension iff the reduced input satisfies only 41 substituted borrow obligations |
| `reemit_parsed_projection` | norm-accepted finite private tuples parse and re-emit coordinate-for-coordinate |
| `decodedPrivateColumn` | each private source column is exactly the 41-word linear decode |
| `loweredRows_iff_sourceRows` | arbitrary source R1CS rows are sound and complete under sparse linear substitution |
| `honest_complete` | any source-row witness lifts through a separately supplied right-inverse materializer |
| `derivedBorrowEquation_holds_iff` | each explicit polynomial equation is exactly the corresponding reconstructed borrow obligation |
| `derivedBorrowEquation_degree_le_three` | every equation in the fixed 41-equation schedule has total degree at most three |
| `maximumDerivedBorrowDegree_eq_three` | degree three is attained by the concrete schedule |
| `eligible_or_explicitlyExcluded` | every source-column role is either an ordinary private field or an explicit non-eligible class |
| `ExactPartition.existsUniqueOwner` | first-start-zero, positive, abutting runs ending at the declared count give every in-range coordinate one owner |
| `ExactPartition.distinctOwnersDisjoint` | distinct owners in an exact partition cannot share a coordinate |
| `Manifest.Valid.existsUniqueSlotForSource` | every source column has one unique classified segment |
| `Manifest.Valid.encodedCoordinateHasUniqueOwner` | every encoded coordinate in one generated artifact has one unique owner |
| `Manifest.Valid.ceCoordinateHasUniqueOwner` | every CE coordinate in one generated artifact has one unique owner |
| `Manifest.Valid.ordinaryOwnerFor` | every eligible source segment has one source-backed encoded/CE owner of exactly 41 coordinates per field |
| `Manifest.Valid.sourceZeroHasConstantOneOwner` | source column zero is owned by the constant-one role |
| `Manifest.Valid.encodedZeroHasExcludedOwner` | encoded coordinate zero exists and its owner is explicitly non-eligible |
| `Manifest.Valid.ceZeroHasExcludedOwner` | CE coordinate zero exists and its owner is explicitly non-eligible |
| `eligibleSlots_share_committed_freshCe_assignment` | conditionally, every eligible word coordinate is read from the same fresh CE assignment whose Ajtai commitment and `b = 2` norm are checked |
| `normBounded_word_can_decode_nonCentered_source` | regression: a norm-bounded 41-coordinate word can decode to source residue `2`, which is not itself centered |
| `selectorComposed_sound` | changing only declared inactive support preserves acceptance in the forward direction when the new assignment satisfies every always-on obligation |
| `selectorComposed_complete` | the corresponding reverse direction holds when the old assignment satisfies every always-on obligation |
| `selectorComposed_acceptance_iff` | selected acceptance is invariant when selector and selected-equation supports are disjoint from the inactive support and both assignments satisfy always-on obligations |
| `authorityOutput_invariant` | authority-visible semantic outputs are invariant when their exact read support is disjoint from the inactive support |
| `inactiveNoninterference` | selector, acceptance, and authority-visible output invariance compose under one branch-relative inactive support |

These theorems do not inspect the production Rust assignment or generated CCS
artifact.

### Paper-derived requirements

HyperNova Definition 12 requires a deterministic polynomial-time, efficiently
invertible `enc` for NP-completeness
(`docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md:5-8`).
Its H.2 proof parses an arbitrary satisfying CCS tuple and re-encodes the same
tuple
(`docs/hypernova-paper/39_H_2_Proof_of_Lemma_3_Folding_CCS_NIVC_compatibility.md:34-46`).
The H.3 extractor calls `enc^-1` on a satisfying tuple
(`docs/hypernova-paper/40_H_3_Proof_of_Theorem_4_HyperNova.md:84-90,106-113`).

SuperNeo CCS and CE commit the exact witness and require
`||z||_inf < b`
(`docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:11-19`). With
`b = 2`, every committed coordinate is one of `-1`, `0`, and `1`. Since
`3^40 < p < 3^41`, 41 is the information-theoretic minimum width.

### Production-refinement obligations

None of the following may be inferred from the model theorem:

1. Separate generated base and recursive artifacts each instantiate
   `Manifest.Valid` and `CompilerBinding`. Within each artifact, source segments
   exactly partition source-column order, while coordinate owners independently
   partition encoded and CE order. Runs start at zero, have positive lengths,
   abut exactly, and end at their declared count; handwritten counts or profiler
   totals are not authority. A combined artifact remains out of scope until a
   fixed materializer and selector-composition proof exist.
2. The actual recursive acceptance path instantiates
   `eligibleSlots_share_committed_freshCe_assignment`, so every eligible
   41-coordinate word is aligned with the same fresh CE assignment whose Ajtai
   commitment and verifier-owned `b = 2` norm are checked.
3. The generated layout classifies every source column as constant-one,
   ordinary private field, private Boolean, public bit, canonical-u64, SIS
   opening, linearly derived, structural balanced alias, gadget-derived, or
   product-derived, with gadget temporaries called out separately. Only the
   ordinary private-field role may enter this optimization. Exact source-run
   partitioning must prove that no source column was omitted or classified
   twice. Coordinate-only owners classify selectors, alignment padding, and
   synthetic lowered fields that have no source column. Encoded and CE
   coordinate zero must each have an explicitly excluded owner and therefore
   cannot begin an ordinary field placement.
4. Every source equation consumes the exact 41-word decoder by structural
   substitution, without an unconstrained alias.
5. The production emitter is refined to the explicit derived-borrow polynomial
   schedule, including any product wires required by a bilinear R1CS backend.
6. Every removed selector-zero row satisfies the noninterference theorem below.
7. A standalone decider that sees only matrix satisfaction also checks the
   omitted norm predicate. Otherwise it must retain local alphabet gates.

Until these are closed, the results are safe optimization candidates, not
production row-removal authority.

The authority direction is deliberately one-way: the strict norm applies to
each of the 41 coordinates in the committed CE assignment. It does **not**
apply to the decoded source residue. That residue is the radix-three weighted
decode and can be `2` (or any canonical field value) while every coordinate is
in `{-1,0,1}`. The regression theorem fixes this distinction in Lean.

The concrete low-norm ABI fixes assignment coordinate zero to ONE and commits
that whole assignment as the CE witness. The schema therefore reserves both
encoded coordinate zero and CE coordinate zero with explicitly excluded
owners. It does not assume source order is encoded order, or that encoded and
CE starts coincide. Each generated branch artifact states those two orders
directly; `CompilerBinding` then proves the exact contiguous 41-coordinate
placement for each eligible source field. This ABI needs no affine mapping: the
committed CE vector is the exact low-norm assignment vector. If a future ABI
introduces striding or permutation, it requires a separately proved schema
extension.

The current gadget-native compiler materializes ordinary private fields with a
41-coordinate footprint. The generated source-role census and cost tree pin
the exact field count and aggregate width. This Lean schema describes the
stronger exact-placement artifact still required to connect each concrete
source column to its production 41-coordinate word.

## Why local alphabet rows are redundant only conditionally

The outer SuperNeo relation already includes the norm predicate. Lean's
`Opening.Holds` includes `normBounded`, concrete CCS membership expands it, and
Pi_CCS reconstructs the fresh assignment under that condition. Therefore an
accepted `R'_2 = CCS(b=2,L)` opening already proves each committed coordinate
is centered.

The theorem in `NormDischarged.lean` shows this premise is mathematically
identical to the local `d^3-d=0` alphabet obligation. Duplicating both checks is
semantic bloat.

The premise is nevertheless authority-sensitive. A native witness-generator
check, a prover-supplied digest, or plain R1CS matrix satisfaction does not
replace the verifier-owned norm predicate. Any backend that omits the outer
norm must retain the 41 local alphabet gates.

Public Boolean slots are also separate: norm `< 2` permits `-1`, so it does not
prove bitness.

## HyperNova invertibility and the augmented witness

The old semantic witness is not exactly invertible. Raw radix-three targets
`0` and `p` fit in 41 trits, produce distinct accepted words, and decode to the
same Goldilocks residue. Lean proves this counterexample.

The fix is to make the exact accepted word part of the nondeterministic witness:

```text
w_aug = { digits : Fin 41 -> F, accepted : every digit is centered }
source(w_aug) = sum_i digits[i] * 3^i mod p
F_aug(x, w_aug) = F_old(x, source(w_aug))
```

The exact committed word is retained and can be re-emitted. This gives a
two-sided parser/emitter and existential equivalence with the old relation for
any finite tuple of eligible fields. It is not a digest shortcut and does not
make the decoded residue authoritative over the committed word.

The finite-tuple theorem remains parameterized. Lean now has a fail-closed
manifest schema with two independent orders: source segments partition source
columns, while coordinate-owner runs partition encoded and CE coordinates.
Coordinate-only owners cover lowered coordinates with no source column. One
manifest describes one concrete production branch; base and recursive layouts
therefore require separate generated artifacts. Each eligible source segment
batches adjacent ordinary fields: a source run of length `n` has one
source-backed owner whose encoded and CE runs each have length `n * 41`. The
compiler binding derives every field's exact block without materializing a flat
coordinate or field list. Every excluded role remains outside this path. There
is not yet a generated base or recursive artifact instantiating that interface
and proving the production ABI. A profiler count alone cannot prove those
facts.

## Derived-negative boundary

For a centered digit `d`, the old negative witness is uniquely:

```text
n = d(d - 1) / 2 mod p
```

It is `1` only for `d = -1`. `DerivedNegative.lean` proves this formula,
defines a reduced assignment whose old negative interval is ignored, and
reconstructs the 41 deleted values. It then proves:

```text
82-row conservative acceptance
  iff
41 substituted borrow obligations on the reduced input, with the deleted
negative values reconstructed rather than constrained.
```

This closes semantic redundancy of the 41 negative-definition rows and 41
negative columns at the reduced-layout model boundary. `Refinement/DerivedBorrow.lean`
also gives the substituted predicate an explicit polynomial representation,
proves every one of the 41 equations has degree at most three, proves that the
maximum is exactly three, and connects the schedule back to `DerivedAccepts`.

Concrete lowering is still open. The current source rows are bilinear R1CS,
so a strict R1CS backend may need replacement product wires. The future
`balanced_ternary/derived_negative.rs` emitter and a generated equality proof
between every emitted constraint and the Lean polynomial schedule are the
remaining production owners.

## Generic linear compiler boundary

`LinearCompiler.lean` states the generic theorem needed before any fixed
F-prime census is trusted. A proof-carrying `Layout fieldCount` supplies:

- the source column for each ordinary private field;
- its 41 encoded coordinate columns;
- the complete sparse expansion for all source columns; and
- an equality proving each private expansion is exactly the radix-three
  weighted coordinate list.

For any source row list, not a hand-selected row shape, Lean proves:

```text
Satisfies(map linearSubstitution sourceRows, encoded)
  iff
Satisfies(sourceRows, decodedAssignment(encoded)).
```

Under the projected external norm, the parser retains every exact word and
the decoded private source columns equal those parsed sources. A separate
`HonestMaterializer` right-inverse gives existential completeness from any
canonical source witness. That interface deliberately includes non-private
columns; the field component cannot invent their encoding.

The concrete fixed layout is still open. `LayoutManifest.lean` defines the
generator-owned source census, an independently ordered encoded/CE ownership
census for one concrete branch, indexed eligible placement, validity
obligations, `CompilerBinding`, and the conditional theorem that places all
eligible words inside one committed, norm-checked fresh CE assignment.
Production must still generate separate exact base and recursive censuses and
instantiate that interface plus `HonestMaterializer`. There is no fixed
combined materializer or selector-composition proof yet.
`InactiveNoninterference.lean` now supplies the generic support theorem, but no
generated production support manifest instantiates it. No Lean definition in
this component hard-codes the profiler's current field count as authority.

## Inactive zeroing

Under the augmented witness, an inactive ordinary word need not be the all-zero
word for HyperNova invertibility or existential semantics. The parser retains
the exact word. Changing inactive coordinates changes the CE commitment, so the
commitment must be recomputed from the exact changed witness; no theorem claims
the old commitment remains valid.

Zero rows may be removed only after proving this exact noninterference shape:

```text
For selector b and assignments e, e':
  e and e' agree on ONE, selector, public inputs, active branch slots,
  every shared or authority-linked slot, and every column outside
  InactiveOrdinary(b);
  e' satisfies the outer norm, and inactive bit slots retain bitness.

Then fixed-relation acceptance(e) iff fixed-relation acceptance(e'),
and both produce the same semantic outputs.
```

`InactiveNoninterference.lean` proves this shape without assuming the target
equivalence. A `Boundary` declares selector, always-on, selected-equation, and
authority-output functions. Extensional read-support proofs are required for
the selector, selected equations, and output. The exported soundness and
completeness theorems then derive acceptance invariance from change confinement,
support disjointness, and explicit old/new always-on premises. Separate test
witnesses show that dropping confinement, any support-disjointness premise,
either always-on premise, or any read-support proof can change the conclusion.

The concrete proof must still show inactive ordinary coordinates are absent
from selected semantic equations, hashes, transcripts, public outputs, and
authority bindings except the full CE commitment. The commitment is recomputed
from the exact changed witness; it is deliberately not an invariant output of
the theorem. Public fields, direct canonical-u64 interfaces, SIS-authoritative
openings, synthetic derived fields, and any digest-omitted data remain excluded
until separate theorems exist.

Current fixed-selector accounting contains:

| Inactive family | Rows | Current classification |
|---|---:|---|
| one-bit zero bindings | 985,574 | retain; concrete selected-support proof open |
| SIS-word zero bindings | 11,863 | retain pending authority and selected-support proof |
| direct/synthetic canonical-word zero bindings | 3,075 | retain; includes the two direct canonical-u64 interfaces |
| ordinary weighted-decode zero bindings | 161,381 | retain; exact role census exists, selected-support proof open |
| aggregate-acceptance bindings | 14,400 | retain; concrete selector bridge open |
| packed Mod-5 low bits | 12,480 | retain; packed selector materializer and support bridge open |
| packed Mod-5 residue pairs | 960 | retain; abstract inactive equation proved, concrete selector materializer open |
| **total** | **1,189,733** | exact current estimator formula |

The generated role census now identifies the ordinary subset exactly, but a
role census is not a selected-support proof. The currently authorized removal
count therefore remains zero. In particular, recomputing a digest after
changing an inactive word does not prove that the word was absent from every
selected equation or authority output.

### Smallest concrete support artifact

The future combined gadget-native materializer should generate one compact
artifact, not flat million-entry Lean lists:

| Artifact field | Exact source | Required check |
|---|---|---|
| encoded owner runs for each branch | `GadgetNativePlan` slot ranges plus synthetic/packed owners | exact nonoverlapping partition and exhaustive role/family tag |
| row owner runs | materialized fixed CCS matrices | exact partition into always-on, base-selected, recursive-selected, and inactive-binding rows |
| selector support | materialized selector column and selector polynomial | exact matrix support |
| selected support per branch | union of nonzero encoded columns in that branch's selected rows, retaining public/hash/transcript/authority-binding stage tags | recomputed from matrices, not caller supplied |
| authority-output support per branch | public and semantic accumulator/state output projections | recomputed union with explicit family tags |
| inactive support per branch | opposite branch's eligible coordinate-owner runs | exact equality with candidate binding-row owners |

Lean should define each support as membership in compact coordinate runs and
instantiate `SupportManifest`; it should not expand those runs into a literal
column list. Rust must independently recompute every read support from the
materialized matrices/traces and reject any mismatch. A digest may identify the
artifact but is never authority for its contents.

The current Rust tree cannot emit this artifact. `selector_gated.rs` consumes
only aggregate branch estimates, and therefore has neither combined branch
slots nor matrices. Individual `GadgetNativePlan`s expose some source and
synthetic ranges, while Poseidon2 traces expose source-side input columns, but
there is no branch-offset mapping, selected-row support union, authority-output
union, or exact inactive-binding row list for the optimized fixed relation.
The older generic fixed-shape lowering materializes a different canonical-bit
relation and cannot instantiate this formula.

## Integrated production arithmetic

The 41-coordinate ordinary representation is no longer conditional arithmetic.
The generated source-role census and reconciled cost tree pin:

| Profile | Ordinary fields | Ordinary coordinates | Pair rows | Tail rows |
|---|---:|---:|---:|---:|
| Base | 3,050 | 125,050 | 62,524 | 2 |
| Recursive | 154,747 | 6,344,627 | 3,171,786 | 1,055 |
| Fixed selector formula | 157,797 | 6,469,677 | 3,234,838 | 1 |

The complete fixed formula is 6,184,892 rows by 8,262,817 columns. Thus
6,469,677 columns are already forced by the present decision to represent each
ordinary source field independently with 41 coordinates. This is a lower
bound for that implementation architecture, not for SuperNeo itself. Reaching
one million columns requires proving many source fields derived/removable or
introducing a compact proof-backed transition boundary; another packing pass
over the same 157,797 fields cannot suffice.

Historical deltas must not be added to this integrated total. The profiler and
reconciled protocol/phase/family tree are the accounting authority.

## Closure checklist

1. Export the exact ordinary source-to-coordinate placement from each concrete
   base and recursive plan, using compact runs plus pointwise Rust drift checks;
   do not hand-author or flatten the census in Lean.
2. Instantiate `CompilerBinding`, `HonestMaterializer`, and the finite-tuple
   chosen-word theorem over each artifact.
3. Instantiate the conditional CE theorem against the production fresh opening
   and exact coordinate alignment.
4. Prove the production ordinary 41-word allocation and every substituted
   source row instantiate the Lean compiler theorem.
5. Choose the conservative 82-row shifted core or implement the proved
   41-equation polynomial schedule and prove exact generated-gate equality,
   including any strict-R1CS product wires.
6. Prove inactive ordinary/private-bit noninterference before deleting zero
   rows.
7. Bind the exact low-norm vector through CE/Ajtai commitment correspondence.
8. Attribute every emitted row and column under protocol/phase/family paths and
   mechanically reconcile every parent.
9. Re-run the complete measured fixed F-prime estimate; no handwritten total
   may replace the artifact-backed profiler.
