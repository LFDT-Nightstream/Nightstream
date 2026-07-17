# F' Encoding Boundary

There are two different encodings in Construction 2. Keeping them separate is
load-bearing.

## `enc_inst(h)`

`enc_inst` maps the raw F' output hash into the public input of the next fresh
CCS instance. The four Poseidon2 lanes are ordinary Goldilocks values while
they are computed, but SuperNeo with `b = 2` cannot commit those values
directly. The canonical public representation is therefore

```text
h            = [h0, h1, h2, h3]
enc_inst(h)  = bits64(h0) || bits64(h1) || bits64(h2) || bits64(h3)
public x     = [1 || enc_inst(h)]
```

This is 257 low-norm coordinates. Canonical Goldilocks decomposition is
enforced, so the bit string for the modulus cannot alias zero.

## `enc_str(F')` and `enc(F', witness)`

The larger encoding turns the complete augmented relation and one execution
into a SuperNeo CCS structure and low-norm assignment. It must cover, in one
fixed language:

- the base/recursive selector;
- the application transition and semantic-state links;
- the prior public recursive link;
- complete fixed-shape `NIFS.V` (`Pi_CCS -> Pi_RLC -> Pi_DEC`);
- running-accumulator continuity and its derived parent authority;
- counters, program counter, and exact next `x_out`.

The current field-valued implementation reference is
`frontends/r1cs_f_prime/full_relation.rs`. Its verifier-owned
`FullFPrimeRelation` records the NIFS configuration, application R1CS, state
column schema, and context anchors. It is useful for implementation
differential tests, but it is not the semantic authority for deciding which
checks are sufficient or necessary. That authority must come from the
independent paper-level Lean semantics and their concrete refinement. Native
compiler checks are not part of its acceptance condition.

The generic differential reference in `frontends/f_prime/low_norm_r1cs.rs` is
deterministic and invertible:

1. Public `enc_inst` bits are placed immediately after the constant-one lane.
2. Each remaining field wire is represented by its canonical 64-bit value.
3. Explicitly proved Boolean wires may use one bit in the derived mode.
4. Canonicality auxiliaries are deterministic prefix products.
5. Every source R1CS row is replayed over the decoded linear combinations.

The decoder reconstructs the exact source witness and rejects inconsistent
auxiliaries. Small tests also cross the real
`CcsInstance::from_low_norm_assignment` boundary.

## Implemented encoding study

The generic encoding remains an implementation-level differential reference.
It can detect disagreement with the current source R1CS, but agreement does
not prove that the source R1CS is a sound or minimal SuperNeo/HyperNova
verifier. The optimized study in `frontends/f_prime/gadget_native.rs` is
derived from the same R1CS emission; it does not use `RecursiveStepImagePlan`
or caller-supplied trace counts.

The R1CS builder records exact row provenance for Poseidon2 S-boxes,
quadratic-extension multiplication, production Toom-3 ring multiplication,
and first-accepted selection blocks.
The optimized compiler then:

1. validates every recorded row against the current source R1CS;
2. rejects overlapping trace ranges and any projected temporary used outside
   its recorded gadget;
3. projects the three `x^2/x^4/x^6` Poseidon temporaries;
4. replaces three Karatsuba products with two exact extension-field equations;
5. replaces each 1,620-product Toom trace with 175 exact convolution
   coefficients and 54 linear output equations;
6. replaces each traced selection product block with three exact aggregate
   ProductSum equations after validating its product definitions and binding
   rows;
7. validates each balanced-ternary opening against its exact 124-row source
   program, then represents its field by 41 centered coordinates while aliasing
   its digit wires and using one-bit slots for sign and borrow auxiliaries;
8. uses the proved exact opening reduction: it retains 82 source product
   obligations, omits 42 source rows and 81 proved-redundant common binary
   gates, then lowers the surviving centered obligations through deterministic
   stage-local residual pairs plus an ordinary tail when needed;
9. replaces each validated Pi_RLC projection identity by its exact 34-output,
   70-synthetic-field ProductSum plan and lowers seeded `Phi81` rows through
   the exact expanded-source fallback path;
10. lowers every unclassified source row generically; and
11. reconstructs the complete source witness, including projected products, for
   differential comparison with the oracle.

The CCS gate polynomial has at most 18 product terms plus `x^7` and uses degree
8 when an internal base/recursive selector is applied. Small materialized tests
prove

```text
source R1CS witness == decoded gadget-native assignment
```

and reject retained-output, synthetic-convolution, and escaped-temporary
tampering.

## Current dimension ledger

The implementation-accounting fixture is one application instruction, one
fresh CCS claim, the fixed `k_rho = 14` running accumulator, and one complete
recursive `NIFS.V` under the Goldilocks profile (`b = 2`, `D = 54`,
`kappa = 18`). The
ledger below is produced on top of `nico/nebula-m0-frontend`; the previously
reported 59.5M-row fixed result was computed on the wrong base and is not a
reference. Source R1CS dimensions are unchanged by the lowering. The
gadget-native dimensions include the proved balanced-opening reduction,
the three-equation selection aggregates, the exact projection-identity
compaction, the packed Mod-5 lowering, and the 41-coordinate ordinary-private
field representation described below.

```text
base branch R1CS          22,812 rows x     22,353 columns
recursive branch R1CS  2,576,416 rows x  2,399,107 columns
materialized source selector 7,575,344 rows x 4,998,209 columns
un-audited direct-selector estimate 258,444,060 rows x 190,149,709 columns

gadget-native base           66,358 rows x    125,695 columns
gadget-native recursive   4,933,049 rows x  8,137,378 columns
gadget-native fixed       6,184,892 rows x  8,262,817 columns (formula only)
```

These numbers do not all have the same assurance tier:

| Dimension | Evidence | Status |
|---|---|---|
| Base/recursive source R1CS | Materialized satisfied snapshots with complete source row/column ownership trees | executable source relations |
| 7,575,344 x 4,998,209 source selector R1CS | Materialized base and recursive executions share one relation, have no unconstrained columns, and reject a disconnected recursive witness; a complete per-family ownership artifact is not yet generated | executable selector-composed source relation; not a low-norm encoding |
| 258,444,060 x 190,149,709 direct selector | Arithmetic in `estimate_selector_gated_r1cs_encoding`; no materializer, emitted-row replay, or selector theorem | un-audited estimate |
| 66,358 x 125,695 gadget-native base | Materialized lowering plus pointwise source-plan checks | executable lowered base relation |
| 4,933,049 x 8,137,378 gadget-native recursive | Exact trace validator, estimator, and mechanically reconciled stage tree; the complete 8.1M-column relation is not materialized by the full-relation test | trace-reconciled estimate |
| 6,184,892 x 8,262,817 gadget-native fixed | Branch estimates plus selector/inactive-binding cost arithmetic | formula only |

For transparency, the 258M estimator performs exactly this arithmetic:

| Direct-selector formula component | Columns | Rows |
|---|---:|---:|
| Public prefix including constant one | 257 | 0 |
| Branch selector | 1 | 0 |
| Private one-bit slots | 38,111 | 0 |
| 2,001,172 canonical field slots at 95 columns each | 190,111,340 | 0 |
| Coordinate bitness | 0 | 190,149,708 |
| Canonicality at 32 rows per canonical field | 0 | 64,037,504 |
| Inactive decoded-value zero equations | 0 | 2,039,283 |
| Surviving base and recursive branch equations | 0 | 2,217,565 |
| **Formula total** | **190,149,709** | **258,444,060** |

This table audits the estimator's addition only. It does not show that the
proposed inactive bindings are sufficient, that selector gating is sound, or
that any emitted CCS matrices have these dimensions.

The prior correct-base estimator snapshot reported **214,119,163 rows x
157,492,794 columns**; it is historical and is not a baseline. Subtracting the
current formula-only gadget-native fixed estimate (`6,184,892 x 8,262,817`)
from the un-audited direct-selector formula gives 252,259,168 rows (97.61%)
and 181,886,892 columns (95.65%). Those percentages are not a measured circuit
reduction: neither fixed encoding has been materialized and reconciled against
emitted constraints, and the direct-selector construction has not been proved
selector-sound.
The fixed language adds 125,439 columns and 1,251,843 rows over the recursive branch for base
slots, selector gating, and branch-specific inactive bindings. It is not the
dominant cost.
The test pins the two materialized source-R1CS branch dimensions, materializes
the lowered base branch, and separately pins the recursive trace estimate and
fixed selector formula. A semantic change or optimization must update the
snapshot and the stage tree together; neither the fixed total nor the 258M
direct-selector number is a materialized relation or selector-soundness claim.

### Full-relation cost-accounting tree

Every current number below is asserted by `tests/f_prime/full_relation.rs`
against the production estimator. Within a tree, every parent is the exact sum
of its descendants and each leaf names the lowering owner whose validated
trace or source family is counted. For the recursive gadget-native branch this
is estimator reconciliation, not replay of a fully materialized 8.1M-column
relation.

The stable ownership paths live in `paper/f_prime/stage.rs`. The base and
recursive roots cover every traced source row and every traced source column;
their gadget-native profiles also cover every lowering metric and row family.
The one global constant-one column is a formula input rather than an emitted
stage range, so the root's owned-column count plus one equals the complete
profile total.

```text
fprime.base
├── verifier_key
├── step
│   ├── prelude
│   ├── source
│   ├── initial
│   ├── advance
│   └── output
└── finalize.{context_link,application,semantic_links}
```

| Base path | Source rows | Owned source columns | Encoded rows | Owned encoded columns | Omitted exact Boolean rows |
|---|---:|---:|---:|---:|---:|
| `fprime.base` | **22,812** | **22,352** | **66,358** | **125,694** | **644** |
| `verifier_key` | 6,056 | 6,068 | 18,736 | 35,752 | 0 |
| `step` | 11,890 | 11,436 | 32,781 | 61,652 | 644 |
| `step.prelude` | 5,454 | 5,484 | 16,891 | 32,226 | 0 |
| `step.source` | 463 | 454 | 298 | 571 | 448 |
| `step.initial` | 10 | 0 | 0 | 0 | 0 |
| `step.advance` | 526 | 332 | 387 | 374 | 128 |
| `step.output` | 5,437 | 5,166 | 15,205 | 28,481 | 68 |
| `finalize` | 4,866 | 4,848 | 14,841 | 28,290 | 0 |
| `finalize.context_link` | 12 | 0 | 0 | 0 | 0 |
| `finalize.application` | 2 | 4 | 41 | 82 | 0 |
| `finalize.semantic_links` | 4,852 | 4,844 | 14,800 | 28,208 | 0 |

Thus the base root plus the global one column is exactly `22,812 x 22,353`
at the source and `66,358 x 125,695` after lowering.

```text
fprime.recursive
├── verifier_key
├── step
│   ├── prelude
│   ├── transcript
│   ├── nifs
│   │   ├── nifs.pi_ccs
│   │   ├── nifs.pi_rlc
│   │   ├── nifs.pi_dec
│   │   └── nifs.point_binding
│   ├── prior_link
│   ├── nebula
│   ├── accumulator.{input_link,output_authority}
│   ├── counters
│   └── output
└── finalize.{context_link,application,semantic_links}
```

`fprime.recursive.step.nifs` is an emitted zero-cost organizational checkpoint.
It aggregates the existing absolute PiCCS and PiRLC trees plus the PiDEC and
point-binding tail; it does not rename those trees or fabricate a source range.

| NIFS child | Source rows | Owned source columns | Encoded rows | Owned encoded columns | Omitted exact Boolean rows |
|---|---:|---:|---:|---:|---:|
| `nifs.pi_ccs` | 1,602,597 | 1,421,733 | 2,703,670 | 4,203,231 | 1 |
| `nifs.pi_rlc` | 666,223 | 677,578 | 1,240,656 | 2,058,157 | 23,505 |
| `nifs.pi_dec` | 10,597 | 3,780 | 85,255 | 154,980 | 0 |
| `nifs.point_binding` | 2 | 0 | 0 | 0 | 0 |
| **`step.nifs`** | **2,279,419** | **2,103,091** | **4,029,581** | **6,416,368** | **23,506** |

The second ordinary-child `y_zcol` elision is physically owned by
`nifs.pi_rlc.shape.allocate`, because that is where the Π_DEC parent/children
wires are allocated. This is cost attribution only. The NC authority audit now
has a concrete laundering counterexample: the current accumulator handle does
not bind the parent projection, the next step replaces its point without an
old-point check, and terminal child checks cannot recover the discarded
obligation. No row-removal permission follows from this measured elision.

| Recursive path | Source rows | Owned source columns | Encoded rows | Owned encoded columns | Omitted exact Boolean rows |
|---|---:|---:|---:|---:|---:|
| `fprime.recursive` | **2,576,416** | **2,399,106** | **4,933,049** | **8,137,377** | **24,539** |
| `verifier_key` | 6,056 | 6,068 | 18,736 | 35,752 | 0 |
| `step` | 2,565,498 | 2,388,190 | 4,899,472 | 8,073,335 | 24,539 |
| `step.prelude` | 6,176 | 6,197 | 17,734 | 33,818 | 705 |
| `step.transcript` | 13,262 | 13,262 | 40,678 | 77,572 | 0 |
| `step.nifs` | 2,279,419 | 2,103,091 | 4,029,581 | 6,416,368 | 23,506 |
| `step.prior_link` | 5,832 | 5,298 | 15,575 | 28,691 | 196 |
| `step.nebula` | 0 | 0 | 0 | 0 | 0 |
| `step.accumulator` | 254,919 | 254,918 | 780,451 | 1,488,300 | 0 |
| `step.accumulator.input_link` | 4 | 0 | 4 | 0 | 0 |
| `step.accumulator.output_authority` | 254,915 | 254,918 | 780,447 | 1,488,300 | 0 |
| `step.counters` | 522 | 324 | 305 | 210 | 128 |
| `step.output` | 5,368 | 5,100 | 15,148 | 28,376 | 4 |
| `finalize` | 4,862 | 4,848 | 14,841 | 28,290 | 0 |
| `finalize.context_link` | 8 | 0 | 0 | 0 | 0 |
| `finalize.application` | 2 | 4 | 41 | 82 | 0 |
| `finalize.semantic_links` | 4,852 | 4,844 | 14,800 | 28,208 | 0 |

Thus the recursive root plus the global one column is exactly `2,576,416 x
2,399,107` at the source and `4,933,049 x 8,137,378` after lowering. The
test rejects duplicate parents, unreachable nodes, missing zero-cost nodes,
unowned metrics, and disagreement between any parent and its immediate
children across the complete metric set.

The NC terminal identity has a finer source-row overlay. These are
`record_row_family` ranges inside the single physical
`nifs.pi_ccs.output_binding_and_terminal_checks.nc_terminal.identity` stage;
they do not introduce lowering boundaries or change the circuit.

| NC terminal family | Occurrences | Source rows each | Total source rows |
|---|---:|---:|---:|
| equality factors | 1 | 175 | 175 |
| `chi_alpha` basis | 1 | 632 | 632 |
| gamma powers | 1 | 77 | 77 |
| output evaluations | 15 | 323 | 4,845 |
| range products | 15 | 12 | 180 |
| weighted sum | 1 | 78 | 78 |
| final product | 1 | 5 | 5 |
| **identity total** | **35** | — | **5,992** |
| final-sum pin | 1 | 2 | 2 |
| **NC terminal total** | **36** | — | **5,994** |

The test pins every occurrence, each per-occurrence shape, contiguity, and the
5,992-row reconciliation with the physical identity stage. Lowered rows and
columns remain attributed to that physical parent because canonical and
Boolean pairing may cross these diagnostic family boundaries.

#### Fixed selector cost formula

The fixed selector is not one contiguous source trace, so it has a separate
cost-formula audit instead of fake stage ownership. Every term is derived from
the two complete branch estimates and the current selector estimator, and the
test mechanically sums these terms to the pinned estimated dimensions. This
is cost accounting, not a proof that a selector-gated gadget-native relation
has been materialized soundly.

| Formula component | Encoded rows | Encoded columns |
|---|---:|---:|
| Global constant one | 0 | 1 |
| One-bit committed coordinates | 0 | 999,271 |
| Aggregate-acceptance tree outputs | 0 | 13,440 |
| Ordinary-private coordinates (`157,797 x 41`) | 0 | 6,469,677 |
| Balanced coordinates (`11,863 x 41`) | 0 | 486,383 |
| Packed Mod-5 residue coordinates | 0 | 1,920 |
| Direct canonical-u64 raw-bit coordinates | 0 | 128 |
| Direct canonical-u64 prefix coordinates | 0 | 62 |
| Synthetic ring raw/prefix coordinates | 0 | 0 |
| Synthetic ProductSum raw-bit coordinates | 0 | 196,672 |
| Synthetic ProductSum prefix coordinates | 0 | 95,263 |
| Common Boolean pair rows | 12,464 | 0 |
| Common Boolean tail rows | 0 | 0 |
| Ordinary-private centered pair rows | 3,234,838 | 0 |
| Ordinary-private centered tail rows | 1 | 0 |
| SIS centered pair rows | 243,191 | 0 |
| SIS centered tail rows | 1 | 0 |
| Direct canonical-u64 raw-bit pair rows | 64 | 0 |
| Direct canonical-u64 prefix pair rows | 31 | 0 |
| Synthetic ring pair/tail rows | 0 | 0 |
| Synthetic ProductSum raw-bit pair rows | 98,336 | 0 |
| Synthetic ProductSum raw-bit tail rows | 0 | 0 |
| Synthetic ProductSum prefix pair rows | 47,631 | 0 |
| Synthetic ProductSum prefix tail rows | 1 | 0 |
| Remaining canonical relation pairs | 49,200 | 0 |
| Base semantic rows | 3,510 | 0 |
| Recursive semantic rows | 1,309,475 | 0 |
| Inactive one-bit zero bindings | 985,574 | 0 |
| Inactive SIS-word zero bindings | 11,863 | 0 |
| Inactive canonical-word zero bindings | 3,075 | 0 |
| Inactive ordinary weighted-decode bindings | 157,797 | 0 |
| Inactive aggregate-acceptance bindings | 14,400 | 0 |
| Inactive packed Mod-5 low bits | 12,480 | 0 |
| Inactive packed Mod-5 residue pairs | 960 | 0 |
| **Fixed total (formula only)** | **6,184,892** | **8,262,817** |

The row/column asymmetry is intentional. Coordinates remain committed while
their Boolean obligations are paired independently within each fixed-formula
family; odd prefixes retain explicit tail rows. Balanced-opening binary
auxiliaries retain their validated source obligations, while the 12,480 packed
Mod-5 low-bit coordinates are enforced by packed rows rather than common
Boolean rows.

The `FPRIME_FIXED_FORMULA` test output is deliberately described as formula
ownership, not trace ownership. This makes the 1,251,843-row / 125,439-column
fixed overhead visible without assigning composition-only constraints to a
source verifier phase.

#### Current per-field floor

The ordinary-private role census and the cost tree now reconcile exactly:

| Profile | Ordinary fields | 41-coordinate columns | Stage-local pair rows | Stage-local tails |
|---|---:|---:|---:|---:|
| Base branch | 3,050 | 125,050 | 62,524 | 2 |
| Recursive branch | 154,747 | 6,344,627 | 3,171,786 | 1,055 |
| Fixed formula | 157,797 | 6,469,677 | 3,234,838 | 1 |

This is an artifact-checked floor **for the current per-source-field design**:
if all 157,797 fields remain independent ordinary inputs, their exact radix-3
representation alone consumes 6,469,677 columns. It is not a theorem that the
protocol needs 157,797 independent fields. Reducing the complete fixed
relation toward one million columns therefore requires a semantic proof that
whole source-field families are derived, redundant, or replaceable by a
compact transition proof. More packing inside the same 41-coordinate model
cannot cross this floor.

The next conformance layer is an exact generated placement artifact connecting
each ordinary source column to its 41 concrete gadget-native coordinates. The
existing source-role census proves the role counts, and the generic Lean
compiler proves the per-field mathematics, but neither alone proves exact
production placement or authorizes deleting a verifier obligation.

There is a separate HyperNova compatibility obligation. The current ordinary
encoder deterministically derives one 41-coordinate word from the source field
value. The low-norm relation can admit a different centered word with the same
field decode, so exact accepted-word round-tripping is not yet proved. A strict
efficient-invertibility argument must either preserve that chosen word as part
of the nondeterministic input or prove that the relation accepts only the
canonical word. Aggregate width and placement evidence do not close this gap.

The intended selector semantics are:

- bit, centered residual-pair/tail, and canonical-slot encoding gates remain
  active for both branch slot sets;
- base semantic rows are multiplied by `is_base`, while recursive semantic
  rows are multiplied by `1 - is_base`; and
- one extra decoded-value equation forces each ordinary inactive private source
  slot to zero. Binary auxiliaries are separately included among the inactive
  one-bit rows. Aggregate acceptance is not zero-filled: all-zero chunk inputs
  canonically derive fourteen zero tree outputs, retained `accept = 1`, and the
  projected inverse of `-65535`. The fifteen bindings cover only the fourteen
  committed tree outputs plus the committed accept coordinate. The inverse is
  uncommitted/projected and reconstructed by the decoder; it does not add a
  sixteenth binding. No executable fixed materializer exists yet. A packed
  Mod-5 chunk instead uses thirteen inactive
  low-bit rows and one shifted nonresidue equation
  `(L + 1)^2 - 7(R + 1)^2 = 0` for its two residue coordinates.

For a balanced slot, Lean proves that 41 centered digits satisfying
`sum d_i 3^i = 0` in Goldilocks must be the all-zero word. Their integer sum
lies in `[-shift, shift]`, where
`shift = (3^41 - 1) / 2 = 18,236,498,188,585,393,201 < p`; therefore field zero
implies integer zero, and fixed-width balanced-ternary uniqueness forces every
digit to zero. Lean now exports this model result as
`ShiftedTernaryCenteredZero.centered_zero_unique`, and the isolated production
opening artifact conditionally proves equivalence between the reduced gates
and all 124 source rows. Its explicit `ProjectiveSevenNonresidue` premise is
not yet discharged in the Nightstream Lean project; the related SuperNeo
extension-field theorem is evidence, not an imported refinement bridge. Lean
also proves that the shifted packed-Mod-5 equation fixes
`L = R = -1`, the inactive encoding of residue zero. What remains open here is
the materialized fixed selector: the estimator has not yet been refined to one
selector-gated constraint system in which every inactive-slot equation
instantiates those results. Until that bridge is closed, the fixed total above
is an exact cost formula only, not a selector-soundness result.

#### SIS-aware balanced-opening boundary

This pass changes representation, not the source verifier. One field opening
is represented by 41 centered committed coordinates; its 41 digit wires alias
those coordinates, and its 41 sign indicators plus 40 internal borrow wires
use binary coordinates. Relative to the conservative replay, the exact
logical reduction:

- retains 82 source product rows;
- retains 41 centered obligations;
- omits 42 proved-redundant source rows; and
- omits 81 proved-redundant common binary gates.

That logical result saves exactly **123 encoded rows per opening**, or
1,459,149 rows across 11,863 production openings. The concrete fixed schedule
then groups all 486,383 centered obligations deterministically within their
stages into 243,191 residual-pair rows and one ordinary tail, saving another
243,191 rows. The combined opening lowering therefore saves 1,702,340 rows in
this fixture. The lowering preserves the candidate source witness exactly
instead of normalizing a second opening from the field value.

The three dominant SIS-backed stages now have these exact snapshots:

| SIS-backed stage | Source rows x columns | Encoded rows x columns | Ordinary fields | SIS fields | SIS aliases | SIS binary auxiliaries | Centered coordinates |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nifs.pi_ccs.output_message_hashes` | 852,733 x 839,151 | 728,983 x 890,658 | 1,516 | 6,791 | 278,431 | 550,071 | 340,587 |
| `nifs.pi_rlc.verify.projection_binding.sis_digest` | 472,218 x 464,770 | 414,412 x 516,484 | 1,516 | 3,724 | 152,684 | 301,644 | 214,840 |
| `nifs.pi_ccs.fresh_claim_hashes` | 177,605 x 174,909 | 170,883 x 226,612 | 1,516 | 1,348 | 55,268 | 109,188 | 117,424 |

The encoded-row decomposition is additive and separately pinned:

| SIS-backed stage | Ordinary pair rows | SIS pair/tail rows | Retained fallback rows | Poseidon2 S-box rows | Total encoded rows |
|---|---:|---:|---:|---:|---:|
| Π_CCS output-message binding | 31,078 | 139,216 | 557,227 | 1,462 | 728,983 |
| Π_RLC projection SIS digest | 31,078 | 76,342 | 305,530 | 1,462 | 414,412 |
| Π_CCS fresh-claim binding | 31,078 | 27,634 | 110,709 | 1,462 | 170,883 |
| **Total** | **93,234** | **243,192** | **973,466** | **4,386** | **1,314,278** |

The same snapshots record 4,436 / 4,432 / 4,432 linearly derived source
columns and 4,386 gadget-derived source columns in each respective stage.
Those source columns are reconstructed, not separately committed. The old
canonical lowering charged these three stages 182,863,022 rows; the current
representation charges 1,314,278 while preserving their source acceptance.

The opening reduction has an exact isolated Rust/Lean bridge. Rust first
matches the full 124-row source program and the concrete shared-slot aliases;
the generated schema-3 artifact then binds the production gate polynomial,
ordered LEFT/RIGHT matrix roles, and exact schedule to 20 residual-pair rows,
one ordinary centered tail, and 82 retained product rows. Lean proves that
these 103 physical rows encode the 123 logical obligations and conditionally
accept exactly when all 124 source rows accept, under
`ProjectiveSevenNonresidue`, verifier-fixed one, and the structural field/digit
alias. This permission is specific to the opening program. Seeded `Phi81`
arithmetic remains on the exact expanded-source path, and no digest is treated
as authority.

#### Historical pre-SIS-aware tree

The following table is the correct-base 214,119,163-row estimator snapshot
immediately before SIS-aware and ordinary-private lowering. It is retained as
formula history; no complete relation with those dimensions was materialized.
Its shares and encoded dimensions are not the current 6,184,892-row fixed
formula.

| Major implementation-tree cost center | Raw rows | Encoded columns | Encoded rows | Fixed-row share |
|---|---:|---:|---:|---:|
| `nifs.pi_ccs.output_message_hashes` | 852,733 | 77,571,680 | 104,531,214 | 48.82% |
| `nifs.pi_rlc.verify` | 538,447 | 48,355,475 | 65,160,031 | 30.43% |
| `nifs.pi_ccs.fresh_claim_hashes` | 177,605 | 15,521,480 | 20,915,848 | 9.77% |
| `f_prime.output_parent_authority_hash` | 254,915 | 3,448,500 | 4,646,397 | 2.17% |
| `nifs.pi_ccs.running_parent_hash` | 254,937 | 3,447,740 | 4,645,377 | 2.17% |
| `nifs.pi_ccs.allocate_and_normalize` | 183 | 3,038,005 | 4,061,333 | 1.90% |
| `nifs.pi_rlc.shape.allocate_parent_and_children` | 165 | 2,263,280 | 3,025,723 | 1.41% |
| `nifs.pi_rlc.challenge` | 126,651 | 1,229,955 | 1,710,699 | 0.80% |

##### What the 214M-row snapshot was measuring

The pre-SIS-aware lowering did not consume the source builder's centered-unit,
balanced-ternary, or `SeededPhi81` trace metadata. It therefore classified the
SIS witness columns as unrestricted Goldilocks fields. Every such field
received a 95-column canonical slot plus 32 canonicality rows.

The exact fixed-fixture snapshots are:

| SIS-backed stage | Canonical source fields | Value-encoding rows | Canonicality rows | Fallback + S-box rows | Total encoded rows |
|---|---:|---:|---:|---:|---:|
| Π_CCS output-message binding | 816,544 | 77,571,680 | 26,129,408 | 830,126 | 104,531,214 |
| Π_RLC projection SIS digest | 448,504 | 42,607,880 | 14,352,128 | 455,952 | 57,415,960 |
| Π_CCS fresh-claim binding | 163,384 | 15,521,480 | 5,228,288 | 166,080 | 20,915,848 |
| **Total** | **1,428,432** | **135,701,040** | **45,709,824** | **1,452,158** | **182,863,022** |

These three stages are 85.40% of the 214,119,163-row fixed estimate. Their
1,428,432 canonical fields alone account for 181,410,864 rows: 95 value
columns plus 32 canonicality rows per field. That is a lowering cost, not a
proof that the protocol needs 181M independent constraints.

The current lowering consumes the balanced-opening metadata, applies the exact
123-row opening reduction, and classifies ordinary private fields into the
41-coordinate centered representation. It preserves the accepted source word
and leaves seeded `Phi81` rows expanded. The historical snapshot remains
useful evidence of the canonical-field tax, not the live fixed-relation size.

#### `Pi_RLC.verify`

The algebra cost center uses stable full paths from
`pi_rlc_circuit::stage`. Organizational nodes are emitted even when they own
zero rows or columns; `aggregate_prefix` therefore exposes this complete tree,
and the full-relation test reconciles every parent against its immediate
children.

```text
nifs.pi_rlc
├── challenge
├── shape
│   ├── allocate_parent_and_children
│   ├── output_parity
│   ├── parent
│   └── d_pad
└── verify
    ├── fold_wires.{commitment,adv,x,y_ring,y_zcol}
    ├── consistency.{s_col,fold_digest}
    ├── projection_binding
    │   ├── domain
    │   ├── combined.{commitment,adv,x,y_ring,y_zcol}
    │   ├── quotient.{commitment,adv,x,y_ring,y_zcol}
    │   ├── sis_digest
    │   └── transcript_beta
    ├── projection_shared.{beta_ladder,rho_evaluations}
    ├── identities.{commitment,adv,x,y_ring,y_zcol}
    └── padding.{x,y_ring,y_zcol}
```

No new algebra is introduced by these boundaries: combined values and quotient
advice retain their transcript order, the beta ladder and rho evaluations stay
shared, and padding remains explicit verifier glue.

The tree is quantitative, not merely descriptive. The current major nodes are:

| Pi_RLC tree node | Raw rows | Encoded columns | Encoded rows |
|---|---:|---:|---:|
| `nifs.pi_rlc` | 666,223 | 2,058,157 | 1,240,656 |
| `challenge` | 127,611 | 370,383 | 198,567 |
| `shape` | 165 | 834,678 | 417,414 |
| `shape.allocate_parent_and_children` | 90 | 834,678 | 417,339 |
| `shape.output_parity` | 70 | 0 | 70 |
| `shape.parent` | 5 | 0 | 5 |
| `shape.d_pad` | 0 | 0 | 0 |
| `verify` | 538,447 | 853,096 | 624,675 |
| `fold_wires` | 0 | 0 | 0 |
| `consistency` | 330 | 0 | 308 |
| `projection_binding` | 475,533 | 534,114 | 423,956 |
| `projection_binding.domain` | 8 | 0 | 8 |
| `projection_binding.combined` | 92 | 0 | 92 |
| `projection_binding.quotient` | 199 | 0 | 199 |
| `projection_binding.sis_digest` | 472,218 | 516,484 | 414,412 |
| `projection_binding.transcript_beta` | 3,016 | 17,630 | 9,245 |
| `projection_shared` | 1,892 | 69,618 | 36,507 |
| `identities` | 59,396 | 249,364 | 162,688 |
| `padding` | 1,296 | 0 | 1,216 |

The test prints every node and leaf as a `PI_RLC_TREE` record and proves that
each parent equals the sum of its immediate children across source rows,
source columns, encoded rows, encoded columns, column classes, fallback rows,
gate families, Poseidon2 counts, product-sum counts, the exact duplicate-Boolean
omission count, and hash histograms. Exact
totals stay in that executable audit rather than being copied into every source
header. The complete Π_RLC lifecycle is **1,240,656 rows x 2,058,157
columns**. Within it, the centered-field and common-Boolean reductions lower
projection binding to 423,956 rows, and the algebra-verification subtree is
**624,675 x 853,096**.

The identity family contains exactly 31 active identities: 18 commitment, 5
public-X, 6 `y_ring`, and 2 `y_zcol`; advice is absent in this fixture. Each
identity has this mechanically pinned formula:

| Identity phase | Source rows x columns | Ordinary retained fields | Projected source fields | Synthetic ProductSum fields | ProductSum rows | Encoded rows x columns |
|---|---:|---:|---:|---:|---:|---:|
| 15 input evaluations | 1,620 x 1,620 | 30 | 1,590 | 60 | 90 | 4,515 x 6,930 |
| Parent-output evaluation | 108 x 108 | 2 | 106 | 4 | 6 | 301 x 462 |
| Quotient evaluation | 106 x 106 | 2 | 104 | 4 | 6 | 301 x 462 |
| 15 rho/input products | 75 x 75 | 0 | 75 | 0 | 0 | 0 x 0 |
| Quotient/Phi products | 5 x 5 | 0 | 5 | 0 | 0 | 0 x 0 |
| Two final limb checks | 2 x 0 | 0 | 0 | 2 | 4 | 131 x 190 |
| **Per identity** | **1,916 x 1,914** | **34** | **1,880** | **70** | **106** | **5,248 x 8,044** |

Across all 31 identities, the compact plan is **162,688 rows x 249,364
columns**, down from the historical expanded-source lowering of 7,261,502 x
5,389,350. It removes 7,098,814 rows and 5,139,986 columns.

The implementation arithmetic boundary is exact for the recorded trace but
deliberately layered. Rust replays every source row, validates SSA order,
rejects overlaps and escaping temporaries, and binds the exact retained-output
and ProductSum schedule. This makes the source trace a differential arithmetic
reference, not semantic authority for the protocol. Lean proves arithmetic
preservation for the corresponding abstract source/emitted model and checks
the generated identity schema and cost formulas. The concrete refinement from
the generated production rows, assignments, and decoders into that abstract
model remains open. A separate bridge from the two retained base-field limbs
to the polynomial-evaluation statement used by the exact-or-bad-root security
reduction also remains open; the arithmetic theorem alone is not a
deterministic projection-security proof.

#### `Pi_RLC.challenge`

The 21,367,779-row challenge figure is historical: it predates product
elimination and the fixed `j..j+10` selection window. Those two reductions
produced the later 1,649,454-row current-base snapshot. That second figure is
also historical now because it still lowered each Mod-5 block generically.

Lean proves that the product/aggregate block is equivalent to three direct
aggregate ProductSum equations per output position; the pointwise guarded form
is a separate optional consequence of one-hotness. Lean also proves that, with
at most ten rejections, output `j` can only select chunk `j..j+10`. Rust
validates the exact source product definitions and three binding rows before
projecting the product columns and emitting those aggregates.

```text
nifs.pi_rlc.challenge
├── transcript
│   ├── bind_outputs_digest
│   ├── rho_domain_separator
│   ├── digest_rounds
│   └── lane_bit_decomposition
└── sampler
    ├── initialize
    ├── chunk
    │   ├── accept
    │   │   └── packed
    │   │       ├── tree_bit_pairs
    │   │       ├── product_aggregate
    │   │       └── root_binding
    │   ├── mod5
    │   │   └── packed
    │   │       ├── low_bit_pairs
    │   │       ├── high_bit_pair
    │   │       └── residue_pair
    │   └── symbol_and_prefix
    ├── acceptance_bound
    └── selection
        ├── initialize
        ├── one_hot
        ├── products
        └── bind
```

The same tree audit used for Π_RLC algebra checks this challenge hierarchy:
every named node and zero-cost organizational checkpoint exists, and every
parent equals its immediate children across source classes, lowered rows and
columns, gate families, Poseidon2 counts, ProductSum counts, and hash
histograms.

| Challenge leaf | Occurrences | Encoded columns | Encoded rows | Omitted exact Boolean rows |
|---|---:|---:|---:|---:|
| Output-digest binding | 1 | 7,052 | 3,698 | 0 |
| Rho domain separator | 15 | 3,526 | 1,849 | 0 |
| Sample initialization | 15 | 0 | 0 | 0 |
| Transcript digests | 60 | 264,450 | 138,675 | 0 |
| Lane bit decomposition | 240 | 25,200 | 13,680 | 15,360 |
| Acceptance tree bit pairs | 960 | 13,440 | 6,720 | 0 |
| Acceptance product aggregate | 960 | 0 | 960 | 0 |
| Acceptance root binding | 960 | 960 | 960 | 0 |
| Packed Mod-5 low-bit pairs | 960 | 11,520 | 5,760 | 0 |
| Packed Mod-5 high-bit pair | 960 | 960 | 960 | 0 |
| Packed Mod-5 residue pair | 960 | 1,920 | 960 | 0 |
| Chunk symbol and prefix | 960 | 0 | 0 | 0 |
| Acceptance bound | 15 | 45 | 45 | 45 |
| Selection initialization | 15 | 0 | 0 | 0 |
| Selection one-hot | 810 | 8,100 | 4,860 | 8,100 |
| Selection products | 810 | 0 | 0 | 0 |
| Selection accept aggregate | 810 | 0 | 810 | 0 |
| Selection prefix aggregate | 810 | 0 | 810 | 0 |
| Selection symbol aggregate | 810 | 33,210 | 17,820 | 0 |
| **Challenge total** |  | **370,383** | **198,567** | **23,505** |

The current profiler attributes the aggregate-acceptance subtree to 960
chunks; these totals remain diagnostic until the recursive physical image is
artifact-checked. It reports a source side of **3,840 rows x 1,920 columns**:
bitness, the
accepted zero branch, the accepted canonical-inverse branch, and the rejected
inverse-zero branch over the accept and inverse columns. Its emitted side is
reported as **8,640 rows x 14,400 coordinates**, split into 6,720 shared quadratic
bit-pair rows over 13,440 tree outputs, 960 radix-3 ProductSum aggregates, and
960 root bindings that retain the accept coordinate. The inverse is projected
through an explicit canonical decoder, and the trace validator rejects any
use outside that exact four-row source block. In the active Lean project,
`Sampler.Chunk.Acceptance.Aggregate` independently proves the abstract
nine-row candidate sound, complete, uniquely extendable, and necessary by
family. The proof is conditional on `EuclidPrime goldilocksP` and
`SevenNonresidue`. A fresh schema-2 artifact fixes the active arity-56 gate,
40 role bindings, nine normalized rows, and exact 25-term specialization; a
handwritten Lean evaluator proves those generated rows equivalent to the
independent relation. The recursive
outer-decoder/physical-placement and inactive fixed-selector bridges remain
separate open obligations, so the leaf theorem authorizes zero production row
removals by itself.

The current outer-image exporter has been exercised on the real production
emitter with the 64-chunk private sampler fixture, including sparse derived
terminal bits, removed-definition provenance, singleton/translated Boolean
owners, and selected physical CCS rows. That is not a 960-chunk F' bridge. On
2026-07-16, after removing an eager all-matrix reservation that would have
requested roughly 31 GB at the previously observed recursive dimensions, an
actual recursive materialization was attempted under the mandatory 300-second
cap. Release compilation used 37.91 seconds; the test then ran for roughly
4 minutes 22 seconds and was killed at the cap before producing an outer-image
census. Peak RSS was not measured. The temporary caller was removed. A
selective production-row audit or a materially cheaper complete materializer
is therefore required before generating the 960-chunk artifact.

The deprecated aggregate artifact is not a fallback: it hardcodes gate arity
48, while the active schema has arity 56, and its 64-chunk fixture forces every
input bit into a singleton retained coordinate. Its generated data and proof
wrappers provide no active conformance evidence. The fresh arity-56 leaf is
now checked separately; only the recursive outer-image refinement remains to
close physical placement.

The Mod-5 subtree accounts for 960 independently validated chunks. Its source
side is exactly **19,200 rows x 18,240 columns**: 12 rows/12 columns for the
low-bit family, 4/3 for the high-bit family, and 4/4 for the residue family per
chunk. Its emitted side is exactly **7,680 rows x 14,400 coordinates**, split
as 5,760 low-bit-pair rows, 960 high-bit-pair rows, and 960 residue-pair rows.

The challenge profile also pins 7,758 ordinary-private source fields, 26,169 linearly
derived source columns, 50,694 gadget-derived source columns, and 23,505 exact
duplicate Boolean source rows omitted in favor of their common gates. The
parent row equation is explicit:

```text
  198,567 =    11,745 common Boolean pair rows
          +        15 common Boolean tail rows
          +   158,514 ordinary centered pair rows
          +     1,050 ordinary centered tail rows
          +     1,785 retained generic fallback
          +     6,708 Poseidon2 x^7
          +     6,720 acceptance-tree bit pairs
          +       960 acceptance product aggregates
          +       960 acceptance root bindings
          +     5,760 packed Mod-5 low-bit pairs
          +       960 packed Mod-5 high-bit pairs
          +       960 packed Mod-5 residue pairs
          +       810 acceptance aggregates
          +       810 prefix aggregates
          +       810 symbol/output aggregates

   23,505 =    15,360 lane-decomposition coordinates
          +        45 acceptance-bound coordinates
          +     8,100 selection one-hot coordinates

  318,078 =     7,758 ordinary fields * 41 coordinates
```

Pairing resets at every physical stage and family. In particular, the 1,050
ordinary tails are not derivable from the even aggregate coordinate count: 240
come from the lane-decomposition stages and 810 from the per-symbol selection
bindings. The profiler stores pair and tail counts separately so no aggregate
formula can silently pair across either boundary.

Each omitted row is the normalized source equation `v * (v - 1) = 0`,
possibly with the two multiplicands exchanged. Rust omits it only when the
concrete source-column map expands `v` to the same singleton Boolean slot whose
common encoding gate already enforces that equation. Same-column near misses,
nonzero right-hand sides, coefficient changes, non-singleton mappings, and
rows already owned by another replacement remain fail-closed.

For each chunk, Rust replays all 20 source rows, checks the exact 19-column
layout and projected decoder, and tests the materialized eight-row/fifteen-
coordinate CCS relation. The generated Lean artifact records and drift-gates
the role-normalized 20 source rows, all six decoder definitions, the exact
active row schedule, matrix roles `0/44/45/54/55`, and the twelve sparse
polynomial terms. In the active `formal/nightstream-lean` project, Lean proves
the exact generated shape and degrees, normalizes the 20 generated source rows
to the readable candidate-zero relation, and proves the generated bit/residue
polynomials at explicit role points. It also proves that the generated high-bit
decoder has the exact expected role and evaluates to the independent
`derivedQuotientHigh` formula under an explicit low-coordinate alias boundary.
Independently, it proves the eight packed residual equations equivalent to the
sixteen scalar residual equations under the explicit `SevenNonresidue` premise.

This is an artifact-checked **isolated active-leaf schema**, not a closed
production row-removal refinement. Lean does not yet prove that the eight
materialized matrix rows evaluate at those role points, that the full recursive
decoder/linear-substitution image supplies the required chunk and quotient
coordinates, or that a fixed-selector inactive branch is sound. The older
theorems with names such as `generatedEmittedAccepts_iff_packed` live only in
the deprecated `formal/superneo-lean` project and do not contribute assurance
until their statements are independently re-established against the active
types and artifact. No production row removal is authorized by this leaf yet.

On the pre-Nebula-base snapshot, product elimination reduced the fixed
estimator formula from 79,363,506 to 59,767,986 rows, and the proved
`j..j+10` selection window reduced it again to 59,510,406 rows. Those totals,
the 21,367,779-row challenge, and the later 1,649,454-row challenge are
historical optimization evidence, not current measurements. The current
challenge subtree is **198,567 rows x 370,383 columns** inside the
6,184,892-row fixed formula. This audit relation is deliberately distinct
from the shipped selective folded relation described below: it remains the
differential reference used to expose accidental verifier cost without
treating either it or a cheaper compiler as semantic authority.

The legacy manual projection image is a **cost prototype**, not `enc(F')`:
that shell reserves K-mul slots without emitting the equations that relate
them to verifier inputs and outputs. The shipped selective compiler described
below does not rely on those slots. The deliberately red gate
`folded_f_prime_kmul_slots_must_be_semantically_constrained` pins this gap.
The complete field-native F' R1CS remains an implementation-level arithmetic
reference. It is not authoritative for the protocol obligations or their
minimality.

## Open design questions for `enc(F')` — now answered quantitatively

The 2026-07-06 inventory (every figure reproduces from the tests named
below and in the "Reproduce" section) answered these:

- *What is the low-norm assignment for an F' execution?* The production
  shell measures **94,330,948 committed bits per recursive step**
  (`system_phase_1_4a_fibonacci_structure::phase_1_4a_production_config_pins_emitter_counts`):
  465 ring-action pairs × 196,992 bits (97 %, dominated by each pair's
  D² partial-product region), 7,100 K-mul slots × 384 bits (2.7M), and
  ~135k for the state-hash trace + boundary + counters.
- *Which values are public `x`?* Plain F′ uses the 257-slot `enc_inst`
  boundary. Nebula F′ appends the current `S_mem.x`, segment-open bit, and
  `D_pre` bits; the following recursive step consumes that suffix together
  with the same claim's `adv` (HyperNova/Nebula one-step delay).
- *Which are derived, never committed?* Canonical-u64 lanes — rows
  substitute `Σ 2^i · z[bit]` directly; already implemented.
- *Do digit encodings help?* No: the measured SignedDigit ladder
  (`perf_ring_action_low_norm_prototype`: 3,079 full-field / 39,853
  SignedDigit / 200,071 U64 cols per pair) is **invalid on production
  wires** — the ring action acts on commitments, which are full-range
  mod q; only ρ is low-norm (a 1.6 % saving). The earlier figures once
  quoted here ("F' ≈ 10M rows post-optimization", "~3.2B bit slots
  naive") are historical estimates superseded by this measurement.

**REGIME DECIDED (Nico, 2026-07-08): the folded regime (Road A).**
F' becomes a foldable low-norm instance each step, with the ring-action
obligations discharged by the projection check (candidate E below).
The terminal-Spartan road (H) remains the compression story for
proof-size/portability later, but is not the induction mechanism.
Integration order: (1) β transcript schedule — **DONE**: native
`pi_rlc` owns it on both prove and verify paths (recompute per-lane
quotients from authoritative inputs → absorb c* and every q_lane →
squeeze β; wire-identity check fails closed if the mixer is not the
ring action), the NIFS.V circuit replays it bit-for-bit and now enforces
the complete product-commitment (`c + adv`) projection identities using
the exact transcript-bound q and β wires, and
`tests/system/rlc_projection.rs` drives a real fold through the
schedule and the projection-trace encoders with zero residual,
(2) complete the field-native implementation relation: the `c + adv`
product commitment folds through PiCCS/PiRLC/PiDEC and is opened by the
terminal relation; the recursive relation consumes the prior fresh
claim's suffix/`adv`, enforces the delayed `NebulaLane` transition, composes
current `S_mem`, and projection-checks the c/adv, X, and y clients. Shape-only
synthesis covers base, bootstrap-recursive, and steady-recursive execution.
The accumulator handle now reuses the already-computed running-parent CE
digest after native and in-circuit NIFS.V verify strict Pi_DEC consistency
between that parent and every child. The implementation removed the repeated
child rehashes, and its child-tamper tests still fail after rebuilding the
compact handle. That evidence does not prove the hashes semantically
redundant: independent Π_CCS/Π_RLC/Π_DEC refinement and a necessity/derivation
theorem are still required before this removal is treated as protocol-safe.
The R2 authority mechanics and the R3 selective compiler are implemented, but
the R3 cost gate is **not closed on the current base**. Five
witness-proportional claim/projection/leaf roles use independent
rank-2 seeded SIS/Ajtai maps followed by one independent short rank-1 map and
a domain-separated Poseidon2 digest. Each long map consumes the same 41
centered unit digits that encode its authoritative source fields, rather than
a second 64-bit serialization. The v3 digest envelope binds the role, field
count, and primary rank. `CscWithSeededPhi81` keeps both maps compact through
CCS and SuperNeo evaluation. The selective compiler lowers Poseidon2 S-boxes,
projection evaluations, K multiplication, rejection selection, and centered
PiDEC checks directly instead of committing their R1CS temporaries. Full field
values without an existing canonical decomposition use 41 balanced-ternary
digits in `{-1,0,1}`; canonical-u64 fields retain their shared 64-bit slots.
This is still `w = 1` at the committed-coordinate boundary: radix 3 lives in
the verifier-owned matrix coefficients, while every witness digit has norm at
most one. Private final Poseidon outputs are substituted linearly, five product
pairs share one direct CCS row, and long evaluations use telescoping
accumulators. K dot products use the exact Karatsuba sums `P`, `Q`, and `R`
instead of retaining every per-term K output. SplitNc checks FE over the row
domain and NC over the assignment domain, so the selective relation does not
need an identity matrix or square row padding.

The correct-base rerun does not reproduce the older fixed-point claims. The
reduced-profile test currently exits with `CompileBudgetExceeded` at a
19,624,600-coordinate candidate (19,624,626 minimum including the public
boundary) against the 16M cap, before it can reach its embedded
10,000,318-row / 11,516,688-coordinate snapshot. The production preflight's
stale expectation is 15,730,104 coordinates; the correct-base census before
the subsequent authority additions was already 18,376,624, and the current
census is 28,047,523. Consequently the older 2,486,540 / 9,613,188 and 2,819,360 /
15,612,210 fixed points are historical, not current passing measurements.
This is an open selective-lowering regression; the budget must not be raised
to make it disappear.
R4's shipped encoder
and R5's terminal induction are **DONE**: `NebulaFPrimeChainBuilder` deposits
the fixed relation with serial `K=1`, recursive steps consume the prior claim's
delayed suffix, finalization consumes the trailing claim, and the terminal-only
verifier accepts the final accumulator plus terminal fold without the audit
history. The active
`r4_shipped_encoder_verifies_multistep_memory_chain` test traverses all three
arms over three one-step segments and rejects link, suffix, lane, and history
tampering. Focused delayed-suffix tests cover the absent-`D_pre` interior
encoding without another production-sized fold. The plain shipped
encoder is exercised by `r1cs_stateful_linked_fibonacci_chain_verifies_end_to_end`.
The active R5 gate `multi_chunk_f_prime_chain_must_verify_terminal_only`
additionally rejects a changed pre-final running commitment, so earlier folded
history remains load-bearing without audit replay.
Legacy and generic F' frontends remain terminal-only fail-closed. The old
14,040,452-bit manual shell remains prototype evidence only.
Lemma 5 carries an author self-review whose one
novel claim (a Φ(β) = 0 completeness caveat) was **refuted by external
review** and is retained in the note as a correction record — the
honest identity holds identically at roots of Φ, and Φ_81 has no roots
in K at these parameters anyway. The non-author review remains an open
tracked flag, proceeding at Nico's direction; the refuted self-review
finding is itself the argument for keeping that flag.

## Candidates, costed (ring-action term per step)

| # | Candidate | Bits/step | Status |
|---|---|---|---|
| A | U64 status quo | 91.6M (94.3M total) | works; ~1,650× the S_mem app circuit it recurses |
| B | SignedDigit as measured | — | invalid: operands are full-range commitments |
| C | Mixed (ρ = SignedDigit{5}, rest U64) | 90.1M | valid; saves 1.6 % — not a lever |
| D | Digit-decompose c, act on digits | ~260M | valid; strictly worse (14 SignedDigit pairs vs 1 U64 pair, ×2.8) |
| E | Projection check: verify `Σ_i ρ_i·c_i = out (mod Φ)` as `Σ ρ_i(X)c_i(X) = q(X)Φ(X) + out(X)` at a post-commitment `β ∈ K` | The primitive is measured at ~21k vs ~196k bits per pair. The current reduced candidate is 19,624,600 coordinates and the production census is 28,047,523; both exceed the 16M cap. The old 14,040,452-bit manual shell remains a non-authoritative reference. | The current NIFS.V/F′ projection implementation covers `c + adv`, X/y projection, delayed lane transition, current `S_mem`, and terminal-only lifecycle induction. Its semantic sufficiency remains an independent Lean refinement obligation. Cost closure remains open: the current correct-base selective tests fail rather than establish the former sub-16M claim. Lemma 5's maximum-geometry census is `P=2,250`, batched `J=150`; conservative `J≤2,250`. |
| F | Fewer pairs (arity/κ trades) | linear only | doesn't touch the 197k/pair |
| G | SIS accumulators (C14/L2) | A role-specific rank-2 map binds the authoritative 41-trit encoding; an independent short rank-1 map compresses its 108-field output before Poseidon2. | **Adopted for five R2 binding roles**, with compact seeded matrices, native/circuit parity, stage-tamper tests, concrete rank-2/rank-1 estimates, and security-note Lemma 6's hash-then-FS reduction. Replacing the carried `D` chains remains deferred. |
| H | Terminal-proof regime (PR5): never commit F' | 0 per step | field-native cost once per chain (~1–3M-constraint relation); sidesteps enc(F') entirely |

Bottom line: E reduced the ring-action wall, the verified-parent handle removes
the child hash chain in the current implementation, and the SIS/selective
compiler implements the R2/R3 mechanics. Whether the removed hash family was
semantically redundant remains open at `FPR-OBLIGATION-NECESSITY`. R4-R6
consume that relation through the shipped encoder and
terminal-only memory induction. The current selective compiler does **not** fit
the unchanged 16M ceiling; its reduced and production cost gates fail. Generic
gadget-native lowering therefore remains the differential reference for the
current implementation. The source R1CS trace remains the exact arithmetic
reference for that implementation only; neither artifact is the protocol
specification or a safe baseline for retaining checks. The balanced-opening
and projection-identity reductions each have exact Rust trace validation and
Lean model-level arithmetic-preservation results; they do not delegate
authority to a digest. For projection identities, the concrete refinement from
generated production rows, assignments, and decoders into the abstract Lean
model remains open. Projection security additionally requires the open
two-limb-to-polynomial bridge. The fixed selector remains a cost formula until
its gated relation is materialized and refined. The old manual shell remains a
non-authoritative reference.

The generic-lowering tax is now measured on the current emitted object, not
just the C14 toy: the complete current NIFS.V source relation over an honest
two-fold chain at the small direct-CCS app shape is **5,934,125 field cols /
5,893,265 rows / 45.4M nnz**, and its complete low-norm lowering is
**371,089,193 committed bits / 376,982,457 rows** — 62.5 bits per field
col, 64.0 rows per row, satisfiability-checked on both sides
(`perf_lowered_nifs_v`). Wire-level lowering without selective commitment
is therefore not a road: it is ~26× the shell cost model and ~4× the D²
shell it was meant to replace. Any completion must keep the bulk of the
verifier's wires derived (row-substituted linear forms), committing bits
only at range-checked and hashed boundaries — which is what the one-bit
audit above assumes and what the shell did by hand.

## Reproduce every number

```bash
# Complete fixed F′ source/gadget dimensions and the full stage/cost tree:
cargo test -p neo-fold-clean --release --test f_prime_full_relation \
  complete_recursive_relation_folds_one_fresh_instance_and_binds_the_application \
  -- --exact --nocapture

# Current reduced cost gate: fails at a 19,624,600-coordinate candidate > 16M.
cargo test -p neo-fold-clean --release --test nebula_f_prime \
  road_a_reduced_profile_fixed_point_stabilizes_within_budget -- --exact --nocapture

# Current production census: 28,047,523; correct-base predecessor: 18,376,624;
# stale embedded expectation: 15,730,104.
cargo test -p neo-fold-clean --release --test perf_nebula \
  nebula_v3_targets_folded_f_prime_production_preflight -- --exact --nocapture

# R4 shipped encoder over two multi-step memory segments:
cargo test -p neo-fold-clean --release --test nebula_f_prime \
  r4_shipped_encoder_verifies_multistep_memory_chain -- --exact --nocapture

# R4 plain multi-step encoder through the encoded F' audit relation:
cargo test -p neo-fold-clean --release --test system_r1cs_compiler_stateful \
  r1cs_stateful_linked_fibonacci_chain_verifies_end_to_end -- --exact

# R5 final accumulator + latest fold, with no audit-history authority:
cargo test -p neo-fold-clean --release --test nebula_f_prime \
  multi_chunk_f_prime_chain_must_verify_terminal_only -- --exact --nocapture

# Complete current NIFS.V source relation, low-norm lowered (371M bits, 62.5 bits/col):
cargo test -p neo-fold-clean --release --test perf_lowered_nifs_v -- --ignored --nocapture

# Historical manual-shell cost model (not complete enc(F')):
cargo test -p neo-fold-clean --release --test system_phase_1_4a_fibonacci_structure \
  phase_1_4a_production_config_pins_emitter_counts -- --nocapture

# Authoritative recursive F' gadget census after commitment projection:
cargo test -p neo-fold-clean --release --test system_phase_1_3d_step_parity \
  phase_1_3d_kmul_ring_action_coverage_full_step_three_way_parity -- --nocapture

# Encoding ladder (3,079 full-field / 39,853 SignedDigit / 200,071 U64 cols per pair):
cargo test -p neo-fold-clean --release --test perf_ring_action_low_norm_prototype -- --nocapture

# 156,740-bit canonical shipped image (state-hash authority only):
cargo test -p neo-fold-clean --release --test system_fibonacci_f_prime_layout_budget -- --nocapture

# Candidate C14 primitive, including one final Poseidon2 digest:
cargo test -p neo-fold-clean --release --test reductions_accumulator_sis -- --nocapture

# S_mem app-circuit comparison point (55,434 rows / 55,418 cols):
cargo test -p neo-fold-clean --release --test perf_nebula -- --ignored --nocapture \
  nebula_v3_targets_structure_snapshot
```
