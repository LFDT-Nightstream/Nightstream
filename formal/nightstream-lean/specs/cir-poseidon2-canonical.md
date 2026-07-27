# CIR-POSEIDON2-CANONICAL — canonical width-8 Poseidon2 permutation encoding

```text
property_id: CIR-POSEIDON2-CANONICAL
claim:
  The selected width-8 Goldilocks Poseidon2 permutation has a canonical
  Lean-owned never-materialize encoding whose round schedule is a closed term:
  both linear layers are concrete matrices, the 8/22 round structure is
  instantiated, every S-box index resolves to exactly one (phase, round, lane)
  triple, and the resulting row program is 352 rows over 344 auxiliary columns
  with both figures folded from receipts. Carried combinations reference at
  most 31 columns anywhere in the permutation: a full round resets support to
  its eight fresh S-box outputs, a partial round adds exactly one, and the
  constant wire adds the last.
assumptions:
  - Column 0 is the constant-one wire; round constants ride it rather than
    costing rows.
  - Canonical residues for all assignments (`Satisfies`, `lcEval`).
  - Poseidon2 and its parameters are production choices, not paper-derived.
    `neo-params` owns width/rate/capacity/digest/seed; p3 and the Rust circuit
    own `x^7`, the 8/22 split, and both linear layers.
non_goals:
  - NOT conformance to the RUST implementation. Semantic conformance is proved
    against `Poseidon2Reference.referencePermutation`, a Lean function
    transcribed statement-for-statement from
    `absorb_words_then_permute_values`. That the Lean reference and the Rust
    routine denote the same function is transcription fidelity, not a theorem —
    POSEIDON2-RUST-CONFORMANCE. The gain is that the trusted surface shrinks
    from "the encoding is right" to "these ~30 lines were copied correctly",
    and the reference is deliberately free of `Row`, `Layout` and `Satisfies`
    so it can be read against the Rust without understanding the encoding.
  - NOT a bit-for-bit claim. Round constants are universally quantified, never
    pinned. This is strictly stronger than fixing the 86 sampled values, but it
    means no digest equality against Rust is available
    (POSEIDON2-ROUND-CONSTANT-CONFORMANCE).
  - NOT an honest-completeness claim. No satisfying assignment is constructed,
    so soundness direction only. This is coupled to layout well-formedness
    below: an honest witness cannot exist for a layout whose ports alias the
    auxiliary range, so the two must be closed together.
  - NOT a nonzero-coefficient count. The support bound bounds the columns a
    combination references; it does not yet total the coefficients, which is
    what a fair density comparison against production needs.
  - NOT minimality in any class.
  - NOT layout well-formedness. `Layout` is unconstrained: nothing forbids
    `inputPort`/`outputPort` from aliasing the auxiliary range. Auxiliary
    columns are proved mutually distinct, ports are not proved disjoint from
    them.
paper_sources:
  - none. Neither SuperNeo nor HyperNova selects a hash; HyperNova
    Construction 2 takes "a cryptographic hash function" and Appendix B takes
    an abstract random oracle. Citing either as authority for Poseidon2 would
    be false.
rust_surfaces:
  - crates/neo-fold-clean/src/engine/ccs_native/poseidon2.rs
    (`absorb_words_then_permute_values` for the round order; `value_apply_mat4`,
    `value_external_linear`, `value_internal_linear`, `internal_diag` for the
    matrices). Values only, per the CIR-POSEIDON2-CANONICAL provenance rule.
  - crates/neo-fold-clean/src/engine/r1cs_circuit/poseidon2.rs
    (WIDTH 8, HALF_FULL_ROUNDS 4, PARTIAL_ROUNDS 22, `enforce_sbox_x7`).
circuit_or_encoding_artifacts:
  - none. No row count, row layout, or column index is read from any generated
    artifact. 352 and 344 are theorems; 600 and 608 appear only as the
    comparison target recorded under CIR-POSEIDON2.
failure_class:
  An encoding that carries state symbolically but whose support is unbounded is
  not implementable at the claimed row count: it trades 42% fewer rows for
  combinations of unbounded width, and the row figure becomes meaningless. The
  recurrence is what forecloses that.
counterexample_or_witness:
  `partialState_mentions_fresh` and `partialState_zero_mentions_output` prove
  the bound is not vacuous — each partial round's fresh output really is
  referenced by the next state, so support genuinely grows by one per round and
  30 is an upper bound on something nonempty. Without them
  `partialState_mentions_subset` would hold of a state that mentioned nothing.
lean_theorems:
  - Poseidon2RoundInduction.canonicalProgram_computes_reference
  - Poseidon2RoundInduction.{initialState_eval,partialState_eval,terminalState_eval}
  - Poseidon2RoundInduction.{output_eq_sbox7,scheduleOf_initial,scheduleOf_terminal}
  - Poseidon2Eval.{lcEval_applyMatrix,lcEval_scale,lcEval_addConstant,rawSum_scale_mod}
  - Poseidon2Matrices.{mat4_nonzero,externalMatrix_nonzero,internalMatrix_nonzero}
  - Poseidon2Matrices.{internalDiag_half_inverse,internalDiag_neg_half}
  - Poseidon2Schedule.{sboxIndex_partition,initialSboxIndex_roundtrip}
  - Poseidon2Schedule.{partialSboxIndex_roundtrip,terminalSboxIndex_roundtrip}
  - Poseidon2Schedule.{canonicalProgram_length,canonicalProgram_cost}
  - Poseidon2Schedule.canonicalProgram_sbox_chains
  - Poseidon2Support.{mentions_applyMatrix,mentions_addConstant}
  - Poseidon2Support.{partialSupportList_length,partialState_mentions_subset}
  - Poseidon2Support.{partialSupport_bound,partialSboxInput_mentions_bound}
  - Poseidon2Support.{partialState_mentions_fresh,partialState_zero_mentions_output}
  - Poseidon2Support.{terminalState_zero_mentions_subset,terminalState_succ_mentions}
  - Poseidon2Support.scheduleOf_partial
axiom_report:
  [propext, Quot.sound] throughout. Every `decide`-closed fact — the matrix
  entry properties, the half-inverse checks, the index partition, all three
  roundtrips — depends on NO axioms. No theorem depends on Lean.trustCompiler
  and no module uses native_decide. Guarded in
  tests/Axioms/CanonicalPoseidon2{Matrices,Schedule,Support}.lean.
conformance_status:
  model-proved, structural only. Semantic conformance is open.
retest_commands:
  - cd formal/nightstream-lean && lake build
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
      tests.Axioms.CanonicalPoseidon2Matrices
      tests.Axioms.CanonicalPoseidon2Schedule
      tests.Axioms.CanonicalPoseidon2Support
```

## What the concrete schedule changed

`Poseidon2Program` left `Schedule` abstract deliberately, so allocation,
ownership and row counts could be proved independently of matrix arithmetic.
That factoring was right, but it meant 352/344 described a *family* of
hypothetical programs indexed by a schedule, none of which was known to exist.

`Poseidon2Schedule.canonicalProgram` is a closed term, so the figures now
describe an object.

**The row count itself did not change and was never schedule-dependent.**
`sboxRows` emits four rows whatever combination it is handed, so
`permutationProgram_length_eq` already held for every schedule. What was bought
is existence, and the prerequisite for round induction and the support
recurrence — not a better-justified 352.

## Round structure

Transcribed from `absorb_words_then_permute_values`:

```text
state = external(input)                            -- pre-layer, no S-box
4x:  sbox all 8 lanes (+ initial[r][lane]);  state = external(sbox)
22x: sbox lane 0 only  (+ internal[r]);      state = internal(sbox_in)
4x:  sbox all 8 lanes (+ terminal[r][lane]); state = external(sbox)
```

S-box index families, partitioning `[0, 86)`:

| family | indices | index map |
|---|---|---|
| initial full | `[0, 32)` | `round * 8 + lane` |
| partial | `[32, 54)` | `32 + round` |
| terminal full | `[54, 86)` | `54 + round * 8 + lane` |

`initialState` and `terminalState` are non-recursive because a full round
S-boxes every lane and each output is a fresh column. `partialState` *is*
recursive because lanes 1..7 are not S-boxed and flow onward. That asymmetry
is the whole content of the support recurrence.

## Matrices

```text
mat4     = circulant [2,3,1,1]
external = [[2·mat4,   mat4], [  mat4, 2·mat4]]
internal = J + diag(d)                       (J all-ones)
d        = [-2, 1, 2, 2⁻¹, 3, -2⁻¹, -3, -4]
```

Both are dense — no entry vanishes — which is what will later make the
coefficient count a product rather than a survey. `internalDiag_half_inverse`
and `internalDiag_neg_half` pin the two entries that are not small integers, so
a transcription slip in `2⁻¹` fails the build rather than silently changing the
permutation.

## Support recurrence

```text
full round      support = 8            (the round's fresh S-box outputs)
partial round r support ≤ 8 + r
entering terminal block   ≤ 30
plus the constant wire    ≤ 31
```

The bound is on *syntactic* support — the columns actually listed. Since
`mentions_map_scale` keeps a column listed even when its coefficient scales to
zero, this is an upper bound on true support, which is the safe direction.
`LinCombNormal.normalize` realizes it and `lcEval_normalize` makes using the
normalized form sound.

## Semantic conformance

`canonicalProgram_computes_reference` is the obligation both `Poseidon2Core`
and `Poseidon2Schedule` deferred. It states: for any layout, any constants, and
any canonical-residue assignment with the constant wire set, satisfying the
352-row program forces every output port to the reference image of the input
ports.

Three inductions in the Rust phase order, each composing two facts:

| fact | supplies |
|---|---|
| `lcEval_applyMatrix` | a linear layer computes the matrix-vector product, so emitting no row loses nothing |
| `sboxRows_chain` | the four emitted rows force the `1 → 2 → 3 → 6 → 7` chain |

Only the partial phase's induction step consults the previous state, mirroring
the fact that lanes 1..7 are not S-boxed. That is the same asymmetry that drives
the support recurrence, showing up twice.

**The trusted surface after this.** Before, "the encoding is Poseidon2" rested
on reading `Poseidon2Schedule` against the Rust and believing the schedule was
right. Now it rests on reading `Poseidon2Reference` — about thirty lines with no
`Row`, `Layout` or `Satisfies` in them — against
`absorb_words_then_permute_values`. That is a much smaller and much more
checkable obligation, but it is still an obligation, and it is not a theorem.

## Comparison against production — provenance stated

| | rows | auxiliary columns | provenance |
|---|---|---|---|
| production (`CIR-POSEIDON2`) | 608 | 600 | **measured**, artifact-checked |
| canonical (this property) | 352 | 344 | **derived**, model-proved |

Production is 600 SSA rows plus eight gated visible-output copies; canonical is
344 S-box rows plus eight terminal binding rows. The structures are parallel,
so the comparison is at the same boundary.

**This is not yet a claim that either encoding is better.** The row difference
is a materialization policy: production materializes every intermediate, the
canonical form carries them symbolically. Trading 256 rows for combinations of
up to 31 terms is only an improvement if the coefficient total moves the right
way, and that total is not yet derived. Nothing in this property licenses a
change to Rust.
