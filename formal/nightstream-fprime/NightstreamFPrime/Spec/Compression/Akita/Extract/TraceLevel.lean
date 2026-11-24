import NightstreamFPrime.Spec.Compression.Akita.Extract.PackingFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Recursion
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceSource

/-! A level of the opening suffix of Akita with the evaluation-trace opening method (ePrint
2026/1983, revision of 2026-11-23, Sections 5.3, 7.1 and 7.2, Corollary 10.32 and
Proposition 10.44 with its proof).

- `TraceClaim` is the claim of a trace level: `TraceStatement` without the matrices.
- `trace` is the opening method of a trace fold: `Witness.TraceSatisfies`, the weak opening
  `Weak.TraceOpens` and `traceForkCollision` with `traceExtract`. `trace_sound` is
  `traceExtract_opens_or_collision` for it.
- `committedTrace` is the `Tier` of a trace fold whose flat witness the next level commits to,
  with the checks `foldCheck` for the quotient-free reduction of `traceRelation`.
  `committedTrace_suffix` is `Tier.suffix_before` for it: the trace level over a suffix of the
  later stages, with `foldError` for the sum of the bounds of the draws.
- `clearTrace` is the `Tier` of a last trace fold, whose proof has the digit vectors in the
  clear, and `clearTrace_suffix` is `Tier.suffix_last` for it.

The link from a level to a next level with evaluation trace is in `TraceLink`. The quotient-lift
reduction, more than one claim or group, and the setup offloading are not in this file. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The claim of one trace level (Akita Definition 10.22, "The fixed folding statement", without
the matrices): the payloads `u_pub` and `v_pub`, the embedding of the fixed subfield, the weights
`χ_blk` and `a`, the functional `T_{ρ_pack}`, the factor `θ` and the reduced target. -/
structure TraceClaim {S : Shape} (O : OpeningShape S) (E : Type) [CommRing E] [Module Zq E] where
  payload : S.Payload
  openPayload : O.Payload
  subfield : E →+* Rq S.innerDegree
  block : Fin S.blocks → E
  position : Fin S.blockLength → E
  functional : Rq S.innerDegree →ₗ[Zq] E
  scale : E
  value : E

/-- The folding statement of a trace claim for the matrices of the level. -/
@[reducible] def TraceClaim.statement {S : Shape} {O : OpeningShape S} {E : Type} [CommRing E]
    [Module Zq E] (key : Key S) (openKey : OpeningKey O) (claim : TraceClaim O E) :
    TraceStatement O E :=
  ⟨key, openKey, claim.payload, claim.openPayload, claim.subfield, claim.block, claim.position,
    claim.functional, claim.scale, claim.value⟩

section Method

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [CommRing E]
  [Module Zq E]

/-- The method of a fold with evaluation trace for one claim (Akita Sections 5.3, 7.1 and 7.2):
`Witness.TraceSatisfies`, the weak opening `Weak.TraceOpens` with `κ̄_1 = 2ω` and
`β̄_∞ = Δ^cert_f` to the table that its source blocks decode to, and `traceForkCollision` with
`traceExtract`. The fold challenges are elements of `R_{q,d_A}`, and `weight` is their norm
`ω = ‖c‖_1`. -/
noncomputable def trace (key : Key S) (openKey : OpeningKey O) (weight : ℕ) :
    Method (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold)
      (Weak O) where
  Satisfies w claim challenge := w.TraceSatisfies (claim.statement key openKey) challenge
  Opens ω claim := ω.TraceOpens (2 * weight) (fold.sourceBound S) (claim.statement key openKey)
    (S.decode ω.source)
  merge claim first second w :=
    match traceForkCollision (claim.statement key openKey)
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩
        w.toCarried with
    | some solution => .inr (some (.inr solution))
    | none => .inl (traceExtract
        ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ w.toCarried)
  broken solution := solution.isSome

/-- Akita Corollary 10.32 for the method `trace` (`traceExtract_opens_or_collision`). The
challenge family has units as differences (Akita Definition 9.2, E1) and members of the norm `ω`,
and `b_z ≥ 2`. -/
theorem trace_sound {key : Key S} {openKey : OpeningKey O} {weight peak : ℕ}
    {family : Set (Rq S.innerDegree)} (units : UnitDifferences family)
    (sparse : ∀ challenge ∈ family, Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) :
    (trace (fold := fold) (E := E) key openKey weight).Sound fun _ => family := by
  intro claim first second w challenge firstMember secondMember different agree firstSatisfies
    secondSatisfies good
  have result := traceExtract_opens_or_collision units sparse base
    (fork := ⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩)
    ⟨firstMember, secondMember, different, agree, firstSatisfies, secondSatisfies⟩
    (Witness.TraceSatisfies.carried good)
  unfold trace
  rcases collision : traceForkCollision (claim.statement key openKey)
      (⟨fun i => ⟨(first i).1, (first i).2⟩, fun i => ⟨(second i).1, (second i).2⟩⟩ :
        TraceFork O fold) w.toCarried with _ | solution
  · rw [collision] at result
    simp only [collision]
    exact result.resolve_left (by simp)
  · simp only [collision]
    rfl

end Method

section Levels

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- A trace fold for one claim whose flat witness the next level commits to, as a `Tier` with
the first stage `start`. A slot of an answer of the oracle is one uniform element of `Slot`. The
successor source is a table, `OpeningShape.ofFlat` gives its digit vectors, and the checks are
`foldCheck` for the quotient-free reduction of `traceRelation`. A final claim `(r_2, v_W)`
is true for a table `w` when `v_W = w̃(r_2)`. -/
noncomputable def committedTrace (key : Key S) (openKey : OpeningKey O)
    (claim : Wire → TraceClaim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : Slot → Option (Rq S.innerDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
        Wire Coord Slot
      (Option (List E × E)) Table where
  method := trace key openKey weight
  claim := claim
  block b := some (block b)
  decode := fun _ => decode
  read := read
  digits := O.ofFlat fold
  check wire challenge table :=
    foldCheck (traceReduction ((claim wire).statement key openKey) challenge μ fits batch
      rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase (O.compressed fold)
  bits := (foldLosses (2 * S.innerDegree - 1) μ batch S.certifiedBase (O.compressed fold)).map
    workBits
  foldBits := foldWorkBits S.blocks (setBits false S.innerDegree)
  start := start

/-- A last trace fold for one claim, as a `Tier` with the first stage `start`. The successor
source is the family of the digit vectors, which the proof has in the clear, and the checks are
`foldCheck` for the quotient-free reduction of `traceRelation` and the flat witness of the digit
vectors. A final claim `(r_2, v_W)` is true when `v_W = w̃(r_2)` for this flat witness. -/
noncomputable def clearTrace (key : Key S) (openKey : OpeningKey O)
    (claim : Wire → TraceClaim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : Slot → Option (Rq S.innerDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
        Wire Coord Slot
      (Option (List E × E)) (Witness O fold) where
  method := trace key openKey weight
  claim := claim
  block b := some (block b)
  decode := fun _ => decode
  read := read
  digits candidate := candidate
  check wire challenge candidate :=
    foldCheck (traceReduction ((claim wire).statement key openKey) challenge μ fits batch
      rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) candidate.flat
  passes candidate result :=
    ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E candidate.flat)
  costs := foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase (O.compressed fold)
  bits := (foldLosses (2 * S.innerDegree - 1) μ batch S.certifiedBase (O.compressed fold)).map
    workBits
  foldBits := foldWorkBits S.blocks (setBits false S.innerDegree)
  start := start

variable {key : Key S} {openKey : OpeningKey O} {claim : Wire → TraceClaim O E} {μ : ℕ}
  {fits : O.cells fold ≤ 2 ^ μ} {batch : ℕ} {block : Fin S.blocks → Coord}
  {decode : Slot → Option (Rq S.innerDegree)} {read : Reader Wire} {weight start : ℕ}

/-- A committed trace fold satisfies the conditions of `Tier.Valid`. The hypotheses are: the
blocks have different coordinates, the challenge family of `R_{q,d_A}` has units as differences
(Akita Definition 9.2, E1) and members with `‖c‖_1 = ω`, `b_z ≥ 2`, `b⋆ < q`, and an answer of the
oracle has `μ'` slots or more and `s_rel` slots or more. -/
theorem committedTrace_valid [Nonempty Coord] {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch}
    {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (committedTrace key openKey claim μ fits batch rowsFit block decode read weight
      start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := trace_sound units sparse base
  sound wire challenge table :=
    (traceTableFold_satisfies ((claim wire).statement key openKey) challenge μ fits small batch
      rowsFit table).mono fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged wire challenge table :=
    foldCheck_staged (traceReduction ((claim wire).statement key openKey) challenge μ fits
      batch rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) table varsFit
        batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [committedTrace, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

/-- A last trace fold satisfies the conditions of `Tier.Valid`, with the hypotheses of
`committedTrace_valid`. -/
theorem clearTrace_valid [Nonempty Coord] {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch} {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (clearTrace key openKey claim μ fits batch rowsFit block decode read weight start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := trace_sound units sparse base
  sound wire challenge candidate :=
    (traceFold_satisfies ((claim wire).statement key openKey) challenge μ fits small batch
      rowsFit candidate).mono fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged wire challenge candidate :=
    foldCheck_staged (traceReduction ((claim wire).statement key openKey) challenge μ fits
      batch rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) candidate.flat
        varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _ _
  draws := by simp [clearTrace, foldCosts]
  aligned := (List.length_map ..).trans (foldLosses_length ..)

omit [Fintype Coord] [DecidableEq Coord] [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] in
/-- The verifier of the checks of a committed trace fold does not read the table. -/
theorem committedTrace_uniform {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch} (wire : Wire)
    (challenge : Fin S.blocks → Rq S.innerDegree) (table other : Table) :
    bare ((committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).check wire challenge table) =
      bare ((committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).check wire challenge other) :=
  foldCheck_bare _ _ _ _ table other

variable [Nonempty Coord] {total : ℕ} {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {Inner BreakT CandT KeyT : Type}

/-- Akita Proposition 10.44 for one more level: a committed trace fold over a suffix of the
later stages is a suffix. A first certificate has a weak opening `Weak.TraceOpens` of the claim
of the level, with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`. Its stage errors add `r·p + ε_fold`: `r` is
the number of blocks, `p` is the largest probability of one fold challenge, and `ε_fold` is
`foldError` of the quotient-free reduction, which is the same for each `statement` and each
`challenge`. Its retry loops add `r + 1`, and its tree size is multiplied by `2·(r + 1)`. `source`
maps the witness of a first certificate of the later stages to the table of the level, and
`decodes`, `passed` and `clashes` are the hypotheses of `Tier.fed_before`. A witness of `R_bind` is
a solution of Module-SIS for `B`, `D` or a compression map of the level
(`traceExtract_opens_or_collision`), or a witness of `R_bind` of the later stages. -/
theorem committedTrace_suffix {Wit' : Type}
    {Opens' : Prefix Wire (Option Coord → Slot) total → Wit' → Prop}
    {tail : Structure (Address Wire total) (Prefix Wire (Option Coord → Slot) total) (Option Coord) Slot (ℕ → Wire) Inner
      BreakT CandT KeyT
      Wit'}
    {error : ℝ} {loops tree : ℕ} {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch} {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (later : Tier.Suffix accept
      (committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).finish Opens' tail error loops tree)
    (inside : (committedTrace key openKey claim μ fits batch rowsFit block decode read weight
      start).finish ≤ total)
    (source : Wit' → Table) (clash : Wit' → Wit' → BreakT)
    (decodes : ∀ point, accept point →
      (committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).Decodes point)
    (passed : ∀ point witness, accept point → Opens' point witness →
      (committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).Passed (source witness) point)
    (clashes : ∀ point other witness witness', accept point → accept other →
      Opens' point witness → Opens' other witness' →
      point.take (start + 1) = other.take (start + 1) → source witness ≠ source witness' →
      tail.breaks (clash witness witness')) :
    Tier.Suffix accept start
      (committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).Opens
      ((committedTrace key openKey claim μ fits batch rowsFit block decode read weight
        start).over accept tail (fun inner => source (tail.witness inner))
        (fun inner other => clash (tail.witness inner) (tail.witness other)))
      ((committedTrace key openKey claim μ fits batch rowsFit block decode read weight
          start).levelError +
        error)
      (S.blocks + 1 + loops) (2 * (S.blocks + 1) * tree) := by
  exact Tier.suffix_before (Fintype.card_fin _)
    (committedTrace_valid injective units sparse base small varsFit batchFit) later inside source
    clash decodes passed clashes

/-- The suffix of a last trace fold, which is the extraction theorem of one trace fold with
`Tier.Suffix.extraction_le`: `r·p + ε_fold` for the stage errors, with `foldError` of the
quotient-free reduction, `r + 1` retry loops and the tree size `2·(r + 1)`. `passed`: the verifier
accepts only when each fold challenge decodes and the final claim of the checks is true for the
flat witness of the digit vectors `witnessOf` of the proof. -/
theorem clearTrace_suffix {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch} {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (witnessOf : Wire → Witness O fold)
    (inside : (clearTrace key openKey claim μ fits batch rowsFit block decode read weight
      start).finish ≤ total)
    (passed : ∀ point, accept point →
      (clearTrace key openKey claim μ fits batch rowsFit block decode read weight start).Passed
        (witnessOf (point.wire (start + 1))) point) :
    Tier.Suffix accept start
      (clearTrace key openKey claim μ fits batch rowsFit block decode read weight start).Opens
      ((clearTrace key openKey claim μ fits batch rowsFit block decode read weight start).over
        accept (finish (transcript total accept))
        (fun point => witnessOf (point.wire (start + 1))) (fun _ _ => ()))
      ((clearTrace key openKey claim μ fits batch rowsFit block decode read weight
          start).levelError)
      (S.blocks + 1) (2 * (S.blocks + 1)) := by
  exact Tier.suffix_last (Fintype.card_fin _) (clearTrace_valid injective units sparse base small
    varsFit batchFit)
    witnessOf inside passed

end Levels

end NightstreamFPrime.Spec.Compression.Akita.Extract
