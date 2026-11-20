import NightstreamFPrime.Spec.Compression.Akita.Extract.Tower

/-! The checks of one fold of Akita on the quotient-lift route with the row `w_cells = 0` in
`R_{q,1}` (ePrint 2026/1983, revision of 2026-10-15, Section 5.5 with (76), Section 7.3 with (111)
and (112), Lemma 7.1 and Lemma 10.42 (i)).

`liftFold` has the row `w_cells = 0` in `R_{q,d_car}`, and its quotient has `δ_quot·d_car` cells.
The level 3 of `key30.aks` has 1024 cells after the quotient digits of its ring rows, and this row
with the zero cell needs 1025 cells (`Key30.level₃_tight`). The row in `R_{q,1}` reads the same
cell and its quotient has `δ_quot` cells.

- `slimRelation` has the rows and the scalar row of `flatRelation`, and the row `w_cells = 0` in
  `R_{q,1}`. `slimRelation_locked` states that a table with it has `lockedRelation`.
- `OpeningShape.slimRoom` is the first cell after the quotient digits:
  `cells + 1 + δ_quot·(Σ_r d_r + 1)`.
- `slimFold` is the level `liftFold` for `slimRelation`, and `LevelSized.slim` gives `Tier.Valid`
  for it with the tensor reduction. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Rows

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor) {E : Type} [Field E] [Algebra Zq E]
  (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold < 2 ^ μ)

/-- The ring rows of `flatRows`, and the row `w_cells = 0` in `R_{q,1}`. -/
noncomputable def slimRows : Fin (O.rowCount + 1) → RingRow :=
  Fin.append (flatRows O fold β st challenge) fun _ : Fin 1 => (zeroForm 1 (O.cells fold)).row 0

theorem slimRows_below (r : Fin (O.rowCount + 1)) :
    (slimRows O fold β st challenge r).Below (O.cells fold) := by
  revert r
  exact (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr
    ⟨flatRows_below O fold β st challenge,
      fun _ => Form.Below.row (f := zeroForm 1 (O.cells fold)) (fun _ _ => le_rfl) 0⟩

theorem slimRows_degree_le (r : Fin (O.rowCount + 1)) :
    (slimRows O fold β st challenge r).degree ≤ O.largestDegree := by
  revert r
  exact (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr
    ⟨flatRows_degree_le O fold β st challenge,
      fun _ => (Nat.one_le_iff_ne_zero.mpr (NeZero.ne O.carrierDegree)).trans
        (le_max_of_le_left (le_max_left O.carrierDegree S.innerDegree))⟩

theorem slimRows_degree_sum :
    ∑ r, (slimRows O fold β st challenge r).degree = O.degreeSum + 1 := by
  unfold slimRows
  rw [sum_append RingRow.degree, flatRows_degree_sum]
  rfl

/-- The relation of one fold for a table that the next level commits to: the rows and the scalar
row of `flatRelation`, and the row `w_cells = 0` in `R_{q,1}`. -/
noncomputable def slimRelation : FlatRelation E where
  vars := μ
  ring := O.rowCount + 1
  row := slimRows O fold β st challenge
  inside r c j _ := (slimRows_below O fold β st challenge r c j).trans_lt fits
  weight := scalarWeight O fold β st.point
  value := st.value

variable {O fold β st challenge μ fits}

/-- A table with `slimRelation` has `lockedRelation`: the two rows `w_cells = 0` state that the
cell `cells` is 0. -/
theorem slimRelation_locked {table : Table}
    (holds : (slimRelation O fold β st challenge μ fits).Holds table) :
    (lockedRelation O fold β st challenge μ fits).Holds table := by
  have rows : ∀ r, (Fin.append (flatRows O fold β st challenge)
      (fun _ : Fin 1 => (zeroForm 1 (O.cells fold)).row 0) r).Holds table := holds.1
  obtain ⟨flat, slim⟩ := (forall_append (RingRow.Holds · table)
    (flatRows O fold β st challenge) (fun _ : Fin 1 => (zeroForm 1 (O.cells fold)).row 0)).mp rows
  have zero := zero_of_zeroForm (slim 0)
  refine ⟨(forall_append (RingRow.Holds · table) (flatRows O fold β st challenge)
    (fun _ : Fin 1 => (zeroForm O.carrierDegree (O.cells fold)).row 0)).mpr
    ⟨flat, fun _ => ((zeroForm O.carrierDegree (O.cells fold)).row_holds 0 table).mpr ?_⟩,
    holds.2⟩
  show (∑ _c : Unit, (1 : Rq O.carrierDegree) * ofCoeffs fun _ => table (O.cells fold)) = 0
  rw [zero, ofCoeffs_eq (x := 0) fun j _ => (coeff_zero j).symm]
  simp

variable (O fold) in
/-- The first cell after the quotient digits of one fold with the row `w_cells = 0` in `R_{q,1}`:
the flat witness, the zero cell, and `δ_quot·d_r` cells for each ring row `r` and `δ_quot` cells
for the row `w_cells = 0` (Akita (76), (348)). -/
def OpeningShape.slimRoom (digits : ℕ) : ℕ := O.cells fold + 1 + digits * (O.degreeSum + 1)

variable [Fintype E] (β st challenge μ fits)

omit [Fintype E] in
theorem slimRelation_quotientEnd (digits : ℕ) :
    (slimRelation O fold β st challenge μ fits).quotientEnd digits (O.cells fold + 1) =
      O.slimRoom fold digits := by
  rw [FlatRelation.quotientEnd_eq]
  exact congrArg (fun total => O.cells fold + 1 + digits * total)
    (slimRows_degree_sum O fold β st challenge)

/-- The quotient-lift reduction of `slimRelation` (Akita Section 7.3 and Lemma 7.1), with the
quotient digits of each ring row in the cells from `cells + 1` (Akita (348)), in the base `b⋆`
with `digits` digits, and `2^batch` places for the rows. Its bad set has at most `2·d_max − 1`
points. -/
noncomputable def slimReduction (digits : ℕ) (room : O.slimRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) : Reduction E :=
  ((slimRelation O fold β st challenge μ fits).withQuotients S.certifiedBase digits
    (O.cells fold + 1)
    ((slimRelation_quotientEnd β st challenge μ fits digits).trans_le room)).lift
    (algebraMap Zq E).injective O.largestDegree batch (slimRows_degree_le O fold β st challenge)
    rowsFit

/-- Akita Lemma 10.42 (i) on the quotient-lift route for `slimRelation`: on a path of `foldCheck`
that avoids the bad sets, a final claim that is true for the table gives the exact branch
relation `Witness.Satisfies` of the digit vectors `ofFlat` of the table. -/
theorem slimFold_satisfies [DecidableEq E] (small : S.certifiedBase < fieldModulus) (digits : ℕ)
    (room : O.slimRoom fold digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch)
    (table : Table) :
    (foldCheck (slimReduction β st challenge μ fits digits room batch rowsFit) S.certifiedBase
      (O.binarySupport fold) table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (O.ofFlat fold table).Satisfies β st challenge :=
  (foldCheck_avoids (slimReduction β st challenge μ fits digits room batch rowsFit)
    S.certifiedBase (O.binarySupport fold) table).mono fun _ checked r2 v equal opening =>
      satisfies_of_locked β st challenge μ fits small
        (slimRelation_locked (LiftRelation.holds_flat _ (checked r2 v equal opening).1))
        (checked r2 v equal opening).2.1 (checked r2 v equal opening).2.2

end Rows

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)

attribute [local instance low] Classical.propDecidable

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The level `liftFold` with the row `w_cells = 0` in `R_{q,1}`: the checks are `foldCheck` for
`slimReduction`. -/
noncomputable def slimFold (β : Module.Basis (Fin O.planes) Zq E) (key : Key S)
    (openKey : OpeningKey O) (claim : Wire → Claim O E) (μ : ℕ) (fits : O.cells fold < 2 ^ μ)
    (digits : ℕ) (room : O.slimRoom fold digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq O.carrierDegree)) (read : Reader Wire) (weight start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq O.carrierDegree) (Claim O E)
      (LevelBreak key openKey (2 * (2 * weight) * fold.sourceBound S)) (Witness O fold) (Weak O)
      Wire Coord E (Option (List E × E)) Table where
  method := packing β key openKey weight
  claim := claim
  block := block
  decode := fun _ => decode
  read := read
  digits := O.ofFlat fold
  check wire challenge table :=
    foldCheck (slimReduction β ((claim wire).statement key openKey) challenge μ fits digits
      room batch rowsFit) S.certifiedBase (O.binarySupport fold) table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCosts E (2 * O.largestDegree - 1) μ batch S.certifiedBase
  start := start

variable [Degree2 E] [Nonempty Coord] {β : Module.Basis (Fin O.planes) Zq E} {key : Key S}
  {openKey : OpeningKey O} {claim : Wire → Claim O E} {μ : ℕ} {fits : O.cells fold < 2 ^ μ}
  {digits : ℕ} {room : O.slimRoom fold digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch} {block : Fin S.blocks → Coord}
  {decode : E → Option (Rq O.carrierDegree)} {read : Reader Wire} {weight start ν : ℕ}

/-- The level `slimFold` with the tensor reduction satisfies the conditions of `Tier.Valid`. -/
theorem LevelSized.slim
    (sized : LevelSized block decode weight fold.responseBase S.certifiedBase μ batch) :
    (Tier.tensored (slimFold β key openKey claim μ fits digits room batch rowsFit block decode
      read weight start) ν).Valid := by
  obtain ⟨peak, sparse⟩ := sized.sparse
  refine Tier.tensored_valid
    { injective := sized.injective
      merges := packing_sound sized.units sparse sized.base
      sound := fun wire challenge table =>
        (slimFold_satisfies β ((claim wire).statement key openKey) challenge μ fits sized.small
          digits room batch rowsFit table).mono fun _ implies passed => by
            obtain ⟨r2, v, ended, opened⟩ := passed
            exact implies r2 v ended opened
      staged := fun wire challenge table =>
        foldCheck_staged (slimReduction β ((claim wire).statement key openKey) challenge μ fits
          digits room batch rowsFit) S.certifiedBase (O.binarySupport fold) table sized.varsFit
          sized.batchFit
      nonnegative := foldCosts_nonnegative _ _ _ _
      draws := by simp [slimFold, foldCosts] }
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
    fun _ _ table rest tail rejected ended =>
      foldCheck_stagedTo _ _ _ table sized.varsFit sized.batchFit rest tail rejected ended

end Extract

end NightstreamFPrime.Spec.Compression.Akita
