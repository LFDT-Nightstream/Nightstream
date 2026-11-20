import NightstreamFPrime.Spec.Compression.Akita.FlatFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged

/-! The bounds of the draws of the checks `foldCheck` of one packing fold of Akita
(ePrint 2026/1983, revision of 2026-10-15, the terms of (189) after the fold challenge).

- `foldCosts` has the bounds of the draws of `foldCheck` in their order, `foldCosts_sum` states
  that their sum is `foldError`, and `foldCheck_staged` states that each draw has its bound when an
  answer of the oracle has a slot for each coordinate of a draw. `foldCheck_stagedTo` is this
  statement for a game that continues after an accepted end. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Rounds

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {Reply : Type} [Fintype Reply]

/-- Each draw of `sumcheckWith` has the bound `(d + e)/|K|`, and an accepted end has the entries
`rest` after the bounds of its rounds. -/
theorem sumcheckWith_staged (fair : ∃ decode : Reply → K, Fair decode) (d bound e : ℕ)
    (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d)
    (extra : List K → Finset K) (small : ∀ point, (extra point).card ≤ e) (rest : List ℝ)
    (tail : Option (List K × K) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      StagedTo Reply (sumcheckWith d round extra rounds point claim)
        (List.replicate rounds (((d + e : ℕ) : ℝ) / Fintype.card K) ++ rest) tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _
      exact ended _
  | succ rounds induction =>
      intro point claim fits message
      dsimp only
      split
      · rw [List.replicate_succ, List.cons_append]
        exact stagedTo_draw fair (density_le _ _ ((Finset.card_union_le _ _).trans
          (Nat.add_le_add (zeros_card_le_of_natDegree_le ((natDegree_sub_le _ _).trans
            (max_le (coeffPoly_natDegree message) (degree point (by omega))))) (small point))))
          fun c => induction (point ++ [c]) _ (by simp; omega)
      · exact rejected _

/-- Each draw of `Game.sumcheck` has the bound `d/|K|`, and an accepted end has the entries
`rest` after the bounds of its rounds. -/
theorem sumcheck_staged (fair : ∃ decode : Reply → K, Fair decode) (d bound : ℕ)
    (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d)
    (rest : List ℝ) (tail : Option (List K × K) → List ℝ → Prop)
    (rejected : ∀ costs, tail none costs) (ended : ∀ final, tail (some final) rest) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      StagedTo Reply (sumcheck d round rounds point claim)
        (List.replicate rounds ((d : ℝ) / Fintype.card K) ++ rest) tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _
      exact ended _
  | succ rounds induction =>
      intro point claim fits message
      dsimp only
      split
      · rw [List.replicate_succ, List.cons_append]
        exact stagedTo_draw fair (density_le _ _ (zeros_card_le_of_natDegree_le
          ((natDegree_sub_le _ _).trans (max_le (coeffPoly_natDegree message)
            (degree point (by omega)))))) fun c => induction (point ++ [c]) _ (by simp; omega)
      · exact rejected _

end Rounds

section Costs

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The bounds of the draws of `foldCheck`, in the order of the draws: `α`, `τ_0`, `τ_1`, the
rounds of the range sum-check, `ζ_bin`, `γ` and the rounds of the fused sum-check
(Akita (189)). -/
noncomputable def foldCosts (E : Type) [Fintype E] (badBound vars batch base : ℕ) : List ℝ :=
  ((badBound : ℝ) / Fintype.card E) :: ((vars : ℝ) / Fintype.card E) ::
    ((batch : ℝ) / Fintype.card E) ::
      (List.replicate vars (((base / 2 + 1 + 1 : ℕ) : ℝ) / Fintype.card E) ++
        (((1 : ℕ) : ℝ) / Fintype.card E) :: (((1 : ℕ) : ℝ) / Fintype.card E) ::
          List.replicate vars (((3 : ℕ) : ℝ) / Fintype.card E))

omit [DecidableEq E] in
/-- The sum of the bounds of the draws is `foldError`. -/
theorem foldCosts_sum (R : Reduction E) (base : ℕ) :
    (foldCosts E R.badBound R.vars R.batch base).sum = foldError R base := by
  simp only [foldCosts, foldError, List.sum_cons, List.sum_append, List.sum_replicate,
    nsmul_eq_mul]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem foldCosts_nonnegative (badBound vars batch base : ℕ) :
    ∀ cost ∈ foldCosts E badBound vars batch base, 0 ≤ cost := by
  intro cost member
  simp only [foldCosts, List.mem_cons, List.mem_append, List.mem_replicate] at member
  rcases member with rfl | rfl | rfl | ⟨_, rfl⟩ | rfl | rfl | ⟨_, rfl⟩ <;> positivity

variable [Nonempty Coord]

/-- Each draw of `fusedCheck` has the bound `3/|E|`, and an accepted end has the entries `rest`
after the bounds of its rounds. -/
theorem fusedCheck_stagedTo (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) (rest : List ℝ)
    (tail : Option (List E × E) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) (fusedCheck μ support w weight target r sClaim ζ γ)
      (List.replicate μ (((3 : ℕ) : ℝ) / Fintype.card E) ++ rest) tail := by
  refine (stagedTo_bind _ _ _ _).mpr (sumcheck_staged (Reply := Coord → E) exists_fair_slot 3 μ _
    (fused_roundTruth μ w weight (virtWeight support r γ ζ)).degree rest _ ?_ ?_ μ []
    (target + γ * sClaim) (by simp))
  · exact fun _ => rejected _
  · intro final v
    dsimp only
    split
    · exact ended _
    · exact rejected _

/-- Each draw of `foldCheck` has its bound of `foldCosts`, and an accepted end has the entries
`rest` after these bounds, when an answer of the oracle has `μ'` slots or more and `s_rel` slots
or more. -/
theorem foldCheck_stagedTo (R : Reduction E) (base : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (varsFit : R.vars ≤ Fintype.card Coord)
    (batchFit : R.batch ≤ Fintype.card Coord) (rest : List ℝ)
    (tail : Option (List E × E) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) (foldCheck R base support w)
      (foldCosts E R.badBound R.vars R.batch base ++ rest) tail := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  unfold foldCheck foldCosts
  simp only [List.cons_append, List.append_assoc]
  refine stagedTo_draw field (density_le (C := E) (R.bad w) _ (R.bad_card w)) fun α => ?_
  refine stagedTo_draw (exists_fair_slots varsFit) (density_rangeBad_le (E := E) base R.vars w)
    fun τ0 => ?_
  refine stagedTo_draw (exists_fair_slots batchFit)
    ((R.rows α).density_batchBad_le R.vars (R.fits α) w) fun τ1 => ?_
  refine (stagedTo_bind _ _ _ _).mpr (sumcheckWith_staged field (base / 2 + 1) R.vars 1 _
    (range_roundTruth base R.vars (List.ofFn τ0) w).degree _ (fun _ => bindBad_card_le _ _) _ _
    ?_ ?_ R.vars [] 0 (by simp))
  · exact fun _ => rejected _
  rintro ⟨r, value⟩ sClaim
  dsimp only
  split
  · exact stagedTo_draw field (density_le _ 1
      (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun ζ =>
        stagedTo_draw field (density_le _ 1
          (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun γ =>
            fusedCheck_stagedTo R.vars support w _ _ r sClaim ζ γ rest tail rejected ended
  · exact rejected _

/-- Each draw of `foldCheck` has its bound of `foldCosts`, when an answer of the oracle has `μ'`
slots or more and `s_rel` slots or more. -/
theorem foldCheck_staged (R : Reduction E) (base : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (varsFit : R.vars ≤ Fintype.card Coord)
    (batchFit : R.batch ≤ Fintype.card Coord) :
    Staged (Coord → E) (foldCheck R base support w)
      (foldCosts E R.badBound R.vars R.batch base) := by
  have staged := foldCheck_stagedTo R base support w varsFit batchFit [] (fun _ _ => True)
    (fun _ => trivial) (fun _ => trivial)
  rwa [List.append_nil] at staged

end Costs

end NightstreamFPrime.Spec.Compression.Akita.Extract
