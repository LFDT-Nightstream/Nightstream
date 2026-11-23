import NightstreamFPrime.Spec.Compression.Akita.FlatFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Staged

/-! The bounds of the draws of the checks `foldCheck` of one packing fold of Akita
(ePrint 2026/1983, revision of 2026-11-23, the terms of (189) after the fold challenge).

- `treeCosts` and `rangeCosts` have the bounds of the draws of the range tree of Akita
  Section 6.1 in their order, with `rangeTree_stagedTo` and `rangeCheck_stagedTo`.
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

/-- The bounds of the draws of the range tree before its last stage, in the order of the draws:
for each product level of degree `a`, the `μ'` rounds of degree `a + 1` and the batching
challenge of the claims of the children (Akita (187)). -/
noncomputable def treeCosts (E : Type) [Fintype E] (μ : ℕ) : List ℕ → ℕ → List ℝ
  | [], _ => []
  | a :: rest, nodes => List.replicate μ (((a + 1 : ℕ) : ℝ) / Fintype.card E) ++
      (((nodes * a - 1 : ℕ) : ℝ) / Fintype.card E) :: treeCosts E μ rest (nodes * a)

/-- The bounds of the draws of `rangeCheck`: `treeCosts`, and then the `μ'` rounds of the last
stage with the binary-support point (Akita (187) and Lemma 10.37 (ii)). -/
noncomputable def rangeCosts (E : Type) [Fintype E] (μ b : ℕ) : List ℝ :=
  treeCosts E μ (rangeArities b) 1 ++
    List.replicate μ (((rangeLeaf b + 1 + 1 : ℕ) : ℝ) / Fintype.card E)

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem treeCosts_sum (μ : ℕ) : ∀ (arities : List ℕ) (nodes : ℕ),
    (treeCosts E μ arities nodes).sum =
      ((treeNumerator μ arities nodes : ℕ) : ℝ) / Fintype.card E := by
  intro arities
  induction arities with
  | nil => simp [treeCosts, treeNumerator]
  | cons a rest step =>
      intro nodes
      simp only [treeCosts, treeNumerator, List.sum_append, List.sum_cons, List.sum_replicate,
        nsmul_eq_mul, step]
      push_cast
      ring

omit [Field E] [DecidableEq E] [Algebra Zq E] in
/-- The sum of the bounds of the draws of `rangeCheck` is `rangeNumerator/|E|`. -/
theorem rangeCosts_sum (μ b : ℕ) :
    (rangeCosts E μ b).sum = ((rangeNumerator μ b : ℕ) : ℝ) / Fintype.card E := by
  simp only [rangeCosts, rangeNumerator, List.sum_append, treeCosts_sum, List.sum_replicate,
    nsmul_eq_mul]
  push_cast
  ring

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem treeCosts_nonnegative (μ : ℕ) : ∀ (arities : List ℕ) (nodes : ℕ),
    ∀ cost ∈ treeCosts E μ arities nodes, 0 ≤ cost := by
  intro arities
  induction arities with
  | nil => simp [treeCosts]
  | cons a rest step =>
      intro nodes cost member
      simp only [treeCosts, List.mem_append, List.mem_cons, List.mem_replicate] at member
      rcases member with ⟨_, rfl⟩ | rfl | later
      · positivity
      · positivity
      · exact step _ cost later

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem rangeCosts_nonnegative (μ b : ℕ) : ∀ cost ∈ rangeCosts E μ b, 0 ≤ cost := by
  intro cost member
  rcases List.mem_append.mp member with tree | last
  · exact treeCosts_nonnegative μ _ _ cost tree
  · rw [List.eq_of_mem_replicate last]
    positivity

/-- The bound of the draw of `drawBin`: `1/|E|` at a level with compression digits. -/
noncomputable def binCosts (E : Type) [Fintype E] : Bool → List ℝ
  | true => [((1 : ℕ) : ℝ) / Fintype.card E]
  | false => []

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem binCosts_sum (bin : Bool) :
    (binCosts E bin).sum = ((binDraws bin : ℕ) : ℝ) / Fintype.card E := by
  cases bin <;> simp [binCosts, binDraws]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem binCosts_nonnegative (bin : Bool) : ∀ cost ∈ binCosts E bin, 0 ≤ cost := by
  cases bin <;> simp only [binCosts, List.mem_singleton, List.not_mem_nil, false_imp_iff,
    implies_true, forall_eq]
  positivity

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem binCosts_length (bin : Bool) : (binCosts E bin).length = binDraws bin := by
  cases bin <;> rfl

/-- The losses of the sites of the Stage 1 draws of the range tree before its last stage in the
Akita code (`transcript_grinding/plan.rs`, `append_nonterminal`), in the order of `treeCosts`: for
each product level of degree `a`, the `μ'` rounds of the loss `a + 1` (`SumcheckRound`) and the
powers batch of the claims of the children (`Stage1InterstageBatch`). -/
def treeLosses (μ : ℕ) : List ℕ → ℕ → List ℕ
  | [], _ => []
  | a :: rest, nodes => List.replicate μ (a + 1) ++ (nodes * a - 1) :: treeLosses μ rest (nodes * a)

/-- The degree of the leaf stage of the range tree in the Akita code
(`DigitRangePlan::leaf_degree`): 2 for the base 4, and 4 for each other base. -/
def leafDegree (b : ℕ) : ℕ := if b = 4 then 2 else 4

/-- The losses of the sites of the draws of `rangeCheck` in the order of `rangeCosts`: the sites of
`treeLosses`, and the `μ'` rounds of the leaf with the loss `leafDegree + 1`. -/
def rangeLosses (μ b : ℕ) : List ℕ :=
  treeLosses μ (rangeArities b) 1 ++ List.replicate μ (leafDegree b + 1)

/-- The loss of the site `CompressionBinary` (1) at a level with compression digits. -/
def binLosses : Bool → List ℕ
  | true => [1]
  | false => []

/-- The losses of the sites of the draws of `foldCheck` in the Akita code, in the order of
`foldCosts`: `RingSwitchAlpha` (`alpha`), `Tau0Point` (`μ'`), `Tau1Point` (`s_rel`), the Stage 1
sites, `CompressionBinary`, `Stage2Batch` (1) and the Stage 2 rounds (3). -/
def foldLosses (alpha vars batch base : ℕ) (bin : Bool) : List ℕ :=
  alpha :: vars :: batch ::
    (rangeLosses vars base ++ (binLosses bin ++ 1 :: List.replicate vars 3))

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem treeLosses_length (μ : ℕ) : ∀ (arities : List ℕ) (nodes : ℕ),
    (treeLosses μ arities nodes).length = (treeCosts E μ arities nodes).length := by
  intro arities
  induction arities with
  | nil => intro nodes; rfl
  | cons a rest step =>
      intro nodes
      simp only [treeLosses, treeCosts, List.length_append, List.length_replicate,
        List.length_cons, step]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem rangeLosses_length (μ b : ℕ) : (rangeLosses μ b).length = (rangeCosts E μ b).length := by
  simp only [rangeLosses, rangeCosts, List.length_append, List.length_replicate,
    treeLosses_length (E := E)]

theorem binLosses_length (bin : Bool) : (binLosses bin).length = binDraws bin := by
  cases bin <;> rfl

/-- The bounds of the draws of `foldCheck`, in the order of the draws: `α`, `τ_0`, `τ_1`, the
draws of the range tree, `ζ_bin` at a level with compression digits, `γ` and the rounds of the
fused sum-check (Akita (189)). -/
noncomputable def foldCosts (E : Type) [Fintype E] (badBound vars batch base : ℕ) (bin : Bool) :
    List ℝ :=
  ((badBound : ℝ) / Fintype.card E) :: ((vars : ℝ) / Fintype.card E) ::
    ((batch : ℝ) / Fintype.card E) ::
      (rangeCosts E vars base ++ (binCosts E bin ++
        (((1 : ℕ) : ℝ) / Fintype.card E) ::
          List.replicate vars (((3 : ℕ) : ℝ) / Fintype.card E)))

omit [DecidableEq E] in
/-- The sum of the bounds of the draws is `foldError`. -/
theorem foldCosts_sum (R : Reduction E) (base : ℕ) (bin : Bool) :
    (foldCosts E R.badBound R.vars R.batch base bin).sum = foldError R base bin := by
  simp only [foldCosts, foldError, List.sum_cons, List.sum_append, List.sum_replicate,
    nsmul_eq_mul, rangeCosts_sum, binCosts_sum]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem foldLosses_length (alpha badBound vars batch base : ℕ) (bin : Bool) :
    (foldLosses alpha vars batch base bin).length =
      (foldCosts E badBound vars batch base bin).length := by
  simp only [foldLosses, foldCosts, List.length_cons, List.length_append, List.length_replicate,
    rangeLosses_length (E := E), binLosses_length, binCosts_length]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem foldCosts_nonnegative (badBound vars batch base : ℕ) (bin : Bool) :
    ∀ cost ∈ foldCosts E badBound vars batch base bin, 0 ≤ cost := by
  intro cost member
  simp only [foldCosts, List.mem_cons, List.mem_append, List.mem_replicate] at member
  rcases member with rfl | rfl | rfl | range | binned | rfl | ⟨_, rfl⟩
  · positivity
  · positivity
  · positivity
  · exact rangeCosts_nonnegative vars base cost range
  · exact binCosts_nonnegative bin cost binned
  · positivity
  · positivity

variable [Nonempty Coord]

omit [DecidableEq E] [Algebra Zq E] in
/-- The draw of `drawBin` has its bound of `binCosts`. -/
theorem drawBin_stagedTo {β : Type} (bin : Bool) {bad : Finset E} {next : E → Game β}
    {costs : List ℝ} {tail : β → List ℝ → Prop} (card : bad.card ≤ 1)
    (nexts : ∀ ζ, StagedTo (Coord → E) (next ζ) costs tail) :
    StagedTo (Coord → E) (drawBin bin bad next) (binCosts E bin ++ costs) tail := by
  cases bin
  · exact nexts 0
  · exact stagedTo_draw exists_fair_slot (density_le _ 1 card) nexts

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

/-- Each draw of the range tree before its last stage has its bound of `treeCosts`, when each
draw of the last stage has its bound of `rest`. -/
theorem rangeTree_stagedTo {β : Type} (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β)) (rest : List ℝ)
    (tail : Option β → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (lastStaged : ∀ nodes weights point claim,
      StagedTo (Coord → E) (last nodes weights point claim) rest tail) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      StagedTo (Coord → E) (rangeTree leaf μ w last arities nodes weights point claim)
        (treeCosts E μ arities nodes ++ rest) tail := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  intro arities
  induction arities with
  | nil =>
      intro nodes weights point claim
      exact lastStaged nodes weights point claim
  | cons a others step =>
      intro nodes weights point claim
      rw [treeCosts, List.append_assoc, List.cons_append]
      refine (stagedTo_bind _ _ _ _).mpr (sumcheck_staged (Reply := Coord → E) field (a + 1) μ _
        (stage_roundTruth a (others.prod * leaf) nodes μ weights point w).degree _ _ ?_ ?_ μ []
        claim (by simp))
      · exact fun _ => rejected _
      · rintro ⟨r, value⟩ child
        dsimp only
        split
        · exact stagedTo_draw field (density_le _ _ (zeros_card_le_of_natDegree_le
            ((natDegree_sub_le _ _).trans (max_le (rowPoly_natDegree_le _ _)
              (rowPoly_natDegree_le _ _))))) fun γ => step _ _ _ _
        · exact rejected _

/-- Each draw of `rangeCheck` has its bound of `rangeCosts`, and an accepted end has the entries
`rest` after these bounds. -/
theorem rangeCheck_stagedTo (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (τ0 : List E) (rest : List ℝ) (tail : Option (List E × E) → List ℝ → Prop)
    (rejected : ∀ costs, tail none costs) (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) (rangeCheck b μ support w τ0) (rangeCosts E μ b ++ rest) tail := by
  rw [rangeCosts, List.append_assoc]
  refine rangeTree_stagedTo (rangeLeaf b) μ w _ _ tail rejected (fun nodes weights point claim =>
    (stagedTo_bind _ _ _ _).mpr (sumcheckWith_staged (Reply := Coord → E) exists_fair_slot
      (rangeLeaf b + 1) μ 1 _ (leaf_roundTruth (rangeLeaf b) nodes μ weights point w).degree _
      (fun _ => bindBad_card_le _ _) rest _ ?_ ?_ μ [] claim (by simp))) (rangeArities b) 1
    (fun _ => 1) τ0 0
  · exact fun _ => rejected _
  · rintro ⟨r, value⟩ sClaim
    dsimp only
    split
    · exact ended _
    · exact rejected _

/-- Each draw of `foldCheck` has its bound of `foldCosts`, and an accepted end has the entries
`rest` after these bounds, when an answer of the oracle has `μ'` slots or more and `s_rel` slots
or more. -/
theorem foldCheck_stagedTo (R : Reduction E) (base : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) (varsFit : R.vars ≤ Fintype.card Coord)
    (batchFit : R.batch ≤ Fintype.card Coord) (rest : List ℝ)
    (tail : Option (List E × E) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) (foldCheck R base support bin w)
      (foldCosts E R.badBound R.vars R.batch base bin ++ rest) tail := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  unfold foldCheck foldCosts
  simp only [List.cons_append, List.append_assoc]
  refine stagedTo_draw field (density_le (C := E) (R.bad w) _ (R.bad_card w)) fun α => ?_
  refine stagedTo_draw (exists_fair_slots varsFit) (density_rangeBad_le (E := E) base R.vars w)
    fun τ0 => ?_
  refine stagedTo_draw (exists_fair_slots batchFit)
    ((R.rows α).density_batchBad_le R.vars (R.fits α) w) fun τ1 => ?_
  refine (stagedTo_bind _ _ _ _).mpr (rangeCheck_stagedTo base R.vars support w _ _ _ ?_ ?_)
  · exact fun _ => rejected _
  rintro ⟨r, sClaim⟩
  exact drawBin_stagedTo bin (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)) fun ζ =>
      stagedTo_draw field (density_le _ 1
        (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun γ =>
          fusedCheck_stagedTo R.vars support w _ _ r sClaim ζ γ rest tail rejected ended

/-- Each draw of `foldCheck` has its bound of `foldCosts`, when an answer of the oracle has `μ'`
slots or more and `s_rel` slots or more. -/
theorem foldCheck_staged (R : Reduction E) (base : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) (varsFit : R.vars ≤ Fintype.card Coord)
    (batchFit : R.batch ≤ Fintype.card Coord) :
    Staged (Coord → E) (foldCheck R base support bin w)
      (foldCosts E R.badBound R.vars R.batch base bin) := by
  have staged := foldCheck_stagedTo R base support bin w varsFit batchFit [] (fun _ _ => True)
    (fun _ => trivial) (fun _ => trivial)
  rwa [List.append_nil] at staged

end Costs

end NightstreamFPrime.Spec.Compression.Akita.Extract
