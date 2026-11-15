import NightstreamFPrime.Spec.Compression.Verifier.Game

/-! Soundness of the verifier game (`game_avoids`): on each path that avoids the bad sets, an
accepted proof gives `Relation` for the committed `W`. The steps, from the end of the transcript:
the column point fixes the values of `V` (`columnBad`), the key sum-check fixes the key part
(`key_sum`, `weightAt_spec`, `folded_at`), the first sum-check fixes its sum (`mle_weightTable` and
the closed forms), the challenges fix every error to zero (`sum_sub_claimed`, `challengeBad`), and
the lookup challenges fix each cell of a claim with a table into its table (`lookup_sound`). -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game Polynomial

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

local notation "ι" => algebraMap F K

variable {P : Params F K}

/-- The counts of equal cells of a slice with a table are nonzero in `F`: `2^vars < char F`. -/
def CountsNonzero (P : Params F K) : Prop :=
  ∀ c, (P.shape c).lookup = true → ∀ n : ℕ, 0 < n → n ≤ 2 ^ (P.shape c).witness.vars → (n : F) ≠ 0

omit [Fintype K] [DecidableEq K] in
theorem relation_of_good (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (counts : CountsNonzero P)
    (τ_ok : ∀ c, (P.shape c).lookup = true → ∀ k, ∀ v < 2 * (P.shape c).bound + 1,
      τ k ≠ tableValue (P.shape c).bound v)
    (good : τ ∉ tauBad P W) (holds : ¬ Fails P st W V τ) : Relation P st W := by
  simp only [Fails, not_or, not_exists, not_and] at holds
  obtain ⟨rings, lookups⟩ := holds
  intro c
  have claim := rings c
  simp only [ClaimFails, not_or, not_exists, not_and, not_not] at claim
  refine ⟨fun e e_lt l lane => claim.1 e e_lt l lane, fun y cell => ?_, fun table => ?_⟩
  · exact (algebraMap F K).injective (claim.2 y cell)
  · have outside : τ ∉ lookupBad P.lookups (2 ^ (P.shape c).witness.vars)
        (2 * (P.shape c).bound + 1) (fun y => W ((P.shape c).witness.offset + y))
        (tableValue (P.shape c).bound) (fun v => W ((P.shape c).counts.offset + v)) := by
      intro inside
      apply good
      simp only [tauBad, mem_biUnion, mem_univ, true_and]
      exact ⟨c, by rw [ite_eq_left table]; exact inside⟩
    have errors (k : Fin P.lookups) := lookups c table k
    have inverse (k : Fin P.lookups) (y : ℕ) (cell : y < 2 ^ (P.shape c).witness.vars) :
        (τ k - W ((P.shape c).witness.offset + y)) * inverseOf P V c k y = 1 := by
      have := (errors k).1 y cell
      simp only [inverseError] at this
      exact sub_eq_zero.mp (not_not.mp this)
    refine lookup_sound P.lookups _ _ _ _ _ (counts c table) τ outside (fun k => ⟨fun y cell => ?_,
      τ_ok c table k⟩) (fun k => ?_)
    · intro same
      have := inverse k y cell
      rw [same, sub_self, zero_mul] at this
      exact zero_ne_one this
    · have sums := not_not.mp (errors k).2
      simp only [sumError, sub_eq_zero] at sums
      rw [sum_congr rfl fun y member => (eq_one_div_of_mul_eq_one_right
        (inverse k y (mem_range.mp member))).symm, sums]
      exact sum_congr rfl fun v _ => mul_one_div _ _

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem main_truth_nil (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) :
    combTruth P.witnessVars (mainTables P st W V τ ch) mainComb [] =
      ∑ x ∈ range (2 ^ P.witnessVars), (ι (W x) * (weightTable P st τ ch x -
        checkTable P ch x * aTable P V ch x) + checkTable P ch x * bTable P V τ ch x +
        sumTable P ch x * aTable P V ch x) := by
  rw [combTruth_nil, cubeSum_eq_sum_range]
  refine sum_congr rfl fun x _ => ?_
  simp [mainComb, mainTables]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem main_truth_final (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) (ρ : List K) (length : ρ.length = P.witnessVars) :
    combTruth P.witnessVars (mainTables P st W V τ ch) mainComb ρ =
      mle ρ (fun x => ι (W x)) * (mle ρ (weightTable P st τ ch) -
        mle ρ (checkTable P ch) * mle ρ (aTable P V ch)) +
        mle ρ (checkTable P ch) * mle ρ (bTable P V τ ch) +
        mle ρ (sumTable P ch) * mle ρ (aTable P V ch) := by
  rw [← length, combTruth_final]
  simp [mainComb, mainTables]

/-- On each path of `game` that avoids the bad sets, an accepted proof gives `Relation` for the
committed `W`. -/
theorem game_avoids (vf : VerifierData P) (wf : WellFormed P) (counts : CountsNonzero P)
    (matrix : MatrixSpec vf) (columns : P.lookups ≤ 2 ^ vf.columnVars)
    (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars) :
    (game vf).Avoids fun r => ∀ o, r = some o → Relation P o.1 o.2 := by
  intro st W τ τ_good V ch ch_good
  have main_truth := combRound_truth P.witnessVars 3 (mainTables P st W V τ ch) mainComb
    mainComb_degree
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine ((sumcheck_avoids 3 P.witnessVars _ _ main_truth P.witnessVars [] (claimedSum P st ch)
    (by simp)).and_ends (sumcheck_ends 3 _ P.witnessVars [] _)).mono ?_
  rintro first ⟨sound₁, ends₁⟩
  rcases first with _ | ⟨ρ, value⟩
  · intro o none_eq
    cases none_eq
  have ρ_length : ρ.length = P.witnessVars := by
    obtain ⟨cs, eq, len⟩ := ends₁ ρ value rfl
    rw [eq, List.nil_append, len]
  intro vals col col_good
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine ((sumcheck_avoids 2 (P.laneVars + P.keyBlockVars) _ _ key_truth _ [] vals.2.2
    (by simp)).and_ends (sumcheck_ends 2 _ _ [] _)).mono ?_
  rintro second ⟨sound₂, ends₂⟩
  rcases second with _ | ⟨kp, kfinal⟩
  · intro o none_eq
    cases none_eq
  have kp_length : kp.length = P.laneVars + P.keyBlockVars := by
    obtain ⟨ks, eq, len⟩ := ends₂ kp kfinal rfl
    rw [eq, List.nil_append, len]
  intro kv o equal
  split at equal
  · rename_i accepts
    cases equal
    obtain ⟨shaped, τ_ok, regular, key_check, main_check, kv_open, wv_open, v_open⟩ := accepts
    have cols : ∀ k < 2 ^ vf.columnVars, vals.2.1 k = columnAt P V ρ k := by
      by_contra differ
      push Not at differ
      apply col_good
      simp only [columnBad, differ, ite_true, mem_filter, mem_univ, true_and]
      rw [mle_sub, v_open, mle_V_opening V ρ wf ρ_length, sub_self]
    have cols' : ∀ k : Fin P.lookups, vals.2.1 k = columnAt P V ρ k :=
      fun k => cols k (lt_of_lt_of_le k.isLt columns)
    have point_length (c : Fin P.claims) :
        (uses P ch ρ c).point.length = (P.shape c).witness.vars - P.laneVars := by
      have := (wf.witness_fits c).2.2
      simp [uses, ρ_length, this]
    have block_fits (c : Fin P.claims) : (uses P ch ρ c).blocks ≤ 2 ^ P.keyBlockVars :=
      (shaped.blocks_le c).trans (Nat.pow_le_pow_right two_pos (key_fits c))
    have kw : vals.2.2 = keyTrue P ch ρ := by
      by_contra differ
      have truth_nil : combTruth (P.laneVars + P.keyBlockVars) (keyTables ch ρ) productComb [] =
          keyTrue P ch ρ := by
        rw [product_truth_nil, cubeSum_eq_sum_range, show 2 ^ (P.laneVars + P.keyBlockVars) =
          P.lanes * 2 ^ P.keyBlockVars by rw [pow_add]; rfl]
        exact key_sum P.lanes P.keyBlockVars _ _ _ block_fits (pow_pos two_pos _)
      apply sound₂ kp kfinal rfl (by rw [truth_nil]; exact differ)
      rw [← kp_length, product_truth_final]
      simp only [keyTables, Matrix.cons_val_zero, Matrix.cons_val_one, Params.lanes]
      rw [weightAt_spec P.laneVars P.keyBlockVars _ _ kp kp_length fun c =>
        ⟨by rw [point_length]; exact key_fits c, by rw [point_length]; exact shaped.blocks_le c⟩,
        folded_at (2 ^ P.laneVars) P.keyBlockVars P.rows P.rowVars _ ch.omega kp
          (by rw [kp_length, pow_add]) rows regular]
      rw [key_check, kv_open]
      simp only [Params.lanes]
      ring
    have claimed : claimedSum P st ch = ∑ x ∈ range (2 ^ P.witnessVars),
        (ι (W x) * (weightTable P st τ ch x - checkTable P ch x * aTable P V ch x) +
          checkTable P ch x * bTable P V τ ch x + sumTable P ch x * aTable P V ch x) := by
      by_contra differ
      apply sound₁ ρ value rfl (by rw [main_truth_nil]; exact differ)
      rw [main_truth_final st W V τ ch ρ ρ_length, main_check, wv_open, kw,
        matrix st ch ρ ρ_length, mle_weightTable st τ ch ρ wf shaped ρ_length,
        mle_checkTable ch ρ wf ρ_length, mle_sumTable ch ρ wf ρ_length,
        mle_aTable V ch ρ wf ρ_length, mle_bTable V τ ch ρ wf ρ_length]
      simp only [cols']
    have zero := sum_sub_claimed st W V τ ch wf
    rw [← claimed, sub_self] at zero
    refine relation_of_good st W V τ counts τ_ok τ_good fun fails => ch_good ?_
    simp only [challengeBad, fails, ite_true, mem_filter, mem_univ, true_and]
    exact zero.symm
  · cases equal

end NightstreamFPrime.Spec.Compression.Verifier
