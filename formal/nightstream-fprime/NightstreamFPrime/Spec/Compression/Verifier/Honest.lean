import NightstreamFPrime.Spec.Compression.Verifier.Complete

/-! The honest prover of SPEC.md Algorithm 9, steps 6 to 11, against the verifier game of
Algorithm 10 with ideal openings. Its failures are the two events of Algorithm 9, Rejects. -/

namespace NightstreamFPrime.Spec.Compression

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open Polynomial

namespace Game

variable {α : Type}

inductive Plays : Game α → Game α → Prop
  | pure {a b : α} : a = b → Plays (.pure a) (.pure b)
  | draw {C : Type} {fintype : Fintype C} {nonempty : Nonempty C} {bad bad' : Finset C}
      {next next' : C → Game α} :
      (∀ c, Plays (next c) (next' c)) →
        Plays (@Game.draw α C fintype nonempty bad next)
          (@Game.draw α C fintype nonempty bad' next')
  | recv {M : Type} {nonempty : Nonempty M} {next : M → Game α} {rest : Game α} (m : M) :
      Plays (next m) rest → Plays (@Game.recv α M nonempty next) rest

theorem plays_bind {β : Type} {g h : Game α} {f f' : α → Game β} (played : Plays g h)
    (each : ∀ a, Plays (f a) (f' a)) : Plays (bind g f) (bind h f') := by
  induction played with
  | pure equal => subst equal; exact each _
  | draw _ ih => exact .draw ih
  | recv m _ ih => exact .recv m ih

theorem Ends.avoids {g : Game α} {P : α → Prop} (ends : g.Ends P) : g.Avoids P := by
  induction g with
  | pure a => exact ends
  | draw C bad next ih => exact fun c _ => ih c (ends c)
  | recv M next ih => exact fun m => ih m (ends m)

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

noncomputable def honestRound (d : ℕ) (round : List K → K[X]) (point : List K) :
    Fin (d + 1) → K :=
  fun i => (round point).coeff i

noncomputable def honestSumcheck (d : ℕ) (round : List K → K[X]) :
    ℕ → List K → K → Game (Option (List K × K))
  | 0, point, claim => .pure (some (point, claim))
  | rounds + 1, point, claim =>
      if (coeffPoly (honestRound d round point)).eval 0 +
          (coeffPoly (honestRound d round point)).eval 1 = claim then
        .draw K ∅ fun c => honestSumcheck d round rounds (point ++ [c])
          ((coeffPoly (honestRound d round point)).eval c)
      else .pure none

theorem plays_sumcheck (d : ℕ) (round : List K → K[X]) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      Plays (sumcheck d round rounds point claim) (honestSumcheck d round rounds point claim)
  | 0, _, _ => .pure rfl
  | rounds + 1, point, claim => by
      refine .recv (honestRound d round point) ?_
      simp only [honestSumcheck]
      split
      · exact .draw fun c => plays_sumcheck d round rounds _ _
      · exact .pure rfl

theorem honestSumcheck_ends (d bound : ℕ) (truth : List K → K) (round : List K → K[X])
    (true_rounds : RoundTruth d bound truth round) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      claim = truth point → (honestSumcheck d round rounds point claim).Ends fun result =>
        ∃ cs : List K, cs.length = rounds ∧ result = some (point ++ cs, truth (point ++ cs))
  | 0, point, claim, _, true_claim => ⟨[], rfl, by simp [true_claim]⟩
  | rounds + 1, point, claim, fits, true_claim => by
      have inside : point.length < bound := by omega
      have message : coeffPoly (honestRound d round point) = round point :=
        coeffPoly_coeff _ (true_rounds.degree point inside)
      simp only [honestSumcheck, message, true_claim, true_rounds.halves point inside, ite_true]
      intro c
      refine (honestSumcheck_ends d bound truth round true_rounds rounds (point ++ [c]) _
        (by simp; omega) (true_rounds.binds point c inside)).mono ?_
      rintro result ⟨cs, length, rfl⟩
      exact ⟨c :: cs, by simp [length], by simp⟩

theorem honestSumcheck_costLe (d : ℕ) (round : List K → K[X]) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      (honestSumcheck d round rounds point claim).CostLe 0
  | 0, _, _ => le_rfl
  | rounds + 1, point, claim => by
      simp only [honestSumcheck]
      split
      · exact ⟨0, by simp, fun c => honestSumcheck_costLe d round rounds _ _⟩
      · exact le_rfl

end Game

namespace Verifier

open Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

local notation "ι" => algebraMap F K

section Events

variable (P : Params F K)

/-- SPEC.md Algorithm 9, step 6. -/
def tableHit : Finset (Fin P.lookups → F) :=
  univ.filter fun τ => ∃ c, (P.shape c).lookup = true ∧
    ∃ k, ∃ v < 2 * (P.shape c).bound + 1, τ k = tableValue (P.shape c).bound v

/-- SPEC.md Algorithm 9, step 10. -/
def omegaBad : Finset (Challenges P) :=
  univ.filter fun ch => ∃ i < P.rowVars, 1 + ch.omega ^ 2 ^ i = 0

noncomputable def tableError : ℝ :=
  ∑ c : Fin P.claims, if (P.shape c).lookup then
    ((P.lookups * (2 * (P.shape c).bound + 1) : ℕ) : ℝ) / Fintype.card F
  else 0

noncomputable def omegaError : ℝ :=
  ((univ.filter fun ω : K => ∃ i < P.rowVars, 1 + ω ^ 2 ^ i = 0).card : ℝ) / Fintype.card K

end Events

section Densities

variable {P : Params F K}

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem density_coordinate {n : ℕ} (k : Fin n) (a : F) :
    density (univ.filter fun τ : Fin n → F => τ k = a) = 1 / Fintype.card F := by
  rw [density_congr (Equiv.funSplitAt k F) _ ({a} ×ˢ univ) fun τ => by simp [eq_comm],
    density_product, density_univ_one, mul_one]
  simp [density]

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem density_tableHit_le : density (tableHit P) ≤ tableError P := by
  have cover : tableHit P ⊆ univ.biUnion fun c => if (P.shape c).lookup then
      univ.biUnion fun k => (range (2 * (P.shape c).bound + 1)).biUnion fun v =>
        univ.filter fun τ : Fin P.lookups → F => τ k = tableValue (P.shape c).bound v
    else ∅ := by
    intro τ member
    simp only [tableHit, mem_filter, mem_univ, true_and] at member
    obtain ⟨c, table, k, v, lt, equal⟩ := member
    refine mem_biUnion.mpr ⟨c, mem_univ _, ?_⟩
    rw [ite_eq_left table]
    simp only [mem_biUnion, mem_univ, true_and, mem_range, mem_filter]
    exact ⟨k, v, lt, equal⟩
  refine (density_mono cover).trans ((density_biUnion_le _ _).trans (sum_le_sum fun c _ => ?_))
  split
  · refine (density_biUnion_le _ _).trans ?_
    calc ∑ k : Fin P.lookups, density ((range (2 * (P.shape c).bound + 1)).biUnion fun v =>
            univ.filter fun τ : Fin P.lookups → F => τ k = tableValue (P.shape c).bound v)
        ≤ ∑ _k : Fin P.lookups, ((2 * (P.shape c).bound + 1 : ℕ) : ℝ) / Fintype.card F :=
          sum_le_sum fun k _ => (density_biUnion_le _ _).trans (by
            rw [sum_congr rfl fun v _ => density_coordinate k _, sum_const, card_range,
              nsmul_eq_mul, mul_one_div])
      _ = ((P.lookups * (2 * (P.shape c).bound + 1) : ℕ) : ℝ) / Fintype.card F := by
          rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_mul, mul_div_assoc]
  · rw [density_empty]

omit [Field F] [Fintype F] [DecidableEq F] [Algebra F K] in
theorem density_omegaBad : density (omegaBad P) = omegaError P := by
  have product : omegaBad P = ((univ : Finset K) ×ˢ
      ((univ.filter fun ω : K => ∃ i < P.rowVars, 1 + ω ^ 2 ^ i = 0) ×ˢ
        (univ : Finset (K × K)))) ×ˢ
      (univ : Finset (LookupDraw K P.lookups P.claims P.spanVars)) := by
    ext ch
    rw [omegaBad, mem_filter, mem_product, mem_product, mem_product, mem_filter]
    simp [Challenges.omega]
  rw [product, density_product, density_product, density_product, density_univ_one,
    density_univ_one, density_univ_one, one_mul, mul_one, mul_one]
  unfold omegaError density
  congr

omit [Field F] [Fintype F] [DecidableEq F] [Algebra F K] in
theorem omegaError_le (rowVars_le : P.rowVars ≤ 5) (two : (2 : K) ≠ 0)
    (v2 : ¬ 16 ∣ Fintype.card K - 1) : omegaError P ≤ 7 / Fintype.card K := by
  unfold omegaError
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  refine Nat.cast_le.mpr ((card_le_card fun ω member => ?_).trans
    (rowPoint_degenerate_card_le two v2))
  simp only [mem_filter, mem_univ, true_and] at member ⊢
  obtain ⟨i, lt, zero⟩ := member
  exact ⟨i, by omega, zero⟩

end Densities

section Honest

variable {P : Params F K} (vf : VerifierData P)

/-- The inverse witness of SPEC.md Algorithm 9, step 6. -/
noncomputable def honestInverses (P : Params F K) (W : ℕ → F) (τ : Fin P.lookups → F)
    (x : ℕ) : F :=
  if ∃ c, (P.shape c).lookup = true ∧ (P.shape c).witness.offset ≤ x ∧
      x < (P.shape c).witness.offset + 2 ^ (P.shape c).witness.vars then
    1 / denominator τ (W x)
  else 0

omit [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem honestInverses_spec (W : ℕ → F) (τ : Fin P.lookups → F) :
    HonestInverses P W (honestInverses P W τ) τ := by
  intro c table y cell
  exact ite_eq_left ⟨c, table, Nat.le_add_right _ _, by omega⟩

/-- `Â(ζ, s)` of SPEC.md Algorithm 9, step 10. -/
noncomputable def honestKey (ch : Challenges P) (kp : List K) : K :=
  mle (kp ++ rowPoint P.rowVars ch.omega)
    (keyTable P.lanes P.keyBlockVars P.rows fun i j h => ι (P.key i j h))

/-- `W(ρ)`, the shared inverse value and `K_w` of SPEC.md Algorithm 9, step 9. -/
noncomputable def honestValues (W V : ℕ → F) (ch : Challenges P) (ρ : List K) : Values K :=
  (mle ρ fun x => ι (W x), inverseAt P V ρ, keyTrue P ch ρ)

noncomputable def honestAfterKey (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) (ρ : List K) (value : K) (vals : Values K) :
    Option (List K × K) → Game (Option (Statement P × (ℕ → F)))
  | none => .pure none
  | some (kp, kfinal) => by
      classical
      exact .pure (if Accepts vf st W V τ ch ρ value vals kp kfinal (honestKey ch kp) then
        some (st, W) else none)

noncomputable def honestAfterMain (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) : Option (List K × K) → Game (Option (Statement P × (ℕ → F)))
  | none => .pure none
  | some (ρ, value) =>
      bind (honestSumcheck 2 (combRound (P.laneVars + P.keyBlockVars) (keyTables ch ρ)
          productComb) (P.laneVars + P.keyBlockVars) [] (honestValues W V ch ρ).2.2)
        (honestAfterKey vf st W V τ ch ρ value (honestValues W V ch ρ))

/-- SPEC.md Algorithm 9, steps 6 to 11, with Rejects as the bad sets. -/
noncomputable def honestGame (st : Statement P) (W : ℕ → F) :
    Game (Option (Statement P × (ℕ → F))) :=
  .draw (Fin P.lookups → F) (tableHit P) fun τ =>
    .draw (Challenges P) (omegaBad P) fun ch =>
      bind (honestSumcheck (P.lookups + 2) (combRound P.witnessVars
          (mainTables P st W (honestInverses P W τ) τ ch) (mainComb P τ ch)) P.witnessVars []
          (claimedSum P st ch))
        (honestAfterMain vf st W (honestInverses P W τ) τ ch)

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem plays_afterKey (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) (ρ : List K) (value : K) (vals : Values K) :
    ∀ result, Plays (afterKey vf st W V τ ch ρ value vals result)
      (honestAfterKey vf st W V τ ch ρ value vals result)
  | none => .pure rfl
  | some (kp, kfinal) => .recv (honestKey ch kp) (.pure (by split_ifs <;> rfl))

omit [Fintype F] [DecidableEq F] in
theorem plays_afterMain (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
    (ch : Challenges P) :
    ∀ result, Plays (afterMain vf st W V τ ch result) (honestAfterMain vf st W V τ ch result)
  | none => .pure rfl
  | some (ρ, value) => .recv (honestValues W V ch ρ)
      (plays_bind (plays_sumcheck _ _ _ _ _) (plays_afterKey vf st W V τ ch ρ value _))

theorem plays_honest (st : Statement P) (W : ℕ → F) : Plays (game vf) (honestGame vf st W) :=
  .recv st (.recv W (.draw fun τ => .recv (honestInverses P W τ) (.draw fun ch =>
    plays_bind (plays_sumcheck _ _ _ _ _) (plays_afterMain vf st W _ τ ch))))

theorem honest_avoids (st : Statement P) (W : ℕ → F) (wf : WellFormed P) (shaped : Shaped P st)
    (relation : Relation P st W) (honest : HonestCounts P W) (distinct : TablesDistinct P)
    (matrix : MatrixSpec vf) (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars) :
    (honestGame vf st W).Avoids fun r => r = some (st, W) := by
  intro τ τ_good ch ch_good
  have τ_ok : ∀ c, (P.shape c).lookup = true → ∀ k, ∀ v < 2 * (P.shape c).bound + 1,
      τ k ≠ tableValue (P.shape c).bound v := fun c table k v lt equal =>
    τ_good (by
      simp only [tableHit, mem_filter, mem_univ, true_and]
      exact ⟨c, table, k, v, lt, equal⟩)
  have regular : ∀ i < P.rowVars, 1 + ch.omega ^ 2 ^ i ≠ 0 := fun i lt zero =>
    ch_good (by
      simp only [omegaBad, mem_filter, mem_univ, true_and]
      exact ⟨i, lt, zero⟩)
  have inverses := honestInverses_spec W τ
  have main_truth := combRound_truth P.witnessVars (P.lookups + 2)
    (mainTables P st W (honestInverses P W τ) τ ch) (mainComb P τ ch) (mainComb_degree P τ ch)
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine (Ends.avoids (honestSumcheck_ends (P.lookups + 2) P.witnessVars _ _ main_truth P.witnessVars []
    (claimedSum P st ch) (by simp) (by
      rw [main_truth_nil]
      exact (honest_sum st W _ τ ch wf relation honest inverses distinct τ_ok).symm))).mono ?_
  rintro first ⟨ρ, ρ_length, rfl⟩
  rw [List.nil_append]
  have point_length (c : Fin P.claims) :
      (uses P ch ρ c).point.length = (P.shape c).witness.vars - P.laneVars := by
    have := (wf.witness_fits c).2.2
    simp [uses, ρ_length, this]
  have block_fits (c : Fin P.claims) : (uses P ch ρ c).blocks ≤ 2 ^ P.keyBlockVars :=
    (shaped.blocks_le c).trans (Nat.pow_le_pow_right two_pos (key_fits c))
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine (Ends.avoids (honestSumcheck_ends 2 (P.laneVars + P.keyBlockVars) _ _ key_truth _ []
    _ (by simp) (by
      rw [product_truth_nil, cubeSum_eq_sum_range, show 2 ^ (P.laneVars + P.keyBlockVars) =
        P.lanes * 2 ^ P.keyBlockVars by rw [pow_add]; rfl]
      exact (key_sum P.lanes P.keyBlockVars _ _ _ block_fits (pow_pos two_pos _)).symm))).mono ?_
  rintro second ⟨kp, kp_length, rfl⟩
  rw [List.nil_append]
  refine ite_eq_left ⟨shaped, τ_ok, regular, ?_, ?_, rfl, rfl, ?_⟩
  · rw [← kp_length, product_truth_final]
    exact key_check_honest ch ρ kp wf st shaped ρ_length kp_length rows regular key_fits
  · exact main_check_honest vf st W _ τ ch ρ wf shaped ρ_length matrix
  · exact (mle_V_opening _ ρ _).symm

theorem honest_costLe (st : Statement P) (W : ℕ → F) :
    (honestGame vf st W).CostLe (tableError P + omegaError P) := by
  refine ⟨omegaError P, by
    have := density_tableHit_le (P := P)
    unfold density at this
    linarith, fun τ => ⟨0, by
      have := density_omegaBad (P := P)
      unfold density at this
      linarith, fun ch => ?_⟩⟩
  refine (costLe_bind (δ := 0) (honestSumcheck_costLe _ _ _ _ _) fun first => ?_).mono (by simp)
  rcases first with _ | ⟨ρ, value⟩
  · exact le_rfl
  refine (costLe_bind (δ := 0) (honestSumcheck_costLe _ _ _ _ _) fun second => ?_).mono
    (by simp)
  rcases second with _ | ⟨kp, kfinal⟩
  · exact le_rfl
  · exact le_rfl

theorem honest_complete (st : Statement P) (W : ℕ → F) (wf : WellFormed P)
    (shaped : Shaped P st) (relation : Relation P st W) (honest : HonestCounts P W)
    (distinct : TablesDistinct P) (matrix : MatrixSpec vf) (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (s : Strategy (Option (Statement P × (ℕ → F)))) :
    Plays (game vf) (honestGame vf st W) ∧
      prob s (honestGame vf st W) (fun r => r ≠ some (st, W)) ≤ tableError P + omegaError P :=
  ⟨plays_honest vf st W, prob_not_le s _ _ _
    (honest_avoids vf st W wf shaped relation honest distinct matrix rows key_fits)
    (honest_costLe vf st W)⟩

end Honest

end Verifier

end NightstreamFPrime.Spec.Compression
