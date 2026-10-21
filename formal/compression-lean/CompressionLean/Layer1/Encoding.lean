import CompressionLean.Layer1.Late

/-!
Owns the encoding identity of the two-level scatter (`matrix.rs`): with the
true slot sums, the weights the slots put on block `b` (`slotWeights`) are the
Eval_A part of the batched ring weights at `ζ`,
`Σ_{l < D} τ_l·Σ_j (λ^{re}·Re + λ^{im}·Im)(M_jᵀ χ_r)_{D·b + l}` (`slotWeights_eq`).

The proof splits the late leaves into (slot, part), keeps the two blocks a
run can touch, swaps the slot and lane sums, and regroups the runs by slot.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- The Eval_A weights of column `c`, batched by the Eval_A rows' powers of `λ`. -/
noncomputable def evalAColumn (s : Shape) (S : Setup F) (st : Statement F) (lam : K) (c : ℕ) : K :=
  ∑ j ∈ Finset.range s.matrices,
    (lam ^ s.evalARow j 0 * ι (evalAWeight S st s.D j c).1 +
      lam ^ s.evalARow j 1 * ι (evalAWeight S st s.D j c).2)

/-- The run part of a column weight: what a run's slot puts on it. -/
theorem runWeight_block (s : Shape) (S : Setup F) (fits : S.Fits s) (σ b l : ℕ) (lane : l < s.D)
    (inside : runWeight (K := K) s S σ (s.D * b + l) ≠ 0) :
    b = S.slotBlock σ ∨ b = S.slotBlock σ + 1 := by
  unfold runWeight slotStart at inside
  split at inside
  · rename_i covers
    have lane_lt := fits.lane_lt σ
    have length_le := fits.length_le σ
    obtain ⟨low, high⟩ := covers
    have lower : S.slotBlock σ ≤ b := by
      by_contra smaller
      push Not at smaller
      have : s.D * (b + 1) ≤ s.D * S.slotBlock σ := Nat.mul_le_mul_left _ smaller
      rw [mul_add, mul_one] at this
      omega
    have upper : b < S.slotBlock σ + 2 := by
      by_contra larger
      push Not at larger
      have : s.D * (S.slotBlock σ + 2) ≤ s.D * b := Nat.mul_le_mul_left _ larger
      rw [mul_add] at this
      omega
    omega
  · exact absurd rfl inside

open Classical in
/-- The late leaves split into (slot, part). -/
theorem slotWeights_split (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (U : ℕ → ℕ → F × F) (b : ℕ) :
    slotWeights s S lam τ U b =
      ∑ σ ∈ Finset.range (2 ^ s.slotVars),
        ((if blockColumn S σ + ((0 : ℕ) : K) = (b : K) then ubar s lam U σ * laneSplit s S τ 0 σ else 0) +
          (if blockColumn S σ + ((1 : ℕ) : K) = (b : K) then ubar s lam U σ * laneSplit s S τ 1 σ else 0)) := by
  classical
  unfold slotWeights
  rw [Finset.sum_filter, pow_succ, mul_two, Finset.sum_range_add, Finset.sum_add_distrib]
  congr 1
  · apply Finset.sum_congr rfl
    intro σ inside
    have lt := Finset.mem_range.mp inside
    rw [Nat.mod_eq_of_lt lt, Nat.div_eq_of_lt lt]
  · apply Finset.sum_congr rfl
    intro σ inside
    have lt := Finset.mem_range.mp inside
    have high : (2 ^ s.slotVars + σ) / 2 ^ s.slotVars = 1 := by
      rw [Nat.add_div_left _ (by positivity), Nat.div_eq_of_lt lt]
    have low : (2 ^ s.slotVars + σ) % 2 ^ s.slotVars = σ := by
      rw [Nat.add_mod_left, Nat.mod_eq_of_lt lt]
    rw [high, low]

open Classical in
/-- One slot: its two parts are the lane sums of its weights on block `b`. -/
theorem slot_parts (s : Shape) (S : Setup F) (fits : S.Fits s) (τ : ℕ → K) (u : K) (σ b : ℕ)
    (N : ℕ) (cast : ∀ a c : ℕ, a < N → c < N → (a : K) = c → a = c) (b_lt : b < N)
    (block_lt : S.slotBlock σ + 1 < N) (outside : S.slots ≤ σ → u = 0) :
    (if blockColumn S σ + ((0 : ℕ) : K) = (b : K) then u * laneSplit s S τ 0 σ else 0) +
        (if blockColumn S σ + ((1 : ℕ) : K) = (b : K) then u * laneSplit s S τ 1 σ else 0) =
      u * ∑ l ∈ Finset.range s.D, τ l * runWeight s S σ (s.D * b + l) := by
  by_cases real : σ < S.slots
  · have column : (blockColumn S σ : K) = (S.slotBlock σ : K) := by simp [blockColumn, real]
    have iff (t : ℕ) (t_le : t ≤ 1) :
        (blockColumn S σ + ((t : ℕ) : K) = (b : K)) ↔ S.slotBlock σ + t = b := by
      rw [column, ← Nat.cast_add]
      exact ⟨fun h => cast _ _ (by omega) b_lt h, fun h => by rw [h]⟩
    simp only [laneSplit, real, if_true]
    rw [if_congr (iff 0 (by norm_num)) rfl rfl, if_congr (iff 1 le_rfl) rfl rfl]
    by_cases here : S.slotBlock σ + 0 = b
    · rw [if_pos here, if_neg (by omega), add_zero, ← here, add_zero]
    · by_cases next : S.slotBlock σ + 1 = b
      · rw [if_neg here, if_pos next, zero_add, ← next]
      · rw [if_neg here, if_neg next, add_zero, eq_comm, mul_eq_zero]
        right
        apply Finset.sum_eq_zero
        intro l inside
        by_contra nonzero
        have := runWeight_block (K := K) s S fits σ b l (Finset.mem_range.mp inside)
          (right_ne_zero_of_mul nonzero)
        omega
  · have zero := outside (by omega)
    simp [zero]

/-- One run's term in the batched Eval_A weights of matrix `j`. -/
noncomputable def runTerm (s : Shape) (S : Setup F) (st : Statement F) (lam : K) (j k : ℕ) : K :=
  lam ^ s.evalARow j 0 * ι (S.runCoef k) * ι (st.chi (S.runRow k)).1 +
    lam ^ s.evalARow j 1 * ι (S.runCoef k) * ι (st.chi (S.runRow k)).2

open Classical in
theorem ubar_true (s : Shape) (S : Setup F) (st : Statement F) (lam : K) (σ : ℕ) :
    ubar s lam (trueU s S st) σ =
      ∑ j ∈ Finset.range s.matrices, ∑ k ∈ Finset.range (2 ^ s.runVars),
        if S.runMatrix k = j ∧ S.runSlot k = σ then runTerm s S st lam j k else 0 := by
  unfold ubar trueU
  apply Finset.sum_congr rfl
  intro j _
  rw [Prod.fst_sum, Prod.snd_sum, map_sum, map_sum, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro k _
  split
  · simp only [runTerm, Prod.smul_fst, Prod.smul_snd, smul_eq_mul, map_mul]
    ring
  · rfl

open Classical in
theorem evalAColumn_true (s : Shape) (S : Setup F) (st : Statement F) (fits : S.Fits s) (lam : K) (c : ℕ) :
    evalAColumn s S st lam c =
      ∑ j ∈ Finset.range s.matrices, ∑ k ∈ Finset.range (2 ^ s.runVars),
        if S.runMatrix k = j then runTerm s S st lam j k * runWeight s S (S.runSlot k) c else 0 := by
  unfold evalAColumn evalAWeight
  apply Finset.sum_congr rfl
  intro j _
  rw [Prod.fst_sum, Prod.snd_sum, map_sum, map_sum, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib, Finset.sum_filter]
  rw [← Finset.sum_subset (Finset.range_mono fits.runs_le)]
  · apply Finset.sum_congr rfl
    intro k _
    unfold runWeight slotStart
    by_cases matrix : S.runMatrix k = j
    · by_cases covers : s.D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) ≤ c ∧
          c < s.D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) + S.slotLength (S.runSlot k)
      · rw [if_pos ⟨matrix, covers⟩, if_pos matrix, if_pos covers]
        simp only [runTerm, Prod.smul_fst, Prod.smul_snd, smul_eq_mul, map_mul, map_pow]
        ring
      · rw [if_neg (fun h => covers h.2), if_pos matrix, if_neg covers, mul_zero]
    · rw [if_neg (fun h => matrix h.1), if_neg matrix]
  · intro k inside outside
    have zero := (fits.padding k (by simp at inside outside; omega)).1
    split
    · simp [runTerm, zero]
    · rfl

open Classical in
/-- The slots' weights against a column regroup into the batched Eval_A weights. -/
theorem ubar_runWeight (s : Shape) (S : Setup F) (st : Statement F) (fits : S.Fits s) (lam : K) (c : ℕ) :
    ∑ σ ∈ Finset.range (2 ^ s.slotVars), ubar s lam (trueU s S st) σ * runWeight s S σ c =
      evalAColumn s S st lam c := by
  rw [evalAColumn_true s S st fits]
  simp only [ubar_true, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  have slot_lt := fits.slot_lt k
  by_cases matrix : S.runMatrix k = j
  · rw [if_pos matrix]
    have collapse : ∑ σ ∈ Finset.range (2 ^ s.slotVars),
        (if S.runMatrix k = j ∧ S.runSlot k = σ then runTerm s S st lam j k else 0) * runWeight s S σ c =
        ∑ σ ∈ Finset.range (2 ^ s.slotVars),
          if S.runSlot k = σ then runTerm s S st lam j k * runWeight s S σ c else 0 := by
      apply Finset.sum_congr rfl
      intro σ _
      by_cases slot : S.runSlot k = σ
      · rw [if_pos ⟨matrix, slot⟩, if_pos slot]
      · rw [if_neg (fun h => slot h.2), if_neg slot, zero_mul]
    rw [collapse, Finset.sum_ite_eq, if_pos (Finset.mem_range.mpr slot_lt)]
  · rw [if_neg matrix]
    apply Finset.sum_eq_zero
    intro σ _
    rw [if_neg (fun h => matrix h.1), zero_mul]

theorem ubar_true_outside (s : Shape) (S : Setup F) (st : Statement F) (fits : S.Fits s) (lam : K)
    (σ : ℕ) (outside : S.slots ≤ σ) : ubar s lam (trueU s S st) σ = 0 := by
  classical
  rw [ubar_true]
  apply Finset.sum_eq_zero
  intro j _
  apply Finset.sum_eq_zero
  intro k inside
  split
  · rename_i h
    by_cases real : k < S.runs
    · exact absurd h.2 (by have := fits.run_slot k real; omega)
    · simp [runTerm, (fits.padding k (by omega)).1]
  · rfl

/-- The encoding identity: with the true slot sums, the slots' weights on
block `b` are the Eval_A part of the batched ring weights at `ζ`. -/
theorem slotWeights_eq (s : Shape) (S : Setup F) (st : Statement F) (fits : S.Fits s) (lam : K)
    (τ : ℕ → K) (b N : ℕ) (cast : ∀ a c : ℕ, a < N → c < N → (a : K) = c → a = c) (b_lt : b < N)
    (block_lt : ∀ σ, σ < S.slots → S.slotBlock σ + 1 < N) :
    slotWeights s S lam τ (trueU s S st) b =
      ∑ l ∈ Finset.range s.D, τ l * evalAColumn s S st lam (s.D * b + l) := by
  classical
  rw [slotWeights_split]
  have per_slot : ∀ σ ∈ Finset.range (2 ^ s.slotVars),
      ((if blockColumn S σ + ((0 : ℕ) : K) = (b : K) then
          ubar s lam (trueU s S st) σ * laneSplit s S τ 0 σ else 0) +
        (if blockColumn S σ + ((1 : ℕ) : K) = (b : K) then
          ubar s lam (trueU s S st) σ * laneSplit s S τ 1 σ else 0)) =
      ubar s lam (trueU s S st) σ * ∑ l ∈ Finset.range s.D, τ l * runWeight s S σ (s.D * b + l) := by
    intro σ _
    by_cases real : σ < S.slots
    · exact slot_parts s S fits τ _ σ b N cast b_lt (block_lt σ real)
        (fun outside => absurd real (by omega))
    · have zero := ubar_true_outside s S st fits lam σ (by omega)
      simp [zero]
  rw [Finset.sum_congr rfl per_slot]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro l _
  rw [← ubar_runWeight s S st fits lam (s.D * b + l), Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro σ _
  ring

end CompressionLean.Layer1
