import CompressionLean.Shrink.Cost
import CompressionLean.Layer1.Restoration

/-! The budget hypothesis is Plonky3's `Pcs::new` check (p3-security `report.rs`, p3-sumcheck `prescribed_pcs.rs`): the composed error is at most `2^-t`. -/

namespace CompressionLean.Shrink

open CompressionLean.Layer1

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

def rustDegrees (m n : ℕ) : ℕ := m + 8 * m + (8 - 1) + 2 * (n + 1)

theorem shrinkCount_eq_rust (m n L : ℕ) : shrinkCount m n L = L * rustDegrees m n := by
  unfold shrinkCount rustDegrees
  ring

theorem shrink_budget (m n L size t : ℕ) (big : 2 ^ 191 ≤ size) (whir : ℝ)
    (rust : whir + L * rustDegrees m n / 2 ^ 191 ≤ 1 / 2 ^ t) :
    whir + shrinkError m n L size ≤ 1 / 2 ^ t := by
  unfold shrinkError
  rw [shrinkCount_eq_rust]
  have positive : (0 : ℝ) < 2 ^ 191 := by positivity
  have sizeBig : (2 : ℝ) ^ 191 ≤ size := by exact_mod_cast big
  have field : ((L * rustDegrees m n : ℕ) : ℝ) / size ≤ L * rustDegrees m n / 2 ^ 191 := by
    push_cast
    exact div_le_div₀ (by positivity) le_rfl positive sizeBig
  linarith

def Extracts {I : Type} (T : Template F) (s : Counts) (prog : I → Run F)
    (r : Option (Transcript I F K)) : Prop :=
  ∀ tr, r = some tr → ∃ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w

theorem shrink_restore_sound {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F)
    (L : ℕ) (wf : T.WellFormed) (valid : ∀ i, (prog i).Valid T)
    (cast : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b) (calls : ℕ)
    (prover : Game.Strategy (Option (Option (Transcript I F K)))) :
    Game.prob prover (Game.restore (shrinkGame T s prog L) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts T s prog a) ≤
      calls * shrinkError (Layout.rowVars T s) (Layout.cellVars T s) L (Fintype.card K) :=
  Game.restore_sound prover _ _ _
    (shrinkGame_avoids T s prog L wf valid (shrinkNodes cast 7 le_rfl) (shrinkNodes cast 1 (by norm_num)))
    (shrinkGame_cost T s prog L (shrinkNodes cast 7 le_rfl) (shrinkNodes cast 1 (by norm_num))) calls

theorem shrink_restore_budget {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F)
    (L : ℕ) (wf : T.WellFormed) (valid : ∀ i, (prog i).Valid T)
    (cast : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b) (big : 2 ^ 191 ≤ Fintype.card K) (t : ℕ)
    (rust : (L : ℝ) * rustDegrees (Layout.rowVars T s) (Layout.cellVars T s) / 2 ^ 191 ≤ 1 / 2 ^ t)
    (calls : ℕ) (prover : Game.Strategy (Option (Option (Transcript I F K)))) :
    Game.prob prover (Game.restore (shrinkGame T s prog L) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts T s prog a) ≤ calls / 2 ^ t := by
  refine (shrink_restore_sound T s prog L wf valid cast calls prover).trans ?_
  have budget := shrink_budget _ _ L (Fintype.card K) t big 0 (by simpa using rust)
  rw [zero_add] at budget
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left budget (Nat.cast_nonneg _)

end CompressionLean.Shrink
