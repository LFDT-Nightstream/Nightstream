import CompressionLean.Layer1.Budget
import CompressionLean.Layer1.Restoration

namespace CompressionLean.Layer1

open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

def Extracts (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) {L : ℕ}
    (r : Option (Transcript F K s.queries L)) : Prop :=
  ∀ tr, r = some tr → ∃ w ∈ tr.early.candidates, Relation s Φ bar S tr.statement w.z

theorem layer1_restore_sound (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F)
    (L0 L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K) (N : ℕ) (sizes : s.Sizes N)
    (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (distinct : ∀ i j, i < L → j < L → q i = q j → i = j) (fits : S.Fits s) (monic : Φ.Monic)
    (degree : Φ.natDegree = s.D) (two : 2 ≤ s.D) (calls : ℕ)
    (prover : Game.Strategy (Option (Option (Transcript F K s.queries L)))) :
    Game.prob prover (Game.restore (layer1Game s Φ bar S L0 L1 L P q) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts s Φ bar S a) ≤
      calls * layer1Error s L0 L1 L P (Fintype.card K) := by
  have proper : Φ ≠ 1 := by
    rintro rfl
    rw [natDegree_one] at degree
    omega
  exact Game.restore_sound prover _ _ _
    (layer1Game_avoids s Φ bar S L0 L1 L P q N sizes cast fits monic proper)
    (layer1Game_cost s Φ bar S L0 L1 L P q distinct (nodes_of_cast N cast sizes.big_nodes) degree two) calls

theorem layer1_restore_budget (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (L0 L1 : ℕ)
    (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (big : 2 ^ 191 ≤ Fintype.card K) (index : s.setupVars ≤ 35) (t : ℕ)
    (p0 : (L0 : ℝ) * L1 * rustDegrees s / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ (s.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (q : ℕ → K) (N : ℕ) (sizes : s.Sizes N) (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (distinct : ∀ i j, i < 2 ^ (s.setupVars - 2) → j < 2 ^ (s.setupVars - 2) → q i = q j → i = j)
    (fits : S.Fits s) (monic : Φ.Monic) (degree : Φ.natDegree = s.D) (two : 2 ≤ s.D) (calls : ℕ)
    (prover : Game.Strategy (Option (Option (Transcript F K s.queries (2 ^ (s.setupVars - 2)))))) :
    Game.prob prover (Game.restore (layer1Game s Φ bar S L0 L1 (2 ^ (s.setupVars - 2)) goldilocks q) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts s Φ bar S a) ≤ calls / 2 ^ t := by
  refine (layer1_restore_sound s Φ bar S L0 L1 _ goldilocks q N sizes cast distinct fits monic degree two calls
    prover).trans ?_
  have budget := layer1_budget s L0 L1 (Fintype.card K) t h0 h1 big index 0 0 (by simpa using p0)
    (by simpa using p1)
  rw [zero_add, zero_add] at budget
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left budget (Nat.cast_nonneg _)

end CompressionLean.Layer1
