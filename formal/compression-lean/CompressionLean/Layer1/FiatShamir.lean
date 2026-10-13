import CompressionLean.Layer1.Budget
import CompressionLean.Layer1.Restoration

/-!
Owns layer-1 soundness against a Fiat–Shamir prover in the
state-restoration experiment (`layer1_restore_sound`), and its value at
production sizes for `2^64` queries (`production_restore`).

Each query costs at most the whole interactive error `layer1Error`. The
largest single term is the setup-query draw, so a prover that hashes `Q`
times gains about `Q·L1·2^-t`.

Does not own: the step from Poseidon2 to the experiment (see
`Restoration.lean`); the bias of `sample_bits`, which takes the low 24 bits
of a uniform Goldilocks element (each index is at most `1 + 2^-40` times as
likely as under the uniform law, so the query term grows by a factor below
`(1 + 2^-40)^t`); or the Fiat–Shamir soundness of WHIR itself.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

/-- An accepted layer-1 result has a P0 candidate that is a CE(B) witness. -/
def Extracts (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) {L : ℕ}
    (r : Option (Transcript F K s.queries L)) : Prop :=
  ∀ tr, r = some tr → ∃ w ∈ tr.early.candidates, Relation s Φ bar S st w.z

/-- Layer-1 soundness against a state-restoration prover with `calls` queries. -/
theorem layer1_restore_sound (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (L0 L1 L : ℕ) [NeZero L] (q : ℕ → K) (N : ℕ) (sizes : s.Sizes N)
    (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (distinct : ∀ i j, i < L → j < L → q i = q j → i = j) (fits : S.Fits s) (monic : Φ.Monic)
    (degree : Φ.natDegree = s.D) (two : 2 ≤ s.D) (calls : ℕ)
    (prover : Game.Strategy (Option (Option (Transcript F K s.queries L)))) :
    Game.prob prover (Game.restore (layer1Game s Φ bar S st L0 L1 L q) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts s Φ bar S st a) ≤
      calls * layer1Error s L0 L1 L (Fintype.card K) := by
  have proper : Φ ≠ 1 := by
    rintro rfl
    rw [natDegree_one] at degree
    omega
  exact Game.restore_sound prover _ _ _
    (layer1Game_avoids s Φ bar S st L0 L1 L q N sizes cast fits monic proper)
    (layer1Game_cost s Φ bar S st L0 L1 L q distinct (nodes_of_cast N cast sizes.big_nodes) degree two) calls

/-- At production sizes, with at least 123 setup queries, a prover with at
most `2^64` queries breaks layer 1 with probability below `2^-53`. -/
theorem production_restore (s : Shape) (prod : s.Production) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F)
    (st : Statement F) (L0 L1 : ℕ) (h0 : L0 ≤ 32) (h1 : L1 ≤ 32) (big : 2 ^ 189 ≤ Fintype.card K)
    (queries : 123 ≤ s.queries) (q : ℕ → K) (N : ℕ) (sizes : s.Sizes N)
    (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (distinct : ∀ i j, i < 2 ^ (s.setupVars - 2) → j < 2 ^ (s.setupVars - 2) → q i = q j → i = j)
    (fits : S.Fits s) (monic : Φ.Monic) (degree : Φ.natDegree = s.D) (two : 2 ≤ s.D)
    (calls : ℕ) (few : calls ≤ 2 ^ 64)
    (prover : Game.Strategy (Option (Option (Transcript F K s.queries (2 ^ (s.setupVars - 2)))))) :
    Game.prob prover (Game.restore (layer1Game s Φ bar S st L0 L1 (2 ^ (s.setupVars - 2)) q) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts s Φ bar S st a) ≤ 1 / 2 ^ 53 := by
  refine (layer1_restore_sound s Φ bar S st L0 L1 _ q N sizes cast distinct fits monic degree two calls
    prover).trans ?_
  have error := production_error s prod L0 L1 (Fintype.card K) h0 h1 big
  have query : (1 / 2 : ℝ) ^ s.queries ≤ (1 / 2) ^ 123 := pow_le_pow_of_le_one (by norm_num) (by norm_num) queries
  have bound : layer1Error s L0 L1 (2 ^ (s.setupVars - 2)) (Fintype.card K) ≤
      (2 : ℝ) ^ 40 / 2 ^ 189 + 32 * (1 / 2) ^ 123 := by linarith
  have callsBound : (calls : ℝ) ≤ 2 ^ 64 := by exact_mod_cast few
  calc (calls : ℝ) * layer1Error s L0 L1 (2 ^ (s.setupVars - 2)) (Fintype.card K)
      ≤ calls * ((2 : ℝ) ^ 40 / 2 ^ 189 + 32 * (1 / 2) ^ 123) :=
        mul_le_mul_of_nonneg_left bound (by positivity)
    _ ≤ 2 ^ 64 * ((2 : ℝ) ^ 40 / 2 ^ 189 + 32 * (1 / 2) ^ 123) :=
        mul_le_mul_of_nonneg_right callsBound (by positivity)
    _ ≤ 1 / 2 ^ 53 := by norm_num

end CompressionLean.Layer1
