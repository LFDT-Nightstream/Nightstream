import Mathlib.Data.Nat.Notation

/-!
Owns generic facts about natural numbers: `Nat.all` as a bounded `∀`.
-/

namespace CompressionLean.Lib

theorem nat_all_iff {n : ℕ} {f : (i : ℕ) → i < n → Bool} : Nat.all n f = true ↔ ∀ i h, f i h = true := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Nat.all_succ, Bool.and_eq_true, ih]
      exact ⟨fun ⟨below, top⟩ i h => (Nat.lt_succ_iff_lt_or_eq.mp h).elim (below i) fun e => e ▸ top,
        fun all => ⟨fun i h => all i _, all n _⟩⟩

end CompressionLean.Lib
