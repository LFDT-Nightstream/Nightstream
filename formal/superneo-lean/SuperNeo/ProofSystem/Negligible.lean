/-!
Negligible-function surface inspired by VCV-io:
- VCVio/CryptoFoundations/Asymptotics/Negligible.lean
- Apache-2.0, Copyright (c) 2024 Devon Tuma

This repository currently avoids a hard Mathlib dependency in the proof system
facade, so we use a compact eventual-zero negligible model for now.
-/

namespace SuperNeo.ProofSystem

/-- Error functions are indexed by the security parameter. -/
abbrev ErrorFn := Nat → Nat

/--
Compact negligible predicate used in the current security boundary: an error is
negligible when it becomes identically zero beyond some threshold.
-/
def IsNegligible (f : ErrorFn) : Prop :=
  ∀ _c : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → f n = 0

@[simp] theorem isNegligible_iff
  (f : ErrorFn) :
  IsNegligible f ↔ ∀ _c : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → f n = 0 := Iff.rfl

theorem isNegligible_zero : IsNegligible (fun _ => 0) :=
  by
    intro _c
    exact ⟨0, by intro n _hn; rfl⟩

theorem isNegligible_of_zero
  {f : ErrorFn}
  (hf : ∀ n, f n = 0) :
  IsNegligible f := by
  intro _c
  exact ⟨0, by intro n _hn; exact hf n⟩

end SuperNeo.ProofSystem
