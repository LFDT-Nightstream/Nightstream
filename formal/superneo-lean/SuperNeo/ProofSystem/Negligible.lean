import Mathlib

/-!
Negligible-function surface inspired by VCV-io:
- VCVio/CryptoFoundations/Asymptotics/Negligible.lean
- Apache-2.0, Copyright (c) 2024 Devon Tuma

The proof-system layer uses a theorem-facing asymptotic negligible model over
security-parameter indexed rational error functions.
-/

namespace SuperNeo.ProofSystem

/-- Error functions are indexed by the security parameter. -/
abbrev ErrorFn := Nat → Rat

/-- Canonical inverse-polynomial tail bound used by negligible statements. -/
def invPolyBound (c n : Nat) : Rat :=
  1 / ((n + 1 : Rat) ^ c)

/--
Asymptotic negligible predicate:
for every polynomial degree `c`, the error is eventually bounded by
`1 / (n+1)^c`.
-/
def IsNegligible (f : ErrorFn) : Prop :=
  ∀ c : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → f n ≤ invPolyBound c n

@[simp] theorem isNegligible_iff
  (f : ErrorFn) :
  IsNegligible f ↔
    ∀ c : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → f n ≤ invPolyBound c n := Iff.rfl

private theorem invPolyBound_nonneg (c n : Nat) : 0 ≤ invPolyBound c n := by
  unfold invPolyBound
  positivity

private theorem invPolyBound_add_self_le
  (c n : Nat)
  (hn : 1 ≤ n) :
  invPolyBound (c + 2) n + invPolyBound (c + 2) n ≤ invPolyBound c n := by
  let x : Rat := n + 1
  have hx2 : (2 : Rat) ≤ x := by
    have hNat : 2 ≤ n + 1 := Nat.succ_le_succ hn
    change (2 : Rat) ≤ (n + 1 : Rat)
    exact_mod_cast hNat
  have hsq : (2 : Rat) ≤ x ^ 2 := by
    have hxnonneg : (0 : Rat) ≤ x := by
      positivity
    nlinarith [sq_nonneg (x - 2)]
  have hcalc : (2 : Rat) / x ^ (c + 2) ≤ x ^ 2 / x ^ (c + 2) := by
    have hxpow : 0 ≤ x ^ (c + 2) := by
      positivity
    gcongr
  have hpowsplit : x ^ (c + 2) = x ^ c * x ^ 2 := by
    rw [pow_add]
  have hxpowc : x ^ c ≠ 0 := ne_of_gt (by positivity : 0 < x ^ c)
  have hxpow2 : x ^ 2 ≠ 0 := ne_of_gt (by positivity : 0 < x ^ 2)
  have hxpow : x ^ (c + 2) ≠ 0 := ne_of_gt (by positivity : 0 < x ^ (c + 2))
  calc
    invPolyBound (c + 2) n + invPolyBound (c + 2) n
        = (2 : Rat) / x ^ (c + 2) := by
            simp [invPolyBound, x]
            field_simp [hxpow]
            norm_num
    _ ≤ x ^ 2 / x ^ (c + 2) := hcalc
    _ = 1 / x ^ c := by
          rw [hpowsplit]
          field_simp [hxpowc, hxpow2]
    _ = invPolyBound c n := by
          simp [invPolyBound, x]

theorem isNegligible_zero : IsNegligible (fun _ => 0) :=
  by
    intro c
    refine ⟨0, ?_⟩
    intro n _hn
    simpa using (invPolyBound_nonneg c n)

theorem isNegligible_of_zero
  {f : ErrorFn}
  (hf : ∀ n, f n = 0) :
  IsNegligible f := by
  intro c
  refine ⟨0, ?_⟩
  intro n _hn
  simpa [hf n] using (invPolyBound_nonneg c n)

theorem isNegligible_singleton
  (k : Nat)
  (a : Rat) :
  IsNegligible (fun n => if n = k then a else 0) := by
  intro c
  refine ⟨k + 1, ?_⟩
  intro n hn
  have hNe : n ≠ k := by omega
  simp [hNe]
  exact invPolyBound_nonneg c n

theorem isNegligible_add
  {f g : ErrorFn}
  (hf : IsNegligible f)
  (hg : IsNegligible g) :
  IsNegligible (fun n => f n + g n) := by
  intro c
  rcases hf (c + 2) with ⟨Nf, hNf⟩
  rcases hg (c + 2) with ⟨Ng, hNg⟩
  refine ⟨max (max Nf Ng) 1, ?_⟩
  intro n hn
  have hNf' : Nf ≤ n := by
    exact le_trans (le_max_left _ _) (le_trans (le_max_left _ _) hn)
  have hNg' : Ng ≤ n := by
    exact le_trans (le_max_right Nf Ng) (le_trans (le_max_left _ _) hn)
  have h1 : 1 ≤ n := by
    exact le_trans (le_max_right (max Nf Ng) 1) hn
  calc
    f n + g n ≤ invPolyBound (c + 2) n + invPolyBound (c + 2) n := by
      exact add_le_add (hNf n hNf') (hNg n hNg')
    _ ≤ invPolyBound c n := invPolyBound_add_self_le c n h1

end SuperNeo.ProofSystem
