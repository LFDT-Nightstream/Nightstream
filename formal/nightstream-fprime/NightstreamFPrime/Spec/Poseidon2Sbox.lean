import Mathlib.FieldTheory.Finite.Basic
import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.Poseidon2

/-! The Poseidon2 S-box is a permutation of `F` (`decisions/poseidon2-p64-instance.md`). The power
map is proved bijective for each exponent coprime to `fieldModulus − 1`, so a change of the S-box
degree, for example to `x^3`, needs only a new coprimality fact. -/

namespace NightstreamFPrime.Spec.Poseidon2

theorem pow_bijective_of_coprime {α : Nat} (coprime : Nat.Coprime α (fieldModulus - 1)) :
    Function.Bijective (fun x : F => x ^ α) := by
  have : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩
  have positive : α ≠ 0 := by
    rintro rfl
    rw [Nat.coprime_zero_left] at coprime
    exact absurd coprime (by decide)
  have unitsCoprime : (Nat.card (ZMod fieldModulus)ˣ).Coprime α := by
    rw [Nat.card_eq_fintype_card, ZMod.card_units]
    exact coprime.symm
  have injective : Function.Injective (fun x : ZMod fieldModulus => x ^ α) := by
    intro x y same
    simp only at same
    by_cases hx : x = 0
    · subst hx
      rw [zero_pow positive] at same
      exact ((pow_eq_zero_iff positive).mp same.symm).symm
    by_cases hy : y = 0
    · subst hy
      rw [zero_pow positive] at same
      exact (pow_eq_zero_iff positive).mp same
    have units := unitsCoprime.pow_left_bijective.injective
      (a₁ := Units.mk0 x hx) (a₂ := Units.mk0 y hy) (Units.ext (by simpa using same))
    simpa using congrArg Units.val units
  exact Finite.injective_iff_bijective.mp injective

theorem coprime_seven : Nat.Coprime 7 (fieldModulus - 1) := by
  decide

theorem coprime_three : Nat.Coprime 3 (fieldModulus - 1) := by
  decide

theorem sbox_eq_pow (x : F) : sbox x = x ^ 7 :=
  (by intro y; ring : ∀ y : ZMod fieldModulus, y * y * (y * y) * (y * y) * y = y ^ 7) x

theorem sbox_bijective : Function.Bijective sbox := by
  have same : sbox = fun x : F => x ^ 7 := funext sbox_eq_pow
  rw [same]
  exact pow_bijective_of_coprime coprime_seven

end NightstreamFPrime.Spec.Poseidon2
