import Mathlib.SetTheory.Cardinal.Finite
import NightstreamFPrime.Spec.Profile

namespace NightstreamFPrime.Spec.FieldTower

private def coefficients : K ≃ F × F where
  toFun value := (value.c0, value.c1)
  invFun value := ⟨value.1, value.2⟩
  left_inv value := by cases value; rfl
  right_inv value := by cases value; rfl

theorem base_cardinality : Nat.card F = goldilocksModulus := Nat.card_fin _

theorem extension_cardinality :
    Nat.card K = goldilocksModulus ^ productionProfile.extensionDegree := by
  rw [Nat.card_congr coefficients, Nat.card_prod, base_cardinality]
  rfl

theorem embed_injective : Function.Injective K.embed := by
  intro left right same
  exact congrArg K.c0 same

theorem basis_reconstruct (value : K) :
    K.add (K.embed value.c0) (K.mul (K.embed value.c1) ⟨0, 1⟩) = value := by
  cases value
  simp [K.add, K.mul, K.embed]

end NightstreamFPrime.Spec.FieldTower
