import CompressionLean.Layer1.CubicField
import Mathlib.FieldTheory.Finiteness
import Mathlib.RingTheory.AdjoinRoot

/-! `Cubic` is Plonky3's `CubicTrinomialExtensionField`. -/

namespace CompressionLean.Final

open Polynomial NightstreamFPrime.Spec

abbrev Gl := ZMod goldilocksModulus

instance goldilocks_fact : Fact (Nat.Prime goldilocksModulus) :=
  ⟨GoldilocksPrime.goldilocks_natPrime⟩

noncomputable def cubic : Gl[X] := X ^ 3 - X - 1

instance cubic_fact : Fact (Irreducible cubic) := ⟨Layer1.cubic_irreducible⟩

abbrev Cubic := AdjoinRoot cubic

theorem cubic_natDegree : cubic.natDegree = 3 := by
  unfold cubic
  compute_degree!

theorem cubic_ne_zero : cubic ≠ 0 := cubic_fact.out.ne_zero

instance : Module.Finite Gl Cubic := (AdjoinRoot.powerBasis cubic_ne_zero).finite

noncomputable instance : Fintype Cubic :=
  haveI : Finite Cubic := Module.finite_of_finite Gl
  Fintype.ofFinite Cubic

noncomputable instance : DecidableEq Cubic := Classical.decEq _

theorem card_cubic : Fintype.card Cubic = goldilocksModulus ^ 3 := by
  rw [Fintype.card_eq_nat_card, Module.natCard_eq_pow_finrank (K := Gl), Nat.card_zmod,
    (AdjoinRoot.powerBasis cubic_ne_zero).finrank, AdjoinRoot.powerBasis_dim, cubic_natDegree]

theorem cubic_big : 2 ^ 191 ≤ Fintype.card Cubic := by
  rw [card_cubic]
  norm_num [goldilocksModulus]

theorem natCast_injective (N : ℕ) (small : N ≤ goldilocksModulus) :
    ∀ a b : ℕ, a < N → b < N → (a : Cubic) = b → a = b := by
  intro a b ha hb equal
  have base : (a : Gl) = b := by
    apply (algebraMap Gl Cubic).injective
    simpa using equal
  have := congrArg ZMod.val base
  rwa [ZMod.val_natCast_of_lt (by omega), ZMod.val_natCast_of_lt (by omega)] at this

end CompressionLean.Final
