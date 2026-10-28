import CompressionLean.Layer1.CubicField
import Mathlib.FieldTheory.Finiteness
import Mathlib.RingTheory.AdjoinRoot

/-!
Owns the concrete fields of the production chain.

- `Gl`: the Goldilocks field. `Spec.F = Fin goldilocksModulus` is the same
  type, so the terminal's words are `Gl` values without a map.
- `Cubic = Gl[x]/(x³ - x - 1)`: the layer-1 challenge field, Plonky3's
  cubic trinomial extension (`crates/neo-spartan/src/field.rs`). It is a
  field because the trinomial is irreducible (`CubicField.lean`), and it has
  `p³` elements (`card_cubic`).
- `natCast_injective`: integers below `p` stay distinct in `Cubic`.
-/

namespace CompressionLean.Final

open Polynomial NightstreamFPrime.Spec

/-- The Goldilocks field. -/
abbrev Gl := ZMod goldilocksModulus

instance goldilocks_fact : Fact (Nat.Prime goldilocksModulus) :=
  ⟨GoldilocksPrime.goldilocks_natPrime⟩

/-- `x³ - x - 1`, the trinomial of the challenge field. -/
noncomputable def cubic : Gl[X] := X ^ 3 - X - 1

instance cubic_fact : Fact (Irreducible cubic) := ⟨Layer1.cubic_irreducible⟩

/-- The layer-1 challenge field. -/
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

/-- The challenge field has `p³` elements. -/
theorem card_cubic : Fintype.card Cubic = goldilocksModulus ^ 3 := by
  rw [Module.card_eq_pow_finrank (K := Gl), ZMod.card,
    (AdjoinRoot.powerBasis cubic_ne_zero).finrank, AdjoinRoot.powerBasis_dim, cubic_natDegree]

/-- The challenge field is larger than `2^191`, as Rust's charges assume
(`Budget.lean`). -/
theorem cubic_big : 2 ^ 191 ≤ Fintype.card Cubic := by
  rw [card_cubic]
  norm_num [goldilocksModulus]

/-- Integers below `N ≤ p` stay distinct in `Cubic`. -/
theorem natCast_injective (N : ℕ) (small : N ≤ goldilocksModulus) :
    ∀ a b : ℕ, a < N → b < N → (a : Cubic) = b → a = b := by
  intro a b ha hb equal
  have base : (a : Gl) = b := by
    apply (algebraMap Gl Cubic).injective
    simpa using equal
  have := congrArg ZMod.val base
  rwa [ZMod.val_natCast_of_lt (by omega), ZMod.val_natCast_of_lt (by omega)] at this

end CompressionLean.Final
