import CompressionLean.Lib.CubicField
import Mathlib.FieldTheory.Finiteness
import Mathlib.RingTheory.AdjoinRoot

/-!
Owns the fields of the library.

- `Gl`: the Goldilocks field, with its one prime fact. `Spec.F = Fin goldilocksModulus`
  is the same type, so the terminal's words are `Gl` values without a map.
- `specField`: `Spec.F` with the field structure of `Gl`, so the claims
  need no instance opened at their site.
- `Cubic = Gl[x]/(x³ - x - 1)`: the layer-1 challenge field, Plonky3's
  cubic trinomial extension (`crates/neo-spartan/src/field.rs`). It is a
  field because the trinomial is irreducible (`Lib/CubicField.lean`), and it has
  `p³` elements (`card_cubic`).
- `ChallengeField`: a field of `p³` elements with a root of `x³ - x - 1`.
  `Cubic` is one, by a global instance.
-/

namespace CompressionLean

open Polynomial NightstreamFPrime.Spec

/-! ## Goldilocks -/

/-- The Goldilocks field. -/
abbrev Gl := ZMod goldilocksModulus

instance goldilocks_fact : Fact (Nat.Prime goldilocksModulus) :=
  ⟨GoldilocksPrime.goldilocks_natPrime⟩

theorem three_le_modulus : 3 ≤ goldilocksModulus := by decide

/-- Fermat in a finite field: `a^(q-2)` is the inverse, zero at zero. -/
theorem pow_card_sub_two {L : Type} [Field L] (q : ℕ) (three : 3 ≤ q) (a : L)
    (fermat : a ≠ 0 → a ^ (q - 1) = 1) : a ^ (q - 2) = a⁻¹ := by
  by_cases zero : a = 0
  · rw [zero, zero_pow (by omega), inv_zero]
  · have one := fermat zero
    rw [show q - 1 = q - 2 + 1 by omega, pow_succ] at one
    exact eq_inv_of_mul_eq_one_left one

/-! ## The challenge field -/

/-- The challenge field of layer 1: `p³` elements and a root of `X³ - X - 1`. -/
class ChallengeField (K : Type) [Field K] [Algebra Gl K] [Fintype K] where
  root : K
  cubic : root ^ 3 = root + 1
  card : Fintype.card K = goldilocksModulus ^ 3

/-- `x³ - x - 1`, the trinomial of the challenge field. -/
noncomputable def cubic : Gl[X] := X ^ 3 - X - 1

instance cubic_fact : Fact (Irreducible cubic) := ⟨Lib.cubic_irreducible⟩

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
theorem cubic_natCast_injective (N : ℕ) (small : N ≤ goldilocksModulus) :
    ∀ a b : ℕ, a < N → b < N → (a : Cubic) = b → a = b := by
  intro a b ha hb equal
  have base : (a : Gl) = b := by
    apply (algebraMap Gl Cubic).injective
    simpa using equal
  have := congrArg ZMod.val base
  rwa [ZMod.val_natCast_of_lt (by omega), ZMod.val_natCast_of_lt (by omega)] at this

/-- `Cubic` with the root of `x³ - x - 1` (`field.rs`). -/
noncomputable instance challengeField : ChallengeField Cubic where
  root := AdjoinRoot.root cubic
  cubic := by
    have h : AdjoinRoot.mk cubic (X ^ 3 - X - 1) = 0 := AdjoinRoot.mk_self
    rw [map_sub, map_sub, map_pow, AdjoinRoot.mk_X, map_one] at h
    linear_combination h
  card := card_cubic

/-- F′'s field `Spec.F` is `Gl`, with its field structure, as a global
instance: every claim reads `Spec.F` arithmetic through it. -/
instance specField : Field F := inferInstanceAs (Field Gl)

/-- `Spec.F` has characteristic `p`, as `Gl` has. -/
instance : CharP F goldilocksModulus := inferInstanceAs (CharP Gl goldilocksModulus)

end CompressionLean
