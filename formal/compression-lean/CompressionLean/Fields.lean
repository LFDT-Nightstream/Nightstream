import CompressionLean.Replay.Field
import CompressionLean.Layer1.CubicField
import Mathlib.FieldTheory.Finiteness
import Mathlib.RingTheory.AdjoinRoot

/-!
Owns the fields of the library and the maps between them and replay words.

- `Gl`: the Goldilocks field, with its one prime fact. `Spec.F = Fin goldilocksModulus`
  is the same type, so the terminal's words are `Gl` values without a map.
- `Cubic = Gl[x]/(x³ - x - 1)`: the layer-1 challenge field, Plonky3's
  cubic trinomial extension (`crates/neo-spartan/src/field.rs`). It is a
  field because the trinomial is irreducible (`CubicField.lean`), and it has
  `p³` elements (`card_cubic`).
- `ChallengeField`: a field of `p³` elements with a root of `x³ - x - 1`.
  `Cubic` is one, by a global instance.
- `Replay.Word.toF`: the residue of a replay word in `Gl`. It keeps every
  word operation, and it is injective.
- `Replay.Word.cast`: a word in any ring, as the circuit writes it.
  `Replay.Word.ofF`: an F′ field value as a word; `cast_ofF` reads it back.
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

end CompressionLean

namespace CompressionLean.Replay

open NightstreamFPrime.Spec NightstreamFPrime.Export.NativePoseidon2

/-! ## Words -/

namespace Word

/-- The residue of a word. -/
def toF (a : Word) : Gl := (a.word.toNat : Gl)

theorem toF_eq (a : Word) : a.toF = ((a.word.denote).val : Gl) := by
  simp [toF, UInt64.denote, Poseidon2.ofNat, Nat.mod_eq_of_lt a.canonical]

theorem add_mk (a b : Word) : a + b = ⟨add64 a.word b.word, add64_canonical _ _ a.canonical b.canonical⟩ := by
  rfl

theorem mul_mk (a b : Word) : a * b = ⟨mul64 a.word b.word, mul64_canonical _ _⟩ := by
  rfl

theorem sub_mk (a b : Word) : a - b = ⟨sub64 a.word b.word, sub64_canonical _ _ a.canonical b.canonical⟩ := by
  rfl

theorem toF_add (a b : Word) : (a + b).toF = a.toF + b.toF := by
  rw [add_mk, toF_eq, toF_eq a, toF_eq b]
  dsimp only
  rw [add64_denote _ _ a.canonical b.canonical, Fin.val_add, ZMod.natCast_mod, Nat.cast_add]

theorem toF_mul (a b : Word) : (a * b).toF = a.toF * b.toF := by
  rw [mul_mk, toF_eq, toF_eq a, toF_eq b]
  dsimp only
  rw [mul64_denote _ _ a.canonical b.canonical, Fin.val_mul, ZMod.natCast_mod, Nat.cast_mul]

theorem toF_sub (a b : Word) : (a - b).toF = a.toF - b.toF := by
  rw [sub_mk, toF_eq, toF_eq a, toF_eq b]
  dsimp only
  rw [sub64_denote _ _ a.canonical b.canonical, Fin.sub_def]
  dsimp only
  rw [ZMod.natCast_mod, Nat.cast_add, Nat.cast_sub (b.word.denote).isLt.le, ZMod.natCast_self]
  ring

theorem toF_ofNat (n : ℕ) : (Word.ofNat n).toF = (n : Gl) := by
  rw [toF_eq]
  show (((ofNat64 n).denote).val : Gl) = _
  rw [ofNat64_denote]
  simp [Poseidon2.ofNat]

@[simp] theorem toF_lit (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Word)).toF = (OfNat.ofNat n : Gl) := by
  rw [show (OfNat.ofNat n : Word) = Word.ofNat n from rfl, toF_ofNat]
  rfl

@[simp] theorem toF_zero : (0 : Word).toF = 0 := by
  rw [show (0 : Word) = Word.ofNat 0 from rfl, toF_ofNat, Nat.cast_zero]

@[simp] theorem toF_one : (1 : Word).toF = 1 := by
  rw [show (1 : Word) = Word.ofNat 1 from rfl, toF_ofNat, Nat.cast_one]

theorem toF_neg (a : Word) : (-a).toF = -a.toF := by
  rw [show -a = 0 - a from rfl, toF_sub, toF_zero, zero_sub]

theorem toF_pow (a : Word) (e : ℕ) : (a.pow e).toF = a.toF ^ e := by
  induction e using Nat.strong_induction_on generalizing a with
  | _ e ih =>
    rw [Word.pow]
    by_cases zero : e = 0
    · rw [if_pos zero, zero, toF_one, pow_zero]
    · rw [if_neg zero]
      dsimp only
      have half := ih (e / 2) (by omega) (a * a)
      rw [toF_mul] at half
      have split := Nat.div_add_mod e 2
      split_ifs with odd
      · rw [toF_mul, half, ← pow_two, ← pow_mul, ← pow_succ]
        congr 1
        omega
      · rw [half, ← pow_two, ← pow_mul]
        congr 1
        omega

theorem toF_inv (a : Word) : a.inv.toF = a.toF⁻¹ := by
  rw [Word.inv, toF_pow]
  exact pow_card_sub_two goldilocksModulus three_le_modulus _ ZMod.pow_card_sub_one_eq_one

theorem eq_of_word {a b : Word} (words : a.word = b.word) : a = b := by
  obtain ⟨wa, ca⟩ := a
  obtain ⟨wb, cb⟩ := b
  cases words
  rfl

theorem eq_of_toF {a b : Word} (equal : a.toF = b.toF) : a = b := by
  have same := (ZMod.natCast_eq_natCast_iff' _ _ _).mp equal
  rw [Nat.mod_eq_of_lt a.canonical, Nat.mod_eq_of_lt b.canonical] at same
  exact eq_of_word (UInt64.toNat_inj.mp same)

theorem eq_of_beq {a b : Word} (equal : (a == b) = true) : a = b :=
  eq_of_word (beq_iff_eq.mp equal)

/-- A word in a ring `F` (`Gl::from_u64`). -/
@[pp_nodot] def cast {F : Type} [CommRing F] (g : Word) : F := (g.toNat : F)

/-- A word of F′'s field as a canonical word. -/
def ofF (value : F) : Word := Word.ofNat value.val

open Fin.CommRing in
/-- A word of F′'s field reads back. -/
theorem cast_ofF (x : F) : (ofF x).cast = x := by
  have denote := ofNat64_denote x.val
  have canonical := ofNat64_canonical x.val
  have value : (ofNat64 x.val).toNat = x.val := by
    have := congrArg Fin.val denote
    simpa [UInt64.denote, Poseidon2.ofNat, Nat.mod_eq_of_lt canonical, Nat.mod_eq_of_lt x.isLt] using this
  show ((ofNat64 x.val).toNat : F) = x
  rw [value]
  exact Fin.cast_val_eq_self x

end Word

end CompressionLean.Replay
