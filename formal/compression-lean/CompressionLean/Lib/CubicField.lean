import Mathlib.Algebra.Polynomial.SpecificDegree
import Mathlib.FieldTheory.Finite.Basic
import Mathlib.Tactic.ComputeDegree
import Mathlib.Tactic.LinearCombination
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
Owns the irreducibility of `x³ - x - 1` over the Goldilocks field
(`cubic_irreducible`). Plonky3's `CubicTrinomialExtensionField`, the layer-1
challenge field (`crates/neo-spartan/src/field.rs`), is built on this
trinomial, so it is a field with `p³` elements.

The proof computes `x^p` in `F_p[x]/(x³ - x - 1)` with a loop the kernel
evaluates (`cubicPow`, after `GoldilocksPrime.powMod`). If the cubic had a
root `a` in `F_p`, then `a^p = a` would make `g = x^p - x` vanish at `a`. But
`g` has an explicit inverse in the quotient ring, so `0 = 1`.

The primality of `p` is an instance argument; `Fields.lean` owns the fact.
-/

namespace CompressionLean.Lib

open Polynomial NightstreamFPrime.Spec

/-- `a + b·x + c·x²` in `F_p[x]/(x³ - x - 1)`, as residues. -/
abbrev CubicResidues := ℕ × ℕ × ℕ

/-- The product, with `x³ = x + 1` and `x⁴ = x² + x`. -/
def cubicMul (p : ℕ) (u v : CubicResidues) : CubicResidues :=
  ((u.1 * v.1 + (u.2.1 * v.2.2 + u.2.2 * v.2.1)) % p,
    (u.1 * v.2.1 + u.2.1 * v.1 + (u.2.1 * v.2.2 + u.2.2 * v.2.1) + u.2.2 * v.2.2) % p,
    (u.1 * v.2.2 + u.2.1 * v.2.1 + u.2.2 * v.1 + u.2.2 * v.2.2) % p)

/-- Power by repeated squaring, structural on `fuel` so the kernel reduces it.
Fuel exhaustion returns the poison value `0`. -/
def cubicPow (p : ℕ) : ℕ → CubicResidues → ℕ → CubicResidues
  | 0, _, _ => (0, 0, 0)
  | _, _, 0 => (1 % p, 0, 0)
  | fuel + 1, base, exponent =>
      if exponent % 2 = 1 then cubicMul p (cubicPow p fuel (cubicMul p base base) (exponent / 2)) base
      else cubicPow p fuel (cubicMul p base base) (exponent / 2)

/-- The value of a residue triple at `a`. -/
def cubicEval {p : ℕ} (a : ZMod p) (u : CubicResidues) : ZMod p :=
  (u.1 : ZMod p) + (u.2.1 : ZMod p) * a + (u.2.2 : ZMod p) * a ^ 2

theorem cubicEval_mul {p : ℕ} (a : ZMod p) (root : a ^ 3 - a - 1 = 0) (u v : CubicResidues) :
    cubicEval a (cubicMul p u v) = cubicEval a u * cubicEval a v := by
  obtain ⟨u0, u1, u2⟩ := u
  obtain ⟨v0, v1, v2⟩ := v
  simp only [cubicMul, cubicEval, ZMod.natCast_mod]
  push_cast
  linear_combination (-((u1 : ZMod p) * v2 + (u2 : ZMod p) * v1 + (u2 : ZMod p) * v2 * a)) * root

theorem cubicPow_eval {p : ℕ} (a : ZMod p) (root : a ^ 3 - a - 1 = 0) (fuel : ℕ) :
    ∀ (base : CubicResidues) (exponent : ℕ), exponent < 2 ^ fuel →
      cubicEval a (cubicPow p (fuel + 1) base exponent) = cubicEval a base ^ exponent := by
  induction fuel with
  | zero =>
      intro base exponent small
      obtain rfl : exponent = 0 := by simpa using small
      simp [cubicPow, cubicEval, ZMod.natCast_mod]
  | succ fuel ih =>
      intro base exponent small
      by_cases zero : exponent = 0
      · subst zero
        simp [cubicPow, cubicEval, ZMod.natCast_mod]
      have half : exponent / 2 < 2 ^ fuel := by
        apply (Nat.div_lt_iff_lt_mul (by norm_num : 0 < 2)).2
        simpa [pow_succ, Nat.mul_comm] using small
      obtain ⟨k, hk⟩ : ∃ k, exponent = k + 1 := ⟨exponent - 1, by omega⟩
      rw [show fuel + 1 + 1 = (fuel + 1) + 1 by omega, hk]
      simp only [cubicPow]
      rw [← hk]
      have square : cubicEval a (cubicMul p base base) = cubicEval a base ^ 2 := by
        rw [cubicEval_mul a root, pow_two]
      rcases Nat.mod_two_eq_zero_or_one exponent with even | odd
      · rw [if_neg (by omega), ih _ _ half, square, ← pow_mul]
        congr 1
        omega
      · rw [if_pos odd, cubicEval_mul a root, ih _ _ half, square, ← pow_mul, ← pow_succ]
        congr 1
        omega

set_option maxRecDepth 100000 in -- fixed-size: one 64-bit exponent, not artifact data
/-- `x^p` in the quotient ring. -/
theorem frobenius :
    cubicPow goldilocksModulus 70 (0, 1, 0) goldilocksModulus =
      (10615703402128488253, 10050274602728160328, 11746561000929144102) := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: one product, not artifact data
/-- `x^p - x` times its inverse is `1`. -/
theorem frobenius_inverse :
    cubicMul goldilocksModulus (10615703402128488253, 10050274602728160327, 11746561000929144102)
        (991315191999139912, 15238614666038134874, 11086490780951302402) = (1, 0, 0) := by
  decide

/-- `x³ - x - 1` has no root in the Goldilocks field. -/
theorem cubic_no_root [Fact (Nat.Prime goldilocksModulus)] (a : ZMod goldilocksModulus) :
    a ^ 3 - a - 1 ≠ 0 := by
  intro root
  have power := cubicPow_eval a root 69 (0, 1, 0) goldilocksModulus (by decide)
  rw [show 69 + 1 = 70 from rfl, frobenius, ZMod.pow_card] at power
  have vanish : cubicEval a (10615703402128488253, 10050274602728160327, 11746561000929144102) = 0 := by
    simp only [cubicEval] at power ⊢
    push_cast at power ⊢
    linear_combination power
  have one := cubicEval_mul a root (10615703402128488253, 10050274602728160327, 11746561000929144102)
    (991315191999139912, 15238614666038134874, 11086490780951302402)
  rw [frobenius_inverse, vanish, zero_mul] at one
  simp [cubicEval] at one

/-- `x³ - x - 1` is irreducible over the Goldilocks field. -/
theorem cubic_irreducible [Fact (Nat.Prime goldilocksModulus)] :
    Irreducible (X ^ 3 - X - 1 : (ZMod goldilocksModulus)[X]) := by
  apply Polynomial.irreducible_of_degree_le_three_of_not_isRoot
  · have degree : (X ^ 3 - X - 1 : (ZMod goldilocksModulus)[X]).natDegree = 3 := by compute_degree!
    simp [degree]
  · intro a isRoot
    apply cubic_no_root a
    simpa [IsRoot] using isRoot

end CompressionLean.Lib
