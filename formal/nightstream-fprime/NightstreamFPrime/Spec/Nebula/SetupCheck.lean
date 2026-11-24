import Mathlib.Analysis.SpecialFunctions.Pow.Real
import NightstreamFPrime.Spec.Nebula.Plan

/-! Owns SPEC.md section 7.1, rule 6, the memory setup check: a geometry passes
exactly when one fingerprint check over `K` has error at most the
owner-approved floor `2^−109.91`, that is, when `R + M + N · B_ops ≤ 139,509`.
The check bounds the per-check value; it does not own the Fiat–Shamir query
loss. -/

namespace NightstreamFPrime.Spec.Nebula

/-- SPEC.md section 7.1, rule 6: `m_mem = R + M + N · B_ops ≤ 139,509`. -/
def Geometry.Secure (p : Geometry) : Prop := p.maxTuples ≤ 139509

/-- The integer form of `2·m/q² ≤ 2^−109.91`. -/
private theorem floor_nat_iff (m : ℕ) :
    (2 * m) ^ 100 * 2 ^ 10991 ≤ fieldModulus ^ 200 ↔ m ≤ 139509 := by
  constructor
  · intro bound
    by_contra large
    have grows : (2 * 139510) ^ 100 * 2 ^ 10991 ≤ (2 * m) ^ 100 * 2 ^ 10991 :=
      Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
    have edge : fieldModulus ^ 200 < (2 * 139510) ^ 100 * 2 ^ 10991 := by decide +kernel
    omega
  · intro small
    have grows : (2 * m) ^ 100 * 2 ^ 10991 ≤ (2 * 139509) ^ 100 * 2 ^ 10991 :=
      Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
    have edge : (2 * 139509) ^ 100 * 2 ^ 10991 ≤ fieldModulus ^ 200 := by decide +kernel
    omega

private theorem cast_floor_bound (m q a b c : ℕ) :
    (2 * m) ^ a * 2 ^ b ≤ q ^ c ↔ (2 * (m : ℝ)) ^ a * 2 ^ b ≤ (q : ℝ) ^ c := by
  rw [← Nat.cast_le (α := ℝ)]
  push_cast
  rfl

attribute [local irreducible] Real.instLT Real.instLE in
/-- Rule 6 holds exactly when
`ε_test = 2·m_mem/q² ≤ 2^−109.91`. -/
theorem Geometry.secure_iff (p : Geometry) :
    p.Secure ↔
      (2 * p.maxTuples : ℝ) / (fieldModulus : ℝ) ^ 2 ≤ (2 : ℝ) ^ (-(10991 / 100 : ℝ)) := by
  have floor : ((2 : ℝ) ^ (-(10991 / 100 : ℝ))) ^ 100 = ((2 : ℝ) ^ 10991)⁻¹ := by
    rw [← Real.rpow_mul_natCast (by norm_num),
      show (-(10991 / 100 : ℝ)) * ((100 : ℕ) : ℝ) = -((10991 : ℕ) : ℝ) by norm_num,
      Real.rpow_neg (by norm_num), Real.rpow_natCast]
  have left : (0 : ℝ) ≤ (2 * p.maxTuples : ℝ) / (fieldModulus : ℝ) ^ 2 := by positivity
  have right : (0 : ℝ) ≤ (2 : ℝ) ^ (-(10991 / 100 : ℝ)) := by positivity
  have modulus : (0 : ℝ) < (fieldModulus : ℝ) ^ 200 :=
    pow_pos (by norm_num [fieldModulus]) _
  have two : (0 : ℝ) < (2 : ℝ) ^ 10991 := pow_pos two_pos _
  rw [Geometry.Secure, ← floor_nat_iff, ← pow_le_pow_iff_left₀ left right (by norm_num : 100 ≠ 0),
    floor, div_pow, ← pow_mul, show 2 * 100 = 200 from rfl, div_le_iff₀ modulus,
    ← div_eq_inv_mul, le_div_iff₀ two]
  exact cast_floor_bound p.maxTuples fieldModulus 100 10991 200

end NightstreamFPrime.Spec.Nebula
