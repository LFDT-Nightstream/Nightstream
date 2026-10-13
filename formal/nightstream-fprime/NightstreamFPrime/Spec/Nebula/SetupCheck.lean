import Mathlib.Analysis.SpecialFunctions.Pow.Real
import NightstreamFPrime.Spec.Nebula.Plan

/-! Owns spec §4.2 rule 6, the setup check of security note §5: a plan passes
exactly when one fingerprint check over `K` has error at most the
owner-approved floor `2^−109.91`, that is, when `R + M + N · B_ops ≤ 139,509`.
The check bounds the per-check value; it does not own the Fiat–Shamir query
loss. -/

namespace NightstreamFPrime.Spec.Nebula

/-- Spec §4.2 rule 6: `m_mem = R + M + N · B_ops ≤ 139,509`. -/
def Plan.Secure (p : Plan) : Prop := p.maxTuples ≤ 139509

/-- The integer form of `2·m/q² ≤ 2^−109.91`. -/
private theorem floor_nat_iff (m : ℕ) :
    (2 * m) ^ 100 * 2 ^ 10991 ≤ goldilocksModulus ^ 200 ↔ m ≤ 139509 := by
  constructor
  · intro bound
    by_contra large
    have grows : (2 * 139510) ^ 100 * 2 ^ 10991 ≤ (2 * m) ^ 100 * 2 ^ 10991 :=
      Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
    have edge : goldilocksModulus ^ 200 < (2 * 139510) ^ 100 * 2 ^ 10991 := by decide +kernel
    omega
  · intro small
    have grows : (2 * m) ^ 100 * 2 ^ 10991 ≤ (2 * 139509) ^ 100 * 2 ^ 10991 :=
      Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
    have edge : (2 * 139509) ^ 100 * 2 ^ 10991 ≤ goldilocksModulus ^ 200 := by decide +kernel
    omega

/-- Security note §5: rule 6 holds exactly when
`ε_test = 2·m_mem/q² ≤ 2^−109.91`. -/
theorem Plan.secure_iff (p : Plan) :
    p.Secure ↔
      (2 * p.maxTuples : ℝ) / (goldilocksModulus : ℝ) ^ 2 ≤ (2 : ℝ) ^ (-(10991 / 100 : ℝ)) := by
  have floor : ((2 : ℝ) ^ (-(10991 / 100 : ℝ))) ^ 100 = ((2 : ℝ) ^ 10991)⁻¹ := by
    rw [← Real.rpow_mul_natCast (by norm_num),
      show (-(10991 / 100 : ℝ)) * ((100 : ℕ) : ℝ) = -((10991 : ℕ) : ℝ) by norm_num,
      Real.rpow_neg (by norm_num), Real.rpow_natCast]
  have left : (0 : ℝ) ≤ (2 * p.maxTuples : ℝ) / (goldilocksModulus : ℝ) ^ 2 := by positivity
  have right : (0 : ℝ) ≤ (2 : ℝ) ^ (-(10991 / 100 : ℝ)) := by positivity
  have modulus : (0 : ℝ) < (goldilocksModulus : ℝ) ^ 200 :=
    pow_pos (by norm_num [goldilocksModulus]) _
  have two : (0 : ℝ) < (2 : ℝ) ^ 10991 := pow_pos two_pos _
  rw [Plan.Secure, ← floor_nat_iff, ← pow_le_pow_iff_left₀ left right (by norm_num : 100 ≠ 0),
    floor, div_pow, ← pow_mul, show 2 * 100 = 200 from rfl, div_le_iff₀ modulus,
    ← div_eq_inv_mul, le_div_iff₀ two]
  norm_cast

end NightstreamFPrime.Spec.Nebula
