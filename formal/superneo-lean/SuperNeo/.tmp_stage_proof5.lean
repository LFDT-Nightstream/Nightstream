import SuperNeo.Decomp

namespace SuperNeo

private def q0Stage (a : Int) (b : Nat) : Int :=
  if a >= 0 then a / Int.ofNat b else - ((-a) / Int.ofNat b)

private def r0Stage (a : Int) (b : Nat) : Int :=
  a - q0Stage a b * Int.ofNat b

private theorem r0Stage_eq_mod_of_nonneg
  (a : Int) (b : Nat)
  (ha : a >= 0) :
  r0Stage a b = a % Int.ofNat b := by
  have hq : q0Stage a b = a / Int.ofNat b := by simp [q0Stage, ha]
  have hdecomp : Int.ofNat b * (a / Int.ofNat b) + a % Int.ofNat b = a := Int.mul_ediv_add_emod a (Int.ofNat b)
  have hdecomp' : (a / Int.ofNat b) * Int.ofNat b + a % Int.ofNat b = a := by
    simpa [Int.mul_comm] using hdecomp
  unfold r0Stage
  rw [hq]
  omega

private theorem r0Stage_eq_neg_mod_of_neg
  (a : Int) (b : Nat)
  (ha : ¬ a >= 0) :
  r0Stage a b = - ((-a) % Int.ofNat b) := by
  have hq : q0Stage a b = - ((-a) / Int.ofNat b) := by simp [q0Stage, ha]
  have hdecomp : Int.ofNat b * ((-a) / Int.ofNat b) + (-a) % Int.ofNat b = -a := Int.mul_ediv_add_emod (-a) (Int.ofNat b)
  have hdecomp' : ((-a) / Int.ofNat b) * Int.ofNat b + (-a) % Int.ofNat b = -a := by
    simpa [Int.mul_comm] using hdecomp
  have hnegMul : -((-a) / Int.ofNat b) * Int.ofNat b = - (((-a) / Int.ofNat b) * Int.ofNat b) := by
    simpa using (Int.neg_mul ((-a) / Int.ofNat b) (Int.ofNat b))
  unfold r0Stage
  rw [hq, hnegMul]
  omega

private theorem r0Stage_range
  (a : Int) {b : Nat}
  (hb : 2 ≤ b) :
  -(Int.ofNat b) < r0Stage a b ∧ r0Stage a b < Int.ofNat b := by
  have hbPos : 0 < b := Nat.lt_of_lt_of_le (by decide : 0 < 2) hb
  have hbiNe : Int.ofNat b ≠ 0 := Int.ofNat_ne_zero.mpr (Nat.ne_of_gt hbPos)
  by_cases ha : a >= 0
  · have hEq : r0Stage a b = a % Int.ofNat b := r0Stage_eq_mod_of_nonneg a b ha
    have hLoMod : 0 ≤ a % Int.ofNat b := Int.emod_nonneg a hbiNe
    have hHiMod : a % Int.ofNat b < Int.ofNat b := by
      simpa using (Int.emod_lt a hbiNe)
    constructor
    · rw [hEq]
      omega
    · rw [hEq]
      exact hHiMod
  · have hEq : r0Stage a b = - ((-a) % Int.ofNat b) := r0Stage_eq_neg_mod_of_neg a b ha
    have hLoMod : 0 ≤ (-a) % Int.ofNat b := Int.emod_nonneg (-a) hbiNe
    have hHiMod : (-a) % Int.ofNat b < Int.ofNat b := by
      simpa using (Int.emod_lt (-a) hbiNe)
    constructor
    · rw [hEq]
      omega
    · rw [hEq]
      omega

end SuperNeo
