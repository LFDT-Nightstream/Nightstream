import SuperNeo.Decomp

namespace SuperNeo

private def q0Stage (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  if a >= 0 then a / bi else - ((-a) / bi)

private def r0Stage (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  a - q0Stage a b * bi

private theorem r0Stage_decompose (a : Int) (b : Nat) :
  a = q0Stage a b * Int.ofNat b + r0Stage a b := by
  unfold r0Stage
  omega

private theorem r0Stage_eq_mod_of_nonneg
  (a : Int) (b : Nat)
  (ha : a >= 0) :
  r0Stage a b = a % Int.ofNat b := by
  let bi : Int := Int.ofNat b
  have hq : q0Stage a b = a / bi := by
    simp [q0Stage, bi, ha]
  have hdecomp : a = bi * (a / bi) + a % bi := by
    simpa [Int.mul_comm] using (Int.ediv_add_emod a bi)
  unfold r0Stage
  rw [hq]
  omega

private theorem r0Stage_eq_neg_mod_of_neg
  (a : Int) (b : Nat)
  (ha : ¬ a >= 0) :
  r0Stage a b = - ((-a) % Int.ofNat b) := by
  let bi : Int := Int.ofNat b
  have hq : q0Stage a b = - ((-a) / bi) := by
    simp [q0Stage, bi, ha]
  have hdecomp : (-a) = bi * ((-a) / bi) + (-a) % bi := by
    simpa [Int.mul_comm] using (Int.ediv_add_emod (-a) bi)
  unfold r0Stage
  rw [hq]
  omega

private theorem r0Stage_range
  (a : Int) {b : Nat}
  (hb : 2 ≤ b) :
  -(Int.ofNat b) < r0Stage a b ∧ r0Stage a b < Int.ofNat b := by
  have hbPos : 0 < b := Nat.lt_of_lt_of_le (by decide : 0 < 2) hb
  have hbiNe : Int.ofNat b ≠ 0 := by
    exact Int.ofNat_ne_zero.mpr (Nat.ne_of_gt hbPos)
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

private theorem balancedResidue_divisible
  (a : Int) {b : Nat} (hb : 2 ≤ b) :
  ∃ q : Int, a = q * Int.ofNat b + balancedResidue a b := by
  let bi : Int := Int.ofNat b
  let half : Int := Int.ofNat (b / 2)
  let q0 : Int := q0Stage a b
  let r0 : Int := r0Stage a b
  let r1 : Int := if r0 > half then r0 - bi else r0
  have hA : a = q0 * bi + r0 := by
    simpa [q0, r0, bi] using r0Stage_decompose a b
  by_cases hUp : r0 > half
  · have hr1 : r1 = r0 - bi := by simp [r1, hUp]
    by_cases hDn : r1 < -half
    · have hBr : balancedResidue a b = r1 + bi := by
        simp [balancedResidue, q0Stage, r0Stage, bi, half, q0, r0, r1, hUp, hDn]
      refine ⟨q0, ?_⟩
      rw [hBr]
      rw [hr1]
      omega
    · have hBr : balancedResidue a b = r1 := by
        simp [balancedResidue, q0Stage, r0Stage, bi, half, q0, r0, r1, hUp, hDn]
      refine ⟨q0 + 1, ?_⟩
      rw [hBr]
      rw [hr1]
      have hMul : (q0 + 1) * bi = q0 * bi + bi := by
        simpa [Int.one_mul] using (Int.add_mul q0 1 bi)
      rw [hMul]
      omega
  · have hr1 : r1 = r0 := by simp [r1, hUp]
    by_cases hDn : r1 < -half
    · have hBr : balancedResidue a b = r1 + bi := by
        simp [balancedResidue, q0Stage, r0Stage, bi, half, q0, r0, r1, hUp, hDn]
      refine ⟨q0 - 1, ?_⟩
      rw [hBr]
      rw [hr1]
      have hMul : (q0 - 1) * bi = q0 * bi - bi := by
        have h : (q0 + (-1)) * bi = q0 * bi + (-1) * bi := by
          simpa using (Int.add_mul q0 (-1) bi)
        calc
          (q0 - 1) * bi = (q0 + (-1)) * bi := by rfl
          _ = q0 * bi + (-1) * bi := h
          _ = q0 * bi - bi := by simp
      rw [hMul]
      omega
    · have hBr : balancedResidue a b = r1 := by
        simp [balancedResidue, q0Stage, r0Stage, bi, half, q0, r0, r1, hUp, hDn]
      refine ⟨q0, ?_⟩
      rw [hBr]
      rw [hr1]
      exact hA

end SuperNeo
