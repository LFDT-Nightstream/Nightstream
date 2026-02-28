import SuperNeo.Decomp

namespace SuperNeo

private def balancedResidue' (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  let half := Int.ofNat (b / 2)
  let q0 := if a >= 0 then a / bi else - ((-a) / bi)
  let r0 := a - q0 * bi
  let r1 := if r0 > half then r0 - bi else r0
  if r1 < -half then r1 + bi else r1

private def q0Stage (a : Int) (b : Nat) : Int :=
  if a >= 0 then a / Int.ofNat b else - ((-a) / Int.ofNat b)

private def r0Stage (a : Int) (b : Nat) : Int :=
  a - q0Stage a b * Int.ofNat b

private theorem r0Stage_decompose (a : Int) (b : Nat) :
  a = q0Stage a b * Int.ofNat b + r0Stage a b := by
  unfold r0Stage
  have h : a - q0Stage a b * Int.ofNat b + q0Stage a b * Int.ofNat b = a := by
    simpa using (Int.sub_add_cancel a (q0Stage a b * Int.ofNat b))
  omega

private theorem balancedResidue_divisible
  (a : Int) {b : Nat} (_hb : 2 ≤ b) :
  ∃ q : Int, a = q * Int.ofNat b + balancedResidue' a b := by
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
    · have hBr : balancedResidue' a b = r1 + bi := by
        unfold balancedResidue'
        simp [q0Stage, r0Stage, bi, half, r0, r1, hUp, hDn]
      refine ⟨q0, ?_⟩
      rw [hBr, hr1]
      omega
    · have hBr : balancedResidue' a b = r1 := by
        unfold balancedResidue'
        simp [q0Stage, r0Stage, bi, half, r0, r1, hUp, hDn]
      refine ⟨q0 + 1, ?_⟩
      rw [hBr, hr1]
      have hMul : (q0 + 1) * bi = q0 * bi + bi := by
        simpa [Int.one_mul] using (Int.add_mul q0 1 bi)
      rw [hMul]
      omega
  · have hr1 : r1 = r0 := by simp [r1, hUp]
    by_cases hDn : r1 < -half
    · have hBr : balancedResidue' a b = r1 + bi := by
        unfold balancedResidue'
        simp [q0Stage, r0Stage, bi, half, r0, r1, hUp, hDn]
      refine ⟨q0 - 1, ?_⟩
      rw [hBr, hr1]
      have hMul : (q0 - 1) * bi = q0 * bi - bi := by
        have h : (q0 + (-1)) * bi = q0 * bi + (-1) * bi := by
          simpa using (Int.add_mul q0 (-1) bi)
        calc
          (q0 - 1) * bi = (q0 + (-1)) * bi := by rfl
          _ = q0 * bi + (-1) * bi := h
          _ = q0 * bi - bi := by simp
      rw [hMul]
      omega
    · have hBr : balancedResidue' a b = r1 := by
        unfold balancedResidue'
        simp [q0Stage, r0Stage, bi, half, r0, r1, hUp, hDn]
      refine ⟨q0, ?_⟩
      rw [hBr, hr1]
      exact hA

end SuperNeo
