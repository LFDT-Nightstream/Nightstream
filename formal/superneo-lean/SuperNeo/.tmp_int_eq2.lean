import SuperNeo.Decomp

example (a q0 bi r0 r : Int)
  (hA : a = q0 * bi + r0)
  (hr : r = r0 - bi) :
  a = (q0 + 1) * bi + r := by
  have hMul : (q0 + 1) * bi = q0 * bi + bi := by
    simpa [Int.one_mul] using (Int.add_mul q0 1 bi)
  rw [hMul, hr]
  omega

example (a q0 bi r0 r : Int)
  (hA : a = q0 * bi + r0)
  (hr : r = r0 + bi) :
  a = (q0 - 1) * bi + r := by
  have hMul : (q0 - 1) * bi = q0 * bi - bi := by
    have h : (q0 + (-1)) * bi = q0 * bi + (-1) * bi := by
      simpa using (Int.add_mul q0 (-1) bi)
    have hneg : (-1) * bi = -bi := by
      simpa using (Int.neg_one_mul bi)
    calc
      (q0 - 1) * bi = (q0 + (-1)) * bi := by rfl
      _ = q0 * bi + (-1) * bi := h
      _ = q0 * bi + (-bi) := by simpa [hneg]
      _ = q0 * bi - bi := by rfl
  rw [hMul, hr]
  omega
