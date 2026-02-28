import SuperNeo.Decomp

example (q0 bi : Int) : (q0 + 1) * bi = q0 * bi + bi := by
  simpa [Int.one_mul] using (Int.add_mul q0 1 bi)

example (q0 bi : Int) : (q0 - 1) * bi = q0 * bi - bi := by
  calc
    (q0 - 1) * bi = (q0 + (-1)) * bi := by rfl
    _ = q0 * bi + (-1) * bi := by simpa using (Int.add_mul q0 (-1) bi)
    _ = q0 * bi - bi := by simp

example (a q0 bi r0 r : Int)
  (hA : a = q0 * bi + r0)
  (hr : r = r0 - bi) :
  a = (q0 + 1) * bi + r := by
  calc
    a = q0 * bi + r0 := hA
    _ = (q0 * bi + bi) + (r0 - bi) := by
      have h : q0 * bi + r0 = (q0 * bi + bi) + (r0 - bi) := by
        omega
      exact h
    _ = (q0 + 1) * bi + r := by
      simp [hr, Int.add_mul, Int.one_mul, Int.mul_comm, Int.mul_left_comm, Int.mul_assoc, Int.add_assoc, Int.add_left_comm, Int.add_comm]

