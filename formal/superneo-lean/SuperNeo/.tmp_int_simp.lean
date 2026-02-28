import SuperNeo.Decomp

namespace SuperNeo

example (r0 bi : Int) : bi + (r0 - bi) = r0 := by
  simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

example (q0 bi r0 : Int) : q0 * bi + r0 = (q0 + 1) * bi + (r0 - bi) := by
  simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm, Int.add_mul, Int.mul_add]

end SuperNeo
