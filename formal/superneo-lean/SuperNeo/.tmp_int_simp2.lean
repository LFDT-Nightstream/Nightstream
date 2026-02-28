import SuperNeo.Decomp

namespace SuperNeo

example (r0 bi : Int) : bi + (r0 - bi) = r0 := by
  simp [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm]

example (q0 bi r0 : Int) : q0 * bi + r0 = (q0 + 1) * bi + (r0 - bi) := by
  simp [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm, Int.add_mul]

example (q0 bi r0 : Int) : q0 * bi + r0 = (q0 - 1) * bi + (r0 + bi) := by
  simp [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm, Int.sub_eq_add_neg, Int.add_mul]

end SuperNeo
