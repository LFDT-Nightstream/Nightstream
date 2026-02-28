import SuperNeo.Decomp

namespace SuperNeo

private def q0Stage (a : Int) (b : Nat) : Int :=
  if a >= 0 then a / Int.ofNat b else - ((-a) / Int.ofNat b)

private def r0Stage (a : Int) (b : Nat) : Int :=
  a - q0Stage a b * Int.ofNat b

example (a : Int) (b : Nat) : a = q0Stage a b * Int.ofNat b + r0Stage a b := by
  unfold r0Stage
  have h : a - q0Stage a b * Int.ofNat b + q0Stage a b * Int.ofNat b = a := by
    simpa using (Int.sub_add_cancel a (q0Stage a b * Int.ofNat b))
  omega

example (a : Int) (b : Nat) (ha : a >= 0) :
  r0Stage a b = a % Int.ofNat b := by
  have hq : q0Stage a b = a / Int.ofNat b := by simp [q0Stage, ha]
  have hdecomp : Int.ofNat b * (a / Int.ofNat b) + a % Int.ofNat b = a := Int.mul_ediv_add_emod a (Int.ofNat b)
  have hdecomp' : (a / Int.ofNat b) * Int.ofNat b + a % Int.ofNat b = a := by
    simpa [Int.mul_comm] using hdecomp
  unfold r0Stage
  rw [hq]
  omega

example (a : Int) (b : Nat) (ha : ¬ a >= 0) :
  r0Stage a b = - ((-a) % Int.ofNat b) := by
  have hq : q0Stage a b = - ((-a) / Int.ofNat b) := by simp [q0Stage, ha]
  have hdecomp : Int.ofNat b * ((-a) / Int.ofNat b) + (-a) % Int.ofNat b = -a := Int.mul_ediv_add_emod (-a) (Int.ofNat b)
  have hdecomp' : ((-a) / Int.ofNat b) * Int.ofNat b + (-a) % Int.ofNat b = -a := by
    simpa [Int.mul_comm] using hdecomp
  unfold r0Stage
  rw [hq]
  omega

end SuperNeo
