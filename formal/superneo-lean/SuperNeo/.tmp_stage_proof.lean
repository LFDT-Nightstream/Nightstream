import SuperNeo.Decomp

namespace SuperNeo

private def q0Stage (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  if a >= 0 then a / bi else - ((-a) / bi)

private def r0Stage (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  a - q0Stage a b * bi

example (a : Int) (b : Nat) : a = q0Stage a b * Int.ofNat b + r0Stage a b := by
  unfold r0Stage
  let bi : Int := Int.ofNat b
  have h : a - q0Stage a b * bi + q0Stage a b * bi = a := by
    simpa [bi] using (Int.sub_add_cancel a (q0Stage a b * bi))
  omega

example (a : Int) (b : Nat) (ha : a >= 0) :
  r0Stage a b = a % Int.ofNat b := by
  let bi : Int := Int.ofNat b
  have hq : q0Stage a b = a / bi := by simp [q0Stage, bi, ha]
  have hdecomp : bi * (a / bi) + a % bi = a := Int.mul_ediv_add_emod a bi
  unfold r0Stage
  rw [hq]
  omega

example (a : Int) (b : Nat) (ha : ¬ a >= 0) :
  r0Stage a b = - ((-a) % Int.ofNat b) := by
  let bi : Int := Int.ofNat b
  have hq : q0Stage a b = - ((-a) / bi) := by simp [q0Stage, bi, ha]
  have hdecomp : bi * ((-a) / bi) + (-a) % bi = -a := Int.mul_ediv_add_emod (-a) bi
  unfold r0Stage
  rw [hq]
  omega

end SuperNeo
