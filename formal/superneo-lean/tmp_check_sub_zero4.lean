import SuperNeo.Field

namespace SuperNeo
open F

example (a : F) (ha : F.Canonical a) : a - 0 = a := by
  cases a with
  | mk v =>
      unfold F.Canonical at ha
      change F.ofNat (v + q - 0) = { val := v }
      rw [Nat.sub_zero]
      have hmod : (v + q) % q = v := by
        rw [Nat.add_mod]
        simp [Nat.mod_eq_of_lt ha]
      unfold F.ofNat
      simp [hmod]

end SuperNeo
