import SuperNeo.Field

namespace SuperNeo
open F

example (a : F) (ha : F.Canonical a) : a - 0 = a := by
  simp

example (a : F) (ha : F.Canonical a) : a - 0 = a := by
  change F.ofNat (a.val + q - (0:F).val) = a
  simp [F.zero_val]

example (a : F) (ha : F.Canonical a) : a - 0 = a := by
  have hcanon : a.val < q := ha
  have hmod : ((a.val + q) % q) = a.val := by
    rw [Nat.add_mod]
    simp [Nat.mod_eq_of_lt hcanon]
  change F.ofNat (a.val + q) = a
  have hof : F.ofNat (a.val + q) = F.ofNat a.val := by
    cases a
    simp [F.ofNat, hmod]
  rw [hof]
  exact F.ofNat_val_eq_of_canonical ha

end SuperNeo
