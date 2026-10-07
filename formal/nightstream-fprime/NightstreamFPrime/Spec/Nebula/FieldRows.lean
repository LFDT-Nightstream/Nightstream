import Mathlib.Algebra.CharP.Basic
import NightstreamFPrime.Spec.Algebra

/-! Owns the three places where the field form of the spec §8 rows differs
from the typed rows of `Rows`: the bit row of O1 and S1, the pad gate of O8 and
O9, and the integer meaning of row O4 under `W_ts ≤ 62` (spec §4.2 rule 3).
The Stage 2 circuit builder uses these to refine its rows to `StepRows`. -/

namespace NightstreamFPrime.Spec.Nebula

/-- Rows O1 and S1: `c · (c − 1) = 0` makes `c` a bit, in any domain. -/
theorem bitRow_iff {R : Type} [CommRing R] [IsDomain R] (c : R) :
    c * (c - 1) = 0 ↔ c = 0 ∨ c = 1 := by
  rw [mul_eq_zero, sub_eq_zero]

/-- Rows O8 and O9: the factor `pad + (1 − pad) · f` is `1` on a pad slot and
`f` on an active slot. -/
theorem gateRow_eq {E : Type} [CommRing E] (b : Bool) (f : E) :
    ((b.toNat : ℕ) : E) + (1 - ((b.toNat : ℕ) : E)) * f = if b then 1 else f := by
  cases b <;> simp

/-- Row O4 is an integer relation: with all terms below `2 ^ W_ts` and
`W_ts ≤ 62`, the field equation cannot wrap modulo `q`. -/
theorem o4Row_iff {R : Type} [CommRing R] [CharP R goldilocksModulus]
    {wTs wt rt diff : ℕ} (narrow : wTs ≤ 62) (hwt : wt < 2 ^ wTs) (hrt : rt < 2 ^ wTs)
    (hdiff : diff < 2 ^ wTs) :
    ((wt : R) - rt - 1 - diff = 0) ↔ wt = rt + 1 + diff := by
  have wide : 2 ^ wTs ≤ 2 ^ 62 := Nat.pow_le_pow_right Nat.zero_lt_two narrow
  have modulus : 2 * 2 ^ 62 < goldilocksModulus := by simp [goldilocksModulus]
  rw [sub_sub, sub_sub, sub_eq_zero, ← add_assoc, ← Nat.cast_one, ← Nat.cast_add, ← Nat.cast_add,
    CharP.natCast_eq_natCast R goldilocksModulus]
  exact ⟨fun h => h.eq_of_lt_of_lt (by omega) (by omega),
    fun h => by subst h; exact Nat.ModEq.refl _⟩

/-- Row O4 gives `rt < wt`. -/
theorem o4Row_fresh {R : Type} [CommRing R] [CharP R goldilocksModulus]
    {wTs wt rt diff : ℕ} (narrow : wTs ≤ 62) (hwt : wt < 2 ^ wTs) (hrt : rt < 2 ^ wTs)
    (hdiff : diff < 2 ^ wTs) (row : (wt : R) - rt - 1 - diff = 0) : rt < wt := by
  have := (o4Row_iff narrow hwt hrt hdiff).1 row
  omega

end NightstreamFPrime.Spec.Nebula
