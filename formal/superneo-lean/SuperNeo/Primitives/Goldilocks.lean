namespace SuperNeo

namespace Goldilocks

/-- Goldilocks prime modulus used by SuperNeo. -/
def q : Nat := 18446744069414584321

/-- Half-modulus floor used by centered representations. -/
def halfQ : Nat := q / 2

theorem q_pos : 0 < q := by
  decide

theorem q_ne_zero : q ≠ 0 :=
  Nat.ne_of_gt q_pos

theorem q_gt_one : 1 < q := by
  exact Nat.lt_of_lt_of_le (by decide : 1 < 2) (by
    -- `q` is a concrete constant, so this closes directly.
    decide)

theorem halfQ_lt_q : halfQ < q := by
  have hTwo : 1 < (2 : Nat) := by decide
  simpa [halfQ] using Nat.div_lt_self q_pos hTwo

theorem halfQ_le_q : halfQ ≤ q :=
  Nat.le_of_lt halfQ_lt_q

theorem one_le_halfQ : 1 ≤ halfQ := by
  decide

end Goldilocks

end SuperNeo
