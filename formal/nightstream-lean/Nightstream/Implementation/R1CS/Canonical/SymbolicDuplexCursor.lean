import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex

/-!
Contract: reduce the symbolic duplex cursor to a pure length recurrence.

Owns: the exact cursor step and the theorem that `absorbMany` depends on the
absorbed list only through its length.

Does not own: any protocol serialization, transcript state, row program, or
security claim.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-- Static cursor transition for one overwrite absorption. -/
def step (absorbed : Nat) : Nat :=
  if Poseidon2Sponge.rate ≤ absorbed then 1 else absorbed + 1

/-- Cursor after `count` consecutive absorptions. -/
def after : Nat → Nat → Nat
  | absorbed, 0 => absorbed
  | absorbed, count + 1 => after (step absorbed) count

@[simp] theorem absorb_absorbed
    (base : Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder) :
    (SymbolicDuplex.absorb base value builder).absorbed =
      step builder.absorbed := by
  unfold SymbolicDuplex.absorb SymbolicDuplex.guarded step
  by_cases full : Poseidon2Sponge.rate ≤ builder.absorbed
  · simp [full, SymbolicDuplex.permute]
  · simp [full]

/-- Field values affect lanes, but never the static cursor schedule. -/
theorem absorbMany_absorbed
    (base : Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      (SymbolicDuplex.absorbMany base values builder).absorbed =
        after builder.absorbed values.length
  | [], _ => rfl
  | value :: rest, builder => by
      simp only [SymbolicDuplex.absorbMany, after]
      rw [absorbMany_absorbed base rest, absorb_absorbed]

/-- Cursor iteration composes by addition of absorbed lengths. -/
theorem after_add (absorbed first second : Nat) :
    after absorbed (first + second) =
      after (after absorbed first) second := by
  induction first generalizing absorbed with
  | zero => simp only [Nat.zero_add, after]
  | succ first inductionHypothesis =>
      simp only [Nat.succ_add, after]
      exact inductionHypothesis (step absorbed)

/-- Four absorptions return cursor one to cursor one. -/
theorem after_one_four : after 1 4 = 1 := by
  decide

/-- Any whole number of rate-sized blocks preserves cursor one. -/
theorem after_one_four_mul (blocks : Nat) :
    after 1 (4 * blocks) = 1 := by
  induction blocks with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      rw [Nat.mul_succ, after_add, inductionHypothesis, after_one_four]

/-- The selected 13-matrix PiCCS output serialization leaves cursor one. -/
theorem after_zero_19713 : after 0 19713 = 1 := by
  rw [show 19713 = 1 + 4 * 4928 by decide, after_add]
  change after 1 (4 * 4928) = 1
  exact after_one_four_mul 4928

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor
