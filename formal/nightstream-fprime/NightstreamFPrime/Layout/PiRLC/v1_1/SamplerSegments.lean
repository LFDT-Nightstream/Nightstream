import NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns structural row projection for one PiRLC scalar sampler. The heavy sampler
owner exposes only its ordered child list; this module projects held rows
without unfolding any entry, digest-window, or selector child.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Sampler

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

private theorem rowsHold_appendAll_iff (env : Env)
    (lists : List (List Expr)) (start : Nat) :
    R1CS.RowsHold env (R1CS.lowerConstraints (appendAll lists) start).rows ↔
      R1CS.SegmentsHold env lists start := by
  induction lists generalizing start with
  | nil => simp [appendAll, R1CS.SegmentsHold, R1CS.RowsHold,
      R1CS.lowerConstraints]
  | cons first rest inductionHypothesis =>
      cases rest with
      | nil => simp [appendAll, R1CS.SegmentsHold]
      | cons second tail =>
          rw [appendAll, R1CS.lowerConstraints_append_rows,
            R1CS.rowsHold_append]
          simp only [R1CS.SegmentsHold]
          exact and_congr Iff.rfl
            (inductionHypothesis _)

/-- Held rows of one scalar sampler project to its entry, eight digest-window,
and selector child segments without unfolding any child constraint list. -/
theorem rowsHold_implies_childSegments
    (interface : Logical.Interface) (coordinate offset : Nat)
    (env : Env) (start : Nat)
    (rows : R1CS.RowsHold env
      (R1CS.lowerConstraints
        (logicalConstraints interface coordinate offset) start).rows) :
    R1CS.SegmentsHold env (childConstraintLists interface coordinate offset)
      start := by
  rw [logicalConstraints_eq_ordered] at rows
  exact (rowsHold_appendAll_iff env _ start).mp rows

/-- Held child segments project to one of the eight digest windows. The exact
start follows only from the entry and window child fresh counts. -/
theorem childSegments_imply_window
    (interface : Logical.Interface) (coordinate offset : Nat)
    (env : Env) (start : Nat)
    (entryFresh :
      R1CS.totalFreshCount
        (childConstraints (Logical.entryCircuit interface coordinate)
          (Logical.entryOffset offset)) = 0)
    (windowFresh : ∀ round : Nat,
      R1CS.totalFreshCount
        (childConstraints
          (Logical.windowCircuit interface coordinate offset round)
          (Logical.windowOffset offset round)) = 1212)
    (holds : R1CS.SegmentsHold env
      (childConstraintLists interface coordinate offset) start)
    (round : Fin 8) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints
        (childConstraints
          (Logical.windowCircuit interface coordinate offset round.val)
          (Logical.windowOffset offset round.val))
        (start + round.val * 1212)).rows := by
  simp only [childConstraintLists, R1CS.SegmentsHold] at holds
  rw [entryFresh, windowFresh 0, windowFresh 1, windowFresh 2,
    windowFresh 3, windowFresh 4, windowFresh 5, windowFresh 6,
    windowFresh 7] at holds
  rcases holds with
    ⟨_, window0, window1, window2, window3, window4, window5, window6,
      window7, _, _⟩
  fin_cases round
  · simpa using window0
  · simpa [Nat.add_assoc] using window1
  · simpa [Nat.add_assoc] using window2
  · simpa [Nat.add_assoc] using window3
  · simpa [Nat.add_assoc] using window4
  · simpa [Nat.add_assoc] using window5
  · simpa [Nat.add_assoc] using window6
  · simpa [Nat.add_assoc] using window7

/-- Held child segments project to the selector after the entry and all eight
digest windows. -/
theorem childSegments_imply_selector
    (interface : Logical.Interface) (coordinate offset : Nat)
    (env : Env) (start : Nat)
    (entryFresh :
      R1CS.totalFreshCount
        (childConstraints (Logical.entryCircuit interface coordinate)
          (Logical.entryOffset offset)) = 0)
    (windowFresh : ∀ round : Nat,
      R1CS.totalFreshCount
        (childConstraints
          (Logical.windowCircuit interface coordinate offset round)
          (Logical.windowOffset offset round)) = 1212)
    (holds : R1CS.SegmentsHold env
      (childConstraintLists interface coordinate offset) start) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints
        (childConstraints
          (Logical.selectorCircuit interface coordinate offset)
          (Logical.selectorOffset offset))
        (start + 9696)).rows := by
  simp only [childConstraintLists, R1CS.SegmentsHold] at holds
  rw [entryFresh, windowFresh 0, windowFresh 1, windowFresh 2,
    windowFresh 3, windowFresh 4, windowFresh 5, windowFresh 6,
    windowFresh 7] at holds
  rcases holds with
    ⟨_, _, _, _, _, _, _, _, _, selector, _⟩
  simpa [Nat.add_assoc] using selector

end NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
