import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindow
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns structural row projection for one PiRLC digest window. The heavy window
owner exposes the exact child order; this module selects children without
unfolding a digest lane or Poseidon2 permutation.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindow

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

def childConstraintLists (interface : Logical.Interface) (offset : Nat) :
    List (List Expr) :=
  [flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
        interface offset 0).main (Logical.laneOffset offset 0)),
   flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
        interface offset 1).main (Logical.laneOffset offset 1)),
   flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
        interface offset 2).main (Logical.laneOffset offset 2)),
   flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
        interface offset 3).main (Logical.laneOffset offset 3)),
   flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.permutationCircuit
        interface offset).main (Logical.permutationOffset offset))]

private theorem logicalConstraints_eq_childConstraintLists
    (interface : Logical.Interface) (offset : Nat) :
    logicalConstraints interface offset =
      (childConstraintLists interface offset).flatten := by
  unfold logicalConstraints
  change flatConstraints
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.opsAt
        interface offset) = _
  rw [Logical.flatConstraints_opsAt]
  simp only [childConstraintLists, List.flatten_cons, List.flatten_nil,
    List.append_nil, List.append_assoc]

/-- Held digest-window rows project to the four lane and one permutation
constraint segments in exact parent order. -/
theorem rowsHold_implies_childSegments
    (interface : Logical.Interface) (offset : Nat) (env : Env) (start : Nat)
    (rows : R1CS.RowsHold env
      (R1CS.lowerConstraints (logicalConstraints interface offset) start).rows) :
    R1CS.SegmentsHold env (childConstraintLists interface offset) start := by
  rw [logicalConstraints_eq_childConstraintLists] at rows
  exact (R1CS.rowsHold_flatten_iff env _ start).mp rows

/-- Held digest-window segments project to one exact digest lane. -/
theorem childSegments_imply_lane
    (interface : Logical.Interface) (offset : Nat) (env : Env) (start : Nat)
    (laneFresh : ∀ lane : Fin 4,
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
            interface offset lane).main
          (Logical.laneOffset offset lane))) = 303)
    (holds : R1CS.SegmentsHold env (childConstraintLists interface offset)
      start)
    (lane : Fin 4) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints
        (flatConstraints (Circuit.ops
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
            interface offset lane).main
          (Logical.laneOffset offset lane)))
        (start + lane.val * 303)).rows := by
  simp only [childConstraintLists, R1CS.SegmentsHold] at holds
  rw [laneFresh 0, laneFresh 1, laneFresh 2, laneFresh 3] at holds
  rcases holds with ⟨lane0, lane1, lane2, lane3, _, _⟩
  fin_cases lane
  · simpa using lane0
  · simpa [Nat.add_assoc] using lane1
  · simpa [Nat.add_assoc] using lane2
  · simpa [Nat.add_assoc] using lane3

/-- Held digest-window segments project to the final owned Poseidon2
permutation after all four digest lanes. -/
theorem childSegments_imply_permutation
    (interface : Logical.Interface) (offset : Nat) (env : Env) (start : Nat)
    (laneFresh : ∀ lane : Fin 4,
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
            interface offset lane).main
          (Logical.laneOffset offset lane))) = 303)
    (holds : R1CS.SegmentsHold env (childConstraintLists interface offset)
      start) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints
        (flatConstraints (Circuit.ops
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.permutationCircuit
            interface offset).main (Logical.permutationOffset offset)))
        (start + 1212)).rows := by
  simp only [childConstraintLists, R1CS.SegmentsHold] at holds
  rw [laneFresh 0, laneFresh 1, laneFresh 2, laneFresh 3] at holds
  rcases holds with ⟨_, _, _, _, permutation, _⟩
  simpa [Nat.add_assoc] using permutation

/-- The selected raw permutation child is the exact public owned-permutation
logical constraint list. -/
theorem childSegments_imply_permutationLogical
    (interface : Logical.Interface) (offset : Nat) (env : Env) (start : Nat)
    (laneFresh : ∀ lane : Fin 4,
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
            interface offset lane).main
          (Logical.laneOffset offset lane))) = 303)
    (holds : R1CS.SegmentsHold env (childConstraintLists interface offset)
      start) :
    R1CS.RowsHold env
      (R1CS.lowerConstraints
        (NightstreamFPrime.Layout.Poseidon2.PermutationOwned.logicalConstraints
          (Logical.permutationInterface interface offset)
          (Logical.permutationOffset offset))
        (start + 1212)).rows := by
  exact childSegments_imply_permutation interface offset env start laneFresh
    holds

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindow
