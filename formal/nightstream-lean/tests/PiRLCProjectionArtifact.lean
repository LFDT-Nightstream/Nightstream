import Nightstream.Implementation.R1CS.PiRLCProjectionSound

namespace NightstreamTests.PiRLCProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRLCProjection
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.ProjectionCheck

set_option maxRecDepth 262144

private theorem assignmentCanonical_of_list (witness : List Nat)
    (valuesCanonical : ∀ value ∈ witness, value < goldilocksP) :
    ∀ column, assignmentOf witness column < goldilocksP := by
  intro column
  simp only [assignmentOf]
  by_cases inRange : column < witness.length
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inRange]
    simp only [Option.getD_some]
    exact valuesCanonical witness[column] (List.getElem_mem inRange)
  · rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_none (Nat.not_lt.mp inRange)]
    decide

private theorem honestCanonical :
    ∀ column, assignmentOf honestWitness column < goldilocksP :=
  assignmentCanonical_of_list honestWitness (by native_decide)

private theorem badRootCanonical :
    ∀ column, assignmentOf badRootWitness column < goldilocksP :=
  assignmentCanonical_of_list badRootWitness (by native_decide)

example : BatchAccepted K.ops
    [projectionTrace.identity (assignmentOf honestWitness)] := by
  exact exactRows_imply_batchAccepted honestCanonical (by native_decide)
    honest_satisfies

example : BatchExact
    [projectionTrace.identity (assignmentOf honestWitness)] := by
  intro identity member
  simp only [List.mem_singleton] at member
  subst identity
  native_decide

/-- The soundness theorem intentionally accepts this row-satisfying witness;
the coefficient mismatch is precisely the explicit bad-root branch. -/
example : BatchAccepted K.ops
    [projectionTrace.identity (assignmentOf badRootWitness)] := by
  exact exactRows_imply_batchAccepted badRootCanonical (by native_decide)
    badRoot_satisfies

example : ¬ BatchExact
    [projectionTrace.identity (assignmentOf badRootWitness)] := by
  intro exact
  have identityExact := exact
    (projectionTrace.identity (assignmentOf badRootWitness)) (by simp)
  have notExact : ¬ (projectionTrace.identity
      (assignmentOf badRootWitness)).Exact := by
    native_decide
  exact notExact identityExact

def forgedWitness : List Nat :=
  honestWitness.set 929 ((honestWitness.getD 929 0 + 1) % goldilocksP)

/-- Mutating a constrained derived column is rejected by the exact rows. -/
example : ¬ Satisfies rows (assignmentOf forgedWitness) := by
  native_decide

end NightstreamTests.PiRLCProjectionArtifact
