import Mathlib.Data.List.GetD
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimDigestWitnesses
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound

/-!
Contract: assignment views and canonical-residue facts for the four generated
streaming claim-digest domain witnesses.

Owns only the interpretation of the generated witness vectors as R1CS
assignments. It does not own their row satisfaction or transcript meaning.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimDigestWitnesses

def checkpoint1Assignment : Nat → Nat := assignmentOf checkpoint1Witness
def checkpoint2Assignment : Nat → Nat := assignmentOf checkpoint2Witness
def checkpoint3Assignment : Nat → Nat := assignmentOf checkpoint3Witness
def checkpoint4Assignment : Nat → Nat := assignmentOf checkpoint4Witness

private theorem assignmentOf_canonical
    {values : List Nat}
    (canonical : ∀ value ∈ values, value < goldilocksP) :
    ∀ column, assignmentOf values column < goldilocksP := by
  intro column
  by_cases bounded : column < values.length
  · rw [assignmentOf, List.getD_eq_getElem values 0 bounded]
    exact canonical values[column] (List.getElem_mem bounded)
  · rw [assignmentOf,
      List.getD_eq_default values 0 (Nat.le_of_not_gt bounded)]
    decide

theorem checkpoint1_assignment_canonical :
    ∀ column, checkpoint1Assignment column < goldilocksP := by
  apply assignmentOf_canonical
  native_decide

theorem checkpoint2_assignment_canonical :
    ∀ column, checkpoint2Assignment column < goldilocksP := by
  apply assignmentOf_canonical
  native_decide

theorem checkpoint3_assignment_canonical :
    ∀ column, checkpoint3Assignment column < goldilocksP := by
  apply assignmentOf_canonical
  native_decide

theorem checkpoint4_assignment_canonical :
    ∀ column, checkpoint4Assignment column < goldilocksP := by
  apply assignmentOf_canonical
  native_decide

theorem checkpoint_assignments_one :
    checkpoint1Assignment 0 = 1 ∧
      checkpoint2Assignment 0 = 1 ∧
      checkpoint3Assignment 0 = 1 ∧
      checkpoint4Assignment 0 = 1 := by
  native_decide

theorem checkpoint1_assignment_one : checkpoint1Assignment 0 = 1 :=
  checkpoint_assignments_one.1

theorem checkpoint2_assignment_one : checkpoint2Assignment 0 = 1 :=
  checkpoint_assignments_one.2.1

theorem checkpoint3_assignment_one : checkpoint3Assignment 0 = 1 :=
  checkpoint_assignments_one.2.2.1

theorem checkpoint4_assignment_one : checkpoint4Assignment 0 = 1 :=
  checkpoint_assignments_one.2.2.2

theorem checkpoint1_rows_satisfied :
    Satisfies Poseidon2Permutation.rows checkpoint1Assignment := by
  native_decide

theorem checkpoint2_rows_satisfied :
    Satisfies Poseidon2Permutation.rows checkpoint2Assignment := by
  native_decide

theorem checkpoint3_rows_satisfied :
    Satisfies Poseidon2Permutation.rows checkpoint3Assignment := by
  native_decide

theorem checkpoint4_rows_satisfied :
    Satisfies Poseidon2Permutation.rows checkpoint4Assignment := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
