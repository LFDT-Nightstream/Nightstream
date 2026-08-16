import Mathlib.Data.List.GetD
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyDigestWitness
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound

/-!
Contract: assignment view and canonical-residue facts for the generated
PiRLC family-digest fourth checkpoint.

Owns only the interpretation of the generated witness vector as an R1CS
assignment. It does not own row semantics or transcript meaning.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyDigestWitness

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

theorem checkpoint4_assignment_canonical :
    ∀ column, checkpoint4Assignment column < goldilocksP := by
  apply assignmentOf_canonical
  native_decide

theorem checkpoint4_assignment_one : checkpoint4Assignment 0 = 1 := by
  native_decide

theorem checkpoint4_rows_satisfied :
    Satisfies Poseidon2Permutation.rows checkpoint4Assignment := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain
