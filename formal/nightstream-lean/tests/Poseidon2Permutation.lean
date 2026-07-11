import Nightstream.Implementation.R1CS.Poseidon2PermutationSound

namespace NightstreamTests.Poseidon2Permutation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Permutation
open Nightstream.Implementation.R1CS.Poseidon2PermutationSound

set_option maxRecDepth 65536

def sampleInput : List Nat := [1, 17, 30, 43, 56, 69, 82, 95, 108]

private theorem sampleCanonical :
    ∀ column, assignmentOf sampleInput column < goldilocksP := by
  intro column
  simp only [assignmentOf]
  by_cases inRange : column < sampleInput.length
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inRange]
    simp only [Option.getD_some]
    have member : sampleInput[column] ∈ sampleInput := List.getElem_mem inRange
    exact (by native_decide : ∀ value ∈ sampleInput, value < goldilocksP)
      sampleInput[column] member
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.not_lt.mp inRange)]
    decide

example : rows.length = 600 := by native_decide
example : definitions.length = 600 := by native_decide
example : Satisfies rows (interpret (assignmentOf sampleInput)) := by
  exact poseidon2Permutation_complete sampleCanonical (by decide)

end NightstreamTests.Poseidon2Permutation
