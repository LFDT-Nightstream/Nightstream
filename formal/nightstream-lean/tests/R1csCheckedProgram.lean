import Nightstream.Implementation.R1CS.Core.CheckedProgram

namespace NightstreamTests.R1csCheckedProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

def sampleProgram : List Instruction :=
  [.define ⟨2, .linear [(1, 1), (0, 1)]⟩,
   .check (bitRow 1)]

def sampleInputColumns : List Nat := [0, 1]

example : WellFormed sampleInputColumns (definitions sampleProgram) := by decide

example : ∀ definition ∈ definitions sampleProgram, definition.Canonical := by
  decide

example :
    ChecksReference
      (knownAfter sampleInputColumns (definitions sampleProgram)) sampleProgram := by
  decide

def honestInput : Nat → Nat := assignmentOf [1, 1]
def rejectedInput : Nat → Nat := assignmentOf [1, 2]

example : ChecksHold honestInput sampleProgram := by native_decide
example : ¬ ChecksHold rejectedInput sampleProgram := by native_decide

example : Satisfies (rows sampleProgram) (interpret honestInput sampleProgram) := by
  apply complete (inputColumns := sampleInputColumns)
    (instructions := sampleProgram) (by decide) (by decide)
  · intro column
    simp only [honestInput, assignmentOf]
    by_cases inRange : column < [1, 1].length
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inRange]
      simp only [Option.getD_some]
      have member : [1, 1][column] ∈ [1, 1] := List.getElem_mem inRange
      exact (by decide : ∀ value ∈ [1, 1], value < goldilocksP)
        [1, 1][column] member
    · rw [List.getD_eq_getElem?_getD,
        List.getElem?_eq_none (Nat.not_lt.mp inRange)]
      decide
  · decide
  · decide
  · native_decide

end NightstreamTests.R1csCheckedProgram
