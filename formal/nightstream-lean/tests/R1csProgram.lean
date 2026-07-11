import Nightstream.Implementation.R1CS.Program

namespace NightstreamTests.R1csProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def sampleDefinitions : List Definition :=
  [ ⟨3, .linear [(1, 1), (2, 1)]⟩
  , ⟨4, .product [(3, 1)] [(2, 1)]⟩
  ]

example : WellFormed [0, 1, 2] sampleDefinitions := by
  apply WellFormed.cons
  · simp [ReferencesOnly, Rhs.refs]
  · simp
  · apply WellFormed.cons
    · simp [ReferencesOnly, Rhs.refs]
    · simp
    · exact .nil _

def sampleWitness : List Nat := [1, 5, 7, 12, 84]

example : Satisfies (sampleDefinitions.map Definition.row)
    (assignmentOf sampleWitness) := by native_decide

example : ∀ definition ∈ sampleDefinitions, definition.Canonical := by
  native_decide

example : Satisfies (sampleDefinitions.map Definition.builderRow)
    (assignmentOf sampleWitness) := by native_decide

example : (run (assignmentOf sampleWitness) sampleDefinitions) 4 = 84 := by
  native_decide

example : Satisfies (sampleDefinitions.map Definition.builderRow)
    (run (assignmentOf sampleWitness) sampleDefinitions) := by
  apply run_satisfies_builder_rows
      (known := [0, 1, 2])
  · apply WellFormed.cons
    · simp [ReferencesOnly, Rhs.refs]
    · simp
    · apply WellFormed.cons
      · simp [ReferencesOnly, Rhs.refs]
      · simp
      · exact .nil _
  · intro column
    simp only [assignmentOf]
    by_cases inRange : column < sampleWitness.length
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inRange]
      simp only [Option.getD_some]
      have member : sampleWitness[column] ∈ sampleWitness := List.getElem_mem inRange
      exact (by native_decide : ∀ value ∈ sampleWitness, value < goldilocksP)
        sampleWitness[column] member
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.not_lt.mp inRange)]
      decide
  · simp
  · decide
  · native_decide

end NightstreamTests.R1csProgram
