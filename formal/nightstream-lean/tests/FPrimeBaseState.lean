import Nightstream.Implementation.R1CS.FPrimeBaseStateSound

namespace NightstreamTests.FPrimeBaseState

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeBaseState
open Nightstream.Implementation.R1CS.FPrimeBaseStateSound

example : rows.length = 31 := by decide
example : (assignmentOf honestWitness) 0 = 1 := by decide
example : Satisfies rows (assignmentOf honestWitness) := by native_decide

example : Holds (assignmentOf honestWitness) := by
  apply fPrimeBaseState_sound
  · intro column
    simp only [assignmentOf]
    by_cases inRange : column < honestWitness.length
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inRange]
      simp only [Option.getD_some]
      have : honestWitness[column] ∈ honestWitness := List.getElem_mem inRange
      exact (by native_decide : ∀ value ∈ honestWitness, value < goldilocksP)
        honestWitness[column] this
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.not_lt.mp inRange)]
      decide
  · decide
  · native_decide

end NightstreamTests.FPrimeBaseState
