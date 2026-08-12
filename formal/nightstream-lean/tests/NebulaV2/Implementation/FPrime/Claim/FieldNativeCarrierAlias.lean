import Nightstream.Implementation.NebulaV2.FPrime.Claim.FieldNativeCarrierAlias

/-! Regression and missing-alias countermodel for the field-native carrier. -/

set_option autoImplicit false

namespace tests.NebulaV2FieldNativeCarrierAlias

open Nightstream.Implementation.NebulaV2.FieldNativeCarrierAlias
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

def aliasRowVariables : Nat := 26

def carrierColumns
    (_ : Fin (runningFieldCoordinatesFor aliasRowVariables)) : Nat := 1

def differentNifsColumns
    (_ : Fin (runningFieldCoordinatesFor aliasRowVariables)) : Nat := 2

def counterexampleAssignment (column : Nat) : Nat :=
  if column = 2 then 1 else 0

/-- Empty rows accept every assignment, but different physical columns can
decode different running claims. Thus the alias contract is necessary. -/
theorem rows_without_alias_do_not_bind_running_values :
    Nightstream.Implementation.R1CS.Satisfies rows counterexampleAssignment /\
      (fun coordinate : Fin (runningFieldCoordinatesFor aliasRowVariables) =>
          counterexampleAssignment (differentNifsColumns coordinate)) ≠
        (fun coordinate : Fin (runningFieldCoordinatesFor aliasRowVariables) =>
          counterexampleAssignment (carrierColumns coordinate)) := by
  constructor
  · exact rows_satisfied counterexampleAssignment
  · intro equal
    have atZero := congrFun equal
      (⟨0, by decide⟩ : Fin (runningFieldCoordinatesFor aliasRowVariables))
    simp [counterexampleAssignment, differentNifsColumns, carrierColumns] at atZero

#check runningValuesFor_eq
#check bundleValuesFor_eq

end tests.NebulaV2FieldNativeCarrierAlias
