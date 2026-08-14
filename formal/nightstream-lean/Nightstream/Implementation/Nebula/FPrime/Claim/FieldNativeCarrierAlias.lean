import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-!
Contract: zero-copy field-native NIFS carrier alias.

The successor relation can avoid bit decomposition and equality rows for the
complete running carrier and mandatory bundle only when the generated
manifest proves physical column identity. This file states that condition and
derives equality of all decoded values for every assignment.

Empty arithmetic rows do not establish the alias. The manifest-owned column
identity is mandatory and is tested by a countermodel.

Does not own canonical external bytes, generated manifests, NIFS arithmetic,
or a selected production profile.

Emits constraints: zero.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FieldNativeCarrierAlias

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

def runningCarrierValues
    (contract : AliasContract) (assignment : Nat -> Nat) :
    Fin runningFieldCoordinates -> Nat :=
  fun coordinate => assignment (contract.runningCarrierColumn coordinate)

def nifsRunningValues
    (contract : AliasContract) (assignment : Nat -> Nat) :
    Fin runningFieldCoordinates -> Nat :=
  fun coordinate => assignment (contract.nifsRunningColumn coordinate)

def bundleCarrierValues
    (contract : AliasContract) (assignment : Nat -> Nat) :
    Fin bundleFieldCoordinates -> Nat :=
  fun coordinate => assignment (contract.bundleCarrierColumn coordinate)

def nifsBundleValues
    (contract : AliasContract) (assignment : Nat -> Nat) :
    Fin bundleFieldCoordinates -> Nat :=
  fun coordinate => assignment (contract.nifsBundleColumn coordinate)

/-- Manifest column identity makes the complete running NIFS input identical
to the carried running state without copy rows. -/
theorem runningValues_eq
    (contract : AliasContract) (assignment : Nat -> Nat) :
    nifsRunningValues contract assignment =
      runningCarrierValues contract assignment := by
  funext coordinate
  rw [nifsRunningValues, runningCarrierValues,
    contract.runningColumnsEqual coordinate]

/-- Manifest column identity makes the complete fresh bundle identical to
the carried mandatory bundle without copy rows. -/
theorem bundleValues_eq
    (contract : AliasContract) (assignment : Nat -> Nat) :
    nifsBundleValues contract assignment =
      bundleCarrierValues contract assignment := by
  funext coordinate
  rw [nifsBundleValues, bundleCarrierValues,
    contract.bundleColumnsEqual coordinate]

/-! ## Generated-exponent alias -/

def runningCarrierValuesFor
    {rowVariables : Nat} (contract : AliasContractFor rowVariables)
    (assignment : Nat -> Nat) :
    Fin (runningFieldCoordinatesFor rowVariables) -> Nat :=
  fun coordinate => assignment (contract.runningCarrierColumn coordinate)

def nifsRunningValuesFor
    {rowVariables : Nat} (contract : AliasContractFor rowVariables)
    (assignment : Nat -> Nat) :
    Fin (runningFieldCoordinatesFor rowVariables) -> Nat :=
  fun coordinate => assignment (contract.nifsRunningColumn coordinate)

def bundleCarrierValuesFor
    {rowVariables : Nat} (contract : AliasContractFor rowVariables)
    (assignment : Nat -> Nat) : Fin bundleFieldCoordinates -> Nat :=
  fun coordinate => assignment (contract.bundleCarrierColumn coordinate)

def nifsBundleValuesFor
    {rowVariables : Nat} (contract : AliasContractFor rowVariables)
    (assignment : Nat -> Nat) : Fin bundleFieldCoordinates -> Nat :=
  fun coordinate => assignment (contract.nifsBundleColumn coordinate)

theorem runningValuesFor_eq
    {rowVariables : Nat} (contract : AliasContractFor rowVariables)
    (assignment : Nat -> Nat) :
    nifsRunningValuesFor contract assignment =
      runningCarrierValuesFor contract assignment := by
  funext coordinate
  rw [nifsRunningValuesFor, runningCarrierValuesFor,
    contract.runningColumnsEqual coordinate]

theorem bundleValuesFor_eq
    {rowVariables : Nat} (contract : AliasContractFor rowVariables)
    (assignment : Nat -> Nat) :
    nifsBundleValuesFor contract assignment =
      bundleCarrierValuesFor contract assignment := by
  funext coordinate
  rw [nifsBundleValuesFor, bundleCarrierValuesFor,
    contract.bundleColumnsEqual coordinate]

/-- The alias itself emits no arithmetic constraints. -/
def rows : List Row := []

@[simp] theorem rows_length : rows.length = 0 := rfl

theorem rows_satisfied (assignment : Nat -> Nat) :
    Satisfies rows assignment := by
  simp [rows, Satisfies]

end Nightstream.Implementation.Nebula.FieldNativeCarrierAlias
