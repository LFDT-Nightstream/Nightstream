import NightstreamFPrime.Layout.Stage1.PackageOrder
import NightstreamFPrime.Lifecycle.Stage1.Application

namespace NightstreamFPrime.Layout.Stage1.ApplicationInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout

def currentWordStart : Nat := 28909

def inputSourceColumn
    (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  PilotProduction.priorPreimageStart + currentWordStart + index.val

def outputSourceColumn
    (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  PilotProduction.outputPreimageStart + currentWordStart + index.val

def inputColumn (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  PackageOrder.sourceToPackage (inputSourceColumn index)

def outputColumn (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  PackageOrder.sourceToPackage (outputSourceColumn index)

def witnessStart : Nat := PackageOrder.privateColumnCount

def witnessColumn {witnessWordCount : Nat}
    (index : Fin witnessWordCount) : Nat :=
  witnessStart + index.val

def localStart (program : Lifecycle.Stage1.Application.Program) : Nat :=
  witnessStart + program.witnessWordCount

def interface (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.Application.Interface program.witnessWordCount where
  input := fun _ index => .var (inputColumn index)
  witness := fun _ index => .var (witnessColumn index)
  output := fun _ index => .var (outputColumn index)

theorem inputColumn_value
    (index : Lifecycle.Stage1.Application.StateIndex) :
    inputColumn index = currentWordStart + index.val := by
  have bound := index.isLt
  simp only [Lifecycle.Stage1.Application.stateWordCount] at bound
  change PackageOrder.sourceToPackage (28909 + index.val) = 28909 + index.val
  unfold PackageOrder.sourceToPackage
  rw [ite_eq_left (by
    norm_num [PackageOrder.pilotSourceColumnCount]
    omega)]
  unfold PilotPackageOrder.sourceToPackage
  rw [ite_eq_left (by rw [PilotPackageOrder.priorPublicStart_value]; omega)]
  unfold PackageOrder.liftPilotColumn
  rw [ite_eq_left (by
    norm_num [PackageOrder.pilotInputPrivateColumnCount]
    omega)]

theorem outputColumn_value
    (index : Lifecycle.Stage1.Application.StateIndex) :
    outputColumn index = 57822 + index.val := by
  have bound := index.isLt
  simp only [Lifecycle.Stage1.Application.stateWordCount] at bound
  change PackageOrder.sourceToPackage (58142 + index.val) = 57822 + index.val
  unfold PackageOrder.sourceToPackage
  rw [ite_eq_left (by
    norm_num [PackageOrder.pilotSourceColumnCount]
    omega)]
  unfold PilotPackageOrder.sourceToPackage
  rw [ite_eq_right (by rw [PilotPackageOrder.priorPublicStart_value]; omega)]
  rw [ite_eq_right (by rw [PilotPackageOrder.outputPreimageStart_value]; omega)]
  rw [ite_eq_left (by rw [PilotPackageOrder.outputDigestStart_value]; omega)]
  rw [PilotPackageOrder.secondPrivateStart_value,
    PilotPackageOrder.outputPreimageStart_value]
  have difference : 58142 + index.val - 29233 = 28909 + index.val := by omega
  rw [difference]
  unfold PackageOrder.liftPilotColumn
  rw [ite_eq_left (by
    norm_num [PackageOrder.pilotInputPrivateColumnCount]
    omega)]
  omega

abbrev ExternalBelow (program : Lifecycle.Stage1.Application.Program) : Prop :=
  Lifecycle.Stage1.Application.InputsBelow
    (interface program) (localStart program)

theorem externalBelow (program : Lifecycle.Stage1.Application.Program) :
    ExternalBelow program := by
  have pilotBelow : PackageOrder.pilotPrivateColumnCount ≤ localStart program := by
    calc
      _ ≤ PackageOrder.constantColumn := PackageOrder.pilotPrivateColumnCount_le_constantColumn
      _ = PackageOrder.privateColumnCount := PackageOrder.constantColumn_eq_private
      _ ≤ localStart program := by
        unfold localStart witnessStart
        exact Nat.le_add_right _ _
  refine {
    input := fun index => ?_
    witness := fun index => ?_
    output := fun index => ?_ }
  · simp only [interface, Expr.VarsBelow]
    apply Nat.lt_of_lt_of_le ?_ pilotBelow
    rw [inputColumn_value]
    have bound := index.isLt
    norm_num [PackageOrder.pilotPrivateColumnCount,
      Lifecycle.Stage1.Application.stateWordCount, currentWordStart] at bound ⊢
    omega
  · simp only [interface, Expr.VarsBelow, witnessColumn]
    have bound := index.isLt
    unfold localStart
    omega
  · simp only [interface, Expr.VarsBelow]
    apply Nat.lt_of_lt_of_le ?_ pilotBelow
    rw [outputColumn_value]
    have bound := index.isLt
    norm_num [PackageOrder.pilotPrivateColumnCount,
      Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
    omega

end NightstreamFPrime.Layout.Stage1.ApplicationInputs
