import NightstreamFPrime.Layout.Stage1.Spartan
import NightstreamFPrime.Lifecycle.Stage1.Application

/-!
Owns the zero-copy Stage 1 application column map.

The input state is the four-word `current` block of the prior state preimage.
The output state is the same block of the next state preimage. Application
witness words occupy the first new private suffix. No copy row is added.

This module does not select a concrete application or change the prefix
Spartan layout.
-/

namespace NightstreamFPrime.Layout.Stage1.ApplicationInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout

/-- In `serializePreimage`, the current-state block starts after the domain,
key block, iteration, and initial-state block. -/
def currentWordStart : Nat := 35

def inputSourceColumn
    (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  PilotProduction.priorPreimageStart + currentWordStart + index.val

def outputSourceColumn
    (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  PilotProduction.outputPreimageStart + currentWordStart + index.val

def inputColumn (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  Spartan.sourceToSpartan (inputSourceColumn index)

def outputColumn (index : Lifecycle.Stage1.Application.StateIndex) : Nat :=
  Spartan.sourceToSpartan (outputSourceColumn index)

/-- New caller-owned application witness words begin at the current private
endpoint. The final per-application layout moves the constant and public suffix
after this complete new private interval. -/
def witnessStart : Nat := Spartan.privateColumnCount

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
  change Spartan.sourceToSpartan (35 + index.val) = 35 + index.val
  unfold Spartan.sourceToSpartan
  rw [if_pos (by
    norm_num [Spartan.pilotSourceColumnCount]
    omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_pos (by rw [PilotSpartan.priorPublicStart_value]; omega)]
  unfold Spartan.liftPilotColumn
  rw [if_pos (by
    norm_num [Spartan.pilotInputPrivateColumnCount]
    omega)]

theorem outputColumn_value
    (index : Lifecycle.Stage1.Application.StateIndex) :
    outputColumn index = 49428 + index.val := by
  have bound := index.isLt
  simp only [Lifecycle.Stage1.Application.stateWordCount] at bound
  change Spartan.sourceToSpartan (49698 + index.val) = 49428 + index.val
  unfold Spartan.sourceToSpartan
  rw [if_pos (by
    norm_num [Spartan.pilotSourceColumnCount]
    omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by rw [PilotSpartan.priorPublicStart_value]; omega)]
  rw [if_neg (by rw [PilotSpartan.outputPreimageStart_value]; omega)]
  rw [if_pos (by rw [PilotSpartan.outputDigestStart_value]; omega)]
  rw [PilotSpartan.secondPrivateStart_value,
    PilotSpartan.outputPreimageStart_value]
  have difference : 49698 + index.val - 49663 = 35 + index.val := by omega
  rw [difference]
  unfold Spartan.liftPilotColumn
  rw [if_pos (by
    norm_num [Spartan.pilotInputPrivateColumnCount]
    omega)]
  omega

abbrev ExternalBelow (program : Lifecycle.Stage1.Application.Program) : Prop :=
  Lifecycle.Stage1.Application.InputsBelow
    (interface program) (localStart program)

theorem externalBelow (program : Lifecycle.Stage1.Application.Program) :
    ExternalBelow program := by
  refine {
    input := fun index => ?_
    witness := fun index => ?_
    output := fun index => ?_ }
  · simp only [interface, Expr.VarsBelow]
    rw [inputColumn_value]
    have bound := index.isLt
    norm_num [localStart, witnessStart, Spartan.privateColumnCount,
      Lifecycle.Stage1.Application.stateWordCount, currentWordStart] at bound ⊢
    omega
  · simp only [interface, Expr.VarsBelow, witnessColumn]
    have bound := index.isLt
    unfold localStart
    omega
  · simp only [interface, Expr.VarsBelow]
    rw [outputColumn_value]
    have bound := index.isLt
    norm_num [localStart, witnessStart, Spartan.privateColumnCount,
      Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
    omega

end NightstreamFPrime.Layout.Stage1.ApplicationInputs
