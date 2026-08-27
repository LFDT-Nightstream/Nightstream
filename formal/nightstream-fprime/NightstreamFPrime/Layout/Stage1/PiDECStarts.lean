import NightstreamFPrime.Layout.Stage1.PiDECInputs

/-!
Owns all constrained-input, logical, physical-row, and R1CS-fresh starts for
the canonical PiDEC Stage 1 packet. Export modules consume these definitions
and do not restate layout constants.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

open NightstreamFPrime.Lifecycle.PiDEC.v1_1

def phaseLogicalStart : Nat := PiDECInputs.phaseOffset
def phaseRowStart : Nat := 25556958
def phaseFreshStart : Nat := phaseLogicalStart + Formal.logicalPrivateCount

def inputLogicalStart : Nat := Formal.inputBindingOffset phaseLogicalStart
def publicInputLogicalStart : Nat := Formal.publicInputOffset phaseLogicalStart
def commitmentLogicalStart : Nat := Formal.commitmentOffset phaseLogicalStart
def evalKLogicalStart : Nat := Formal.evalKOffset phaseLogicalStart
def evalALogicalStart : Nat := Formal.evalAOffset phaseLogicalStart
def outputLogicalStart : Nat := Formal.outputBindingOffset phaseLogicalStart

def inputRowStart : Nat := phaseRowStart
def publicInputRowStart : Nat := inputRowStart
def commitmentRowStart : Nat := publicInputRowStart + 4536
def evalKRowStart : Nat := commitmentRowStart + 972
def evalARowStart : Nat := evalKRowStart + 108
def outputRowStart : Nat := evalARowStart + 1512

def inputFreshStart : Nat := phaseFreshStart
def publicInputFreshStart : Nat := inputFreshStart
def commitmentFreshStart : Nat := publicInputFreshStart + 3564
def evalKFreshStart : Nat := commitmentFreshStart
def evalAFreshStart : Nat := evalKFreshStart
def outputFreshStart : Nat := evalAFreshStart

def scalarLogicalStart (source : Nat) : Nat :=
  publicInputLogicalStart + source

def scalarRowStart (source : Nat) : Nat :=
  publicInputRowStart + source * 84

def scalarFreshStart (source : Nat) : Nat :=
  publicInputFreshStart + source * 66

def signRowStart (source : Nat) : Nat := scalarRowStart source
def signFreshStart (source : Nat) : Nat := scalarFreshStart source

def digitRowStart (source child : Nat) : Nat :=
  scalarRowStart source + 3 + child * 5

def digitFreshStart (source child : Nat) : Nat :=
  scalarFreshStart source + 2 + child * 4

def recompositionRowStart (source : Nat) : Nat :=
  scalarRowStart source + 83

theorem phaseStarts_eq :
    [phaseLogicalStart, phaseRowStart, phaseFreshStart] =
      [25711399, 25556958, 25711453] := by
  rfl

theorem childLogicalStarts_eq :
    [inputLogicalStart, publicInputLogicalStart, commitmentLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [25711399, 25711399, 25711453, 25711453, 25711453, 25711453] := by
  rfl

theorem childRowStarts_eq :
    [inputRowStart, publicInputRowStart, commitmentRowStart, evalKRowStart,
      evalARowStart, outputRowStart] =
    [25556958, 25556958, 25561494, 25562466, 25562574, 25564086] := by
  rfl

theorem childFreshStarts_eq :
    [inputFreshStart, publicInputFreshStart, commitmentFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [25711453, 25711453, 25715017, 25715017, 25715017, 25715017] := by
  rfl

theorem scalarStarts_eq (source : Nat) :
    scalarLogicalStart source = 25711399 + source ∧
      scalarRowStart source = 25556958 + source * 84 ∧
      scalarFreshStart source = 25711453 + source * 66 := by
  exact ⟨rfl, rfl, rfl⟩

theorem finalBoundaries_eq :
    outputRowStart = 25564086 ∧ outputFreshStart = 25715017 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiDECStarts
