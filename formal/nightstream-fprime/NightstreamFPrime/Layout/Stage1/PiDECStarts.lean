import NightstreamFPrime.Layout.Stage1.PiDECInputs

/-!
Owns all constrained-input, logical, physical-row, and R1CS-fresh starts for
the canonical PiDEC Stage 1 packet. Export modules consume these definitions
and do not restate layout constants.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

open NightstreamFPrime.Lifecycle.PiDEC.v1_1

def phaseLogicalStart : Nat := PiDECInputs.phaseOffset
def phaseRowStart : Nat := 27237625
def phaseFreshStart : Nat := phaseLogicalStart + Formal.logicalPrivateCount

def inputLogicalStart : Nat := Formal.inputBindingOffset phaseLogicalStart
def publicInputLogicalStart : Nat := Formal.publicInputOffset phaseLogicalStart
def commitmentLogicalStart : Nat := Formal.commitmentOffset phaseLogicalStart
def evalKLogicalStart : Nat := Formal.evalKOffset phaseLogicalStart
def evalALogicalStart : Nat := Formal.evalAOffset phaseLogicalStart
def outputLogicalStart : Nat := Formal.outputBindingOffset phaseLogicalStart

def inputRowStart : Nat := phaseRowStart
def publicInputRowStart : Nat := inputRowStart
def commitmentRowStart : Nat := publicInputRowStart + 22680
def evalKRowStart : Nat := commitmentRowStart + 972
def evalARowStart : Nat := evalKRowStart + 108
def outputRowStart : Nat := evalARowStart + 1512

def inputFreshStart : Nat := phaseFreshStart
def publicInputFreshStart : Nat := inputFreshStart
def commitmentFreshStart : Nat := publicInputFreshStart + 17820
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
      [27402496, 27237625, 27402766] := by
  rfl

theorem childLogicalStarts_eq :
    [inputLogicalStart, publicInputLogicalStart, commitmentLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [27402496, 27402496, 27402766, 27402766, 27402766, 27402766] := by
  rfl

theorem childRowStarts_eq :
    [inputRowStart, publicInputRowStart, commitmentRowStart, evalKRowStart,
      evalARowStart, outputRowStart] =
    [27237625, 27237625, 27260305, 27261277, 27261385, 27262897] := by
  rfl

theorem childFreshStarts_eq :
    [inputFreshStart, publicInputFreshStart, commitmentFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [27402766, 27402766, 27420586, 27420586, 27420586, 27420586] := by
  rfl

theorem scalarStarts_eq (source : Nat) :
    scalarLogicalStart source = 27402496 + source ∧
      scalarRowStart source = 27237625 + source * 84 ∧
      scalarFreshStart source = 27402766 + source * 66 := by
  exact ⟨rfl, rfl, rfl⟩

theorem finalBoundaries_eq :
    outputRowStart = 27262897 ∧ outputFreshStart = 27420586 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiDECStarts
