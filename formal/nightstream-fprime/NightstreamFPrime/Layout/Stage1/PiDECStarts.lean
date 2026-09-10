import NightstreamFPrime.Layout.Stage1.PiDECInputs

/-!
Owns all constrained-input, logical, physical-row, and R1CS-fresh starts for
the canonical PiDEC Stage 1 packet. Export modules consume these definitions
and do not restate layout constants.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

open NightstreamFPrime.Lifecycle.PiDEC.v1_1

def phaseLogicalStart : Nat := PiDECInputs.phaseOffset
def phaseRowStart : Nat := PiRLCStarts.outputRowStart
def phaseFreshStart : Nat := phaseLogicalStart + Formal.logicalPrivateCount

def inputLogicalStart : Nat := Formal.inputBindingOffset phaseLogicalStart
def publicInputLogicalStart : Nat := Formal.publicInputOffset phaseLogicalStart
def commitmentLogicalStart : Nat := Formal.commitmentOffset phaseLogicalStart
def evalKLogicalStart : Nat := Formal.evalKOffset phaseLogicalStart
def evalALogicalStart : Nat := Formal.evalAOffset phaseLogicalStart
def outputLogicalStart : Nat := Formal.outputBindingOffset phaseLogicalStart

def inputRowStart : Nat := phaseRowStart
def publicInputRowStart : Nat := inputRowStart
def commitmentRowStart : Nat := publicInputRowStart + PiDEC.v1_1.PublicInputSplit.physicalRowCount
def evalKRowStart : Nat := commitmentRowStart + PiDEC.v1_1.CommitmentRecomposition.physicalRowCount
def evalARowStart : Nat := evalKRowStart + PiDEC.v1_1.EvalKRecomposition.physicalRowCount
def outputRowStart : Nat := evalARowStart + PiDEC.v1_1.EvalARecomposition.physicalRowCount

def inputFreshStart : Nat := phaseFreshStart
def publicInputFreshStart : Nat := inputFreshStart
def commitmentFreshStart : Nat := publicInputFreshStart + PiDEC.v1_1.PublicInputSplit.freshColumnCount
def evalKFreshStart : Nat := commitmentFreshStart
def evalAFreshStart : Nat := evalKFreshStart
def outputFreshStart : Nat := evalAFreshStart

def scalarLogicalStart (source : Nat) : Nat :=
  PublicInputSplit.sourceOffset publicInputLogicalStart source

def scalarRowStart (source : Nat) : Nat :=
  publicInputRowStart + source * PiDEC.v1_1.Leaves.SignedSplitScalar.physicalRowCount

def scalarFreshStart (source : Nat) : Nat :=
  publicInputFreshStart + source * PiDEC.v1_1.Leaves.SignedSplitScalar.freshColumnCount

def signRowStart (source : Nat) : Nat := scalarRowStart source
def signFreshStart (source : Nat) : Nat := scalarFreshStart source

def digitRowStart (source child : Nat) : Nat :=
  scalarRowStart source + PiDEC.v1_1.Leaves.SignedSplitScalar.signRowCount +
    child * PiDEC.v1_1.Leaves.SignedSplitScalar.digitRowCount

def digitFreshStart (source child : Nat) : Nat :=
  scalarFreshStart source + PiDEC.v1_1.Leaves.SignedSplitScalar.signFreshCount +
    child * PiDEC.v1_1.Leaves.SignedSplitScalar.digitFreshCount

def recompositionRowStart (source : Nat) : Nat :=
  scalarRowStart source + PiDEC.v1_1.Leaves.SignedSplitScalar.signRowCount +
    PiDECInputs.childCount *
      PiDEC.v1_1.Leaves.SignedSplitScalar.digitRowCount


end NightstreamFPrime.Layout.Stage1.PiDECStarts
