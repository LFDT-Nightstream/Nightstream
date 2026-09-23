import NightstreamFPrime.Layout.Stage1.Wide.PilotPiCCSPiRLC

/-!
Owns all source-column, physical-row, and R1CS-fresh starts for the canonical
PiRLC Stage 1 packet.

The formulas follow the exact parent order. Export modules consume these
starts and do not restate layout constants.
-/

namespace NightstreamFPrime.Layout.Stage1.Wide.PiRLCStarts

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.Wide
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Completed PiCCS boundaries. -/
def phaseLogicalStart : Nat := PiRLCInputs.phaseOffset
def phaseRowStart : Nat := 19385261

/-- The phase lowering starts after all seven logical child intervals. -/
def phaseFreshStart : Nat :=
  phaseLogicalStart + Formal.logicalPrivateCount

def samplerLogicalStart : Nat := Formal.samplerOffset phaseLogicalStart
def commitmentLogicalStart : Nat := Formal.commitmentOffset phaseLogicalStart
def publicInputLogicalStart : Nat := Formal.publicInputOffset phaseLogicalStart
def evalKLogicalStart : Nat := Formal.evalKOffset phaseLogicalStart
def evalALogicalStart : Nat := Formal.evalAOffset phaseLogicalStart
def outputLogicalStart : Nat := Formal.outputBindingOffset phaseLogicalStart

def samplerRowStart : Nat := phaseRowStart
def commitmentRowStart : Nat := samplerRowStart + 58939
def publicInputRowStart : Nat := commitmentRowStart + 3049596
def evalKRowStart : Nat := publicInputRowStart + 693090
def evalARowStart : Nat := evalKRowStart + 277236
def outputRowStart : Nat := evalARowStart + 3881304

def samplerFreshStart : Nat := phaseFreshStart
def commitmentFreshStart : Nat := samplerFreshStart + 26316
def publicInputFreshStart : Nat := commitmentFreshStart + 3029400
def evalKFreshStart : Nat := publicInputFreshStart + 688500
def evalAFreshStart : Nat := evalKFreshStart + 275400
def outputFreshStart : Nat := evalAFreshStart + 3855600

/-- One scalar has two Poseidon permutations and one executable range gadget. -/
def samplerSourceLogicalStart (source : Nat) : Nat :=
  samplerLogicalStart + source * 3205

def samplerSourceRowStart (source : Nat) : Nat :=
  samplerRowStart + source * 3413

def samplerSourceFreshStart (source : Nat) : Nat :=
  samplerFreshStart + source * 1548

def entryLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source
def entryRowStart (source : Nat) : Nat := samplerSourceRowStart source

def rangeLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source + 592
def rangeRowStart (source : Nat) : Nat := samplerSourceRowStart source + 592

def advanceLogicalStart (source : Nat) : Nat := rangeLogicalStart source + 2021
def advanceRowStart (source : Nat) : Nat := rangeRowStart source + 2229

/-- Temporary digit words preserve the existing ring-product recipe shape. -/
def challengeWordStart (source : Nat) : Nat :=
  samplerLogicalStart + 54485 + source * 54

theorem challengeWordStart_eq (source : Nat) :
    challengeWordStart source = phaseLogicalStart + 54485 + source * 54 := by
  rfl

theorem phaseLogicalStart_eq : phaseLogicalStart = 19513117 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 19620846 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 19647162 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 22676562 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 23365062 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 23640462 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [19513117, 19568520, 19588716, 19593306, 19595142, 19620846] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [19385261, 19444200, 22493796, 23186886, 23464122, 27345426] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [19620846, 19647162, 22676562, 23365062, 23640462, 27496062] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 27345426 ∧ outputFreshStart = 27496062 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.Wide.PiRLCStarts
