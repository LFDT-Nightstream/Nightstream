import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.First54
import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLC

/-!
Owns all source-column, physical-row, and R1CS-fresh starts for the canonical
PiRLC Stage 1 packet.

The formulas follow the exact parent order. Export modules consume these
starts and do not restate layout constants.
-/

namespace NightstreamFPrime.Layout.Stage1.PiRLCStarts

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Completed PiCCS boundaries. -/
def phaseLogicalStart : Nat := PiRLCInputs.phaseOffset
def phaseRowStart : Nat := 18835765

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
def commitmentRowStart : Nat := samplerRowStart + 1008848
def publicInputRowStart : Nat := commitmentRowStart + 2495124
def evalKRowStart : Nat := publicInputRowStart + 693090
def evalARowStart : Nat := evalKRowStart + 277236
def outputRowStart : Nat := evalARowStart + 3881304

def samplerFreshStart : Nat := phaseFreshStart
def commitmentFreshStart : Nat := samplerFreshStart + 743631
def publicInputFreshStart : Nat := commitmentFreshStart + 2478600
def evalKFreshStart : Nat := publicInputFreshStart + 688500
def evalAFreshStart : Nat := evalKFreshStart + 275400
def outputFreshStart : Nat := evalAFreshStart + 3855600

/-- One scalar sampler owns 15,504 logical columns, 59,344 physical rows,
and 43,743 R1CS-fresh columns. -/
def samplerSourceLogicalStart (source : Nat) : Nat :=
  samplerLogicalStart + source * 15504

def samplerSourceRowStart (source : Nat) : Nat :=
  samplerRowStart + source * 59344

def samplerSourceFreshStart (source : Nat) : Nat :=
  samplerFreshStart + source * 43743

def entryLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source
def entryRowStart (source : Nat) : Nat := samplerSourceRowStart source

def windowLogicalStart (source round : Nat) : Nat :=
  samplerSourceLogicalStart source + 592 + round * 992

def windowRowStart (source round : Nat) : Nat :=
  samplerSourceRowStart source + 592 + round * 2216

def windowFreshStart (source round : Nat) : Nat :=
  samplerSourceFreshStart source + round * 1212

def digestLaneLogicalStart (source round lane : Nat) : Nat :=
  windowLogicalStart source round + lane * 100

def digestLaneRowStart (source round lane : Nat) : Nat :=
  windowRowStart source round + lane * 406

def digestLaneFreshStart (source round lane : Nat) : Nat :=
  windowFreshStart source round + lane * 303

def digestPermutationLogicalStart (source round : Nat) : Nat :=
  windowLogicalStart source round + 400

def digestPermutationRowStart (source round : Nat) : Nat :=
  windowRowStart source round + 1624

def selectorLogicalStart (source : Nat) : Nat :=
  samplerSourceLogicalStart source + 8528

def selectorRowStart (source : Nat) : Nat :=
  samplerSourceRowStart source + 18320

def selectorFreshStart (source : Nat) : Nat :=
  samplerSourceFreshStart source + 9696

/-- The final First54 value row owns 54 contiguous raw words. The centered
challenge is each word minus two. -/
def challengeWordStart (source : Nat) : Nat :=
  selectorLogicalStart source + 6922

theorem challengeWordStart_eq (source : Nat) :
    challengeWordStart source = phaseLogicalStart + source * 15504 + 15450 := by
  unfold challengeWordStart selectorLogicalStart samplerSourceLogicalStart
    samplerLogicalStart
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
  omega

theorem phaseLogicalStart_eq : phaseLogicalStart = 18956449 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 19268671 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 20012302 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 22490902 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 23179402 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 23454802 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [18956449, 19220017, 19236541, 19241131, 19242967, 19268671] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [18835765, 19844613, 22339737, 23032827, 23310063, 27191367] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [19268671, 20012302, 22490902, 23179402, 23454802, 27310402] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 27191367 ∧ outputFreshStart = 27310402 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiRLCStarts
