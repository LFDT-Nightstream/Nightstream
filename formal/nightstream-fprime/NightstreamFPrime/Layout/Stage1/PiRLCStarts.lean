import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PilotPiCCS

namespace NightstreamFPrime.Layout.Stage1.PiRLCStarts

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def phaseLogicalStart : Nat := PiRLCInputs.phaseOffset
def phaseRowStart : Nat := 6524761

def phaseFreshStart : Nat :=
  phaseLogicalStart + Formal.logicalPrivateCount

def samplerLogicalStart : Nat := Formal.samplerOffset phaseLogicalStart
def commitmentLogicalStart : Nat := Formal.commitmentOffset phaseLogicalStart
def publicInputLogicalStart : Nat := Formal.publicInputOffset phaseLogicalStart
def evalKLogicalStart : Nat := Formal.evalKOffset phaseLogicalStart
def evalALogicalStart : Nat := Formal.evalAOffset phaseLogicalStart
def outputLogicalStart : Nat := Formal.outputBindingOffset phaseLogicalStart

def samplerRowStart : Nat := phaseRowStart
def commitmentRowStart : Nat := samplerRowStart + 52717
def publicInputRowStart : Nat := commitmentRowStart + 2565504
def evalKRowStart : Nat := publicInputRowStart + 712640
def evalARowStart : Nat := evalKRowStart + 285056
def outputRowStart : Nat := evalARowStart + 1140224

def samplerFreshStart : Nat := phaseFreshStart
def commitmentFreshStart : Nat := samplerFreshStart
def publicInputFreshStart : Nat := commitmentFreshStart + 2545920
def evalKFreshStart : Nat := publicInputFreshStart + 707200
def evalAFreshStart : Nat := evalKFreshStart + 282880
def outputFreshStart : Nat := evalAFreshStart + 1131520

def samplerSourceLogicalStart (source : Nat) : Nat :=
  SamplerChain.sourceOffset samplerLogicalStart source

def samplerSourceRowStart (source : Nat) : Nat := samplerRowStart + source * 3101

def samplerSourceFreshStart (_source : Nat) : Nat := samplerFreshStart

def entryLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source

def entryRowStart (source : Nat) : Nat := samplerSourceRowStart source

def rangeLogicalStart (source : Nat) : Nat := Sampler.rangeOffset (samplerSourceLogicalStart source)

def rangeRowStart (source : Nat) : Nat := samplerSourceRowStart source + 1096

def rangeFreshStart (source : Nat) : Nat := samplerSourceFreshStart source

def advanceLogicalStart (source : Nat) : Nat := Sampler.advanceOffset (samplerSourceLogicalStart source)

def advanceRowStart (source : Nat) : Nat := rangeRowStart source + 845

def challengeWordStart (source : Nat) : Nat := Sampler.wordsOffset (samplerSourceLogicalStart source)

def challengeWordRowStart (source : Nat) : Nat := advanceRowStart source + 1096

theorem challengeWordStart_eq (source : Nat) :
    challengeWordStart source = phaseLogicalStart + source * 4873 + 4809 := by
  unfold challengeWordStart samplerSourceLogicalStart samplerLogicalStart
    Formal.samplerOffset SamplerChain.sourceOffset Sampler.wordsOffset Sampler.advanceOffset Sampler.rangeOffset
  rw [Sampler.counts.1, Gadgets.Sampling.WideReduction.Program.privateCount_eq]

theorem phaseLogicalStart_eq : phaseLogicalStart = 6594873 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 6713618 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 6713618 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 9259538 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 9966738 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 10249618 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [6594873, 6677714, 6697298, 6702738, 6704914, 6713618] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [6524761, 6577478, 9142982, 9855622, 10140678, 11280902] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [6713618, 6713618, 9259538, 9966738, 10249618, 11381138] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 11280902 ∧ outputFreshStart = 11381138 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiRLCStarts
