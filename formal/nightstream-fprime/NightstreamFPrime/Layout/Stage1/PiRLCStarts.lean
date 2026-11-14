import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PilotPiCCS

namespace NightstreamFPrime.Layout.Stage1.PiRLCStarts

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def phaseLogicalStart : Nat := PiRLCInputs.phaseOffset
def phaseRowStart : Nat := 6522569

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

theorem phaseLogicalStart_eq : phaseLogicalStart = 6592681 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 6711426 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 6711426 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 9257346 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 9964546 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 10247426 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [6592681, 6675522, 6695106, 6700546, 6702722, 6711426] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [6522569, 6575286, 9140790, 9853430, 10138486, 11278710] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [6711426, 6711426, 9257346, 9964546, 10247426, 11378946] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 11278710 ∧ outputFreshStart = 11378946 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiRLCStarts
