import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PilotPiCCS

namespace NightstreamFPrime.Layout.Stage1.PiRLCStarts

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def phaseLogicalStart : Nat := PiRLCInputs.phaseOffset
def phaseRowStart : Nat := 6117773

def phaseFreshStart : Nat :=
  phaseLogicalStart + Formal.logicalPrivateCount

def samplerLogicalStart : Nat := Formal.samplerOffset phaseLogicalStart
def commitmentLogicalStart : Nat := Formal.commitmentOffset phaseLogicalStart
def publicInputLogicalStart : Nat := Formal.publicInputOffset phaseLogicalStart
def evalKLogicalStart : Nat := Formal.evalKOffset phaseLogicalStart
def evalALogicalStart : Nat := Formal.evalAOffset phaseLogicalStart
def outputLogicalStart : Nat := Formal.outputBindingOffset phaseLogicalStart

def samplerRowStart : Nat := phaseRowStart
def commitmentRowStart : Nat := samplerRowStart + 48855
def publicInputRowStart : Nat := commitmentRowStart + 2263680
def evalKRowStart : Nat := publicInputRowStart + 628800
def evalARowStart : Nat := evalKRowStart + 251520
def outputRowStart : Nat := evalARowStart + 1006080

def samplerFreshStart : Nat := phaseFreshStart
def commitmentFreshStart : Nat := samplerFreshStart
def publicInputFreshStart : Nat := commitmentFreshStart + 2246400
def evalKFreshStart : Nat := publicInputFreshStart + 624000
def evalAFreshStart : Nat := evalKFreshStart + 249600
def outputFreshStart : Nat := evalAFreshStart + 998400

def samplerSourceLogicalStart (source : Nat) : Nat :=
  SamplerChain.sourceOffset samplerLogicalStart source

def samplerSourceRowStart (source : Nat) : Nat := samplerRowStart + source * 3257

def samplerSourceFreshStart (_source : Nat) : Nat := samplerFreshStart

def entryLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source

def entryRowStart (source : Nat) : Nat := samplerSourceRowStart source

def rangeLogicalStart (source : Nat) : Nat := Sampler.rangeOffset (samplerSourceLogicalStart source)

def rangeRowStart (source : Nat) : Nat := samplerSourceRowStart source + 1174

def rangeFreshStart (source : Nat) : Nat := samplerSourceFreshStart source

def advanceLogicalStart (source : Nat) : Nat := Sampler.advanceOffset (samplerSourceLogicalStart source)

def advanceRowStart (source : Nat) : Nat := rangeRowStart source + 845

def challengeWordStart (source : Nat) : Nat := Sampler.wordsOffset (samplerSourceLogicalStart source)

def challengeWordRowStart (source : Nat) : Nat := advanceRowStart source + 1174

theorem challengeWordStart_eq (source : Nat) :
    challengeWordStart source = phaseLogicalStart + source * 5029 + 4965 := by
  unfold challengeWordStart samplerSourceLogicalStart samplerLogicalStart
    Formal.samplerOffset SamplerChain.sourceOffset Sampler.wordsOffset Sampler.advanceOffset Sampler.rangeOffset
  rw [Sampler.counts.1, Gadgets.Sampling.WideReduction.Program.privateCount_eq]

theorem phaseLogicalStart_eq : phaseLogicalStart = 6179213 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 6286328 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 6286328 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 8532728 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 9156728 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 9406328 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [6179213, 6254648, 6271928, 6276728, 6278648, 6286328] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [6117773, 6166628, 8430308, 9059108, 9310628, 10316708] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [6286328, 6286328, 8532728, 9156728, 9406328, 10404728] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 10316708 ∧ outputFreshStart = 10404728 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiRLCStarts
