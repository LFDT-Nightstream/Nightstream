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
def phaseRowStart : Nat := 19936967

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

/-- One scalar owns 3,259 logical columns, 3,467 physical rows,
and 1,548 R1CS lowering columns. -/
def samplerSourceLogicalStart (source : Nat) : Nat :=
  SamplerChain.sourceOffset samplerLogicalStart source

def samplerSourceRowStart (source : Nat) : Nat := samplerRowStart + source * 3467

def samplerSourceFreshStart (source : Nat) : Nat := samplerFreshStart + source * 1548

def entryLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source

def entryRowStart (source : Nat) : Nat := samplerSourceRowStart source

def rangeLogicalStart (source : Nat) : Nat := Sampler.rangeOffset (samplerSourceLogicalStart source)

def rangeRowStart (source : Nat) : Nat := samplerSourceRowStart source + 592

def rangeFreshStart (source : Nat) : Nat := samplerSourceFreshStart source

def advanceLogicalStart (source : Nat) : Nat := Sampler.advanceOffset (samplerSourceLogicalStart source)

def advanceRowStart (source : Nat) : Nat := rangeRowStart source + 2229

/-- The sampler checks each digit word before the combination circuit reads it. -/
def challengeWordStart (source : Nat) : Nat := Sampler.wordsOffset (samplerSourceLogicalStart source)

def challengeWordRowStart (source : Nat) : Nat := advanceRowStart source + 592

theorem challengeWordStart_eq (source : Nat) :
    challengeWordStart source = phaseLogicalStart + source * 3259 + 3205 := by
  unfold challengeWordStart samplerSourceLogicalStart samplerLogicalStart
    Formal.samplerOffset SamplerChain.sourceOffset Sampler.wordsOffset Sampler.advanceOffset Sampler.rangeOffset
  rw [Sampler.counts.1, Gadgets.Sampling.WideReduction.Program.privateCount_eq]

theorem phaseLogicalStart_eq : phaseLogicalStart = 20064823 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 20172552 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 20198868 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 23228268 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 23916768 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 24192168 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [20064823, 20120226, 20140422, 20145012, 20146848, 20172552] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [19936967, 19995906, 23045502, 23738592, 24015828, 27897132] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [20172552, 20198868, 23228268, 23916768, 24192168, 28047768] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 27897132 ∧ outputFreshStart = 28047768 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiRLCStarts
