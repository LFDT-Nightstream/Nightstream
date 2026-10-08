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
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Completed PiCCS boundaries. -/
def phaseLogicalStart : Nat := PiRLCInputs.phaseOffset
def phaseRowStart : Nat := 7132243

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
def commitmentRowStart : Nat := samplerRowStart + 52207
def publicInputRowStart : Nat := commitmentRowStart + 3049596
def evalKRowStart : Nat := publicInputRowStart + 693090
def evalARowStart : Nat := evalKRowStart + 277236
def outputRowStart : Nat := evalARowStart + 1108944

def samplerFreshStart : Nat := phaseFreshStart
def commitmentFreshStart : Nat := samplerFreshStart + 2448
def publicInputFreshStart : Nat := commitmentFreshStart + 3029400
def evalKFreshStart : Nat := publicInputFreshStart + 688500
def evalAFreshStart : Nat := evalKFreshStart + 275400
def outputFreshStart : Nat := evalAFreshStart + 1101600

/-- One scalar owns 4,267 logical columns, 3,071 physical rows,
and 144 R1CS lowering columns. -/
def samplerSourceLogicalStart (source : Nat) : Nat :=
  SamplerChain.sourceOffset samplerLogicalStart source

def samplerSourceRowStart (source : Nat) : Nat := samplerRowStart + source * 3071

def samplerSourceFreshStart (source : Nat) : Nat := samplerFreshStart + source * 144

def entryLogicalStart (source : Nat) : Nat := samplerSourceLogicalStart source

def entryRowStart (source : Nat) : Nat := samplerSourceRowStart source

def rangeLogicalStart (source : Nat) : Nat := Sampler.rangeOffset (samplerSourceLogicalStart source)

def rangeRowStart (source : Nat) : Nat := samplerSourceRowStart source + 1096

def rangeFreshStart (source : Nat) : Nat := samplerSourceFreshStart source

def advanceLogicalStart (source : Nat) : Nat := Sampler.advanceOffset (samplerSourceLogicalStart source)

def advanceRowStart (source : Nat) : Nat := rangeRowStart source + 825

/-- The sampler checks each digit word before the combination circuit reads it. -/
def challengeWordStart (source : Nat) : Nat := Sampler.wordsOffset (samplerSourceLogicalStart source)

def challengeWordRowStart (source : Nat) : Nat := advanceRowStart source + 1096

theorem challengeWordStart_eq (source : Nat) :
    challengeWordStart source = phaseLogicalStart + source * 4267 + 4213 := by
  unfold challengeWordStart samplerSourceLogicalStart samplerLogicalStart
    Formal.samplerOffset SamplerChain.sourceOffset Sampler.wordsOffset Sampler.advanceOffset Sampler.rangeOffset
  rw [Sampler.counts.1, Gadgets.Sampling.WideReduction.Program.privateCount_eq]

theorem phaseLogicalStart_eq : phaseLogicalStart = 7207123 := by
  rfl

theorem phaseRowStart_matches
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseRowStart = PilotPiCCS.physicalRowCount relation := by
  rw [PilotPiCCS.physicalRowCount_eq]
  rfl

theorem phaseFreshStart_eq : phaseFreshStart = 7313628 := by
  rfl

theorem commitmentFreshStart_eq : commitmentFreshStart = 7316076 := by
  rfl

theorem publicInputFreshStart_eq : publicInputFreshStart = 10345476 := by
  rfl

theorem evalKFreshStart_eq : evalKFreshStart = 11033976 := by
  rfl

theorem evalAFreshStart_eq : evalAFreshStart = 11309376 := by
  rfl

theorem childLogicalStarts_eq :
    [samplerLogicalStart, commitmentLogicalStart, publicInputLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [7207123, 7279662, 7299858, 7304448, 7306284, 7313628] := by
  rfl

theorem childRowStarts_eq :
    [samplerRowStart, commitmentRowStart, publicInputRowStart,
      evalKRowStart, evalARowStart, outputRowStart] =
    [7132243, 7184450, 10234046, 10927136, 11204372, 12313316] := by
  rfl

theorem childFreshStarts_eq :
    [samplerFreshStart, commitmentFreshStart, publicInputFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [7313628, 7316076, 10345476, 11033976, 11309376, 12410976] := by
  rfl

theorem finalBoundaries_eq :
    outputRowStart = 12313316 ∧ outputFreshStart = 12410976 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiRLCStarts
