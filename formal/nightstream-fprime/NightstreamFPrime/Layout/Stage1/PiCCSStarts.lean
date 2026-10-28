import NightstreamFPrime.Layout.Stage1.PilotPiCCS
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.FormalRows

/-! Paper authority: SuperNeo v1.2, section 7.3, PiCCS Steps 1--5. -/

namespace NightstreamFPrime.Layout.Stage1.PiCCSStarts

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def prefixStarts : Nat → List Nat → List Nat
  | _, [] => []
  | base, delta :: deltas => base :: prefixStarts (base + delta) deltas

def rowBase : Nat := 5086126

theorem rowBase_eq_layout :
    rowBase = Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset := by
  rw [rowBase, PilotProduction.physicalRowCount_eq]

def statementBindingRowStart : Nat := rowBase
def statementAbsorptionRowStart : Nat := statementBindingRowStart + 4712
def challengeRowStart : Nat := statementAbsorptionRowStart + 134808
def roundTranscriptRowStart : Nat := challengeRowStart + 4384
def initialClaimRowStart : Nat := roundTranscriptRowStart + 61376
def sumcheckRowStart : Nat := initialClaimRowStart + 12957
def evalKRowStart : Nat := sumcheckRowStart + 728
def evalARowStart : Nat := evalKRowStart + 3364
def ccsRowStart : Nat := evalARowStart + 11140
def normRowStart : Nat := ccsRowStart + 23
def finalIdentityRowStart : Nat := normRowStart + 800
def outputBindingRowStart : Nat := finalIdentityRowStart + 3593

def rowStarts : List Nat :=
  [statementBindingRowStart, statementAbsorptionRowStart, challengeRowStart,
    roundTranscriptRowStart, initialClaimRowStart, sumcheckRowStart,
    evalKRowStart, evalARowStart, ccsRowStart, normRowStart,
    finalIdentityRowStart, outputBindingRowStart]

def statementBindingLogicalStart : Nat := PiCCSInputs.phaseOffset
def statementWitnessStart : Nat := statementBindingLogicalStart + 270
def challengeWitnessStart : Nat := statementWitnessStart + 134808
def roundTranscriptWitnessStart : Nat := challengeWitnessStart + 4384
def initialClaimLogicalStart : Nat := roundTranscriptWitnessStart + 61376
def sumcheckLogicalStart : Nat := initialClaimLogicalStart + 12957
def evalKLogicalStart : Nat := sumcheckLogicalStart + 672
def evalALogicalStart : Nat := evalKLogicalStart + 2699
def ccsLogicalStart : Nat := evalALogicalStart + 10475
def normLogicalStart : Nat := ccsLogicalStart + 23
def finalIdentityLogicalStart : Nat := normLogicalStart + 48
def outputBindingWitnessStart : Nat := finalIdentityLogicalStart + 2717

theorem phaseOffset_le_statementWitnessStart :
    PiCCSInputs.phaseOffset ≤ statementWitnessStart :=
  Nat.le_add_right _ _

theorem statementWitnessStart_eq : statementWitnessStart = 5157226 := by
  unfold statementWitnessStart statementBindingLogicalStart
  rw [PiCCSInputs.phaseOffset_eq]

theorem challengeWitnessStart_eq : challengeWitnessStart = 5292034 := by
  unfold challengeWitnessStart
  rw [statementWitnessStart_eq]

theorem roundTranscriptWitnessStart_eq :
    roundTranscriptWitnessStart = 5296418 := by
  unfold roundTranscriptWitnessStart
  rw [challengeWitnessStart_eq]

theorem outputBindingWitnessStart_eq :
    outputBindingWitnessStart = 5387385 := by
  unfold outputBindingWitnessStart finalIdentityLogicalStart
    normLogicalStart ccsLogicalStart evalALogicalStart evalKLogicalStart
    sumcheckLogicalStart initialClaimLogicalStart
  rw [roundTranscriptWitnessStart_eq]

theorem outputBindingWitnessStart_matches
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    outputBindingWitnessStart =
      Formal.outputBindingOffset relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset := by
  unfold Formal.outputBindingOffset Formal.nextOffset Formal.childLength
  erw [Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation,
    Formal.finalIdentityCircuit,
    NightstreamFPrime.Circuit.FormalCircuit.withConstantFootprint_main,
    FinalIdentity.localLength_eq]
  rfl

def logicalFreshBase : Nat := PiCCSInputs.phaseOffset + 1068869

theorem logicalFreshBase_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    logicalFreshBase =
      NightstreamFPrime.Layout.PiCCS.v1_2.logicalColumnCount relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset := by
  rw [NightstreamFPrime.Layout.PiCCS.v1_2.logicalColumnCount_eq_of_degreeBound_eq_eight
    relation (PiCCSInputs.interface logicalWidth publicFits)
      PiCCSInputs.phaseOffset rfl]
  rfl

def statementBindingFreshStart : Nat := logicalFreshBase
def statementAbsorptionFreshStart : Nat := statementBindingFreshStart
def challengeFreshStart : Nat := statementAbsorptionFreshStart
def roundTranscriptFreshStart : Nat := challengeFreshStart
def initialClaimFreshStart : Nat := roundTranscriptFreshStart
def sumcheckFreshStart : Nat := initialClaimFreshStart
def evalKFreshStart : Nat := sumcheckFreshStart
def evalAFreshStart : Nat := evalKFreshStart + 665
def ccsFreshStart : Nat := evalAFreshStart + 665
def normFreshStart : Nat := ccsFreshStart
def finalIdentityFreshStart : Nat := normFreshStart + 752
def outputBindingFreshStart : Nat := finalIdentityFreshStart + 665 + 209

def freshStarts : List Nat :=
  [statementBindingFreshStart, statementAbsorptionFreshStart,
    challengeFreshStart, roundTranscriptFreshStart, initialClaimFreshStart,
    sumcheckFreshStart, evalKFreshStart, evalAFreshStart, ccsFreshStart,
    normFreshStart, finalIdentityFreshStart, outputBindingFreshStart]

theorem rowStarts_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    rowStarts = prefixStarts rowBase
      (NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset) := by
  let inputs :=
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.inputShapes relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)
  rw [NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rw [NightstreamFPrime.Layout.PiCCS.v1_2.terminalRowCost_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rfl

theorem freshStarts_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    freshStarts = prefixStarts logicalFreshBase
      (NightstreamFPrime.Layout.PiCCS.v1_2.physicalFreshDeltas relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset) := by
  let inputs :=
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.inputShapes relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)
  rw [NightstreamFPrime.Layout.PiCCS.v1_2.physicalFreshDeltas_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rw [NightstreamFPrime.Layout.PiCCS.v1_2.terminalFreshCost_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rfl

end NightstreamFPrime.Layout.Stage1.PiCCSStarts
