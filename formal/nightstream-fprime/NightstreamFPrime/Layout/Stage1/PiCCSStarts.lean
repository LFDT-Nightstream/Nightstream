import NightstreamFPrime.Layout.Stage1.PiCCSInputs
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
    RingCarrierLayout.carrierWidth logicalWidth}

def prefixStarts : Nat → List Nat → List Nat
  | _, [] => []
  | base, delta :: deltas => base :: prefixStarts (base + delta) deltas

def rowBase : Nat := 4957224

theorem rowBase_eq_layout :
    rowBase = Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset := by
  rw [rowBase, PilotProduction.physicalRowCount_eq]

def statementBindingRowStart : Nat := rowBase
def statementAbsorptionRowStart : Nat := statementBindingRowStart + 4992
def challengeRowStart : Nat := statementAbsorptionRowStart + 145576
def roundTranscriptRowStart : Nat := challengeRowStart + 4696
def initialClaimRowStart : Nat := roundTranscriptRowStart + 32872
def sumcheckRowStart : Nat := initialClaimRowStart + 13437
def evalKRowStart : Nat := sumcheckRowStart + 392
def evalARowStart : Nat := evalKRowStart + 3460
def ccsRowStart : Nat := evalARowStart + 11524
def normRowStart : Nat := ccsRowStart + 11
def finalIdentityRowStart : Nat := normRowStart + 700
def outputBindingRowStart : Nat := finalIdentityRowStart + 3689

def rowStarts : List Nat :=
  [statementBindingRowStart, statementAbsorptionRowStart, challengeRowStart,
    roundTranscriptRowStart, initialClaimRowStart, sumcheckRowStart,
    evalKRowStart, evalARowStart, ccsRowStart, normRowStart,
    finalIdentityRowStart, outputBindingRowStart]

def statementBindingLogicalStart : Nat := PiCCSInputs.phaseOffset
def statementWitnessStart : Nat := statementBindingLogicalStart + 320
def challengeWitnessStart : Nat := statementWitnessStart + 145576
def roundTranscriptWitnessStart : Nat := challengeWitnessStart + 4696
def initialClaimLogicalStart : Nat := roundTranscriptWitnessStart + 32872
def sumcheckLogicalStart : Nat := initialClaimLogicalStart + 13437
def evalKLogicalStart : Nat := sumcheckLogicalStart + 336
def evalALogicalStart : Nat := evalKLogicalStart + 2795
def ccsLogicalStart : Nat := evalALogicalStart + 10859
def normLogicalStart : Nat := ccsLogicalStart + 11
def finalIdentityLogicalStart : Nat := normLogicalStart + 42
def outputBindingWitnessStart : Nat := finalIdentityLogicalStart + 2813

theorem phaseOffset_le_statementWitnessStart :
    PiCCSInputs.phaseOffset ≤ statementWitnessStart :=
  Nat.le_add_right _ _

theorem statementWitnessStart_eq : statementWitnessStart = 5023714 := by
  unfold statementWitnessStart statementBindingLogicalStart
  rw [PiCCSInputs.phaseOffset_eq]

theorem challengeWitnessStart_eq : challengeWitnessStart = 5169290 := by
  unfold challengeWitnessStart
  rw [statementWitnessStart_eq]

theorem roundTranscriptWitnessStart_eq :
    roundTranscriptWitnessStart = 5173986 := by
  unfold roundTranscriptWitnessStart
  rw [challengeWitnessStart_eq]

theorem outputBindingWitnessStart_eq :
    outputBindingWitnessStart = 5237151 := by
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

def logicalFreshBase : Nat := PiCCSInputs.phaseOffset + 1152957

theorem logicalFreshBase_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    logicalFreshBase =
      NightstreamFPrime.Layout.PiCCS.v1_2.logicalColumnCount relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset := by
  rw [NightstreamFPrime.Layout.PiCCS.v1_2.logicalColumnCount_eq_of_degreeBound_eq_four
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
def finalIdentityFreshStart : Nat := normFreshStart + 658
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
