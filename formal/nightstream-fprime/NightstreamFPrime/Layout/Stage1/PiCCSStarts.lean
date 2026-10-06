import NightstreamFPrime.Layout.Stage1.PilotPiCCS
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FormalRows

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS Steps 1--5.
Obligation: Own the cumulative physical starts of the twelve PiCCS leaves in
the same order as the logical parent and physical lowering.

This module materializes only fixed prefix sums. The proofs below connect the
row starts to `physicalRowDeltas`, the R1CS-fresh starts to
`physicalFreshDeltas`, and both bases to their existing layout owners.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSStarts

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Start of each child interval from an initial base and ordered deltas. -/
def prefixStarts : Nat → List Nat → List Nat
  | _, [] => []
  | base, delta :: deltas => base :: prefixStarts (base + delta) deltas

/-- The completed pilot owns the physical row prefix. -/
def rowBase : Nat := 6815614

theorem rowBase_eq_layout :
    rowBase = Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset := by
  rw [rowBase, PilotProduction.physicalRowCount_eq]

def statementBindingRowStart : Nat := rowBase
def statementAbsorptionRowStart : Nat := statementBindingRowStart + 160
def challengeRowStart : Nat := statementAbsorptionRowStart + 140288
def roundTranscriptRowStart : Nat := challengeRowStart + 92064
def initialClaimRowStart : Nat := roundTranscriptRowStart + 147960
def sumcheckRowStart : Nat := initialClaimRowStart + 20733
def evalKRowStart : Nat := sumcheckRowStart + 783
def evalARowStart : Nat := evalKRowStart + 3336
def ccsRowStart : Nat := evalARowStart + 18888
def normRowStart : Nat := ccsRowStart + 56
def finalIdentityRowStart : Nat := normRowStart + 800
def outputBindingRowStart : Nat := finalIdentityRowStart + 3574

/-- Row starts in the exact twelve-child parent order. -/
def rowStarts : List Nat :=
  [statementBindingRowStart, statementAbsorptionRowStart, challengeRowStart,
    roundTranscriptRowStart, initialClaimRowStart, sumcheckRowStart,
    evalKRowStart, evalARowStart, ccsRowStart, normRowStart,
    finalIdentityRowStart, outputBindingRowStart]

/-- The logical child starts are also the witness starts for the four
Poseidon2 invocation packets. -/
def statementWitnessStart : Nat := PiCCSInputs.phaseOffset
def challengeWitnessStart : Nat := statementWitnessStart + 140288
def roundTranscriptWitnessStart : Nat := challengeWitnessStart + 92064
def initialClaimLogicalStart : Nat := roundTranscriptWitnessStart + 147960
def sumcheckLogicalStart : Nat := initialClaimLogicalStart + 20733
def evalKLogicalStart : Nat := sumcheckLogicalStart + 729
def evalALogicalStart : Nat := evalKLogicalStart + 2695
def ccsLogicalStart : Nat := evalALogicalStart + 18247
def normLogicalStart : Nat := ccsLogicalStart + 56
def finalIdentityLogicalStart : Nat := normLogicalStart + 48
def outputBindingWitnessStart : Nat := finalIdentityLogicalStart + 2722

theorem statementWitnessStart_eq : statementWitnessStart = 6906620 := by
  unfold statementWitnessStart
  exact PiCCSInputs.phaseOffset_eq

theorem challengeWitnessStart_eq : challengeWitnessStart = 7046908 := by
  unfold challengeWitnessStart
  rw [statementWitnessStart_eq]

theorem roundTranscriptWitnessStart_eq :
    roundTranscriptWitnessStart = 7138972 := by
  unfold roundTranscriptWitnessStart
  rw [challengeWitnessStart_eq]

theorem outputBindingWitnessStart_eq :
    outputBindingWitnessStart = 7332162 := by
  unfold outputBindingWitnessStart finalIdentityLogicalStart
    normLogicalStart ccsLogicalStart evalALogicalStart evalKLogicalStart
    sumcheckLogicalStart initialClaimLogicalStart
  rw [roundTranscriptWitnessStart_eq]

/-- The materialized output-child start is the exact start selected by the
canonical PiCCS parent for every production relation. -/
theorem outputBindingWitnessStart_matches
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    outputBindingWitnessStart =
      Formal.outputBindingOffset relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset := by
  unfold Formal.outputBindingOffset Formal.nextOffset Formal.childLength
  rw [Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation,
    Formal.finalIdentityCircuit,
    NightstreamFPrime.Circuit.FormalCircuit.withConstantFootprint_main,
    FinalIdentity.localLength_eq]
  rfl

/-- Generic R1CS multiplication columns begin after all PiCCS logical
variables. -/
def logicalFreshBase : Nat := PiCCSInputs.phaseOffset + 1768142

theorem logicalFreshBase_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    logicalFreshBase =
      NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset := by
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount_eq_of_degreeBound_eq_nine
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
def evalAFreshStart : Nat := evalKFreshStart + 641
def ccsFreshStart : Nat := evalAFreshStart + 641
def normFreshStart : Nat := ccsFreshStart
def finalIdentityFreshStart : Nat := normFreshStart + 752
def outputBindingFreshStart : Nat := finalIdentityFreshStart + 641 + 209

/-- R1CS-fresh starts in the exact twelve-child parent order. -/
def freshStarts : List Nat :=
  [statementBindingFreshStart, statementAbsorptionFreshStart,
    challengeFreshStart, roundTranscriptFreshStart, initialClaimFreshStart,
    sumcheckFreshStart, evalKFreshStart, evalAFreshStart, ccsFreshStart,
    normFreshStart, finalIdentityFreshStart, outputBindingFreshStart]

/-- The materialized row starts are exactly the cumulative proved physical
row deltas. -/
theorem rowStarts_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    rowStarts = prefixStarts rowBase
      (NightstreamFPrime.Layout.PiCCS.v1_1.physicalRowDeltas relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset) := by
  let inputs :=
    NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.inputShapes relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.physicalRowDeltas_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.terminalRowCost_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rfl

/-- The materialized R1CS-fresh starts are exactly the cumulative proved
fresh-column deltas. -/
theorem freshStarts_eq_layout
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    freshStarts = prefixStarts logicalFreshBase
      (NightstreamFPrime.Layout.PiCCS.v1_1.physicalFreshDeltas relation
        (PiCCSInputs.interface logicalWidth publicFits)
        PiCCSInputs.phaseOffset) := by
  let inputs :=
    NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.inputShapes relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.physicalFreshDeltas_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.terminalFreshCost_eq relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
      inputs]
  rfl

end NightstreamFPrime.Layout.Stage1.PiCCSStarts
