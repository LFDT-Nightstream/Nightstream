import NightstreamFPrime.Layout.Stage1.PilotPiCCS

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
def rowBase : Nat := PilotProduction.physicalRowCountValue

theorem rowBase_eq_layout :
    rowBase = Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset := by
  rw [rowBase, PilotProduction.physicalRowCountValue_eq,
    PilotProduction.physicalRowCount_eq]

def statementBindingRowStart : Nat := rowBase
def statementAbsorptionRowStart : Nat := statementBindingRowStart + 160
def challengeRowStart : Nat := statementAbsorptionRowStart + 160432
def roundTranscriptRowStart : Nat := challengeRowStart + 46176
def initialClaimRowStart : Nat := roundTranscriptRowStart + 133200
def sumcheckRowStart : Nat := initialClaimRowStart + 116631
def evalKRowStart : Nat := sumcheckRowStart + 378610
def evalARowStart : Nat := evalKRowStart + 8458
def ccsRowStart : Nat := evalARowStart + 109546
def normRowStart : Nat := ccsRowStart + 20794
def finalIdentityRowStart : Nat := normRowStart + 752
def outputBindingRowStart : Nat := finalIdentityRowStart + 130419

/-- Row starts in the exact twelve-child parent order. -/
def rowStarts : List Nat :=
  [statementBindingRowStart, statementAbsorptionRowStart, challengeRowStart,
    roundTranscriptRowStart, initialClaimRowStart, sumcheckRowStart,
    evalKRowStart, evalARowStart, ccsRowStart, normRowStart,
    finalIdentityRowStart, outputBindingRowStart]

/-- The logical child starts are also the witness starts for the four
Poseidon2 invocation packets. -/
def statementWitnessStart : Nat := PiCCSInputs.phaseOffset
def challengeWitnessStart : Nat := statementWitnessStart + 160432
def roundTranscriptWitnessStart : Nat := challengeWitnessStart + 46176
def initialClaimLogicalStart : Nat := roundTranscriptWitnessStart + 133200
def sumcheckLogicalStart : Nat := initialClaimLogicalStart + 25918
def evalKLogicalStart : Nat := sumcheckLogicalStart
def evalALogicalStart : Nat := evalKLogicalStart + 1824
def ccsLogicalStart : Nat := evalALogicalStart + 24288
def normLogicalStart : Nat := ccsLogicalStart + 2
def finalIdentityLogicalStart : Nat := normLogicalStart + 32
def outputBindingWitnessStart : Nat := finalIdentityLogicalStart + 27746

theorem statementWitnessStart_eq : statementWitnessStart = 12688104 := by
  exact PiCCSInputs.phaseOffset_eq

theorem challengeWitnessStart_eq : challengeWitnessStart = 12848536 := by
  rw [challengeWitnessStart, statementWitnessStart_eq]

theorem roundTranscriptWitnessStart_eq :
    roundTranscriptWitnessStart = 12894712 := by
  rw [roundTranscriptWitnessStart, challengeWitnessStart_eq]

theorem outputBindingWitnessStart_eq :
    outputBindingWitnessStart = 13107722 := by
  norm_num [outputBindingWitnessStart, finalIdentityLogicalStart,
    normLogicalStart, ccsLogicalStart, evalALogicalStart, evalKLogicalStart,
    sumcheckLogicalStart, initialClaimLogicalStart,
    roundTranscriptWitnessStart_eq]

/-- Generic R1CS multiplication columns begin after all PiCCS logical
variables. -/
def logicalFreshBase : Nat := PiCCSInputs.phaseOffset + 4496130

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
def sumcheckFreshStart : Nat := initialClaimFreshStart + 90713
def evalKFreshStart : Nat := sumcheckFreshStart + 378560
def evalAFreshStart : Nat := evalKFreshStart + 6634
def ccsFreshStart : Nat := evalAFreshStart + 85258
def normFreshStart : Nat := ccsFreshStart + 20792
def finalIdentityFreshStart : Nat := normFreshStart + 720
def outputBindingFreshStart : Nat := finalIdentityFreshStart + 97347 + 5324

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
