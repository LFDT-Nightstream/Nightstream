import NightstreamFPrime.Layout.Stage1.PiDECStarts

/-!
Owns the Stage 1 prefix through the exact PiDEC v1_1 phase.

The 49,248-word PiDEC input ABI follows the completed PiRLC physical endpoint.
The PiDEC packet then adds 25,488 rows and 18,090 logical-plus-R1CS private
columns. No public column, copy row, or boundary row is added.
-/

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def piDecOffset : Nat := PiDECInputs.phaseOffset

theorem piDecOffset_eq : piDecOffset = 29022496 := by
  rfl

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  PilotPiCCSPiRLC.physicalRows relation ++
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows relation
      (PiDECInputs.interface logicalWidth publicFits) piDecOffset

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  (physicalRows relation).length

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max PiDECInputs.phaseOffset
    (NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnCount relation
      (PiDECInputs.interface logicalWidth publicFits) piDecOffset)

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (physicalRowCount relation) (physicalColumnCount relation)

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation)

theorem physicalHolds_iff
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PhysicalHolds relation env ↔
      PilotPiCCSPiRLC.PhysicalHolds relation env ∧
        NightstreamFPrime.Layout.PiDEC.v1_1.PhysicalHolds relation
          (PiDECInputs.interface logicalWidth publicFits) piDecOffset env := by
  exact R1CS.rowsHold_append env _ _

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount relation = 28872529 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change PilotPiCCSPiRLC.physicalRowCount relation +
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowCount relation
      (PiDECInputs.interface logicalWidth publicFits) piDecOffset = 28872529
  rw [PilotPiCCSPiRLC.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowCount_eq_production
      relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset
      (PiDECInputs.inputShapes relation)]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 29040586 := by
  unfold physicalColumnCount
  rw [NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnCount_eq_production
    relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset
    (PiDECInputs.inputShapes relation), piDecOffset_eq]
  change max 29022496 29040586 = 29040586
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 29040586 := by
  unfold jointDomain
  rw [physicalRowCount_eq relation, physicalColumnCount_eq relation]
  norm_num

theorem jointDomain_le_twoPow28
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation ≤ 2 ^ 28 := by
  rw [jointDomain_eq relation]
  norm_num

def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiDEC.v1_1.cumulativePhysicalRows relation
    (PiDECInputs.interface logicalWidth publicFits) piDecOffset).map
      (28847041 + ·)

def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiDEC.v1_1.cumulativePhysicalColumns relation
    (PiDECInputs.interface logicalWidth publicFits) piDecOffset).map
      (piDecOffset + ·)

def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  List.zipWith max (cumulativePhysicalRows relation)
    (cumulativePhysicalColumns relation)

theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowDeltas relation
        (PiDECInputs.interface logicalWidth publicFits) piDecOffset =
        [0, 22680, 1188, 108, 1512, 0] ∧
      NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnDeltas relation
        (PiDECInputs.interface logicalWidth publicFits) piDecOffset =
        [0, 18090, 0, 0, 0, 0] ∧
      cumulativePhysicalRows relation =
        [28847041, 28869721, 28870909, 28871017, 28872529, 28872529] ∧
      cumulativePhysicalColumns relation =
        [29022496, 29040586, 29040586, 29040586, 29040586, 29040586] ∧
      cumulativeJointDomains relation =
        [29022496, 29040586, 29040586, 29040586, 29040586, 29040586] := by
  let inputs := PiDECInputs.inputShapes relation
  have rows := NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowDeltas_eq
    relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset inputs
  have columns := NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnDeltas_eq
    relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset inputs
  have cumulative := NightstreamFPrime.Layout.PiDEC.v1_1.cumulativeFootprints_eq
    relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset inputs
  rcases cumulative with ⟨cumulativeRows, cumulativeColumns, cumulativeJoint⟩
  refine ⟨rows, columns, ?_, ?_, ?_⟩
  · rw [cumulativePhysicalRows, cumulativeRows]
    norm_num
  · rw [cumulativePhysicalColumns, cumulativeColumns, piDecOffset_eq]
    norm_num
  · rw [cumulativeJointDomains, cumulativePhysicalRows,
      cumulativePhysicalColumns, cumulativeRows, cumulativeColumns,
      piDecOffset_eq]
    norm_num

end NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC
