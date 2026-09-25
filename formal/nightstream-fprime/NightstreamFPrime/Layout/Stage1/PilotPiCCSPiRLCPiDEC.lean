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
    physicalRowCount relation = PiDECStarts.outputRowStart := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change PilotPiCCSPiRLC.physicalRowCount relation +
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowCount relation
      (PiDECInputs.interface logicalWidth publicFits) piDecOffset = PiDECStarts.outputRowStart
  rw [PilotPiCCSPiRLC.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowCount_eq_production
      relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset
      (PiDECInputs.inputShapes relation)]
  rfl

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = PiDECStarts.outputFreshStart := by
  unfold physicalColumnCount
  rw [NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnCount_eq_production
    relation (PiDECInputs.interface logicalWidth publicFits) piDecOffset
    (PiDECInputs.inputShapes relation)]
  dsimp only [piDecOffset]
  rw [Nat.max_eq_right (Nat.le_add_right _ _)]
  rfl

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation =
      max PiDECStarts.outputRowStart PiDECStarts.outputFreshStart := by
  unfold jointDomain
  rw [physicalRowCount_eq relation, physicalColumnCount_eq relation]

theorem jointDomain_le_twoPow28
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation ≤ 2 ^ 28 := by
  rw [jointDomain_eq relation]
  decide

def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiDEC.v1_1.cumulativePhysicalRows relation
    (PiDECInputs.interface logicalWidth publicFits) piDecOffset).map
      (PiDECStarts.phaseRowStart + ·)

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

/-- Cumulative phase footprints follow the certified child deltas. -/
theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    cumulativePhysicalRows relation =
      (PiDEC.v1_1.cumulativeFrom 0 PiDEC.v1_1.exactRowDeltas).map
        (PiDECStarts.phaseRowStart + ·) ∧
    cumulativePhysicalColumns relation =
      (PiDEC.v1_1.cumulativeFrom 0 PiDEC.v1_1.exactPhysicalColumnDeltas).map
        (piDecOffset + ·) := by
  have inputs := PiDECInputs.inputShapes relation
  constructor
  · simp only [cumulativePhysicalRows, PiDEC.v1_1.cumulativePhysicalRows, piDecOffset,
      PiDEC.v1_1.physicalRowDeltas_eq relation _ _ inputs]
  · simp only [cumulativePhysicalColumns, PiDEC.v1_1.cumulativePhysicalColumns, piDecOffset,
      PiDEC.v1_1.physicalColumnDeltas_eq relation _ _ inputs]

end NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC
