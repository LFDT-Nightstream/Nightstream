import NightstreamFPrime.Layout.PiRLC.v1_1.Preservation
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PilotPiCCS

/-!
Owns the Stage 1 prefix through the exact PiRLC v1_1 phase.

The PiRLC packet starts at the proved final PiCCS source-column boundary. It
adds no public input, constant column, copy row, or parent assertion row. The
seven PiRLC child packets remain in canonical parent order.
-/

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLC

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The exact PiRLC source-column start is the completed PiCCS endpoint. -/
def piRlcOffset : Nat := PiRLCInputs.phaseOffset

theorem piRlcOffset_eq : piRlcOffset = 17869582 := by
  rfl

/-- Exact physical row order of the Stage 1 prefix through PiRLC. -/
def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  PilotPiCCS.physicalRows relation ++
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  (physicalRows relation).length

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (PilotPiCCS.physicalColumnCount relation)
    (NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset)

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (physicalRowCount relation) (physicalColumnCount relation)

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation)

/-- Stage 1 row endpoints after the seven PiRLC children. -/
def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiRLC.v1_1.cumulativePhysicalRows relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) piRlcOffset).map
      (17755828 + ·)

/-- Stage 1 source-column endpoints after the seven PiRLC children. -/
def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiRLC.v1_1.cumulativePhysicalColumns relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) piRlcOffset).map
      (piRlcOffset + ·)

def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  List.zipWith max (cumulativePhysicalRows relation)
    (cumulativePhysicalColumns relation)

theorem physicalHolds_iff
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PhysicalHolds relation env ↔
      PilotPiCCS.PhysicalHolds relation env ∧
        NightstreamFPrime.Layout.PiRLC.v1_1.PhysicalHolds relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth)
            (publicFits := publicFits)) piRlcOffset env := by
  exact R1CS.rowsHold_append env _ _

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount relation = 25556958 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change PilotPiCCS.physicalRowCount relation +
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalRowCount relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset = 25556958
  rw [PilotPiCCS.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalRowCount_eq_production
    relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) piRlcOffset
    (PiRLCInputs.inputShapes relation)]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 25669063 := by
  unfold physicalColumnCount
  rw [PilotPiCCS.physicalColumnCount_eq,
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset
      (PiRLCInputs.inputShapes relation),
    piRlcOffset_eq]
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 25669063 := by
  unfold jointDomain
  rw [physicalRowCount_eq relation, physicalColumnCount_eq relation]
  norm_num

theorem jointDomain_le_twoPow25
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation ≤ 2 ^ 25 := by
  rw [jointDomain_eq relation]
  norm_num

/-- One transported ledger states every PiRLC delta and every Stage 1
endpoint after the completed PiCCS prefix. -/
theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalRowDeltas relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) piRlcOffset =
        [0, 1008848, 2495124, 138618, 277236, 3881304, 0] ∧
      NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnDeltas relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) piRlcOffset =
        [0, 1007199, 2495124, 138618, 277236, 3881304, 0] ∧
      cumulativePhysicalRows relation =
        [17755828, 18764676, 21259800, 21398418, 21675654, 25556958,
          25556958] ∧
      cumulativePhysicalColumns relation =
        [17869582, 18876781, 21371905, 21510523, 21787759, 25669063,
          25669063] ∧
      cumulativeJointDomains relation =
        [17869582, 18876781, 21371905, 21510523, 21787759, 25669063,
          25669063] := by
  let inputs := PiRLCInputs.inputShapes relation
  have rows :=
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalRowDeltas_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset inputs
  have columns :=
    NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnDeltas_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset inputs
  have cumulative :=
    NightstreamFPrime.Layout.PiRLC.v1_1.cumulativeFootprints_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset inputs
  rcases cumulative with ⟨cumulativeRows, cumulativeColumns, cumulativeJoint⟩
  refine ⟨rows, columns, ?_, ?_, ?_⟩
  · rw [cumulativePhysicalRows, cumulativeRows]
    norm_num
  · rw [cumulativePhysicalColumns, cumulativeColumns, piRlcOffset_eq]
    norm_num
  · rw [cumulativeJointDomains, cumulativePhysicalRows,
      cumulativePhysicalColumns, cumulativeRows, cumulativeColumns,
      piRlcOffset_eq]
    norm_num

end NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLC
