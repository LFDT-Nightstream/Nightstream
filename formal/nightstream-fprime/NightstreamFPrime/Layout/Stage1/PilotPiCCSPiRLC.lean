import NightstreamFPrime.Layout.PiRLC.v1_2.Preservation
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PilotPiCCS

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLC

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

def piRlcOffset : Nat := PiRLCInputs.phaseOffset

theorem piRlcOffset_eq : piRlcOffset = 5806589 := by
  rfl

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  PilotPiCCS.physicalRows relation ++
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalRows relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  (physicalRows relation).length

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (PilotPiCCS.physicalColumnCount relation)
    (NightstreamFPrime.Layout.PiRLC.v1_2.physicalColumnCount relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset)

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (physicalRowCount relation) (physicalColumnCount relation)

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation)

def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiRLC.v1_2.cumulativePhysicalRows relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) piRlcOffset).map
      (5744925 + ·)

def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiRLC.v1_2.cumulativePhysicalColumns relation
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
        NightstreamFPrime.Layout.PiRLC.v1_2.PhysicalHolds relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth)
            (publicFits := publicFits)) piRlcOffset env := by
  exact R1CS.rowsHold_append env _ _

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount relation = 9941520 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change PilotPiCCS.physicalRowCount relation +
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalRowCount relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset = 9941520
  rw [PilotPiCCS.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalRowCount_eq_production
    relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) piRlcOffset
    (PiRLCInputs.inputShapes relation)]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 10029764 := by
  unfold physicalColumnCount
  rw [PilotPiCCS.physicalColumnCount_eq,
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalColumnCount_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset
      (PiRLCInputs.inputShapes relation),
    piRlcOffset_eq]
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 10029764 := by
  unfold jointDomain
  rw [physicalRowCount_eq relation, physicalColumnCount_eq relation]
  norm_num

theorem jointDomain_le_twoPow28
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation ≤ 2 ^ 28 := by
  rw [jointDomain_eq relation]
  norm_num

theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalRowDeltas relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) piRlcOffset =
        [0, 46515, 2263680, 628800, 251520, 1006080, 0] ∧
      NightstreamFPrime.Layout.PiRLC.v1_2.physicalColumnDeltas relation
        (PiRLCInputs.interface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) piRlcOffset =
        [0, 73095, 2263680, 628800, 251520, 1006080, 0] ∧
      cumulativePhysicalRows relation =
        [5744925, 5791440, 8055120, 8683920, 8935440, 9941520,
          9941520] ∧
      cumulativePhysicalColumns relation =
        [5806589, 5879684, 8143364, 8772164, 9023684, 10029764,
          10029764] ∧
      cumulativeJointDomains relation =
        [5806589, 5879684, 8143364, 8772164, 9023684, 10029764,
          10029764] := by
  let inputs := PiRLCInputs.inputShapes relation
  have rows :=
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalRowDeltas_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset inputs
  have columns :=
    NightstreamFPrime.Layout.PiRLC.v1_2.physicalColumnDeltas_eq_production
      relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth)
        (publicFits := publicFits)) piRlcOffset inputs
  have cumulative :=
    NightstreamFPrime.Layout.PiRLC.v1_2.cumulativeFootprints_eq_production
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
