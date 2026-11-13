import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.PiCCS.v1_2.Preservation
import NightstreamFPrime.Layout.Stage1.PiCCSInputs

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

def interface : Formal.Interface logicalWidth 8 publicFits :=
  PiCCSInputs.interface logicalWidth publicFits

def piCcsOffset : Nat := PiCCSInputs.phaseOffset

theorem piCcsOffset_eq : piCcsOffset = 5360990 := by
  exact PiCCSInputs.phaseOffset_eq

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  Pilot.physicalRows PilotProduction.interface PilotProduction.witnessOffset ++
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRows relation
      (interface (publicFits := publicFits)) piCcsOffset

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  (physicalRows relation).length

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max
    (Pilot.physicalColumnCount PilotProduction.interface
      PilotProduction.witnessOffset)
    (NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnCount relation
      (interface (publicFits := publicFits)) piCcsOffset)

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
  (NightstreamFPrime.Layout.PiCCS.v1_2.cumulativePhysicalRows relation
    (interface (publicFits := publicFits)) piCcsOffset).map
      (PilotProduction.physicalRowCountValue + ·)

def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiCCS.v1_2.cumulativePhysicalColumns relation
    (interface (publicFits := publicFits)) piCcsOffset).map
      (piCcsOffset + ·)

def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  List.zipWith max (cumulativePhysicalRows relation)
    (cumulativePhysicalColumns relation)

theorem physicalHolds_iff
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PhysicalHolds relation env ↔
      Pilot.PhysicalHolds PilotProduction.interface
          PilotProduction.witnessOffset env ∧
        NightstreamFPrime.Layout.PiCCS.v1_2.PhysicalHolds relation
          (interface (publicFits := publicFits)) piCcsOffset env := by
  exact R1CS.rowsHold_append env _ _

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount relation = 6524761 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset +
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowCount relation
      (interface (publicFits := publicFits)) piCcsOffset = 6524761
  rw [PilotProduction.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.physicalRowCount_eq
      relation (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 6594873 := by
  unfold physicalColumnCount
  rw [PilotProduction.physicalColumnCount_eq,
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.physicalColumnCount_eq
      relation (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits),
    piCcsOffset_eq]
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 6594873 := by
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
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas relation
        (interface (publicFits := publicFits)) piCcsOffset =
        [5632, 135904, 4384, 61376, 15357, 728, 3844, 13060,
          23, 800, 4073, 994072] ∧
      NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnDeltas relation
        (interface (publicFits := publicFits)) piCcsOffset =
        [320, 135904, 4384, 61376, 15357, 672, 3844, 13060,
          23, 800, 4071, 994072] ∧
      cumulativePhysicalRows relation =
        [5291140, 5427044, 5431428, 5492804, 5508161, 5508889,
          5512733, 5525793, 5525816, 5526616, 5530689, 6524761] ∧
      cumulativePhysicalColumns relation =
        [5361310, 5497214, 5501598, 5562974, 5578331, 5579003,
          5582847, 5595907, 5595930, 5596730, 5600801, 6594873] ∧
      cumulativeJointDomains relation =
        [5361310, 5497214, 5501598, 5562974, 5578331, 5579003,
          5582847, 5595907, 5595930, 5596730, 5600801, 6594873] := by
  let inputs :=
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.inputShapes relation
      (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)
  have rows :=
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas_eq_production
      relation (interface (publicFits := publicFits)) piCcsOffset inputs
  have columns :=
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnDeltas_eq_production
      relation (interface (publicFits := publicFits)) piCcsOffset inputs
  have cumulative :=
    NightstreamFPrime.Layout.PiCCS.v1_2.cumulativeFootprints_eq_production
      relation (interface (publicFits := publicFits)) piCcsOffset inputs
  rcases cumulative with ⟨cumulativeRows, cumulativeColumns, cumulativeJoint⟩
  refine ⟨rows, columns, ?_, ?_, ?_⟩
  · rw [cumulativePhysicalRows, cumulativeRows,
      PilotProduction.physicalRowCountValue_eq]
    norm_num
  · rw [cumulativePhysicalColumns, cumulativeColumns, piCcsOffset_eq]
    norm_num
  · rw [cumulativeJointDomains, cumulativePhysicalRows,
      cumulativePhysicalColumns, cumulativeRows, cumulativeColumns,
      PilotProduction.physicalRowCountValue_eq, piCcsOffset_eq]
    norm_num

end NightstreamFPrime.Layout.Stage1.PilotPiCCS
