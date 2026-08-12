import Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding

/-!
Contract: exact product-commitment lane geometry for the reference V2 fresh
relation.

The one-step reference profile keeps the complete operations, initial
snapshot, and final snapshot bit block directly after the ten public rings.
The three projection shapes have exactly the aligned widths from
`ConcreteLaneGeometry`. `Exact` rejects any other starts, widths, overlap, or
profile-selected lane layout.

This file owns geometry only. It does not own commitment keys, memory-row
placement, compact-chain rows, NIFS extraction, or cryptographic binding.

Assurance tier: implementation model.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionFreshLaneGeometry

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Commitment-only operations projection shape. -/
def operationsShape : Phi81Relation.Shape where
  rowVariables := 0
  logicalWidth := operationsLaneWidth
  matrixCount := 0
  publicRingColumns := 0
  publicFits := by simp

/-- Commitment-only shared snapshot projection shape. -/
def snapshotShape : Phi81Relation.Shape where
  rowVariables := 0
  logicalWidth := snapshotLaneWidth
  matrixCount := 0
  publicRingColumns := 0
  publicFits := by simp

@[simp] theorem operationsShape_carrierWidth :
    operationsShape.carrierWidth = operationsLaneWidth := by
  norm_num [operationsShape, Shape.carrierWidth,
    Phi81CarrierLayout.carrierWidth, Phi81ColumnLayout.blockCount,
    operationsLaneWidth_exact, ringDegree]

@[simp] theorem snapshotShape_carrierWidth :
    snapshotShape.carrierWidth = snapshotLaneWidth := by
  norm_num [snapshotShape, Shape.carrierWidth,
    Phi81CarrierLayout.carrierWidth, Phi81ColumnLayout.blockCount,
    snapshotLaneWidth_exact, ringDegree]

/-- The sole authority-bearing lane placement for the reference profile. -/
def placement (privateWidth : Nat) :
    ConcreteLaneGeometry.Placement
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth)).carrierWidth where
  base := publicWidth
  assignmentAligned := by
    change
      (Phi81ColumnLayout.blockCount
          (ProductPaperAlgebra.FullShape
            (logicalWidth privateWidth) (publicFits privateWidth)).logicalWidth *
        54) % 54 = 0
    simp
  baseAligned := by
    norm_num [LaneLayout.Aligned, publicWidth, LaneLayout.ringDegree]
  blockWithin := by
    change publicWidth + blockWidth <=
      Phi81CarrierLayout.carrierWidth (logicalWidth privateWidth)
    rw [logicalWidth_carrier_exact]
    have payload := payloadWidth_le_logicalWidth privateWidth
    have direct : publicWidth + blockWidth = directWidth := by
      norm_num [publicWidth, blockWidth_exact, directWidth]
    rw [direct]
    exact Nat.le_trans (directWidth_le_payloadWidth privateWidth) payload

/-- A verifier configuration is exact only when its lane layout is the
canonical placement above. Commitment keys remain separate verifier-key
data. -/
def Exact {privateWidth : Nat}
    (config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape) : Prop :=
  config.lanes = (placement privateWidth).layout

theorem exact_operations_start
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    (layoutExact : Exact config) :
    config.lanes.operationsStart = publicWidth := by
  rw [layoutExact]
  rfl

theorem exact_initial_snapshot_start
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    (layoutExact : Exact config) :
    config.lanes.initialSnapshotStart = publicWidth + operationsLaneWidth := by
  rw [layoutExact]
  rfl

theorem exact_final_snapshot_start
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    (layoutExact : Exact config) :
    config.lanes.finalSnapshotStart =
      publicWidth + operationsLaneWidth + snapshotLaneWidth := by
  rw [layoutExact]
  exact ConcreteLaneGeometry.Placement.layout_finalSnapshotStart _

def operationsBlockIndex
    (index : Fin operationsShape.carrierWidth) : Fin laneWidth :=
  ⟨index.val, by
    have bounded : index.val < operationsLaneWidth := by
      simpa only [operationsShape_carrierWidth] using index.isLt
    norm_num [laneWidth, operationsLaneWidth_exact] at bounded ⊢
    omega⟩

def initialSnapshotBlockIndex
    (index : Fin snapshotShape.carrierWidth) : Fin laneWidth :=
  ⟨operationsLaneWidth + index.val, by
    have bounded : index.val < snapshotLaneWidth := by
      simpa only [snapshotShape_carrierWidth] using index.isLt
    norm_num [laneWidth, operationsLaneWidth_exact,
      snapshotLaneWidth_exact] at bounded ⊢
    omega⟩

def finalSnapshotBlockIndex
    (index : Fin snapshotShape.carrierWidth) : Fin laneWidth :=
  ⟨operationsLaneWidth + snapshotLaneWidth + index.val, by
    have bounded : index.val < snapshotLaneWidth := by
      simpa only [snapshotShape_carrierWidth] using index.isLt
    norm_num [laneWidth, operationsLaneWidth_exact,
      snapshotLaneWidth_exact] at bounded ⊢
    omega⟩

def fullLaneColumn {privateWidth : Nat} (laneIndex : Fin laneWidth) :
    Fin (ProductPaperAlgebra.FullShape
      (logicalWidth privateWidth) (publicFits privateWidth)).carrierWidth :=
  Phi81CarrierLayout.embedLogical
    (payloadColumn (finSumFinEquiv (Sum.inl
      (⟨publicWidth + laneIndex.val, by
        have laneBound := laneIndex.isLt
        norm_num [publicWidth, directWidth, laneWidth] at laneBound ⊢
        omega⟩ : Fin directWidth) :
      Fin directWidth ⊕ Fin (privateWidth *
        Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.digitCount))))

@[simp] theorem fullLaneColumn_val
    {privateWidth : Nat} (laneIndex : Fin laneWidth) :
    (fullLaneColumn (privateWidth := privateWidth) laneIndex).val =
      publicWidth + laneIndex.val := rfl

/-- The operations commitment reads the exact direct operations bits. -/
theorem operations_project_exact
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    (layoutExact : Exact config)
    (source : SourceAssignment privateWidth)
    (index : Fin operationsShape.carrierWidth) :
    config.operationsSlice.project (encodeCarrier source) index =
      source (laneSourceColumn (operationsBlockIndex index)) := by
  rw [AlignedLaneAction.Slice.project_apply]
  have startEq : config.operationsSlice.start = publicWidth := by
    exact exact_operations_start layoutExact
  have coordinateEq :
      (⟨config.operationsSlice.start + index.val, by
        have bounded := index.isLt
        have within := config.operationsSlice.within
        omega⟩ : Fin (ProductPaperAlgebra.FullShape
          (logicalWidth privateWidth) (publicFits privateWidth)).carrierWidth) =
        fullLaneColumn (privateWidth := privateWidth)
          (operationsBlockIndex index) := by
    apply Fin.ext
    simp [startEq, operationsBlockIndex]
  rw [coordinateEq]
  simpa [fullLaneColumn] using
    encodeCarrier_lane source (operationsBlockIndex index)

/-- The initial-snapshot commitment reads the exact direct snapshot bits. -/
theorem initial_snapshot_project_exact
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    (layoutExact : Exact config)
    (source : SourceAssignment privateWidth)
    (index : Fin snapshotShape.carrierWidth) :
    config.initialSnapshotSlice.project (encodeCarrier source) index =
      source (laneSourceColumn (initialSnapshotBlockIndex index)) := by
  rw [AlignedLaneAction.Slice.project_apply]
  have startEq : config.initialSnapshotSlice.start =
      publicWidth + operationsLaneWidth := by
    exact exact_initial_snapshot_start layoutExact
  have coordinateEq :
      (⟨config.initialSnapshotSlice.start + index.val, by
        have bounded := index.isLt
        have within := config.initialSnapshotSlice.within
        omega⟩ : Fin (ProductPaperAlgebra.FullShape
          (logicalWidth privateWidth) (publicFits privateWidth)).carrierWidth) =
        fullLaneColumn (privateWidth := privateWidth)
          (initialSnapshotBlockIndex index) := by
    apply Fin.ext
    simp [startEq, initialSnapshotBlockIndex, Nat.add_assoc]
  rw [coordinateEq]
  simpa [fullLaneColumn] using
    encodeCarrier_lane source (initialSnapshotBlockIndex index)

/-- The final-snapshot commitment reads the exact direct snapshot bits. -/
theorem final_snapshot_project_exact
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    (layoutExact : Exact config)
    (source : SourceAssignment privateWidth)
    (index : Fin snapshotShape.carrierWidth) :
    config.finalSnapshotSlice.project (encodeCarrier source) index =
      source (laneSourceColumn (finalSnapshotBlockIndex index)) := by
  rw [AlignedLaneAction.Slice.project_apply]
  have startEq : config.finalSnapshotSlice.start =
      publicWidth + operationsLaneWidth + snapshotLaneWidth := by
    exact exact_final_snapshot_start layoutExact
  have coordinateEq :
      (⟨config.finalSnapshotSlice.start + index.val, by
        have bounded := index.isLt
        have within := config.finalSnapshotSlice.within
        omega⟩ : Fin (ProductPaperAlgebra.FullShape
          (logicalWidth privateWidth) (publicFits privateWidth)).carrierWidth) =
        fullLaneColumn (privateWidth := privateWidth)
          (finalSnapshotBlockIndex index) := by
    apply Fin.ext
    simp [startEq, finalSnapshotBlockIndex, Nat.add_assoc]
  rw [coordinateEq]
  simpa [fullLaneColumn] using
    encodeCarrier_lane source (finalSnapshotBlockIndex index)

end Nightstream.Implementation.NebulaV2.ProductionFreshLaneGeometry
