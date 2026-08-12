import Nightstream.Implementation.NebulaV2.ProductionFreshLaneGeometry
import Nightstream.Implementation.NebulaV2.ProductionFreshLinearSubstitution
import Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics

/-!
Contract: bind the reference V2 producer memory rows to the exact committed
fresh-assignment lanes.

The reference profile has one checked step per fresh claim. Its memory source
rows use the same finite source assignment that the exact fresh compiler
encodes and commits. `LayoutExact` fixes the row lane base at coordinate 540.
The projection theorems then show, coordinate by coordinate, that all three
commitment lanes are the producer row bits. No consumer replay assignment is
used as authority for these records.

This file does not own compact-root updates, NIFS extraction, commitment
binding, source-row satisfaction, or Rust refinement.

Assurance tier: implementation-to-protocol bridge.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionFreshLaneAuthority

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionFreshLaneGeometry
open Nightstream.Implementation.NebulaV2.ProductionFreshLinearSubstitution
open Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

/-- The unique checked-step index of the reference `E = 1` profile. -/
def soleStep : Fin (ProductionMemoryCheckedBatchRows.StepCount .e1) :=
  ⟨0, by decide⟩

/-- The producer checked-step rows directly own the exact lane block. -/
structure LayoutExact
    (layout : ProductionMemoryCheckedBatchRows.Layout .e1) : Prop where
  laneBase :
    (layout.steps soleStep).source.product.laneBase = publicWidth

/-- All alignment-only tail coordinates are canonical zero. These bits are
inside the committed lanes even though they do not encode a record field. -/
structure PaddingZero {privateWidth : Nat}
    (source : SourceAssignment privateWidth) : Prop where
  operations : ∀ offset (_lower : operationPayloadWidth ≤ offset)
      (upper : offset < operationsLaneWidth),
    source (laneSourceColumn
      ⟨offset, by
        have within : operationsLaneWidth ≤ laneWidth := by
          norm_num [operationsLaneWidth_exact, laneWidth]
        exact Nat.lt_of_lt_of_le upper within⟩) = 0
  initialSnapshot : ∀ offset (_lower : snapshotPayloadWidth ≤ offset)
      (upper : offset < snapshotLaneWidth),
    source (laneSourceColumn
      ⟨operationsLaneWidth + offset, by
        have offsetBound : offset < 3564 := by
          simpa only [snapshotLaneWidth_exact] using upper
        norm_num [operationsLaneWidth_exact, laneWidth] at ⊢
        omega⟩) = 0
  finalSnapshot : ∀ offset (_lower : snapshotPayloadWidth ≤ offset)
      (upper : offset < snapshotLaneWidth),
    source (laneSourceColumn
      ⟨operationsLaneWidth + snapshotLaneWidth + offset, by
        have offsetBound : offset < 3564 := by
          simpa only [snapshotLaneWidth_exact] using upper
        norm_num [operationsLaneWidth_exact, snapshotLaneWidth_exact,
          laneWidth] at ⊢
        omega⟩) = 0

/-- Every operations projection coordinate is the same physical source value
read by the producer checked-step relation. -/
theorem operations_projection_eq_row
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    (configExact : ProductionFreshLaneGeometry.Exact config)
    (layoutExact : LayoutExact layout)
    (source : SourceAssignment privateWidth)
    (index : Fin operationsShape.carrierWidth) :
    (config.operationsSlice.project (encodeCarrier source) index).val =
      sourceNat source
        ((layout.steps soleStep).source.product.laneBase + index.val) := by
  rw [operations_project_exact configExact source index]
  calc
    (source (laneSourceColumn (operationsBlockIndex index))).val =
        sourceNat source
          (laneSourceColumn (operationsBlockIndex index)).val :=
      (sourceNat_sourceColumn source
        (laneSourceColumn (operationsBlockIndex index))).symm
    _ = sourceNat source
        ((layout.steps soleStep).source.product.laneBase + index.val) := by
      apply congrArg (sourceNat source)
      simp [layoutExact.laneBase, laneSourceColumn, operationsBlockIndex,
        directSourceColumn]

/-- Every initial-snapshot projection coordinate is the same physical source
value read by the producer checked-step relation. -/
theorem initial_snapshot_projection_eq_row
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    (configExact : ProductionFreshLaneGeometry.Exact config)
    (layoutExact : LayoutExact layout)
    (source : SourceAssignment privateWidth)
    (index : Fin snapshotShape.carrierWidth) :
    (config.initialSnapshotSlice.project (encodeCarrier source) index).val =
      sourceNat source
        (MemoryProductUpdateRows.initialSnapshotStart
          (layout.steps soleStep).source.product +
          index.val) := by
  rw [initial_snapshot_project_exact configExact source index]
  calc
    (source (laneSourceColumn (initialSnapshotBlockIndex index))).val =
        sourceNat source
          (laneSourceColumn (initialSnapshotBlockIndex index)).val :=
      (sourceNat_sourceColumn source
        (laneSourceColumn (initialSnapshotBlockIndex index))).symm
    _ = sourceNat source
        (MemoryProductUpdateRows.initialSnapshotStart
          (layout.steps soleStep).source.product + index.val) := by
      apply congrArg (sourceNat source)
      simp [MemoryProductUpdateRows.initialSnapshotStart,
        layoutExact.laneBase, laneSourceColumn, initialSnapshotBlockIndex,
        directSourceColumn, initialSnapshotRelativeStart, Nat.add_assoc]

/-- Every final-snapshot projection coordinate is the same physical source
value read by the producer checked-step relation. -/
theorem final_snapshot_projection_eq_row
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    (configExact : ProductionFreshLaneGeometry.Exact config)
    (layoutExact : LayoutExact layout)
    (source : SourceAssignment privateWidth)
    (index : Fin snapshotShape.carrierWidth) :
    (config.finalSnapshotSlice.project (encodeCarrier source) index).val =
      sourceNat source
        (MemoryProductUpdateRows.finalSnapshotStart
          (layout.steps soleStep).source.product +
          index.val) := by
  rw [final_snapshot_project_exact configExact source index]
  calc
    (source (laneSourceColumn (finalSnapshotBlockIndex index))).val =
        sourceNat source
          (laneSourceColumn (finalSnapshotBlockIndex index)).val :=
      (sourceNat_sourceColumn source
        (laneSourceColumn (finalSnapshotBlockIndex index))).symm
    _ = sourceNat source
        (MemoryProductUpdateRows.finalSnapshotStart
          (layout.steps soleStep).source.product + index.val) := by
      apply congrArg (sourceNat source)
      simp [MemoryProductUpdateRows.finalSnapshotStart,
        layoutExact.laneBase, laneSourceColumn, finalSnapshotBlockIndex,
        directSourceColumn, finalSnapshotRelativeStart, Nat.add_assoc]

/-- One producer result is authority-bound only when it was derived from the
same exact source assignment that is encoded and committed. -/
structure BoundResult
    {privateWidth : Nat}
    (config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape)
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    (source : SourceAssignment privateWidth)
    (headers : FPrime.ChainHeaders Digest.Value)
    (result : ProductionMemoryCheckedBatchRows.Result layout
      (sourceNat source) headers) : Prop where
  configExact : ProductionFreshLaneGeometry.Exact config
  layoutExact : LayoutExact layout
  paddingZero : PaddingZero source

namespace BoundResult

theorem operations
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    {source : SourceAssignment privateWidth}
    {headers : FPrime.ChainHeaders Digest.Value}
    {result : ProductionMemoryCheckedBatchRows.Result layout
      (sourceNat source) headers}
    (bound : BoundResult config source headers result)
    (index : Fin operationsShape.carrierWidth) :
    (config.operationsSlice.project (encodeCarrier source) index).val =
      sourceNat source
        ((layout.steps soleStep).source.product.laneBase + index.val) :=
  operations_projection_eq_row bound.configExact bound.layoutExact source index

theorem initialSnapshot
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    {source : SourceAssignment privateWidth}
    {headers : FPrime.ChainHeaders Digest.Value}
    {result : ProductionMemoryCheckedBatchRows.Result layout
      (sourceNat source) headers}
    (bound : BoundResult config source headers result)
    (index : Fin snapshotShape.carrierWidth) :
    (config.initialSnapshotSlice.project (encodeCarrier source) index).val =
      sourceNat source
        (MemoryProductUpdateRows.initialSnapshotStart
          (layout.steps soleStep).source.product +
          index.val) :=
  initial_snapshot_projection_eq_row bound.configExact bound.layoutExact source
    index

theorem finalSnapshot
    {privateWidth : Nat}
    {config : ProductCommitmentAlgebra.Config
      (ProductPaperAlgebra.FullShape
        (logicalWidth privateWidth) (publicFits privateWidth))
      operationsShape snapshotShape}
    {layout : ProductionMemoryCheckedBatchRows.Layout .e1}
    {source : SourceAssignment privateWidth}
    {headers : FPrime.ChainHeaders Digest.Value}
    {result : ProductionMemoryCheckedBatchRows.Result layout
      (sourceNat source) headers}
    (bound : BoundResult config source headers result)
    (index : Fin snapshotShape.carrierWidth) :
    (config.finalSnapshotSlice.project (encodeCarrier source) index).val =
      sourceNat source
        (MemoryProductUpdateRows.finalSnapshotStart
          (layout.steps soleStep).source.product +
          index.val) :=
  final_snapshot_projection_eq_row bound.configExact bound.layoutExact source
    index

end BoundResult

end Nightstream.Implementation.NebulaV2.ProductionFreshLaneAuthority
