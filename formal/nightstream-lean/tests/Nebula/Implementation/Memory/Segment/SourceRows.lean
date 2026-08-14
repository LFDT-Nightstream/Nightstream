import Nightstream.Implementation.Nebula.Memory.Segment.SourceRows
import Nightstream.Implementation.Nebula.FPrime.Manifest.RecursiveSchema

/-! Focused gates for the complete checked-step memory source relation. -/

set_option autoImplicit false

namespace tests.NebulaMemorySourceRows

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.SuperNeo.Concrete

theorem exact_operation_source_count (layout : OperationPrefixRows.Layout) :
    (OperationPrefixRows.rows layout).length = 12168 :=
  OperationPrefixRows.rows_length_exact layout

theorem exact_snapshot_slot_count (layout : SnapshotSlotRows.Layout) :
    (SnapshotSlotRows.rows layout).length = 82 :=
  SnapshotSlotRows.rows_length_exact layout

theorem exact_snapshot_chunk_count (layout : SnapshotChunkRows.Layout) :
    (SnapshotChunkRows.rows layout).length = 10496 :=
  SnapshotChunkRows.rows_length_exact layout

theorem exact_complete_checked_step_count (layout : MemorySourceRows.Layout) :
    (MemorySourceRows.checkedRows layout).length = 26736 :=
  MemorySourceRows.checkedRows_length_exact layout

theorem all_eight_products_come_from_rows
    {layout : MemorySourceRows.Layout} {assignment : Nat → Nat}
    {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (MemorySourceRows.checkedRows layout) assignment) :
    MemoryClaimProductUpdate.mapState claim.productsAfter =
      Nightstream.Protocol.Nebula.ProductState.update
        Nightstream.Implementation.Nebula.ConcreteField.encode
        (MemoryClaimProductUpdate.mapChallenges claim.challenge)
        (MemoryClaimProductUpdate.mapState claim.productsBefore)
        (MemorySourceRows.sound canonical one parsed
          (by
            intro row member
            exact holds row (List.mem_append_left _ member))).records.chunk :=
  MemorySourceRows.checked_step_product_update canonical one parsed holds

theorem manifest_has_complete_checked_step_rows
    {widths : FullClaimEnvelope.CompilerWidths}
    (artifact : RecursiveManifestSchema.Artifact widths) :
    (MemorySourceRows.checkedRows artifact.layouts.memorySource).length =
      26736 :=
  artifact.memoryCheckedStep_rows_length

end tests.NebulaMemorySourceRows
