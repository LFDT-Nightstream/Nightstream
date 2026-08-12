import Nightstream.Implementation.NebulaV2.Memory.Operation.PrefixRows
import Nightstream.Implementation.NebulaV2.Memory.Snapshot.ChunkRows

/-!
Contract: complete record-source and product-update relation for one V2
checked step.

Assurance tier: implementation-to-protocol bridge.

Owns the shared product layout, all 63 operation sources, all 128 snapshot
sources, and the inclusion of the eight product chains. From one satisfying
26,736-row block it derives the complete independent `ProductState.update`.

Does not own the 3-by-21 WASM-port refinement, cross-step scan coverage,
segment continuity, absolute column disjointness, or the generated artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemorySourceRows

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry

structure Layout where
  product : MemoryProductUpdateRows.Layout
  countColumn : OperationPrefixRows.CountIndex → Nat
  countBitStart : OperationPrefixRows.CountIndex → Nat
  writeTimestampBitStart : Fin operationSlots → Nat
  operationAux : Fin operationSlots → OperationSlotRows.AuxColumns
  writeTimestampLinked : ∀ slot,
    product.writeTimestamp slot = [((operationAux slot).writeTimestamp, 1)]
  snapshotAux : SnapshotRole → Fin scanSlots →
    SnapshotSlotRows.AuxColumns

def Layout.operation (layout : Layout) : OperationPrefixRows.Layout :=
  { product := layout.product
    countColumn := layout.countColumn
    countBitStart := layout.countBitStart
    writeTimestampBitStart := layout.writeTimestampBitStart
    slotAux := layout.operationAux
    writeTimestampLinked := layout.writeTimestampLinked }

def Layout.snapshot (layout : Layout) : SnapshotChunkRows.Layout :=
  { product := layout.product
    aux := layout.snapshotAux }

/-- Challenge-independent source rows only. -/
def rows (layout : Layout) : List Row :=
  OperationPrefixRows.rows layout.operation ++
    SnapshotChunkRows.rows layout.snapshot

/-- Complete checked-step block: source rows followed by the exact eight
running-product chains. -/
def checkedRows (layout : Layout) : List Row :=
  rows layout ++ MemoryProductUpdateRows.rows layout.product

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 22664 := by
  simp [rows, OperationPrefixRows.rows_length_exact,
    SnapshotChunkRows.rows_length_exact]

theorem checkedRows_length_exact (layout : Layout) :
    (checkedRows layout).length = 26736 := by
  simp [checkedRows, rows_length_exact,
    MemoryProductUpdateRows.rows_length_exact]

private theorem operation_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (OperationPrefixRows.rows layout.operation) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem snapshot_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (SnapshotChunkRows.rows layout.snapshot) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

private theorem source_holds_of_checked
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (checkedRows layout) assignment) :
    Satisfies (rows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem product_holds_of_checked
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (checkedRows layout) assignment) :
    Satisfies (MemoryProductUpdateRows.rows layout.product) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

/-- Complete source meaning for the 63 operation slots and 128 snapshot
slots. No field in this proposition states a product endpoint equality. -/
structure Sound
    (layout : Layout) (assignment : Nat → Nat) (claim : Claim) where
  operation : OperationPrefixRows.Sound layout.operation assignment claim
  snapshot : SnapshotChunkRows.Sound layout.snapshot assignment claim

def Sound.records
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (sound : Sound layout assignment claim) : CheckedStepRecords :=
  { reads := fun slot => sound.operation.records .reads slot
    writes := fun slot => sound.operation.records .writes slot
    initialSnapshot := fun slot =>
      sound.snapshot.records .initialSnapshot slot
    finalSnapshot := fun slot => sound.snapshot.records .finalSnapshot slot }

theorem sound
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    Sound layout assignment claim :=
  { operation := OperationPrefixRows.sound canonical one parsed
      (operation_holds holds)
    snapshot := SnapshotChunkRows.sound canonical one parsed
      (snapshot_holds holds) }

/-- The source rows derive the exact source-only premise used by the
independent product theorem. -/
theorem source_refines
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (derived : Sound layout assignment claim) :
    SourceRefines assignment layout.product derived.records where
  operation := by
    intro repetition role
    cases role with
    | reads =>
      simpa [Sound.records, CheckedStepRecords.operation, Layout.operation] using
        OperationPrefixRows.operation_source_refines canonical one
          (operation_holds holds) derived.operation repetition .reads
    | writes =>
      simpa [Sound.records, CheckedStepRecords.operation, Layout.operation] using
        OperationPrefixRows.operation_source_refines canonical one
          (operation_holds holds) derived.operation repetition .writes
  snapshot := by
    intro repetition role
    cases role with
    | initialSnapshot =>
      simpa [Sound.records, CheckedStepRecords.snapshot, Layout.snapshot] using
        SnapshotChunkRows.snapshot_source_refines canonical one parsed
          (snapshot_holds holds) derived.snapshot repetition .initialSnapshot
    | finalSnapshot =>
      simpa [Sound.records, CheckedStepRecords.snapshot, Layout.snapshot] using
        SnapshotChunkRows.snapshot_source_refines canonical one parsed
          (snapshot_holds holds) derived.snapshot repetition .finalSnapshot

/-- A supplied row-derived source witness and the complete checked block give
the exact independent product-state update. -/
theorem product_update
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (checkedRows layout) assignment)
    (derived : Sound layout assignment claim) :
    mapState claim.productsAfter =
      ProductState.update
        Nightstream.Implementation.NebulaV2.ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore) derived.records.chunk := by
  exact MemoryClaimProductUpdate.claim_product_update canonical one parsed
    (product_holds_of_checked holds) derived.records
    (source_refines canonical one parsed (source_holds_of_checked holds) derived)

/-- One satisfying checked-step block derives the complete update of all
eight products. The premises contain no record-source or endpoint conclusion. -/
theorem checked_step_product_update
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (checkedRows layout) assignment) :
    mapState claim.productsAfter =
      ProductState.update
        Nightstream.Implementation.NebulaV2.ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore)
        (sound canonical one parsed (source_holds_of_checked holds)).records.chunk := by
  let derived := sound canonical one parsed (source_holds_of_checked holds)
  exact product_update canonical one parsed holds derived

end Nightstream.Implementation.NebulaV2.MemorySourceRows
