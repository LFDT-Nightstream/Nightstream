import Nightstream.Implementation.Nebula.Memory.Claim.ProductUpdate
import Nightstream.Implementation.Nebula.Memory.Snapshot.SlotRows

/-!
Contract: exact aggregate snapshot-source relation for one V2 checked step.

Assurance tier: implementation-to-protocol bridge.

Owns all 64 initial-snapshot slots and all 64 final-snapshot slots. It derives
their typed source records from 10,496 rows and proves that both fingerprint
repetitions consume those records in structural slot order.

Does not own product accumulation, cross-step full-scan scheduling, boundary
snapshot continuity, absolute column disjointness, or the generated artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.SnapshotChunkRows

open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.MemoryClaimProductUpdate
open Nightstream.Implementation.Nebula.MemoryProductClaimBridge
open Nightstream.Implementation.Nebula.MemoryProductSemanticBridge
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Implementation.Nebula.SnapshotSlotRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.SnapshotSlot

structure Layout where
  product : MemoryProductUpdateRows.Layout
  aux : SnapshotRole → Fin scanSlots → SnapshotSlotRows.AuxColumns

def Layout.slotLayout (layout : Layout) (role : SnapshotRole)
    (slot : Fin scanSlots) : SnapshotSlotRows.Layout :=
  { product := layout.product
    role := role
    slot := slot
    aux := layout.aux role slot }

def roleRows (layout : Layout) (role : SnapshotRole) : List Row :=
  (List.ofFn fun slot : Fin scanSlots =>
    SnapshotSlotRows.rows (layout.slotLayout role slot)).flatten

def rows (layout : Layout) : List Row :=
  roleRows layout .initialSnapshot ++ roleRows layout .finalSnapshot

private theorem flatten_ofFn_length
    {alpha : Type} {count width : Nat} (blocks : Fin count → List alpha)
    (each : ∀ index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : ∀ value ∈ (List.ofFn blocks).map List.length,
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

theorem roleRows_length (layout : Layout) (role : SnapshotRole) :
    (roleRows layout role).length = 5248 := by
  have exactLength := flatten_ofFn_length (width := 82)
    (fun slot : Fin scanSlots =>
      SnapshotSlotRows.rows (layout.slotLayout role slot))
    (fun slot => SnapshotSlotRows.rows_length_exact
      (layout.slotLayout role slot))
  change (List.ofFn fun slot : Fin scanSlots =>
    SnapshotSlotRows.rows (layout.slotLayout role slot)).flatten.length = 5248
  calc
    _ = scanSlots * 82 := exactLength
    _ = 5248 := by decide

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 10496 := by
  simp [rows, roleRows_length]

private theorem role_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) (role : SnapshotRole) :
    Satisfies (roleRows layout role) assignment := by
  cases role with
  | initialSnapshot =>
      intro row member
      exact holds row (List.mem_append_left _ member)
  | finalSnapshot =>
      intro row member
      exact holds row (List.mem_append_right _ member)

private theorem slot_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) (role : SnapshotRole)
    (slot : Fin scanSlots) :
    Satisfies (SnapshotSlotRows.rows (layout.slotLayout role slot))
      assignment := by
  have group := role_holds holds role
  exact (satisfies_flatten_iff _ _).mp group _
    (List.mem_ofFn.mpr ⟨slot, rfl⟩)

/-- Complete row-derived source meaning for all 128 structural snapshot
slots. The result contains no fingerprint product endpoint. -/
structure Sound
    (layout : Layout) (assignment : Nat → Nat) (claim : Claim) where
  valid : ∀ role slot,
    SnapshotSlot.ValidAt
      (SnapshotSlotRows.decoded (layout.slotLayout role slot) assignment)
      claim.stepIndex.val (SnapshotSlotRows.boundaryValue claim role)

def Sound.records
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (sound : Sound layout assignment claim) (role : SnapshotRole)
    (slot : Fin scanSlots) : BoundedTuple :=
  (sound.valid role slot).boundedTuple slot

theorem sound
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    Sound layout assignment claim := by
  refine ⟨?_⟩
  intro role slot
  exact SnapshotSlotRows.sound canonical one parsed
    (slot_holds holds role slot)

/-- Both repetitions use the same 128 row-derived snapshot records. -/
theorem snapshot_source_refines
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.product.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (derived : Sound layout assignment claim)
    (repetition : Fin 2) (role : SnapshotRole) :
    List.Forall₂ (GateRepresents assignment)
      (layout.product.snapshotChain repetition role).entries
      (snapshotRecords fun slot => derived.records role slot) := by
  simp only [MemoryProductUpdateRows.Layout.snapshotChain,
    MemoryProductUpdateRows.Layout.snapshotEntries, snapshotRecords]
  apply List.forall₂_of_length_eq_of_get
  · simp
  · intro index leftBound _rightBound
    have indexBound : index < scanSlots := by
      simpa using leftBound
    let slot : Fin scanSlots := ⟨index, indexBound⟩
    have represented := SnapshotSlotRows.gate_represents
      (layout := layout.slotLayout role slot) canonical one parsed
      (slot_holds holds role slot) (derived.valid role slot) repetition
    simpa [List.get_ofFn, slot, Layout.slotLayout, Sound.records] using
      represented

end Nightstream.Implementation.Nebula.SnapshotChunkRows
