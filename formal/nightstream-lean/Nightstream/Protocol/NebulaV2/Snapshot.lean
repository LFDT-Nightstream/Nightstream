import Mathlib.Data.List.FinRange
import Nightstream.Protocol.NebulaV2.Memory

/-!
Contract: canonical full-memory snapshots for PaddedRowIdentityMemoryV2.

Assurance tier: model-level.

Owns the function-valued memory image and its canonical conversion to one
typed tuple for each structural global index.

Does not own commitment chunks, chain roots, circuit extraction, or hashes.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2

/-- The private state of one scanned cell at a segment boundary. -/
structure CellState where
  value : Nat
  lastTimestamp : Nat
deriving DecidableEq, Repr

/-- A complete snapshot has one cell state for every structural global index.
Using a function makes omission and duplication unrepresentable. -/
abbrev Snapshot := Fin scannedCells → CellState

namespace Snapshot

/-- Canonical verifier-owned initial memory image. -/
def ofImage (image : Fin scannedCells → Nat) : Snapshot :=
  fun index => ⟨image index, 0⟩

def ImageInRange (image : Fin scannedCells → Nat) : Prop :=
  ∀ index, image index < valueLimit

/-- Boundary values are canonical and no cell claims a future timestamp. -/
def ValidAt (snapshot : Snapshot) (timestamp : Nat) : Prop :=
  ∀ index,
    (snapshot index).value < valueLimit ∧
      (snapshot index).lastTimestamp ≤ timestamp

def tupleAt (snapshot : Snapshot) (index : Fin scannedCells) : MemTuple :=
  { timestamp := (snapshot index).lastTimestamp
    globalIndex := index
    value := (snapshot index).value }

/-- Exact increasing-index scan order. -/
def tupleList (snapshot : Snapshot) : List MemTuple :=
  List.ofFn (tupleAt snapshot)

def tuples (snapshot : Snapshot) : Multiset MemTuple :=
  tupleList snapshot

theorem tupleList_length (snapshot : Snapshot) :
    snapshot.tupleList.length = scannedCells := by
  change (List.ofFn (tupleAt snapshot)).length = scannedCells
  exact List.length_ofFn

/-- The structural index projection is exactly the full canonical index list.
This is the exact-cover statement used by the semantic model. -/
theorem tupleList_indices (snapshot : Snapshot) :
    snapshot.tupleList.map MemTuple.globalIndex =
      List.ofFn (fun index : Fin scannedCells => index.val) := by
  change
    (List.ofFn (tupleAt snapshot)).map MemTuple.globalIndex =
      List.ofFn (fun index : Fin scannedCells => index.val)
  rw [List.map_ofFn]
  rfl

theorem tupleList_indices_nodup (snapshot : Snapshot) :
    (snapshot.tupleList.map MemTuple.globalIndex).Nodup := by
  rw [snapshot.tupleList_indices]
  exact List.nodup_ofFn.mpr Fin.val_injective

theorem tupleAt_mem (snapshot : Snapshot) (index : Fin scannedCells) :
    snapshot.tupleAt index ∈ snapshot.tuples := by
  change snapshot.tupleAt index ∈
    (List.ofFn (tupleAt snapshot) : Multiset MemTuple)
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

theorem tuple_mem_validAt
    {snapshot : Snapshot}
    {timestamp : Nat}
    (valid : snapshot.ValidAt timestamp)
    {entry : MemTuple}
    (member : entry ∈ snapshot.tuples) :
    entry.value < valueLimit ∧ entry.timestamp ≤ timestamp := by
  have listMember : entry ∈ List.ofFn (tupleAt snapshot) := member
  rcases List.mem_ofFn.mp listMember with ⟨index, rfl⟩
  exact valid index

theorem tuple_mem_has_bounded_index
    (snapshot : Snapshot)
    {entry : MemTuple}
    (member : entry ∈ snapshot.tuples) :
    entry.globalIndex < scannedCells := by
  have listMember : entry ∈ List.ofFn (tupleAt snapshot) := member
  rcases List.mem_ofFn.mp listMember with ⟨index, rfl⟩
  exact index.isLt

theorem tupleAt_injective (snapshot : Snapshot) :
    Function.Injective snapshot.tupleAt := by
  intro left right equal
  apply Fin.ext
  exact congrArg MemTuple.globalIndex equal

theorem tupleList_nodup (snapshot : Snapshot) :
    snapshot.tupleList.Nodup := by
  exact List.nodup_ofFn.mpr snapshot.tupleAt_injective

theorem ofImage_validAt_zero
    {image : Fin scannedCells → Nat}
    (imageInRange : ImageInRange image) :
    ValidAt (ofImage image) 0 := by
  intro index
  exact ⟨imageInRange index, Nat.le_refl 0⟩

theorem ofImage_injective : Function.Injective ofImage := by
  intro left right equal
  funext index
  have cellEqual := congrFun equal index
  exact congrArg CellState.value cellEqual

end Snapshot

end Nightstream.Protocol.NebulaV2
