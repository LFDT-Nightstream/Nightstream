import Nightstream.Implementation.NebulaV2.Memory.Claim.Rows
import Nightstream.Implementation.NebulaV2.Memory.Product.ChainRows
import Nightstream.Implementation.R1CS.Core.LinearSubstitution
import Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry

/-!
Contract: fixed-shape two-repetition product-update rows for one Nebula V2
checked step.

Assurance tier: implementation model.

Owns all eight product chains: reads, writes, initial snapshot, and final
snapshot for both repetitions. It derives the 63-operation-slot and
64-scan-slot shapes from the normative lane geometry and uses the exact parsed
claim columns for challenges and product endpoints.

Does not own the operation write-timestamp prefix-counter rows, typed port
decoding, snapshot timestamp comparisons, frame-column disjointness, absolute
lane placement, honest auxiliary construction, or the generated V2 artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryProductChainRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry

inductive OperationRole where
  | reads
  | writes
deriving DecidableEq, Fintype, Repr

inductive SnapshotRole where
  | initialSnapshot
  | finalSnapshot
deriving DecidableEq, Fintype, Repr

def operationRoles : List OperationRole := [.reads, .writes]
def snapshotRoles : List SnapshotRole := [.initialSnapshot, .finalSnapshot]

structure LeafFrames where
  scale : Frame
  gate : Frame
  update : Frame

structure Frames where
  operation : Fin 2 → OperationRole → Fin operationSlots → LeafFrames
  snapshot : Fin 2 → SnapshotRole → Fin scanSlots → LeafFrames

/-- The generated operation relation supplies the write timestamp for each
physical slot. The product rows consume this linear combination. A later
refinement theorem must prove that it is the global timestamp after all prior
active physical slots plus one. -/
structure Layout where
  claim : MemoryClaimRows.Layout
  laneBase : Nat
  writeTimestamp : Fin operationSlots → LinComb
  frames : Frames

def operationStart (layout : Layout) : Nat := layout.laneBase

def initialSnapshotStart (layout : Layout) : Nat :=
  layout.laneBase + initialSnapshotRelativeStart

def finalSnapshotStart (layout : Layout) : Nat :=
  layout.laneBase + finalSnapshotRelativeStart

def Layout.operationSlotStart (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  operationStart layout + slot.val * operationSlotPayloadWidth

def Layout.snapshotSlotStart (layout : Layout) (role : SnapshotRole)
    (slot : Fin scanSlots) : Nat :=
  (match role with
   | .initialSnapshot => initialSnapshotStart layout
   | .finalSnapshot => finalSnapshotStart layout) +
    slot.val * snapshotSlotPayloadWidth

def bitWordBase (start width : Nat) : LinComb :=
  (List.range width).map fun offset => (start + offset, 2 ^ offset)

/-- Public scaling uses the shared canonical linear-substitution operation.
This makes the row-to-word refinement use the same coefficient-reduction rule
as every other canonical R1CS linear combination. -/
def bitWord (start width coefficient : Nat) : LinComb :=
  LinearSubstitution.scaleTerms coefficient (bitWordBase start width)

def constantWord (value : Nat) : LinComb :=
  if value = 0 then [] else [(0, value % goldilocksP)]

def Layout.claimFieldColumn (layout : Layout)
    (slot : MemoryClaimFieldRows.Slot) : Nat :=
  Relabel.column (layout.claim.fieldColumnMap slot) CanonicalU64.varCol

def Layout.challenge (layout : Layout) (repetition coordinate : Fin 2) :
    Carried :=
  ⟨[(layout.claimFieldColumn (.challenge repetition coordinate 0), 1)],
    [(layout.claimFieldColumn (.challenge repetition coordinate 1), 1)]⟩

def productRole : OperationRole → ProductRole
  | .reads => .reads
  | .writes => .writes

def snapshotProductRole : SnapshotRole → ProductRole
  | .initialSnapshot => .initialSnapshot
  | .finalSnapshot => .finalSnapshot

def Layout.product (layout : Layout) (side repetition : Fin 2)
    (role : ProductRole) : Carried :=
  ⟨[(layout.claimFieldColumn (.product side repetition role 0), 1)],
    [(layout.claimFieldColumn (.product side repetition role 1), 1)]⟩

def Layout.operationPadColumn (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  layout.operationSlotStart slot

def Layout.operationAddressStart (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  layout.operationSlotStart slot + 3

def Layout.operationReadValueStart (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  layout.operationAddressStart slot + addressBits

def Layout.operationWriteValueStart (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  layout.operationReadValueStart slot + ConcreteLaneGeometry.valueBits

def Layout.operationReadTimestampStart (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  layout.operationWriteValueStart slot + ConcreteLaneGeometry.valueBits

def Layout.operationIsRamColumn (layout : Layout)
    (slot : Fin operationSlots) : Nat :=
  layout.operationSlotStart slot + 2

def Layout.operationGlobalIndex (layout : Layout)
    (slot : Fin operationSlots) : LinComb :=
  bitWord (layout.operationAddressStart slot) addressBits 1 ++
    [(layout.operationIsRamColumn slot, romCells)]

def Layout.operationReadPacked (layout : Layout)
    (slot : Fin operationSlots) : LinComb :=
  bitWord (layout.operationReadTimestampStart slot)
      ConcreteLaneGeometry.timestampBits 1 ++
    bitWord (layout.operationAddressStart slot) addressBits timestampLimit ++
    [(layout.operationIsRamColumn slot, timestampLimit * romCells)]

def Layout.operationWritePacked (layout : Layout)
    (slot : Fin operationSlots) : LinComb :=
  layout.writeTimestamp slot ++
    bitWord (layout.operationAddressStart slot) addressBits timestampLimit ++
    [(layout.operationIsRamColumn slot, timestampLimit * romCells)]

def Layout.operationValue (layout : Layout) (role : OperationRole)
    (slot : Fin operationSlots) : LinComb :=
  match role with
  | .reads =>
      bitWord (layout.operationReadValueStart slot)
        ConcreteLaneGeometry.valueBits 1
  | .writes =>
      bitWord (layout.operationWriteValueStart slot)
        ConcreteLaneGeometry.valueBits 1

def Layout.operationPacked (layout : Layout) (role : OperationRole)
    (slot : Fin operationSlots) : LinComb :=
  match role with
  | .reads => layout.operationReadPacked slot
  | .writes => layout.operationWritePacked slot

def operationEntry (layout : Layout) (repetition : Fin 2)
    (role : OperationRole) (slot : Fin operationSlots) : Entry :=
  let frames := layout.frames.operation repetition role slot
  { packed := layout.operationPacked role slot
    value := layout.operationValue role slot
    activation := .padded (layout.operationPadColumn slot)
    scaleFrame := frames.scale
    gateFrame := frames.gate
    updateFrame := frames.update }

def Layout.operationEntries (layout : Layout) (repetition : Fin 2)
    (role : OperationRole) : List Entry :=
  List.ofFn fun slot : Fin operationSlots =>
    operationEntry layout repetition role slot

def Layout.operationChain (layout : Layout) (repetition : Fin 2)
    (role : OperationRole) : MemoryProductChainRows.Layout :=
  { gamma1 := layout.challenge repetition 0
    gamma2 := layout.challenge repetition 1
    initial := layout.product 0 repetition (productRole role)
    final := layout.product 1 repetition (productRole role)
    entries := layout.operationEntries repetition role }

def Layout.snapshotValueStart (layout : Layout) (role : SnapshotRole)
    (slot : Fin scanSlots) : Nat :=
  layout.snapshotSlotStart role slot

def Layout.snapshotTimestampStart (layout : Layout) (role : SnapshotRole)
    (slot : Fin scanSlots) : Nat :=
  layout.snapshotValueStart role slot + ConcreteLaneGeometry.valueBits

def Layout.snapshotGlobalIndex (layout : Layout)
    (slot : Fin scanSlots) : LinComb :=
  [(layout.claim.counterValueColumn .stepIndex, scanSlots)] ++
    constantWord slot.val

def Layout.snapshotPacked (layout : Layout) (role : SnapshotRole)
    (slot : Fin scanSlots) : LinComb :=
  bitWord (layout.snapshotTimestampStart role slot)
      ConcreteLaneGeometry.timestampBits 1 ++
    [(layout.claim.counterValueColumn .stepIndex,
      timestampLimit * scanSlots)] ++
    constantWord (timestampLimit * slot.val)

def Layout.snapshotValue (layout : Layout) (role : SnapshotRole)
    (slot : Fin scanSlots) : LinComb :=
  bitWord (layout.snapshotValueStart role slot)
    ConcreteLaneGeometry.valueBits 1

def snapshotEntry (layout : Layout) (repetition : Fin 2)
    (role : SnapshotRole) (slot : Fin scanSlots) : Entry :=
  let frames := layout.frames.snapshot repetition role slot
  { packed := layout.snapshotPacked role slot
    value := layout.snapshotValue role slot
    activation := .always
    scaleFrame := frames.scale
    gateFrame := frames.gate
    updateFrame := frames.update }

def Layout.snapshotEntries (layout : Layout) (repetition : Fin 2)
    (role : SnapshotRole) : List Entry :=
  List.ofFn fun slot : Fin scanSlots =>
    snapshotEntry layout repetition role slot

def Layout.snapshotChain (layout : Layout) (repetition : Fin 2)
    (role : SnapshotRole) : MemoryProductChainRows.Layout :=
  { gamma1 := layout.challenge repetition 0
    gamma2 := layout.challenge repetition 1
    initial := layout.product 0 repetition (snapshotProductRole role)
    final := layout.product 1 repetition (snapshotProductRole role)
    entries := layout.snapshotEntries repetition role }

def operationChains (layout : Layout) : List MemoryProductChainRows.Layout :=
  (List.ofFn fun repetition : Fin 2 =>
    operationRoles.map fun role => layout.operationChain repetition role).flatten

def snapshotChains (layout : Layout) : List MemoryProductChainRows.Layout :=
  (List.ofFn fun repetition : Fin 2 =>
    snapshotRoles.map fun role => layout.snapshotChain repetition role).flatten

def chains (layout : Layout) : List MemoryProductChainRows.Layout :=
  operationChains layout ++ snapshotChains layout

def rows (layout : Layout) : List Row :=
  (chains layout).flatMap MemoryProductChainRows.rows

theorem operationEntries_length (layout : Layout) (repetition : Fin 2)
    (role : OperationRole) :
    (layout.operationEntries repetition role).length = operationSlots := by
  simp [Layout.operationEntries]

theorem snapshotEntries_length (layout : Layout) (repetition : Fin 2)
    (role : SnapshotRole) :
    (layout.snapshotEntries repetition role).length = scanSlots := by
  simp [Layout.snapshotEntries]

theorem operationChain_rows_length (layout : Layout) (repetition : Fin 2)
    (role : OperationRole) :
    (MemoryProductChainRows.rows
      (layout.operationChain repetition role)).length = 632 := by
  rw [MemoryProductChainRows.rows_length]
  simp [Layout.operationChain, Layout.operationEntries, operationEntry, Entry.rowCount,
    operationSlots]

theorem snapshotChain_rows_length (layout : Layout) (repetition : Fin 2)
    (role : SnapshotRole) :
    (MemoryProductChainRows.rows
      (layout.snapshotChain repetition role)).length = 386 := by
  rw [MemoryProductChainRows.rows_length]
  simp [Layout.snapshotChain, Layout.snapshotEntries, snapshotEntry, Entry.rowCount,
    scanSlots]

/-- Four operation chains cost `4 * 632`; four snapshot chains cost
`4 * 386`. -/
theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 4072 := by
  simp [rows, chains, operationChains, snapshotChains, operationRoles,
    snapshotRoles, operationChain_rows_length, snapshotChain_rows_length]

private theorem one_chain_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    {chain : MemoryProductChainRows.Layout}
    (member : chain ∈ chains layout) :
    Satisfies (MemoryProductChainRows.rows chain) assignment := by
  intro row rowMember
  exact satisfies row (by
    apply List.mem_flatMap.mpr
    exact ⟨chain, member, rowMember⟩)

theorem operationChain_sound
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (repetition : Fin 2) (role : OperationRole) :
    carriedValue assignment
        (layout.operationChain repetition role).final =
      MemoryProductChainRows.productValue assignment
        (layout.operationChain repetition role).gamma1
        (layout.operationChain repetition role).gamma2
        (carriedValue assignment
          (layout.operationChain repetition role).initial)
        (layout.operationChain repetition role).entries := by
  apply MemoryProductChainRows.final_sound one
  apply one_chain_holds satisfies
  simp [chains, operationChains, operationRoles]
  fin_cases repetition <;> cases role <;> simp [operationRoles]

theorem snapshotChain_sound
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (repetition : Fin 2) (role : SnapshotRole) :
    carriedValue assignment
        (layout.snapshotChain repetition role).final =
      MemoryProductChainRows.productValue assignment
        (layout.snapshotChain repetition role).gamma1
        (layout.snapshotChain repetition role).gamma2
        (carriedValue assignment
          (layout.snapshotChain repetition role).initial)
        (layout.snapshotChain repetition role).entries := by
  apply MemoryProductChainRows.final_sound one
  apply one_chain_holds satisfies
  simp [chains, snapshotChains, snapshotRoles]
  fin_cases repetition <;> cases role <;> simp [snapshotRoles]

end Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
