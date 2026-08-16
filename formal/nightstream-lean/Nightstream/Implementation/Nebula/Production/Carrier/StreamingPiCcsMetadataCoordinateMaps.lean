import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingClaimSchedule
import Nightstream.Implementation.R1CS.Core.SeededAjtai
import Nightstream.Implementation.R1CS.Core.SeededPhi81

/-!
Contract: verifier-owned geometry for the two PiCCS metadata coordinate maps.

Assurance tier: model-level serialization and setup profile.

Owns the two field counts, frame-position maps, claim-chunk partitions,
rank-two message widths, fixed Rust seeds, and domain identifiers. Together,
the maps bind the statement, fresh instance, running commitments, running
public inputs, prior point, and running evaluations.

Does not own generated rows, sampler conformance, additive accumulator rows,
terminal equality, Module-SIS hardness, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

inductive MapKind where
  | statementFresh
  | runningMetadata
deriving DecidableEq, Inhabited, Repr

def MapKind.fieldCount : MapKind → Nat
  | .statementFresh => 25_648
  | .runningMetadata => 61_992

def MapKind.messageColumnCount (kind : MapKind) : Nat :=
  (kind.fieldCount * digitCount + 54 - 1) / 54

def MapKind.seedByte : MapKind → Nat
  | .statementFresh => 0xC8
  | .runningMetadata => 0xCA

def MapKind.rustSeedBytes (kind : MapKind) : List Nat :=
  List.replicate 32 kind.seedByte

def MapKind.rustDomain : MapKind → Nat
  | .statementFresh => 0x5049_4356_4152_4244
  | .runningMetadata => 0x5049_4352_554E_4D44

def MapKind.expectedSchedule (kind : MapKind) : SeededPhi81.SeedSchedule :=
  SeededAjtai.schedule kind.rustSeedBytes 2 kind.messageColumnCount 16

/-- Canonical placement-free block used only to certify sampler liveness and
shape for one verifier-owned map profile. Physical Rust blocks receive this
certificate through `SeededPhi81.Block.Valid.transfer`. -/
def MapKind.certificateBlock (kind : MapKind) : SeededPhi81.Block where
  rowStart := 0
  wordStarts := List.replicate kind.fieldCount 0
  wordWidth := digitCount
  kappa := 2
  messageCols := kind.messageColumnCount
  outputColumns := List.range 108
  superneoTransformedColumns := false
  schedule := kind.expectedSchedule

theorem MapKind.certificateBlock_exact_geometry (kind : MapKind) :
    kind.certificateBlock.wordStarts.length = kind.fieldCount /\
      kind.certificateBlock.wordWidth = 41 /\
      kind.certificateBlock.kappa = 2 /\
      kind.certificateBlock.messageCols = kind.messageColumnCount /\
      kind.certificateBlock.outputColumns.length = 108 /\
      kind.certificateBlock.schedule = kind.expectedSchedule := by
  simp [MapKind.certificateBlock, digitCount]

theorem exact_map_geometry :
    MapKind.statementFresh.fieldCount = 25_648 /\
      MapKind.statementFresh.messageColumnCount = 19_474 /\
      MapKind.runningMetadata.fieldCount = 61_992 /\
      MapKind.runningMetadata.messageColumnCount = 47_068 /\
      MapKind.statementFresh.fieldCount +
        MapKind.runningMetadata.fieldCount = 87_640 /\
      MapKind.statementFresh.messageColumnCount ≤ 50_371 /\
      MapKind.runningMetadata.messageColumnCount ≤ 50_371 := by
  decide

theorem exact_rust_identities :
    MapKind.statementFresh.rustSeedBytes = List.replicate 32 200 /\
      MapKind.statementFresh.rustDomain = 5785229234076271172 /\
      MapKind.runningMetadata.rustSeedBytes = List.replicate 32 202 /\
      MapKind.runningMetadata.rustDomain = 5785229217231686980 := by
  decide

/-- Position of one map field in the authoritative 88,023-field claim
frame. The order within each map is the fixed coordinate order used by the
seeded rank-two binding. -/
def MapKind.framePosition :
    (kind : MapKind) → Fin kind.fieldCount → Nat
  | .statementFresh, field =>
      if field.val < 52 then
        383 + field.val
      else if field.val < 21_220 then
        62_427 + (field.val - 52)
      else if field.val < 25_108 then
        83_595 + (field.val - 21_220)
      else
        87_483 + (field.val - 25_108)
  | .runningMetadata, field =>
      if field.val < 54_432 then
        435 + field.val
      else
        54_867 + (field.val - 54_432)

theorem MapKind.framePosition_lt
    (kind : MapKind) (field : Fin kind.fieldCount) :
    kind.framePosition field < claimFrameLength := by
  cases kind with
  | statementFresh =>
      have bound := field.isLt
      change field.val < 25_648 at bound
      simp only [MapKind.framePosition]
      unfold claimFrameLength
      split
      · omega
      · split
        · omega
        · split <;> omega
  | runningMetadata =>
      have bound := field.isLt
      change field.val < 61_992 at bound
      simp only [MapKind.framePosition]
      unfold claimFrameLength
      split <;> omega

def MapKind.claimChunk
    (kind : MapKind) (field : Fin kind.fieldCount) : Fin claimChunkCount :=
  ⟨kind.framePosition field / claimChunkWidth, by
    have bound := kind.framePosition_lt field
    unfold claimFrameLength at bound
    unfold claimChunkWidth claimChunkCount
    omega⟩

def MapKind.claimChunkOffset
    (kind : MapKind) (field : Fin kind.fieldCount) : Fin claimChunkWidth :=
  ⟨kind.framePosition field % claimChunkWidth, Nat.mod_lt _ (by decide)⟩

theorem MapKind.framePosition_recompose
    (kind : MapKind) (field : Fin kind.fieldCount) :
    (kind.claimChunk field).val * claimChunkWidth +
        (kind.claimChunkOffset field).val =
      kind.framePosition field := by
  unfold MapKind.claimChunk MapKind.claimChunkOffset
  simpa [Nat.mul_comm] using
    Nat.div_add_mod (kind.framePosition field) claimChunkWidth

def MapKind.activeFields
    (kind : MapKind) (chunk : Fin claimChunkCount) :
    List (Fin kind.fieldCount) :=
  (List.finRange kind.fieldCount).filter fun field =>
    kind.claimChunk field = chunk

theorem MapKind.activeFields_nodup
    (kind : MapKind) (chunk : Fin claimChunkCount) :
    (kind.activeFields chunk).Nodup := by
  exact (List.nodup_finRange kind.fieldCount).filter _

@[simp] theorem MapKind.mem_activeFields
    (kind : MapKind) (chunk : Fin claimChunkCount)
    (field : Fin kind.fieldCount) :
    field ∈ kind.activeFields chunk ↔ kind.claimChunk field = chunk := by
  simp [MapKind.activeFields]

def claimChunkFieldCount (chunk : Fin claimChunkCount) : Nat :=
  if chunk.val = 85 then 983 else 1024

theorem claimChunkFieldCount_le (chunk : Fin claimChunkCount) :
    claimChunkFieldCount chunk ≤ claimChunkWidth := by
  unfold claimChunkFieldCount claimChunkWidth
  split <;> omega

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
