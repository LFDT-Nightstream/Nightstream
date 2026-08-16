import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMaps

/-!
Contract: exact active-field counts for the production PiCCS metadata maps.

Assurance tier: structural verifier-profile certificate.

Owns the three map-and-chunk counts used by the Rust full and final claim
arms. The proofs reduce the piecewise verifier-owned frame-position maps.

Does not own Rust placement, sampler execution, or lifecycle semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataActiveFieldCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps

def firstChunk : Fin claimChunkCount := ⟨0, by decide⟩

def finalChunk : Fin claimChunkCount := ⟨85, by decide⟩

theorem statementFresh_firstChunk_length :
    (MapKind.statementFresh.activeFields firstChunk).length = 52 := by
  rfl

theorem activeFields_length_eq_card
    (kind : MapKind) (chunk : Fin claimChunkCount) :
    (kind.activeFields chunk).length =
      (Finset.univ.filter fun field : Fin kind.fieldCount =>
        kind.claimChunk field = chunk).card := by
  rfl

def runningFirstEmbedding :
    Fin 589 ↪ Fin MapKind.runningMetadata.fieldCount where
  toFun field := ⟨field.val, by
    change field.val < 61_992
    omega⟩
  inj' := by
    intro left right equal
    apply Fin.ext
    exact congrArg
      (fun value : Fin MapKind.runningMetadata.fieldCount => value.val) equal

theorem runningMetadata_firstChunk_iff
    (field : Fin MapKind.runningMetadata.fieldCount) :
    MapKind.runningMetadata.claimChunk field = firstChunk ↔
      field.val < 589 := by
  constructor
  · intro equal
    have values := congrArg (fun chunk : Fin claimChunkCount => chunk.val) equal
    change
      (if field.val < 54_432 then 435 + field.val
        else 54_867 + (field.val - 54_432)) / 1024 = 0 at values
    split at values
    · rw [Nat.div_eq_zero_iff_lt (by decide)] at values
      omega
    · rw [Nat.div_eq_zero_iff_lt (by decide)] at values
      omega
  · intro bound
    apply Fin.ext
    change
      (if field.val < 54_432 then 435 + field.val
        else 54_867 + (field.val - 54_432)) / 1024 = 0
    split
    · rw [Nat.div_eq_zero_iff_lt (by decide)]
      omega
    · omega

theorem runningMetadata_firstChunk_length :
    (MapKind.runningMetadata.activeFields firstChunk).length = 589 := by
  rw [activeFields_length_eq_card]
  have selected :
      (Finset.univ.filter fun field :
        Fin MapKind.runningMetadata.fieldCount =>
          MapKind.runningMetadata.claimChunk field = firstChunk) =
        Finset.univ.map runningFirstEmbedding := by
    ext field
    rw [Finset.mem_filter]
    simp only [Finset.mem_univ, true_and, Finset.mem_map]
    rw [runningMetadata_firstChunk_iff]
    constructor
    · intro bound
      let source : Fin 589 := ⟨field.val, bound⟩
      exact ⟨source, by
        apply Fin.ext
        rfl⟩
    · rintro ⟨source, equal⟩
      rw [← equal]
      exact source.isLt
  rw [selected, Finset.card_map, Finset.card_univ, Fintype.card_fin]

def statementFreshFinalEmbedding :
    Fin 983 ↪ Fin MapKind.statementFresh.fieldCount where
  toFun field := ⟨24_665 + field.val, by
    change 24_665 + field.val < 25_648
    omega⟩
  inj' := by
    intro left right equal
    apply Fin.ext
    have values := congrArg
      (fun value : Fin MapKind.statementFresh.fieldCount => value.val) equal
    simp only at values
    omega

theorem statementFresh_finalChunk_iff
    (field : Fin MapKind.statementFresh.fieldCount) :
    MapKind.statementFresh.claimChunk field = finalChunk ↔
      24_665 ≤ field.val := by
  constructor
  · intro equal
    have values := congrArg (fun chunk : Fin claimChunkCount => chunk.val) equal
    change
      (if field.val < 52 then 383 + field.val
        else if field.val < 21_220 then 62_427 + (field.val - 52)
        else if field.val < 25_108 then 83_595 + (field.val - 21_220)
        else 87_483 + (field.val - 25_108)) / 1024 = 85 at values
    split at values
    · rw [Nat.div_eq_iff (by decide)] at values
      omega
    · split at values
      · rw [Nat.div_eq_iff (by decide)] at values
        omega
      · split at values
        · rw [Nat.div_eq_iff (by decide)] at values
          omega
        · rw [Nat.div_eq_iff (by decide)] at values
          omega
  · intro lower
    apply Fin.ext
    have fieldLt : field.val < 25_648 := by
      simpa [MapKind.fieldCount] using field.isLt
    change
      (if field.val < 52 then 383 + field.val
        else if field.val < 21_220 then 62_427 + (field.val - 52)
        else if field.val < 25_108 then 83_595 + (field.val - 21_220)
        else 87_483 + (field.val - 25_108)) / 1024 = 85
    split
    · omega
    · split
      · omega
      · split
        · rw [Nat.div_eq_iff (by decide)]
          omega
        · rw [Nat.div_eq_iff (by decide)]
          omega

theorem statementFresh_finalChunk_length :
    (MapKind.statementFresh.activeFields finalChunk).length = 983 := by
  rw [activeFields_length_eq_card]
  have selected :
      (Finset.univ.filter fun field :
        Fin MapKind.statementFresh.fieldCount =>
          MapKind.statementFresh.claimChunk field = finalChunk) =
        Finset.univ.map statementFreshFinalEmbedding := by
    ext field
    rw [Finset.mem_filter]
    simp only [Finset.mem_univ, true_and, Finset.mem_map]
    rw [statementFresh_finalChunk_iff]
    constructor
    · intro lower
      have upper : field.val - 24_665 < 983 := by
        have := field.isLt
        change field.val < 25_648 at this
        omega
      let source : Fin 983 := ⟨field.val - 24_665, upper⟩
      exact ⟨source, by
        apply Fin.ext
        change 24_665 + (field.val - 24_665) = field.val
        omega⟩
    · rintro ⟨source, equal⟩
      have values := congrArg
        (fun value : Fin MapKind.statementFresh.fieldCount => value.val) equal
      change 24_665 + source.val = field.val at values
      omega
  rw [selected, Finset.card_map, Finset.card_univ, Fintype.card_fin]

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataActiveFieldCertificate
