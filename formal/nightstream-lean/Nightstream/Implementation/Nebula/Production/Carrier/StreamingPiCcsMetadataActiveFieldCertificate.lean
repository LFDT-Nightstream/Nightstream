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

def finalChunk : Fin claimChunkCount := ⟨97, by decide⟩

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
    Fin 589 ↪ Fin MapKind.runningCommitments.fieldCount where
  toFun field := ⟨field.val, by
    change field.val < 62_208
    omega⟩
  inj' := by
    intro left right equal
    apply Fin.ext
    exact congrArg
      (fun value : Fin MapKind.runningCommitments.fieldCount => value.val) equal

theorem runningCommitments_firstChunk_iff
    (field : Fin MapKind.runningCommitments.fieldCount) :
    MapKind.runningCommitments.claimChunk field = firstChunk ↔
      field.val < 589 := by
  constructor
  · intro equal
    have values := congrArg (fun chunk : Fin claimChunkCount => chunk.val) equal
    change
      (435 + field.val) / 1024 = 0 at values
    rw [Nat.div_eq_zero_iff_lt (by decide)] at values
    omega
  · intro bound
    apply Fin.ext
    change
      (435 + field.val) / 1024 = 0
    rw [Nat.div_eq_zero_iff_lt (by decide)]
    omega

theorem runningCommitments_firstChunk_length :
    (MapKind.runningCommitments.activeFields firstChunk).length = 589 := by
  rw [activeFields_length_eq_card]
  have selected :
      (Finset.univ.filter fun field :
        Fin MapKind.runningCommitments.fieldCount =>
          MapKind.runningCommitments.claimChunk field = firstChunk) =
        Finset.univ.map runningFirstEmbedding := by
    ext field
    rw [Finset.mem_filter]
    simp only [Finset.mem_univ, true_and, Finset.mem_map]
    rw [runningCommitments_firstChunk_iff]
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
    Fin 575 ↪ Fin MapKind.statementFresh.fieldCount where
  toFun field := ⟨28_097 + field.val, by
    change 28_097 + field.val < 28_672
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
      28_097 ≤ field.val := by
  constructor
  · intro equal
    have values := congrArg (fun chunk : Fin claimChunkCount => chunk.val) equal
    change
      (if field.val < 52 then 383 + field.val
        else if field.val < 24_244 then 71_283 + (field.val - 52)
        else if field.val < 28_132 then 95_475 + (field.val - 24_244)
        else 99_363 + (field.val - 28_132)) / 1024 = 97 at values
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
    have fieldLt : field.val < 28_672 := by
      simpa [MapKind.fieldCount] using field.isLt
    change
      (if field.val < 52 then 383 + field.val
        else if field.val < 24_244 then 71_283 + (field.val - 52)
        else if field.val < 28_132 then 95_475 + (field.val - 24_244)
        else 99_363 + (field.val - 28_132)) / 1024 = 97
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
    (MapKind.statementFresh.activeFields finalChunk).length = 575 := by
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
      have upper : field.val - 28_097 < 575 := by
        have := field.isLt
        change field.val < 28_672 at this
        omega
      let source : Fin 575 := ⟨field.val - 28_097, upper⟩
      exact ⟨source, by
        apply Fin.ext
        change 28_097 + (field.val - 28_097) = field.val
        omega⟩
    · rintro ⟨source, equal⟩
      have values := congrArg
        (fun value : Fin MapKind.statementFresh.fieldCount => value.val) equal
      change 28_097 + source.val = field.val at values
      omega
  rw [selected, Finset.card_map, Finset.card_univ, Fintype.card_fin]

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataActiveFieldCertificate
