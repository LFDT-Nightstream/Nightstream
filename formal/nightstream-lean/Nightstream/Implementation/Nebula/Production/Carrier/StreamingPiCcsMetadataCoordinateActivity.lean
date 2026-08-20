import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMaps

/-!
Contract: structural activity ranges for the three PiCCS metadata maps.

Assurance tier: model-level serialization geometry.

Owns the proof that a map has no active fields outside its exact claim-frame
range. The proofs use verifier-owned frame positions and do not evaluate the
generated active-field lists.

Does not own generated calls, sampler schedules, physical rows, accumulator
updates, lifecycle selection, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateActivity

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps

/-- The statement-and-fresh map is inactive between claim chunk zero and the
first evaluation chunk. -/
theorem statementFresh_activeFields_empty
    (chunk : Fin claimChunkCount)
    (positive : 0 < chunk.val) (beforeEvaluations : chunk.val < 69) :
    MapKind.statementFresh.activeFields chunk = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro field member
  have selected :=
    (MapKind.mem_activeFields MapKind.statementFresh chunk field).mp member
  have selectedVal := congrArg Fin.val selected
  have fieldBound := field.isLt
  change field.val < 28_672 at fieldBound
  unfold MapKind.claimChunk at selectedVal
  simp only [MapKind.framePosition] at selectedVal
  unfold claimChunkWidth at selectedVal
  split at selectedVal
  · omega
  · split at selectedVal
    · omega
    · split at selectedVal <;> omega

/-- The running-commitment map is inactive after claim chunk 61. -/
theorem runningCommitments_activeFields_empty
    (chunk : Fin claimChunkCount) (afterCommitments : 62 <= chunk.val) :
    MapKind.runningCommitments.activeFields chunk = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro field member
  have selected :=
    (MapKind.mem_activeFields MapKind.runningCommitments chunk field).mp member
  have selectedVal := congrArg Fin.val selected
  have fieldBound := field.isLt
  change field.val < 62_208 at fieldBound
  unfold MapKind.claimChunk at selectedVal
  simp only [MapKind.framePosition] at selectedVal
  unfold claimChunkWidth at selectedVal
  omega

/-- The running-public map is inactive before claim chunk 61. -/
theorem runningPublic_activeFields_empty_of_lt
    (chunk : Fin claimChunkCount) (beforePublic : chunk.val < 61) :
    MapKind.runningPublic.activeFields chunk = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro field member
  have selected :=
    (MapKind.mem_activeFields MapKind.runningPublic chunk field).mp member
  have selectedVal := congrArg Fin.val selected
  unfold MapKind.claimChunk at selectedVal
  simp only [MapKind.framePosition] at selectedVal
  unfold claimChunkWidth at selectedVal
  omega

/-- The running-public map is inactive after claim chunk 69. -/
theorem runningPublic_activeFields_empty_of_gt
    (chunk : Fin claimChunkCount) (afterPublic : 69 < chunk.val) :
    MapKind.runningPublic.activeFields chunk = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro field member
  have selected :=
    (MapKind.mem_activeFields MapKind.runningPublic chunk field).mp member
  have selectedVal := congrArg Fin.val selected
  have fieldBound := field.isLt
  change field.val < 8_640 at fieldBound
  unfold MapKind.claimChunk at selectedVal
  simp only [MapKind.framePosition] at selectedVal
  unfold claimChunkWidth at selectedVal
  omega

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateActivity
