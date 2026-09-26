import NightstreamFPrime.Export.Stage1.Wide.RunningSourceRows
import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignmentNorm

/-! Norm of copied coordinates and the direct wide witness. The proof ignores
the removed sampler bit blocks and uses the checked shared transition flag. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentNorm

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint ProductionRelation
open PerApplicationCanonicalAssignment PerApplicationAssignmentPlan

private theorem block_valid {program : RetainedLayout.Program} (raw : RawValues program)
    (flag : LowNormSlot.Valid .bit (raw.retainedSource (RunningTransitionReducedRetainedBlocks.flagSource program)))
    (kind : BlockKind) (notReject : kind ≠ .first54Reject) (notPosition : kind ≠ .first54Position) :
    ∀ slot, LowNormSlot.Valid (kind.expand raw).block.kind
      ((kind.expand raw).source ((kind.expand raw).block.source slot)) := by
  cases kind <;> intro slot
  case first54Reject => exact False.elim (notReject rfl)
  case first54Position => exact False.elim (notPosition rfl)
  case runningFlag =>
    simpa only [BlockKind.expand, BlockKind.template, Canonical.ofBlock,
      CanonicalBlockAssignment.ofBlock, RunningTransitionReducedRetainedBlocks.flagBlock] using flag
  case applicationLocal =>
    simp only [BlockKind.expand, BlockKind.template, Canonical.ofBlock,
      CanonicalBlockAssignment.ofBlock, ApplicationRetainedBlocks.localBlock_kind]
    trivial
  all_goals trivial

private theorem region_valid {program : RetainedLayout.Program} (raw : RawValues program)
    (flag : LowNormSlot.Valid .bit (raw.retainedSource (RunningTransitionReducedRetainedBlocks.flagSource program)))
    (kinds : List BlockKind) (notReject : .first54Reject ∉ kinds) (notPosition : .first54Position ∉ kinds) :
    ∀ entry ∈ kinds.map (fun kind => kind.expand raw), ∀ slot, LowNormSlot.Valid entry.block.kind
      (entry.source (entry.block.source slot)) := by
  intro entry member
  obtain ⟨kind, kindMember, rfl⟩ := List.mem_map.mp member
  exact block_valid raw flag kind (fun same => notReject (same ▸ kindMember))
    (fun same => notPosition (same ▸ kindMember))

private theorem split_schedule {program : RetainedLayout.Program} (raw : RawValues program) (count : Nat) :
    raw.schedule =
      (canonicalKinds.take count).map (fun kind => kind.expand raw) ++
      (canonicalKinds.drop count).map (fun kind => kind.expand raw) := by
  rw [← expand_eq_schedule, expand, ← List.map_append, List.take_append_drop]

/-- The common source regions are bounded without any removed sampler-bit premise. -/
theorem copied_norm {program : RetainedLayout.Program} (raw : RawValues program)
    (flag : LowNormSlot.Valid .bit (raw.retainedSource (RunningTransitionReducedRetainedBlocks.flagSource program)))
    (column : Fin (PerApplicationFixedPoint.logicalWidth program))
    (copied : CoordinateRecovery.CommonSource program column.val) :
    centeredMagnitude (raw.assignment column) < 2 := by
  rcases copied with hash | shared | application
  · change centeredMagnitude (CanonicalBlockAssignment.assignment (encodedHashCells raw.outputDigest) raw.schedule column) < 2
    rw [split_schedule raw 3]
    apply CanonicalBlockAssignment.assignment_norm_prefix
    · intro index
      exact encHash_norm (publicFits := PerApplicationFixedPoint.publicFits program) raw.outputDigest index
    · exact region_valid raw flag (canonicalKinds.take 3) (by decide) (by decide)
    · rw [List.map_take]
      change column.val < ProductionAssignment.publicWidth + CanonicalBlockAssignment.coordinateCount (raw.schedule.take 3)
      rw [← PerApplicationCanonicalEncodes.poseidon_prefix_count]
      apply lt_of_lt_of_le hash
      unfold RetainedLayout.hashEnd PiRLCRetainedGeometry.productGroupStart
      rw [LaterPoseidonRetainedBlocks.samplerStart_eq, PiRLCRetainedGeometry.laterPoseidonBlock_coordinateCount]
      omega
  all_goals
    change centeredMagnitude (CanonicalBlockAssignment.assignment (encodedHashCells raw.outputDigest) raw.schedule column) < 2
    rw [split_schedule raw 10]
    apply CanonicalBlockAssignment.assignment_norm_suffix
    · exact region_valid raw flag (canonicalKinds.drop 10) (by decide) (by decide)
    · rw [List.map_take]
      change ProductionAssignment.publicWidth + CanonicalBlockAssignment.coordinateCount (raw.schedule.take 10) ≤ column.val
      rw [← PerApplicationCanonicalEncodes.shared_prefix_count]
      first
      | exact shared.1
      | have lower := application.1
        rw [(RetainedLayout.boundaries program).2.2.2] at lower
        rw [PiRLCRetainedGeometry.prefixLogicalWidth_eq]
        omega

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

theorem flag_valid (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (running : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    LowNormSlot.Valid .bit ((SourceAssignment.raw program env application).retainedSource
      (RunningTransitionReducedRetainedBlocks.flagSource program)) := by
  rw [RawValues.retainedSource, RunningTransitionReducedRetainedSemantics.flagSource_value]
  change PiRLCFirst54DirectPlan.baseEnv program (SourceAssignment.raw program env application).base
    Layout.Stage1.RunningTransitionReducedRows.flagIndex = 0 ∨
    PiRLCFirst54DirectPlan.baseEnv program (SourceAssignment.raw program env application).base
      Layout.Stage1.RunningTransitionReducedRows.flagIndex = 1
  rw [SourceAssignment.raw_source program env application _ (by decide)]
  exact Layout.Stage1.RunningTransitionReducedRows.flag_boolean _
    (RunningSourceRows.reduced_rows relation env running)

/-- Every logical coordinate of the constructed wide witness has magnitude below two. -/
theorem assignment_norm (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (running : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    ∀ column, centeredMagnitude (SourceAssignment.assignment program env application column) < 2 := by
  exact AssignmentProjection.assignment_norm program (SourceAssignment.raw program env application).assignment
    (copied_norm (SourceAssignment.raw program env application) (flag_valid program env application relation running))

end NightstreamFPrime.Export.Stage1.Wide.AssignmentNorm
