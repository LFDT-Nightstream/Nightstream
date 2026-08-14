import Nightstream.Implementation.Nebula.Commitment.Bundle.FieldRows
import Nightstream.Implementation.Nebula.Commitment.Compact.LaneStepRows
import Nightstream.Implementation.Nebula.Memory.Claim.Rows

/-!
Contract: one exact three-lane compact-chain update for a checked V2 claim.

Assurance tier: implementation-to-protocol bridge.

Owns one shared canonical decoder for the complete four-component commitment
bundle, the operations lane update, the initial-snapshot lane update, the
final-snapshot lane update, exact reuse of the verified memory-claim roots,
and the row-derived theorem for all three after-roots.

Does not own full-claim section placement, memory-claim parsing, header-root
initialization, transcript rows, absolute generated columns, or Rust
conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.CompactCheckedStepChainRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.CompactChainHashFrameRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.MemoryWireGeometry
open Nightstream.Implementation.Nebula.MemoryClaimCodec

structure Layout where
  bundleFields : CommitmentBundleFieldRows.Layout
  memoryClaim : MemoryClaimRows.Layout
  operations : CompactLaneStepRows.Layout
  initialSnapshot : CompactLaneStepRows.Layout
  finalSnapshot : CompactLaneStepRows.Layout

def Layout.rootColumn (layout : Layout) (stage : RootStage)
    (role : RootRole) (lane : Fin 4) : Nat :=
  Relabel.column
    (layout.memoryClaim.fieldColumnMap (.root stage role lane))
    CanonicalU64.varCol

/-- Column reuse and local schedule facts only. It contains no placement or
row-satisfaction conclusion. -/
structure Layout.Valid
    (manifest : SeedSchedule.Manifest) (layout : Layout) : Prop where
  operationsValid :
    layout.operations.Valid manifest .operations
  initialSnapshotValid :
    layout.initialSnapshot.Valid manifest .memory
  finalSnapshotValid :
    layout.finalSnapshot.Valid manifest .memory
  operationsCommitmentColumns :
    layout.operations.token.commitmentFieldColumn =
      layout.bundleFields.fieldColumn .operations
  initialSnapshotCommitmentColumns :
    layout.initialSnapshot.token.commitmentFieldColumn =
      layout.bundleFields.fieldColumn .initialSnapshot
  finalSnapshotCommitmentColumns :
    layout.finalSnapshot.token.commitmentFieldColumn =
      layout.bundleFields.fieldColumn .finalSnapshot
  operationsPriorColumns :
    layout.operations.priorDigestColumn =
      layout.rootColumn .seenBefore .operations
  initialSnapshotPriorColumns :
    layout.initialSnapshot.priorDigestColumn =
      layout.rootColumn .seenBefore .initialSnapshot
  finalSnapshotPriorColumns :
    layout.finalSnapshot.priorDigestColumn =
      layout.rootColumn .seenBefore .finalSnapshot
  operationsAfterColumns :
    layout.operations.afterDigestColumn =
      layout.rootColumn .seenAfter .operations
  initialSnapshotAfterColumns :
    layout.initialSnapshot.afterDigestColumn =
      layout.rootColumn .seenAfter .initialSnapshot
  finalSnapshotAfterColumns :
    layout.finalSnapshot.afterDigestColumn =
      layout.rootColumn .seenAfter .finalSnapshot
  operationsIndexColumn :
    layout.operations.linkFrame.indexColumn =
      layout.memoryClaim.counterValueColumn .stepIndex
  initialSnapshotIndexColumn :
    layout.initialSnapshot.linkFrame.indexColumn =
      layout.memoryClaim.counterValueColumn .stepIndex
  finalSnapshotIndexColumn :
    layout.finalSnapshot.linkFrame.indexColumn =
      layout.memoryClaim.counterValueColumn .stepIndex

def rows (manifest : SeedSchedule.Manifest) (layout : Layout) : List Row :=
  CommitmentBundleFieldRows.rows layout.bundleFields ++
    CompactLaneStepRows.rows manifest .operations layout.operations ++
    CompactLaneStepRows.rows manifest .memory layout.initialSnapshot ++
    CompactLaneStepRows.rows manifest .memory layout.finalSnapshot

theorem rows_length_exact
    {manifest : SeedSchedule.Manifest} {layout : Layout}
    (valid : layout.Valid manifest) :
    (rows manifest layout).length = 957423 := by
  simp [rows, CommitmentBundleFieldRows.rows_length_exact,
    CompactLaneStepRows.rows_length_exact valid.operationsValid,
    CompactLaneStepRows.rows_length_exact valid.initialSnapshotValid,
    CompactLaneStepRows.rows_length_exact valid.finalSnapshotValid]

private theorem bundle_rows_hold
    {manifest : SeedSchedule.Manifest}
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies (CommitmentBundleFieldRows.rows layout.bundleFields)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem operations_rows_hold
    {manifest : SeedSchedule.Manifest}
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies
      (CompactLaneStepRows.rows manifest .operations layout.operations)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem initial_snapshot_rows_hold
    {manifest : SeedSchedule.Manifest}
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies
      (CompactLaneStepRows.rows manifest .memory
        layout.initialSnapshot) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem final_snapshot_rows_hold
    {manifest : SeedSchedule.Manifest}
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest layout) assignment) :
    Satisfies
      (CompactLaneStepRows.rows manifest .memory layout.finalSnapshot)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

def selectedRoot (claim : Claim) (stage : RootStage)
    (role : RootRole) : Digest.Value :=
  let roots :=
    match stage with
    | .precommit => claim.dPre
    | .seenBefore => claim.dSeenBefore
    | .seenAfter => claim.dSeenAfter
  match role with
  | .operations => roots.operations
  | .initialSnapshot => roots.initialSnapshot
  | .finalSnapshot => roots.finalSnapshot

private theorem root_placed
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryClaim assignment claim)
    (stage : RootStage) (role : RootRole) :
    DigestPlaced (layout.rootColumn stage role) assignment
      (selectedRoot claim stage role) := by
  intro lane
  have exactField := parsed.fields
    (MemoryClaimFieldRows.Slot.root stage role lane)
  cases stage <;> cases role <;> exact exactField

private theorem component_placed
    {layout : Layout} {assignment : Nat → Nat}
    {bundle : CommitmentBundleCodec.Value}
    (typedFields : ∀ component coordinate,
      assignment (layout.bundleFields.fieldColumn component coordinate) =
        (bundle component coordinate).val)
    (lane : CompactLaneStepRows.Layout) (component : Component)
    (columns : lane.token.commitmentFieldColumn =
      layout.bundleFields.fieldColumn component) :
    CompactTokenRows.CommitmentPlaced lane.token assignment
      (bundle component) := by
  intro coordinate
  change assignment (lane.token.commitmentFieldColumn coordinate) =
    (bundle component coordinate).val
  rw [columns]
  exact typedFields component coordinate

def LaneExact
    (manifest : SeedSchedule.Manifest) (role : CompactCommit.Role)
    (index : Fin claimsPerSegment) (layout : CompactLaneStepRows.Layout)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (commitment : CompactCommit.CommitmentEncoding)
    (prior after : Digest.Value) : Prop :=
  let token := (CompactTokenRows.key manifest).token role commitment
  let leafDigest := CompactLaneStepRows.outputDigest
    layout.leafTrace assignment canonical
  (∀ lane : Fin 4,
      (leafDigest.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.leaf role manifest.profile manifest.plan token) lane.val) ∧
    (∀ lane : Fin 4,
      (after.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.link role index prior leafDigest) lane.val)

/-- The three compact chains use the exact typed commitment fields supplied
by the already parsed full claim. This is the field-native authority form;
it does not require a second authority-bearing bit copy of the bundle. -/
theorem all_lanes_exact_of_typed_fields
    {manifest : SeedSchedule.Manifest}
    {layout : Layout} {assignment : Nat → Nat}
    {bundle : CommitmentBundleCodec.Value} {claim : Claim}
    (valid : layout.Valid manifest)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (typedFields : ∀ component coordinate,
      assignment (layout.bundleFields.fieldColumn component coordinate) =
        (bundle component coordinate).val)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryClaim assignment claim)
    (holds : Satisfies (rows manifest layout) assignment) :
    LaneExact manifest .operations claim.stepIndex layout.operations assignment canonical
        (bundle .operations)
        claim.dSeenBefore.operations claim.dSeenAfter.operations ∧
      LaneExact manifest .memory claim.stepIndex layout.initialSnapshot assignment canonical
        (bundle .initialSnapshot)
        claim.dSeenBefore.initialSnapshot claim.dSeenAfter.initialSnapshot ∧
      LaneExact manifest .memory claim.stepIndex layout.finalSnapshot assignment canonical
        (bundle .finalSnapshot)
        claim.dSeenBefore.finalSnapshot claim.dSeenAfter.finalSnapshot := by
  have operationsCommitment := component_placed typedFields layout.operations
    .operations valid.operationsCommitmentColumns
  have initialCommitment := component_placed typedFields layout.initialSnapshot
    .initialSnapshot valid.initialSnapshotCommitmentColumns
  have finalCommitment := component_placed typedFields layout.finalSnapshot
    .finalSnapshot valid.finalSnapshotCommitmentColumns
  have operationsPrior : DigestPlaced layout.operations.priorDigestColumn
      assignment claim.dSeenBefore.operations := by
    rw [valid.operationsPriorColumns]
    exact root_placed parsed .seenBefore .operations
  have operationsAfter : DigestPlaced layout.operations.afterDigestColumn
      assignment claim.dSeenAfter.operations := by
    rw [valid.operationsAfterColumns]
    exact root_placed parsed .seenAfter .operations
  have initialPrior : DigestPlaced layout.initialSnapshot.priorDigestColumn
      assignment claim.dSeenBefore.initialSnapshot := by
    rw [valid.initialSnapshotPriorColumns]
    exact root_placed parsed .seenBefore .initialSnapshot
  have initialAfter : DigestPlaced layout.initialSnapshot.afterDigestColumn
      assignment claim.dSeenAfter.initialSnapshot := by
    rw [valid.initialSnapshotAfterColumns]
    exact root_placed parsed .seenAfter .initialSnapshot
  have finalPrior : DigestPlaced layout.finalSnapshot.priorDigestColumn
      assignment claim.dSeenBefore.finalSnapshot := by
    rw [valid.finalSnapshotPriorColumns]
    exact root_placed parsed .seenBefore .finalSnapshot
  have finalAfter : DigestPlaced layout.finalSnapshot.afterDigestColumn
      assignment claim.dSeenAfter.finalSnapshot := by
    rw [valid.finalSnapshotAfterColumns]
    exact root_placed parsed .seenAfter .finalSnapshot
  have stepIndexExact :
      assignment (layout.memoryClaim.counterValueColumn .stepIndex) =
        claim.stepIndex.val := by
    exact parsed.counters .stepIndex
  have operationsIndex :
      assignment layout.operations.linkFrame.indexColumn =
        claim.stepIndex.val := by
    rw [valid.operationsIndexColumn]
    exact stepIndexExact
  have initialIndex :
      assignment layout.initialSnapshot.linkFrame.indexColumn =
        claim.stepIndex.val := by
    rw [valid.initialSnapshotIndexColumn]
    exact stepIndexExact
  have finalIndex :
      assignment layout.finalSnapshot.linkFrame.indexColumn =
        claim.stepIndex.val := by
    rw [valid.finalSnapshotIndexColumn]
    exact stepIndexExact
  exact
    ⟨CompactLaneStepRows.after_root_exact valid.operationsValid canonical one
        operationsIndex operationsCommitment operationsPrior operationsAfter
        (operations_rows_hold holds),
      CompactLaneStepRows.after_root_exact valid.initialSnapshotValid canonical
        one initialIndex initialCommitment initialPrior initialAfter
        (initial_snapshot_rows_hold holds),
      CompactLaneStepRows.after_root_exact valid.finalSnapshotValid canonical
        one finalIndex finalCommitment finalPrior finalAfter
        (final_snapshot_rows_hold holds)⟩

/-- All three authority-bearing lane roots are derived from the same verified
bundle and the exact typed before/after roots in the memory claim. -/
theorem all_lanes_exact
    {manifest : SeedSchedule.Manifest}
    {layout : Layout} {assignment : Nat → Nat}
    {bundle : CommitmentBundleCodec.Value} {claim : Claim}
    (valid : layout.Valid manifest)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bundleBitsPlaced : CommitmentBundleFieldRows.BitsPlaced
      layout.bundleFields assignment bundle)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryClaim assignment claim)
    (holds : Satisfies (rows manifest layout) assignment) :
    LaneExact manifest .operations claim.stepIndex layout.operations assignment canonical
        (bundle .operations)
        claim.dSeenBefore.operations claim.dSeenAfter.operations ∧
      LaneExact manifest .memory claim.stepIndex layout.initialSnapshot assignment canonical
        (bundle .initialSnapshot)
        claim.dSeenBefore.initialSnapshot claim.dSeenAfter.initialSnapshot ∧
      LaneExact manifest .memory claim.stepIndex layout.finalSnapshot assignment canonical
        (bundle .finalSnapshot)
        claim.dSeenBefore.finalSnapshot claim.dSeenAfter.finalSnapshot := by
  have typedFields := CommitmentBundleFieldRows.typed_columns_of_bits_and_rows
    canonical one (bundle_rows_hold holds) bundleBitsPlaced
  exact all_lanes_exact_of_typed_fields valid canonical one typedFields parsed
    holds

end Nightstream.Implementation.Nebula.CompactCheckedStepChainRows
