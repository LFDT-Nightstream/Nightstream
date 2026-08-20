import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateActivity
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimCoordinateOverlay
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayRunningScheduleCertificate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayStatementFreshScheduleCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateOverlayArtifact

/-!
Contract: bounded identity and semantic contract for every Rust-emitted
claim-coordinate overlay arm.

Assurance tier: Rust-conformant row-to-semantics certificate.

Owns the exact map order, chunk identity, and named schedule use for all 98
active arms. Six adjacent 16-arm leaves and one 2-arm final remainder cover
the generated list. The final theorem composes this identity with the generic
row refinement.

Does not own phase-state physical links, an accepted assignment, complete
accumulator execution, lifecycle selection, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayContractCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateActivity
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimCoordinateOverlay
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayArtifact

/-- Exact map order emitted for one claim chunk. -/
def expectedMapKinds (chunkIndex : Nat) : List MapKind :=
  if chunkIndex = 0 then
    [.statementFresh, .runningCommitments]
  else if chunkIndex < 61 then
    [.runningCommitments]
  else if chunkIndex = 61 then
    [.runningCommitments, .runningPublic]
  else if chunkIndex < 69 then
    [.runningPublic]
  else if chunkIndex = 69 then
    [.statementFresh, .runningPublic]
  else
    [.statementFresh]

/-- A physical call reuses the one Rust schedule certified for its map. The
field equalities are symbolic projections; they do not compare seed lists. -/
def UsesNamedSchedule (call : CoordinateCall) : Prop :=
  match call.mapKind with
  | .statementFresh =>
      call.chunkSize = statementFreshSchedule.chunkSize /\
        call.seedsByOutput = statementFreshSchedule.seedsByOutput
  | .runningCommitments =>
      call.chunkSize = runningCommitmentsSchedule.chunkSize /\
        call.seedsByOutput = runningCommitmentsSchedule.seedsByOutput
  | .runningPublic =>
      call.chunkSize = runningPublicSchedule.chunkSize /\
        call.seedsByOutput = runningPublicSchedule.seedsByOutput

def CallIdentity (arm : RawActiveArm) (call : CoordinateCall) : Prop :=
  call.chunkIndex = arm.chunkIndex /\ UsesNamedSchedule call

/-- Compact identity used after the generated-data boundary. It excludes all
rows, sampler coefficients, witness data, and active-field lists. -/
def ArmIdentity (arm : RawActiveArm) : Prop :=
  arm.chunkIndex < claimChunkCount /\
    arm.coordinateCalls.map (fun call => call.mapKind) =
      expectedMapKinds arm.chunkIndex /\
    ∀ call, call ∈ arm.coordinateCalls → CallIdentity arm call

private theorem scheduleValid_of_usesNamedSchedule
    (call : CoordinateCall) (uses : UsesNamedSchedule call) :
    call.ScheduleValid := by
  cases kindEq : call.mapKind with
  | statementFresh =>
      have fields :
          call.chunkSize = statementFreshSchedule.chunkSize /\
            call.seedsByOutput = statementFreshSchedule.seedsByOutput := by
        simpa [UsesNamedSchedule, kindEq] using uses
      unfold CoordinateCall.ScheduleValid
      rw [kindEq]
      change
        { chunkSize := call.chunkSize
          seedsByOutput := call.seedsByOutput
          rejectionFuel := 16 } =
        MapKind.statementFresh.expectedSchedule
      rw [fields.1, fields.2]
      change statementFreshSchedule =
        MapKind.statementFresh.expectedSchedule
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStatementFreshScheduleCertificate.schedule_exact
  | runningCommitments =>
      have fields :
          call.chunkSize = runningCommitmentsSchedule.chunkSize /\
            call.seedsByOutput = runningCommitmentsSchedule.seedsByOutput := by
        simpa [UsesNamedSchedule, kindEq] using uses
      unfold CoordinateCall.ScheduleValid
      rw [kindEq]
      change
        { chunkSize := call.chunkSize
          seedsByOutput := call.seedsByOutput
          rejectionFuel := 16 } =
        MapKind.runningCommitments.expectedSchedule
      rw [fields.1, fields.2]
      change runningCommitmentsSchedule =
        MapKind.runningCommitments.expectedSchedule
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRunningScheduleCertificate.commitments_schedule_exact
  | runningPublic =>
      have fields :
          call.chunkSize = runningPublicSchedule.chunkSize /\
            call.seedsByOutput = runningPublicSchedule.seedsByOutput := by
        simpa [UsesNamedSchedule, kindEq] using uses
      unfold CoordinateCall.ScheduleValid
      rw [kindEq]
      change
        { chunkSize := call.chunkSize
          seedsByOutput := call.seedsByOutput
          rejectionFuel := 16 } =
        MapKind.runningPublic.expectedSchedule
      rw [fields.1, fields.2]
      change runningPublicSchedule = MapKind.runningPublic.expectedSchedule
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRunningScheduleCertificate.public_schedule_exact

private theorem mapKind_not_mem_of_callFor_none
    (arm : RawActiveArm) (kind : MapKind)
    (selected : arm.callFor kind = none) :
    kind ∉ arm.coordinateCalls.map (fun call => call.mapKind) := by
  intro member
  rcases List.mem_map.mp member with ⟨call, callMember, rfl⟩
  have rawSelected :
      arm.coordinateCalls.find?
          (fun candidate => candidate.mapKind = call.mapKind) = none := by
    simpa [RawActiveArm.callFor] using selected
  have rejected := (List.find?_eq_none.mp rawSelected) call callMember
  simpa using rejected

private theorem statementFresh_inactive_region
    (chunkIndex : Nat)
    (absent : MapKind.statementFresh ∉ expectedMapKinds chunkIndex) :
    0 < chunkIndex /\ chunkIndex < 69 := by
  unfold expectedMapKinds at absent
  split at absent
  · simp at absent
  · split at absent
    · omega
    · split at absent
      · omega
      · split at absent
        · omega
        · split at absent
          · simp at absent
          · simp at absent

private theorem runningCommitments_inactive_region
    (chunkIndex : Nat)
    (absent : MapKind.runningCommitments ∉ expectedMapKinds chunkIndex) :
    62 <= chunkIndex := by
  unfold expectedMapKinds at absent
  split at absent
  · simp at absent
  · split at absent
    · simp at absent
    · split at absent
      · simp at absent
      · split at absent
        · omega
        · split at absent <;> omega

private theorem runningPublic_inactive_region
    (chunkIndex : Nat)
    (absent : MapKind.runningPublic ∉ expectedMapKinds chunkIndex) :
    chunkIndex < 61 \/ 69 < chunkIndex := by
  unfold expectedMapKinds at absent
  split at absent
  · omega
  · split at absent
    · omega
    · split at absent
      · simp at absent
      · split at absent
        · simp at absent
        · split at absent
          · simp at absent
          · omega

/-- The compact identity implies exactly the contract consumed by the generic
row-refinement theorem. -/
theorem armContract_of_identity
    (arm : RawActiveArm) (identity : ArmIdentity arm) : ArmContract arm := by
  rcases identity with ⟨chunkBound, mapKinds, callIdentities⟩
  constructor
  · intro kind call selected
    have rawSelected :
        arm.coordinateCalls.find?
            (fun candidate => candidate.mapKind = kind) = some call := by
      simpa [RawActiveArm.callFor] using selected
    have member : call ∈ arm.coordinateCalls :=
      List.mem_of_find?_eq_some rawSelected
    have callKind : call.mapKind = kind := by
      have foundKind := List.find?_some rawSelected
      simpa using foundKind
    rcases callIdentities call member with ⟨sameChunk, namedSchedule⟩
    have chunkExact : call.chunk = arm.chunk := by
      apply Fin.ext
      simp [CoordinateCall.chunk, RawActiveArm.chunk, sameChunk]
    exact ⟨member, callKind, chunkExact,
      scheduleValid_of_usesNamedSchedule call namedSchedule⟩
  · intro kind selected
    have absentGenerated := mapKind_not_mem_of_callFor_none arm kind selected
    have absentExpected : kind ∉ expectedMapKinds arm.chunkIndex := by
      rw [← mapKinds]
      exact absentGenerated
    have chunkVal : arm.chunk.val = arm.chunkIndex := by
      simp [RawActiveArm.chunk, Nat.mod_eq_of_lt chunkBound]
    cases kind with
    | statementFresh =>
        have region := statementFresh_inactive_region arm.chunkIndex
          absentExpected
        apply statementFresh_activeFields_empty arm.chunk
        · simpa [chunkVal] using region.1
        · simpa [chunkVal] using region.2
    | runningCommitments =>
        apply runningCommitments_activeFields_empty arm.chunk
        simpa [chunkVal] using
          runningCommitments_inactive_region arm.chunkIndex absentExpected
    | runningPublic =>
        rcases runningPublic_inactive_region arm.chunkIndex absentExpected with
          before | after
        · exact runningPublic_activeFields_empty_of_lt arm.chunk (by
            simpa [chunkVal] using before)
        · exact runningPublic_activeFields_empty_of_gt arm.chunk (by
            simpa [chunkVal] using after)

/-- Recursive tails make all certificate leaves adjacent by construction. -/
def tail16 {alpha : Type} (items : List alpha) : Nat → List alpha
  | 0 => items
  | index + 1 => (tail16 items index).drop 16

def chunk16 {alpha : Type} (items : List alpha) (index : Nat) : List alpha :=
  (tail16 items index).take 16

private theorem partitionStep
    {alpha : Type} (items : List alpha) (index : Nat) :
    tail16 items index = chunk16 items index ++ tail16 items (index + 1) := by
  unfold chunk16
  rw [show tail16 items (index + 1) =
      (tail16 items index).drop 16 by rfl]
  exact (List.take_append_drop 16 (tail16 items index)).symm

/-- Exact coverage follows only from adjacent `take` and `drop` operations.
No generated element can be skipped or used by two index ranges. -/
theorem activeArms_partition_exact :
    activeArms =
      chunk16 activeArms 0 ++
      (chunk16 activeArms 1 ++
      (chunk16 activeArms 2 ++
      (chunk16 activeArms 3 ++
      (chunk16 activeArms 4 ++
      (chunk16 activeArms 5 ++ tail16 activeArms 6))))) := by
  calc
    activeArms = tail16 activeArms 0 := rfl
    _ = chunk16 activeArms 0 ++ tail16 activeArms 1 :=
      partitionStep activeArms 0
    _ = chunk16 activeArms 0 ++
        (chunk16 activeArms 1 ++ tail16 activeArms 2) := by
      rw [partitionStep activeArms 1]
    _ = chunk16 activeArms 0 ++
        (chunk16 activeArms 1 ++
        (chunk16 activeArms 2 ++ tail16 activeArms 3)) := by
      rw [partitionStep activeArms 2]
    _ = chunk16 activeArms 0 ++
        (chunk16 activeArms 1 ++
        (chunk16 activeArms 2 ++
        (chunk16 activeArms 3 ++ tail16 activeArms 4))) := by
      rw [partitionStep activeArms 3]
    _ = chunk16 activeArms 0 ++
        (chunk16 activeArms 1 ++
        (chunk16 activeArms 2 ++
        (chunk16 activeArms 3 ++
        (chunk16 activeArms 4 ++ tail16 activeArms 5)))) := by
      rw [partitionStep activeArms 4]
    _ = chunk16 activeArms 0 ++
        (chunk16 activeArms 1 ++
        (chunk16 activeArms 2 ++
        (chunk16 activeArms 3 ++
        (chunk16 activeArms 4 ++
        (chunk16 activeArms 5 ++ tail16 activeArms 6))))) := by
      rw [partitionStep activeArms 5]

/-- Every bounded leaf has exactly 16 arms. The final remainder has exactly
two arms, so no unbounded tail remains. -/
theorem activeArms_partition_geometry :
    (chunk16 activeArms 0).length = 16 /\
      (chunk16 activeArms 1).length = 16 /\
      (chunk16 activeArms 2).length = 16 /\
      (chunk16 activeArms 3).length = 16 /\
      (chunk16 activeArms 4).length = 16 /\
      (chunk16 activeArms 5).length = 16 /\
      (tail16 activeArms 6).length = 2 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

private theorem identity_leaf0 :
    ∀ arm, arm ∈ chunk16 activeArms 0 → ArmIdentity arm := by
  simp [chunk16, tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

private theorem identity_leaf1 :
    ∀ arm, arm ∈ chunk16 activeArms 1 → ArmIdentity arm := by
  simp [chunk16, tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

private theorem identity_leaf2 :
    ∀ arm, arm ∈ chunk16 activeArms 2 → ArmIdentity arm := by
  simp [chunk16, tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

private theorem identity_leaf3 :
    ∀ arm, arm ∈ chunk16 activeArms 3 → ArmIdentity arm := by
  simp [chunk16, tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

private theorem identity_leaf4 :
    ∀ arm, arm ∈ chunk16 activeArms 4 → ArmIdentity arm := by
  simp [chunk16, tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

private theorem identity_leaf5 :
    ∀ arm, arm ∈ chunk16 activeArms 5 → ArmIdentity arm := by
  simp [chunk16, tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

private theorem identity_remainder :
    ∀ arm, arm ∈ tail16 activeArms 6 → ArmIdentity arm := by
  simp [tail16, activeArms, ArmIdentity, CallIdentity,
    UsesNamedSchedule, expectedMapKinds, claimChunkCount]

/-- Complete compact identity for every Rust-emitted active arm. Downstream
proofs consume this theorem and do not unfold the generated list. -/
theorem generated_arms_identity :
    ∀ arm, arm ∈ activeArms → ArmIdentity arm := by
  intro arm member
  rw [activeArms_partition_exact] at member
  simp only [List.mem_append] at member
  rcases member with member | member | member | member | member | member | member
  · exact identity_leaf0 arm member
  · exact identity_leaf1 arm member
  · exact identity_leaf2 arm member
  · exact identity_leaf3 arm member
  · exact identity_leaf4 arm member
  · exact identity_leaf5 arm member
  · exact identity_remainder arm member

/-- Every generated active arm satisfies the semantic contract required by
the row-refinement theorem. -/
theorem generated_arm_contract
    (arm : RawActiveArm) (member : arm ∈ activeArms) : ArmContract arm :=
  armContract_of_identity arm (generated_arms_identity arm member)

/-- Exact generated rows and verifier-owned frame links imply the
authoritative claim-accumulator step for every emitted active arm. -/
theorem generated_arm_rows_and_frame_link_imply_step
    (layout : StateBases) (arm : RawActiveArm)
    (member : arm ∈ activeArms)
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied layout assignment)
    (linked : arm.FrameLinked frame assignment) :
    Step frame arm.chunk
      (decodedBefore layout assignment) (decodedAfter layout assignment) := by
  exact rows_and_frame_link_imply_step layout arm frame assignment canonical
    one satisfied linked (generated_arm_contract arm member)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayContractCertificate
