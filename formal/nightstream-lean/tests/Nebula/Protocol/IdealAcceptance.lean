import Nightstream.Protocol.Nebula.IdealAcceptance

set_option autoImplicit false

namespace tests.NebulaIdealAcceptance

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.IdealAcceptance
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.IdealSequence
open Nightstream.Protocol.Nebula.Memory
open Nightstream.Protocol.Nebula.SequenceBinding
open Nightstream.Protocol.Nebula.Soundness

section GenericSoundness

variable {ChallengeField Profile Plan Commitment Digest : Type}
variable [Field ChallengeField]
variable {schema : FullClaim.Schema}
variable {config :
  Config ChallengeField Profile Plan Commitment Digest}
variable {bundleComponent :
  schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
variable {Program ApplicationState : Type}
variable {applicationSemantics :
  ApplicationTrace.Semantics Program ApplicationState}
variable {statement : PublicStatement Program ApplicationState Digest}
variable {verify : FullVerifier schema Digest ChallengeField}

/-- The composed theorem consumes only raw ideal acceptance. It does not take
the old assumed reduction or a prebuilt execution witness. -/
theorem raw_acceptance_has_a_derived_outcome
    (acceptance :
      IdealAcceptV2 config schema bundleComponent verify applicationSemantics
        statement) :
    Failure config ∨ CertifiedExecution acceptance :=
  ideal_acceptance_implies_execution_or_failure acceptance

/-- If an accepted raw segment is unbalanced, the composed segment theorem
must report a failure. It cannot return a semantic segment by assumption. -/
theorem unbalanced_segment_cannot_hide_in_valid_branch
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut)
    (unbalanced : ¬ Balanced initial.tuples accesses final.tuples) :
    Failure config := by
  rcases segment.valid_or_failure with failure | valid
  · exact failure
  · exact False.elim (unbalanced valid.balanced)

theorem checked_fingerprint_points_bind_the_actual_lanes
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex initial
      timestampIn accesses final timestampOut) :
    segment.fingerprint.challenges =
      config.deriveChallenge segmentIndex timestampIn accesses.length
        (config.authoritativeRoots initial accesses final) :=
  segment.fingerprint_challenges_bind_authoritative_lanes

end GenericSoundness

def zeroImage : Fin scannedCells → Nat := fun _ => 0
def oneImage : Fin scannedCells → Nat := fun _ => 1

def zeroSnapshot : Snapshot := Snapshot.ofImage zeroImage
def oneSnapshot : Snapshot := Snapshot.ofImage oneImage
def firstIndex : Fin scannedCells := ⟨0, by decide⟩

theorem zeroSnapshot_ne_oneSnapshot : zeroSnapshot ≠ oneSnapshot := by
  intro equal
  have atZero := congrFun equal firstIndex
  have valueEqual := congrArg CellState.value atZero
  change 0 = 1 at valueEqual
  omega

def constantSnapshotRoot (_snapshot : Snapshot) : Nat := 0

/-- Equal digest values alone do not establish snapshot authority. This is why
the soundness theorem returns an explicit snapshot-root collision branch. -/
theorem constant_root_collision :
    SnapshotRootCollision constantSnapshotRoot :=
  ⟨zeroSnapshot, oneSnapshot, zeroSnapshot_ne_oneSnapshot, rfl⟩

theorem equal_constant_roots_do_not_imply_equal_snapshots :
    constantSnapshotRoot zeroSnapshot = constantSnapshotRoot oneSnapshot ∧
      zeroSnapshot ≠ oneSnapshot :=
  ⟨rfl, zeroSnapshot_ne_oneSnapshot⟩

/-- A detached F-prime precommit and sequence precommit can disagree. The
`SegmentCheck.dPreMatches` field excludes this exact model. -/
theorem detached_precommit_countermodel :
    ∃ fprimeRoot sequenceRoot : Roots Nat, fprimeRoot ≠ sequenceRoot := by
  exact
    ⟨⟨0, 0, 0⟩, ⟨1, 0, 0⟩, by
      intro equal
      have operations := congrArg Roots.operations equal
      change 0 = 1 at operations
      omega⟩

/-- An arbitrary carried challenge need not equal transcript derivation. The
`SegmentCheck.challengeDerived` field excludes this model. -/
theorem detached_challenge_countermodel :
    ∃ (derive : Nat → Nat) (input carried : Nat),
      carried ≠ derive input := by
  exact ⟨fun value => value + 1, 4, 9, by decide⟩

/-- A correctly derived carried challenge does not constrain fingerprint
challenges unless both values are linked. `SegmentCheck.fingerprintChallenges`
excludes this model. -/
theorem detached_fingerprint_challenges_countermodel :
    ∃ (derive : Nat → Nat) (pairs : Nat → Nat) (input carried used : Nat),
      carried = derive input ∧ used ≠ pairs carried := by
  exact ⟨fun value => value + 1, fun value => value + 2, 4, 5, 9,
    rfl, by decide⟩

/-- A lane root can bind a sequence that does not encode the records checked
by the fingerprint. The three `SegmentCheck.*Authority` fields exclude this
model before challenge derivation. -/
theorem detached_lane_records_countermodel :
    ∃ (canonical committed : Nat), committed ≠ canonical :=
  ⟨0, 1, by decide⟩

end tests.NebulaIdealAcceptance
