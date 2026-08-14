import Nightstream.Implementation.Nebula.Memory.Claim.Codec
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-!
Contract: lossless frame and concrete Poseidon2 digest for one complete V2
memory suffix.

The frame contains every tagged memory-claim field in the exact codec order.
It is injective before hashing. Equal four-lane digests therefore give equal
canonical claims or one explicit Poseidon2 collision event.

This file owns value semantics only. Generated digest rows and absolute
manifest placement are separate modules.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryClaimPoseidonBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Protocol.Nebula

abbrev Claim := MemoryClaimCodec.Claim
abbrev Digest := Fin 4 → Nat

def domainTag : Nat := 0x4e534d51
def frameVersion : Nat := 1

def profileFields : List Nat := [2, 2, 1, 1]

def fixedPrefix : List Nat :=
  [domainTag, frameVersion] ++ profileFields ++
    [MemoryWireGeometry.stepPublicBits, MemoryClaimCodec.schema.length]

theorem schema_length_exact : MemoryClaimCodec.schema.length = 83 := by
  decide

theorem fixedPrefix_exact :
    fixedPrefix = [0x4e534d51, 1, 2, 2, 1, 1, 4980, 83] := by
  decide

def claimFields (claim : Claim) : List Nat :=
  MemoryClaimCodec.schema.map claim.fieldValue

theorem claimFields_length (claim : Claim) :
    (claimFields claim).length = 83 := by
  simp [claimFields, schema_length_exact]

private theorem mapped_value_eq_at
    {Alpha Beta : Type} {tags : List Alpha} {left right : Alpha → Beta}
    (equal : tags.map left = tags.map right)
    {tag : Alpha} (member : tag ∈ tags) : left tag = right tag := by
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq] at equal
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact equal.1
      · exact inductionHypothesis equal.2 tailMember

theorem claimFields_injective : Function.Injective claimFields := by
  intro left right equal
  apply MemoryClaimCodec.Claim.fieldValue_injective
  funext tag
  exact mapped_value_eq_at equal tag.mem_schema

/-- Exact 91-field, domain-separated memory-claim frame. -/
def frame (claim : Claim) : List Nat := fixedPrefix ++ claimFields claim

theorem frame_length (claim : Claim) : (frame claim).length = 91 := by
  simp [frame, fixedPrefix_exact, claimFields_length]

theorem frame_injective : Function.Injective frame := by
  intro left right equal
  apply claimFields_injective
  have tails := congrArg (List.drop fixedPrefix.length) equal
  simpa [frame] using tails

private theorem fieldValue_canonical
    {claim : Claim} (canonical : claim.Canonical)
    (tag : MemoryClaimCodec.FieldTag) :
    claim.fieldValue tag < goldilocksP := by
  cases tag with
  | segmentIndex =>
      exact canonical.segmentIndex.trans (by decide)
  | stepIndex =>
      exact claim.stepIndex.isLt.trans (by decide)
  | timestampIn =>
      exact canonical.timestampIn.trans (by decide)
  | timestampOut =>
      exact canonical.timestampOut.trans (by decide)
  | segmentStartTimestamp =>
      exact canonical.segmentStartTimestamp.trans (by decide)
  | segmentEndTimestamp =>
      exact canonical.segmentEndTimestamp.trans (by decide)
  | activeAccessCount =>
      exact canonical.activeAccessCount.trans (by decide)
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;>
        simp only [Claim.fieldValue, challengeValue, kLimbValue] <;>
        exact Fin.isLt _
  | product side repetition role limb =>
      fin_cases side <;> cases role <;> fin_cases limb <;>
        simp only [Claim.fieldValue, productValue, kLimbValue, if_pos,
          if_neg] <;>
        exact Fin.isLt _
  | root stage role lane =>
      cases stage <;> cases role <;>
        simp only [Claim.fieldValue, rootValue] <;>
        first
        | exact (claim.dPre.operations.lanes lane).property
        | exact (claim.dPre.initialSnapshot.lanes lane).property
        | exact (claim.dPre.finalSnapshot.lanes lane).property
        | exact (claim.dSeenBefore.operations.lanes lane).property
        | exact (claim.dSeenBefore.initialSnapshot.lanes lane).property
        | exact (claim.dSeenBefore.finalSnapshot.lanes lane).property
        | exact (claim.dSeenAfter.operations.lanes lane).property
        | exact (claim.dSeenAfter.initialSnapshot.lanes lane).property
        | exact (claim.dSeenAfter.finalSnapshot.lanes lane).property

theorem frame_fields_canonical
    {claim : Claim} (canonical : claim.Canonical) :
    ∀ value ∈ frame claim, value < goldilocksP := by
  intro value member
  rw [frame, List.mem_append] at member
  rcases member with fixed | claimField
  · rw [fixedPrefix_exact] at fixed
    simp only [List.mem_cons, List.not_mem_nil, or_false] at fixed
    rcases fixed with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      decide
  · rcases List.mem_map.mp claimField with ⟨tag, _member, rfl⟩
    exact fieldValue_canonical canonical tag

/-- Ninety-one fields use 22 full absorbs, one three-field absorb, and one
terminal pad permutation. -/
def expectedSchedule : List ValueSchedule :=
  List.replicate 22 (.absorb 4) ++ [.absorb 3, .pad]

theorem expectedSchedule_exact :
    expectedSchedule.length = 24 ∧
      (expectedSchedule.filter (· = .absorb 4)).length = 22 ∧
      (expectedSchedule.filter (· = .absorb 3)).length = 1 ∧
      (expectedSchedule.filter (· = .pad)).length = 1 := by
  decide

def representativeRound : ValueSchedule → Round
  | .absorb count =>
      { (default : Round) with kind := .absorb (List.replicate count 0) }
  | .pad => { (default : Round) with kind := .pad }

theorem representativeRound_schedule (schedule : ValueSchedule) :
    (representativeRound schedule).valueSchedule = schedule := by
  cases schedule <;> simp [representativeRound, Round.valueSchedule]

def representativeRounds : List Round :=
  expectedSchedule.map representativeRound

theorem representativeRounds_schedule :
    valueSchedules representativeRounds = expectedSchedule := by
  rw [representativeRounds, valueSchedules, List.map_map]
  change expectedSchedule.map
      (fun schedule => (representativeRound schedule).valueSchedule) =
    expectedSchedule
  generalize expectedSchedule = schedules
  induction schedules with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq]
      exact ⟨representativeRound_schedule head, inductionHypothesis⟩

def pureDigest (values : List Nat) (lane : Nat) : Nat :=
  runValueRounds representativeRounds values (fun _ => 0) lane

def digest (claim : Claim) : Digest :=
  fun lane => pureDigest (frame claim) lane.val

theorem digest_canonical (claim : Claim) (lane : Fin 4) :
    digest claim lane < goldilocksP := by
  exact runValueRounds_canonical _ _ _
    (fun _ => by norm_num [goldilocksP]) lane.val

def canonicalDigest (claim : Claim) : Digest.Value where
  lanes := fun lane => ⟨digest claim lane, digest_canonical claim lane⟩

structure CanonicalClaim where
  claim : Claim
  canonical : claim.Canonical

def PoseidonCollision : Prop :=
  ∃ left right : CanonicalClaim,
    frame left.claim ≠ frame right.claim ∧
      digest left.claim = digest right.claim

/-- Equal concrete digests recover the full typed suffix or expose the exact
framed Poseidon2 collision that the security proof must price. -/
theorem claim_eq_or_poseidon_collision
    {left right : Claim}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : digest left = digest right) :
    left = right ∨ PoseidonCollision := by
  classical
  by_cases same : left = right
  · exact Or.inl same
  · have frameDifferent : frame left ≠ frame right :=
      fun frameEqual => same (frame_injective frameEqual)
    exact Or.inr
      ⟨⟨left, leftCanonical⟩, ⟨right, rightCanonical⟩,
        frameDifferent, equal⟩

end Nightstream.Implementation.Nebula.MemoryClaimPoseidonBinding
