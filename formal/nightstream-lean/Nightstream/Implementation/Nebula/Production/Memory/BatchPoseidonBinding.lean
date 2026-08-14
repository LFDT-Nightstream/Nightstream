import Nightstream.Implementation.Nebula.Memory.Claim.PoseidonBinding
import Nightstream.Implementation.Nebula.Application.Wasm.ResultCodec
import Nightstream.Protocol.Nebula.ProductionBatchedFPrime

/-!
Contract: lossless frame and concrete Poseidon2 digest for one complete
candidate-specific memory-suffix batch.

The frame includes the successor profile identity and every field of all `E`
ordered memory suffixes. Equal digests give equal canonical batches or one
explicit framed Poseidon2 collision.

Does not own generated sponge rows, CCS public placement, collision security,
Rust conformance, candidate selection, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding

open Nightstream.Implementation.Nebula.MemoryClaimPoseidonBinding
open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

abbrev Claim := MemoryClaimCodec.Claim
abbrev Batch (candidate : Id) :=
  SuffixBatch candidate Digest.Value (ProductState.Challenges K)
    (ProductState.State K)
abbrev BatchDigest := Fin 4 -> Nat

/-- ASCII `NSMB`, encoded as one Goldilocks field. -/
def domainTag : Nat := 0x4e534d42
def frameVersion : Nat := 1

def profileFields (candidate : Id) : List Nat :=
  [3, version candidate, checkedStepsPerFreshClaim candidate, 1]

def fixedPrefix (candidate : Id) : List Nat :=
  [domainTag, frameVersion] ++ profileFields candidate ++
    [MemoryWireGeometry.stepPublicBits, MemoryClaimCodec.schema.length]

theorem fixedPrefix_length (candidate : Id) :
    (fixedPrefix candidate).length = 8 := rfl

theorem fixedPrefix_injective : Function.Injective fixedPrefix := by
  intro left right equal
  have selected := congrArg (fun values : List Nat => (values.drop 2).take 4)
    equal
  have profiles : profileFields left = profileFields right := by
    simpa [fixedPrefix, profileFields] using selected
  cases left <;> cases right <;>
    simp [profileFields, version, checkedStepsPerFreshClaim] at profiles ⊢

def claimBlocks {candidate : Id} (batch : Batch candidate) :
    List (List Nat) :=
  batch.suffixes.map claimFields

theorem claimBlocks_lengths
    {candidate : Id} (batch : Batch candidate) :
    (claimBlocks batch).map List.length =
      List.replicate (checkedStepsPerFreshClaim candidate) 83 := by
  apply List.eq_replicate_iff.mpr
  constructor
  · simp [claimBlocks, batch.length_exact]
  · intro width member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_map.mp blockMember with ⟨claim, _claimMember, rfl⟩
    exact claimFields_length claim

def batchFields {candidate : Id} (batch : Batch candidate) : List Nat :=
  (claimBlocks batch).flatten

theorem batchFields_length
    {candidate : Id} (batch : Batch candidate) :
    (batchFields batch).length =
      checkedStepsPerFreshClaim candidate * 83 := by
  rw [batchFields, List.length_flatten, claimBlocks_lengths]
  simp

theorem batchFields_injective
    (candidate : Id) : Function.Injective
      (batchFields : Batch candidate -> List Nat) := by
  intro left right equal
  have blocksEqual : claimBlocks left = claimBlocks right :=
    WasmResultCodec.flatten_injective_of_lengths
      (claimBlocks_lengths left) (claimBlocks_lengths right) equal
  have suffixesEqual : left.suffixes = right.suffixes := by
    apply (List.map_injective_iff.mpr claimFields_injective)
    exact blocksEqual
  exact SuffixBatch.ext suffixesEqual

/-- Complete candidate batch frame. -/
def frame {candidate : Id} (batch : Batch candidate) : List Nat :=
  fixedPrefix candidate ++ batchFields batch

def frameFieldCount (candidate : Id) : Nat :=
  8 + checkedStepsPerFreshClaim candidate * 83

theorem frame_length
    {candidate : Id} (batch : Batch candidate) :
    (frame batch).length = frameFieldCount candidate := by
  simp [frame, frameFieldCount, fixedPrefix_length, batchFields_length]

theorem frame_injective
    (candidate : Id) : Function.Injective
      (frame : Batch candidate -> List Nat) := by
  intro left right equal
  apply batchFields_injective candidate
  have tails := congrArg (List.drop (fixedPrefix candidate).length) equal
  simpa [frame] using tails

structure CanonicalBatch (candidate : Id) where
  batch : Batch candidate
  claimsCanonical : ∀ claim ∈ batch.suffixes,
    MemoryClaimCodec.Claim.Canonical claim

private theorem claimFields_canonical
    {claim : Claim}
    (canonical : MemoryClaimCodec.Claim.Canonical claim) :
    ∀ value ∈ claimFields claim, value < goldilocksP := by
  intro value member
  exact MemoryClaimPoseidonBinding.frame_fields_canonical canonical value
    (by simp [MemoryClaimPoseidonBinding.frame, member])

theorem frame_fields_canonical
    {candidate : Id} (canonical : CanonicalBatch candidate) :
    ∀ value ∈ frame canonical.batch, value < goldilocksP := by
  intro value member
  rw [frame, List.mem_append] at member
  rcases member with fixed | batchMember
  · simp only [fixedPrefix, profileFields, List.mem_append,
      List.mem_cons, List.not_mem_nil, or_false] at fixed
    rcases fixed with ((rfl | rfl) | rfl | rfl | rfl | rfl) | rfl | rfl
    all_goals
      cases candidate <;>
        norm_num [domainTag, frameVersion, version,
          checkedStepsPerFreshClaim, MemoryWireGeometry.stepPublicBits_exact,
          MemoryClaimPoseidonBinding.schema_length_exact, goldilocksP]
  · rcases List.mem_flatten.mp batchMember with
        ⟨block, blockMember, fieldMember⟩
    rcases List.mem_map.mp blockMember with
      ⟨claim, claimMember, rfl⟩
    exact claimFields_canonical (canonical.claimsCanonical claim claimMember)
      value fieldMember

def fullAbsorbCount (candidate : Id) : Nat := frameFieldCount candidate / 4
def trailingFieldCount (candidate : Id) : Nat := frameFieldCount candidate % 4

def expectedSchedule (candidate : Id) : List ValueSchedule :=
  List.replicate (fullAbsorbCount candidate) (.absorb 4) ++
    (if trailingFieldCount candidate = 0 then []
      else [.absorb (trailingFieldCount candidate)]) ++
    [.pad]

theorem schedule_table :
    frameFieldCount .e1 = 91 /\
      (expectedSchedule .e1).length = 24 /\
    frameFieldCount .e4 = 340 /\
      (expectedSchedule .e4).length = 86 /\
    frameFieldCount .e8 = 672 /\
      (expectedSchedule .e8).length = 169 /\
    frameFieldCount .e16 = 1336 /\
      (expectedSchedule .e16).length = 335 := by
  decide

def representativeRound : ValueSchedule -> Round
  | .absorb count =>
      { (default : Round) with kind := .absorb (List.replicate count 0) }
  | .pad => { (default : Round) with kind := .pad }

theorem representativeRound_schedule (schedule : ValueSchedule) :
    (representativeRound schedule).valueSchedule = schedule := by
  cases schedule <;> simp [representativeRound, Round.valueSchedule]

def representativeRounds (candidate : Id) : List Round :=
  (expectedSchedule candidate).map representativeRound

theorem representativeRounds_schedule (candidate : Id) :
    valueSchedules (representativeRounds candidate) =
      expectedSchedule candidate := by
  rw [representativeRounds, valueSchedules, List.map_map]
  change (expectedSchedule candidate).map
      (fun schedule => (representativeRound schedule).valueSchedule) =
    expectedSchedule candidate
  generalize expectedSchedule candidate = schedules
  induction schedules with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq]
      exact ⟨representativeRound_schedule head, inductionHypothesis⟩

def pureDigest (candidate : Id) (values : List Nat) (lane : Nat) : Nat :=
  runValueRounds (representativeRounds candidate) values (fun _ => 0) lane

def digest {candidate : Id} (batch : Batch candidate) : BatchDigest :=
  fun lane => pureDigest candidate (frame batch) lane.val

theorem digest_canonical
    {candidate : Id} (batch : Batch candidate) (lane : Fin 4) :
    digest batch lane < goldilocksP := by
  exact runValueRounds_canonical _ _ _
    (fun _ => by norm_num [goldilocksP]) lane.val

def canonicalDigest {candidate : Id} (batch : Batch candidate) :
    Digest.Value where
  lanes := fun lane => ⟨digest batch lane, digest_canonical batch lane⟩

def PoseidonCollision (candidate : Id) : Prop :=
  ∃ left right : CanonicalBatch candidate,
    frame left.batch ≠ frame right.batch /\
      digest left.batch = digest right.batch

/-- Equal concrete digests recover the full canonical batch or expose the
exact framed Poseidon2 collision event. -/
theorem batch_eq_or_poseidon_collision
    {candidate : Id} {left right : Batch candidate}
    (leftCanonical : ∀ claim ∈ left.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (rightCanonical : ∀ claim ∈ right.suffixes,
      MemoryClaimCodec.Claim.Canonical claim)
    (equal : digest left = digest right) :
    left = right ∨ PoseidonCollision candidate := by
  classical
  by_cases same : left = right
  · exact Or.inl same
  · have frameDifferent : frame left ≠ frame right :=
      fun frameEqual => same (frame_injective candidate frameEqual)
    exact Or.inr
      ⟨⟨left, leftCanonical⟩, ⟨right, rightCanonical⟩,
        frameDifferent, equal⟩

end Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding
