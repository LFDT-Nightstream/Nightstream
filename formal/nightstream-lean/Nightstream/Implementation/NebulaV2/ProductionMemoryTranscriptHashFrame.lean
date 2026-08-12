import Nightstream.Implementation.NebulaV2.MemoryTranscriptHashFrame
import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-!
Contract: exact candidate-specific memory challenge frame for the field-native
successor profiles.

The frame keeps the V2 authority fields and 53-field width. It replaces the
reference profile prefix and factor-one claim count with the exact candidate
identity and `1088 / E` batch count. Different candidates have disjoint frames
before hashing.

Does not own Poseidon2 rows, Fiat-Shamir security, generated placement, Rust
encoding, candidate selection, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryTranscriptHashFrame

open Nightstream.Implementation.NebulaV2.MemoryTranscriptHashFrame
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

abbrev Input := MemoryTranscriptHashFrame.Input
abbrev CountersCanonical := MemoryTranscriptHashFrame.Input.CountersCanonical

theorem candidateProfileCanonical (candidate : Id) :
    MemoryTranscriptHashFrame.ProfileCanonical (identity candidate) := by
  cases candidate <;> constructor <;> decide

/-- ASCII `NTRF`, encoded as one Goldilocks field. -/
def domainTag : Nat := 0x4e545246
def frameVersion : Nat := 1

/-- Name, version, checked-step factor, and commitment encoding. -/
def profileFields (candidate : Id) : List Nat :=
  [3, version candidate, checkedStepsPerFreshClaim candidate, 1]

theorem profileFields_length (candidate : Id) :
    (profileFields candidate).length = 4 := rfl

theorem profileFields_injective : Function.Injective profileFields := by
  intro left right equal
  cases left <;> cases right <;>
    simp [profileFields, version, checkedStepsPerFreshClaim] at equal ⊢

def fixedPrefix (candidate : Id) : List Nat :=
  [domainTag, frameVersion] ++ profileFields candidate

theorem fixedPrefix_length (candidate : Id) :
    (fixedPrefix candidate).length = 6 := by
  simp [fixedPrefix, profileFields_length]

def counterFields (candidate : Id) (input : Input) : List Nat :=
  [ input.segmentIndex
  , input.segmentStartTimestamp
  , input.activeAccessCount
  , input.segmentEndTimestamp
  , claimsPerSegment candidate
  , 63
  , 64
  ]

theorem counterFields_length (candidate : Id) (input : Input) :
    (counterFields candidate input).length = 7 := rfl

/-- Exact candidate semantic frame. -/
def Input.toSemanticFrame (candidate : Id) (input : Input) :
    Transcript.Frame Digest.Value where
  profile := identity candidate
  verifierKeyDigest := input.verifierKeyDigest
  applicationRelationDigest := input.applicationRelationDigest
  programDigest := input.programDigest
  memoryPlanDigest := input.memoryPlanDigest
  laneLayoutDigest := input.laneLayoutDigest
  priorStateDigest := input.priorStateDigest
  runningAccumulatorDigest := input.runningAccumulatorDigest
  segmentIndex := input.segmentIndex
  segmentStartTimestamp := input.segmentStartTimestamp
  activeAccessCount := input.activeAccessCount
  segmentEndTimestamp := input.segmentEndTimestamp
  roots := input.roots

@[simp]
theorem Input.toSemanticFrame_profile (candidate : Id) (input : Input) :
    (input.toSemanticFrame candidate).profile = identity candidate := rfl

/-- Normative 53-field candidate frame. -/
def encode (candidate : Id) (input : Input) : List Nat :=
  (fixedPrefix candidate ++ authorityDigestFields input) ++
    (counterFields candidate input ++ rootFields input)

/-- The candidate encoder is exactly the common profile-indexed encoder.
This equality is the bridge used by the row and challenge implementations. -/
theorem encode_eq_encodeFor (candidate : Id) (input : Input) :
    encode candidate input =
      MemoryTranscriptHashFrame.encodeFor (identity candidate) input := by
  cases candidate <;>
    rfl

/-- A successor candidate cannot silently reuse the version-2 challenge
frame. The frames differ before Poseidon2 for every dynamic input. -/
theorem encode_ne_v2 (candidate : Id) (input : Input) :
    encode candidate input ≠ MemoryTranscriptHashFrame.encode input := by
  rw [encode_eq_encodeFor, ← MemoryTranscriptHashFrame.encodeFor_v2]
  intro equal
  have profileExact :=
    MemoryTranscriptHashFrame.profile_eq_of_encodeFor_eq equal
  exact identity_ne_v2 candidate profileExact

theorem encode_length (candidate : Id) (input : Input) :
    (encode candidate input).length = 53 := by
  simp [encode, fixedPrefix_length, authorityDigestFields_length,
    counterFields_length, rootFields_length]

/-- For one fixed candidate, the field frame is lossless. -/
theorem encode_injective (candidate : Id) :
    Function.Injective (encode candidate) := by
  intro left right equal
  have authorityFieldsEqual :
      authorityDigestFields left = authorityDigestFields right := by
    have middle := congrArg
      (fun values : List Nat => (values.drop 6).take 28) equal
    simpa [encode, fixedPrefix_length, authorityDigestFields_length]
      using middle
  have counterEqual :
      counterFields candidate left = counterFields candidate right := by
    have middle := congrArg
      (fun values : List Nat => (values.drop 34).take 7) equal
    simpa [encode, fixedPrefix_length, authorityDigestFields_length,
      counterFields_length] using middle
  have rootFieldsEqual : rootFields left = rootFields right := by
    have tails := congrArg (List.drop 41) equal
    simpa [encode, fixedPrefix_length, authorityDigestFields_length,
      counterFields_length] using tails
  have authorityEqual : authorityDigests left = authorityDigests right :=
    encodeDigests_injective authorityFieldsEqual
  have rootsEqual : rootDigests left = rootDigests right :=
    encodeDigests_injective rootFieldsEqual
  rcases left with
    ⟨leftKey, leftApplication, leftProgram, leftPlan, leftLayout,
      leftPrior, leftAccumulator, leftIndex, leftStart, leftCount, leftEnd,
      leftRoots⟩
  rcases right with
    ⟨rightKey, rightApplication, rightProgram, rightPlan, rightLayout,
      rightPrior, rightAccumulator, rightIndex, rightStart, rightCount,
      rightEnd, rightRoots⟩
  simp only [authorityDigests, List.cons.injEq] at authorityEqual
  simp only [counterFields, List.cons.injEq] at counterEqual
  simp only [rootDigests, List.cons.injEq] at rootsEqual
  rcases authorityEqual with
    ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, _authorityEnd⟩
  rcases counterEqual with ⟨rfl, rfl, rfl, rfl, _counterEnd⟩
  rcases rootsEqual with
    ⟨operations, initialSnapshot, finalSnapshot, _rootsEnd⟩
  have rootEqual : leftRoots = rightRoots := by
    apply FPrime.Roots.ext <;> assumption
  cases rootEqual
  rfl

/-- Equal frames from two candidate versions force the same candidate. -/
theorem candidate_eq_of_encode_eq
    {leftCandidate rightCandidate : Id}
    {left right : Input}
    (equal : encode leftCandidate left = encode rightCandidate right) :
    leftCandidate = rightCandidate := by
  have prefixes := congrArg (List.take 6) equal
  have profileEqual :
      profileFields leftCandidate = profileFields rightCandidate := by
    simpa [encode, fixedPrefix, profileFields_length] using prefixes
  exact profileFields_injective profileEqual

/-- Candidate plus input is injective. Cross-version transcript reuse is not
possible before the Poseidon2 assumption is invoked. -/
theorem encode_joint_injective
    {leftCandidate rightCandidate : Id}
    {left right : Input}
    (equal : encode leftCandidate left = encode rightCandidate right) :
    leftCandidate = rightCandidate ∧ left = right := by
  have candidateEqual := candidate_eq_of_encode_eq equal
  subst rightCandidate
  exact ⟨rfl, encode_injective leftCandidate equal⟩

private theorem digestFields_canonical (digest : Digest.Value) :
    ∀ value ∈ digestFields digest, value < goldilocksP := by
  intro value member
  rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
  exact (digest.lanes lane).property

theorem encode_fields_canonical
    {candidate : Id} {input : Input}
    (counters : input.CountersCanonical) :
    ∀ value ∈ encode candidate input, value < goldilocksP := by
  intro value member
  simp only [encode, List.mem_append] at member
  rcases member with fixedOrAuthority | countersOrRoots
  · rcases fixedOrAuthority with fixed | authority
    · simp only [fixedPrefix, profileFields, List.mem_append,
        List.mem_cons, List.not_mem_nil, or_false] at fixed
      rcases fixed with (rfl | rfl) | rfl | rfl | rfl | rfl
      all_goals
        cases candidate <;>
          norm_num [domainTag, frameVersion, version,
            checkedStepsPerFreshClaim, goldilocksP]
    · rcases List.mem_flatMap.mp authority with
          ⟨digest, _digestMember, fieldMember⟩
      exact digestFields_canonical digest value fieldMember
  · rcases countersOrRoots with counter | root
    · simp only [counterFields, List.mem_cons, List.not_mem_nil, or_false]
        at counter
      rcases counter with rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact counters.segmentIndex.trans_le (by decide)
      · exact counters.segmentStartTimestamp.trans_le (by decide)
      · exact counters.activeAccessCount.trans_le (by decide)
      · exact counters.segmentEndTimestamp.trans_le (by decide)
      all_goals
        cases candidate <;>
          norm_num [claimsPerSegment, checkedStepsPerFreshClaim,
            ProductionProfileCandidates.stepsPerSegment, goldilocksP]
    · rcases List.mem_flatMap.mp root with
          ⟨digest, _digestMember, fieldMember⟩
      exact digestFields_canonical digest value fieldMember

end Nightstream.Implementation.NebulaV2.ProductionMemoryTranscriptHashFrame
