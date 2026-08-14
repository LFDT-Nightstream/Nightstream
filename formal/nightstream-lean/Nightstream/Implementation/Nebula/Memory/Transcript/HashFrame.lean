import Mathlib.Data.List.OfFn
import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.Protocol.Nebula.Digest
import Nightstream.Protocol.Nebula.Transcript

/-!
Contract: exact canonical field frame for the V2 memory challenge transcript.

Assurance tier: implementation model and cryptographic boundary.

Owns the numeric transcript and coordinate tags, fixed V2 profile fields,
exact 53-field order, bounded counter domain, canonical-field proof, and the
refinement to the independent protocol transcript frame.

Does not own Poseidon2 permutation rows, challenge unpredictability,
prechallenge extraction, generated absolute columns, or Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryTranscriptHashFrame

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Lifecycle

def domainTag : Nat := 0x4e545232
def frameVersion : Nat := 1

/-- Numeric profile-name encoding used only inside the memory transcript.
The values are part of the protocol frame and are not enum ordinals. -/
def profileNameValue : Profile.Name → Nat
  | .paddedRowIdentityMemoryV2 => 2
  | .paddedRowIdentityMemoryFieldNative => 3

/-- Numeric commitment-encoding identifier used inside the transcript. -/
def commitmentEncodingValue : Profile.CommitmentEncoding → Nat
  | .shiftedTernary41V1 => 1

/-- Each profile family has a distinct memory-transcript domain. -/
def domainTagFor (profile : Profile.Identity) : Nat :=
  match profile.name with
  | .paddedRowIdentityMemoryV2 => domainTag
  | .paddedRowIdentityMemoryFieldNative => 0x4e545246

def profileFieldsFor (profile : Profile.Identity) : List Nat :=
  [ profileNameValue profile.name
  , profile.version
  , profile.checkedStepsPerFreshClaim
  , commitmentEncodingValue profile.commitmentEncoding
  ]

def fixedPrefixFor (profile : Profile.Identity) : List Nat :=
  [domainTagFor profile, frameVersion] ++ profileFieldsFor profile

/-- Number of fresh claims in one 1,088-step memory segment for this
profile. A selected profile separately proves that its positive factor divides
1,088 exactly. -/
def checkedStepCountFor (profile : Profile.Identity) : Nat :=
  claimsPerSegment / profile.checkedStepsPerFreshClaim

def coordinateTag : Fin 4 → Nat
  | ⟨0, _⟩ => 0x4e544730
  | ⟨1, _⟩ => 0x4e544731
  | ⟨2, _⟩ => 0x4e544732
  | ⟨3, _⟩ => 0x4e544733

theorem coordinateTags_exact :
    List.ofFn coordinateTag =
      [0x4e544730, 0x4e544731, 0x4e544732, 0x4e544733] := by
  decide

theorem all_tags_pairwise_distinct :
    List.Nodup
      [ domainTag
      , coordinateTag 0, coordinateTag 1
      , coordinateTag 2, coordinateTag 3
      ] := by
  decide

def profileFields : List Nat := [2, 2, 1, 1]

theorem profileFields_eq_v2 : profileFieldsFor Profile.v2 = profileFields := by
  rfl

def digestFields (digest : Digest.Value) : List Nat :=
  List.ofFn fun lane => (digest.lanes lane).val

def encodeDigests (digests : List Digest.Value) : List Nat :=
  digests.flatMap digestFields

theorem digestFields_length (digest : Digest.Value) :
    (digestFields digest).length = 4 := by
  simp [digestFields, Digest.laneCount]

theorem digestFields_injective : Function.Injective digestFields := by
  intro left right equal
  apply Digest.Value.ext
  funext lane
  apply Subtype.ext
  exact congrFun (List.ofFn_injective equal) lane

theorem encodeDigests_length (digests : List Digest.Value) :
    (encodeDigests digests).length = digests.length * 4 := by
  induction digests with
  | nil => rfl
  | cons digest rest inductionHypothesis =>
      simp [encodeDigests, digestFields_length]
      omega

theorem encodeDigests_injective : Function.Injective encodeDigests := by
  intro left
  induction left with
  | nil =>
      intro right equal
      cases right with
      | nil => rfl
      | cons head tail =>
          have lengths := congrArg List.length equal
          simp [encodeDigests, digestFields_length] at lengths
          omega
  | cons leftHead leftTail inductionHypothesis =>
      intro right equal
      cases right with
      | nil =>
          have lengths := congrArg List.length equal
          simp [encodeDigests, digestFields_length] at lengths
      | cons rightHead rightTail =>
          have headFields := congrArg (List.take 4) equal
          have headEqual : leftHead = rightHead := by
            apply digestFields_injective
            simpa [encodeDigests, digestFields_length] using headFields
          have tailFields := congrArg (List.drop 4) equal
          have tailEqual : leftTail = rightTail := by
            apply inductionHypothesis
            simpa [encodeDigests, digestFields_length] using tailFields
          rw [headEqual, tailEqual]

/-- The V2-only transcript input. The profile is not caller-selected. -/
@[ext] structure Input where
  verifierKeyDigest : Digest.Value
  applicationRelationDigest : Digest.Value
  programDigest : Digest.Value
  memoryPlanDigest : Digest.Value
  laneLayoutDigest : Digest.Value
  priorStateDigest : Digest.Value
  runningAccumulatorDigest : Digest.Value
  segmentIndex : Nat
  segmentStartTimestamp : Nat
  activeAccessCount : Nat
  segmentEndTimestamp : Nat
  roots : FPrime.Roots Digest.Value
deriving DecidableEq

/-- Exact bridge to the independent semantic frame. -/
def Input.toSemanticFrame (input : Input) : Transcript.Frame Digest.Value where
  profile := Profile.v2
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

theorem toSemanticFrame_profile (input : Input) :
    input.toSemanticFrame.profile = Profile.v2 :=
  rfl

def fixedPrefix : List Nat :=
  [domainTag, frameVersion] ++ profileFields

theorem fixedPrefix_eq_v2 : fixedPrefixFor Profile.v2 = fixedPrefix := by
  rfl

def authorityDigests (input : Input) : List Digest.Value :=
  [ input.verifierKeyDigest
  , input.applicationRelationDigest
  , input.programDigest
  , input.memoryPlanDigest
  , input.laneLayoutDigest
  , input.priorStateDigest
  , input.runningAccumulatorDigest
  ]

def authorityDigestFields (input : Input) : List Nat :=
  encodeDigests (authorityDigests input)

def counterFields (input : Input) : List Nat :=
  [ input.segmentIndex
  , input.segmentStartTimestamp
  , input.activeAccessCount
  , input.segmentEndTimestamp
  , claimsPerSegment
  , 63
  , 64
  ]

def counterFieldsFor (profile : Profile.Identity) (input : Input) : List Nat :=
  [ input.segmentIndex
  , input.segmentStartTimestamp
  , input.activeAccessCount
  , input.segmentEndTimestamp
  , checkedStepCountFor profile
  , 63
  , 64
  ]

theorem counterFieldsFor_v2 (input : Input) :
    counterFieldsFor Profile.v2 input = counterFields input := by
  rfl

def rootDigests (input : Input) : List Digest.Value :=
  [ input.roots.operations
  , input.roots.initialSnapshot
  , input.roots.finalSnapshot
  ]

def rootFields (input : Input) : List Nat :=
  encodeDigests (rootDigests input)

/-- Normative 53-field frame from `SPEC.md` section 12. -/
def encode (input : Input) : List Nat :=
  (fixedPrefix ++ authorityDigestFields input) ++
    (counterFields input ++ rootFields input)

/-- Profile-indexed 53-field frame. This is the authority-bearing encoder for
successor profiles. The fixed V2 `encode` above is its version-2 instance. -/
def encodeFor (profile : Profile.Identity) (input : Input) : List Nat :=
  (fixedPrefixFor profile ++ authorityDigestFields input) ++
    (counterFieldsFor profile input ++ rootFields input)

theorem encodeFor_v2 (input : Input) :
    encodeFor Profile.v2 input = encode input := by
  rfl

theorem fixedPrefix_length : fixedPrefix.length = 6 := by
  rfl

theorem authorityDigestFields_length (input : Input) :
    (authorityDigestFields input).length = 28 := by
  rw [authorityDigestFields, encodeDigests_length]
  rfl

theorem counterFields_length (input : Input) :
    (counterFields input).length = 7 := by
  rfl

theorem rootFields_length (input : Input) :
    (rootFields input).length = 12 := by
  rw [rootFields, encodeDigests_length]
  rfl

theorem profileFieldsFor_length (profile : Profile.Identity) :
    (profileFieldsFor profile).length = 4 := by
  rfl

theorem fixedPrefixFor_length (profile : Profile.Identity) :
    (fixedPrefixFor profile).length = 6 := by
  simp [fixedPrefixFor, profileFieldsFor_length]

theorem counterFieldsFor_length (profile : Profile.Identity) (input : Input) :
    (counterFieldsFor profile input).length = 7 := by
  rfl

theorem encodeFor_length (profile : Profile.Identity) (input : Input) :
    (encodeFor profile input).length = 53 := by
  simp [encodeFor, fixedPrefixFor_length, authorityDigestFields_length,
    counterFieldsFor_length, rootFields_length]

/-- The four numeric profile fields are lossless. -/
theorem profileFieldsFor_injective : Function.Injective profileFieldsFor := by
  intro left right equal
  cases left with
  | mk leftName leftVersion leftSteps leftEncoding =>
      cases right with
      | mk rightName rightVersion rightSteps rightEncoding =>
          simp only [profileFieldsFor, List.cons.injEq] at equal
          rcases equal with
            ⟨nameEqual, versionEqual, stepsEqual, encodingEqual, _⟩
          have nameExact : leftName = rightName := by
            cases leftName <;> cases rightName <;>
              simp_all [profileNameValue]
          have encodingExact : leftEncoding = rightEncoding := by
            cases leftEncoding <;> cases rightEncoding <;> simp_all
          subst rightName
          subst rightVersion
          subst rightSteps
          subst rightEncoding
          rfl

/-- For a fixed profile, the numeric frame is lossless in all dynamic
authority fields before hashing. -/
theorem encodeFor_injective (profile : Profile.Identity) :
    Function.Injective (encodeFor profile) := by
  intro left right equal
  have authorityFieldsEqual :
      authorityDigestFields left = authorityDigestFields right := by
    have middle := congrArg
      (fun values : List Nat => (values.drop 6).take 28) equal
    simpa [encodeFor, fixedPrefixFor_length,
      authorityDigestFields_length] using middle
  have counterEqual :
      counterFieldsFor profile left = counterFieldsFor profile right := by
    have middle := congrArg
      (fun values : List Nat => (values.drop 34).take 7) equal
    simpa [encodeFor, fixedPrefixFor_length,
      authorityDigestFields_length, counterFieldsFor_length] using middle
  have rootFieldsEqual : rootFields left = rootFields right := by
    have tails := congrArg (List.drop 41) equal
    simpa [encodeFor, fixedPrefixFor_length,
      authorityDigestFields_length, counterFieldsFor_length] using tails
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
  simp only [counterFieldsFor, List.cons.injEq] at counterEqual
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

/-- Equality of complete numeric frames also fixes the profile identity. -/
theorem profile_eq_of_encodeFor_eq
    {leftProfile rightProfile : Profile.Identity}
    {left right : Input}
    (equal : encodeFor leftProfile left = encodeFor rightProfile right) :
    leftProfile = rightProfile := by
  have prefixes := congrArg (fun values : List Nat => (values.take 6).drop 2)
    equal
  apply profileFieldsFor_injective
  simpa [encodeFor, fixedPrefixFor, profileFieldsFor_length] using prefixes

theorem encodeFor_joint_injective
    {leftProfile rightProfile : Profile.Identity}
    {left right : Input}
    (equal : encodeFor leftProfile left = encodeFor rightProfile right) :
    leftProfile = rightProfile ∧ left = right := by
  have profileExact := profile_eq_of_encodeFor_eq equal
  subst rightProfile
  exact ⟨rfl, encodeFor_injective leftProfile equal⟩

theorem encode_length (input : Input) :
    (encode input).length = 53 := by
  simp [encode, fixedPrefix_length, authorityDigestFields_length,
    counterFields_length, rootFields_length]

/-- The numeric frame is lossless. This theorem prevents two different
authority inputs from sharing one field frame before any hash assumption is
used. -/
theorem encode_injective : Function.Injective encode := by
  intro left right equal
  have authorityFieldsEqual :
      authorityDigestFields left = authorityDigestFields right := by
    have middle := congrArg (fun values : List Nat => (values.drop 6).take 28)
      equal
    simpa [encode, fixedPrefix_length, authorityDigestFields_length]
      using middle
  have counterEqual : counterFields left = counterFields right := by
    have middle := congrArg (fun values : List Nat => (values.drop 34).take 7)
      equal
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

/-- All variable integer fields must already satisfy their V2 range
relations. The transcript does not repair or reduce an invalid value. -/
structure Input.CountersCanonical (input : Input) : Prop where
  segmentIndex : input.segmentIndex < 2 ^ 7
  segmentStartTimestamp : input.segmentStartTimestamp < 2 ^ 23
  activeAccessCount : input.activeAccessCount < 2 ^ 17
  segmentEndTimestamp : input.segmentEndTimestamp < 2 ^ 23

/-- Numeric well-formedness needed to place one profile identity in canonical
Goldilocks transcript fields. Protocol selection adds the stronger exact-factor
and version requirements. -/
structure ProfileCanonical (profile : Profile.Identity) : Prop where
  version : profile.version < goldilocksP
  checkedStepsPositive : 0 < profile.checkedStepsPerFreshClaim
  checkedSteps : profile.checkedStepsPerFreshClaim < goldilocksP

theorem v2ProfileCanonical : ProfileCanonical Profile.v2 := by
  constructor <;> decide

private theorem profileNameValue_lt (name : Profile.Name) :
    profileNameValue name < goldilocksP := by
  cases name <;> decide

private theorem commitmentEncodingValue_lt
    (encoding : Profile.CommitmentEncoding) :
    commitmentEncodingValue encoding < goldilocksP := by
  cases encoding <;> decide

private theorem domainTagFor_lt (profile : Profile.Identity) :
    domainTagFor profile < goldilocksP := by
  cases profile with
  | mk name version checkedSteps encoding =>
      cases name <;> norm_num [domainTagFor, domainTag, goldilocksP]

theorem fixedPrefixFor_fields_canonical
    {profile : Profile.Identity} (valid : ProfileCanonical profile) :
    ∀ value ∈ fixedPrefixFor profile, value < goldilocksP := by
  intro value member
  simp only [fixedPrefixFor, profileFieldsFor, List.mem_append,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with (rfl | rfl) | rfl | rfl | rfl | rfl
  · exact domainTagFor_lt profile
  · decide
  · exact profileNameValue_lt profile.name
  · exact valid.version
  · exact valid.checkedSteps
  · exact commitmentEncodingValue_lt profile.commitmentEncoding

theorem checkedStepCountFor_lt
    {profile : Profile.Identity} (_valid : ProfileCanonical profile) :
    checkedStepCountFor profile < goldilocksP := by
  unfold checkedStepCountFor
  have upper : claimsPerSegment < goldilocksP := by decide
  exact (Nat.div_le_self claimsPerSegment _).trans_lt upper

private theorem digestFields_canonical (digest : Digest.Value) :
    ∀ value ∈ digestFields digest, value < goldilocksP := by
  intro value member
  rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
  exact (digest.lanes lane).property

theorem encodeFor_fields_canonical
    {profile : Profile.Identity} {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (counters : input.CountersCanonical) :
    ∀ value ∈ encodeFor profile input, value < goldilocksP := by
  intro value member
  simp only [encodeFor, List.mem_append] at member
  rcases member with fixedOrAuthority | countersOrRoots
  · rcases fixedOrAuthority with fixed | authority
    · exact fixedPrefixFor_fields_canonical profileCanonical value fixed
    · rcases List.mem_flatMap.mp authority with
          ⟨digest, _digestMember, fieldMember⟩
      exact digestFields_canonical digest value fieldMember
  · rcases countersOrRoots with counter | root
    · simp only [counterFieldsFor, List.mem_cons, List.not_mem_nil,
          or_false] at counter
      rcases counter with rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact counters.segmentIndex.trans_le (by decide)
      · exact counters.segmentStartTimestamp.trans_le (by decide)
      · exact counters.activeAccessCount.trans_le (by decide)
      · exact counters.segmentEndTimestamp.trans_le (by decide)
      · exact checkedStepCountFor_lt profileCanonical
      all_goals decide
    · rcases List.mem_flatMap.mp root with
          ⟨digest, _digestMember, fieldMember⟩
      exact digestFields_canonical digest value fieldMember

theorem encode_fields_canonical
    {input : Input} (counters : input.CountersCanonical) :
    ∀ value ∈ encode input, value < goldilocksP := by
  intro value member
  simp only [encode, List.mem_append] at member
  rcases member with fixedOrAuthority | countersOrRoots
  · rcases fixedOrAuthority with fixed | authority
    · simp only [fixedPrefix, profileFields, List.mem_append,
        List.mem_cons, List.not_mem_nil, or_false] at fixed
      rcases fixed with (rfl | rfl) | rfl | rfl | rfl | rfl <;>
        decide
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
      all_goals decide
    · rcases List.mem_flatMap.mp root with
          ⟨digest, _digestMember, fieldMember⟩
      exact digestFields_canonical digest value fieldMember

end Nightstream.Implementation.Nebula.MemoryTranscriptHashFrame
