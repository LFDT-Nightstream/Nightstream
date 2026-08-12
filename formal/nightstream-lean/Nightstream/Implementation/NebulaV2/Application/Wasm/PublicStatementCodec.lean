import Nightstream.Implementation.NebulaV2.Application.Wasm.ResultCodec
import Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding

/-!
Contract: canonical bit codec for the complete V2 WASM public statement.

Assurance tier: implementation model.

Owns the exact 7,868-bit field order, fixed widths, tagged manifest-digest
order, and injectivity for statements accepted by the independent public
decoding relation.

Does not own byte-container framing, Rust parsing, digest evaluation,
authoritative digest openings, generated public columns, or proof checking.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec

open Nightstream.Implementation.NebulaV2.WasmResultCodec
open Nightstream.Implementation.NebulaV2.WasmStateCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Digest
open Nightstream.Protocol.NebulaV2.Profile
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement

def profileNameValue : Profile.Name → Nat
  | .paddedRowIdentityMemoryV2 => 2
  | .paddedRowIdentityMemoryFieldNative => 3

def commitmentEncodingValue : Profile.CommitmentEncoding → Nat
  | .shiftedTernary41V1 => 1

def profileBlockWidths : List Nat := [16, 16, 16, 16]

def profileBlocks (identity : Profile.Identity) : List (List Nat) :=
  [ encodeWord 16 (profileNameValue identity.name)
  , encodeWord 16 identity.version
  , encodeWord 16 identity.checkedStepsPerFreshClaim
  , encodeWord 16 (commitmentEncodingValue identity.commitmentEncoding)
  ]

def encodeProfile (identity : Profile.Identity) : List Nat :=
  (profileBlocks identity).flatten

theorem profileBlocks_lengths (identity : Profile.Identity) :
    (profileBlocks identity).map List.length = profileBlockWidths := by
  simp [profileBlocks, profileBlockWidths, encodeWord_length]

theorem encodeProfile_length (identity : Profile.Identity) :
    (encodeProfile identity).length = profileSerializedBitCount := by
  simp [encodeProfile, List.length_flatten, profileBlocks_lengths,
    profileBlockWidths, profileSerializedBitCount]

theorem encodeProfile_binary
    (identity : Profile.Identity) (digit : Nat)
    (member : digit ∈ encodeProfile identity) :
    digit < 2 := by
  rcases List.mem_flatten.mp member with ⟨block, blockMember, digitMember⟩
  simp only [profileBlocks, List.mem_cons, List.not_mem_nil, or_false]
    at blockMember
  rcases blockMember with rfl | rfl | rfl | rfl
  all_goals exact encodeWord_binary _ _ _ digitMember

/-- Digest fields in the exact authority-bearing tag order. -/
def identityDigests
    (identity : StatementIdentity Digest.Value) : List Digest.Value :=
  [ identity.verifierKey.digest
  , identity.verifierKey.relationManifestDigest
  , identity.verifierKey.laneLayoutDigest
  , identity.verifierKey.setupManifestDigest
  , identity.verifierKey.transcriptManifestDigest
  , identity.verifierKey.codecManifestDigest
  , identity.verifierKey.terminalManifestDigest
  , identity.verifierKey.applicationStateSchemaDigest
  , identity.applicationRelationDigest
  , identity.programDigest
  , identity.memoryPlanDigest
  ]

theorem identityDigests_length
    (identity : StatementIdentity Digest.Value) :
    (identityDigests identity).length = identityDigestCount :=
  rfl

def encodeDigests : List Digest.Value → List Nat
  | [] => []
  | digest :: rest => encodeDigest digest ++ encodeDigests rest

theorem encodeDigests_length (digests : List Digest.Value) :
    (encodeDigests digests).length = digests.length * 256 := by
  induction digests with
  | nil => rfl
  | cons digest rest inductionHypothesis =>
      simp [encodeDigests, encodeDigest_exact_length, inductionHypothesis,
        Nat.succ_mul]
      omega

theorem encodeDigests_binary
    (digests : List Digest.Value) (digit : Nat)
    (member : digit ∈ encodeDigests digests) :
    digit < 2 := by
  induction digests with
  | nil => simp [encodeDigests] at member
  | cons head tail inductionHypothesis =>
      simp only [encodeDigests, List.mem_append] at member
      rcases member with inHead | inTail
      · exact encodeDigest_binary head digit inHead
      · exact inductionHypothesis inTail

theorem encodeDigests_injective : Function.Injective encodeDigests := by
  intro left
  induction left with
  | nil =>
      intro right equal
      cases right with
      | nil => rfl
      | cons head tail =>
          have lengths := congrArg List.length equal
          simp [encodeDigests, encodeDigest_exact_length] at lengths
          omega
  | cons leftHead leftTail inductionHypothesis =>
      intro right equal
      cases right with
      | nil =>
          have lengths := congrArg List.length equal
          simp [encodeDigests, encodeDigest_exact_length] at lengths
      | cons rightHead rightTail =>
          have headWords := congrArg (List.take 256) equal
          have headEqual : leftHead = rightHead := by
            apply encodeDigest_injective
            simpa [encodeDigests, encodeDigest_exact_length] using headWords
          have tailWords := congrArg (List.drop 256) equal
          have tailEqual : leftTail = rightTail := by
            apply inductionHypothesis
            simpa [encodeDigests, encodeDigest_exact_length] using tailWords
          rw [headEqual, tailEqual]

def encodeIdentity
    (identity : StatementIdentity Digest.Value) : List Nat :=
  encodeProfile identity.profile ++ encodeDigests (identityDigests identity)

theorem encodeIdentity_length
    (identity : StatementIdentity Digest.Value) :
    (encodeIdentity identity).length = identitySerializedBitCount := by
  rw [encodeIdentity, List.length_append, encodeProfile_length,
    encodeDigests_length, identityDigests_length]
  decide

theorem encodeIdentity_binary
    (identity : StatementIdentity Digest.Value) (digit : Nat)
    (member : digit ∈ encodeIdentity identity) :
    digit < 2 := by
  simp only [encodeIdentity, List.mem_append] at member
  rcases member with inProfile | inDigests
  · exact encodeProfile_binary identity.profile digit inProfile
  · exact encodeDigests_binary (identityDigests identity) digit inDigests

private theorem identityDigests_injective_with_profile
    {left right : StatementIdentity Digest.Value}
    (profileEqual : left.profile = right.profile)
    (digestsEqual : identityDigests left = identityDigests right) :
    left = right := by
  rcases left with
    ⟨leftProfile,
      ⟨leftKey, leftRelation, leftLayout, leftSetup, leftTranscript,
        leftCodec, leftTerminal, leftStateSchema⟩,
      leftApplication, leftProgram, leftPlan⟩
  rcases right with
    ⟨rightProfile,
      ⟨rightKey, rightRelation, rightLayout, rightSetup, rightTranscript,
        rightCodec, rightTerminal, rightStateSchema⟩,
      rightApplication, rightProgram, rightPlan⟩
  simp_all [identityDigests]

theorem encodeIdentity_injective_on_profile
    {left right : StatementIdentity Digest.Value}
    {expectedProfile : Profile.Identity}
    (leftProfile : left.profile = expectedProfile)
    (rightProfile : right.profile = expectedProfile)
    (equal : encodeIdentity left = encodeIdentity right) :
    left = right := by
  have profileEqual : left.profile = right.profile :=
    leftProfile.trans rightProfile.symm
  apply identityDigests_injective_with_profile profileEqual
  apply encodeDigests_injective
  have tails := congrArg (List.drop profileSerializedBitCount) equal
  simpa [encodeIdentity, encodeProfile_length] using tails

theorem encodeIdentity_injective_on_v2
    {left right : StatementIdentity Digest.Value}
    (leftProfile : left.profile = Profile.v2)
    (rightProfile : right.profile = Profile.v2)
    (equal : encodeIdentity left = encodeIdentity right) :
    left = right :=
  encodeIdentity_injective_on_profile leftProfile rightProfile equal

def statementBlockWidths : List Nat := [2880, 2293, 7, 23, 2665]

/-- Field blocks in the exact order of `SPEC.md` section 17. -/
def blocks (image : PublicImage) : List (List Nat) :=
  [ encodeIdentity image.identity
  , WasmStateCodec.encode image.initialApplicationState
  , encodeWord segmentCountBitWidth image.segmentCount
  , encodeWord finalTimestampBitWidth image.finalGlobalTimestamp
  , WasmResultCodec.encode image.result
  ]

def encode (image : PublicImage) : List Nat :=
  (blocks image).flatten

theorem blocks_lengths (image : PublicImage) :
    (blocks image).map List.length = statementBlockWidths := by
  simp [blocks, statementBlockWidths, encodeIdentity_length,
    identitySerializedBitCount_eq, WasmStateCodec.encode_exact_length,
    encodeWord_length, WasmResultCodec.encode_length,
    segmentCountBitWidth, finalTimestampBitWidth]

theorem encode_length (image : PublicImage) :
    (encode image).length = 7868 := by
  simp [encode, List.length_flatten, blocks_lengths, statementBlockWidths]

theorem encode_binary
    (image : PublicImage) (digit : Nat)
    (member : digit ∈ encode image) :
    digit < 2 := by
  rcases List.mem_flatten.mp member with ⟨block, blockMember, digitMember⟩
  simp only [blocks, List.mem_cons, List.not_mem_nil, or_false]
    at blockMember
  rcases blockMember with rfl | rfl | rfl | rfl | rfl
  · exact encodeIdentity_binary _ _ digitMember
  · exact WasmStateCodec.encode_binary _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact WasmResultCodec.encode_binary _ _ digitMember

private theorem component_words_equal
    {left right : PublicImage}
    (equal : blocks left = blocks right) :
    encodeIdentity left.identity = encodeIdentity right.identity ∧
      WasmStateCodec.encode left.initialApplicationState =
        WasmStateCodec.encode right.initialApplicationState ∧
      encodeWord segmentCountBitWidth left.segmentCount =
        encodeWord segmentCountBitWidth right.segmentCount ∧
      encodeWord finalTimestampBitWidth left.finalGlobalTimestamp =
        encodeWord finalTimestampBitWidth right.finalGlobalTimestamp ∧
      WasmResultCodec.encode left.result =
        WasmResultCodec.encode right.result := by
  simpa [blocks] using equal

/-- The complete flattened 7,868-bit public statement has one accepted typed
preimage. This theorem does not replace digest-opening checks. -/
theorem encode_injective_of_decodesFor
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement : ProductionStatement Program}
    {expectedProfile : Profile.Identity}
    (leftDecoded : left.DecodesFor expectedProfile leftStatement)
    (rightDecoded : right.DecodesFor expectedProfile rightStatement)
    (equal : encode left = encode right) :
    left = right := by
  have words := component_words_equal
    (flatten_injective_of_lengths
      (blocks_lengths left) (blocks_lengths right) equal)
  apply PublicImage.ext
  · exact encodeIdentity_injective_on_profile
      leftDecoded.image_profile_exact rightDecoded.image_profile_exact words.1
  · exact WasmStateCodec.encode_injective_on_canonical
      leftDecoded.initial_state_canonical rightDecoded.initial_state_canonical
      words.2.1
  · exact encodeWord_injective_of_bound
      leftDecoded.image_segment_count_bound
      rightDecoded.image_segment_count_bound words.2.2.1
  · exact encodeWord_injective_of_bound
      leftDecoded.image_final_timestamp_bound
      rightDecoded.image_final_timestamp_bound words.2.2.2.1
  · exact WasmResultCodec.encode_injective_of_decodes
      leftDecoded.image_result_decodes rightDecoded.image_result_decodes
      words.2.2.2.2

theorem encode_injective_of_decodes
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement : ProductionStatement Program}
    (leftDecoded : left.Decodes leftStatement)
    (rightDecoded : right.Decodes rightStatement)
    (equal : encode left = encode right) :
    left = right :=
  encode_injective_of_decodesFor leftDecoded.toDecodesFor
    rightDecoded.toDecodesFor equal

end Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec
