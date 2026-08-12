import Nightstream.Protocol.NebulaV2.Lifecycle
import Nightstream.Protocol.NebulaV2.WasmStatement

/-!
Contract: exact typed public image for one production V2 WASM statement.

Assurance tier: model-level.

Owns the authority-bearing statement field order, the exact profile
restriction, canonical public bounds, and the relation from the public image
to the independent semantic statement.

Does not own bit or byte encoding, digest evaluation, authoritative digest
openings, generated rows, proof parsing, or deployed verification.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Digest
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.Profile
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.WasmStateEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement

def profileSerializedBitCount : Nat := 64
def identityDigestCount : Nat := 11
def identitySerializedBitCount : Nat :=
  profileSerializedBitCount + identityDigestCount * Digest.serializedBitCount
def segmentCountBitWidth : Nat := 7
def finalTimestampBitWidth : Nat := 23
def serializedBitCount : Nat :=
  identitySerializedBitCount + WasmStateEncoding.serializedBitCount +
    segmentCountBitWidth + finalTimestampBitWidth + 2665

theorem identitySerializedBitCount_eq : identitySerializedBitCount = 2880 := by
  decide

theorem serializedBitCount_eq : serializedBitCount = 7868 := by
  decide

/-- Complete typed external public image. Each manifest digest remains a
separate authority-bearing field. -/
@[ext] structure PublicImage where
  identity : StatementIdentity Digest.Value
  initialApplicationState : Image
  segmentCount : Nat
  finalGlobalTimestamp : Nat
  result : ProductionResultImage

def PublicImage.ofStatement
    {Program : Type} (statement : ProductionStatement Program) : PublicImage :=
  { identity := statement.base.identity
    initialApplicationState :=
      WasmStateEncoding.encode statement.base.initialApplicationState
    segmentCount := statement.base.segmentCount
    finalGlobalTimestamp := statement.base.finalGlobalTimestamp
    result := statement.resultImage }

/-- Canonical public decoding relation. It checks structural and range facts;
it does not claim that a proof establishes the statement. -/
structure PublicImage.Decodes
    {Program : Type}
    (image : PublicImage)
    (statement : ProductionStatement Program) : Prop where
  exactImage : image = PublicImage.ofStatement statement
  exactProfile : statement.base.identity.profile = Profile.v2
  segmentCountPositive : 0 < statement.base.segmentCount
  segmentCountBound : statement.base.segmentCount ≤ maximumSegments
  finalTimestampBound : statement.base.finalGlobalTimestamp < 2 ^ finalTimestampBitWidth
  realRowsFitDeclaredSegments :
    statement.base.expectedResult.realApplicationRowCount ≤
      Completion.segmentCapacity statement.base.segmentCount
  smallestSegmentCount :
    statement.base.segmentCount = Completion.minimumSegmentCount
      statement.base.expectedResult.realApplicationRowCount

/-- Profile-parameterized canonical public decoding. Successor relations use
this form so that their public statement cannot retain the V2 profile. -/
structure PublicImage.DecodesFor
    {Program : Type}
    (expectedProfile : Profile.Identity)
    (image : PublicImage)
    (statement : ProductionStatement Program) : Prop where
  exactImage : image = PublicImage.ofStatement statement
  exactProfile : statement.base.identity.profile = expectedProfile
  segmentCountPositive : 0 < statement.base.segmentCount
  segmentCountBound : statement.base.segmentCount ≤ maximumSegments
  finalTimestampBound :
    statement.base.finalGlobalTimestamp < 2 ^ finalTimestampBitWidth
  realRowsFitDeclaredSegments :
    statement.base.expectedResult.realApplicationRowCount ≤
      Completion.segmentCapacity statement.base.segmentCount
  smallestSegmentCount :
    statement.base.segmentCount = Completion.minimumSegmentCount
      statement.base.expectedResult.realApplicationRowCount

namespace PublicImage.Decodes

theorem initial_state_canonical
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    image.initialApplicationState.Canonical := by
  rw [decoded.exactImage]
  exact (canonical_encode_iff statement.base.initialApplicationState).2
    statement.initialApplicationStateValid

theorem image_profile_exact
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    image.identity.profile = Profile.v2 := by
  rw [decoded.exactImage]
  exact decoded.exactProfile

theorem image_segment_count_bound
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    image.segmentCount < 2 ^ segmentCountBitWidth := by
  rw [decoded.exactImage]
  change statement.base.segmentCount < 2 ^ segmentCountBitWidth
  have := decoded.segmentCountBound
  norm_num [segmentCountBitWidth, maximumSegments] at this ⊢
  omega

theorem image_final_timestamp_bound
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    image.finalGlobalTimestamp < 2 ^ finalTimestampBitWidth := by
  rw [decoded.exactImage]
  change statement.base.finalGlobalTimestamp < 2 ^ finalTimestampBitWidth
  exact decoded.finalTimestampBound

theorem image_result_decodes
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    image.result.Decodes statement.base.expectedResult := by
  rw [decoded.exactImage]
  exact statement.resultDecoded

/-- Canonical public decoding contains every arithmetic condition needed for
completed-trace minimality. The row-kind list remains the canonical list
defined from the public result. -/
def completionTrace
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    Completion.ValidCompletedTrace statement.base.expectedResult
      statement.base.segmentCount
      (Completion.canonicalRows statement.base.expectedResult
        statement.base.segmentCount) where
  segmentCountPositive := decoded.segmentCountPositive
  segmentCountBound := decoded.segmentCountBound
  realRowCountPositive := statement.resultDecoded.realRowCountPositive
  realRowCountBound := statement.resultDecoded.realRowCountBound
  fitsDeclaredSegments := decoded.realRowsFitDeclaredSegments
  smallestSegmentCount := decoded.smallestSegmentCount
  rowsCanonical := rfl

end PublicImage.Decodes

namespace PublicImage.DecodesFor

theorem initial_state_canonical
    {Program : Type}
    {expectedProfile : Profile.Identity}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement) :
    image.initialApplicationState.Canonical := by
  rw [decoded.exactImage]
  exact (canonical_encode_iff statement.base.initialApplicationState).2
    statement.initialApplicationStateValid

theorem image_profile_exact
    {Program : Type}
    {expectedProfile : Profile.Identity}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement) :
    image.identity.profile = expectedProfile := by
  rw [decoded.exactImage]
  exact decoded.exactProfile

theorem image_segment_count_bound
    {Program : Type}
    {expectedProfile : Profile.Identity}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement) :
    image.segmentCount < 2 ^ segmentCountBitWidth := by
  rw [decoded.exactImage]
  change statement.base.segmentCount < 2 ^ segmentCountBitWidth
  have bound := decoded.segmentCountBound
  norm_num [segmentCountBitWidth, maximumSegments] at bound ⊢
  omega

theorem image_final_timestamp_bound
    {Program : Type}
    {expectedProfile : Profile.Identity}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement) :
    image.finalGlobalTimestamp < 2 ^ finalTimestampBitWidth := by
  rw [decoded.exactImage]
  change statement.base.finalGlobalTimestamp < 2 ^ finalTimestampBitWidth
  exact decoded.finalTimestampBound

theorem image_result_decodes
    {Program : Type}
    {expectedProfile : Profile.Identity}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement) :
    image.result.Decodes statement.base.expectedResult := by
  rw [decoded.exactImage]
  exact statement.resultDecoded

def completionTrace
    {Program : Type}
    {expectedProfile : Profile.Identity}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.DecodesFor expectedProfile statement) :
    Completion.ValidCompletedTrace statement.base.expectedResult
      statement.base.segmentCount
      (Completion.canonicalRows statement.base.expectedResult
        statement.base.segmentCount) where
  segmentCountPositive := decoded.segmentCountPositive
  segmentCountBound := decoded.segmentCountBound
  realRowCountPositive := statement.resultDecoded.realRowCountPositive
  realRowCountBound := statement.resultDecoded.realRowCountBound
  fitsDeclaredSegments := decoded.realRowsFitDeclaredSegments
  smallestSegmentCount := decoded.smallestSegmentCount
  rowsCanonical := rfl

end PublicImage.DecodesFor

def PublicImage.Decodes.toDecodesFor
    {Program : Type}
    {image : PublicImage}
    {statement : ProductionStatement Program}
    (decoded : image.Decodes statement) :
    image.DecodesFor Profile.v2 statement where
  exactImage := decoded.exactImage
  exactProfile := decoded.exactProfile
  segmentCountPositive := decoded.segmentCountPositive
  segmentCountBound := decoded.segmentCountBound
  finalTimestampBound := decoded.finalTimestampBound
  realRowsFitDeclaredSegments := decoded.realRowsFitDeclaredSegments
  smallestSegmentCount := decoded.smallestSegmentCount

end Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
