import Nightstream.Implementation.NebulaV2.Memory.Transcript.PoseidonRows
import Nightstream.Protocol.NebulaV2.Soundness

/-!
Contract: concrete V2 specialization of the semantic segment-open transition.

Assurance tier: implementation model.

Owns the seven authority digests, exact transcript input derived from a closed
carry and precommit roots, fixed Poseidon2 challenge derivation, and the
specialized `FPrime.openSegment` result.

Does not own authority-column placement, transcript rows, precommit
extraction, Fiat--Shamir unpredictability, or Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryOpenSegment

open Nightstream.Implementation.NebulaV2.MemoryTranscriptHashFrame
open Nightstream.Implementation.NebulaV2.MemoryTranscriptPoseidonRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

local instance concreteKOne : One K := ⟨K.one⟩

@[ext] structure Authority where
  verifierKeyDigest : Digest.Value
  applicationRelationDigest : Digest.Value
  programDigest : Digest.Value
  memoryPlanDigest : Digest.Value
  laneLayoutDigest : Digest.Value
  priorStateDigest : Digest.Value
  runningAccumulatorDigest : Digest.Value
deriving DecidableEq

def Authority.digestFields (authority : Authority) : List Nat :=
  encodeDigests
    [ authority.verifierKeyDigest
    , authority.applicationRelationDigest
    , authority.programDigest
    , authority.memoryPlanDigest
    , authority.laneLayoutDigest
    , authority.priorStateDigest
    , authority.runningAccumulatorDigest
    ]

def Authority.digestAt (authority : Authority) : Fin 7 -> Digest.Value
  | ⟨0, _⟩ => authority.verifierKeyDigest
  | ⟨1, _⟩ => authority.applicationRelationDigest
  | ⟨2, _⟩ => authority.programDigest
  | ⟨3, _⟩ => authority.memoryPlanDigest
  | ⟨4, _⟩ => authority.laneLayoutDigest
  | ⟨5, _⟩ => authority.priorStateDigest
  | ⟨6, _⟩ => authority.runningAccumulatorDigest

theorem Authority.digestFields_eq_indexed (authority : Authority) :
    authority.digestFields =
      (List.ofFn authority.digestAt).flatMap
        MemoryTranscriptHashFrame.digestFields := by
  simp [Authority.digestFields, Authority.digestAt,
    MemoryTranscriptHashFrame.encodeDigests]

/-- The ordered 28-field authority frame is lossless before hashing. -/
theorem Authority.digestFields_injective :
    Function.Injective Authority.digestFields := by
  intro left right equal
  have digestListEqual :
      [ left.verifierKeyDigest
      , left.applicationRelationDigest
      , left.programDigest
      , left.memoryPlanDigest
      , left.laneLayoutDigest
      , left.priorStateDigest
      , left.runningAccumulatorDigest ] =
        [ right.verifierKeyDigest
        , right.applicationRelationDigest
        , right.programDigest
        , right.memoryPlanDigest
        , right.laneLayoutDigest
        , right.priorStateDigest
        , right.runningAccumulatorDigest ] := by
    apply MemoryTranscriptHashFrame.encodeDigests_injective
    exact equal
  simp only [List.cons.injEq] at digestListEqual
  rcases digestListEqual with
    ⟨verifierKey, applicationRelation, program, memoryPlan, laneLayout,
      priorState, runningAccumulator, _⟩
  exact Authority.ext verifierKey applicationRelation program memoryPlan
    laneLayout priorState runningAccumulator

/-- The unique memory-challenge authority selected by a verifier-owned
statement identity and the two authenticated dynamic F-prime digests. -/
def Authority.ofIdentityAndState
    (identity :
      Nightstream.Protocol.NebulaV2.Soundness.StatementIdentity Digest.Value)
    (priorStateDigest runningAccumulatorDigest : Digest.Value) : Authority :=
  { verifierKeyDigest := identity.verifierKey.digest
    applicationRelationDigest := identity.applicationRelationDigest
    programDigest := identity.programDigest
    memoryPlanDigest := identity.memoryPlanDigest
    laneLayoutDigest := identity.verifierKey.laneLayoutDigest
    priorStateDigest := priorStateDigest
    runningAccumulatorDigest := runningAccumulatorDigest }

@[simp] theorem Authority.ofIdentityAndState_digestFields
    (identity :
      Nightstream.Protocol.NebulaV2.Soundness.StatementIdentity Digest.Value)
    (priorStateDigest runningAccumulatorDigest : Digest.Value) :
    (Authority.ofIdentityAndState identity priorStateDigest
        runningAccumulatorDigest).digestFields =
      encodeDigests
        [ identity.verifierKey.digest
        , identity.applicationRelationDigest
        , identity.programDigest
        , identity.memoryPlanDigest
        , identity.verifierKey.laneLayoutDigest
        , priorStateDigest
        , runningAccumulatorDigest
        ] :=
  rfl

def transcriptInput
    (authority : Authority)
    (closed : ClosedCarry Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat) : Input where
  verifierKeyDigest := authority.verifierKeyDigest
  applicationRelationDigest := authority.applicationRelationDigest
  programDigest := authority.programDigest
  memoryPlanDigest := authority.memoryPlanDigest
  laneLayoutDigest := authority.laneLayoutDigest
  priorStateDigest := authority.priorStateDigest
  runningAccumulatorDigest := authority.runningAccumulatorDigest
  segmentIndex := closed.segmentIndex
  segmentStartTimestamp := closed.globalTimestamp
  activeAccessCount := activeAccessCount
  segmentEndTimestamp := closed.globalTimestamp + activeAccessCount
  roots := precommit

theorem transcriptInput_authority_fields
    (authority : Authority)
    (closed : ClosedCarry Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat) :
    authorityDigestFields
        (transcriptInput authority closed precommit activeAccessCount) =
      authority.digestFields :=
  rfl

/-- Challenge derivation for one verifier-selected statement profile. -/
def deriveFor
    (profile : Profile.Identity)
    (authority : Authority)
    (closed : ClosedCarry Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat) : ProductState.Challenges K :=
  MemoryTranscriptPoseidonRows.ProfileIndexed.pureChallenges profile
    (transcriptInput authority closed precommit activeAccessCount)

/-- The reference V2 challenge derivation. -/
def derive
    (authority : Authority)
    (closed : ClosedCarry Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat) : ProductState.Challenges K :=
  deriveFor Profile.v2 authority closed precommit activeAccessCount

@[simp] theorem deriveFor_v2
    (authority : Authority)
    (closed : ClosedCarry Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat) :
    deriveFor Profile.v2 authority closed precommit activeAccessCount =
      derive authority closed precommit activeAccessCount :=
  rfl

def openCarryFor
    (profile : Profile.Identity)
    (authority : Authority)
    (headers : ChainHeaders Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest.Value)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    Carry Digest.Value (ProductState.Challenges K) (ProductState.State K) :=
  FPrime.openSegment
    (fun closed precommit activeAccessCount =>
      deriveFor profile authority closed precommit activeAccessCount)
    headers precommit activeAccessCount closed canOpen activeCountInRange
    endTimestampInRange

def openCarry
    (authority : Authority)
    (headers : ChainHeaders Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest.Value)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    Carry Digest.Value (ProductState.Challenges K) (ProductState.State K) :=
  openCarryFor Profile.v2 authority headers precommit activeAccessCount closed
    canOpen activeCountInRange endTimestampInRange

@[simp] theorem openCarryFor_v2
    (authority : Authority)
    (headers : ChainHeaders Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest.Value)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    openCarryFor Profile.v2 authority headers precommit activeAccessCount
        closed canOpen activeCountInRange endTimestampInRange =
      openCarry authority headers precommit activeAccessCount closed canOpen
        activeCountInRange endTimestampInRange :=
  rfl

theorem open_exact_for
    (profile : Profile.Identity)
    (authority : Authority)
    (headers : ChainHeaders Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest.Value)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    openCarryFor profile authority headers precommit activeAccessCount closed canOpen
        activeCountInRange endTimestampInRange =
      .active
        { segmentIndex := closed.segmentIndex
          stepIndex := ⟨0, by decide⟩
          globalTimestamp := closed.globalTimestamp
          segmentStartTimestamp := closed.globalTimestamp
          segmentActiveAccessCount := activeAccessCount
          segmentEndTimestamp := closed.globalTimestamp + activeAccessCount
          challenge := deriveFor profile authority closed precommit activeAccessCount
          products := ProductState.one
          dPre := precommit
          dSeen := headers.roots
          memoryRoot := closed.memoryRoot } :=
  rfl

theorem open_exact
    (authority : Authority)
    (headers : ChainHeaders Digest.Value)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (closed : ClosedCarry Digest.Value)
    (canOpen : closed.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      closed.globalTimestamp + activeAccessCount < timestampLimit) :
    openCarry authority headers precommit activeAccessCount closed canOpen
        activeCountInRange endTimestampInRange =
      .active
        { segmentIndex := closed.segmentIndex
          stepIndex := ⟨0, by decide⟩
          globalTimestamp := closed.globalTimestamp
          segmentStartTimestamp := closed.globalTimestamp
          segmentActiveAccessCount := activeAccessCount
          segmentEndTimestamp := closed.globalTimestamp + activeAccessCount
          challenge := derive authority closed precommit activeAccessCount
          products := ProductState.one
          dPre := precommit
          dSeen := headers.roots
          memoryRoot := closed.memoryRoot } := by
  exact open_exact_for Profile.v2 authority headers precommit activeAccessCount
    closed canOpen activeCountInRange endTimestampInRange

end Nightstream.Implementation.NebulaV2.MemoryOpenSegment
