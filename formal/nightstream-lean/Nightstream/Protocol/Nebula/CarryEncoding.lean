import Nightstream.Protocol.Nebula.FPrime

/-!
Contract: canonical fixed-shape memory-carry encoding for Nebula V2.

Assurance tier: model-level.

Owns the complete wire shape for both active and closed carries. A closed
encoding fixes every inactive field: step and segment counters are zero,
challenge coordinates are zero, products are one, and both root accumulators
equal the verifier-owned chain headers.

Does not own byte serialization, field-limb codecs, generated rows, or the
state-hash implementation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.CarryEncoding

open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Lifecycle

inductive PhaseTag where
  | closed
  | active
deriving DecidableEq, Repr

/-- Fixed wire shape included in the recursive state-hash preimage. -/
@[ext]
structure WireCarry (Digest ChallengeField : Type) where
  phase : PhaseTag
  segmentIndex : Nat
  stepIndex : Nat
  globalTimestamp : Nat
  segmentStartTimestamp : Nat
  segmentActiveAccessCount : Nat
  segmentEndTimestamp : Nat
  challenges : ProductState.Challenges ChallengeField
  products : ProductState.State ChallengeField
  dPre : Roots Digest
  dSeen : Roots Digest
  memoryRoot : Digest
deriving DecidableEq, Repr

def zeroChallenges
    {ChallengeField : Type} [Zero ChallengeField] :
    ProductState.Challenges ChallengeField :=
  fun _ => { gamma1 := 0, gamma2 := 0 }

def encodeClosed
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    (headers : ChainHeaders Digest)
    (closed : ClosedCarry Digest) : WireCarry Digest ChallengeField :=
  { phase := .closed
    segmentIndex := closed.segmentIndex
    stepIndex := 0
    globalTimestamp := closed.globalTimestamp
    segmentStartTimestamp := 0
    segmentActiveAccessCount := 0
    segmentEndTimestamp := 0
    challenges := zeroChallenges
    products := ProductState.one
    dPre := headers.roots
    dSeen := headers.roots
    memoryRoot := closed.memoryRoot }

def encodeActive
    {Digest ChallengeField : Type}
    (active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)) : WireCarry Digest ChallengeField :=
  { phase := .active
    segmentIndex := active.segmentIndex
    stepIndex := active.stepIndex.val
    globalTimestamp := active.globalTimestamp
    segmentStartTimestamp := active.segmentStartTimestamp
    segmentActiveAccessCount := active.segmentActiveAccessCount
    segmentEndTimestamp := active.segmentEndTimestamp
    challenges := active.challenge
    products := active.products
    dPre := active.dPre
    dSeen := active.dSeen
    memoryRoot := active.memoryRoot }

def ClosedFieldsCanonical
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    (headers : ChainHeaders Digest)
    (wire : WireCarry Digest ChallengeField) : Prop :=
  wire.stepIndex = 0 ∧
    wire.segmentStartTimestamp = 0 ∧
    wire.segmentActiveAccessCount = 0 ∧
    wire.segmentEndTimestamp = 0 ∧
    wire.challenges = zeroChallenges ∧
    wire.products = ProductState.one ∧
    wire.dPre = headers.roots ∧
    wire.dSeen = headers.roots

/-- Normative decoding relation for the fixed shape. A byte or field decoder
is conformant only if it produces this relation. No constructor exists for a
noncanonical closed wire value or an out-of-range active step index. -/
inductive Decodes
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    (headers : ChainHeaders Digest) :
    WireCarry Digest ChallengeField →
      Carry Digest (ProductState.Challenges ChallengeField)
        (ProductState.State ChallengeField) → Prop
  | closed (closed : ClosedCarry Digest) :
      Decodes headers (encodeClosed headers closed) (.closed closed)
  | active
      (active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
        (ProductState.State ChallengeField)) :
      Decodes headers (encodeActive active) (.active active)

theorem closedFieldsCanonical_encodeClosed
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    (headers : ChainHeaders Digest)
    (closed : ClosedCarry Digest) :
    ClosedFieldsCanonical headers
      (encodeClosed (ChallengeField := ChallengeField) headers closed) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem decodes_encodeClosed
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    (headers : ChainHeaders Digest)
    (closed : ClosedCarry Digest) :
    Decodes headers
      (encodeClosed (ChallengeField := ChallengeField) headers closed)
      (.closed closed) :=
  .closed closed

theorem decodes_encodeActive
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    (headers : ChainHeaders Digest)
    (active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)) :
    Decodes headers (encodeActive active) (.active active) :=
  .active active

/-- Every successfully decoded closed value has all required canonical
inactive fields. This is not an assumption of the returned carry. -/
theorem canonical_of_decodes_closed
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    {headers : ChainHeaders Digest}
    {wire : WireCarry Digest ChallengeField}
    {closed : ClosedCarry Digest}
    (decoded : Decodes headers wire (.closed closed)) :
    ClosedFieldsCanonical headers wire := by
  cases decoded
  exact closedFieldsCanonical_encodeClosed headers closed

theorem closed_decodes_exact
    {Digest ChallengeField : Type}
    [Zero ChallengeField] [One ChallengeField]
    {headers : ChainHeaders Digest}
    {wire : WireCarry Digest ChallengeField}
    {closed : ClosedCarry Digest}
    (decoded : Decodes headers wire (.closed closed)) :
    wire = encodeClosed headers closed := by
  cases decoded
  rfl

end Nightstream.Protocol.Nebula.CarryEncoding
