import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyContinuity
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeSchema
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.PureSponge
import Nightstream.Protocol.FPrime.XOut

/-!
Contract: complete `x_out` continuity for a carried PiRLC family state.

Owns the three compression layers between a public recursive-state digest and
the exact 937-field family state. Equal complete `x_out` values first recover
the stateful semantic digest or an outer binding failure. Equal phase-envelope
digests with one common fixed-width payload then recover equal local digests or
an exact phase-envelope Poseidon2 collision. Equal local digests finally
recover the family state or the exact local Poseidon2 collision.

Does not own the bridge from physical phase-envelope rows to typed outer-state
authority, Rust/R1CS assignment conformance, adjacent shared-wire enforcement,
start or finish circuits, or collision resistance.

Emits constraints: no.

Assurance tier: security-reduced. This theorem names every compression
failure. It does not assume that a digest is authority.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Protocol.FPrime

universe uParams uStructure uHeader uRunning uFresh uNebulaDigest

/-- The four-field local family digest carried in the semantic-state lane of
the complete recursive state. -/
abbrev FamilyDigest := Fin 4 -> Field

/-- Exact fixed-width delayed payload absorbed after one local family digest. -/
structure PhasePayload where
  values : List Nat
  length_eq :
    values.length =
      FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.payloadFields
  canonical : forall value, value ∈ values -> value < goldilocksP

@[ext] theorem PhasePayload.ext {left right : PhasePayload}
    (values : left.values = right.values) : left = right := by
  cases left
  cases right
  simp_all

/-- Canonical natural values of one four-lane local digest. -/
def localDigestValues (digest : FamilyDigest) : List Nat :=
  List.ofFn fun lane => (digest lane).val

/-- Exact fixed-width phase-envelope Poseidon2 preimage. -/
def phaseEnvelopePreimage
    (localDigest : FamilyDigest) (payload : PhasePayload) : List Nat :=
  Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope.phaseConstantValues ++
    localDigestValues localDigest ++ payload.values

/-- Exact selected Poseidon2 digest of the local digest and delayed payload. -/
def phaseEnvelopeDigest
    (localDigest : FamilyDigest) (payload : PhasePayload) : FamilyDigest :=
  fun lane => fieldValue
    (Poseidon2Sponge.digest Poseidon2CanonicalConstants.selected
      (Poseidon2PureSponge.fullRateChunks
        (phaseEnvelopePreimage localDigest payload)
        FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.absorbRounds)
      lane)

/-- Witness for a collision in the exact fixed-width phase-envelope domain.
The complete typed inputs differ in the local family digest, the delayed
payload, or both. -/
structure Poseidon2PhaseEnvelopeCollisionWitness where
  leftLocal : FamilyDigest
  rightLocal : FamilyDigest
  leftPayload : PhasePayload
  rightPayload : PhasePayload
  inputDifferent : leftLocal ≠ rightLocal ∨ leftPayload ≠ rightPayload
  digestEqual :
    phaseEnvelopeDigest leftLocal leftPayload =
      phaseEnvelopeDigest rightLocal rightPayload

/-- Named security-reduction event for the exact phase-envelope Poseidon2
application domain. -/
def Poseidon2PhaseEnvelopeCollision : Prop :=
  Nonempty Poseidon2PhaseEnvelopeCollisionWitness

/-- Equal phase-envelope digests recover both complete typed inputs, or
exhibit a collision in that exact application domain. -/
theorem phase_preimage_eq_or_collision
    (leftLocal rightLocal : FamilyDigest)
    (leftPayload rightPayload : PhasePayload)
    (digestEqual :
      phaseEnvelopeDigest leftLocal leftPayload =
        phaseEnvelopeDigest rightLocal rightPayload) :
    (leftLocal = rightLocal ∧ leftPayload = rightPayload) ∨
      Poseidon2PhaseEnvelopeCollision := by
  by_cases localEqual : leftLocal = rightLocal
  · by_cases payloadEqual : leftPayload = rightPayload
    · exact Or.inl ⟨localEqual, payloadEqual⟩
    · exact Or.inr ⟨{
        leftLocal := leftLocal
        rightLocal := rightLocal
        leftPayload := leftPayload
        rightPayload := rightPayload
        inputDifferent := Or.inr payloadEqual
        digestEqual := digestEqual }⟩
  · exact Or.inr ⟨{
      leftLocal := leftLocal
      rightLocal := rightLocal
      leftPayload := leftPayload
      rightPayload := rightPayload
      inputDifferent := Or.inl localEqual
      digestEqual := digestEqual }⟩

/-- Equal phase-envelope digests over one common payload recover equal local
digests, or exhibit a collision in that exact application domain. -/
theorem local_digest_eq_or_phase_envelope_collision
    (left right : FamilyDigest) (payload : PhasePayload)
    (digestEqual :
      phaseEnvelopeDigest left payload = phaseEnvelopeDigest right payload) :
    left = right ∨ Poseidon2PhaseEnvelopeCollision := by
  rcases phase_preimage_eq_or_collision left right payload payload digestEqual with
    preimageExact | collision
  · exact Or.inl preimageExact.1
  · exact Or.inr collision

/-- A complete-output ambiguity can occur at the outer `x_out` layer, at the
phase-envelope layer, or at the framed family-state digest layer. -/
inductive ContinuityFailure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest) : Prop where
  | xOut (failure : XOut.BindingFailure semantics)
  | phaseEnvelope (collision : Poseidon2PhaseEnvelopeCollision)
  | familyState (collision : Poseidon2FamilyStateCollision)

/-- First layer: equal complete recursive-state outputs recover equality of
the semantic-state lane, or expose an outer `x_out` binding failure. -/
theorem semantic_digest_eq_or_xOut_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest)
    (leftMode rightMode : XOut.Mode)
    (leftContext rightContext :
      XOut.Context Params StructureDigest Header FamilyDigest)
    (leftState rightState : State FamilyDigest Running Fresh Nebula)
    (leftPinned :
      XOut.StatePinned semantics leftMode leftContext leftState)
    (rightPinned :
      XOut.StatePinned semantics rightMode rightContext rightState)
    (sameOutput :
      XOut.compute semantics leftMode leftContext leftState =
        XOut.compute semantics rightMode rightContext rightState) :
    leftState.semanticState = rightState.semanticState ∨
      XOut.BindingFailure semantics := by
  rcases XOut.xOut_binding_or_collision semantics leftMode rightMode
      leftContext rightContext leftState rightState leftPinned rightPinned
      sameOutput with authorityEqual | failure
  · left
    exact congrArg (fun authority => authority.semanticState) authorityEqual
  · exact Or.inr failure

/-- Complete three-layer reduction for two independently supplied phase
preimages. Equal complete outputs recover the full family state and delayed
payload, or expose one named compression failure. -/
theorem familyState_payload_eq_or_continuity_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest)
    (leftMode rightMode : XOut.Mode)
    (leftContext rightContext :
      XOut.Context Params StructureDigest Header FamilyDigest)
    (leftState rightState : State FamilyDigest Running Fresh Nebula)
    (leftFamily rightFamily : FamilyState)
    (leftPayload rightPayload : PhasePayload)
    (leftPinned :
      XOut.StatePinned semantics leftMode leftContext leftState)
    (rightPinned :
      XOut.StatePinned semantics rightMode rightContext rightState)
    (leftSemantic :
      leftState.semanticState =
        phaseEnvelopeDigest (familyStateDigest leftFamily) leftPayload)
    (rightSemantic :
      rightState.semanticState =
        phaseEnvelopeDigest (familyStateDigest rightFamily) rightPayload)
    (leftCanonical :
      ∀ value, value ∈ familyStateFields leftFamily -> value < goldilocksP)
    (rightCanonical :
      ∀ value, value ∈ familyStateFields rightFamily -> value < goldilocksP)
    (sameOutput :
      XOut.compute semantics leftMode leftContext leftState =
        XOut.compute semantics rightMode rightContext rightState) :
    (leftFamily = rightFamily ∧ leftPayload = rightPayload) ∨
      ContinuityFailure semantics := by
  rcases semantic_digest_eq_or_xOut_failure semantics leftMode rightMode
      leftContext rightContext leftState rightState leftPinned rightPinned
      sameOutput with semanticEqual | failure
  · have envelopeEqual :
        phaseEnvelopeDigest (familyStateDigest leftFamily) leftPayload =
          phaseEnvelopeDigest (familyStateDigest rightFamily) rightPayload :=
      leftSemantic.symm.trans (semanticEqual.trans rightSemantic)
    rcases phase_preimage_eq_or_collision
        (familyStateDigest leftFamily) (familyStateDigest rightFamily)
        leftPayload rightPayload envelopeEqual with preimageExact | collision
    · rcases familyState_eq_or_poseidon2_collision leftFamily rightFamily
          leftCanonical rightCanonical preimageExact.1 with familyEqual | collision
      · exact Or.inl ⟨familyEqual, preimageExact.2⟩
      · exact Or.inr (.familyState collision)
    · exact Or.inr (.phaseEnvelope collision)
  · exact Or.inr (.xOut failure)

/-- Complete three-layer reduction. The local family digest is absorbed with
the common delayed payload before it reaches the semantic lane of `x_out`; it
is not itself the public recursive-state digest. -/
theorem familyState_eq_or_continuity_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest)
    (leftMode rightMode : XOut.Mode)
    (leftContext rightContext :
      XOut.Context Params StructureDigest Header FamilyDigest)
    (leftState rightState : State FamilyDigest Running Fresh Nebula)
    (leftFamily rightFamily : FamilyState)
    (payload : PhasePayload)
    (leftPinned :
      XOut.StatePinned semantics leftMode leftContext leftState)
    (rightPinned :
      XOut.StatePinned semantics rightMode rightContext rightState)
    (leftSemantic :
      leftState.semanticState =
        phaseEnvelopeDigest (familyStateDigest leftFamily) payload)
    (rightSemantic :
      rightState.semanticState =
        phaseEnvelopeDigest (familyStateDigest rightFamily) payload)
    (leftCanonical :
      ∀ value, value ∈ familyStateFields leftFamily -> value < goldilocksP)
    (rightCanonical :
      ∀ value, value ∈ familyStateFields rightFamily -> value < goldilocksP)
    (sameOutput :
      XOut.compute semantics leftMode leftContext leftState =
        XOut.compute semantics rightMode rightContext rightState) :
    leftFamily = rightFamily ∨ ContinuityFailure semantics := by
  rcases familyState_payload_eq_or_continuity_failure semantics leftMode
      rightMode leftContext rightContext leftState rightState leftFamily
      rightFamily payload payload leftPinned rightPinned leftSemantic
      rightSemantic leftCanonical rightCanonical sameOutput with stateExact | failure
  · exact Or.inl stateExact.1
  · exact Or.inr failure

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity
