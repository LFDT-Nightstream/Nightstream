import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyContinuity
import Nightstream.Protocol.FPrime.XOut

/-!
Contract: complete `x_out` continuity for a carried PiRLC family state.

Owns the two compression layers between a public recursive-state digest and
the exact 937-field family state. Equal complete `x_out` values first recover
the stateful semantic digest or an outer binding failure. Equal semantic
digests then recover the family state or the exact local Poseidon2 collision.

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
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Protocol.FPrime

universe uParams uStructure uHeader uRunning uFresh uNebulaDigest

/-- The four-field local family digest carried in the semantic-state lane of
the complete recursive state. -/
abbrev FamilyDigest := Fin 4 -> Field

/-- A complete-output ambiguity can occur only at the outer `x_out` layer or
at the inner framed family-state digest layer. -/
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

/-- Complete two-layer reduction. The local family digest is a component of
`x_out`; it is not itself the public recursive-state digest. -/
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
    (leftPinned :
      XOut.StatePinned semantics leftMode leftContext leftState)
    (rightPinned :
      XOut.StatePinned semantics rightMode rightContext rightState)
    (leftSemantic :
      leftState.semanticState = familyStateDigest leftFamily)
    (rightSemantic :
      rightState.semanticState = familyStateDigest rightFamily)
    (leftCanonical :
      ∀ value, value ∈ familyStateFields leftFamily -> value < goldilocksP)
    (rightCanonical :
      ∀ value, value ∈ familyStateFields rightFamily -> value < goldilocksP)
    (sameOutput :
      XOut.compute semantics leftMode leftContext leftState =
        XOut.compute semantics rightMode rightContext rightState) :
    leftFamily = rightFamily ∨ ContinuityFailure semantics := by
  rcases semantic_digest_eq_or_xOut_failure semantics leftMode rightMode
      leftContext rightContext leftState rightState leftPinned rightPinned
      sameOutput with semanticEqual | failure
  · have digestEqual :
        familyStateDigest leftFamily = familyStateDigest rightFamily :=
      leftSemantic.symm.trans (semanticEqual.trans rightSemantic)
    rcases familyState_eq_or_poseidon2_collision leftFamily rightFamily
        leftCanonical rightCanonical digestEqual with familyEqual | collision
    · exact Or.inl familyEqual
    · exact Or.inr (.familyState collision)
  · exact Or.inr (.xOut failure)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity
