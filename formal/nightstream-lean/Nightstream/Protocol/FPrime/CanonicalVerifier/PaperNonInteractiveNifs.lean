import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
import Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement

/-!
Layer-safe HyperNova/F-prime adapter for the paper SuperNeo NIFS verifier.

Owns: the exact `NIFS.V` adapter, the deterministic core of frozen
obligation 5, a Construction-2 setup using that verifier, and no-premise
specialization of the two augmented-function refinement directions.

Does not own: a concrete Poseidon2/Ajtai instantiation, event probabilities,
default-instance validity, terminal relation checks, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.

HyperNova Construction 2 consumes one fresh claim.  Every setup and
specialized theorem therefore carries the explicit proof
`shape.freshCount = 1`; the generalized SuperNeo verifier itself remains
well-typed for arbitrary `K`.

| Construction-2 phase | Mathematical obligation | Lean owner |
|---|---|---|
| selected fold | adapt the exact paper NIFS checker to HyperNova's `NIFS.V` interface | `nifsVerifier` |
| NIFS boundary | instantiate deterministic soundness modulo five events and completeness without a correctness premise | `nifsSoundAndCompleteModulo` |
| setup | require exactly one fresh claim as in Construction 2 | `construction2Setup` |
| recursive branch | accepted `F'_j` implies the paper NIFS transition or a named NIFS event | `canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent` |
| honest recursive branch | a paper NIFS transition constructs an accepted `F'_j` proof | `canonicalFprime_paperTransition_implies_exists_nifsProof_accepts` |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Protocol.FPrime.Frozen.Obligations
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

universe uExtension uCommitment uPublicInput uScalar uTranscriptState
  uDigest uAppState uWitness uEncoded

/-- HyperNova's deterministic one-message verifier interface, implemented by
the paper SuperNeo checker. -/
def nifsVerifier
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {TranscriptState : Type uTranscriptState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat} :
    Verifier
      (Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
      (Running Extension Commitment PublicInput shape)
      (Fresh Commitment PublicInput shape)
      (Proof Extension Commitment shape degreeBound) where
  verify :=
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify

/-- Deterministic core of obligation 5: the executable paper NIFS is sound
modulo exactly the five named events and complete for the independently
expanded equations. No correctness bundle is a premise. The quantitative
non-interactive target is discharged separately from the eleven exact event
bounds. -/
theorem nifsSoundAndCompleteModulo
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {TranscriptState : Type uTranscriptState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat} :
    NifsSoundAndCompleteModulo
      (nifsVerifier (Extension := Extension) (Commitment := Commitment)
        (PublicInput := PublicInput) (Scalar := Scalar)
        (TranscriptState := TranscriptState) (shape := shape)
        (columns := columns) (blockCount := blockCount)
        (degreeBound := degreeBound))
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent := by
  constructor
  · intro key running fresh proof result accepted
    exact Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify_sound
      key running fresh proof result accepted
  · intro key running fresh result transition
    exact Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify_complete
      key running fresh result transition

/-- Construction-2 setup pinned to the paper SuperNeo NIFS and the required
single-fresh profile. -/
def construction2Setup
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {TranscriptState : Type uTranscriptState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound slotCount : Nat}
    (_oneFresh : shape.freshCount = 1)
    (keys : Fin slotCount ->
      Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
    (defaultRunning : Running Extension Commitment PublicInput shape) :
    Setup
      (Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
      (Running Extension Commitment PublicInput shape)
      (Fresh Commitment PublicInput shape)
      (Proof Extension Commitment shape degreeBound)
      slotCount where
  verifierKeys := keys
  nifs := nifsVerifier
  defaultRunning := defaultRunning

/-- Canonical Construction-2 soundness with the concrete paper NIFS pinned.
There is no caller-supplied NIFS transition, bad-event predicate, or
correctness premise. -/
theorem canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {TranscriptState : Type uTranscriptState}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {Encoded : Type uEncoded}
    [DecidableEq Extension]
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {shape : Shape}
    {columns blockCount degreeBound slotCount : Nat}
    (oneFresh : shape.freshCount = 1)
    (keys : Fin slotCount ->
      Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
    (defaultRunning : Running Extension Commitment PublicInput shape)
    (machine : Machine
      (Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
      Digest AppState Witness
      (Running Extension Commitment PublicInput shape)
      (Fresh Commitment PublicInput shape)
      Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input
      (Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
      AppState Witness
      (Running Extension Commitment PublicInput shape)
      (Fresh Commitment PublicInput shape)
      (Proof Extension Commitment shape degreeBound) slotCount)
    (output : Output Digest AppState
      (Running Extension Commitment PublicInput shape) slotCount)
    (accepted : Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
      (construction2Setup oneFresh keys defaultRunning) machine functionIndex
      input output) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        (construction2Setup oneFresh keys defaultRunning) machine
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
        functionIndex input output ∨
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SelectedNifsBadEvent
        (construction2Setup oneFresh keys defaultRunning)
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
        input output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.accepts_implies_semanticTransition_or_selectedNifsBadEvent
    (construction2Setup oneFresh keys defaultRunning) machine
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
    nifsSoundAndCompleteModulo functionIndex input output accepted

/-- Every semantic Construction-2 step using the concrete paper NIFS admits
one replacement NIFS message accepted by the canonical executable evaluator. -/
theorem canonicalFprime_paperTransition_implies_exists_nifsProof_accepts
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {TranscriptState : Type uTranscriptState}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {Encoded : Type uEncoded}
    [DecidableEq Extension]
    [DecidableEq AppState]
    [DecidableEq Encoded]
    {shape : Shape}
    {columns blockCount degreeBound slotCount : Nat}
    (oneFresh : shape.freshCount = 1)
    (keys : Fin slotCount ->
      Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
    (defaultRunning : Running Extension Commitment PublicInput shape)
    (machine : Machine
      (Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
      Digest AppState Witness
      (Running Extension Commitment PublicInput shape)
      (Fresh Commitment PublicInput shape)
      Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input
      (Key Extension Commitment PublicInput Scalar TranscriptState shape
        columns blockCount degreeBound)
      AppState Witness
      (Running Extension Commitment PublicInput shape)
      (Fresh Commitment PublicInput shape)
      (Proof Extension Commitment shape degreeBound) slotCount)
    (output : Output Digest AppState
      (Running Extension Commitment PublicInput shape) slotCount)
    (semantic :
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        (construction2Setup oneFresh keys defaultRunning) machine
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
        functionIndex input output) :
    exists nifsProof : Proof Extension Commitment shape degreeBound,
      Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
        (construction2Setup oneFresh keys defaultRunning) machine functionIndex
        (Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.withNifsProof
          input nifsProof)
        output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.semanticTransition_implies_exists_nifsProof_accepts
    (construction2Setup oneFresh keys defaultRunning) machine
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
    nifsSoundAndCompleteModulo functionIndex input output semantic

end Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
