import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
import Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics

/-!
Construction-2 refinement through the parallel paper-exact selected edge.

Assurance tier: model-level.

Owns: projection of the existing active semantic result into the paper-exact
selected NIFS family, and reuse of the generic Construction-2 soundness
theorem for that family.

Does not own: physical acceptance, child openings, deterministic child
equality as paper authority, transcript replay, Rust, R1CS, costs, or row
removal.

Authority boundary: this is a one-way compatibility theorem. The independent
paper family is defined without the richer active result; the proof below
forgets the richer result's implementation sidecars before invoking the
generic outer theorem.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- The existing active semantic result projects to the parallel paper-exact
selected family. No new refinement premise is introduced. -/
theorem paperSelectedNifsRefinement
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount) :
    Refinement setup
      (PaperSelectedNifsSemantics.family (selectedNifsSetup setup)) := by
  refine {
    expectedStructure := ?_
    transition := ?_
  }
  · intro key slot
    rfl
  · intro input slot result accepted
    apply PaperSelectedNifsSemantics.transition_of_result
      (incomingParent := (input.running slot).parent)
      (polynomial := setup.piCcsInput input slot)
      (priorState := setup.priorTranscriptState input slot)
    simpa [PaperSelectedNifsSemantics.contextOf,
      SelectedNifsSemantics.contextOf, selectedNifsSetup, contextAt,
      invocationAt]
      using accepted

/-- Every accepted active semantic execution satisfies Construction 2 with
the paper-exact selected family installed. -/
theorem sound_paperSelectedNifs
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount}
    {machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (accepted : Holds setup machine functionIndex input output) :
    Paper.Construction2.RecursiveHolds
      (PaperSelectedNifsSemantics.family (selectedNifsSetup setup)) machine
      functionIndex input.toPaper output.toPaper :=
  sound (paperSelectedNifsRefinement setup) accepted

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
