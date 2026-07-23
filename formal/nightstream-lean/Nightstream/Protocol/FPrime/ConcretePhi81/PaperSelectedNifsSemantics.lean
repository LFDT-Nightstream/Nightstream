import Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics

/-!
Paper-exact selected-NIFS edge for the ConcretePhi81 Construction-2 instance.

Assurance tier: model-level.

Owns: the public selected source, existential internal verifier context, and
the exact fixed-active paper-profile transition to the public child vector.

Does not own: the richer cached-parent result, deterministic child openings,
canonical private child equality, physical acceptance, transcript replay,
Rust, R1CS, costs, or row removal.

Authority boundary: the internal parent, Split-NC input, and transcript prefix
remain existential witnesses used only to build the verifier context. The
semantic target is the paper verifier's actual child family. No outgoing
parent or `ChildOpenings` premise is present.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.paper_exact.nifs.setup` | reuse the key/slot-owned verifier setup | verifier setup | `Setup` |
| `fprime.paper_exact.nifs.source` | expose one fresh claim and the selected public children | direct dataflow | `Source` |
| `fprime.paper_exact.nifs.internal` | keep parent, polynomial input, and transcript prefix internal | existential semantic witness | `Transition` |
| `fprime.paper_exact.nifs.transition` | accept exactly the fixed-active paper-profile transition to actual public children | independent semantic target | `Transition` |
| `fprime.paper_exact.nifs.family` | install the paper-exact edge as a Construction-2 family | computed | `family` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics

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

universe uOuterKey uTranscriptState

/-- Static setup is shared with the existing selected edge; only the semantic
transition installed below differs. -/
abbrev Setup
    (OuterKey : Type uOuterKey)
    (TranscriptState : Type uTranscriptState)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) :=
  SelectedNifsSemantics.Setup OuterKey TranscriptState shape
    publicRingColumns publicFits verifierRows slotCount

/-- Exact public selected source from HyperNova Construction 2. -/
abbrev Source
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  SelectedNifsSemantics.Source shape publicRingColumns publicFits verifierRows

/-- Exact verifier-visible child family returned by the selected NIFS edge. -/
abbrev Target
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  SelectedNifsSemantics.Target shape publicRingColumns publicFits verifierRows

/-- Reuse the authority-preserving context constructor. This projection adds
no semantic premise and does not expose the cached outgoing parent. -/
abbrev contextOf
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount)
    (key : OuterKey)
    (slot : Fin slotCount)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (incomingParent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (polynomial : PiCCS.SplitNc.Verifier.PublicInput shape)
    (priorState : TranscriptState) :=
  SelectedNifsSemantics.contextOf setup key slot source incomingParent
    polynomial priorState

/-- Public paper-exact ConcretePhi81 NIFS transition.

Only the actual public child vector is returned. The parent cache and the
inputs needed to interpret a physical certificate are internal witnesses. -/
def Transition
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount)
    (key : OuterKey)
    (slot : Fin slotCount)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (target : Target shape publicRingColumns publicFits verifierRows) : Prop :=
  exists incomingParent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows),
    exists polynomial : PiCCS.SplitNc.Verifier.PublicInput shape,
      exists priorState : TranscriptState,
        let context := contextOf setup key slot source incomingParent
          polynomial priorState
        FixedActive.PaperProfile.Transition
          (FixedActive.paperProfileOf context) context.input target

/-- Install the paper-exact public edge in the abstract Construction-2 NIFS
interface without changing the existing richer family. -/
def family
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount) :
    Paper.Construction2.Family OuterKey
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount where
  expectedStructure := setup.expectedStructure
  transition := Transition setup

@[simp] theorem family_expectedStructure
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount)
    (key : OuterKey)
    (slot : Fin slotCount) :
    (family setup).expectedStructure key slot =
      setup.expectedStructure key slot := rfl

@[simp] theorem family_transition
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount)
    (key : OuterKey)
    (slot : Fin slotCount)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (target : Target shape publicRingColumns publicFits verifierRows) :
    (family setup).transition key slot source target =
      Transition setup key slot source target := rfl

/-- A concrete paper-profile transition supplies exactly the internal
witnesses required by the public selected edge. -/
theorem transition_of_paper
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount}
    {key : OuterKey}
    {slot : Fin slotCount}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {incomingParent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    {polynomial : PiCCS.SplitNc.Verifier.PublicInput shape}
    {priorState : TranscriptState}
    {target : Target shape publicRingColumns publicFits verifierRows}
    (accepted :
      let context := contextOf setup key slot source incomingParent polynomial
        priorState
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf context) context.input target) :
    Transition setup key slot source target := by
  exact ⟨incomingParent, polynomial, priorState, accepted⟩

/-- The existing richer result relation forgets its parent cache and
deterministic-child strengthening at this boundary. This theorem is a
projection only; the definition of the paper-exact edge does not depend on the
richer relation. -/
theorem transition_of_result
    {OuterKey : Type uOuterKey}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey TranscriptState shape publicRingColumns
      publicFits verifierRows slotCount}
    {key : OuterKey}
    {slot : Fin slotCount}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {incomingParent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    {polynomial : PiCCS.SplitNc.Verifier.PublicInput shape}
    {priorState : TranscriptState}
    {result : Slot shape publicRingColumns publicFits verifierRows}
    (accepted : FixedActive.ResultTransition
      (contextOf setup key slot source incomingParent polynomial priorState)
      result) :
    Transition setup key slot source result.children := by
  let context :=
    contextOf setup key slot source incomingParent polynomial priorState
  rcases
      (FixedActive.resultTransition_iff_exists_paperDecomposition context
        result).mp accepted with
    ⟨data, witness, decomposed⟩
  exact transition_of_paper
    (incomingParent := incomingParent) (polynomial := polynomial)
    (priorState := priorState) ⟨data, witness, decomposed.paper⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics
