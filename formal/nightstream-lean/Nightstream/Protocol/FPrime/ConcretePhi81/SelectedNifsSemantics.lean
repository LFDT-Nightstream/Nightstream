import Nightstream.Protocol.FPrime.ConcretePhi81.Outer
import Nightstream.Protocol.FPrime.Paper.Construction2

/-!
Public selected-NIFS edge for the ConcretePhi81 Construction-2 instance.

Assurance tier: model-level.

Owns: the key/slot-owned semantic setup, the exact selected public source,
existential internal parent/message carriers, and the independent
`FixedActive.ResultTransition` projected to public output children.

Does not own: the outer F-prime input, a physical certificate, transcript
replay, bad-event bounds, Rust, R1CS, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: cached parents, the Split-NC polynomial input, and the
prior transcript state are internal NIFS witnesses, not additional HyperNova
public inputs. Existence alone gives them no authority: acceptance still
requires `ResultTransition`, whose independent semantics binds the source
family, checks strict incoming-parent recomposition and challenge validity,
and computes the exact outgoing parent and children.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.paper.nifs.concrete.setup` | key and slot select static verifier setup and relation structure | verifier setup | `Setup` |
| `fprime.paper.nifs.concrete.source` | expose exactly one fresh claim and the selected `k` public children | direct dataflow | `Source` |
| `fprime.paper.nifs.concrete.internal` | parents and verifier messages remain internal to NIFS | existential semantic witness | `Transition` |
| `fprime.paper.nifs.concrete.result` | independent ConcretePhi81 semantics yields exactly the public target children | semantic target | `Transition` |
| `fprime.paper.nifs.concrete.family` | install the public edge as the abstract Construction-2 family | computed | `family` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics

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

/-- Static semantic setup selected only by the outer verifier key and slot. -/
structure Setup
    (OuterKey : Type uOuterKey)
    (TranscriptState : Type uTranscriptState)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  template :
    OuterKey -> Fin slotCount ->
      Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template
        shape TranscriptState publicRingColumns publicFits verifierRows
  expectedStructure :
    OuterKey -> Fin slotCount ->
      RelationStructure shape publicRingColumns publicFits

/-- Exact public source passed to one selected ConcretePhi81 NIFS edge. -/
abbrev Source
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Paper.Construction2.SelectedInput
    (RelationStructure shape publicRingColumns publicFits)
    (RelationPublicInput shape publicRingColumns publicFits)
    (RelationPoint shape publicRingColumns publicFits)
    Phi81Relation.Evaluation (CommitmentValue verifierRows)
    productionGlobalParams

/-- Exact public child vector returned by one selected NIFS edge. -/
abbrev Target
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Paper.RunningSlot
    (RelationStructure shape publicRingColumns publicFits)
    (RelationPublicInput shape publicRingColumns publicFits)
    (RelationPoint shape publicRingColumns publicFits)
    Phi81Relation.Evaluation (CommitmentValue verifierRows)
    productionGlobalParams

/-- Build the complete internal context from the public source plus semantic
witnesses that never become additional outer authority. -/
def contextOf
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
    (priorState : TranscriptState) :
    FixedActive.Context shape TranscriptState publicRingColumns publicFits
      verifierRows :=
  (setup.template key slot).build {
    fresh := source.fresh
    running := {
      parent := incomingParent
      children := source.running
    }
    piCcsInput := polynomial
    priorState := priorState
  }

/-- Public ConcretePhi81 NIFS transition. Internal parents and verifier
messages are existential; the public target is exactly the computed child
family of one independent semantic result. -/
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
        exists outgoingParent :
            Phi81Relation.CEStatement
              (RelationShape shape publicRingColumns publicFits)
              (CommitmentValue verifierRows),
          FixedActive.ResultTransition
            (contextOf setup key slot source incomingParent polynomial
              priorState) {
                parent := outgoingParent
                children := target
              }

/-- Canonical installation of the independent ConcretePhi81 public edge into
the abstract Construction-2 NIFS interface. -/
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

/-- Any complete semantic result supplies the internal witnesses required by
the public child-only edge. -/
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
  exact ⟨incomingParent, polynomial, priorState, result.parent, accepted⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics
