import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.PriorLink

/-!
Model-level bridge from the exact ordered-commitment field message to the
recursive F-prime prior link.

Owns: composition of message injectivity, selected-NIFS input authority,
strict PiDEC recomposition, and child-opening binding into one explicit prior
slot theorem.

Does not own: a concrete Poseidon2 implementation or collision bound, Rust
serialization, R1CS decoding, artifact columns, costs, or row removal.

Emits constraints: no.

Authority boundary: equal hash outputs are useful only because both sides are
recomputed over the exact field messages. Outside a field-hash collision and
one indexed Ajtai opening collision, the complete current rich slot equals the
previous semantic NIFS result.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.recursive.prior_link.ordered_commitments.message` | equal digests bind equal typed carrier messages | security boundary | `OrderedCommitmentMessage.digest_eq_or_fieldHashCollision` |
| `fprime.recursive.prior_link.ordered_commitments.openings` | equal ordered commitments bind equal opened children | security boundary | `slot_eq_or_failure` |
| `fprime.recursive.prior_link.ordered_commitments.selected_nifs` | selected NIFS supplies current openings and recomposition | derived composition | `slot_eq_or_failure_of_selectedNifs` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink

set_option maxRecDepth 65536

open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uHashDigest uOuterKey uOuterDigest uAppState uWitness
  uTranscriptState uPreviousState

/-- Exact ordered-message prior link before outer selected-NIFS composition. -/
theorem slot_eq_or_failure
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {HashDigest : Type uHashDigest}
    (hashFields : List F -> HashDigest)
    (key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows)
    (previous current :
      Slot shape publicRingColumns publicFits verifierRows)
    (currentAssignments : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children)
    (currentPiDec :
      PiDEC.Accepted (ConcretePhi81.decAlgebra key) {
        parent := current.parent
        children := current.children
      })
    (currentValid : ∀ child,
      CE.Holds (ConcretePhi81.semantics key) productionGlobalParams
        (current.children child) (currentAssignments child))
    (sameStructure :
      current.parent.constraintSystem = previous.parent.constraintSystem)
    (sameDigest :
      payloadDigest hashFields
          (commitmentFamilyPayload current.parent current.children) =
        payloadDigest hashFields
          (commitmentFamilyPayload previous.parent previous.children)) :
    current = previous ∨
      FieldHashCollision hashFields ∨
      ∃ child, Nonempty
        (Opening.BindingCollision (ConcretePhi81.semantics key)
          productionGlobalParams.b (current.children child).commitment) := by
  rcases digest_eq_or_fieldHashCollision hashFields
      (commitmentFamilyPayload current.parent current.children)
      (commitmentFamilyPayload previous.parent previous.children)
      sameDigest with payloadEq | hashCollision
  · rcases previousCanonical with ⟨previousAssignment, canonical⟩
    let previousAssignments : Fin productionGlobalParams.k ->
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits) :=
      fun child => (ConcretePhi81.decAlgebra key).splitAssignment
        previousAssignment child
    rcases parent_children_eq_or_freshBindingCollision
        (by decide) currentAssignments previousAssignments currentPiDec
        canonical.complete.1 currentValid canonical.complete.2 sameStructure
        payloadEq with exactView | openingCollision
    · rcases exactView with ⟨parentEq, childrenEq⟩
      cases current with
      | mk currentParent currentChildren =>
        cases previous with
        | mk previousParent previousChildren =>
          change currentParent = previousParent at parentEq
          change currentChildren = previousChildren at childrenEq
          cases parentEq
          cases childrenEq
          exact Or.inl rfl
    · exact Or.inr (Or.inr openingCollision)
  · exact Or.inr (Or.inl hashCollision)

/-- The independent selected-NIFS relation supplies every non-cryptographic
premise needed by the exact ordered-message prior link. -/
theorem slot_eq_or_failure_of_selectedNifs
    {OuterKey : Type uOuterKey}
    {OuterDigest : Type uOuterDigest}
    {HashDigest : Type uHashDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {PreviousState : Type uPreviousState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (hashFields : List F -> HashDigest)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey OuterDigest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (previous : Slot shape publicRingColumns publicFits verifierRows)
    (previousContext :
      FixedActive.Context shape PreviousState publicRingColumns publicFits
        verifierRows)
    (previousTransition :
      FixedActive.ResultTransition previousContext previous)
    (sameKey :
      previousContext.key = (contextAt setup input selected).key)
    (obligations :
      Obligations setup machine functionIndex input selected selectedNext)
    (sameStructure :
      (input.running selected).parent.constraintSystem =
        previous.parent.constraintSystem)
    (sameDigest :
      payloadDigest hashFields
          (commitmentFamilyPayload (input.running selected).parent
            (input.running selected).children) =
        payloadDigest hashFields
          (commitmentFamilyPayload previous.parent previous.children)) :
    input.running selected = previous ∨
      FieldHashCollision hashFields ∨
      ∃ child, Nonempty
        (Opening.BindingCollision
          (ConcretePhi81.semantics (contextAt setup input selected).key)
          productionGlobalParams.b
          ((input.running selected).children child).commitment) := by
  have previousCanonical := previousTransition.canonicalChildren
  rw [sameKey] at previousCanonical
  rcases obligations.selectedInputAuthority with
    ⟨inputAssignments, currentPiDec, currentValid⟩
  exact slot_eq_or_failure hashFields
    (contextAt setup input selected).key previous (input.running selected)
    inputAssignments previousCanonical currentPiDec currentValid
    sameStructure sameDigest

end Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink
