import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding
import Nightstream.Protocol.FPrime.Paper.PriorLink
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening
import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren

/-!
Exact rich-carrier prior link for one recursive F-prime slot.

Assurance tier: model-level.

Owns: the two current-step checks needed to carry one complete semantic NIFS
result into the next invocation; exactness of those checks relative to the
previous semantic result; projection of the paper cross-step binding theorem
onto one rich slot; and transport of the NIFS result relation across an
accepted link.

Does not own: concrete serialization, Poseidon2 security, exclusion or bounds
for the paper binding failures, extraction of a previous NIFS result,
certificate checking, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: the previous NIFS result supplies canonical child
authority as an inherited theorem, not as a fresh verifier check. The next
invocation must bind the complete child vector and must check public PiDEC
recomposition for its carried parent cache. One exact digest target retains
all child-specific public payloads. A smaller target retains only shared
context and ordered child commitments, but requires current child openings
and a commitment-binding reduction. `slot_eq_or_commitmentDigest_failure_of_selectedNifs`
derives those openings and current recomposition from the independent
selected-NIFS relation rather than accepting them as extra outer evidence. A
digest can implement either equality only after its separate binding theorem.
The parent-only carrier is deliberately stricter:
`slot_eq_or_canonicalParentDigest_failure_of_openingSources` derives its
canonical-opening premise only from an opening-derived NIFS context whose
computed children are source-valid. The ordinary selected-NIFS interface does
not imply this premise.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.recursive.prior_link.previous` | previous result has one valid parent opening and its deterministic child split | inherited semantic authority | `FixedActive.ResultTransition.canonicalChildren` |
| `fprime.recursive.prior_link.children` | current children equal the complete previous child vector | checked | `Accepted.childrenEq` |
| `fprime.recursive.prior_link.parent_dec` | current cached parent strictly recomposes from the current children | checked | `Accepted.currentPiDec` |
| `fprime.recursive.prior_link.exact` | the complete current slot equals the previous result | derived | `Accepted.current_eq_previous` |
| `fprime.recursive.prior_link.family_digest` | compact shared-context-plus-child-payload handle binds the complete slot | derived or named encoding/hash failure | `slot_eq_or_familyDigest_failure` |
| `fprime.recursive.prior_link.commitment_digest` | smaller shared-context-plus-child-commitment handle binds the complete slot | derived or named encoding/hash/opening failure | `slot_eq_or_commitmentDigest_failure` |
| `fprime.recursive.prior_link.commitment_digest.selected_nifs` | selected NIFS semantics discharges the current opening and recomposition premises | derived composition | `slot_eq_or_commitmentDigest_failure_of_selectedNifs` |
| `fprime.recursive.prior_link.canonical_parent_digest` | point-plus-parent-commitment handle binds a canonically opened child family | derived or named encoding/hash/opening failure | `slot_eq_or_canonicalParentDigest_failure` |
| `fprime.recursive.prior_link.canonical_parent_digest.opening_sources` | opening-derived, source-valid NIFS input discharges current canonicality | derived composition | `slot_eq_or_canonicalParentDigest_failure_of_openingSources` |
| `fprime.recursive.prior_link.paper_projection` | exact paper running-product binding supplies complete rich child equality | derived or named security failure | `accepted_or_securityFailure` |
| `fprime.recursive.prior_link.paper_exact` | paper binding plus prior canonical authority yields exact rich-slot equality | derived or named security failure | `slot_eq_or_securityFailure` |
| `fprime.recursive.prior_link.transport` | the previous semantic NIFS result remains the current semantic slot | derived | `Accepted.resultTransition` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.PriorLink

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

universe uOuterKey uState uWitness uEncoding uDigest
universe uAppState uTranscriptState uHandle uPreviousState uCurrentState

/-- Current-step checks for carrying one complete prior NIFS result.

`childrenEq` is the actual cross-step binding obligation. `currentPiDec`
checks the cached parent rather than treating it as authority. -/
structure Accepted
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows)
    (previous current :
      Slot shape publicRingColumns publicFits verifierRows) : Prop where
  childrenEq : current.children = previous.children
  currentPiDec :
    PiDEC.Accepted (ConcretePhi81.decAlgebra key) {
      parent := current.parent
      children := current.children
    }

namespace Accepted

/-- Canonical authority for the previous result implies its strict public
PiDEC acceptance. -/
theorem previousPiDec
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (_accepted : Accepted key previous current)
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children) :
    PiDEC.Accepted (ConcretePhi81.decAlgebra key) {
      parent := previous.parent
      children := previous.children
    } := by
  rcases previousCanonical with ⟨_assignment, canonical⟩
  exact canonical.complete.1

/-- The two retained checks are sufficient: a canonically authorized previous
slot and an accepted current link are exactly the same rich carrier. -/
theorem current_eq_previous
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (accepted : Accepted key previous current)
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children) :
    current = previous := by
  have parentEq : current.parent = previous.parent :=
    (PiDEC.Accepted.parent_eq_of_children_eq (by decide)
      (accepted.previousPiDec previousCanonical) accepted.currentPiDec
      accepted.childrenEq.symm).symm
  have childrenEq : current.children = previous.children :=
    accepted.childrenEq
  cases current with
  | mk currentParent currentChildren =>
    cases previous with
    | mk previousParent previousChildren =>
      change currentParent = previousParent at parentEq
      change currentChildren = previousChildren at childrenEq
      cases parentEq
      cases childrenEq
      rfl

/-- Exact equality is complete for the two-check prior-link relation whenever
the previous slot has canonical semantic authority. -/
theorem of_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children)
    (equal : current = previous) :
    Accepted key previous current := by
  subst current
  rcases previousCanonical with ⟨_assignment, canonical⟩
  exact {
    childrenEq := rfl
    currentPiDec := canonical.complete.1
  }

/-- Relative to inherited canonical authority, the retained check set is
equivalent to equality of the complete rich slot. -/
theorem accepted_iff_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children) :
    Accepted key previous current ↔ current = previous := by
  exact ⟨fun accepted => accepted.current_eq_previous previousCanonical,
    of_eq previousCanonical⟩

/-- An accepted current slot inherits the exact canonical child authority of
the previous result. -/
theorem currentCanonical
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (accepted : Accepted key previous current)
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children) :
    PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
      current.parent current.children := by
  have equal := accepted.current_eq_previous previousCanonical
  subst current
  exact previousCanonical

/-- One previous semantic NIFS result constructs the reflexive exact link. -/
theorem of_resultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape State publicRingColumns publicFits verifierRows}
    {previous : Slot shape publicRingColumns publicFits verifierRows}
    (transition : FixedActive.ResultTransition context previous) :
    Accepted context.key previous previous := by
  exact of_eq transition.canonicalChildren rfl

/-- An accepted rich-carrier link transports the independent semantic NIFS
result; it does not create a second result relation. -/
theorem resultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape State publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (accepted : Accepted context.key previous current)
    (transition : FixedActive.ResultTransition context previous) :
    FixedActive.ResultTransition context current := by
  have equal := accepted.current_eq_previous transition.canonicalChildren
  subst current
  exact transition

/-- A previous semantic NIFS result closes the relation exactly: accepting a
current carrier is equivalent to carrying that same complete result. -/
theorem accepted_iff_eq_of_resultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape State publicRingColumns publicFits verifierRows}
    {previous current :
      Slot shape publicRingColumns publicFits verifierRows}
    (transition : FixedActive.ResultTransition context previous) :
    Accepted context.key previous current ↔ current = previous := by
  exact accepted_iff_eq transition.canonicalChildren

end Accepted

/-- A compact family handle is sufficient for the exact rich prior link. The
previous semantic result supplies strict PiDEC acceptance; the current step
checks its own cached parent. Equal handles then recover the complete slot or
name the exact family-encoding/hash failure. -/
theorem slot_eq_or_familyDigest_failure
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (ChildPayloadAuthority.FamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows)
    (previous current :
      Slot shape publicRingColumns publicFits verifierRows)
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children)
    (currentPiDec :
      PiDEC.Accepted (ConcretePhi81.decAlgebra key) {
        parent := current.parent
        children := current.children
      })
    (sameDigest :
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.familyDigest
          scheme current.parent
          current.children =
        Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.familyDigest
          scheme previous.parent
          previous.children) :
    current = previous \/
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases previousCanonical with ⟨_assignment, canonical⟩
  rcases Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_failure scheme
      (by decide) currentPiDec canonical.complete.1 sameDigest with
    exactView | failure
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
  · exact Or.inr failure

/-- The smaller ordered-commitment handle is sufficient when the current
children also have explicit valid CE openings. The previous semantic result
provides its canonical openings; any remaining ambiguity is returned as a
concrete compression failure or one indexed fresh-bound opening collision. -/
theorem slot_eq_or_commitmentDigest_failure
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (ChildCommitmentAuthority.CommitmentFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) productionGlobalParams.k)
      Encoding Digest)
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
    (currentValid : forall child,
      CE.Holds (ConcretePhi81.semantics key) productionGlobalParams
        (current.children child) (currentAssignments child))
    (sameStructure :
      current.parent.constraintSystem = previous.parent.constraintSystem)
    (sameDigest :
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.commitmentFamilyDigest
          scheme current.parent current.children =
        Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.commitmentFamilyDigest
          scheme previous.parent previous.children) :
    current = previous ∨
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.CommitmentFamilyFailure
        (ConcretePhi81.semantics key) productionGlobalParams scheme
        current.children := by
  rcases previousCanonical with ⟨previousAssignment, canonical⟩
  let previousAssignments : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits) :=
    fun child => (ConcretePhi81.decAlgebra key).splitAssignment
      previousAssignment child
  rcases Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_commitmentFailure
      scheme (by decide) currentAssignments previousAssignments currentPiDec
      canonical.complete.1 currentValid canonical.complete.2 sameStructure
      sameDigest with
    exactView | failure
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
  · exact Or.inr failure

/-- The ordered-commitment prior link does not need a second current-opening
witness at the outer F-prime boundary. The selected independent NIFS
transition already supplies one coherent opening function and strict incoming
`Pi_DEC` acceptance for the exact current slot. Structure equality remains a
verifier-setup premise, and digest/opening ambiguity remains an explicit
security failure. -/
theorem slot_eq_or_commitmentDigest_failure_of_selectedNifs
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {PreviousState : Type uPreviousState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Handle : Type uHandle}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (ChildCommitmentAuthority.CommitmentFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) productionGlobalParams.k)
      Encoding Handle)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
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
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.commitmentFamilyDigest
          scheme (input.running selected).parent
            (input.running selected).children =
        Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.commitmentFamilyDigest
          scheme previous.parent previous.children) :
    input.running selected = previous ∨
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.CommitmentFamilyFailure
        (ConcretePhi81.semantics (contextAt setup input selected).key)
        productionGlobalParams scheme (input.running selected).children := by
  have previousCanonical := previousTransition.canonicalChildren
  rw [sameKey] at previousCanonical
  rcases obligations.selectedInputAuthority with
    ⟨inputAssignments, currentPiDec, currentValid⟩
  exact slot_eq_or_commitmentDigest_failure scheme
    (contextAt setup input selected).key previous (input.running selected)
    inputAssignments previousCanonical currentPiDec currentValid
    sameStructure sameDigest

/-- Under the stronger current-step canonical-opening premise, the recursive
link needs to hash only the per-step point and combined parent commitment. This
is not derivable from public PiDEC acceptance; the premise is deliberately
visible in the signature. -/
theorem slot_eq_or_canonicalParentDigest_failure
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CanonicalParentAuthority.CanonicalParentPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows)
    (previous current :
      Slot shape publicRingColumns publicFits verifierRows)
    (currentAssignment : Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits))
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        previous.parent previous.children)
    (currentCanonical :
      PiDEC.CanonicalChildren.ForOpening (ConcretePhi81.decAlgebra key)
        current.parent currentAssignment current.children)
    (sameStructure :
      current.parent.constraintSystem = previous.parent.constraintSystem)
    (sameDigest :
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.canonicalParentDigest
          scheme current.parent =
        Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.canonicalParentDigest
          scheme previous.parent) :
    current = previous ∨
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.CanonicalParentFailure
        (ConcretePhi81.semantics key) productionGlobalParams scheme
        current.parent := by
  rcases previousCanonical with ⟨_previousAssignment, previousBound⟩
  rcases Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_canonicalParentFailure
      scheme currentCanonical previousBound sameStructure sameDigest with
    exactView | failure
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
  · exact Or.inr failure

/-- The parent-only prior link can be discharged without accepting a free
`currentCanonical` proposition when the current NIFS input is materialized
from one complete opening and its computed children satisfy the independent
source relation.

This is the exact model-level contract for a future proof-backed canonical
opening boundary. It does not claim that the current Rust certificate carries
or verifies that opening. -/
theorem slot_eq_or_canonicalParentDigest_failure_of_openingSources
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {CurrentState : Type uCurrentState}
    {PreviousState : Type uPreviousState}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CanonicalParentAuthority.CanonicalParentPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (currentContext :
      FixedActive.CanonicalOpening.Context shape CurrentState
        publicRingColumns publicFits verifierRows)
    (previous : Slot shape publicRingColumns publicFits verifierRows)
    (previousContext :
      FixedActive.Context shape PreviousState publicRingColumns publicFits
        verifierRows)
    (previousTransition :
      FixedActive.ResultTransition previousContext previous)
    (sameKey : previousContext.key = currentContext.key)
    (currentSources :
      FixedActive.CanonicalOpening.ChildSourcesValid currentContext)
    (sameStructure :
      currentContext.input.system = previous.parent.constraintSystem)
    (sameDigest :
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.canonicalParentDigest
          scheme
          (currentContext.input.opening.parent currentContext.key
            currentContext.input.system) =
        Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.canonicalParentDigest
          scheme previous.parent) :
    ({
      parent := currentContext.input.opening.parent currentContext.key
        currentContext.input.system
      children := currentContext.input.opening.children currentContext.key
        currentContext.input.system
    } : Slot shape publicRingColumns publicFits verifierRows) = previous \/
      Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.CanonicalParentFailure
        (ConcretePhi81.semantics currentContext.key) productionGlobalParams
        scheme
        (currentContext.input.opening.parent currentContext.key
          currentContext.input.system) := by
  have previousCanonical := previousTransition.canonicalChildren
  rw [sameKey] at previousCanonical
  exact slot_eq_or_canonicalParentDigest_failure scheme currentContext.key
    previous {
      parent := currentContext.input.opening.parent currentContext.key
        currentContext.input.system
      children := currentContext.input.opening.children currentContext.key
        currentContext.input.system
    } currentContext.input.opening.assignment previousCanonical
    (FixedActive.CanonicalOpening.canonicalChildren currentSources)
    sameStructure sameDigest

/-- The paper cross-step theorem binds every child in one selected rich slot,
or exposes the exact instance-encoding/hash failure. The cached parent remains
checked by current PiDEC rather than compressed as authority. -/
theorem accepted_or_securityFailure
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (previousInput currentInput :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (previousOutput :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount)
    (selected : Fin slotCount)
    (key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows)
    (previousOutputHash :
      Paper.OutputHolds machine previousInput.toPaper previousOutput.toPaper)
    (currentPriorPublic :
      currentInput.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage currentInput.toPaper)))
    (freshCarries :
      Paper.PriorLink.FreshCarriesPreviousOutput machine previousOutput.toPaper
        currentInput.toPaper)
    (currentPiDec :
      PiDEC.Accepted (ConcretePhi81.decAlgebra key) {
        parent := (currentInput.running selected).parent
        children := (currentInput.running selected).children
      }) :
    Accepted key (previousOutput.runningNext selected)
        (currentInput.running selected) ∨
      Paper.PriorLink.SecurityFailure machine := by
  rcases Paper.PriorLink.running_eq_or_securityFailure machine
      previousInput.toPaper previousOutput.toPaper currentInput.toPaper
      previousOutputHash currentPriorPublic freshCarries with
    runningEq | failure
  · apply Or.inl
    refine {
      childrenEq := ?_
      currentPiDec := currentPiDec
    }
    have selectedEq := congrFun runningEq selected
    change (currentInput.running selected).children =
      (previousOutput.runningNext selected).children at selectedEq
    exact selectedEq
  · exact Or.inr failure

/-- With inherited canonical authority for the previous semantic result, the
paper binding reduction and current PiDEC recover equality of the complete
rich slot, or name the exact compression failure. -/
theorem slot_eq_or_securityFailure
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (previousInput currentInput :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (previousOutput :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount)
    (selected : Fin slotCount)
    (key : ConcretePhi81.VerifierKey
      shape publicRingColumns publicFits verifierRows)
    (previousCanonical :
      PiDEC.CanonicalChildren.Holds (ConcretePhi81.decAlgebra key)
        (previousOutput.runningNext selected).parent
        (previousOutput.runningNext selected).children)
    (previousOutputHash :
      Paper.OutputHolds machine previousInput.toPaper previousOutput.toPaper)
    (currentPriorPublic :
      currentInput.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage currentInput.toPaper)))
    (freshCarries :
      Paper.PriorLink.FreshCarriesPreviousOutput machine previousOutput.toPaper
        currentInput.toPaper)
    (currentPiDec :
      PiDEC.Accepted (ConcretePhi81.decAlgebra key) {
        parent := (currentInput.running selected).parent
        children := (currentInput.running selected).children
      }) :
    currentInput.running selected = previousOutput.runningNext selected ∨
      Paper.PriorLink.SecurityFailure machine := by
  rcases accepted_or_securityFailure machine previousInput currentInput
      previousOutput selected key previousOutputHash currentPriorPublic
      freshCarries currentPiDec with accepted | failure
  · exact Or.inl (accepted.current_eq_previous previousCanonical)
  · exact Or.inr failure

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.PriorLink
