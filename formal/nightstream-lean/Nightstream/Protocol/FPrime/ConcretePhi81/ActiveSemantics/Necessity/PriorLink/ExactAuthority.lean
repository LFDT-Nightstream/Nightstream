import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.PriorLink
import Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution

/-!
Per-check necessity witnesses for the exact rich-carrier prior link.

Assurance tier: model-level.

Owns: one production-profile countermodel for removing complete child-vector
binding and one for removing current-parent PiDEC recomposition, while the
other retained check and inherited previous authority remain intact.

Does not own: hash/preimage binding, Poseidon2 security, Rust/R1CS refinement,
physical rows, costs, or row removal.

Emits constraints: no.

Authority boundary: these witnesses establish inclusion-minimality only for
the current rich slot carrier, where the parent is carried as a cache. A
future carrier that computes the parent from its children may eliminate the
parent check rather than materialize it.

| Stage path | Removed obligation | Retained evidence | Invalid result | Lean owner |
|---|---|---|---|---|
| `fprime.recursive.prior_link.necessity.children` | complete child-vector equality | previous canonical authority and current strict PiDEC | distinct accepted child substitution | `childrenBinding_necessary` |
| `fprime.recursive.prior_link.necessity.parent_dec` | current cached-parent recomposition | previous canonical authority and exact child equality | same children under a fresh-stage parent | `parentRecomposition_necessary` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ExactAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

namespace Counterexample

def shape : SemanticShape :=
  FPrimeCarrier270.PaddedIdentityEvaluation.semanticShape

def publicRingColumns : Nat := FPrimeCarrier270.publicRingColumns

def publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth := by
  decide

abbrev Slot := Outer.Slot shape publicRingColumns publicFits 0

def key : VerifierKey shape publicRingColumns publicFits 0 :=
  Fixture.key

/-- Canonically authorized result of the previous semantic NIFS step. -/
def previous : Slot where
  parent := Fixture.parent
  children := Fixture.leftChildren

/-- Distinct child substitution that still passes strict public PiDEC. -/
def substituted : Slot where
  parent := Fixture.parent
  children := Fixture.rightChildren

theorem previousCanonical :
    PiDEC.CanonicalChildren.Holds (decAlgebra key)
      previous.parent previous.children := by
  refine ⟨Fixture.parentOpening, ?_⟩
  simpa [key, previous] using
    Fixture.leftCanonical

theorem substitutedPiDec :
    PiDEC.Accepted (decAlgebra key) {
      parent := substituted.parent
      children := substituted.children
    } := by
  simpa [key, substituted] using
    Fixture.rightAccepted

theorem substituted_ne : substituted ≠ previous := by
  intro equal
  apply Fixture.children_ne
  exact (congrArg (fun slot : Slot => slot.children) equal).symm

/-- Parent mutation used to isolate necessity of current PiDEC. The complete
child vector is unchanged. -/
def freshParent := {
  Fixture.parent with
  stage := NormStage.fresh
}

def uncheckedParent : Slot where
  parent := freshParent
  children := Fixture.leftChildren

theorem uncheckedParent_children :
    uncheckedParent.children = previous.children := rfl

theorem uncheckedParent_ne : uncheckedParent ≠ previous := by
  intro equal
  have stageEq := congrArg (fun slot : Slot => slot.parent.stage) equal
  change NormStage.fresh = NormStage.combined at stageEq
  exact (by decide : NormStage.fresh ≠ NormStage.combined) stageEq

theorem uncheckedParent_notPiDec :
    ¬ PiDEC.Accepted (decAlgebra key) {
      parent := uncheckedParent.parent
      children := uncheckedParent.children
    } := by
  intro accepted
  have stageEq := accepted.parentCombined
  change NormStage.fresh = NormStage.combined at stageEq
  exact (by decide : NormStage.fresh ≠ NormStage.combined) stageEq

end Counterexample

/-- Relation left after deleting only complete child-vector binding. -/
def WithoutChildrenBinding
    (current : Counterexample.Slot) : Prop :=
  PiDEC.Accepted (decAlgebra Counterexample.key) {
    parent := current.parent
    children := current.children
  }

/-- Relation left after deleting only current-parent recomposition. -/
def WithoutParentRecomposition
    (current : Counterexample.Slot) : Prop :=
  current.children = Counterexample.previous.children

/-- Removing child-vector equality admits a distinct production-profile slot
while inherited previous authority and current strict PiDEC both hold. -/
theorem childrenBinding_necessary :
    PiDEC.CanonicalChildren.Holds
        (decAlgebra Counterexample.key)
        Counterexample.previous.parent Counterexample.previous.children /\
      WithoutChildrenBinding Counterexample.substituted /\
      Counterexample.substituted ≠ Counterexample.previous /\
      ¬ ActiveSemantics.PriorLink.Accepted Counterexample.key
        Counterexample.previous Counterexample.substituted := by
  refine ⟨Counterexample.previousCanonical, Counterexample.substitutedPiDec,
    Counterexample.substituted_ne, ?_⟩
  intro accepted
  exact Counterexample.substituted_ne
    (accepted.current_eq_previous Counterexample.previousCanonical)

/-- Removing current-parent recomposition admits a distinct carried parent
while inherited previous authority and exact child equality both hold. -/
theorem parentRecomposition_necessary :
    PiDEC.CanonicalChildren.Holds
        (decAlgebra Counterexample.key)
        Counterexample.previous.parent Counterexample.previous.children /\
      WithoutParentRecomposition Counterexample.uncheckedParent /\
      Counterexample.uncheckedParent ≠ Counterexample.previous /\
      ¬ PiDEC.Accepted (decAlgebra Counterexample.key) {
        parent := Counterexample.uncheckedParent.parent
        children := Counterexample.uncheckedParent.children
      } /\
      ¬ ActiveSemantics.PriorLink.Accepted Counterexample.key
        Counterexample.previous Counterexample.uncheckedParent := by
  refine ⟨Counterexample.previousCanonical,
    Counterexample.uncheckedParent_children,
    Counterexample.uncheckedParent_ne,
    Counterexample.uncheckedParent_notPiDec, ?_⟩
  intro accepted
  exact Counterexample.uncheckedParent_notPiDec accepted.currentPiDec

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.PriorLink.ExactAuthority
