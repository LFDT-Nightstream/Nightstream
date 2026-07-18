import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Baseline

/-!
Removal witness for selected-NIFS incoming accumulator authority.

Assurance tier: model-level.

Owns: the active-mode counterexample obtained by deleting the complete
incoming parent carrier while retaining all other context fields and
recomputing the outgoing parent and children.

Does not own: parent hashing, transcript binding, physical rows, costs,
Rust/R1CS refinement, security reduction, or row removal.

Emits constraints: no.

Authority boundary: active acceptance requires a complete checked parent;
a digest is never substituted for that authority.

| Phase | Stage path | Counterexample | Lean owner |
|---|---|---|---|
| incoming | `fprime.active.nifs.running.authority.necessity` | remove only the parent carrier and recompute outputs | `incomingAuthority_necessary` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

/-- Preserve every verifier-owned field except the active incoming parent. -/
def noParentContext :
    FixedActive.Context Sources.shape Unit Context.publicRingColumns
      Context.publicFits Context.verifierRows :=
  { Context.context with runningParent := none }

/-- Raw witness transported to the context with no incoming parent. -/
def noParentWitness : SemanticFold.Witness noParentContext where
  point := baselineWitness.point
  challenges := baselineWitness.challenges

/-- Recompute both result surfaces after deleting only incoming-parent
authority. -/
def noParentCandidate : BaselineCandidate := {
  context := noParentContext
  data := Sources.data
  point := noParentWitness.point
  challenges := noParentWitness.challenges
  parent := SemanticFold.parentOf noParentContext Sources.data noParentWitness
  children :=
    SemanticFold.childrenOf noParentContext Sources.data noParentWitness
}

/-- Active fixed arity cannot authorize a missing parent carrier. -/
theorem noParent_not_authorized :
    ¬RunningAuthority.Accepted noParentContext := by
  intro accepted
  cases accepted with
  | bootstrap mode _ =>
      change RunningMode.active = RunningMode.bootstrap at mode
      cases mode
  | active bound =>
      simpa [noParentContext] using bound.parentBound

/-- Deleting the parent changes only the incoming-authority leaf once the
outgoing parent and children are recomputed. -/
theorem noParent_semantics_iff
    (leaf : SemanticFold.ObligationPlan.Leaf)
    (retained : leaf ≠ .incomingAuthority) :
    baselineSemantics leaf noParentCandidate ↔
      baselineSemantics leaf baselineCandidate := by
  cases leaf with
  | freshCcs => rfl
  | allSourceNorm => rfl
  | carriedEvaluations => rfl
  | polynomialInput => rfl
  | sourceProduct => rfl
  | incomingAuthority => exact (retained rfl).elim
  | challengeStrongSet => rfl
  | parentExact =>
      constructor
      · intro _
        exact baselineAccepted .parentExact
          (SemanticFold.ObligationPlan.mem_checks .parentExact)
      · intro _
        rfl
  | childrenExact =>
      constructor
      · intro _
        exact baselineAccepted .childrenExact
          (SemanticFold.ObligationPlan.mem_checks .childrenExact)
      · intro _
        rfl

/-- The plan without incoming authority accepts the parentless mutation. -/
theorem noParentWeakened :
    CheckPlan.Accepts baselineSemantics
      (CheckPlan.without SemanticFold.ObligationPlan.checks
        .incomingAuthority)
      noParentCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (noParent_semantics_iff leaf retained).mpr
    (baselineAccepted leaf (SemanticFold.ObligationPlan.mem_checks leaf))

/-- The independent target rejects the parentless active context. -/
theorem noParentRejected : ¬baselineTarget noParentCandidate := by
  intro realized
  exact noParent_not_authorized realized.running

/-- Closed inclusion-necessity of incoming checked-parent authority. -/
theorem incomingAuthority_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .incomingAuthority :=
  ⟨noParentCandidate, noParentWeakened, noParentRejected⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
