import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan

/-!
Removal witnesses for the challenge-membership leaf and two computed-result
leaves of the concrete Phi81 NIFS semantic obligation plan.

Protocol: SuperNeo NIFS.
Phases: `Pi_RLC` challenge selection and parent materialization, followed by
`Pi_DEC` child materialization. This file emits no rows.

Assurance tier: model-level.

Owns: one valid raw-plan realization; deterministic challenge-only,
parent-only, and children-only mutations; preservation of every other semantic
leaf; rejection of each mutation by the independent target; and conditional
inclusion-necessity of those three obligations.

Does not own: existence of a closed honest fixture, transcript or extraction
security, executable checking, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: these theorems show that the public result must equal the
canonical computation. Because both leaves are classified `computed`, the
preferred implementation is to construct the values rather than spend rows
rechecking them. A necessity witness for the equality is not evidence that a
separate R1CS equality family must be retained.

| Stage path | Mutation | Preserved leaves | Result |
|---|---|---|---|
| `nifs.semantic.pi_rlc.challenge.strong_set.necessity` | replace every challenge by one explicit nonmember and recompute outputs | all leaves except `challengeStrongSet` | `Realization.challengeNecessary` |
| `nifs.semantic.pi_rlc.parent.exact.necessity` | change only the public parent stage | all leaves except `parentExact` | `Realization.parentNecessary` |
| `nifs.semantic.pi_dec.children.exact.necessity` | change only child zero's stage | all leaves except `childrenExact` | `Realization.childrenNecessary` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Pick a visibly different verifier-owned norm stage. -/
def differentStage : NormStage -> NormStage
  | .fresh => .combined
  | .combined => .fresh
  | .ambient => .fresh

theorem differentStage_ne (stage : NormStage) :
    differentStage stage ≠ stage := by
  cases stage <;> simp [differentStage]

section

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {arity : BatchArity productionGlobalParams}

/-- Change only a CE statement's stage. -/
def withDifferentStage
    (statement :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  { statement with stage := differentStage statement.stage }

theorem withDifferentStage_ne
    (statement :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    withDifferentStage statement ≠ statement := by
  intro equal
  have stageEqual := congrArg (fun candidate => candidate.stage) equal
  exact differentStage_ne statement.stage (by
    simpa [withDifferentStage] using stageEqual)

/-- One valid actual-type baseline for the exact nine-leaf plan. -/
structure Realization where
  candidate :
    Candidate shape State publicRingColumns publicFits verifierRows arity
  accepted : CheckPlan.Accepts semantics checks candidate

local notation "PlanRealization" =>
  Realization (shape := shape) (State := State)
    (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
    (publicFits := publicFits) (arity := arity)

namespace Realization

/-- Mutate only the public parent carrier. -/
def forgedParentCandidate (realization : PlanRealization) :
    Candidate shape State publicRingColumns publicFits verifierRows arity :=
  { realization.candidate with
    parent := withDifferentStage realization.candidate.parent }

theorem forgedParent_ne (realization : PlanRealization) :
    (realization.forgedParentCandidate).parent ≠
      realization.candidate.parent := by
  simpa [forgedParentCandidate] using
    (withDifferentStage_ne realization.candidate.parent)

/-- A parent-only mutation is invisible to every other semantic leaf. -/
theorem forgedParent_semantics_iff
    (realization : PlanRealization)
    (leaf : Leaf)
    (retained : leaf ≠ .parentExact) :
    semantics leaf realization.forgedParentCandidate ↔
      semantics leaf realization.candidate := by
  cases leaf <;>
    simp_all [forgedParentCandidate, semantics, Candidate.witness]

/-- Removing `parentExact` admits the parent-only mutation. -/
theorem parentWeakened (realization : PlanRealization) :
    CheckPlan.Accepts semantics (CheckPlan.without checks .parentExact)
      realization.forgedParentCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (realization.forgedParent_semantics_iff leaf retained).mpr
    (realization.accepted leaf (mem_checks leaf))

/-- The independent target rejects the parent-only mutation. -/
theorem forgedParentRejected (realization : PlanRealization) :
    ¬ target realization.forgedParentCandidate := by
  intro targetHolds
  have forgedAccepted :=
    (accepts_iff_target realization.forgedParentCandidate).mpr targetHolds
  have forgedEq :=
    forgedAccepted .parentExact (mem_checks .parentExact)
  have originalEq :=
    realization.accepted .parentExact (mem_checks .parentExact)
  apply realization.forgedParent_ne
  exact forgedEq.trans originalEq.symm

/-- Conditional inclusion-necessity of the computed parent equality. -/
theorem parentNecessary (realization : PlanRealization) :
    CheckPlan.NecessaryForSoundness
      (semantics (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      (target (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      checks .parentExact :=
  ⟨realization.forgedParentCandidate, realization.parentWeakened,
    realization.forgedParentRejected⟩

/-- Use one explicit out-of-set ring value at every challenge coordinate. -/
def forgedChallengeWitness (realization : PlanRealization) :
    SemanticFold.Witness realization.candidate.context where
  point := realization.candidate.point
  challenges := fun _ => outsideChallenge

/-- Recompute both public result surfaces after changing only the raw
challenge vector. -/
def forgedChallengeCandidate (realization : PlanRealization) :
    Candidate shape State publicRingColumns publicFits verifierRows arity := {
  context := realization.candidate.context
  data := realization.candidate.data
  point := realization.candidate.point
  challenges := fun _ => outsideChallenge
  parent := SemanticFold.parentOf realization.candidate.context
    realization.candidate.data realization.forgedChallengeWitness
  children := SemanticFold.childrenOf realization.candidate.context
    realization.candidate.data realization.forgedChallengeWitness
}

/-- A challenge-only mutation with canonically recomputed outputs preserves
every other semantic leaf. -/
theorem forgedChallenge_semantics_iff
    (realization : PlanRealization)
    (leaf : Leaf)
    (retained : leaf ≠ .challengeStrongSet) :
    semantics leaf realization.forgedChallengeCandidate ↔
      semantics leaf realization.candidate := by
  cases leaf with
  | freshCcs => rfl
  | allSourceNorm => rfl
  | carriedEvaluations => rfl
  | polynomialInput => rfl
  | sourceProduct => rfl
  | incomingAuthority => rfl
  | challengeStrongSet => exact (retained rfl).elim
  | parentExact =>
      constructor
      · intro _
        exact realization.accepted .parentExact (mem_checks .parentExact)
      · intro _
        rfl
  | childrenExact =>
      constructor
      · intro _
        exact realization.accepted .childrenExact (mem_checks .childrenExact)
      · intro _
        rfl

/-- Removing unary challenge membership admits the recomputed mutation. -/
theorem challengeWeakened (realization : PlanRealization) :
    CheckPlan.Accepts semantics
      (CheckPlan.without checks .challengeStrongSet)
      realization.forgedChallengeCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (realization.forgedChallenge_semantics_iff leaf retained).mpr
    (realization.accepted leaf (mem_checks leaf))

/-- The independent target rejects the explicit out-of-set challenge. -/
theorem forgedChallengeRejected (realization : PlanRealization) :
    ¬ target realization.forgedChallengeCandidate := by
  intro targetHolds
  let first : Fin arity.total := ⟨0, arity.totalPositive⟩
  have valid := targetHolds.challengesValid first
  have outsideValid :
      (rlcAlgebra realization.candidate.context.key).challengeValid
        outsideChallenge := by
    simpa [forgedChallengeCandidate, Candidate.witness,
      forgedChallengeWitness] using valid
  exact outsideChallenge_not_member outsideValid

/-- Conditional inclusion-necessity of unary strong-set membership. Sampler
replay may derive this leaf without a separate physical membership check. -/
theorem challengeNecessary (realization : PlanRealization) :
    CheckPlan.NecessaryForSoundness
      (semantics (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      (target (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      checks .challengeStrongSet :=
  ⟨realization.forgedChallengeCandidate, realization.challengeWeakened,
    realization.forgedChallengeRejected⟩

/-- Canonical child chosen for the one-coordinate mutation. -/
def firstChild : Fin productionGlobalParams.k :=
  ⟨0, by decide⟩

/-- Mutate exactly child zero and preserve every other child. -/
def forgedChildren
    (realization : PlanRealization)
    (child : Fin productionGlobalParams.k) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  if child = firstChild then
    withDifferentStage (realization.candidate.children child)
  else
    realization.candidate.children child

theorem forgedChildren_ne (realization : PlanRealization) :
    realization.forgedChildren ≠ realization.candidate.children := by
  intro equal
  have atFirst := congrFun equal firstChild
  have changed :
      withDifferentStage (realization.candidate.children firstChild) =
        realization.candidate.children firstChild := by
    simpa [forgedChildren] using atFirst
  exact withDifferentStage_ne
    (realization.candidate.children firstChild) changed

/-- Mutate only the public child family. -/
def forgedChildrenCandidate (realization : PlanRealization) :
    Candidate shape State publicRingColumns publicFits verifierRows arity :=
  { realization.candidate with
    children := realization.forgedChildren }

/-- A children-only mutation is invisible to every other semantic leaf. -/
theorem forgedChildren_semantics_iff
    (realization : PlanRealization)
    (leaf : Leaf)
    (retained : leaf ≠ .childrenExact) :
    semantics leaf realization.forgedChildrenCandidate ↔
      semantics leaf realization.candidate := by
  cases leaf <;>
    simp_all [forgedChildrenCandidate, semantics, Candidate.witness]

/-- Removing `childrenExact` admits the one-child mutation. -/
theorem childrenWeakened (realization : PlanRealization) :
    CheckPlan.Accepts semantics (CheckPlan.without checks .childrenExact)
      realization.forgedChildrenCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (realization.forgedChildren_semantics_iff leaf retained).mpr
    (realization.accepted leaf (mem_checks leaf))

/-- The independent target rejects the one-child mutation. -/
theorem forgedChildrenRejected (realization : PlanRealization) :
    ¬ target realization.forgedChildrenCandidate := by
  intro targetHolds
  have forgedAccepted :=
    (accepts_iff_target realization.forgedChildrenCandidate).mpr targetHolds
  have forgedEq :=
    forgedAccepted .childrenExact (mem_checks .childrenExact)
  have originalEq :=
    realization.accepted .childrenExact (mem_checks .childrenExact)
  apply realization.forgedChildren_ne
  exact forgedEq.trans originalEq.symm

/-- Conditional inclusion-necessity of the computed child-family equality. -/
theorem childrenNecessary (realization : PlanRealization) :
    CheckPlan.NecessaryForSoundness
      (semantics (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      (target (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      checks .childrenExact :=
  ⟨realization.forgedChildrenCandidate, realization.childrenWeakened,
    realization.forgedChildrenRejected⟩

end Realization

end
end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity
