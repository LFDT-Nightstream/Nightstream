import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan

/-!
Removal witnesses for the two computed result leaves of the concrete Phi81
NIFS semantic obligation plan.

Protocol: SuperNeo NIFS.
Phases: `Pi_RLC` parent materialization and `Pi_DEC` child materialization.
Constraint families: semantic output equalities only; this file emits no rows.

Assurance tier: model-level.

Owns: one valid raw-plan realization; deterministic parent-only and
children-only mutations; preservation of every other semantic leaf; rejection
of each mutation by the independent target; and conditional
inclusion-necessity of both computed output equalities.

Does not own: existence of a closed honest fixture, challenge necessity,
transcript or extraction security, executable checking, Rust/R1CS refinement,
costs, or row removal.

Emits constraints: no.

Authority boundary: these theorems show that the public result must equal the
canonical computation. Because both leaves are classified `computed`, the
preferred implementation is to construct the values rather than spend rows
rechecking them. A necessity witness for the equality is not evidence that a
separate R1CS equality family must be retained.

| Stage path | Mutation | Preserved leaves | Result |
|---|---|---|---|
| `nifs.semantic.pi_rlc.parent.exact.necessity` | change only the public parent stage | all leaves except `parentExact` | `Realization.parentNecessary` |
| `nifs.semantic.pi_dec.children.exact.necessity` | change only child zero's stage | all leaves except `childrenExact` | `Realization.childrenNecessary` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

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

namespace Realization

/-- Mutate only the public parent carrier. -/
def forgedParentCandidate (realization : Realization) :
    Candidate shape State publicRingColumns publicFits verifierRows arity :=
  { realization.candidate with
    parent := withDifferentStage realization.candidate.parent }

theorem forgedParent_ne (realization : Realization) :
    (realization.forgedParentCandidate).parent ≠
      realization.candidate.parent := by
  simpa [forgedParentCandidate] using
    (withDifferentStage_ne realization.candidate.parent)

/-- A parent-only mutation is invisible to every other semantic leaf. -/
theorem forgedParent_semantics_iff
    (realization : Realization)
    (leaf : Leaf)
    (retained : leaf ≠ .parentExact) :
    semantics leaf realization.forgedParentCandidate ↔
      semantics leaf realization.candidate := by
  cases leaf <;>
    simp_all [forgedParentCandidate, semantics, Candidate.witness]

/-- Removing `parentExact` admits the parent-only mutation. -/
theorem parentWeakened (realization : Realization) :
    CheckPlan.Accepts semantics (CheckPlan.without checks .parentExact)
      realization.forgedParentCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (realization.forgedParent_semantics_iff leaf retained).mpr
    (realization.accepted leaf (mem_checks leaf))

/-- The independent target rejects the parent-only mutation. -/
theorem forgedParentRejected (realization : Realization) :
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
theorem parentNecessary (realization : Realization) :
    CheckPlan.NecessaryForSoundness semantics target checks .parentExact :=
  ⟨realization.forgedParentCandidate, realization.parentWeakened,
    realization.forgedParentRejected⟩

/-- Canonical child chosen for the one-coordinate mutation. -/
def firstChild : Fin productionGlobalParams.k :=
  ⟨0, by decide⟩

/-- Mutate exactly child zero and preserve every other child. -/
def forgedChildren
    (realization : Realization)
    (child : Fin productionGlobalParams.k) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  if child = firstChild then
    withDifferentStage (realization.candidate.children child)
  else
    realization.candidate.children child

theorem forgedChildren_ne (realization : Realization) :
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
def forgedChildrenCandidate (realization : Realization) :
    Candidate shape State publicRingColumns publicFits verifierRows arity :=
  { realization.candidate with
    children := realization.forgedChildren }

/-- A children-only mutation is invisible to every other semantic leaf. -/
theorem forgedChildren_semantics_iff
    (realization : Realization)
    (leaf : Leaf)
    (retained : leaf ≠ .childrenExact) :
    semantics leaf realization.forgedChildrenCandidate ↔
      semantics leaf realization.candidate := by
  cases leaf <;>
    simp_all [forgedChildrenCandidate, semantics, Candidate.witness]

/-- Removing `childrenExact` admits the one-child mutation. -/
theorem childrenWeakened (realization : Realization) :
    CheckPlan.Accepts semantics (CheckPlan.without checks .childrenExact)
      realization.forgedChildrenCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (realization.forgedChildren_semantics_iff leaf retained).mpr
    (realization.accepted leaf (mem_checks leaf))

/-- The independent target rejects the one-child mutation. -/
theorem forgedChildrenRejected (realization : Realization) :
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
theorem childrenNecessary (realization : Realization) :
    CheckPlan.NecessaryForSoundness semantics target checks .childrenExact :=
  ⟨realization.forgedChildrenCandidate, realization.childrenWeakened,
    realization.forgedChildrenRejected⟩

end Realization

end
end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity
