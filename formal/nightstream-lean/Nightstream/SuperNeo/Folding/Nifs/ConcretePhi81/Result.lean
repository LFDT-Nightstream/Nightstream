import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Transition
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan

/-!
Arity-independent result of one concrete Phi81 NIFS transition.

Protocol: SuperNeo NIFS.
Phase: accepted `Pi_RLC` parent and canonical `Pi_DEC` children.
Constraint family: semantic result projection only; this file emits no rows.

Assurance tier: model-level.

Owns: the one parent-plus-children result carrier; deterministic projection
from one shared physical execution; the certificate-independent semantic
transition over that carrier; the physical-refinement handoff; child
projection; coherent incoming running openings; source/output structure
preservation; and uniqueness of the derived parent given accepted children.

Does not own: bootstrap or active arity selection, incoming-parent authority,
an executable checker, outer F-prime state, Rust/R1CS lowering, rows, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: `ResultTransition` contains no verifier certificate.
`FoldResult.parent` and `FoldResult.children` must be the values computed by
`SemanticFold` from one independent source family, row point, and valid
challenge vector. `resultOf` remains the physical projection, and
`resultOf_refines` is the sole bridge between those layers.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.result.parent` | expose the verifier-derived `Pi_RLC` parent | computed | `resultOf` |
| `nifs.result.children` | expose the canonical `Pi_DEC` child family | computed | `resultOf` |
| `nifs.result.semantic` | bind both surfaces to one independent source transition | semantic target | `ResultTransition` |
| `nifs.result.obligation_plan` | open the result relation into the exact nine-leaf semantic plan | exact model theorem | `resultTransition_iff_exists_obligationPlan` |
| `nifs.result.refinement` | a certificate-indexed physical result instantiates the independent relation | derived | `resultOf_refines` |
| `nifs.result.parent_opening` | the cached parent has the canonical challenge-folded private opening | derived | `ResultTransition.parentOpening` |
| `nifs.result.canonical_children` | the complete child vector is the deterministic split of that same opening | derived | `ResultTransition.canonicalChildren` |
| `nifs.result.child_opening` | every canonical child has its radix-split private opening | derived | `ResultTransition.childOpening` |
| `nifs.result.input.running_authority` | expose the checked incoming-parent authority used by the same fold | derived | `ResultTransition.runningAuthority` |
| `nifs.result.input.running_openings` | one source witness opens the complete incoming running family | derived | `ResultTransition.inputRunningOpenings` |
| `nifs.result.input.running_structure` | every running source shares the sole fresh-source structure | derived | `ResultTransition.runningStructure_eq_fresh` |
| `nifs.result.child_structure` | every output child shares the sole fresh-source structure | derived | `ResultTransition.childStructure_eq_fresh` |
| `nifs.result.parent_unique` | equal accepted children determine the same derived parent | derived | `ResultTransition.parent_eq_of_children_eq` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Raw certificate at an arbitrary verifier-owned production arity. -/
abbrev Certificate
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity) :=
  ConcretePhi81.Certificate (arity := arity)
    publicRingColumns publicFits verifierRows context.piCcsInput

/-- Complete public result shared by bootstrap and active profiles. -/
structure FoldResult
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  parent :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)
  children : Fin productionGlobalParams.k ->
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

/-- Compute both public result surfaces from one shared phase execution. -/
def resultOf
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity)
    (certificate : Certificate context) :
    FoldResult shape publicRingColumns publicFits verifierRows := {
  parent := (ConcretePhi81.derive context certificate).piRlcOutput
  children := ConcretePhi81.outputChildren context certificate
}

@[simp] theorem resultOf_parent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity)
    (certificate : Certificate context) :
    (resultOf context certificate).parent =
      (ConcretePhi81.derive context certificate).piRlcOutput := rfl

@[simp] theorem resultOf_children
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity)
    (certificate : Certificate context) :
    (resultOf context certificate).children =
      ConcretePhi81.outputChildren context certificate := rfl

/-- Independent semantic transition for the complete result. -/
def ResultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity)
    (result : FoldResult shape publicRingColumns publicFits verifierRows) :
    Prop :=
  ∃ data : Data shape,
    SemanticFold.Holds context data result.parent result.children

/-- The result-level semantic leaf is exactly an existential realization of
the protocol/phase/family obligation plan. This is the formal parent-to-child
edge used by outer F-prime accounting; it is not a physical row mapping. -/
theorem resultTransition_iff_exists_obligationPlan
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity)
    (result : FoldResult shape publicRingColumns publicFits verifierRows) :
    ResultTransition context result ↔
      ∃ candidate :
          SemanticFold.ObligationPlan.Candidate shape State publicRingColumns
            publicFits verifierRows arity,
        candidate.context = context ∧
        candidate.parent = result.parent ∧
        candidate.children = result.children ∧
        CheckPlan.Accepts SemanticFold.ObligationPlan.semantics
          SemanticFold.ObligationPlan.checks candidate := by
  constructor
  · rintro ⟨data, witness, realized⟩
    let candidate :
        SemanticFold.ObligationPlan.Candidate shape State publicRingColumns
          publicFits verifierRows arity := {
      context := context
      data := data
      point := witness.point
      challenges := witness.challenges
      parent := result.parent
      children := result.children
    }
    refine ⟨candidate, rfl, rfl, rfl, ?_⟩
    apply
      (SemanticFold.ObligationPlan.accepts_iff_target candidate).mpr
    exact realized
  · rintro ⟨candidate, contextEq, parentEq, childrenEq, accepted⟩
    have realized :=
      (SemanticFold.ObligationPlan.accepts_iff_target candidate).mp accepted
    change SemanticFold.Realization candidate.context candidate.data
      candidate.parent candidate.children candidate.witness at realized
    subst context
    rw [parentEq, childrenEq] at realized
    exact ⟨candidate.data, candidate.witness, realized⟩

/-- A physical result with complete certificate-refinement evidence
instantiates the certificate-independent result relation. -/
theorem resultOf_refines
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {data : Data shape}
    {certificate : Certificate context}
    (refinement :
      ConcretePhi81.CertificateRefinement context data certificate) :
    ResultTransition context (resultOf context certificate) := by
  refine ⟨data, ?_⟩
  simpa [resultOf] using refinement.toSemanticFold

/-- Every independent semantic result binds its complete public child vector
to one valid combined parent opening. This is the exact semantic carrier that
an F-prime prior-link commitment or proof must preserve. -/
theorem ResultTransition.canonicalChildren
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    ∃ assignment :
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits),
      PiDEC.CanonicalChildren.ForOpening
        (ConcretePhi81.decAlgebra context.key) result.parent assignment
        result.children := by
  rcases transition with ⟨data, holds⟩
  exact holds.canonicalChildren

/-- The cached public parent in any semantic result has the canonical private
opening obtained by folding the independently authorized source assignments. -/
theorem ResultTransition.parentOpening
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    ∃ assignment :
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits),
      CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
        result.parent assignment := by
  rcases transition.canonicalChildren with ⟨assignment, canonical⟩
  exact ⟨assignment, canonical.parentValid⟩

/-- Every public child in a semantic result retains the corresponding
canonical radix-split opening. The physical refinement that establishes
canonicality still requires the explicit extraction/binding boundary. -/
theorem ResultTransition.childOpening
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result)
    (child : Fin productionGlobalParams.k) :
    ∃ assignment :
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits),
      CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
        (result.children child) assignment := by
  rcases transition.canonicalChildren with ⟨assignment, canonical⟩
  exact ⟨(ConcretePhi81.decAlgebra context.key).splitAssignment
      assignment child,
    canonical.complete.2 child⟩

/-- Project the complete semantic result to the public child relation. -/
theorem ResultTransition.children_transition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    ConcretePhi81.Transition context result.children := by
  rcases transition with ⟨data, holds⟩
  exact ⟨data, result.parent, holds⟩

/-- The child-only relation is exactly the projection of a complete result. -/
theorem transition_iff_exists_resultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)} :
    ConcretePhi81.Transition context children ↔
      ∃ result : FoldResult shape publicRingColumns publicFits verifierRows,
        result.children = children ∧ ResultTransition context result := by
  constructor
  · rintro ⟨data, parent, holds⟩
    let result : FoldResult shape publicRingColumns publicFits verifierRows := {
      parent := parent
      children := children
    }
    exact ⟨result, rfl, ⟨data, holds⟩⟩
  · rintro ⟨result, childrenEq, ⟨data, holds⟩⟩
    refine ⟨data, result.parent, ?_⟩
    simpa [childrenEq] using holds

/-- One semantic transition supplies a coherent opening function for the
complete incoming running family. All assignments come from the same
independent source witness; this theorem neither chooses fourteen unrelated
existential witnesses nor adds prover-authoritative inputs. -/
theorem ResultTransition.inputRunningOpenings
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    ∃ inputAssignments :
        Fin (arity.mode.count productionGlobalParams) ->
          Phi81Relation.Assignment
            (RelationShape shape publicRingColumns publicFits),
      ∀ running,
        CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
          (context.input.running running) (inputAssignments running) := by
  rcases transition with ⟨data, witness, realized⟩
  refine ⟨fun running =>
    data.runningAssignments
      (context.alignment.semanticRunningIndex running), ?_⟩
  intro running
  exact InputAuthority.runningSource_holds publicRingColumns publicFits
    (commit context.key) data context.alignment context.input
    production_norm_stages.1 realized.paper running
    (realized.input.sources.running running)

/-- The same semantic realization also retains the exact checked incoming
parent authority. This is public recomposition consistency; the coherent
private openings are exposed separately by `inputRunningOpenings`. -/
theorem ResultTransition.runningAuthority
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    RunningAuthority.Accepted context := by
  rcases transition with ⟨_data, _witness, realized⟩
  exact realized.running

/-- Every running source shares the first fresh source's structure. Bootstrap
specializes this theorem to an empty index type. -/
theorem ResultTransition.runningStructure_eq_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result)
    (running : Fin (arity.mode.count productionGlobalParams)) :
    (context.input.running running).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  rcases transition with ⟨_data, _witness, realized⟩
  exact
    (realized.input.sources.running running).constraintSystem.trans
      (realized.input.sources.fresh
        ⟨0, arity.freshPositive⟩).constraintSystem.symm

/-- Every output child preserves the first fresh source's structure. -/
theorem ResultTransition.childStructure_eq_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result)
    (child : Fin productionGlobalParams.k) :
    (result.children child).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  rcases transition with ⟨data, witness, realized⟩
  calc
    (result.children child).constraintSystem =
        (SemanticFold.childrenOf context data witness child).constraintSystem := by
      rw [realized.children_eq]
    _ = SemanticFold.systemOf context data := by
      rfl
    _ = (context.input.fresh
          ⟨0, arity.freshPositive⟩).constraintSystem :=
      (realized.input.sources.fresh
        ⟨0, arity.freshPositive⟩).constraintSystem.symm

/-- Equal accepted child families determine the same derived parent cache. -/
theorem ResultTransition.parent_eq_of_children_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape State publicRingColumns publicFits
        verifierRows arity}
    {left right : FoldResult shape publicRingColumns publicFits verifierRows}
    (leftAccepted : ResultTransition context left)
    (rightAccepted : ResultTransition context right)
    (childrenEq : left.children = right.children) :
    left.parent = right.parent := by
  rcases leftAccepted with ⟨_leftData, leftHolds⟩
  rcases rightAccepted with ⟨_rightData, rightHolds⟩
  exact PiDEC.Accepted.parent_eq_of_children_eq
    (params := productionGlobalParams) (by decide)
    leftHolds.piDecAccepted rightHolds.piDecAccepted childrenEq

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result
