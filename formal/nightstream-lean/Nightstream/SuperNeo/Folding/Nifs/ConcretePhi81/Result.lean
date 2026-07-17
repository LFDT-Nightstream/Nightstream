import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Transition

/-!
Arity-independent result of one concrete Phi81 NIFS transition.

Protocol: SuperNeo NIFS.
Phase: accepted `Pi_RLC` parent and canonical `Pi_DEC` children.
Constraint family: semantic result projection only; this file emits no rows.

Owns: the one parent-plus-children result carrier; deterministic projection
from one shared phase execution; its independent semantic transition; child
projection; source/output structure preservation; and uniqueness of the
derived parent given accepted children.

Does not own: bootstrap or active arity selection, incoming-parent authority,
an executable checker, outer F-prime state, Rust/R1CS lowering, rows, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: `FoldResult.parent` and `FoldResult.children` are derived
from the same semantic certificate. Neither profile may supply a parent cache
or child family independently. This module is parameterized by `BatchArity`;
fixed-bootstrap and fixed-active modules only select a profile.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.result.parent` | expose the verifier-derived `Pi_RLC` parent | computed | `resultOf` |
| `nifs.result.children` | expose the canonical `Pi_DEC` child family | computed | `resultOf` |
| `nifs.result.semantic` | bind both surfaces to one independent source transition | semantic target | `ResultTransition` |
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
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity) :=
  ConcretePhi81.Certificate (domain := domain) (arity := arity)
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
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity)
    (certificate : Certificate context) :
    FoldResult shape publicRingColumns publicFits verifierRows := {
  parent := (ConcretePhi81.derive context certificate).piRlcOutput
  children := ConcretePhi81.outputChildren context certificate
}

@[simp] theorem resultOf_parent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity)
    (certificate : Certificate context) :
    (resultOf context certificate).parent =
      (ConcretePhi81.derive context certificate).piRlcOutput := rfl

@[simp] theorem resultOf_children
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity)
    (certificate : Certificate context) :
    (resultOf context certificate).children =
      ConcretePhi81.outputChildren context certificate := rfl

/-- Independent semantic transition for the complete result. -/
def ResultTransition
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity)
    (result : FoldResult shape publicRingColumns publicFits verifierRows) :
    Prop :=
  ∃ data : Data shape,
    ∃ certificate : Certificate context,
      result = resultOf context certificate ∧
        ConcretePhi81.Holds context data certificate

/-- Project the complete semantic result to the public child relation. -/
theorem ResultTransition.children_transition
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    ConcretePhi81.Transition context result.children := by
  rcases transition with ⟨data, certificate, resultEq, holds⟩
  subst result
  exact ⟨data, certificate, rfl, holds⟩

/-- The child-only relation is exactly the projection of a complete result. -/
theorem transition_iff_exists_resultTransition
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)} :
    ConcretePhi81.Transition context children ↔
      ∃ result : FoldResult shape publicRingColumns publicFits verifierRows,
        result.children = children ∧ ResultTransition context result := by
  constructor
  · rintro ⟨data, certificate, childrenEq, holds⟩
    refine ⟨resultOf context certificate, ?_, ?_⟩
    · simpa using childrenEq
    · exact ⟨data, certificate, rfl, holds⟩
  · rintro ⟨result, childrenEq, transition⟩
    have projected := transition.children_transition
    simpa [childrenEq] using projected

/-- Every running source shares the first fresh source's structure. Bootstrap
specializes this theorem to an empty index type. -/
theorem ResultTransition.runningStructure_eq_fresh
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result)
    (running : Fin (arity.mode.count productionGlobalParams)) :
    (context.input.running running).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  rcases transition with ⟨_data, _certificate, _resultEq, holds⟩
  exact
    (holds.input.sources.running running).constraintSystem.trans
      (holds.input.sources.fresh
        ⟨0, arity.freshPositive⟩).constraintSystem.symm

/-- Every output child preserves the first fresh source's structure. -/
theorem ResultTransition.childStructure_eq_fresh
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result)
    (child : Fin productionGlobalParams.k) :
    (result.children child).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  rcases transition with ⟨data, certificate, resultEq, holds⟩
  subst result
  calc
    ((resultOf context certificate).children child).constraintSystem =
        ((ConcretePhi81.derive context certificate).piRlcOutput).constraintSystem := by
      simpa [resultOf, ConcretePhi81.Execution.piDecAttempt] using
        holds.tail.piDec.sameStructure child
    _ = context.system := by
      rfl
    _ = (context.input.fresh
          ⟨0, arity.freshPositive⟩).constraintSystem :=
      context.system_eq_firstFresh

/-- Equal accepted child families determine the same derived parent cache. -/
theorem ResultTransition.parent_eq_of_children_eq
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      ConcretePhi81.Context shape domain State publicRingColumns publicFits
        verifierRows arity}
    {left right : FoldResult shape publicRingColumns publicFits verifierRows}
    (leftAccepted : ResultTransition context left)
    (rightAccepted : ResultTransition context right)
    (childrenEq : left.children = right.children) :
    left.parent = right.parent := by
  rcases leftAccepted with
    ⟨leftData, leftCertificate, leftEq, leftHolds⟩
  rcases rightAccepted with
    ⟨rightData, rightCertificate, rightEq, rightHolds⟩
  subst left
  subst right
  apply PiDEC.Accepted.parent_eq_of_children_eq
      (params := productionGlobalParams) (by decide)
      leftHolds.tail.piDec rightHolds.tail.piDec
  simpa using childrenEq

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result
