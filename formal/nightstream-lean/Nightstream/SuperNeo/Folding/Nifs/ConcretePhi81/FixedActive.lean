import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result

/-!
Fixed steady-state carrier and semantic result for the concrete Phi81 NIFS.

Protocol: SuperNeo NIFS.
Phase: one fresh CCS claim plus the complete running `CE(b)^k` accumulator.
Constraint family: typed carrier and semantic result only; this file emits no
rows.

Owns: the exact production arity `1 + 14 = 15` and the fixed-profile context,
certificate, result, and theorem facade.

Does not own: the generic result equations or semantic transition, which are
owned by `Result`; bootstrap folds; an executable checker; outer F-prime
state/hash binding; Rust/R1CS refinement; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: this facade does not redefine result authority.
`FoldResult`, `resultOf`, and `ResultTransition` delegate definitionally to
the generic owner; only active arity selection is profile-specific here.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.arity` | exactly one fresh source plus all `k = 14` running sources | typed/computed | `arity` |
| `nifs.fixed_active.result` | expose the generic complete result at active arity | delegated | `Result.FoldResult`, `Result.resultOf` |
| `nifs.fixed_active.semantic` | expose the generic semantic transition at active arity | delegated | `Result.ResultTransition` |
| `nifs.fixed_active.derived` | expose generic structure and parent-uniqueness facts at active arity | delegated | `Result.ResultTransition.*` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Fixed recursive arity from HyperNova Construction 2:
one fresh CCS claim and all fourteen running CE claims. -/
def arity : BatchArity productionGlobalParams :=
  BatchArity.active productionGlobalParams 1 (by decide) (by decide)

@[simp] theorem arity_freshCount : arity.freshCount = 1 := rfl

@[simp] theorem arity_mode : arity.mode = .active := rfl

@[simp] theorem arity_total : arity.total = 15 := rfl

/-- Concrete verifier context specialized to the fixed active arity. -/
abbrev Context
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  ConcretePhi81.Context shape State publicRingColumns publicFits
    verifierRows arity

/-- Raw verifier-visible certificate specialized to the fixed active arity. -/
abbrev Certificate
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :=
  ConcretePhi81.Certificate (arity := arity)
    publicRingColumns publicFits verifierRows context.piCcsInput

/-- Complete recursive fold result.

The children are the next formal accumulator. The parent is the checked,
deterministically derived combined statement retained as a recomposition
cache for the next transition. It is not a binding commitment to the child
vector. -/
abbrev FoldResult
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Result.FoldResult shape publicRingColumns publicFits verifierRows

/-- Compute both public result surfaces from one shared phase execution. -/
abbrev resultOf
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    FoldResult shape publicRingColumns publicFits verifierRows :=
  Result.resultOf context certificate

@[simp] theorem resultOf_parent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    (resultOf context certificate).parent =
      (ConcretePhi81.derive context certificate).piRlcOutput := rfl

@[simp] theorem resultOf_children
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    (resultOf context certificate).children =
      ConcretePhi81.outputChildren context certificate := rfl

/-- Independent semantic transition for the complete parent-and-children
result. Physical acceptance is intentionally absent from this definition. -/
abbrev ResultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (result : FoldResult shape publicRingColumns publicFits verifierRows) :
    Prop :=
  Result.ResultTransition context result

/-- The complete result transition projects to the existing child-only
transition without assigning independent authority to the cached parent. -/
theorem ResultTransition.children_transition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result) :
    ConcretePhi81.Transition context result.children := by
  exact Result.ResultTransition.children_transition
    (arity := arity) accepted

/-- The derived parent cache has a valid private opening. This does not make
the parent an injective encoding of its children. -/
theorem ResultTransition.parentOpening
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result) :
    ∃ assignment :
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits),
      CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
        result.parent assignment := by
  exact Result.ResultTransition.parentOpening (arity := arity) accepted

/-- Every returned child has an authoritative private opening. -/
theorem ResultTransition.childOpening
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result)
    (child : Fin productionGlobalParams.k) :
    ∃ assignment :
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits),
      CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
        (result.children child) assignment := by
  exact Result.ResultTransition.childOpening (arity := arity) accepted child

/-- The old child-only transition is exactly the projection of some complete
result transition. -/
theorem transition_iff_exists_resultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)} :
    ConcretePhi81.Transition context children ↔
      ∃ result : FoldResult shape publicRingColumns publicFits verifierRows,
        result.children = children ∧ ResultTransition context result := by
  exact Result.transition_iff_exists_resultTransition
    (arity := arity) (context := context) (children := children)

/-- Independent source authority already forces every selected running child
to use the same relation structure as the sole fresh source. This is derived
from the semantic NIFS transition and is not a separate outer `F'` check. -/
theorem ResultTransition.runningStructure_eq_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result)
    (running : Fin productionGlobalParams.k) :
    (context.input.running running).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  exact Result.ResultTransition.runningStructure_eq_fresh
    (arity := arity) accepted running

/-- Strict `Pi_DEC` output binding preserves the same sole fresh-source
structure in every returned accumulator child. No caller-supplied output
structure needs an additional outer check. -/
theorem ResultTransition.childStructure_eq_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result)
    (child : Fin productionGlobalParams.k) :
    (result.children child).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  exact Result.ResultTransition.childStructure_eq_fresh
    (arity := arity) accepted child

/-- Within one fixed context, two semantic results with the same checked child
accumulator necessarily carry the same derived parent cache. -/
theorem ResultTransition.parent_eq_of_children_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {left right : FoldResult shape publicRingColumns publicFits verifierRows}
    (leftAccepted : ResultTransition context left)
    (rightAccepted : ResultTransition context right)
    (childrenEq : left.children = right.children) :
    left.parent = right.parent := by
  exact Result.ResultTransition.parent_eq_of_children_eq
    (arity := arity) leftAccepted rightAccepted childrenEq

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
