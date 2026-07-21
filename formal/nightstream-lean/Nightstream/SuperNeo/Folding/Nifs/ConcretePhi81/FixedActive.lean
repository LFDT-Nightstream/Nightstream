import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result

/-!
Fixed steady-state carrier and semantic result for the concrete Phi81 NIFS.

Protocol: SuperNeo NIFS.
Phase: one fresh CCS claim plus the complete running `CE(b)^k` accumulator.
Constraint family: typed carrier and semantic result only; this file emits no
rows.

Owns: the fixed-profile context, certificate, result, and theorem facade.

Does not own: the exact paper profile and arity, which are owned by
`FixedActive.PaperProfile`; the generic result equations or semantic
transition, which are owned by `Result`; bootstrap folds; an executable
checker; outer F-prime state/hash binding; Rust/R1CS refinement; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: this facade does not redefine result authority.
`FoldResult`, `resultOf`, and `ResultTransition` delegate definitionally to
the generic result owner. The independent paper profile owns active arity;
this facade owns only the exact paper-versus-sidecar-and-strengthening
decomposition.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.arity` | exactly one fresh source plus all `k = 14` running sources | delegated | `PaperProfile`, `arity` |
| `nifs.fixed_active.result` | expose the generic complete result at active arity | delegated | `Result.FoldResult`, `Result.resultOf` |
| `nifs.fixed_active.semantic` | expose the generic semantic transition at active arity | delegated | `Result.ResultTransition` |
| `nifs.fixed_active.paper_boundary` | separate public paper acceptance, lifecycle sidecars, and the extra canonical-child strengthening | derived | `resultTransition_iff_exists_paperDecomposition` |
| `nifs.fixed_active.canonical_children` | expose the valid parent opening and its complete deterministic child split | delegated | `ResultTransition.canonicalChildren` |
| `nifs.fixed_active.input.running_openings` | expose one coherent opening function for all fourteen incoming children | delegated | `ResultTransition.inputRunningOpenings` |
| `nifs.fixed_active.input.running_parent` | expose strict recomposition for the exact carried parent and fourteen children | delegated/derived | `ResultTransition.inputRunningPiDec` |
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

/-- Project a rich verifier context to exactly the setup retained by the
paper relation. No transcript, polynomial, parent, or lifecycle field is
copied. -/
def paperProfileOf
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :
    PaperProfile.Profile shape publicRingColumns publicFits verifierRows := {
  key := context.key
  alignment := context.alignment
}

/-- Forget the implementation-indexed witness type while preserving the raw
paper point and challenge vector exactly. -/
def paperWitnessOf
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    (witness : SemanticFold.Witness context) : PaperProfile.Witness shape := {
  point := witness.point
  challenges := witness.challenges
}

/-- Re-index a raw paper witness for the richer semantic carrier without
adding authority. -/
def semanticWitnessOf
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (witness : PaperProfile.Witness shape) :
    SemanticFold.Witness context := {
  point := witness.point
  challenges := witness.challenges
}

@[simp] theorem paper_parentOf_eq_semantic
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : SemanticFold.Witness context) :
    PaperProfile.parentOf (paperProfileOf context) context.input data
        (paperWitnessOf witness) =
      SemanticFold.parentOf context data witness := rfl

@[simp] theorem paper_childrenOf_eq_semantic
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : SemanticFold.Witness context) :
    PaperProfile.childrenOf (paperProfileOf context) context.input data
        (paperWitnessOf witness) =
      SemanticFold.childrenOf context data witness := rfl

@[simp] theorem semantic_parentOf_eq_paper
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : PaperProfile.Witness shape) :
    SemanticFold.parentOf context data (semanticWitnessOf context witness) =
      PaperProfile.parentOf (paperProfileOf context) context.input data
        witness := rfl

@[simp] theorem semantic_childrenOf_eq_paper
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : PaperProfile.Witness shape) :
    SemanticFold.childrenOf context data (semanticWitnessOf context witness) =
      PaperProfile.childrenOf (paperProfileOf context) context.input data
        witness := rfl

/-- Exact boundary between the paper NIFS relation and richer lifecycle
authority. The indexed `data` and `witness` are fixed outside the fields so a
later necessity countermodel cannot be rescued by existential witness
substitution. `canonicalTarget` records the current semantic fold's extra
honest-prover strengthening: it fixes the prover-supplied child commitments
and evaluations to one selected private parent split. The paper verifier
already computes child public inputs and copied fields, but does not require
that stronger private-opening identity. -/
structure PaperDecomposition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (result : FoldResult shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : PaperProfile.Witness shape) : Prop where
  paper : PaperProfile.Realization (paperProfileOf context) context.input data
    result.children witness
  polynomialInput : SemanticFold.PublicInputBound context data
  runningAuthority : RunningAuthority.Accepted context
  parentMaterialized : result.parent =
    PaperProfile.parentOf (paperProfileOf context) context.input data witness
  canonicalTarget : result.children =
    PaperProfile.childrenOf (paperProfileOf context) context.input data witness

/-- The existing rich semantic transition is exactly the independent paper
relation plus three implementation/lifecycle sidecars: polynomial-input
binding, incoming cached-parent authority, and outgoing parent-cache
materialization; and one semantic strengthening: deterministic canonical
children. The paper's `Pi_DEC` verifier determines child public inputs but
permits any prover commitment/evaluation messages satisfying its two
recomposition checks, so this last field remains an optimization target rather
than protocol authority. Transcript provenance is intentionally absent from
both sides; it belongs to physical refinement. -/
theorem resultTransition_iff_exists_paperDecomposition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (result : FoldResult shape publicRingColumns publicFits verifierRows) :
    ResultTransition context result ↔
      exists data : Data shape,
        exists witness : PaperProfile.Witness shape,
          PaperDecomposition context result data witness := by
  constructor
  · rintro ⟨data, semanticWitness, realized⟩
    let witness := paperWitnessOf semanticWitness
    refine ⟨data, witness, {
      paper := ?_
      polynomialInput := realized.input.publicInput
      runningAuthority := realized.running
      parentMaterialized := ?_
      canonicalTarget := ?_
    }⟩
    · exact {
        paper := realized.paper
        input := realized.input.sources
        challengesValid := realized.challengesValid
        piDecAccepted := by
          have canonical := realized.canonicalChildren
          have accepted :=
            (PiDEC.PaperVerifier.output_complete
              (ConcretePhi81.semantics context.key) productionGlobalParams
              (ConcretePhi81.decAlgebra context.key)
              (PiDECAlgebra.PaperVerifier.publicInputSplit context.key)
              (PiDECAlgebra.PaperVerifier.evaluationArity context.key)
              result.parent
              (SemanticFold.combinedAssignment context data semanticWitness)
              canonical.parentCombined canonical.parentValid).1
          simpa [witness, paperProfileOf, PaperProfile.decAlgebra,
            PaperProfile.decPublicInputSplit, PaperProfile.decEvaluationArity,
            SemanticFold.childrenOf,
            canonical.childrenEq, realized.parent_eq] using accepted
      }
    · simpa [witness] using realized.parent_eq
    · simpa [witness] using realized.children_eq
  · rintro ⟨data, witness, decomposed⟩
    let semanticWitness := semanticWitnessOf context witness
    refine ⟨data, semanticWitness, {
      paper := decomposed.paper.paper
      input := {
        publicInput := decomposed.polynomialInput
        sources := decomposed.paper.input
      }
      running := decomposed.runningAuthority
      challengesValid := decomposed.paper.challengesValid
      parent_eq := ?_
      children_eq := ?_
    }⟩
    · simpa [semanticWitness] using decomposed.parentMaterialized
    · simpa [semanticWitness] using decomposed.canonicalTarget

/-- Every rich fixed-active semantic transition refines the independent
abstract paper profile. The cached parent and polynomial-input sidecars are
forgotten only after the exact decomposition theorem above has checked them. -/
theorem ResultTransition.toPaperProfile
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.Transition
      (PaperProfile.toGenericProfile (paperProfileOf context)) context.input
      result.children := by
  rcases
      (resultTransition_iff_exists_paperDecomposition context result).mp
        accepted with
    ⟨data, witness, decomposed⟩
  exact ⟨PaperProfile.toGenericWitness (paperProfileOf context) data witness,
    decomposed.paper.toGeneric⟩

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

/-- The complete active result is bound to one valid combined parent opening
and its exact deterministic child split. -/
theorem ResultTransition.canonicalChildren
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
      PiDEC.CanonicalChildren.ForOpening
        (ConcretePhi81.decAlgebra context.key) result.parent assignment
        result.children := by
  exact Result.ResultTransition.canonicalChildren (arity := arity) accepted

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

/-- One active transition carries authoritative openings for the complete
incoming `CE(b)^14` accumulator through one shared semantic source witness. -/
theorem ResultTransition.inputRunningOpenings
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result) :
    ∃ inputAssignments : Fin productionGlobalParams.k ->
        Phi81Relation.Assignment
          (RelationShape shape publicRingColumns publicFits),
      ∀ child,
        CE.Holds (ConcretePhi81.semantics context.key) productionGlobalParams
          (context.input.running child) (inputAssignments child) := by
  exact Result.ResultTransition.inputRunningOpenings
    (arity := arity) accepted

/-- Active semantic acceptance already checks the supplied incoming parent
against the exact fourteen public running children. `parentBound` only names
the parent installed in the verifier-owned context; it is not a digest. -/
theorem ResultTransition.inputRunningPiDec
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : ResultTransition context result)
    (parent : Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (parentBound : context.runningParent = some parent) :
    PiDEC.Accepted (ConcretePhi81.decAlgebra context.key) {
      parent := parent
      children := context.input.running
    } := by
  have runningAuthority : RunningAuthority.Accepted context :=
    Result.ResultTransition.runningAuthority (arity := arity) accepted
  rcases
      (RunningAuthority.Accepted.iff_nonemptyBound_of_active
        (context := context) rfl).1 runningAuthority with
    ⟨bound⟩
  have parentEq : bound.parent = parent :=
    Option.some.inj (bound.parentBound.symm.trans parentBound)
  subst parent
  simpa [RunningAuthority.attempt, RunningAuthority.children,
    RunningAuthority.activeIndex, bound.active] using bound.piDec

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
