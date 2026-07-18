import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.TraceNormalForm

/-!
Fixed-history trace-tree packaging over the profile-neutral Phi81 normal form.

Assurance tier: model-level.

Owns: diagnostic public-role shape facts and construction of the historical
`ReductionArtifact`. Does not own: the polynomial proof, active profiles,
generated rows, sampled-root security, transcript authority, costs, or row
removal. Emits constraints: no.

| Stage family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `identities.public.normal_form` | one exact generic trace determines its Phi81 remainder | derived | `ProjectionPhi81.exact_output_eq_phi81Combine` |
| `identities.public.tree` | package every fixed-history public role | direct dataflow | `reductionArtifact_of_exact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

abbrev pairAt {count : Nat} (trace : ProjectionTrace)
    (pairArity : trace.pairs.length = count) (index : Fin count) : PairTrace :=
  ProjectionPhi81.pairAt trace pairArity index

/-- Legacy namespace wrapper for the profile-neutral trace theorem. -/
theorem exact_output_eq_phi81Combine
    {count : Nat} (assignment : Nat -> Nat) (trace : ProjectionTrace)
    (pairArity : trace.pairs.length = count)
    (rhoWidth : ∀ index,
      (pairAt trace pairArity index).rhoColumns.length =
        Concrete.ringDegree)
    (inputWidth : ∀ index,
      (pairAt trace pairArity index).inputColumns.length =
        Concrete.ringDegree)
    (outputWidth : trace.outputColumns.length = Concrete.ringDegree)
    (quotientWidth : trace.quotientColumns.length = 53)
    (maxDegree : trace.maxDegree = 106)
    (exact : (trace.identity assignment).Exact) :
    values assignment trace.outputColumns =
      phi81Combine
        (fun index =>
          values assignment (pairAt trace pairArity index).rhoColumns)
        (fun index =>
          values assignment (pairAt trace pairArity index).inputColumns) :=
  ProjectionPhi81.exact_output_eq_phi81Combine assignment trace pairArity
    rhoWidth inputWidth outputWidth quotientWidth maxDegree exact

/-- Static shape facts consumed by the diagnostic public trace tree. -/
structure TraceShapeArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) : Prop where
  challengeWidth : ∀ role index,
    (tree.publicPairAt role index).rhoColumns.length =
      Concrete.ringDegree
  inputWidth : ∀ role index,
    (tree.publicPairAt role index).inputColumns.length =
      Concrete.ringDegree
  outputWidth : ∀ role,
    (tree.publicTrace role).outputColumns.length =
      Concrete.ringDegree
  quotientWidth : ∀ role,
    (tree.publicTrace role).quotientColumns.length = 53
  maxDegree : ∀ role, (tree.publicTrace role).maxDegree = 106

theorem quotientRemainder_of_exact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {assignment : Nat → Nat} {tree : TraceTree arity matrixCount}
    (shape : TraceShapeArtifact tree)
    (role : PublicRole matrixCount)
    (exact : ((tree.publicTrace role).identity assignment).Exact) :
    values assignment (tree.publicTrace role).outputColumns =
      phi81Combine
        (fun index =>
          values assignment (tree.publicPairAt role index).rhoColumns)
        (fun index =>
          values assignment (tree.publicPairAt role index).inputColumns) := by
  exact exact_output_eq_phi81Combine assignment (tree.publicTrace role)
    (tree.publicPairArity role)
    (shape.challengeWidth role) (shape.inputWidth role)
    (shape.outputWidth role)
    (shape.quotientWidth role) (shape.maxDegree role) exact

/-- Per-role exactness plus the separate shape artifact constructs the
complete deterministic diagnostic reduction artifact. -/
theorem reductionArtifact_of_exact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {assignment : Nat → Nat} {tree : TraceTree arity matrixCount}
    (shape : TraceShapeArtifact tree)
    (exact : ∀ role, ((tree.publicTrace role).identity assignment).Exact) :
    ReductionArtifact assignment tree where
  equation role := quotientRemainder_of_exact shape role (exact role)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction
