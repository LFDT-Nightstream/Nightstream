import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism

/-!
Focused kernel regressions for source-derived `yZcol` base-field transport.

Owns: public-surface checks for the projection leaves, the finite-combination
theorem, and the product-wide production `PiDEC` transport theorem.

Does not own: a witness for the required source/recomposition equalities,
acceptance-to-authority refinement, Rust/R1CS conformance, or row removal.

Emits constraints: no.

| Protocol | Phase | Family | Regression |
|---|---|---|---|
| SplitNC | column projection | zero / add / scale | the independently recomputed leaf theorems remain exported |
| `Pi_DEC` | base-field combination | every 54-lane sidecar | finite combination is available at arbitrary source shape |
| `Pi_CCS` product | canonical source transport | every source / lane | transport requires the explicit per-source recomposition premise |
-/

namespace tests.PiCcsOutputClaimsYZcolBaseLinear

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

#check diagonal_zero
#check diagonal_add
#check diagonal_scale
#check yZcolForAssignment_zero
#check yZcolForAssignment_add
#check yZcolForAssignment_scale
#check yZcolEvaluation_combine
#check yZcolEvaluation_piDecRecompose
#check canonicalYZcol_piDec_transport
#check canonicalYZcol_product_piDec_transport

/-- The finite-combination theorem is independent of any verifier acceptance
predicate and covers all 54 lanes extensionally. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (weights : Fin count -> F)
    (assignments : Fin count ->
      Phi81Relation.Assignment (relationShape shape))
    (sPrime : CubePoint K domain.columnVariables) :
    yZcolEvaluation covers (combineAssignments weights assignments) sPrime =
      combineEvaluations weights
        (fun index => yZcolEvaluation covers (assignments index) sPrime) := by
  exact yZcolEvaluation_combine covers weights assignments sPrime

/-- Product-wide transport remains conditional on one explicit hard
assignment equality for every source; the test keeps that premise visible. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data shape)
    (points : VerifierPoints shape domain)
    (children : Fin shape.sourceCount -> Fin productionGlobalParams.k ->
      Phi81Relation.Assignment (relationShape shape))
    (sourceRecomposition : forall source,
      data.assignment source =
        Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
          (children source)) :
    forall source lane,
      canonicalYZcol covers data points source lane =
        combineEvaluations
          Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
          (fun index =>
            yZcolEvaluation covers (children source index) points.sPrime)
          lane := by
  exact canonicalYZcol_product_piDec_transport covers data points children
    sourceRecomposition

end tests.PiCcsOutputClaimsYZcolBaseLinear
