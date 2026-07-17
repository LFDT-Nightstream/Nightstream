import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct

/-!
Focused model-level regressions for canonical `Pi_CCS` CE materialization.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.output.source_order` | the two index casts are mutual inverses | total-only or permuted source alignment |
| `nifs.pi_ccs.output.fields` | structure/opening/point/evaluations/stage have one exact owner | independent or default-filled CE fields |
| `nifs.pi_ccs.output.y_ring` | CE membership authority is exactly full `yRing` source binding | sampled, short, or digest-only evaluation authority |
| `nifs.pi_ccs.output.y_zcol` | changing only `yZcol` cannot change CE output | accidental ownership of the delayed NC payload |
| `nifs.pi_ccs.output.points` | each authority branch ignores the other branch's point | false FE/NC coupling during composition |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.Tests

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uCommitment

example
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.sourceCount) :
    alignment.semanticIndex (alignment.productIndex source) = source := by
  simp

example
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    let output := materialize publicRingColumns publicFits alignment input
      rPrime message source
    output.constraintSystem = (input.source source).constraintSystem /\
      output.commitment = (input.source source).commitment /\
      output.publicInput = (input.source source).publicInput /\
      output.point = rPrime /\
      output.evaluations =
        claimedEvaluations message (alignment.semanticIndex source) /\
      output.stage = .fresh := by
  simp

/-- This regression permits arbitrary, possibly different `yZcol` families. -/
example
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (left right : OutputMessage shape)
    (sameYRing : left.yRing = right.yRing) :
    materialize publicRingColumns publicFits alignment input rPrime left =
      materialize publicRingColumns publicFits alignment input rPrime right := by
  exact materialize_eq_of_yRing_eq publicRingColumns publicFits alignment
    input rPrime left right sameYRing

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Sources.Data shape)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (points : VerifierPoints shape domain)
    (message : OutputMessage shape) :
    YRingBoundToSources data points message <->
      forall source,
        Phi81Relation.EvaluationsBound
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.assignment (alignment.semanticIndex source)) points.rPrime
          (materialize publicRingColumns publicFits alignment input
            points.rPrime message source).evaluations := by
  exact yRingBoundToSources_iff_outputEvaluationsBound publicRingColumns
    publicFits data alignment input points message

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (data : Sources.Data shape)
    (left right : VerifierPoints shape domain)
    (message : OutputMessage shape)
    (sameRowPoint : left.rPrime = right.rPrime) :
    YRingBoundToSources data left message <->
      YRingBoundToSources data right message := by
  exact yRingBoundToSources_iff_of_rPrime_eq data left right message sameRowPoint

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Sources.Data shape)
    (left right : VerifierPoints shape domain)
    (message : OutputMessage shape)
    (sameColumnPoint : left.sPrime = right.sPrime) :
    YZcolBoundToSources covers data left message <->
      YZcolBoundToSources covers data right message := by
  exact yZcolBoundToSources_iff_of_sPrime_eq covers data left right message
    sameColumnPoint

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.Tests
