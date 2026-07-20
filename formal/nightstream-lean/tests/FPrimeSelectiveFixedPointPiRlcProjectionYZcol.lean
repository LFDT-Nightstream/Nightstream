import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-!
Focused compile-time regressions for the fixed-point PiRLC shared + `y_zcol`
source and compact artifacts and their conditional semantic bridge.

| Tree level | Regression |
|---|---|
| protocol/phase | stable correspondence root exports the source bridge |
| family | exact 14-leaf, 5,724-row, 5,720-fresh-column census remains available |
| leaf | serializer indices and producer/consumer columns remain separate |
| selective coefficients | decoded compact rows have exact source/rewrite owners |
| semantics | selected-row satisfaction implies the typed row interface; honest source equations construct selected rows |
-/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiRlcProjectionYZcol

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionPhi81
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.ProjectionCheck

#check Checked.exactRows
#check Checked.structureValid
#check Checked.sourceStagePathsUnique
#check Checked.sourceStageLeafCounts
#check Census.definitionOutputs_eq_allocatedColumns
#check ProducerBinding.serializerIndicesMatch
#check ProducerBinding.producerColumnsMatchTrace
#check ActiveBridge.rowsSatisfied_of_sourceRows
#check ActiveBridge.rows_decodedOutput_eq_messageAggregate_or_badRoot
#check Selective.Materialized.Checked.uniqueOwner
#check Selective.Materialized.Checked.uniqueOwnerAndFamily
#check Selective.Materialized.Artifact.rowCount
#check Selective.Materialized.Artifact.allRowsDecode_true
#check Selective.SourceDecode.retainedSlotFast_eq
#check Selective.SourceDecode.completeSourcePartition
#check Selective.SourceProgram.compilerSourceOutputDefinitions_exact
#check Selective.SourceProgram.compilerDefinitionColumnsKnown
#check Selective.SourceProgram.sourceAssignmentCompilerDefinitionsHold
#check Selective.RewriteBridge.derivedRecurrenceRegistryExact
#check Selective.RewriteBridge.decodedDerivedRecurrenceRegistryExact
#check Selective.RewriteBridge.decodedDerivedRecurrenceRegistered
#check Selective.RewriteBridge.decodedDerivedOutputBaseZero
#check Selective.RewriteBridge.derivedStepHolds_witnessRecurrence
#check Selective.RewriteBridge.witnessRewriteProgramHolds
#check Selective.RewriteBridge.Coefficients.rewriteCoefficientsMatch_of_shape_check_true
#check Selective.RewriteBridge.Coefficients.rewriteCoefficientChunkRangesOrdered
#check Selective.RewriteBridge.Coefficients.rewriteCoefficientDataChunkLengthsExact
#check Selective.RewriteBridge.Coefficients.rewriteCoefficientDataWithinCertificateLimit
#check Selective.RewriteBridge.Coefficients.rewriteCoefficientChunksExact
#check Selective.RewriteBridge.rewriteCoefficientsExact
#check Selective.RewriteBridge.retainedCoefficientsExact
#check Selective.QuadraticRefinement.groupMatches_of_shape_check_true
#check Selective.QuadraticRefinement.evaluationChunkRangesOrdered
#check Selective.QuadraticRefinement.evaluationChunkRangesCoverCount
#check Selective.QuadraticRefinement.Evaluation.Chunk0.dataLengthExact
#check Selective.QuadraticRefinement.Evaluation.Chunk1.dataLengthExact
#check Selective.QuadraticRefinement.Evaluation.Chunk2.dataLengthExact
#check Selective.QuadraticRefinement.Evaluation.Chunk3.dataLengthExact
#check Selective.QuadraticRefinement.Evaluation.Chunk4.dataLengthExact
#check Selective.QuadraticRefinement.Evaluation.Chunk5.dataLengthExact
#check Selective.QuadraticRefinement.Evaluation.Chunk6.dataLengthExact
#check Selective.QuadraticRefinement.Product.dataLengthExact
#check Selective.QuadraticRefinement.evaluationGroupsExact
#check Selective.QuadraticRefinement.productGroupsExact
#check Selective.Soundness.selectedRows_imply_rowsSatisfied
#check Selective.Soundness.selectedRows_decodedOutput_eq_messageAggregate_or_badRoot
#check Selective.HonestAssignment.materializedWitnessRewriteProgramHolds
#check Selective.HonestAssignment.materializedDerivedWitnessRecurrencesHold
#check Selective.HonestAssignment.TerminalSemantics.exists_selectedRows_of_honestSource

example : Checked.artifact.sourceRows.length = 5724 :=
  Census.rowCount

example : Checked.artifact.allocatedColumns.length = 5720 :=
  Census.allocatedColumnCount

example : Checked.sourceStageLeaves.length = 14 :=
  Census.sourceStageLeafCount

example :
    (Checked.sourceStageLeaves.map SourceStageLeaf.rowCount).sum = 5724 :=
  Census.sourceStageRowCount

example : Selective.Materialized.Artifact.decodedRows.length = 1254 :=
  Selective.Materialized.Artifact.rowCount

example : Selective.Materialized.Artifact.rewriteRows.length = 1250 :=
  Selective.Materialized.Artifact.rewriteRowCount

example : Selective.Materialized.Artifact.retainedRows.length = 4 :=
  Selective.Materialized.Artifact.retainedRowCount

example {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {assignment : Nat → Nat}
    (selected : Selective.Materialized.Semantics.RowsSatisfied
      Selective.Materialized.Artifact.decodedRows assignment)
    (selectorOne :
      assignment Selective.Materialized.Checked.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1) :
    ProjectionIdentity.RowsSatisfied
      (ActiveBridge.tracePair shape sourceCount)
      (Selective.SourceProgram.sourceAssignment assignment) :=
  Selective.Soundness.selectedRows_imply_rowsSatisfied sourceCount selected
    selectorOne constantOne

example {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {assignment : Nat → Nat}
    (selected : Selective.Materialized.Semantics.RowsSatisfied
      Selective.Materialized.Artifact.decodedRows assignment)
    (selectorOne :
      assignment Selective.Materialized.Checked.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1)
    {producer : SourceRole shape → Nat}
    (upstream : ProducerBinding.UpstreamProducerColumnsBound producer)
    {message : OutputMessage shape}
    (yZcolBound : BindingsHoldFor .yZcolOutput
      (semanticAssignment
        (Selective.SourceProgram.sourceAssignment assignment))
      producer message) :
    ProjectionIdentity.decodedOutput
          (ActiveBridge.tracePair shape sourceCount)
          (Selective.SourceProgram.sourceAssignment assignment) =
        sourceAggregate
          (ProjectionIdentity.decodedChallenges
            (ActiveBridge.tracePair shape sourceCount)
            (Selective.SourceProgram.sourceAssignment assignment))
          message.yZcol ∨
      BatchBadRoot K.ops
        (ProjectionProgram.BatchIdentity
          (ActiveBridge.tracePair shape sourceCount).traces
          (Selective.SourceProgram.sourceAssignment assignment)) :=
  Selective.Soundness.selectedRows_decodedOutput_eq_messageAggregate_or_badRoot
    sourceCount selected selectorOne constantOne upstream yZcolBound

example {source : Nat → Nat}
    (honest : Selective.HonestAssignment.HonestSourceBoundary source) :
    ∃ assignment,
      assignment Selective.Materialized.Checked.constantOneColumn = 1 ∧
      assignment Selective.Materialized.Checked.steadySelectorColumn = 1 ∧
      Selective.Materialized.Semantics.AssignmentCanonical assignment ∧
      Selective.Materialized.Semantics.RowsSatisfied
        Selective.Materialized.Artifact.decodedRows assignment :=
  Selective.HonestAssignment.TerminalSemantics.exists_selectedRows_of_honestSource
    honest

example {source seed : Nat → Nat}
    (seedOne : seed 0 = 1)
    (sourceEq : Selective.SourceProgram.sourceAssignment seed = source)
    (lowSampledWireEquation :
      (Checked.lowTrace.pairProductValues source).foldr K.add K.zero =
        K.add
          (Checked.lowTrace.quotientPhiProduct.output.value source)
          (Checked.lowTrace.outputEvaluation.output.value source))
    (highSampledWireEquation :
      (Checked.highTrace.pairProductValues source).foldr K.add K.zero =
        K.add
          (Checked.highTrace.quotientPhiProduct.output.value source)
          (Checked.highTrace.outputEvaluation.output.value source)) :
    Selective.HonestAssignment.HonestSourceBoundary source :=
  ⟨seed, seedOne, sourceEq, lowSampledWireEquation,
    highSampledWireEquation⟩

end Nightstream.Tests.FPrimeSelectiveFixedPointPiRlcProjectionYZcol
