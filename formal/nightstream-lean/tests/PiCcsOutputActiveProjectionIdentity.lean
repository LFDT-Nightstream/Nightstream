import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity

/-!
Public theorem-shape regression for the artifact-independent active PiCCS
`y_zcol` source to PiRLC projection boundary.
-/

namespace tests.PiCcsOutputActiveProjectionIdentity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity
open Nightstream.Implementation.R1CS.ProjectionPhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.ProjectionCheck

#check TracePair
#check ShapeValid
#check RowsSatisfied
#check TracePair.inputConsumer
#check rows_batchAccepted
#check decodedInputs_eq_inputConsumer
#check batchExact_decodedOutput_eq_sourceAggregate
#check batchExact_decodedOutput_eq_messageAggregate

example
    {shape : SemanticShape}
    {pair : TracePair shape}
    {assignment : Nat -> Nat}
    {producer : SourceRole shape -> Nat}
    {message : OutputMessage shape}
    (valid : ShapeValid pair)
    (constantOne : assignment 0 = 1)
    (rows : RowsSatisfied pair assignment)
    (consumerMatch : YZcolConsumer.ConsumerMatches producer
      pair.inputConsumer)
    (yZcolBound : BindingsHoldFor .yZcolOutput
      (semanticAssignment assignment) producer message) :
    decodedOutput pair assignment =
        sourceAggregate (decodedChallenges pair assignment) message.yZcol ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity pair.traces assignment) :=
  rows_decodedOutput_eq_messageAggregate_or_badRoot valid constantOne rows
    consumerMatch yZcolBound

end tests.PiCcsOutputActiveProjectionIdentity
