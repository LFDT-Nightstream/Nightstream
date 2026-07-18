import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.RhoEvaluations

/-!
Public theorem-shape regressions for the exact shared-rho refinement.

| Theorem | Exact premise | Conclusion |
|---|---|---|
| source rows | 1,620 rho rows plus 272 ladder rows | all 15 exact evaluator outputs |
| full rows | whole-R1CS satisfaction plus two indexed embeddings | the same local result |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionRhoEvaluationsRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement

example
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (rhoSourceSatisfies : Satisfies ownedSourceRows assignment)
    (ladderSourceSatisfies : Satisfies betaSourceRows assignment) :
    OutputsCorrect assignment
      (betaOwner.betaColumns.value assignment) :=
  ownedSourceRows_outputs_correct assignmentCanonical constantOne
    rhoSourceSatisfies ladderSourceSatisfies

example
    {fullRows : List Row} {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (rhoEmbedded : SourceRowsEmbedded fullRows)
    (ladderEmbedded : LadderRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    OutputsCorrect assignment
      (betaOwner.betaColumns.value assignment) :=
  fullRows_outputs_correct assignmentCanonical constantOne rhoEmbedded
    ladderEmbedded fullSatisfies

end NightstreamTests.FPrimeRecursivePiRlcProjectionRhoEvaluationsRefinement
