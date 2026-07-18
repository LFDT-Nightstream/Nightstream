import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolNormalForm

/-!
Public theorem-shape regressions for the active PiRLC `y_zcol` normal form.

| Theorem | Explicit boundary | Result |
|---|---|---|
| physical normal form | coefficient-wise exactness of both traces | decoded output is the typed source aggregate |
| semantic binding | challenge, input, and output column equality | semantic parent is the source aggregate |
| row composition | four exact row sets plus semantic column equality | parent equality or named bad root |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionYZcolNormalForm

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.ProjectionCheck

example
    {assignment : Nat → Nat}
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity traces assignment)) :
    decodedOutput assignment =
      sourceAggregate (decodedChallenges assignment)
        (decodedInputs assignment) :=
  batchExact_decodedOutput_eq_sourceAggregate exact

example
    {assignment : Nat → Nat}
    {challenges : Fin sourceCount → RingF}
    {inputs : Fin sourceCount → RingK}
    {parent : RingK}
    (columns : SemanticColumnsMatch assignment challenges inputs parent)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity traces assignment)) :
    parent = sourceAggregate challenges inputs :=
  batchExact_parent_eq_sourceAggregate columns exact

example
    {assignment : Nat → Nat}
    {challenges : Fin sourceCount → RingF}
    {inputs : Fin sourceCount → RingK}
    {parent : RingK}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment)
    (columns : SemanticColumnsMatch assignment challenges inputs parent) :
    parent = sourceAggregate challenges inputs ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity traces assignment) :=
  completeSourceRows_parent_eq_sourceAggregate_or_badRoot
    assignmentCanonical constantOne betaSatisfies rhoSatisfies
    outputSatisfies localSatisfies columns

end NightstreamTests.FPrimeRecursivePiRlcProjectionYZcolNormalForm
