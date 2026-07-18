import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolIdentities

/-!
Public theorem-shape regressions for both complete active PiRLC `y_zcol`
identities.

| Theorem | Exact premises | Conclusion |
|---|---|---|
| batch acceptance | canonical assignment, constant one, and separate beta/rho/output/local source-row satisfaction | both complete identities evaluate correctly |
| deterministic partition | the same four physical row premises | `BatchExact` or the named `BatchBadRoot` event |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionYZcolIdentitiesRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement
open Nightstream.SuperNeo.ProjectionCheck

example
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    BatchAccepted ProjectionProgram.K.ops
      (ProjectionProgram.BatchIdentity traces assignment) :=
  completeSourceRows_batchAccepted assignmentCanonical constantOne
    betaSatisfies rhoSatisfies outputSatisfies localSatisfies

example
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    BatchExact (ProjectionProgram.BatchIdentity traces assignment) ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity traces assignment) :=
  completeSourceRows_batchExact_or_badRoot assignmentCanonical constantOne
    betaSatisfies rhoSatisfies outputSatisfies localSatisfies

end NightstreamTests.FPrimeRecursivePiRlcProjectionYZcolIdentitiesRefinement
