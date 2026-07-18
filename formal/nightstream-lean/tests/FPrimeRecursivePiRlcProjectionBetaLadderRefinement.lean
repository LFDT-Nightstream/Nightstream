import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.BetaLadder

/-!
Public theorem-shape regressions for the exact beta-ladder refinement.

| Theorem | Exact premise | Conclusion |
|---|---|---|
| `ownedSourceRows_ladder_sound` | 272 rows plus field/one invariants | 55 physical powers |
| `ownedSourceRows_y_zcol_sharedPowers` | same exact rows | both `y_zcol` leaves receive valid powers at the physical beta wire |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcProjectionBetaLadderRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement

example
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    owner.ladderTrace.powers.map
        (fun power => power.value assignment) =
      ProjectionProgram.K.powersFrom
        (owner.betaColumns.value assignment)
        ProjectionProgram.K.one owner.ladderTrace.powers.length :=
  ownedSourceRows_ladder_sound assignmentCanonical constantOne
    sourceSatisfies

example
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    FPrimeRecursiveYZcolProjection.Refinement.SharedPowersValid
      assignment (owner.betaColumns.value assignment) :=
  ownedSourceRows_y_zcol_sharedPowers assignmentCanonical constantOne
    sourceSatisfies

end NightstreamTests.FPrimeRecursivePiRlcProjectionBetaLadderRefinement
