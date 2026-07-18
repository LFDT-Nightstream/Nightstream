import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.YZcolProjection

/-!
Public theorem-shape regressions for the fixed-profile two-limb parent
`y_zcol` evaluator refinement.

Owns: explicit checks that algebra transport preserves multiplication, exact
source-row satisfaction reaches the supplied-parent projection, and the
full-row entry point retains an indexed embedding premise.

Does not own: production witnesses, global R1CS satisfaction, beta transcript
derivation, parent authority, security reduction, costs, or row removal.

Emits constraints: no.

| Test | Obligation | Guarantee |
|---|---|---|
| transport | implementation extension multiplication maps to semantic multiplication | exact equality |
| source rows | exact 216 normalized rows plus one shared ladder imply the packed parent projection | theorem-shape regression |
| full rows | whole-R1CS satisfaction requires exact indexed leaf embedding | theorem-shape regression |
-/

namespace NightstreamTests.FPrimeRecursiveYZcolProjectionRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection
open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement

example (left right : ProjectionProgram.K) :
    toSemanticK (ProjectionProgram.K.mul left right) =
      Nightstream.SuperNeo.Concrete.K.mul
        (toSemanticK left) (toSemanticK right) :=
  toSemanticK_mul left right

example {assignment : Nat -> Nat} {point : ProjectionProgram.K}
    {parent : Nightstream.SuperNeo.Concrete.RingK}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceRows : Satisfies ownedSourceRows assignment)
    (sharedPowers : SharedPowersValid assignment point)
    (parentColumns : ParentColumnsMatch assignment parent) :
    Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.PairRightScalarMatches
      parent
      (Nightstream.SuperNeo.Concrete.K.add
        (toSemanticK (limb0Owner.evalTrace.output.value assignment))
        (Nightstream.SuperNeo.Concrete.K.mul
          Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition.extensionGenerator
          (toSemanticK (limb1Owner.evalTrace.output.value assignment))))
      (toSemanticK point) :=
  ownedSourceRows_refine_parentProjection canonical one sourceRows
    sharedPowers parentColumns

example {fullRows : List Row}
    {assignment : Nat -> Nat} {point : ProjectionProgram.K}
    {parent : Nightstream.SuperNeo.Concrete.RingK}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (embedded : SourceRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment)
    (sharedPowers : SharedPowersValid assignment point)
    (parentColumns : ParentColumnsMatch assignment parent) :
    Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.PairRightScalarMatches
      parent
      (Nightstream.SuperNeo.Concrete.K.add
        (toSemanticK (limb0Owner.evalTrace.output.value assignment))
        (Nightstream.SuperNeo.Concrete.K.mul
          Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition.extensionGenerator
          (toSemanticK (limb1Owner.evalTrace.output.value assignment))))
      (toSemanticK point) :=
  fullRows_refine_parentProjection canonical one embedded fullSatisfies
    sharedPowers parentColumns

end NightstreamTests.FPrimeRecursiveYZcolProjectionRefinement
