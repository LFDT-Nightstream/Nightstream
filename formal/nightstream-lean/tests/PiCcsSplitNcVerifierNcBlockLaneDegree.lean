import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree

/-!
Focused regressions for the canonical block×lane NC degree contract.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.degree.bound` | individual degree ceiling is four and width is five | parameter drift |
| `nifs.pi_ccs.nc.block_lane.degree.polynomial.block` | block slices have five coefficients | block-MLE or selector degree drift |
| `nifs.pi_ccs.nc.block_lane.degree.polynomial.lane` | lane slices have five coefficients | lane-MLE or selector degree drift |
| `nifs.pi_ccs.nc.block_lane.degree.sumcheck` | every honest prefix/suffix round has five slots | decoder split or suffix-sum drift |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck

example : ncSumcheckDegreeBound = 4 ∧ ncMessageWidth = 5 := by
  decide

example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = 5 ∧
      message.degreeUpperBound = 4 ∧
      ∀ point, message.evaluate
          Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
          point = Mixing.qAtPoint covers data coins {
            block := cubeSlice before after length point
            lane := lane } :=
  qAtPoint_block_has_five_coefficients
    covers data coins lane before after length

example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = 5 ∧
      message.degreeUpperBound = 4 ∧
      ∀ point, message.evaluate
          Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
          point = Mixing.qAtPoint covers data coins {
            block := block
            lane := cubeSlice before after length point } :=
  qAtPoint_lane_has_five_coefficients
    covers data coins block before after length

/-- The five-slot ceiling applies to every exact honest SumCheck round, not
only to a typed point slice. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (fixed : List K)
    (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.blockVariables + domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = 5 ∧
      message.degreeUpperBound = 4 ∧
      ∀ point, message.evaluate
          ConcreteCarrier.extensionOps.toOps point =
        Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
          ConcreteCarrier.extensionOps.toOps
          (InitialSum.sumcheckPolynomial covers data coins)
          (fixed ++ [point]) remaining :=
  expectedRound_has_five_coefficients
    covers data coins fixed remaining length

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Tests
