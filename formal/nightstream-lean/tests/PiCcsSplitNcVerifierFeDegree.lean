import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree

/-!
Focused type-level regressions for the independent Split-NC FE degree contract.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.degree.row` | honest row rounds use the syntax-derived width | sparse-degree or phase-boundary drift |
| `nifs.pi_ccs.fe.degree.lane` | honest lane rounds use exactly three slots | accidental inheritance of the wider row bound |
| `nifs.pi_ccs.fe.degree.message` | typed phase bounds project unchanged to raw verifier messages | coefficient-order or width drift |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport

example : laneSumcheckDegreeBound = 2 := by
  decide

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (fixed : List K)
    (remaining : Nat)
    (rowPhase : fixed.length < shape.rowVariables)
    (length : fixed.length + 1 + remaining =
      shape.rowVariables + domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length =
        rowSumcheckDegreeBound (PublicInput.ofSources data) + 1 ∧
      message.degreeUpperBound =
        rowSumcheckDegreeBound (PublicInput.ofSources data) ∧
      ∀ point,
        message.evaluate
            Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
            point =
          Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
            Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
            (InitialSum.sumcheckPolynomial profile data coins)
            (fixed ++ [point]) remaining := by
  apply Represents.message_shape
  exact expectedRowRound_bounded profile data coins fixed remaining
    rowPhase length

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (fixed : List K)
    (remaining : Nat)
    (lanePhase : shape.rowVariables <= fixed.length)
    (length : fixed.length + 1 + remaining =
      shape.rowVariables + domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = 3 ∧
      message.degreeUpperBound = 2 ∧
      ∀ point,
        message.evaluate
            Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
            point =
          Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
            Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
            (InitialSum.sumcheckPolynomial profile data coins)
            (fixed ++ [point]) remaining := by
  simpa [laneSumcheckDegreeBound] using
    (Represents.message_shape
      (expectedLaneRound_quadratic profile data coins fixed remaining
        lanePhase length))

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Tests
