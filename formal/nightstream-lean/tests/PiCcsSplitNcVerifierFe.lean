import Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-!
Focused regressions for the independent Split-NC FE polynomial layer.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.domain.row_lane` | serialization is exactly `row ++ lane` | row/lane transcript swap |
| `nifs.pi_ccs.fe.domain.profile` | production-facing entrypoints reject zero fresh sources and non-64 lane cubes | dead shape guard |
| `nifs.pi_ccs.fe.polynomial` | exact serialization decodes to the typed polynomial | truncation or reconstruction drift |
| `nifs.pi_ccs.fe.polynomial.arity` | wrong coordinate count returns `none` | zero-fallback acceptance |
| `nifs.pi_ccs.fe.output.y_ring` | source-bound `yRing` suffices regardless of `yZcol` | hidden cross-branch authority |
| `nifs.pi_ccs.fe.initial.signed` | independent residual identity is facade-visible | hidden carried-bridge premise |
| `nifs.pi_ccs.fe.initial.complete` | honest FE truth equals the generic SumCheck cube sum | drift from semantic truth |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

private def supportedShape : SemanticShape where
  rowVariables := 1
  logicalWidth := 1
  freshCount := 1
  runningCount := 1
  matrixCount := 1

private def twoMatrixShape : SemanticShape where
  rowVariables := 1
  logicalWidth := 1
  freshCount := 1
  runningCount := 1
  matrixCount := 2

private def zeroFreshShape : SemanticShape where
  rowVariables := 1
  logicalWidth := 1
  freshCount := 0
  runningCount := 1
  matrixCount := 1

private def zeroRowShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 1
  freshCount := 1
  runningCount := 1
  matrixCount := 1

private def productionLaneDomain : FlatNcDomain where
  columnVariables := 0
  laneVariables := 6

/-- The fixed one-fresh/one-running, 64-lane profile is admitted. -/
example : SupportedProfile supportedShape productionLaneDomain := by
  constructor <;> decide

/-- A zero-fresh shape cannot enter any production-facing FE evaluator. -/
example : ¬ SupportedProfile zeroFreshShape productionLaneDomain := by
  intro profile
  have impossible : 0 < 0 := by
    simpa [zeroFreshShape] using profile.fresh_nonempty
  omega

/-- Native derives `ell_n` with a minimum padded row cube of size two, so a
zero-row-variable model is not a supported production profile. -/
example : ¬ SupportedProfile zeroRowShape productionLaneDomain := by
  intro profile
  have impossible : 0 < 0 := by
    simpa [zeroRowShape] using profile.row_nonempty
  omega

/-- The carried schedule has inner exponent `K+h+jN` and the terminal adds
the separate outer `N` shift. For `K=1`, `h=0`, `j=1`, `N=2`, these are 3
and 5 respectively. -/
example :
    carriedGammaExponent twoMatrixShape (0 : Fin 1) (1 : Fin 2) = 3 /\
      twoMatrixShape.sourceCount +
        carriedGammaExponent twoMatrixShape (0 : Fin 1) (1 : Fin 2) = 5 := by
  decide

/-- The executable round order is row first even though the mathematical
display convention writes the lane variable first. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (row : CubePoint K shape.rowVariables)
    (lane : CubePoint K domain.laneVariables) :
    (Point.coordinates ({ row := row, lane := lane } : Point shape domain)) =
      row.coordinates ++ lane.coordinates := by
  rfl

/-- Exact-length decoding is a left inverse of typed serialization. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain) :
    polynomial profile data coins point.coordinates =
      some (qAtPoint profile data coins point) := by
  exact polynomial_coordinates_eq_qAtPoint profile data coins point

/-- A malformed round count is rejection, not the zero polynomial. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (coordinates : List K)
    (different : coordinates.length ≠
      shape.rowVariables + domain.laneVariables) :
    polynomial profile data coins coordinates = none := by
  exact polynomial_eq_none_of_length_ne
    profile data coins coordinates different

/-- FE consumes only the source-bound `yRing` branch. An arbitrary independent
`yZcol` payload cannot influence the FE terminal. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (point : Point shape domain)
    (arbitraryYZcol : Fin shape.sourceCount -> Fin ringDegree -> K) :
    terminalFromMessage profile (PublicInput.ofSources data) coins point
        ({ yRing := sourceYRingAt data point.row
           yZcol := arbitraryYZcol } : OutputMessage shape) =
      qAtPoint profile data coins point := by
  apply terminalFromMessage_eq_qAtPoint_of_yRing_eq
  rfl

/-- The facade exposes an unconditional exact residual identity. In
particular, callers do not supply the carried selector bridge as a premise. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    ConcreteCarrier.extensionOps.sub
        (initial profile (PublicInput.ofSources data) coins)
        (InitialSum.hypercubeSum profile data coins) =
      InitialSum.mixedResidual profile data coins :=
  InitialSum.CarriedBridge.initial_sub_hypercubeSum_eq_mixedResidual
    profile data coins

/-- Independent FE truth reproduces the exact recursive Boolean sum consumed
by SumCheck truth construction. Transcript and raw-message acceptance remain
separate obligations. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data) :
    initial profile (PublicInput.ofSources data) coins =
      InitialSum.sumcheckHypercubeSum profile data coins :=
  InitialSum.CarriedBridge.initial_eq_sumcheckHypercubeSum_of_truth
    profile data coins truth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Tests
