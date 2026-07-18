import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

/-!
Focused regressions for independent Split-NC source mixing.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.mixing.exponent` | paper, joint-Q, and Split-V1 schedules remain distinct | convention drift |
| `nifs.pi_ccs.nc.mixing.decode` | typed points round-trip and malformed arity rejects | implicit padding |
| `nifs.pi_ccs.nc.mixing.shift.paper_joint_q` | joint-Q equals `gamma^K` times relative NC | paper shift drift |
| `nifs.pi_ccs.nc.mixing.shift.split_v1` | Split-V1 equals `gamma` times relative NC | hidden production convention |
| `nifs.pi_ccs.nc.mixing.root.zero` | Split-V1 vanishes unconditionally at `gamma=0` | unnamed bad event |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

private def exponentShape : SemanticShape where
  rowVariables := 1
  logicalWidth := 1
  freshCount := 2
  runningCount := 1
  matrixCount := 1

private def sourceOne : Fin exponentShape.sourceCount :=
  ⟨1, by decide⟩

/-- One source index exposes all three schedules: `1`, `K+1=3`, and `2`. -/
example :
    sourceExponent exponentShape .paperNc sourceOne = 1 /\
      sourceExponent exponentShape .paperJointQ sourceOne = 3 /\
      sourceExponent exponentShape .splitV1 sourceOne = 2 := by
  decide

/-- Exact typed serialization evaluates the same equality-gated source mix. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    polynomial convention covers data coins point.coordinates =
      some (qAtPoint convention covers data coins point) :=
  polynomial_coordinates_eq_qAtPoint convention covers data coins point

/-- A malformed product-domain point is rejected. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K)
    (different : coordinates.length ≠
      domain.columnVariables + domain.laneVariables) :
    polynomial convention covers data coins coordinates = none :=
  polynomial_eq_none_of_length_ne
    convention covers data coins coordinates different

/-- The paper joint-Q shift is explicit rather than folded into NC semantics. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    mixedRangeAt .paperJointQ covers data coins point =
      K.mul
        (TargetPolynomial.power ConcreteCarrier.extensionOps.toOps
          coins.gamma shape.freshCount)
        (mixedRangeAt .paperNc covers data coins point) :=
  paperJointQMix_eq_gammaPowFresh_mul_paperNcMix covers data coins point

/-- Split-V1 carries a common gamma factor absent from relative paper NC. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    mixedRangeAt .splitV1 covers data coins point =
      K.mul coins.gamma (mixedRangeAt .paperNc covers data coins point) :=
  splitV1Mix_eq_gamma_mul_paperNcMix covers data coins point

/-- At zero gamma, Split-V1 accepts the zero mixture for every source table. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain)
    (gammaZero : coins.gamma = K.zero) :
    mixedRangeAt .splitV1 covers data coins point = K.zero :=
  splitV1Mix_eq_zero_of_gamma_zero covers data coins point gammaZero

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.Tests
