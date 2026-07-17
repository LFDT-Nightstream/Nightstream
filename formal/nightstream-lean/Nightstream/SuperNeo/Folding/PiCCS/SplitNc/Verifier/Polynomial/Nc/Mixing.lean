import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.Gamma
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath

/-!
Challenge mixing for the independent Split-NC norm polynomial.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: source compression and equality gating before SumCheck.
Constraint family: source gamma weights and column/lane equality selectors;
this file emits no rows.

Owns: the flat-domain NC challenge carrier, source compression,
equality-gated polynomial evaluation, fail-closed list evaluation, and
flat-domain specializations of the shared gamma-schedule relations.

Does not own: gamma schedules, source projection, the zero initial claim,
SumCheck messages, root probability, transcript derivation, `yZcol`, terminal
binding, Rust, R1CS, row emission, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `paperNc` is the paper's relative norm mixture;
`paperJointQ` applies its outer `gamma^K` shift; `splitV1` records the current
production schedule. The comparison theorems do not approve Split-V1. In
particular, its common gamma factor creates a deterministic zero at
`gamma = 0` for every source table.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.mixing.coins` | `betaM`, `betaA`, and `gamma` are typed arguments | security boundary | `Coins` |
| `nifs.pi_ccs.nc.mixing.gamma` | consume the shared named exponent convention | direct dataflow | `Mixing.Gamma` |
| `nifs.pi_ccs.nc.mixing.source` | every source cubic is gamma-compressed once | computed | `mixedRangeAt` |
| `nifs.pi_ccs.nc.mixing.selectors` | column and lane equality gates multiply the source mix | computed | `qAtPoint` |
| `nifs.pi_ccs.nc.mixing.decode` | malformed coordinate arity rejects | checked | `polynomial` |
| assurance | `paperJointQ = gamma^K * paperNc` | derived | `paperJointQMix_eq_gammaPowFresh_mul_paperNcMix` |
| assurance | every Split-V1 source sum is `gamma * paperNc` | derived | `splitV1Sum_eq_gamma_mul_paperNcSum` |
| assurance | pointwise `splitV1 = gamma * paperNc` and vanishes at `gamma=0` | derived | `splitV1Mix_eq_gamma_mul_paperNcMix`, `splitV1Mix_eq_zero_of_gamma_zero` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Verifier challenges consumed by the NC polynomial. A later Poseidon2
machine must derive all three fields after binding the statement. -/
structure Coins (domain : FlatNcDomain) where
  betaM : CubePoint K domain.columnVariables
  betaA : CubePoint K domain.laneVariables
  gamma : K

/-- Gamma compression of every independently derived source cubic at one
typed NC point. -/
def mixedRangeAt
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (sourceExponent shape convention source)
        (SourceProjection.rangeValueAt covers data source point)

/-- Equality-gated NC polynomial over the column/lane product domain. -/
def qAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) : K :=
  K.mul
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.column coins.betaM)
      (SumCheckTruthPath.pointEquality ops point.lane coins.betaA))
    (mixedRangeAt convention covers data coins point)

/-- Fail-closed list evaluator in the fixed column-then-lane order. -/
def polynomial
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K) : Option K :=
  match Point.decode (domain := domain) coordinates with
  | some point => some (qAtPoint convention covers data coins point)
  | none => none

/-- Exact typed serialization evaluates the same NC polynomial. -/
theorem polynomial_coordinates_eq_qAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    polynomial convention covers data coins point.coordinates =
      some (qAtPoint convention covers data coins point) := by
  rw [polynomial, Point.decode_coordinates]

/-- Malformed coordinate arity rejects rather than truncating or padding. -/
theorem polynomial_eq_none_of_length_ne
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K)
    (different : coordinates.length ≠
      domain.columnVariables + domain.laneVariables) :
    polynomial convention covers data coins coordinates = none := by
  rw [polynomial, Point.decode_eq_none_of_length_ne coordinates different]

/-- The paper joint-Q schedule is exactly the relative paper NC schedule
multiplied by the outer `gamma^K` factor. -/
theorem paperJointQMix_eq_gammaPowFresh_mul_paperNcMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    mixedRangeAt .paperJointQ covers data coins point =
      K.mul
        (TargetPolynomial.power ops.toOps coins.gamma shape.freshCount)
        (mixedRangeAt .paperNc covers data coins point) := by
  unfold mixedRangeAt
  exact paperJointQSum_eq_gammaPowFresh_mul_paperNcSum coins.gamma _

/-- The current Split-V1 schedule has one common gamma factor relative to the
paper NC schedule. -/
theorem splitV1Mix_eq_gamma_mul_paperNcMix
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    mixedRangeAt .splitV1 covers data coins point =
      K.mul coins.gamma
        (mixedRangeAt .paperNc covers data coins point) := by
  unfold mixedRangeAt
  exact splitV1Sum_eq_gamma_mul_paperNcSum coins.gamma _

/-- Split-V1's common factor makes `gamma = 0` an unconditional root,
regardless of the source assignments. -/
theorem splitV1Mix_eq_zero_of_gamma_zero
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain)
    (gammaZero : coins.gamma = K.zero) :
    mixedRangeAt .splitV1 covers data coins point = K.zero := by
  rw [splitV1Mix_eq_gamma_mul_paperNcMix]
  rw [gammaZero]
  change ops.mul ops.zero _ = ops.zero
  rw [laws.mul_comm, laws.mul_zero]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
