import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity

/-!
Domain-independent gamma schedules for Split-NC source compression.

Assurance tier: model-level.

Owns: the three named source-exponent conventions and exact algebraic
relations among their finite source mixtures.

Does not own: an NC point domain, source projection, equality selectors,
SumCheck, transcript derivation, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: `.paperNc` is the relative paper NC schedule.
`.paperJointQ` records the paper joint-polynomial outer shift. `.splitV1`
records the current production convention only as a diagnostic; the theorem
below exposes its deterministic common gamma factor.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.mixing.gamma.paper_nc` | source `i` has relative exponent `i` | computed | `sourceExponent .paperNc` |
| `nifs.pi_ccs.nc.mixing.gamma.paper_joint_q` | joint `Q` shifts NC by the fresh-source count | computed | `sourceExponent .paperJointQ` |
| `nifs.pi_ccs.nc.mixing.gamma.split_v1` | production Split-V1 uses exponent `i+1` | diagnostic | `sourceExponent .splitV1` |
| assurance | every joint-Q source sum is `gamma^K * paperNc` | derived | `paperJointQSum_eq_gammaPowFresh_mul_paperNcSum` |
| assurance | every Split-V1 source sum is `gamma * paperNc` | derived | `splitV1Sum_eq_gamma_mul_paperNcSum` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- The three concrete gamma schedules that require an explicit refinement
decision. -/
inductive GammaConvention where
  /-- Relative norm block displayed by SuperNeo Section 7.3. -/
  | paperNc
  /-- Relative paper norm block after the joint `Q` outer `gamma^K` shift. -/
  | paperJointQ
  /-- Current production Split-NC schedule. -/
  | splitV1
deriving Repr, DecidableEq

/-- Source exponent under one named schedule. `freshCount` is the paper's
incoming CCS count `K`. -/
def sourceExponent
    (shape : SemanticShape)
    (convention : GammaConvention)
    (source : Fin shape.sourceCount) : Nat :=
  match convention with
  | .paperNc => source.val
  | .paperJointQ => shape.freshCount + source.val
  | .splitV1 => source.val + 1

private def shiftLaws : TargetPolynomial.ShiftLaws ops.toOps where
  one_mul := laws.one_mul
  mul_assoc := laws.mul_assoc
  mul_zero := laws.mul_zero
  mul_add := laws.left_distrib

/-- The paper joint-Q shift is independent of the values being mixed. -/
theorem paperJointQSum_eq_gammaPowFresh_mul_paperNcSum
    {shape : SemanticShape}
    (gamma : K)
    (values : Fin shape.sourceCount → K) :
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun source =>
          SignedJointIdentity.gammaTerm ops gamma
            (sourceExponent shape .paperJointQ source) (values source)) =
      K.mul
        (TargetPolynomial.power ops.toOps gamma shape.freshCount)
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.sourceCount) (fun source =>
            SignedJointIdentity.gammaTerm ops gamma
              (sourceExponent shape .paperNc source) (values source))) := by
  change
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.sourceCount) _ =
      ops.mul
        (TargetPolynomial.power ops.toOps gamma shape.freshCount)
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.sourceCount) _)
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  unfold SignedJointIdentity.gammaTerm sourceExponent
  rw [TargetPolynomial.power_add ops.toOps shiftLaws]
  exact laws.mul_assoc _ _ _

/-- The common Split-V1 gamma factor is independent of the values being
mixed. At `gamma = 0`, this makes every Split-V1 source sum vanish. -/
theorem splitV1Sum_eq_gamma_mul_paperNcSum
    {shape : SemanticShape}
    (gamma : K)
    (values : Fin shape.sourceCount → K) :
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun source =>
          SignedJointIdentity.gammaTerm ops gamma
            (sourceExponent shape .splitV1 source) (values source)) =
      K.mul gamma
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.sourceCount) (fun source =>
            SignedJointIdentity.gammaTerm ops gamma
              (sourceExponent shape .paperNc source) (values source))) := by
  change
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) _ =
      ops.mul gamma
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.sourceCount) _)
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  unfold SignedJointIdentity.gammaTerm sourceExponent
  change
    K.mul
        (TargetPolynomial.power ops.toOps gamma (source.val + 1)) _ =
      K.mul gamma
        (K.mul
          (TargetPolynomial.power ops.toOps gamma source.val) _)
  simp only [TargetPolynomial.power]
  exact laws.mul_assoc _ _ _

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
