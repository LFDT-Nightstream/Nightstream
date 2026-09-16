import NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial
import NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.ProductionRelation
import NightstreamFPrime.Lifecycle.Types

/-!
One fresh CCS contribution from two retained extension-field rows. The
original production shape and ten coefficient slots are retained. Prepared
selector/power equality is proved here; origin of the retained rows is not.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixPolynomial

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle

/-- Only freshMatrixImage is consumed by the CCS constructor. The selected
profile has one fresh source; the other message fields are unused zeros. -/
def message (row : Vector K ProductionRelation.matrixCount) :
    ProtocolPolynomial.OutputMessage K productionShape where
  freshMatrixImage := fun _ matrix => row.get matrix
  sourceAssignment := fun _ => K.zero
  padImage := fun _ => K.zero
  matrixImage := fun _ => K.zero

/-- The original fresh constructor and its single outer gamma shift. -/
def contribution (input : ProtocolPolynomial.VerifierInput K productionShape)
    (alpha : CubePoint K productionShape.cubeVariables) (challenges : List K)
    {remaining : Nat} (suffix : BooleanVertex remaining) (powers : Nat → K)
    (tables : Array K × Array K)
    (low high : Vector K ProductionRelation.matrixCount) :
    FixedPolynomial K input.sumcheckDegreeBound :=
  FixedPolynomial.scale extensionOps.toOps (powers productionShape.constraintOffset)
    (FixedPolynomial.widen extensionOps.toOps (Nat.le_max_left _ _)
      (PiCCSFreshPolynomial.ccsPolynomialWithPowers extensionOps input powers
        (PiCCSPrefixSelector.cachedSelector extensionOps challenges suffix alpha tables)
        (message low) (message high)))

/-- Prepared lookups preserve every coefficient of the original CCS constructor. -/
theorem contribution_prepared (input : ProtocolPolynomial.VerifierInput K productionShape)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K) (challenges : List K)
    {remaining : Nat} (suffix : BooleanVertex remaining)
    (dimension : productionShape.cubeVariables = challenges.length + remaining + 1)
    (low high : Vector K ProductionRelation.matrixCount) :
    contribution input alpha challenges suffix
        (PiCCSGammaPowers.lookup extensionOps.toOps gamma
          (PiCCSGammaPowers.prepare extensionOps.toOps gamma
            (PiCCSFirstRoundPair.powerCount productionShape)))
        (PiCCSTensorWeights.prepare extensionOps
          (PiCCSPrefixSelector.dropPoint alpha challenges.length).coordinates.tail) low high =
      FixedPolynomial.scale extensionOps.toOps
        (TargetPolynomial.power extensionOps.toOps gamma productionShape.constraintOffset)
        (FixedPolynomial.widen extensionOps.toOps (Nat.le_max_left _ _)
          (PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input
            (TargetPolynomial.power extensionOps.toOps gamma)
            (PiCCSPrefixSelector.selector extensionOps challenges suffix alpha)
            (message low) (message high))) := by
  have powers : PiCCSGammaPowers.lookup extensionOps.toOps gamma
      (PiCCSGammaPowers.prepare extensionOps.toOps gamma
        (PiCCSFirstRoundPair.powerCount productionShape)) =
        TargetPolynomial.power extensionOps.toOps gamma := by
    funext exponent
    exact PiCCSGammaPowers.lookup_prepare _ _ _ exponent
  unfold contribution
  rw [powers,
    PiCCSPrefixSelector.cachedSelector_prepare extensionOps extensionLaws
      challenges suffix alpha dimension,
    PiCCSFreshPolynomial.ccsPolynomialWithPowers_value extensionOps extensionLaws]
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixPolynomial
