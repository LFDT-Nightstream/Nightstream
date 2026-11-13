import NightstreamFPrime.Spec.SumCheck.FieldCausalTrace
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Challenge

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FieldCausal

open NightstreamFPrime.Spec
open SumCheck.Finite
open ConcreteCarrier StrongReduction

universe uCommitment uPublicInput

theorem sequentialRoundRepresentable {shape : Shape}
    (data : ProtocolPolynomial.Data K shape) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (width : Nat)
    (degreeCovers : data.toVerifierInput.sumcheckDegreeBound ≤ width) :
    FixedPhase.Sequential.RoundRepresentable FieldRoots.ops
      (ProtocolPolynomial.polynomial extensionOps data alpha gamma)
      width shape.cubeVariables := by
  intro fixed remaining length
  obtain ⟨polynomial, represents⟩ :=
    ProtocolPolynomialDegree.sequentialRoundRepresentable extensionOps extensionLaws
      data alpha gamma fixed remaining length
  refine ⟨FixedPolynomial.widen extensionOps.toOps degreeCovers polynomial, ?_⟩
  intro point
  change (FixedPolynomial.widen extensionOps.toOps degreeCovers polynomial).evaluate
      extensionOps.toOps point = _
  rw [FixedPolynomial.evaluate_widen extensionOps.toOps
    (ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws)]
  exact represents point

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FieldCausal
