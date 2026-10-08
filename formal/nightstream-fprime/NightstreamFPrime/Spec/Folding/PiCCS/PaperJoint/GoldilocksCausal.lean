import NightstreamFPrime.Spec.SumCheck.GoldilocksCausalTrace
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
Owns the round representability of the paper PiCCS polynomial at the
verifier's selected width: after `α` and `γ`, every round polynomial of every
prefix fits in that width. `RoundByRound` consumes it.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal

open NightstreamFPrime.Spec
open SumCheck.Finite
open ConcreteCarrier StrongReduction

universe uCommitment uPublicInput

/-- The paper polynomial has the verifier's selected width at every prefix.
The representation comes from its concrete CCS, norm, and evaluation terms. -/
theorem sequentialRoundRepresentable {shape : Shape}
    (data : ProtocolPolynomial.Data K shape) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (width : Nat)
    (degreeCovers : data.toVerifierInput.sumcheckDegreeBound ≤ width) :
    FixedPhase.Sequential.RoundRepresentable GoldilocksRoots.ops
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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
