import NightstreamFPrime.Spec.SumCheck.GoldilocksCausalTrace
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
The paper's exact fixed-width PiCCS failure event on a causal message path.
Source assignments are fixed before the independent second execution. After
alpha and gamma are drawn, its semantic q is fixed before all round challenges.
An output witness can depend on the complete second execution; agreement with
the first witness is an event, not a condition on the challenge distribution.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal

open NightstreamFPrime.Spec
open SumCheck.Finite
open ConcreteCarrier StrongReduction
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Strategy)
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)

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
