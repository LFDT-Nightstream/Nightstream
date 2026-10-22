import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientObject
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckTruthPath
import NightstreamFPrime.Spec.SumCheck.VerifierCertificate

/-!
Finite SumCheck initial-claim binding for paper-level joint `Pi_CCS`.

Protocol: SuperNeo v1.2 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: verifier-owned initial claim before the first SumCheck round.
Constraint family: equality between the shifted target and the explicit
Boolean-hypercube sum of `Q`.

Owns: the verifier initial value, the semantic initial value, their placement
in the finite verifier-to-symbolic bridge, exact equivalence with the signed
coefficient polynomial vanishing, the canonical expected-round list derived
from the explicit joint polynomial, and deterministic false-acceptance
dichotomies exposing coefficient/table truth, a signed mixing root, or a
specific SumCheck round collision for logical and executable verification.

Does not own: the proof that the verifier enforces the exact round count, the
degree bound for canonical expected round polynomials, challenge sampling,
root-counting probability, Fiat--Shamir, Rust,
R1CS, or counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckInitial

universe uField

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

def verifierInitial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) : Field :=
  SignedJointIdentity.targetAbsolute ops data gamma

def semanticInitial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  SignedJointIdentity.summedQ ops data alpha gamma

def semanticGhosts
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (expected : List (Field -> Field)) :
    SumCheck.Finite.SemanticGhosts Field where
  trueInitial := semanticInitial ops data alpha gamma
  expected := expected

def canonicalExpected
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challenges : List Field) : List (Field -> Field) :=
  SumCheck.Finite.HypercubeTruth.expectedPolynomials ops.toOps
    (SumCheckTruthPath.jointPolynomial ops data alpha gamma) challenges

def checkJoint
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree : Nat)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (certificate : SumCheck.Finite.Certificate Field) : Bool :=
  SumCheck.Finite.check ops.toOps maxDegree
    (verifierInitial ops data gamma) roundPoint.coordinates
    (SumCheckTruthPath.verifierTerminal ops data alpha gamma
      roundPoint.coordinates)
    certificate

def symbolicInstance
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field)) :
    SumCheck.Instance Field Field :=
  SumCheck.Finite.toSymbolicInstance ops.toOps maxDegree challengeSetSize
    (verifierInitial ops data gamma) challenges terminal certificate
    (semanticGhosts ops data alpha gamma expected)

theorem claimTrue_iff_polynomial_evaluate_eq_zero
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field)) :
    SumCheck.Claim.True
        (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
          challenges terminal certificate expected) ↔
      (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
        ops.toOps gamma = ops.zero := by
  change verifierInitial ops data gamma =
      semanticInitial ops data alpha gamma ↔ _
  rw [← FiniteSumAlgebra.sub_eq_zero_iff ops laws]
  change SignedJointIdentity.paperDifference ops data alpha gamma =
      ops.zero ↔ _
  rw [SignedCoefficientPolynomial.paperDifference_eq_evaluate
    ops laws data alpha gamma]

theorem accepted_implies_polynomial_zero_or_badChallenge
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field))
    (accepted : SumCheck.Finite.Accepted ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate)
    (honestGhosts :
      (semanticGhosts ops data alpha gamma expected).Honest ops.toOps
        maxDegree challengeSetSize (verifierInitial ops data gamma)
        challenges terminal certificate) :
    (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
        ops.toOps gamma = ops.zero ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            challenges terminal certificate expected)
          round := by
  have projected :=
    SumCheck.Finite.accepted_implies_symbolicAccepted_and_truthPath
      ops.toOps maxDegree challengeSetSize (verifierInitial ops data gamma)
      challenges terminal certificate
      (semanticGhosts ops data alpha gamma expected) accepted honestGhosts
  by_cases polynomialZero :
      (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
        ops.toOps gamma = ops.zero
  · exact Or.inl polynomialZero
  · apply Or.inr
    apply SumCheck.false_acceptance_implies_bad_challenge
      ops.toOps.toSymbolic
      (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
        challenges terminal certificate expected)
      projected.1 projected.2
    intro claimTrue
    exact polynomialZero <|
      (claimTrue_iff_polynomial_evaluate_eq_zero
        ops laws data alpha gamma maxDegree challengeSetSize challenges
        terminal certificate expected).1 claimTrue

theorem accepted_implies_coefficientTruth_or_mixingRoot_or_badChallenge
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field))
    (accepted : SumCheck.Finite.Accepted ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate)
    (honestGhosts :
      (semanticGhosts ops data alpha gamma expected).Honest ops.toOps
        maxDegree challengeSetSize (verifierInitial ops data gamma)
        challenges terminal certificate) :
    SignedCoefficientObject.CoefficientTruth ops data ∨
      SignedCoefficientObject.MixingRoot ops data alpha gamma ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            challenges terminal certificate expected)
          round := by
  rcases accepted_implies_polynomial_zero_or_badChallenge
      ops laws data alpha gamma maxDegree challengeSetSize challenges terminal
      certificate expected accepted honestGhosts with polynomialZero | badRound
  · rcases (SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot
          ops laws data alpha gamma).1 polynomialZero with truth | mixingRoot
    · exact Or.inl truth
    · exact Or.inr (Or.inl mixingRoot)
  · exact Or.inr (Or.inr badRound)

theorem checked_implies_polynomial_zero_or_badChallenge
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field))
    (checked : SumCheck.Finite.check ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate = true)
    (honestGhosts :
      (semanticGhosts ops data alpha gamma expected).Honest ops.toOps
        maxDegree challengeSetSize (verifierInitial ops data gamma)
        challenges terminal certificate) :
    (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
        ops.toOps gamma = ops.zero ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            challenges terminal certificate expected)
          round := by
  apply accepted_implies_polynomial_zero_or_badChallenge
    ops laws data alpha gamma maxDegree challengeSetSize challenges terminal
    certificate expected
  · exact (SumCheck.Finite.check_eq_true_iff_accepted ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate).1 checked
  · exact honestGhosts

theorem checked_implies_coefficientTruth_or_mixingRoot_or_badChallenge
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field))
    (checked : SumCheck.Finite.check ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate = true)
    (honestGhosts :
      (semanticGhosts ops data alpha gamma expected).Honest ops.toOps
        maxDegree challengeSetSize (verifierInitial ops data gamma)
        challenges terminal certificate) :
    SignedCoefficientObject.CoefficientTruth ops data ∨
      SignedCoefficientObject.MixingRoot ops data alpha gamma ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            challenges terminal certificate expected)
          round := by
  apply accepted_implies_coefficientTruth_or_mixingRoot_or_badChallenge
    ops laws data alpha gamma maxDegree challengeSetSize challenges terminal
    certificate expected
  · exact (SumCheck.Finite.check_eq_true_iff_accepted ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate).1 checked
  · exact honestGhosts

theorem checked_implies_tableObligations_or_mixingRoot_or_badChallenge
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (terminal : Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (expected : List (Field -> Field))
    (checked : SumCheck.Finite.check ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges terminal certificate = true)
    (honestGhosts :
      (semanticGhosts ops data alpha gamma expected).Honest ops.toOps
        maxDegree challengeSetSize (verifierInitial ops data gamma)
        challenges terminal certificate) :
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops data)).AllHold ∨
      SignedCoefficientObject.MixingRoot ops data alpha gamma ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            challenges terminal certificate expected)
          round := by
  rcases checked_implies_coefficientTruth_or_mixingRoot_or_badChallenge
      ops laws data alpha gamma maxDegree challengeSetSize challenges terminal
      certificate expected checked honestGhosts with truth | bad
  · exact Or.inl <|
      (SignedCoefficientObject.coefficientTruth_iff_tableObligations
        ops zeroLaws data).1 truth
  · exact Or.inr bad

theorem checkedCanonical_implies_tableObligations_or_mixingRoot_or_badChallenge
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (challenges : List Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (challengeLength : challenges.length = shape.cubeVariables)
    (checked : SumCheck.Finite.check ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges
      (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
      certificate = true) :
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops data)).AllHold ∨
      SignedCoefficientObject.MixingRoot ops data alpha gamma ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            challenges
            (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
            certificate
            (canonicalExpected ops data alpha gamma challenges))
          round := by
  have accepted : SumCheck.Finite.Accepted ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges
      (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
      certificate :=
    (SumCheck.Finite.check_eq_true_iff_accepted ops.toOps maxDegree
      (verifierInitial ops data gamma) challenges
      (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
      certificate).1 checked
  have sameLength : certificate.rounds.length = challenges.length :=
    SumCheck.Finite.Chain.messages_length_eq_challenges_length ops.toOps
      maxDegree (verifierInitial ops data gamma)
      (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
      certificate.rounds challenges accepted
  have honestGhosts :
      (semanticGhosts ops data alpha gamma
          (canonicalExpected ops data alpha gamma challenges)).Honest ops.toOps
        maxDegree challengeSetSize (verifierInitial ops data gamma)
        challenges
        (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
        certificate := by
    simpa only [semanticGhosts, semanticInitial, canonicalExpected,
        SumCheckTruthPath.canonicalGhosts]
      using SumCheckTruthPath.canonicalGhosts_honest ops laws data alpha gamma
        maxDegree challengeSetSize (verifierInitial ops data gamma) challenges
        certificate challengeLength sameLength
  exact checked_implies_tableObligations_or_mixingRoot_or_badChallenge
    ops laws zeroLaws data alpha gamma maxDegree challengeSetSize challenges
    (SumCheckTruthPath.verifierTerminal ops data alpha gamma challenges)
    certificate (canonicalExpected ops data alpha gamma challenges) checked
    honestGhosts

theorem checkJoint_implies_tableObligations_or_mixingRoot_or_badChallenge
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (certificate : SumCheck.Finite.Certificate Field)
    (checked : checkJoint ops data alpha gamma maxDegree roundPoint
      certificate = true) :
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops data)).AllHold ∨
      SignedCoefficientObject.MixingRoot ops data alpha gamma ∨
      ∃ round,
        SumCheck.BadChallenge
          (symbolicInstance ops data alpha gamma maxDegree challengeSetSize
            roundPoint.coordinates
            (SumCheckTruthPath.verifierTerminal ops data alpha gamma
              roundPoint.coordinates)
            certificate
            (canonicalExpected ops data alpha gamma roundPoint.coordinates))
          round := by
  apply checkedCanonical_implies_tableObligations_or_mixingRoot_or_badChallenge
    ops laws zeroLaws data alpha gamma maxDegree challengeSetSize
    roundPoint.coordinates certificate roundPoint.dimension
  exact checked

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckInitial
