import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree
import NightstreamFPrime.Spec.SumCheck.FixedPhase.RawCertificate

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ProtocolPolynomialFixedWidth.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Paper-polynomial verification over one verifier-owned SumCheck width.

Source: SuperNeo Definition 6, Section 7.3, and Appendix D.4.  The paper bounds
each round polynomial's degree but does not impose a canonical
variable-length coefficient serialization.

Owns: the fixed-width raw-message PiCCS checker and its deterministic
reduction to residual truth, the alpha/gamma mixing root, one fixed-width
SumCheck bad challenge, or the existing output-message mismatch.

Does not own: challenge distribution, root probabilities, Fiat--Shamir,
transcript serialization, Poseidon2, Ajtai, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.FixedWidth

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField

/-- The paper SumCheck gate over raw transport messages.  Width mismatch is a
verifier rejection, while high zero slots at the selected width are valid. -/
def check
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (width : Nat)
    (input : VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape)
    (certificate : SumCheck.Finite.Certificate Field) : Bool :=
  SumCheck.Finite.FixedPhase.RawCertificate.check ops.toOps width
    (input.initial ops gamma)
    roundPoint.coordinates
    (terminalFromMessage ops input alpha gamma roundPoint message)
    certificate

/-- Exact executable exposure of the decoded certificate and fixed-width
claimed chain. -/
theorem check_eq_true_iff
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (width : Nat)
    (input : VerifierInput Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape)
    (certificate : SumCheck.Finite.Certificate Field) :
    check ops width input alpha gamma roundPoint message certificate = true <->
      exists fixed : SumCheck.Finite.FixedPhase.Certificate Field width,
        SumCheck.Finite.FixedPhase.RawCertificate.decode width certificate =
            some fixed /\
          SumCheck.Finite.FixedPhase.Chain ops.toOps
            (input.initial ops gamma) fixed.rounds roundPoint.coordinates
            (terminalFromMessage ops input alpha gamma roundPoint message) := by
  exact SumCheck.Finite.FixedPhase.RawCertificate.check_eq_true_iff
    ops.toOps width (input.initial ops gamma) roundPoint.coordinates
    (terminalFromMessage ops input alpha gamma roundPoint message) certificate

/-- The exact fixed-width bad-challenge event selected by one decoded
certificate. -/
def SumCheckCollision
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (width challengeSetSize : Nat)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (certificate : SumCheck.Finite.FixedPhase.Certificate Field width) :
    Prop :=
  exists round,
    SumCheck.Finite.FixedPhase.BadChallenge ops.toOps
      (polynomial ops data alpha gamma)
      width challengeSetSize
      (data.toVerifierInput.initial ops gamma)
      roundPoint.coordinates certificate round

/-- Deterministic fixed-width PiCCS soundness boundary.

The selected width may exceed the exact syntactic degree, but that inclusion
is verifier-owned and explicit.  Acceptance yields no encoding escape:
width-invalid raw messages have already been rejected before this theorem is
called. -/
theorem accepted_implies_tableTruth_or_badEvent
    {Field : Type uField}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (width : Nat)
    (degreeCovers : data.toVerifierInput.sumcheckDegreeBound <= width)
    (challengeSetSize : Nat)
    (roundPoint : CubePoint Field shape.cubeVariables)
    (message : OutputMessage Field shape)
    (certificate : SumCheck.Finite.FixedPhase.Certificate Field width)
    (chain : SumCheck.Finite.FixedPhase.Chain ops.toOps
      (data.toVerifierInput.initial ops gamma) certificate.rounds
      roundPoint.coordinates
      (terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint
        message)) :
    (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold \/
      SignedCoefficientObject.MixingRoot ops (data.toJointData ops)
        alpha gamma \/
      SumCheckCollision ops data alpha gamma width challengeSetSize roundPoint
        certificate \/
      OutputMismatch ops data alpha gamma roundPoint message := by
  let q := polynomial ops data alpha gamma
  by_cases outputMatches :
      terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint
          message =
        qAtPoint ops data alpha gamma roundPoint
  · have qAtPointExact :
        qAtPoint ops data alpha gamma roundPoint =
          q roundPoint.coordinates := by
      unfold q polynomial
      rw [dif_pos roundPoint.dimension]
    have terminalExact :
        terminalFromMessage ops data.toVerifierInput alpha gamma roundPoint
            message =
          q roundPoint.coordinates :=
      outputMatches.trans qAtPointExact
    have accepted :
        SumCheck.Finite.FixedPhase.Accepted ops.toOps q
          (data.toVerifierInput.initial ops gamma)
          roundPoint.coordinates certificate := by
      unfold SumCheck.Finite.FixedPhase.Accepted
      rw [← terminalExact]
      exact chain
    have expectedAtExactDegree :=
      ProtocolPolynomialDegree.expectedRoundsRepresentable ops laws data
        alpha gamma roundPoint
    have expectedAtWidth :
        SumCheck.Finite.FixedPhase.ExpectedRoundsRepresentable ops.toOps q
          width roundPoint.coordinates := by
      intro expected expectedIn
      rcases expectedAtExactDegree expected (by simpa [q] using expectedIn) with
        ⟨polynomialAtDegree, represents⟩
      refine ⟨SumCheck.Finite.FixedPolynomial.widen ops.toOps degreeCovers
        polynomialAtDegree, ?_⟩
      intro point
      rw [SumCheck.Finite.FixedPolynomial.evaluate_widen ops.toOps
        (ProtocolPolynomialDegree.Support.polynomialLaws laws)]
      exact represents point
    let jointData := data.toJointData ops
    by_cases tableTruth :
        (TableResidualData.toTableObligations ops
          (SignedCoefficientObject.toTableResidualData ops jointData)).AllHold
    · exact Or.inl tableTruth
    · by_cases mixingRoot :
          SignedCoefficientObject.MixingRoot ops jointData alpha gamma
      · exact Or.inr (Or.inl mixingRoot)
      · have falseInitial :
            data.toVerifierInput.initial ops gamma ≠
              SumCheck.Finite.FixedPhase.semanticInitial ops.toOps q
                roundPoint.coordinates.length := by
          intro initialTrue
          have jointInitialTrue :
              data.toVerifierInput.initial ops gamma =
                SumCheckInitial.semanticInitial ops jointData alpha gamma := by
            calc
              _ = SumCheck.Finite.FixedPhase.semanticInitial ops.toOps q
                    roundPoint.coordinates.length := initialTrue
              _ = _ := by
                rw [roundPoint.dimension]
                unfold SumCheck.Finite.FixedPhase.semanticInitial q jointData
                rw [sumCompletions_polynomial_eq_summedQ ops laws data
                  alpha gamma]
                rfl
          have polynomialZero :
              (SignedCoefficientPolynomial.polynomial ops jointData alpha).evaluate
                ops.toOps gamma = ops.zero := by
            apply (SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero
              ops laws jointData alpha gamma width challengeSetSize
              roundPoint.coordinates (q roundPoint.coordinates)
              (SumCheck.Finite.FixedPhase.RawCertificate.encode certificate)
              []).1
            simpa [SumCheck.Claim.True, SumCheckInitial.symbolicInstance] using
              jointInitialTrue
          rcases
              (SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot
                ops laws jointData alpha gamma).1 polynomialZero with
            coefficientTruth | root
          · exact tableTruth
              ((SignedCoefficientObject.coefficientTruth_iff_tableObligations
                ops zeroLaws jointData).1 coefficientTruth)
          · exact mixingRoot root
        have collision :=
          SumCheck.Finite.FixedPhase.false_acceptance_implies_bad_challenge
            ops.toOps q challengeSetSize
            (data.toVerifierInput.initial ops gamma)
            roundPoint.coordinates certificate expectedAtWidth accepted
            falseInitial
        exact Or.inr (Or.inr (Or.inl collision))
  · exact Or.inr (Or.inr (Or.inr outputMatches))

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.FixedWidth
