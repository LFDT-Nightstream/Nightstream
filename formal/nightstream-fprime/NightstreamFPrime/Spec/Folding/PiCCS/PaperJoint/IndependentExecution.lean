import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability

/-!
SuperNeo v1.1 Appendix B.2, equation (16). Fix the first execution's witness
and one second prover's private tape. The second alpha, gamma, and round
challenges are independent uniform K samples. Success and witness agreement
are tested inside that unchanged distribution, including every aborted run.

The bound holds for every fixed private tape. Averaging arbitrary prover and
setup randomness, and the final extraction-error bound, are separate steps.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution

open scoped BigOperators
open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open ConcreteCarrier StrongReduction CausalExecution
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal
  (uniformAverage uniformAverage_const uniformAverage_congr uniformAverage_mono
    uniformAverage_add sampleSpace_nonempty collisionProbability)

attribute [local instance] Classical.propDecidable

universe uCommitment uPublicInput

/-- The second actual response succeeds in the relaxed target and returns
the same source assignments as the fixed first execution. -/
def AgreedSuccess {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (width : Nat) (firstWitness : OutputWitness shape columns)
    (result : Option (Probe K shape × OutputWitness shape columns)) : Prop :=
  ∃ probe witness, result = some (probe, witness) ∧
    probe.FixedWidthAccepted extensionOps K.embed statement width ∧
    AmbientOutputHolds extensionOps K.embed openingMaps params statement probe witness ∧
    witness = firstWitness

/-- After alpha/gamma are fixed, all fresh round challenge streams are retained. -/
noncomputable def roundAgreementProbability
    {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (firstWitness : OutputWitness shape columns) (prover : Prover shape columns width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) : ℝ :=
  uniformAverage GoldilocksRoots.fullChallengeSet shape.cubeVariables (fun coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      if AgreedSuccess openingMaps params statement width firstWitness
        (run prover alpha gamma ⟨coordinates, dimension⟩) then 1 else 0
    else 0)

/-- Complete independent verifier sampling for the second execution. -/
noncomputable def agreementProbability
    {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (firstWitness : OutputWitness shape columns) (prover : Prover shape columns width) : ℝ :=
  uniformAverage GoldilocksRoots.fullChallengeSet shape.cubeVariables (fun coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      𝔼 gamma ∈ GoldilocksRoots.fullChallengeSet,
        roundAgreementProbability openingMaps params statement firstWitness prover
          ⟨coordinates, dimension⟩ gamma
    else 0)

/-- The two interactive algebraic losses in the paper's test bound. -/
noncomputable def testError (shape : Shape) (width : Nat) : ℝ :=
  (shape.cubeVariables : ℝ) * width / (goldilocksModulus ^ 2 : Nat) +
    ((shape.jointCoefficientCount - 1 + shape.cubeVariables : Nat) : ℝ) /
      (goldilocksModulus ^ 2 : Nat)

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {shape : Shape} {columns blockCount width : Nat}
  (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
  (freshBound : params.b = 2)
  (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
  (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
  (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
  (firstWitness : OutputWitness shape columns)
  (sourceInvalid : ¬ SourceHolds extensionOps K.embed openingMaps params statement firstWitness)
  (prover : Prover shape columns width)

include freshBound constantLaw degreeCovers sourceInvalid

/-- Per alpha/gamma slice, success with agreement is covered by the fixed
mixing event or the causal round collision bound. -/
theorem roundAgreementProbability_le (alpha : CubePoint K shape.cubeVariables) (gamma : K) :
    roundAgreementProbability openingMaps params statement firstWitness prover alpha gamma ≤
      (if SignedCoefficientObject.MixingRoot extensionOps
        ((statement.sourceProtocolData K.embed firstWitness).toJointData extensionOps)
        alpha gamma then (1 : ℝ) else 0) +
      (shape.cubeVariables : ℝ) * width / (goldilocksModulus ^ 2 : Nat) := by
  let data := statement.sourceProtocolData K.embed firstWitness
  by_cases mixing : SignedCoefficientObject.MixingRoot extensionOps
      (data.toJointData extensionOps) alpha gamma
  · have atMostOne :
        roundAgreementProbability openingMaps params statement firstWitness prover alpha gamma ≤ 1 := by
      calc
        _ ≤ uniformAverage GoldilocksRoots.fullChallengeSet shape.cubeVariables (fun _ => 1) := by
          apply uniformAverage_mono
          intro coordinates dimension
          simp only [dif_pos dimension]
          split_ifs <;> norm_num
        _ = 1 := uniformAverage_const _ sampleSpace_nonempty _ _
    rw [if_pos mixing]
    have nonnegative : (0 : ℝ) ≤
        (shape.cubeVariables : ℝ) * width / (goldilocksModulus ^ 2 : Nat) := by positivity
    linarith
  · rw [if_neg mixing, zero_add]
    have inclusion :
        roundAgreementProbability openingMaps params statement firstWitness prover alpha gamma ≤
        collisionProbability GoldilocksRoots.fullChallengeSet
          (ProtocolPolynomial.polynomial extensionOps data alpha gamma)
          (prover.rounds alpha gamma) [] shape.cubeVariables := by
      apply uniformAverage_mono
      intro coordinates dimension
      simp only [dif_pos dimension]
      by_cases event : AgreedSuccess openingMaps params statement width firstWitness
          (run prover alpha gamma ⟨coordinates, dimension⟩)
      · have success := event
        obtain ⟨probe, witness, returned, accepted, ambient, agreement⟩ := event
        have receipt := run_implies_receipt prover alpha gamma ⟨coordinates, dimension⟩
          probe witness returned
        have alternatives := GoldilocksCausal.agreed_accepted_implies_mixing_or_collision
          openingMaps params freshBound statement constantLaw degreeCovers firstWitness sourceInvalid
          (prover.rounds alpha gamma) probe witness agreement receipt.2.2.2 ambient accepted
        simp only [receipt.1, receipt.2.1, receipt.2.2.1] at alternatives
        have collision :
            _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal.Collision
              (ProtocolPolynomial.polynomial extensionOps data alpha gamma)
              (prover.rounds alpha gamma) [] coordinates := by
          exact alternatives.resolve_left mixing
        simp only [if_pos success, if_pos collision, le_refl]
      · rw [if_neg event]
        split_ifs <;> norm_num
    apply inclusion.trans
    have dataCovers : data.toVerifierInput.sumcheckDegreeBound ≤ width := degreeCovers
    exact _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal.collisionProbability_le
      (ProtocolPolynomial.polynomial extensionOps data alpha gamma)
      (prover.rounds alpha gamma)
      (GoldilocksCausal.sequentialRoundRepresentable data alpha gamma width dataCovers)
      [] shape.cubeVariables (by simp)

/-- B.2 equation (16), on the unconditioned second execution's verifier coins.
This bound is uniform in the prover's fixed private tape. -/
theorem agreementProbability_le :
    agreementProbability openingMaps params statement firstWitness prover ≤ testError shape width := by
  let data := (statement.sourceProtocolData K.embed firstWitness).toJointData extensionOps
  let sumcheckError := (shape.cubeVariables : ℝ) * width / (goldilocksModulus ^ 2 : Nat)
  have each (alpha : CubePoint K shape.cubeVariables) :
      (𝔼 gamma ∈ GoldilocksRoots.fullChallengeSet,
        roundAgreementProbability openingMaps params statement firstWitness prover alpha gamma) ≤
      (𝔼 gamma ∈ GoldilocksRoots.fullChallengeSet,
        if SignedCoefficientObject.MixingRoot extensionOps data alpha gamma then (1 : ℝ) else 0) +
          sumcheckError := by
    calc
      _ ≤ (𝔼 gamma ∈ GoldilocksRoots.fullChallengeSet,
          ((if SignedCoefficientObject.MixingRoot extensionOps data alpha gamma then (1 : ℝ) else 0) +
            sumcheckError)) :=
        Finset.expect_le_expect fun gamma _ =>
          roundAgreementProbability_le openingMaps params freshBound statement constantLaw degreeCovers
            firstWitness sourceInvalid prover alpha gamma
      _ = _ := by
        rw [Finset.expect_add_distrib, Finset.expect_const sampleSpace_nonempty]
  have averaged :
      agreementProbability openingMaps params statement firstWitness prover ≤
      uniformAverage GoldilocksRoots.fullChallengeSet shape.cubeVariables (fun coordinates =>
        (if dimension : coordinates.length = shape.cubeVariables then
          𝔼 gamma ∈ GoldilocksRoots.fullChallengeSet,
            if SignedCoefficientObject.MixingRoot extensionOps data ⟨coordinates, dimension⟩ gamma
              then (1 : ℝ) else 0
          else 0) + sumcheckError) := by
    apply uniformAverage_mono
    intro coordinates dimension
    simpa only [dif_pos dimension] using each ⟨coordinates, dimension⟩
  apply averaged.trans
  rw [uniformAverage_add, uniformAverage_const _ sampleSpace_nonempty]
  have mixingBound := SignedMixingProbability.mixingProbability_le
    GoldilocksRoots.fullChallengeSet sampleSpace_nonempty data
  rw [GoldilocksRoots.fullChallengeSet_card] at mixingBound
  have combined := _root_.add_le_add_right mixingBound sumcheckError
  simpa only [SignedMixingProbability.mixingProbability, testError, sumcheckError, add_comm]
    using combined

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution
