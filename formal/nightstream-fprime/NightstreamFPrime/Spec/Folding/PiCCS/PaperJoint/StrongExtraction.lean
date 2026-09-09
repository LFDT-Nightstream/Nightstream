import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.OneRunExtraction

/-!
SuperNeo B.2's probability and expected-time conclusions for the same checked
one-call extractor. Setup contexts and private tapes keep their stated PMFs;
only interactive verifier coins have the proved uniform law. The returned
source relation is the existing CCS/CE product through the concrete Phi81
public prefix. Fiat–Shamir transfer and setup hardness are separate claims.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier WitnessProjection CheckedWitnessExtraction

variable {Context Tape Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
  {blockCount width : Nat}
  (contexts : PMF Context) (tapes : Context → PMF Tape)
  (call : Context → OneRunExtraction.Call Tape shape carrier)
  (program : Context → Program shape carrier)
  (prover : Context → Tape → CausalExecution.Prover shape carrier.carrierWidth width)
  (commit : Context → Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
  (statement : Context → Statement K Commitment (Phi81Relation.PublicInput carrier)
    shape carrier.carrierWidth blockCount baseOps)

attribute [local instance] Classical.propDecidable

/-- Actual valid returns of the implemented extractor under the context law. -/
noncomputable def successProbability : ℝ :=
  StrongProbability.clockMean (StrongProbability.jointTapeLaw contexts tapes)
    fun sample alpha gamma point =>
      if SourceReturned (commit sample.1) params (statement sample.1)
        (OneRunExtraction.run (call sample.1) (program sample.1) sample.2 alpha gamma point).value
      then 1 else 0

/-- The same joint experiment's actual call and checker work. -/
def baseClock (sample : Context × Tape) :
    CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ :=
  OneRunExtraction.baseClock (call sample.1) (program sample.1) sample.2

/-- The same joint experiment's total work, including projection and exits. -/
def totalClock (sample : Context × Tape) :
    CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ :=
  OneRunExtraction.totalClock (call sample.1) (program sample.1) sample.2

variable (callCorrect : ∀ context, OneRunExtraction.CallCorrect (call context) (prover context))
  (correct : ∀ context, CheckedWitnessExtraction.Correct (width := width)
    (program context) (commit context) params (statement context))

include callCorrect correct in
/-- Global probability accounting measures these exact returned witness values. -/
theorem successProbability_eq :
    successProbability contexts tapes call program commit params statement =
      StrongProbability.globalSourceProbability contexts tapes prover
        (fun context => openingMaps (commit context)) params statement := by
  let clock := fun (sample : Context × Tape) alpha gamma point =>
    if SourceReturned (commit sample.1) params (statement sample.1)
      (OneRunExtraction.run (call sample.1) (program sample.1) sample.2 alpha gamma point).value
    then (1 : ℝ) else 0
  have averaged := (StrongProbability.joint_clockMean_eq_nested contexts tapes clock (by
    intro sample alpha gamma point
    dsimp only [clock]
    split_ifs <;> norm_num)).2
  change StrongProbability.clockMean (StrongProbability.jointTapeLaw contexts tapes) clock = _
  rw [averaged]
  change (∑' context, (contexts context).toReal *
    OneRunExtraction.successProbability (tapes context) (call context) (program context)
      (commit context) params (statement context)) =
    ∑' context, (contexts context).toReal * StrongProbability.sourceProbability
      (tapes context) (prover context) (openingMaps (commit context)) params (statement context)
  apply tsum_congr
  intro context
  rw [OneRunExtraction.successProbability_eq (tapes context) (call context) (program context)
    (prover context) (callCorrect context) (commit context) params (statement context) (correct context)]

include callCorrect correct in
/-- One bound states both the paper success loss and expected polynomial work
for the same program. Primitive correctness and their actual polynomial work
bounds are the explicit paper EPT/PPT premises. No source-success premise is
assumed, and the fresh public-prefix condition is already discharged. -/
theorem probability_and_expected_work
    (freshBound : params.b = 2)
    (constantLaw : ∀ context,
      MatrixCoefficientSource.ConstantTermLaw baseOps (statement context).matrixSource.kernel)
    (degreeCovers : ∀ context,
      ((statement context).verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (accessBound : Nat)
    (bounded : ∀ context, CostedWitnessProjection.Bounded (program context).access accessBound)
    (baseSummable : Summable fun sample =>
      (StrongProbability.jointTapeLaw contexts tapes sample).toReal *
        StrongProbability.verifierMean (baseClock call program sample))
    (securityParameter : Nat) (callPolynomial accessPolynomial : Polynomial ℝ)
    (callPPT : StrongProbability.clockMean (StrongProbability.jointTapeLaw contexts tapes)
      (baseClock call program) ≤ callPolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤
      accessPolynomial.eval (securityParameter : ℝ)) :
    (StrongProbability.globalSuccessProbability contexts tapes prover
        (fun context => openingMaps (commit context)) params statement -
      Real.sqrt (StrongProbability.globalDisagreementProbability contexts tapes prover
        (fun context => openingMaps (commit context)) params statement +
          IndependentExecution.testError shape width) ≤
      successProbability contexts tapes call program commit params statement) ∧
    (Summable (fun sample => (StrongProbability.jointTapeLaw contexts tapes sample).toReal *
      StrongProbability.verifierMean (totalClock call program sample)) ∧
      StrongProbability.clockMean (StrongProbability.jointTapeLaw contexts tapes)
        (totalClock call program) ≤
        (callPolynomial +
          Polynomial.C (shape.freshCount : ℝ) *
            (Polynomial.C (privateWidth carrier : ℝ) * (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
          Polynomial.C (shape.runningCount : ℝ) *
            (Polynomial.C (carrier.carrierWidth : ℝ) * (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
          Polynomial.C 6).eval (securityParameter : ℝ)) := by
  constructor
  · rw [successProbability_eq contexts tapes call program prover commit params statement callCorrect correct]
    exact StrongProbability.source_success_ge contexts tapes prover
      (fun context => openingMaps (commit context)) params statement freshBound constantLaw degreeCovers
  · have actual := StrongProbability.clockMean_le_add_const
      (StrongProbability.jointTapeLaw contexts tapes) (baseClock call program) (totalClock call program)
      ((CostedWitnessProjection.workBound shape carrier accessBound + 3 : Nat) : ℝ)
      (fun _ _ _ _ => Nat.cast_nonneg _) baseSummable
      (fun sample => OneRunExtraction.run_work_le (call sample.1) (program sample.1)
        accessBound (bounded sample.1) sample.2)
    refine ⟨actual.1, ?_⟩
    have fresh := mul_le_mul_of_nonneg_left accessPPT
      (by positivity : (0 : ℝ) ≤ (shape.freshCount : ℝ) * (privateWidth carrier : ℝ))
    have running := mul_le_mul_of_nonneg_left accessPPT
      (by positivity : (0 : ℝ) ≤ (shape.runningCount : ℝ) * (carrier.carrierWidth : ℝ))
    have bound := actual.2
    unfold CostedWitnessProjection.workBound at bound
    push_cast at bound
    simp only [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C]
    nlinarith

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongExtraction
