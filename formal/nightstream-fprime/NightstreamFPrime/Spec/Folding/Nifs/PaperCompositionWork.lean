import NightstreamFPrime.Spec.Folding.Nifs.InteractivePrefix
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakLaw
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateExtraction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

/-!
Work of the sequential B.1 extractor. The prefix returns its captured state
and clock together. The same state selects the clocked weak oracle and its
endpoint law; only that selected suffix executes. Its literal returned list
is decoded and checked by the existing one-run PiCCS projection program.

The global moment premise is on these actual call, decode, and check costs.
Individual contexts and calls need no uniform time bound. The finite endpoint
law describes returned values; its construction is not charged as a runtime
table sampler. Retry work retains the existing abort-inclusive clock law.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionWork

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint
open StrongReduction
open PiRLC.PaperForkExtraction
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result Primitives PrimitiveBounds)
open PiRLC.CoordinateChargedOracle PiRLC.CoordinateCheckedCalls
open PiRLC.CoordinateRetryWork PiRLC.CoordinateOracleCost
open PiRLC.CoordinateOracle PiRLC.CoordinateRetry

variable {Assignment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}

/-- Decode this actual weak return, then run the existing checked PiCCS
projection. No source witness is selected by proof choice. -/
def finish (program : CheckedWitnessExtraction.Program shape carrier)
    (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))
    (probe : Probe K shape) (values : Option (List Assignment)) :
    Result (Option (WitnessProjection.SourceWitness shape carrier)) :=
  let decoded := decode values
  let checked := CheckedWitnessExtraction.finish program
    (decoded.value.map fun witness => (probe, witness))
  ⟨checked.value, decoded.work + checked.work + 1⟩

/-- The same decode and checker invocations supply the pre-projection clock. -/
def finishBaseWork (program : CheckedWitnessExtraction.Program shape carrier)
    (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))
    (probe : Probe K shape) (values : Option (List Assignment)) : Nat :=
  let decoded := decode values
  decoded.work + CheckedWitnessExtraction.checkerWork program
    (decoded.value.map fun witness => (probe, witness))

theorem finish_work_le (program : CheckedWitnessExtraction.Program shape carrier)
    (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))
    (accessBound : Nat) (bounded : CostedWitnessProjection.Bounded program.access accessBound)
    (probe : Probe K shape) (values : Option (List Assignment)) :
    (finish program decode probe values).work ≤ finishBaseWork program decode probe values +
      CostedWitnessProjection.workBound shape carrier accessBound + 3 := by
  have bound := CheckedWitnessExtraction.finish_work_le program accessBound bounded
    ((decode values).value.map fun witness => (probe, witness))
  dsimp only [finish, finishBaseWork]
  omega

variable {Structure PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)
  [DecidableEq Scalar] [Fintype (PiRLC.CoordinateForkLaw.Challenge algebra)]
  [Nonempty (PiRLC.CoordinateForkLaw.Challenge algebra)] [Fintype Assignment]

omit [DecidableEq Scalar] in
private theorem query_work_nonnegative
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (check : (Fin arity.total → PiRLC.CoordinateForkLaw.Challenge algebra) → Assignment → Bool) :
    0 ≤ expectedQueryWork law check := by
  unfold expectedQueryWork
  apply add_nonneg
  · exact Finset.expect_nonneg fun vector _ => law.meanWork_nonnegative vector
  · apply tsum_nonneg
    intro rejections
    apply Finset.sum_nonneg
    intro coordinate _
    apply Finset.expect_nonneg
    intro vector _
    rw [workAt_eq, queryTailTerm_eq]
    exact mul_nonneg
      (mul_nonneg ((line law.oracle check).nonnegative vector)
        (pow_nonneg (sub_nonneg.mpr (Line.rate_le_one _)) _))
      (law.lineWork_nonnegative _ _)

private theorem weak_work_nonnegative
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (program : Primitives Scalar Assignment) :
    0 ≤ PiRLC.CoordinateExtraction.expectedTotalWork algebra law checker program := by
  unfold PiRLC.CoordinateExtraction.expectedTotalWork
  apply add_nonneg (query_work_nonnegative algebra _ _)
  rw [← PaperWeakLaw.terminal_work_mean]
  exact Finset.sum_nonneg fun endpoint _ =>
    mul_nonneg ENNReal.toReal_nonneg (Nat.cast_nonneg _)

/-- Both probability and the postprocessing clock use this same endpoint law
of the actual checked coordinate extractor. -/
noncomputable def endpointLaw
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult) :=
  let typed := PiRLC.CoordinateExtraction.typedChecker algebra checker
  PaperWeakLaw.law (withChecker law typed).oracle (PiRLC.CoordinateCheckedCalls.check typed)

omit [DecidableEq Scalar] in
private theorem endpoint_weights_sum
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult) :
    ∑ endpoint, (endpointLaw algebra law checker endpoint).toReal = 1 := by
  simp only [endpointLaw, PaperWeakLaw.law_toReal]
  exact PaperWeakLaw.mass_total _ _

/-- Mean clock after the weak extractor has returned its actual list. The
weak terminal clock is already counted in `expectedTotalWork`. -/
noncomputable def finishMean
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (weak : Primitives Scalar Assignment)
    (strong : CheckedWitnessExtraction.Program shape carrier)
    (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))
    (probe : Probe K shape) : ℝ :=
  ∑ endpoint, (endpointLaw algebra law checker endpoint).toReal *
    ((finish strong decode probe (PaperWeakLaw.terminalValue weak endpoint)).work : ℝ)

noncomputable def finishBaseMean
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (weak : Primitives Scalar Assignment)
    (strong : CheckedWitnessExtraction.Program shape carrier)
    (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))
    (probe : Probe K shape) : ℝ :=
  ∑ endpoint, (endpointLaw algebra law checker endpoint).toReal *
    (finishBaseWork strong decode probe (PaperWeakLaw.terminalValue weak endpoint) : ℝ)

theorem finishMean_le
    (law : Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
    (checker : Response Assignment Scalar params arity → CheckResult)
    (weak : Primitives Scalar Assignment)
    (strong : CheckedWitnessExtraction.Program shape carrier)
    (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))
    (accessBound : Nat) (bounded : CostedWitnessProjection.Bounded strong.access accessBound)
    (probe : Probe K shape) :
    finishMean algebra law checker weak strong decode probe ≤
      finishBaseMean algebra law checker weak strong decode probe +
        ((CostedWitnessProjection.workBound shape carrier accessBound + 3 : Nat) : ℝ) := by
  calc
    _ ≤ ∑ endpoint, (endpointLaw algebra law checker endpoint).toReal *
        ((finishBaseWork strong decode probe (PaperWeakLaw.terminalValue weak endpoint) : ℝ) +
          ((CostedWitnessProjection.workBound shape carrier accessBound + 3 : Nat) : ℝ)) := by
      apply Finset.sum_le_sum
      intro endpoint _
      apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
      exact_mod_cast finish_work_le strong decode accessBound bounded probe
        (PaperWeakLaw.terminalValue weak endpoint)
    _ = _ := by
      simp only [mul_add, Finset.sum_add_distrib, ← Finset.sum_mul,
        endpoint_weights_sum, one_mul, finishBaseMean]

variable {Context State : Type*}
  (call : Context → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables →
    Result (Option (Probe K shape × State)))
  (law : Context → Probe K shape × State →
    Law (Fin arity.total) (PiRLC.CoordinateForkLaw.Challenge algebra) Assignment)
  (checker : Context → Probe K shape × State → Response Assignment Scalar params arity → CheckResult)
  (weak : Primitives Scalar Assignment)
  (strong : Context → CheckedWitnessExtraction.Program shape carrier)
  (decode : Option (List Assignment) → Result (Option (OutputWitness shape carrier.carrierWidth)))

/-- Actual prefix, one uniform checked suffix query, and the same decode/C
checker costs. No successful output or short-running context is selected. -/
noncomputable def baseClock (context : Context) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (point : CubePoint K shape.cubeVariables) : ℝ :=
  let issued := call context alpha gamma point
  (issued.work : ℝ) + match issued.value with
  | none => 0
  | some receipt =>
      (𝔼 vector, (withChecker (law context receipt)
        (PiRLC.CoordinateExtraction.typedChecker algebra (checker context receipt))).meanWork vector) +
      finishBaseMean algebra (law context receipt) (checker context receipt) weak
        (strong context) decode receipt.1

/-- Expected work of the operational sequential extractor conditional on the
original context and the actual C coins. Prefix aborts make no suffix call.
Every present receipt uses its own response/clock law and terminal output. -/
noncomputable def totalClock (context : Context) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (point : CubePoint K shape.cubeVariables) : ℝ :=
  let issued := call context alpha gamma point
  (issued.work : ℝ) + (match issued.value with
  | none => 0
  | some receipt =>
      PiRLC.CoordinateExtraction.expectedTotalWork algebra (law context receipt)
        (checker context receipt) weak +
      finishMean algebra (law context receipt) (checker context receipt) weak
        (strong context) decode receipt.1) + 1

/-- Erasing the prefix clock selects exactly the same receipt as the causal
probability experiment. This equality rules out a separate suffix context. -/
theorem totalClock_on_prefix {width : Nat}
    (firstPhase : Context → InteractivePrefix.Prover State shape width)
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value =
        InteractivePrefix.run (firstPhase context) alpha gamma point)
    (context : Context) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    totalClock algebra call law checker weak strong decode context alpha gamma point =
      ((call context alpha gamma point).work : ℝ) +
      (match InteractivePrefix.run (firstPhase context) alpha gamma point with
      | none => 0
      | some receipt =>
          PiRLC.CoordinateExtraction.expectedTotalWork algebra (law context receipt)
            (checker context receipt) weak +
          finishMean algebra (law context receipt) (checker context receipt) weak
            (strong context) decode receipt.1) + 1 := by
  simp only [totalClock, callCorrect]

theorem totalClock_nonnegative (context : Context)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    0 ≤ totalClock algebra call law checker weak strong decode context alpha gamma point := by
  unfold totalClock
  apply add_nonneg _ zero_le_one
  apply add_nonneg (Nat.cast_nonneg _)
  cases returned : (call context alpha gamma point).value with
  | none => exact le_rfl
  | some receipt =>
      apply add_nonneg (weak_work_nonnegative algebra _ _ _)
      exact Finset.sum_nonneg fun endpoint _ =>
        mul_nonneg ENNReal.toReal_nonneg (Nat.cast_nonneg _)

variable (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra semantics params algebra)
  (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring algebra.challengeValid)
  (correct : PiRLC.PaperForkExtractionWork.Correct laws.ring laws.assignmentModule weak)
  (bounds : PrimitiveBounds)
  (bounded : PiRLC.PaperForkExtractionWork.Bounded laws.ring weak bounds)
  (accessBound : Nat)
  (accessBounded : ∀ context, CostedWitnessProjection.Bounded (strong context).access accessBound)

include strongSet correct bounded accessBounded in
/-- The only amplification is the existing coordinate retry bound. All other
terms are clocks of the actual prefix, decoder, checker, and projection. -/
theorem totalClock_le (context : Context) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (point : CubePoint K shape.cubeVariables) :
    totalClock algebra call law checker weak strong decode context alpha gamma point ≤
      baseClock algebra call law checker weak strong decode context alpha gamma point *
        ((arity.total : ℝ) + 1) +
      ((arity.total * (bounds.coordinateWork + 3) + 2 +
        CostedWitnessProjection.workBound shape carrier accessBound + 4 : Nat) : ℝ) := by
  have prefixNonnegative : (0 : ℝ) ≤ (call context alpha gamma point).work := Nat.cast_nonneg _
  have countNonnegative : (0 : ℝ) ≤ arity.total := Nat.cast_nonneg _
  unfold totalClock baseClock
  cases returned : (call context alpha gamma point).value with
  | none =>
      simp only [returned]
      have overhead : (1 : ℝ) ≤ ((arity.total * (bounds.coordinateWork + 3) + 2 +
          CostedWitnessProjection.workBound shape carrier accessBound + 4 : Nat) : ℝ) := by
        exact_mod_cast (show 1 ≤ arity.total * (bounds.coordinateWork + 3) + 2 +
          CostedWitnessProjection.workBound shape carrier accessBound + 4 by omega)
      simp only [add_zero]
      nlinarith [mul_nonneg countNonnegative prefixNonnegative]
  | some receipt =>
      simp only [returned]
      have weakBound := PiRLC.CoordinateExtraction.expectedTotalWork_bound algebra laws strongSet
        (law context receipt) (checker context receipt) weak correct bounds bounded
      have finishBound := finishMean_le algebra (law context receipt) (checker context receipt)
        weak (strong context) decode accessBound (accessBounded context) receipt.1
      have postNonnegative : 0 ≤ finishBaseMean algebra (law context receipt) (checker context receipt)
          weak (strong context) decode receipt.1 :=
        Finset.sum_nonneg fun endpoint _ =>
          mul_nonneg ENNReal.toReal_nonneg (Nat.cast_nonneg _)
      push_cast at weakBound finishBound ⊢
      nlinarith [mul_nonneg countNonnegative prefixNonnegative,
        mul_nonneg countNonnegative postNonnegative]

include strongSet correct bounded accessBounded in
/-- Global EPT follows from the actual global call/check moment. No constant
bounds the adversary's time at each setup, prefix, or verifier response. -/
theorem expected_work_bound (contexts : PMF Context)
    (baseSummable : Summable fun context => (contexts context).toReal *
      StrongProbability.verifierMean (baseClock algebra call law checker weak strong decode context)) :
    Summable (fun context => (contexts context).toReal *
      StrongProbability.verifierMean (totalClock algebra call law checker weak strong decode context)) ∧
    StrongProbability.clockMean contexts (totalClock algebra call law checker weak strong decode) ≤
      StrongProbability.clockMean contexts (baseClock algebra call law checker weak strong decode) *
        ((arity.total : ℝ) + 1) +
      ((arity.total * (bounds.coordinateWork + 3) + 2 +
        CostedWitnessProjection.workBound shape carrier accessBound + 4 : Nat) : ℝ) := by
  apply StrongProbability.clockMean_le_mul_add_const contexts
    (baseClock algebra call law checker weak strong decode)
    (totalClock algebra call law checker weak strong decode) _ _
  · exact totalClock_nonnegative algebra call law checker weak strong decode
  · exact baseSummable
  · exact totalClock_le algebra call law checker weak strong decode laws strongSet
      correct bounds bounded accessBound accessBounded

include strongSet correct bounded accessBounded in
/-- Polynomial bounds on the global original call/check mean and the actual
extraction/access primitives give a polynomial bound for this same composed
extractor. The source count and field-access counts come from its program. -/
theorem expected_work_polynomial_bound (contexts : PMF Context)
    (baseSummable : Summable fun context => (contexts context).toReal *
      StrongProbability.verifierMean (baseClock algebra call law checker weak strong decode context))
    (securityParameter : Nat) (basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ)
    (basePPT : StrongProbability.clockMean contexts
      (baseClock algebra call law checker weak strong decode) ≤
        basePolynomial.eval (securityParameter : ℝ))
    (primitivePPT : (bounds.coordinateWork : ℝ) ≤
      primitivePolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    Summable (fun context => (contexts context).toReal *
      StrongProbability.verifierMean (totalClock algebra call law checker weak strong decode context)) ∧
    StrongProbability.clockMean contexts (totalClock algebra call law checker weak strong decode) ≤
      (Polynomial.C ((arity.total : ℝ) + 1) * basePolynomial +
        Polynomial.C (arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
        Polynomial.C (shape.freshCount : ℝ) *
          (Polynomial.C (WitnessProjection.privateWidth carrier : ℝ) *
            (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
        Polynomial.C (shape.runningCount : ℝ) *
          (Polynomial.C (carrier.carrierWidth : ℝ) *
            (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
        Polynomial.C 9).eval (securityParameter : ℝ) := by
  have actual := expected_work_bound algebra call law checker weak strong decode laws strongSet
    correct bounds bounded accessBound accessBounded contexts baseSummable
  refine ⟨actual.1, ?_⟩
  have base := mul_le_mul_of_nonneg_right basePPT
    (by positivity : 0 ≤ (arity.total : ℝ) + 1)
  have primitive := mul_le_mul_of_nonneg_left
    (_root_.add_le_add_right primitivePPT 3) (Nat.cast_nonneg arity.total : (0 : ℝ) ≤ arity.total)
  have fresh := mul_le_mul_of_nonneg_left accessPPT
    (by positivity : (0 : ℝ) ≤ (shape.freshCount : ℝ) * (WitnessProjection.privateWidth carrier : ℝ))
  have running := mul_le_mul_of_nonneg_left accessPPT
    (by positivity : (0 : ℝ) ≤ (shape.runningCount : ℝ) * (carrier.carrierWidth : ℝ))
  have bound := actual.2
  unfold CostedWitnessProjection.workBound at bound
  push_cast at bound
  simp only [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C]
  nlinarith

end NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionWork
