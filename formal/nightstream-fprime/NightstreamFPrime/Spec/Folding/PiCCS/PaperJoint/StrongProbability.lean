import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Analysis.Normed.Group.InfiniteSum
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution

/-!
SuperNeo v1.1 Appendix B.2 averaging for arbitrary PMF context and private-coin
laws. No context or prover randomness is replaced by uniform finite sampling.
The local helpers use real weighted sums of the stated normalized PMF.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongProbability

private noncomputable def mean {Sample : Type*} (law : PMF Sample)
    (value : Sample → ℝ) : ℝ :=
  ∑' sample, (law sample).toReal * value sample

private theorem weight_summable {Sample : Type*} (law : PMF Sample) :
    Summable (fun sample => (law sample).toReal) :=
  ENNReal.summable_toReal law.tsum_coe_ne_top

private theorem weight_sum {Sample : Type*} (law : PMF Sample) :
    (∑' sample, (law sample).toReal) = 1 := by
  rw [← ENNReal.tsum_toReal_eq (law.apply_ne_top), law.tsum_coe, ENNReal.toReal_one]

private theorem weighted_summable {Sample : Type*} (law : PMF Sample)
    (value : Sample → ℝ) (bound : ℝ) (bounded : ∀ sample, |value sample| ≤ bound) :
    Summable (fun sample => (law sample).toReal * value sample) := by
  apply Summable.of_norm_bounded ((weight_summable law).mul_right bound)
  intro sample
  rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
  exact mul_le_mul_of_nonneg_left (bounded sample) ENNReal.toReal_nonneg

private theorem mean_const {Sample : Type*} (law : PMF Sample) (value : ℝ) :
    mean law (fun _ => value) = value := by
  rw [mean, tsum_mul_right, weight_sum, one_mul]

private theorem mean_nonnegative {Sample : Type*} (law : PMF Sample)
    (value : Sample → ℝ) (nonnegative : ∀ sample, 0 ≤ value sample) :
    0 ≤ mean law value :=
  tsum_nonneg fun sample => mul_nonneg ENNReal.toReal_nonneg (nonnegative sample)

private theorem mean_mono {Sample : Type*} (law : PMF Sample)
    (left right : Sample → ℝ) (leftBound rightBound : ℝ)
    (leftBounded : ∀ sample, |left sample| ≤ leftBound)
    (rightBounded : ∀ sample, |right sample| ≤ rightBound)
    (ordered : ∀ sample, left sample ≤ right sample) :
    mean law left ≤ mean law right :=
  Summable.tsum_le_tsum (fun sample => mul_le_mul_of_nonneg_left (ordered sample) ENNReal.toReal_nonneg)
    (weighted_summable law left leftBound leftBounded)
    (weighted_summable law right rightBound rightBounded)

private theorem mean_add {Sample : Type*} (law : PMF Sample)
    (left right : Sample → ℝ) (leftBound rightBound : ℝ)
    (leftBounded : ∀ sample, |left sample| ≤ leftBound)
    (rightBounded : ∀ sample, |right sample| ≤ rightBound) :
    mean law (fun sample => left sample + right sample) = mean law left + mean law right := by
  unfold mean
  simp_rw [mul_add]
  exact Summable.tsum_add (weighted_summable law left leftBound leftBounded)
    (weighted_summable law right rightBound rightBounded)

private theorem mean_mul_const {Sample : Type*} (law : PMF Sample)
    (value : Sample → ℝ) (constant : ℝ) :
    mean law (fun sample => value sample * constant) = mean law value * constant := by
  unfold mean
  simp_rw [← mul_assoc]
  exact tsum_mul_right

private theorem mean_unit_interval {Sample : Type*} (law : PMF Sample)
    (value : Sample → ℝ) (bounded : ∀ sample, 0 ≤ value sample ∧ value sample ≤ 1) :
    0 ≤ mean law value ∧ mean law value ≤ 1 := by
  refine ⟨mean_nonnegative law value (fun sample => (bounded sample).1), ?_⟩
  calc
    _ ≤ mean law (fun _ => 1) :=
      mean_mono law value (fun _ => 1) 1 1
        (fun sample => by rw [abs_of_nonneg (bounded sample).1]; exact (bounded sample).2)
        (fun _ => by norm_num) (fun sample => (bounded sample).2)
    _ = 1 := mean_const law 1

private theorem mean_square_le {Sample : Type*} (law : PMF Sample)
    (value : Sample → ℝ) (bounded : ∀ sample, 0 ≤ value sample ∧ value sample ≤ 1) :
    (mean law value) ^ 2 ≤ mean law (fun sample => (value sample) ^ 2) := by
  have valueBounded : ∀ sample, |value sample| ≤ 1 := by
    intro sample
    rw [abs_of_nonneg (bounded sample).1]
    exact (bounded sample).2
  have squareBounded : ∀ sample, |(value sample) ^ 2| ≤ 1 := by
    intro sample
    rw [abs_of_nonneg (sq_nonneg _)]
    have constraints := bounded sample
    nlinarith [mul_nonneg constraints.1 (sub_nonneg.mpr constraints.2)]
  have valueSummable := weighted_summable law value 1 valueBounded
  have squareSummable := weighted_summable law (fun sample => (value sample) ^ 2) 1 squareBounded
  let center := mean law value
  have nonnegative :
      0 ≤ ∑' sample, (law sample).toReal * (value sample - center) ^ 2 :=
    tsum_nonneg fun sample => mul_nonneg ENNReal.toReal_nonneg (sq_nonneg _)
  have expansion (sample : Sample) :
      (law sample).toReal * (value sample - center) ^ 2 =
        ((law sample).toReal * (value sample) ^ 2 -
          2 * center * ((law sample).toReal * value sample)) +
        center ^ 2 * (law sample).toReal := by ring
  simp_rw [expansion] at nonnegative
  rw [Summable.tsum_add (squareSummable.sub (valueSummable.mul_left (2 * center)))
      ((weight_summable law).mul_left (center ^ 2)),
    Summable.tsum_sub squareSummable (valueSummable.mul_left (2 * center)),
    tsum_mul_left, tsum_mul_left, weight_sum] at nonnegative
  change 0 ≤ mean law (fun sample => (value sample) ^ 2) -
    2 * center * mean law value + center ^ 2 * 1 at nonnegative
  dsimp only [center] at nonnegative
  nlinarith

private theorem averaged_error_le_sqrt {Context : Type*} (contexts : PMF Context)
    (error success : Context → ℝ)
    (bounded : ∀ context, 0 ≤ error context ∧ error context ≤ success context ∧ success context ≤ 1)
    (testError collision : ℝ) (testNonnegative : 0 ≤ testError)
    (pairBound : mean contexts (fun context => error context * success context) ≤
      collision + mean contexts error * testError) :
    mean contexts error ≤ Real.sqrt (collision + testError) := by
  have errorRange (context : Context) : 0 ≤ error context ∧ error context ≤ 1 :=
    ⟨(bounded context).1, (bounded context).2.1.trans (bounded context).2.2⟩
  have errorMean := mean_unit_interval contexts error errorRange
  have squareBounded (context : Context) : |(error context) ^ 2| ≤ 1 := by
    rw [abs_of_nonneg (sq_nonneg _)]
    have constraints := errorRange context
    nlinarith [mul_nonneg constraints.1 (sub_nonneg.mpr constraints.2)]
  have productBounded (context : Context) : |error context * success context| ≤ 1 := by
    have errorNonnegative := (bounded context).1
    have successNonnegative := errorNonnegative.trans (bounded context).2.1
    rw [abs_of_nonneg (mul_nonneg errorNonnegative successNonnegative)]
    exact (mul_le_mul (errorRange context).2 (bounded context).2.2
      successNonnegative (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (by norm_num)
  have squareMean : mean contexts (fun context => (error context) ^ 2) ≤
      mean contexts (fun context => error context * success context) := by
    apply mean_mono contexts _ _ 1 1 squareBounded productBounded
    intro context
    simpa only [pow_two] using
      mul_le_mul_of_nonneg_left (bounded context).2.1 (bounded context).1
  have squareBound : (mean contexts error) ^ 2 ≤ collision + testError := by
    have jensen := mean_square_le contexts error errorRange
    have linear := mul_le_mul_of_nonneg_right errorMean.2 testNonnegative
    nlinarith
  have radicandNonnegative : 0 ≤ collision + testError := (sq_nonneg _).trans squareBound
  have sqrtSquared := Real.sq_sqrt radicandNonnegative
  nlinarith [Real.sqrt_nonneg (collision + testError)]

open scoped BigOperators
open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open ConcreteCarrier StrongReduction CausalExecution
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal
  (uniformAverage uniformAverage_const uniformAverage_congr uniformAverage_mono
    uniformAverage_add sampleSpace_nonempty)

attribute [local instance] Classical.propDecidable

private noncomputable def pointMean (dimensionCount : Nat) (value : CubePoint K dimensionCount → ℝ) : ℝ :=
  uniformAverage GoldilocksRoots.fullChallengeSet dimensionCount fun coordinates =>
    if dimension : coordinates.length = dimensionCount then value ⟨coordinates, dimension⟩ else 0

private theorem pointMean_const (dimensionCount : Nat) (value : ℝ) :
    pointMean dimensionCount (fun _ => value) = value := by
  calc
    _ = uniformAverage GoldilocksRoots.fullChallengeSet dimensionCount (fun _ => value) := by
      apply uniformAverage_congr
      intro coordinates dimension
      simp only [dif_pos dimension]
    _ = value := uniformAverage_const _ sampleSpace_nonempty _ _

private theorem pointMean_mono (dimensionCount : Nat) (left right : CubePoint K dimensionCount → ℝ)
    (ordered : ∀ point, left point ≤ right point) :
    pointMean dimensionCount left ≤ pointMean dimensionCount right := by
  apply uniformAverage_mono
  intro coordinates dimension
  simpa only [dif_pos dimension] using ordered ⟨coordinates, dimension⟩

private theorem pointMean_add (dimensionCount : Nat) (left right : CubePoint K dimensionCount → ℝ) :
    pointMean dimensionCount (fun point => left point + right point) =
      pointMean dimensionCount left + pointMean dimensionCount right := by
  unfold pointMean
  calc
    _ = uniformAverage GoldilocksRoots.fullChallengeSet dimensionCount (fun coordinates =>
        (if dimension : coordinates.length = dimensionCount then left ⟨coordinates, dimension⟩ else 0) +
        (if dimension : coordinates.length = dimensionCount then right ⟨coordinates, dimension⟩ else 0)) := by
      apply uniformAverage_congr
      intro coordinates dimension
      simp only [dif_pos dimension]
    _ = _ := uniformAverage_add _ _ _ _

private theorem uniformAverage_mul_const (samples : Finset K) (rounds : Nat)
    (value : List K → ℝ) (constant : ℝ) :
    uniformAverage samples rounds (fun coordinates => value coordinates * constant) =
      uniformAverage samples rounds value * constant := by
  induction rounds generalizing value with
  | zero => rfl
  | succ rounds ih =>
      simp only [uniformAverage]
      simp_rw [ih]
      exact (Finset.expect_mul samples (fun challenge =>
        uniformAverage samples rounds (fun rest => value (challenge :: rest))) constant).symm

private theorem pointMean_mul_const (dimensionCount : Nat) (value : CubePoint K dimensionCount → ℝ)
    (constant : ℝ) :
    pointMean dimensionCount (fun point => value point * constant) =
      pointMean dimensionCount value * constant := by
  unfold pointMean
  calc
    _ = uniformAverage GoldilocksRoots.fullChallengeSet dimensionCount (fun coordinates =>
        (if dimension : coordinates.length = dimensionCount then value ⟨coordinates, dimension⟩ else 0) *
          constant) := by
      apply uniformAverage_congr
      intro coordinates dimension
      simp only [dif_pos dimension]
    _ = _ := uniformAverage_mul_const _ _ _ _

/-- The exact independent alpha, gamma, and round-point sampling law. This
same average is used by outcome probabilities and the one-run work consumer. -/
noncomputable def verifierMean {shape : Shape}
    (value : CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ) : ℝ :=
  pointMean shape.cubeVariables fun alpha =>
    𝔼 gamma ∈ GoldilocksRoots.fullChallengeSet,
      pointMean shape.cubeVariables (value alpha gamma)

theorem verifierMean_const {shape : Shape} (constant : ℝ) :
    verifierMean (shape := shape) (fun _ _ _ => constant) = constant := by
  simp only [verifierMean, pointMean_const, Finset.expect_const sampleSpace_nonempty]

theorem verifierMean_mono {shape : Shape}
    (left right : CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (ordered : ∀ alpha gamma point, left alpha gamma point ≤ right alpha gamma point) :
    verifierMean left ≤ verifierMean right := by
  apply pointMean_mono
  intro alpha
  apply Finset.expect_le_expect
  intro gamma _
  exact pointMean_mono _ _ _ (ordered alpha gamma)

theorem verifierMean_range {shape : Shape}
    (value : CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (bound : ℝ) (range : ∀ alpha gamma point, 0 ≤ value alpha gamma point ∧ value alpha gamma point ≤ bound) :
    0 ≤ verifierMean value ∧ verifierMean value ≤ bound := by
  constructor
  · rw [← verifierMean_const (shape := shape) 0]
    exact verifierMean_mono _ _ (fun alpha gamma point => (range alpha gamma point).1)
  · rw [← verifierMean_const (shape := shape) bound]
    exact verifierMean_mono _ _ (fun alpha gamma point => (range alpha gamma point).2)

theorem verifierMean_add {shape : Shape}
    (left right : CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ) :
    verifierMean (fun alpha gamma point => left alpha gamma point + right alpha gamma point) =
      verifierMean left + verifierMean right := by
  simp only [verifierMean, pointMean_add, Finset.expect_add_distrib]

theorem verifierMean_mul_const {shape : Shape}
    (value : CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (constant : ℝ) :
    verifierMean (fun alpha gamma point => value alpha gamma point * constant) =
      verifierMean value * constant := by
  simp only [verifierMean, pointMean_mul_const, ← Finset.expect_mul]

/-- Arbitrary private coins are sampled independently of the verifier coins.
The PMF can have countably infinite support and need not be uniform. -/
noncomputable def executionMean {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ) : ℝ :=
  mean tapes fun tape => verifierMean fun alpha gamma point => value (run (prover tape) alpha gamma point)

private theorem executionMean_const {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width) (constant : ℝ) :
    executionMean tapes prover (fun _ => constant) = constant := by
  simp only [executionMean, verifierMean_const, mean_const]

private theorem executionMean_range {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (bound : ℝ) (range : ∀ outcome, 0 ≤ value outcome ∧ value outcome ≤ bound) :
    0 ≤ executionMean tapes prover value ∧ executionMean tapes prover value ≤ bound := by
  have each (tape : Tape) := verifierMean_range
    (fun alpha gamma point => value (run (prover tape) alpha gamma point)) bound
    (fun alpha gamma point => range _)
  refine ⟨mean_nonnegative tapes _ (fun tape => (each tape).1), ?_⟩
  calc
    _ ≤ mean tapes (fun _ => bound) :=
      mean_mono tapes _ _ bound |bound|
        (fun tape => by rw [abs_of_nonneg (each tape).1]; exact (each tape).2)
        (fun _ => le_rfl) (fun tape => (each tape).2)
    _ = bound := mean_const tapes bound

private theorem executionMean_mono {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (left right : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (leftBound rightBound : ℝ)
    (leftRange : ∀ outcome, 0 ≤ left outcome ∧ left outcome ≤ leftBound)
    (rightRange : ∀ outcome, 0 ≤ right outcome ∧ right outcome ≤ rightBound)
    (ordered : ∀ outcome, left outcome ≤ right outcome) :
    executionMean tapes prover left ≤ executionMean tapes prover right := by
  apply mean_mono tapes _ _ leftBound rightBound
  · intro tape
    have range := verifierMean_range (fun alpha gamma point => left (run (prover tape) alpha gamma point))
      leftBound (fun _ _ _ => leftRange _)
    rw [abs_of_nonneg range.1]
    exact range.2
  · intro tape
    have range := verifierMean_range (fun alpha gamma point => right (run (prover tape) alpha gamma point))
      rightBound (fun _ _ _ => rightRange _)
    rw [abs_of_nonneg range.1]
    exact range.2
  · intro tape
    exact verifierMean_mono _ _ (fun _ _ _ => ordered _)

private theorem executionMean_add {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (left right : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (leftBound rightBound : ℝ)
    (leftRange : ∀ outcome, 0 ≤ left outcome ∧ left outcome ≤ leftBound)
    (rightRange : ∀ outcome, 0 ≤ right outcome ∧ right outcome ≤ rightBound) :
    executionMean tapes prover (fun outcome => left outcome + right outcome) =
      executionMean tapes prover left + executionMean tapes prover right := by
  unfold executionMean
  simp_rw [verifierMean_add]
  apply mean_add tapes _ _ leftBound rightBound
  · intro tape
    have range := verifierMean_range (fun alpha gamma point => left (run (prover tape) alpha gamma point))
      leftBound (fun _ _ _ => leftRange _)
    rw [abs_of_nonneg range.1]
    exact range.2
  · intro tape
    have range := verifierMean_range (fun alpha gamma point => right (run (prover tape) alpha gamma point))
      rightBound (fun _ _ _ => rightRange _)
    rw [abs_of_nonneg range.1]
    exact range.2

private theorem executionMean_mul_const {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ) (constant : ℝ) :
    executionMean tapes prover (fun outcome => value outcome * constant) =
      executionMean tapes prover value * constant := by
  simp only [executionMean, verifierMean_mul_const, mean_mul_const]

private noncomputable def indicator (event : Prop) : ℝ := if event then 1 else 0

private theorem indicator_range (event : Prop) : 0 ≤ indicator event ∧ indicator event ≤ 1 := by
  unfold indicator
  split_ifs <;> norm_num

private theorem indicator_mono {left right : Prop} (implies : left → right) :
    indicator left ≤ indicator right := by
  unfold indicator
  by_cases holds : left
  · simp only [if_pos holds, if_pos (implies holds), le_refl]
  · simp only [if_neg holds]
    split_ifs <;> norm_num

/-- The actual returned probe succeeds in the verifier and relaxed output relation. -/
def RelaxedSuccess {Commitment PublicInput : Type*} {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (outcome : Option (Probe K shape × OutputWitness shape columns)) : Prop :=
  ∃ probe witness, outcome = some (probe, witness) ∧
    probe.FixedWidthAccepted extensionOps K.embed statement width ∧
    AmbientOutputHolds extensionOps K.embed openingMaps params statement probe witness

/-- Source validity refers to the same witness returned by this execution. -/
def SourceValid {Commitment PublicInput : Type*} {shape : Shape} {columns blockCount : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (outcome : Option (Probe K shape × OutputWitness shape columns)) : Prop :=
  ∃ probe witness, outcome = some (probe, witness) ∧
    SourceHolds extensionOps K.embed openingMaps params statement witness

/-- B.2 Err: successful relaxed output whose same witness fails the source relation. -/
def SourceError {Commitment PublicInput : Type*} {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (outcome : Option (Probe K shape × OutputWitness shape columns)) : Prop :=
  RelaxedSuccess (width := width) openingMaps params statement outcome ∧
    ¬ SourceValid openingMaps params statement outcome

/-- The pair event is a subset of the paper's nonabort witness disagreement.
Both values here are actual successful relaxed outputs. -/
def SuccessfulDisagreement {Commitment PublicInput : Type*} {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (left right : Option (Probe K shape × OutputWitness shape columns)) : Prop :=
  RelaxedSuccess (width := width) openingMaps params statement left ∧
    RelaxedSuccess (width := width) openingMaps params statement right ∧
    ∃ leftProbe leftWitness rightProbe rightWitness,
      left = some (leftProbe, leftWitness) ∧ right = some (rightProbe, rightWitness) ∧
      leftWitness ≠ rightWitness

private theorem executionMean_indicator_range {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (event : Option (Probe K shape × OutputWitness shape columns) → Prop) :
    0 ≤ executionMean tapes prover (fun outcome => indicator (event outcome)) ∧
      executionMean tapes prover (fun outcome => indicator (event outcome)) ≤ 1 :=
  executionMean_range tapes prover _ 1 (fun _ => indicator_range _)

private theorem executionMean_indicator_mono {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (left right : Option (Probe K shape × OutputWitness shape columns) → Prop)
    (implies : ∀ outcome, left outcome → right outcome) :
    executionMean tapes prover (fun outcome => indicator (left outcome)) ≤
      executionMean tapes prover (fun outcome => indicator (right outcome)) :=
  executionMean_mono tapes prover _ _ 1 1 (fun _ => indicator_range _) (fun _ => indicator_range _)
    (fun outcome => indicator_mono (implies outcome))

section FixedContext

variable {Tape Commitment PublicInput : Type*} {shape : Shape} {columns blockCount width : Nat}
  (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
  (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
  (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)

/-- Relaxed success in the actual one-run law at one fixed context. -/
noncomputable def successProbability : ℝ :=
  executionMean tapes prover fun outcome =>
    indicator (RelaxedSuccess (width := width) openingMaps params statement outcome)

/-- Source extraction error in the same actual one-run law. -/
noncomputable def errorProbability : ℝ :=
  executionMean tapes prover fun outcome =>
    indicator (SourceError (width := width) openingMaps params statement outcome)

/-- Successful relaxed output with a valid source witness from that same run. -/
noncomputable def sourceProbability : ℝ :=
  executionMean tapes prover fun outcome =>
    indicator (RelaxedSuccess (width := width) openingMaps params statement outcome ∧
      SourceValid openingMaps params statement outcome)

/-- A caller's value function measures the same source-success event when
its two cases agree. The private indicator's decision procedure stays here. -/
theorem executionMean_eq_sourceProbability
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ)
    (onSuccess : ∀ outcome,
      (RelaxedSuccess (width := width) openingMaps params statement outcome ∧
        SourceValid openingMaps params statement outcome) → value outcome = 1)
    (onFailure : ∀ outcome,
      ¬ (RelaxedSuccess (width := width) openingMaps params statement outcome ∧
        SourceValid openingMaps params statement outcome) → value outcome = 0) :
    executionMean tapes prover value = sourceProbability tapes prover openingMaps params statement := by
  unfold sourceProbability
  apply congrArg (executionMean tapes prover)
  funext outcome
  by_cases successful :
      RelaxedSuccess (width := width) openingMaps params statement outcome ∧
        SourceValid openingMaps params statement outcome
  · simpa only [indicator, if_pos successful] using onSuccess outcome successful
  · simpa only [indicator, if_neg successful] using onFailure outcome successful

private noncomputable def disagreementFrom
    (first : Option (Probe K shape × OutputWitness shape columns)) : ℝ :=
  executionMean tapes prover fun second =>
    indicator (SuccessfulDisagreement (width := width) openingMaps params statement first second)

/-- Two independent executions from the same context and private-tape law.
Every tape and every verifier challenge is resampled in the inner execution. -/
noncomputable def disagreementProbability : ℝ :=
  executionMean tapes prover (disagreementFrom tapes prover openingMaps params statement)

private theorem probability_ranges :
    0 ≤ errorProbability tapes prover openingMaps params statement ∧
    errorProbability tapes prover openingMaps params statement ≤
      successProbability tapes prover openingMaps params statement ∧
    successProbability tapes prover openingMaps params statement ≤ 1 := by
  refine ⟨(executionMean_indicator_range tapes prover _).1, ?_,
    (executionMean_indicator_range tapes prover _).2⟩
  exact executionMean_indicator_mono tapes prover _ _ (fun _ error => error.1)

private theorem disagreementFrom_range (first : Option (Probe K shape × OutputWitness shape columns)) :
    0 ≤ disagreementFrom tapes prover openingMaps params statement first ∧
      disagreementFrom tapes prover openingMaps params statement first ≤ 1 :=
  executionMean_indicator_range tapes prover _

private theorem disagreementProbability_range :
    0 ≤ disagreementProbability tapes prover openingMaps params statement ∧
      disagreementProbability tapes prover openingMaps params statement ≤ 1 :=
  executionMean_range tapes prover _ 1 (disagreementFrom_range tapes prover openingMaps params statement)

private theorem success_partition :
    successProbability tapes prover openingMaps params statement =
      errorProbability tapes prover openingMaps params statement +
      sourceProbability tapes prover openingMaps params statement := by
  have partition (outcome : Option (Probe K shape × OutputWitness shape columns)) :
      indicator (RelaxedSuccess (width := width) openingMaps params statement outcome) =
      indicator (SourceError (width := width) openingMaps params statement outcome) +
      indicator (RelaxedSuccess (width := width) openingMaps params statement outcome ∧
        SourceValid openingMaps params statement outcome) := by
    unfold SourceError indicator
    by_cases success : RelaxedSuccess (width := width) openingMaps params statement outcome <;>
      by_cases source : SourceValid openingMaps params statement outcome <;> simp [success, source]
  unfold successProbability errorProbability sourceProbability
  simp_rw [partition]
  exact executionMean_add tapes prover _ _ 1 1
    (fun _ => indicator_range _) (fun _ => indicator_range _)

variable (freshBound : params.b = 2)
  (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
  (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)

private theorem testError_nonnegative : 0 ≤ IndependentExecution.testError shape width := by
  unfold IndependentExecution.testError
  positivity

include freshBound constantLaw degreeCovers

private theorem agreed_probability_le (firstWitness : OutputWitness shape columns)
    (invalid : ¬ SourceHolds extensionOps K.embed openingMaps params statement firstWitness) :
    executionMean tapes prover (fun outcome =>
      indicator (IndependentExecution.AgreedSuccess openingMaps params statement width firstWitness outcome)) ≤
      IndependentExecution.testError shape width := by
  have valueRange (tape : Tape) :
      0 ≤ IndependentExecution.agreementProbability openingMaps params statement firstWitness (prover tape) ∧
      IndependentExecution.agreementProbability openingMaps params statement firstWitness (prover tape) ≤ 1 :=
    verifierMean_range (shape := shape)
      (fun alpha gamma point => indicator (IndependentExecution.AgreedSuccess
        openingMaps params statement width firstWitness (run (prover tape) alpha gamma point))) 1
      (fun _ _ _ => indicator_range _)
  change mean tapes (fun tape => IndependentExecution.agreementProbability
    openingMaps params statement firstWitness (prover tape)) ≤ _
  calc
    _ ≤ mean tapes (fun _ => IndependentExecution.testError shape width) :=
      mean_mono tapes _ _ 1 |IndependentExecution.testError shape width|
        (fun tape => by rw [abs_of_nonneg (valueRange tape).1]; exact (valueRange tape).2)
        (fun _ => le_rfl)
        (fun tape => IndependentExecution.agreementProbability_le openingMaps params freshBound
          statement constantLaw degreeCovers firstWitness invalid (prover tape))
    _ = _ := mean_const tapes _

private theorem first_error_second_success_le
    (first : Option (Probe K shape × OutputWitness shape columns)) :
    indicator (SourceError (width := width) openingMaps params statement first) *
        successProbability tapes prover openingMaps params statement ≤
      disagreementFrom tapes prover openingMaps params statement first +
      indicator (SourceError (width := width) openingMaps params statement first) *
        IndependentExecution.testError shape width := by
  by_cases firstError : SourceError (width := width) openingMaps params statement first
  · obtain ⟨firstProbe, firstWitness, firstReturned, _firstAccepted, _firstAmbient⟩ := firstError.1
    have invalid : ¬ SourceHolds extensionOps K.embed openingMaps params statement firstWitness := by
      intro source
      exact firstError.2 ⟨firstProbe, firstWitness, firstReturned, source⟩
    have inclusion (second : Option (Probe K shape × OutputWitness shape columns)) :
        indicator (RelaxedSuccess (width := width) openingMaps params statement second) ≤
        indicator (SuccessfulDisagreement (width := width) openingMaps params statement first second) +
        indicator (IndependentExecution.AgreedSuccess openingMaps params statement width firstWitness second) := by
      by_cases secondSuccess : RelaxedSuccess (width := width) openingMaps params statement second
      · have success := secondSuccess
        obtain ⟨secondProbe, secondWitness, secondReturned, secondAccepted, secondAmbient⟩ := secondSuccess
        by_cases agreement : secondWitness = firstWitness
        · have agreed : IndependentExecution.AgreedSuccess openingMaps params statement width firstWitness second :=
            ⟨secondProbe, secondWitness, secondReturned, secondAccepted, secondAmbient, agreement⟩
          simp only [indicator, if_pos success, if_pos agreed]
          split_ifs <;> norm_num
        · have different : SuccessfulDisagreement (width := width) openingMaps params statement first second :=
            ⟨firstError.1, success, firstProbe, firstWitness, secondProbe, secondWitness,
              firstReturned, secondReturned, Ne.symm agreement⟩
          simp only [indicator, if_pos success, if_pos different]
          split_ifs <;> norm_num
      · have leftZero : indicator (RelaxedSuccess (width := width) openingMaps params statement second) = 0 :=
          if_neg secondSuccess
        rw [leftZero]
        exact add_nonneg (indicator_range _).1 (indicator_range _).1
    have averaged : successProbability tapes prover openingMaps params statement ≤
        disagreementFrom tapes prover openingMaps params statement first +
        executionMean tapes prover (fun outcome => indicator
          (IndependentExecution.AgreedSuccess openingMaps params statement width firstWitness outcome)) := by
      calc
        _ ≤ executionMean tapes prover (fun second =>
            indicator (SuccessfulDisagreement (width := width) openingMaps params statement first second) +
            indicator (IndependentExecution.AgreedSuccess openingMaps params statement width firstWitness second)) := by
          apply executionMean_mono tapes prover _ _ 1 2 (fun _ => indicator_range _)
          · intro second
            have left := indicator_range (SuccessfulDisagreement (width := width) openingMaps params statement first second)
            have right := indicator_range (IndependentExecution.AgreedSuccess openingMaps params statement width firstWitness second)
            exact ⟨add_nonneg left.1 right.1, by linarith⟩
          · exact inclusion
        _ = _ := executionMean_add tapes prover _ _ 1 1
          (fun _ => indicator_range _) (fun _ => indicator_range _)
    have tested := agreed_probability_le tapes prover openingMaps params statement
      freshBound constantLaw degreeCovers firstWitness invalid
    simp only [indicator, if_pos firstError, one_mul]
    linarith
  · have zero : indicator (SourceError (width := width) openingMaps params statement first) = 0 :=
      if_neg firstError
    simp only [zero, zero_mul, add_zero]
    exact (disagreementFrom_range tapes prover openingMaps params statement first).1

private theorem pair_bound :
    errorProbability tapes prover openingMaps params statement *
      successProbability tapes prover openingMaps params statement ≤
    disagreementProbability tapes prover openingMaps params statement +
      errorProbability tapes prover openingMaps params statement * IndependentExecution.testError shape width := by
  let error := fun outcome => indicator (SourceError (width := width) openingMaps params statement outcome)
  let success := successProbability tapes prover openingMaps params statement
  let test := IndependentExecution.testError shape width
  have successRange : 0 ≤ success ∧ success ≤ 1 := executionMean_indicator_range tapes prover _
  have testNonnegative : 0 ≤ test := testError_nonnegative
  have leftRange (outcome : Option (Probe K shape × OutputWitness shape columns)) :
      0 ≤ error outcome * success ∧ error outcome * success ≤ 1 := by
    have errorRange := indicator_range (SourceError (width := width) openingMaps params statement outcome)
    constructor
    · exact mul_nonneg errorRange.1 successRange.1
    · nlinarith [mul_nonneg (sub_nonneg.mpr errorRange.2) successRange.1]
  have rightRange (outcome : Option (Probe K shape × OutputWitness shape columns)) :
      0 ≤ disagreementFrom tapes prover openingMaps params statement outcome + error outcome * test ∧
      disagreementFrom tapes prover openingMaps params statement outcome + error outcome * test ≤ 1 + test := by
    have disagreement := disagreementFrom_range tapes prover openingMaps params statement outcome
    have errorRange := indicator_range (SourceError (width := width) openingMaps params statement outcome)
    constructor
    · exact add_nonneg disagreement.1 (mul_nonneg errorRange.1 testNonnegative)
    · nlinarith [mul_le_mul_of_nonneg_right errorRange.2 testNonnegative]
  calc
    _ = executionMean tapes prover (fun outcome => error outcome * success) :=
      (executionMean_mul_const tapes prover error success).symm
    _ ≤ executionMean tapes prover (fun outcome =>
        disagreementFrom tapes prover openingMaps params statement outcome + error outcome * test) :=
      executionMean_mono tapes prover _ _ 1 (1 + test) leftRange rightRange
        (first_error_second_success_le tapes prover openingMaps params statement freshBound constantLaw degreeCovers)
    _ = _ := by
      rw [executionMean_add tapes prover _ _ 1 test
        (disagreementFrom_range tapes prover openingMaps params statement)
        (fun outcome => by
          have range := indicator_range (SourceError (width := width) openingMaps params statement outcome)
          exact ⟨mul_nonneg range.1 testNonnegative,
            (mul_le_mul_of_nonneg_right range.2 testNonnegative).trans_eq (one_mul _)⟩),
        executionMean_mul_const]
      rfl

end FixedContext

section Contexts

variable {Context Tape Commitment PublicInput : Type*} {shape : Shape} {columns blockCount width : Nat}
  (contexts : PMF Context) (tapes : Context → PMF Tape)
  (prover : Context → Tape → Prover shape columns width)
  (openingMaps : Context → OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
  (statement : Context → Statement K Commitment PublicInput shape columns blockCount baseOps)

/-- Setup/adversary contexts have their stated arbitrary probability law. -/
noncomputable def globalSuccessProbability : ℝ :=
  mean contexts fun context => successProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)

/-- The actual source extraction error, averaged over the same contexts. -/
noncomputable def globalErrorProbability : ℝ :=
  mean contexts fun context => errorProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)

/-- Successful source witnesses returned by that same one-run experiment. -/
noncomputable def globalSourceProbability : ℝ :=
  mean contexts fun context => sourceProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)

/-- Two fresh independent executions per context, with the context law unchanged. -/
noncomputable def globalDisagreementProbability : ℝ :=
  mean contexts fun context => disagreementProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)

private theorem global_success_partition :
    globalSuccessProbability contexts tapes prover openingMaps params statement =
      globalErrorProbability contexts tapes prover openingMaps params statement +
      globalSourceProbability contexts tapes prover openingMaps params statement := by
  unfold globalSuccessProbability globalErrorProbability globalSourceProbability
  simp_rw [success_partition]
  apply mean_add contexts _ _ 1 1
  · intro context
    have range := probability_ranges (tapes context) (prover context)
      (openingMaps context) params (statement context)
    rw [abs_of_nonneg range.1]
    exact range.2.1.trans range.2.2
  · intro context
    have range : 0 ≤ sourceProbability (tapes context) (prover context)
        (openingMaps context) params (statement context) ∧
        sourceProbability (tapes context) (prover context)
          (openingMaps context) params (statement context) ≤ 1 :=
      executionMean_indicator_range (tapes context) (prover context) _
    rw [abs_of_nonneg range.1]
    exact range.2

variable (freshBound : params.b = 2)
  (constantLaw : ∀ context, MatrixCoefficientSource.ConstantTermLaw baseOps
    (statement context).matrixSource.kernel)
  (degreeCovers : ∀ context, ((statement context).verifierInput K.embed).sumcheckDegreeBound ≤ width)

include freshBound constantLaw degreeCovers

/-- B.2 equations (13)–(20) for the actual causal execution events. The
square-root loss uses arbitrary context and private-tape PMFs, including aborts.
No numeric error, success, independence, or pair-bound premise is supplied. -/
theorem source_error_le_sqrt :
    globalErrorProbability contexts tapes prover openingMaps params statement ≤
      Real.sqrt (globalDisagreementProbability contexts tapes prover openingMaps params statement +
        IndependentExecution.testError shape width) := by
  let error := fun context => errorProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)
  let success := fun context => successProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)
  let collision := fun context => disagreementProbability (tapes context) (prover context)
    (openingMaps context) params (statement context)
  let test := IndependentExecution.testError shape width
  have testNonnegative : 0 ≤ test := testError_nonnegative
  have ranges (context : Context) :
      0 ≤ error context ∧ error context ≤ success context ∧ success context ≤ 1 :=
    probability_ranges (tapes context) (prover context) (openingMaps context) params (statement context)
  have collisionRange (context : Context) : 0 ≤ collision context ∧ collision context ≤ 1 :=
    disagreementProbability_range (tapes context) (prover context)
      (openingMaps context) params (statement context)
  apply averaged_error_le_sqrt contexts error success ranges test
    (globalDisagreementProbability contexts tapes prover openingMaps params statement) testNonnegative
  have averaged : mean contexts (fun context => error context * success context) ≤
      mean contexts (fun context => collision context + error context * test) := by
    apply mean_mono contexts _ _ 1 (1 + test)
    · intro context
      have range := ranges context
      have successNonnegative := range.1.trans range.2.1
      have errorAtMostOne := range.2.1.trans range.2.2
      rw [abs_of_nonneg (mul_nonneg range.1 successNonnegative)]
      nlinarith [mul_nonneg (sub_nonneg.mpr errorAtMostOne) successNonnegative]
    · intro context
      have range := ranges context
      have errorAtMostOne := range.2.1.trans range.2.2
      rw [abs_of_nonneg (add_nonneg (collisionRange context).1 (mul_nonneg range.1 testNonnegative))]
      have scaled := mul_le_mul_of_nonneg_right errorAtMostOne testNonnegative
      linarith [(collisionRange context).2]
    · intro context
      exact pair_bound (tapes context) (prover context) (openingMaps context) params (statement context)
        freshBound (constantLaw context) (degreeCovers context)
  apply averaged.trans_eq
  rw [mean_add contexts collision (fun context => error context * test) 1 test
      (fun context => by rw [abs_of_nonneg (collisionRange context).1]; exact (collisionRange context).2)
      (fun context => by
        rw [abs_of_nonneg (mul_nonneg (ranges context).1 testNonnegative)]
        exact (mul_le_mul_of_nonneg_right ((ranges context).2.1.trans (ranges context).2.2)
          testNonnegative).trans_eq (one_mul _)),
    mean_mul_const]
  rfl

/-- The one-run extractor's actual source-success probability has the paper's
square-root loss from the actual two-run witness-disagreement probability. -/
theorem source_success_ge :
    globalSuccessProbability contexts tapes prover openingMaps params statement -
      Real.sqrt (globalDisagreementProbability contexts tapes prover openingMaps params statement +
        IndependentExecution.testError shape width) ≤
      globalSourceProbability contexts tapes prover openingMaps params statement := by
  have partition := global_success_partition contexts tapes prover openingMaps params statement
  have error := source_error_le_sqrt contexts tapes prover openingMaps params statement
    freshBound constantLaw degreeCovers
  linarith

/-- The paper's separate witness-disagreement premise can bound the named
actual pair event. Fiat–Shamir and hardness assumptions are not inserted here. -/
theorem source_success_ge_of_disagreement_le (collisionBound : ℝ)
    (bound : globalDisagreementProbability contexts tapes prover openingMaps params statement ≤ collisionBound) :
    globalSuccessProbability contexts tapes prover openingMaps params statement -
      Real.sqrt (collisionBound + IndependentExecution.testError shape width) ≤
      globalSourceProbability contexts tapes prover openingMaps params statement := by
  have direct := source_success_ge contexts tapes prover openingMaps params statement
    freshBound constantLaw degreeCovers
  have loss := Real.sqrt_le_sqrt
    (_root_.add_le_add bound (le_refl (IndependentExecution.testError shape width)))
  linarith

end Contexts

/-- Actual work is averaged on the same private tape and public coins as run. -/
noncomputable def clockMean {Tape : Type*} {shape : Shape} (tapes : PMF Tape)
    (clock : Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ) : ℝ :=
  mean tapes fun tape => verifierMean (clock tape)

/-- Finite suffix outcomes can be averaged before the verifier coins. -/
theorem verifierMean_sum {Index : Type*} {shape : Shape} (indices : Finset Index)
    (value : Index → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ) :
    verifierMean (fun alpha gamma point => ∑ index ∈ indices, value index alpha gamma point) =
      ∑ index ∈ indices, verifierMean (value index) := by
  classical
  induction indices using Finset.induction_on with
  | empty => simp only [Finset.sum_empty, verifierMean_const]
  | @insert index indices absent induction =>
      simp only [Finset.sum_insert absent, verifierMean_add, induction]

/-- The finite private-tape mean and verifier mean commute. The caller
must separately identify the finite tape with its actual suffix law. -/
theorem clockMean_fintype {Tape : Type*} [Fintype Tape] {shape : Shape}
    (tapes : PMF Tape)
    (clock : Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ) :
    clockMean tapes clock = verifierMean (fun alpha gamma point =>
      ∑ tape, (tapes tape).toReal * clock tape alpha gamma point) := by
  simp only [clockMean, mean, tsum_fintype, verifierMean_sum]
  apply Finset.sum_congr rfl
  intro tape _
  simpa only [mul_comm] using (verifierMean_mul_const (clock tape) (tapes tape).toReal).symm

theorem clockMean_mul_const {Tape : Type*} {shape : Shape} (tapes : PMF Tape)
    (clock : Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (factor : ℝ) :
    clockMean tapes (fun tape alpha gamma point => clock tape alpha gamma point * factor) =
      clockMean tapes clock * factor := by
  simp only [clockMean, mean, verifierMean_mul_const, ← mul_assoc, tsum_mul_right]

/-- Bounded event observables have a summable mean under any context law. -/
theorem clockMean_summable_of_bounded {Tape : Type*} {shape : Shape}
    (tapes : PMF Tape)
    (clock : Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (bound : ℝ)
    (range : ∀ tape alpha gamma point,
      0 ≤ clock tape alpha gamma point ∧ clock tape alpha gamma point ≤ bound) :
    Summable fun tape => (tapes tape).toReal * verifierMean (clock tape) := by
  apply weighted_summable tapes _ bound
  intro tape
  have limits := verifierMean_range (clock tape) bound (range tape)
  simpa only [abs_of_nonneg limits.1] using limits.2

/-- The clock and outcome means use the same tape and verifier samples. Keep
the outcome value opaque when applying this equality to a concrete event. -/
theorem clockMean_run_eq_executionMean {Tape : Type*} {shape : Shape} {columns width : Nat}
    (tapes : PMF Tape) (prover : Tape → Prover shape columns width)
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ) :
    clockMean tapes (fun tape alpha gamma point => value (run (prover tape) alpha gamma point)) =
      executionMean tapes prover value := rfl

/-- The total clock has finite expectation and the stated bound whenever
the actual base clock does. Overhead must come from the proved concrete
projection/access/control count at the consuming call site. -/
theorem clockMean_le_add_const {Tape : Type*} {shape : Shape} (tapes : PMF Tape)
    (base total : Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (overhead : ℝ)
    (totalNonnegative : ∀ tape alpha gamma point, 0 ≤ total tape alpha gamma point)
    (baseSummable : Summable fun tape => (tapes tape).toReal * verifierMean (base tape))
    (pointwise : ∀ tape alpha gamma point,
      total tape alpha gamma point ≤ base tape alpha gamma point + overhead) :
    Summable (fun tape => (tapes tape).toReal * verifierMean (total tape)) ∧
      clockMean tapes total ≤ clockMean tapes base + overhead := by
  have nonnegative (tape : Tape) : 0 ≤ verifierMean (total tape) := by
    rw [← verifierMean_const (shape := shape) 0]
    exact verifierMean_mono _ _ (totalNonnegative tape)
  have each (tape : Tape) : verifierMean (total tape) ≤ verifierMean (base tape) + overhead := by
    calc
      _ ≤ verifierMean (fun alpha gamma point => base tape alpha gamma point + overhead) :=
        verifierMean_mono _ _ (pointwise tape)
      _ = _ := by rw [verifierMean_add, verifierMean_const]
  have overheadSummable := (weight_summable tapes).mul_right overhead
  have envelopeSummable := baseSummable.add overheadSummable
  have weighted (tape : Tape) :
      (tapes tape).toReal * verifierMean (total tape) ≤
        (tapes tape).toReal * verifierMean (base tape) + (tapes tape).toReal * overhead := by
    simpa only [mul_add] using mul_le_mul_of_nonneg_left (each tape) ENNReal.toReal_nonneg
  have totalSummable := Summable.of_nonneg_of_le
    (fun tape => mul_nonneg ENNReal.toReal_nonneg (nonnegative tape)) weighted envelopeSummable
  refine ⟨totalSummable, ?_⟩
  calc
    _ ≤ ∑' tape, ((tapes tape).toReal * verifierMean (base tape) + (tapes tape).toReal * overhead) :=
      Summable.tsum_le_tsum weighted totalSummable envelopeSummable
    _ = _ := by
      rw [Summable.tsum_add baseSummable overheadSummable, tsum_mul_right, weight_sum, one_mul]
      rfl

/-- A proved pointwise amplification of the call mean gives a global work
bound without a uniform bound on individual contexts or calls. -/
theorem clockMean_le_mul_add_const {Tape : Type*} {shape : Shape} (tapes : PMF Tape)
    (base total : Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables → ℝ)
    (factor overhead : ℝ)
    (totalNonnegative : ∀ tape alpha gamma point, 0 ≤ total tape alpha gamma point)
    (baseSummable : Summable fun tape => (tapes tape).toReal * verifierMean (base tape))
    (pointwise : ∀ tape alpha gamma point,
      total tape alpha gamma point ≤ base tape alpha gamma point * factor + overhead) :
    Summable (fun tape => (tapes tape).toReal * verifierMean (total tape)) ∧
      clockMean tapes total ≤ clockMean tapes base * factor + overhead := by
  have scaled : Summable fun tape => (tapes tape).toReal *
      verifierMean (fun alpha gamma point => base tape alpha gamma point * factor) := by
    simpa only [verifierMean_mul_const, ← mul_assoc] using baseSummable.mul_right factor
  have bound := clockMean_le_add_const tapes
    (fun tape alpha gamma point => base tape alpha gamma point * factor) total overhead
    totalNonnegative scaled pointwise
  simpa only [clockMean_mul_const] using bound

/-- Sample the stated setup context, then that context's private tape. The
PMF bind/map constructors prove normalization without a finite-support premise. -/
noncomputable def jointTapeLaw {Context Tape : Type*}
    (contexts : PMF Context) (tapes : Context → PMF Tape) : PMF (Context × Tape) :=
  contexts.bind fun context => (tapes context).map fun tape => (context, tape)

theorem jointTapeLaw_apply {Context Tape : Type*}
    (contexts : PMF Context) (tapes : Context → PMF Tape) (context : Context) (tape : Tape) :
    jointTapeLaw contexts tapes (context, tape) = contexts context * tapes context tape := by
  classical
  have mapped (chosen : Context) :
      ((tapes chosen).map (fun inner => (chosen, inner))) (context, tape) =
        if context = chosen then tapes chosen tape else 0 := by
    by_cases same : context = chosen
    · subst chosen
      simp [PMF.map_apply, Prod.mk.injEq, eq_comm]
      symm
      calc
        _ = (if tape = tape then tapes context tape else 0) :=
          tsum_eq_single tape (fun other different => if_neg (Ne.symm different))
        _ = _ := if_pos rfl
    · simp only [PMF.map_apply, Prod.mk.injEq, same, false_and, ite_false, tsum_zero]
  rw [jointTapeLaw, PMF.bind_apply]
  simp_rw [mapped]
  simp only [mul_ite, mul_zero]
  simpa only [eq_comm] using
    (tsum_ite_eq context (fun chosen => contexts chosen * tapes chosen tape))

/-- Bounded run events use exactly the same joint law as the global work
bound. Absolute summability justifies the nested context/tape probability. -/
theorem joint_clockMean_eq_nested {Context Tape : Type*} {shape : Shape}
    (contexts : PMF Context) (tapes : Context → PMF Tape)
    (clock : (Context × Tape) → CubePoint K shape.cubeVariables → K →
      CubePoint K shape.cubeVariables → ℝ)
    (range : ∀ sample alpha gamma point,
      0 ≤ clock sample alpha gamma point ∧ clock sample alpha gamma point ≤ 1) :
    Summable (fun sample =>
      (jointTapeLaw contexts tapes sample).toReal * verifierMean (clock sample)) ∧
    clockMean (jointTapeLaw contexts tapes) clock =
      ∑' context, (contexts context).toReal *
        clockMean (tapes context) (fun tape => clock (context, tape)) := by
  have bounded (sample : Context × Tape) : |verifierMean (clock sample)| ≤ 1 := by
    have limits := verifierMean_range (clock sample) 1 (range sample)
    rw [abs_of_nonneg limits.1]
    exact limits.2
  have totalSummable := weighted_summable (jointTapeLaw contexts tapes)
    (fun sample => verifierMean (clock sample)) 1 bounded
  refine ⟨totalSummable, ?_⟩
  change (∑' sample, (jointTapeLaw contexts tapes sample).toReal * verifierMean (clock sample)) = _
  rw [totalSummable.tsum_prod]
  apply tsum_congr
  intro context
  simp_rw [jointTapeLaw_apply, ENNReal.toReal_mul, mul_assoc]
  rw [tsum_mul_left]
  rfl

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongProbability
