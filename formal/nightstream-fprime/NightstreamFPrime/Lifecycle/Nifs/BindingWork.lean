import NightstreamFPrime.Lifecycle.Nifs.BindingProbability
import NightstreamFPrime.Lifecycle.Nifs.InteractiveWork

/-!
Work of the binding reduction after the original context is supplied, under
the existing interactive coin model. The outer context-generation driver owns
preparation and fixed-key preprocessing work.
Two source runs retain their receipts and weak endpoints. Their actual retry,
decode and projection clocks are the existing NIFS clocks. The same pair law
then measures the additional integer-output work of `BindingReduction.runPair`.
Finite probability sums describe the experiment; no table is sampled at runtime.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.BindingWork

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork PiRLC.CoordinateForkLaw

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))

/-- Only postprocessing remains here. `runPair_work_eq` adds both actual
source clocks, so zero in these temporary records does not erase source work. -/
noncomputable def postClock {State : Type*}
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (left right : PaperCompositionAgreement.Observation State
      (BindingReduction.Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) : ℝ :=
  (∑ coordinate, ((BindingReduction.runPair program access
    ⟨left, 0⟩ ⟨right, 0⟩ coordinate).work : ℝ)) / (PaperProfile.arity.total : ℝ)

theorem postClock_range {State : Type*}
    (bounds : PrimitiveBounds)
    (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (accessBound : Nat) (accessBounded : CostedWitnessProjection.Bounded access accessBound)
    (left right : PaperCompositionAgreement.Observation State
      (BindingReduction.Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape) :
    0 ≤ postClock program access left right ∧ postClock program access left right ≤
      ((BindingOutput.crossWork bounds +
        (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 : Nat) : ℝ) := by
  have positive : (0 : ℝ) < PaperProfile.arity.total := by change (0 : ℝ) < 17; norm_num
  unfold postClock
  constructor
  · exact div_nonneg (Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _) positive.le
  · apply (div_le_iff₀ positive).mpr
    calc
      _ ≤ ∑ _coordinate : Fin PaperProfile.arity.total,
          ((BindingOutput.crossWork bounds +
            (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 : Nat) : ℝ) := by
        apply Finset.sum_le_sum
        intro coordinate _
        have actual := BindingReduction.runPair_work_le ajtai program bounds bounded
          access accessBound accessBounded ⟨left, 0⟩ ⟨right, 0⟩ coordinate
        dsimp only at actual
        simp only [Nat.zero_add] at actual
        exact_mod_cast actual
      _ = _ := by
        simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        rw [Fintype.card_fin]
        exact mul_comm _ _

variable {Context State Tape : Type*}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)
  (call : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))

omit [DecidableEq RingF] in
/-- The endpoint distribution in the binding probability is exactly the one
used for the source extractor's terminal and postprocessing clocks. -/
theorem suffixLaw_eq_workLaw (context : Context) (receipt : Probe K productionShape × State) :
    InteractiveComposition.suffixLaw relation ajtai running fresh continuation context
        receipt.1.coins receipt.1.response.fullOutput receipt.2 =
      PaperCompositionWork.endpointLaw (ProductionKey.key relation ajtai).piRlcAlgebra
        (InteractiveWork.law relation ajtai running fresh continuation context receipt)
        (InteractiveWork.parentChecker relation ajtai running fresh continuation context receipt) := rfl

noncomputable def postMean (context : Context) : ℝ :=
  PaperCompositionAgreement.pairMean
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
    (postClock program (sourceProgram context).access)

/-- Linearity of work uses the two complete source-run marginals plus the
postprocessing mean on their independent endpoint pair. Source runs include
their existing checks and projection even though only their retained endpoint
data is needed for the binding vector. -/
noncomputable def totalClock (context : Context)
    (_alpha : CubePoint K productionShape.cubeVariables) (_gamma : K)
    (_point : CubePoint K productionShape.cubeVariables) : ℝ :=
  2 * StrongProbability.verifierMean
    (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context) +
      postMean ajtai program relation running fresh originalFirstPhase publicCheck continuation sourceProgram context

variable (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (accessBound : Nat)
  (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)

include bounded accessBounded in
omit [DecidableEq RingF] in
theorem postMean_range (context : Context) :
    0 ≤ postMean ajtai program relation running fresh originalFirstPhase publicCheck continuation
        sourceProgram context ∧
    postMean ajtai program relation running fresh originalFirstPhase publicCheck continuation
        sourceProgram context ≤
      ((BindingOutput.crossWork bounds +
        (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 : Nat) : ℝ) := by
  classical
  have lower := PaperCompositionAgreement.pairMean_mono
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
    (fun _ _ => 0) (postClock program (sourceProgram context).access)
    (fun left right _ _ => (postClock_range ajtai program bounds bounded
      (sourceProgram context).access accessBound (accessBounded context) left right).1)
  have upper := PaperCompositionAgreement.pairMean_mono
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
    (postClock program (sourceProgram context).access)
    (fun _ _ => ((BindingOutput.crossWork bounds +
      (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 : Nat) : ℝ))
    (fun left right _ _ => (postClock_range ajtai program bounds bounded
      (sourceProgram context).access accessBound (accessBounded context) left right).2)
  exact ⟨by simpa only [PaperCompositionAgreement.pairMean_const] using lower,
    by simpa only [PaperCompositionAgreement.pairMean_const] using upper⟩

include bounded accessBounded in
/-- The reduction after context preparation inherits the actual source moment.
There is no new premise asserting the reduction's own total work. -/
theorem expected_work_bound (contexts : PMF Context)
    (sourceSummable : Summable fun context => (contexts context).toReal *
      StrongProbability.verifierMean
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context)) :
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean
      (totalClock ajtai program relation running fresh originalFirstPhase publicCheck continuation
        call sourceProgram context)) ∧
    StrongProbability.clockMean contexts
      (totalClock ajtai program relation running fresh originalFirstPhase publicCheck continuation call sourceProgram) ≤
      2 * StrongProbability.clockMean contexts
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram) +
      ((BindingOutput.crossWork bounds +
        (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 : Nat) : ℝ) := by
  let source := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    StrongProbability.verifierMean
      (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context)
  have bound := StrongProbability.clockMean_le_mul_add_const contexts source
    (totalClock ajtai program relation running fresh originalFirstPhase publicCheck continuation call sourceProgram)
    2 ((BindingOutput.crossWork bounds +
      (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 : Nat) : ℝ)
    (by
      intro context _ _ _
      apply add_nonneg
      · apply mul_nonneg (by norm_num : (0 : ℝ) ≤ 2)
        have nonnegative := StrongProbability.verifierMean_mono (shape := productionShape)
          (fun _ _ _ => 0)
          (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context)
          (PaperCompositionWork.totalClock_nonnegative (ProductionKey.key relation ajtai).piRlcAlgebra call
            (InteractiveWork.law relation ajtai running fresh continuation)
            (InteractiveWork.parentChecker relation ajtai running fresh continuation)
            program sourceProgram (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) context)
        simpa only [StrongProbability.verifierMean_const] using nonnegative
      · exact (postMean_range ajtai program relation running fresh originalFirstPhase publicCheck
          continuation sourceProgram bounds bounded accessBound accessBounded context).1)
    (by simpa only [source, StrongProbability.verifierMean_const] using sourceSummable)
    (by
      intro context _ _ _
      have post := (postMean_range ajtai program relation running fresh originalFirstPhase publicCheck
        continuation sourceProgram bounds bounded accessBound accessBounded context).2
      dsimp only [totalClock, source]
      linarith)
  simpa only [StrongProbability.clockMean, source, StrongProbability.verifierMean_const,
    mul_comm _ (2 : ℝ)] using bound

include bounded accessBounded in
/-- The bound is polynomial in the same actual source, primitive, and
coordinate-access budgets. The witness length is the selected carrier width. -/
theorem expected_work_polynomial_bound (contexts : PMF Context)
    (sourceSummable : Summable fun context => (contexts context).toReal *
      StrongProbability.verifierMean
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context))
    (securityParameter : Nat) (sourcePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ)
    (sourcePPT : StrongProbability.clockMean contexts
      (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram) ≤
        sourcePolynomial.eval (securityParameter : ℝ))
    (primitivePPT : (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean
      (totalClock ajtai program relation running fresh originalFirstPhase publicCheck continuation
        call sourceProgram context)) ∧
    StrongProbability.clockMean contexts
      (totalClock ajtai program relation running fresh originalFirstPhase publicCheck continuation call sourceProgram) ≤
      (Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 12).eval (securityParameter : ℝ) := by
  have actual := expected_work_bound ajtai program relation running fresh originalFirstPhase publicCheck
    continuation call sourceProgram bounds bounded accessBound accessBounded contexts sourceSummable
  refine ⟨actual.1, ?_⟩
  have cross : (BindingOutput.crossWork bounds : ℝ) ≤ 3 * bounds.coordinateWork := by
    exact_mod_cast BindingOutput.crossWork_le_coordinateWork bounds
  have access := mul_le_mul_of_nonneg_left accessPPT
    (Nat.cast_nonneg (FullShape logicalWidth publicFits).carrierWidth : (0 : ℝ) ≤
      (FullShape logicalWidth publicFits).carrierWidth)
  have total := actual.2
  push_cast at total
  simp only [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C]
  nlinarith

end NightstreamFPrime.Lifecycle.Nifs.BindingWork
