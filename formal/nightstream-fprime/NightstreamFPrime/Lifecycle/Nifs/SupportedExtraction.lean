import NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput
import NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement
import NightstreamFPrime.Lifecycle.Nifs.BindingProbability
import NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingProbability
import NightstreamFPrime.Lifecycle.Nifs.BindingWork
import NightstreamFPrime.Lifecycle.Nifs.SupportedContinuation
import NightstreamFPrime.Lifecycle.Nifs.InteractiveWork

/-!
Selected interactive extraction from the actual checked prefix and its
reachable continuations. The ring algebra is fixed by the production key;
the public checker is fixed to its existing verifier. Low-norm invertibility
and correctness of costed witness operations remain explicit.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw

variable {Context State Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The selected public verifier runs on the literal probe returned by the
causal prefix. No independently supplied checker can select its success event. -/
def publicCheck (context : Context) (probe : Probe K productionShape) : Bool :=
  let input : ProtocolPolynomial.VerifierInput K productionShape := {
    constraintPolynomial := ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      ProductionRelation.polynomial
    priorPoint := (running context).point
    claimedPadCoefficient := fun coordinate =>
      ((running context).evaluations coordinate.running).pad coordinate.coefficient
    claimedMatrixCoefficient := fun coordinate =>
      ((running context).evaluations coordinate.running).matrix coordinate.matrix coordinate.coefficient }
  let output : ProtocolPolynomial.OutputMessage K productionShape := {
    freshMatrixImage := fun source matrix =>
      probe.response.fullOutput.matrixCoordinate (UnifiedSources.freshSourceIndex source)
        matrix Phi81CoefficientKernel.constant
    sourceAssignment := fun source =>
      probe.response.fullOutput.padCoordinate source Phi81CoefficientKernel.constant
    padImage := fun coordinate =>
      probe.response.fullOutput.padCoordinate (UnifiedSources.runningSourceIndex coordinate.running)
        coordinate.coefficient
    matrixImage := fun coordinate =>
      probe.response.fullOutput.matrixCoordinate (UnifiedSources.runningSourceIndex coordinate.running)
        coordinate.matrix coordinate.coefficient }
  ProtocolPolynomial.FixedWidth.check extensionOps 9 input
    probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
    output probe.response.rounds

/-- Correctness of the checker chosen by this reduction is discharged for
all probes, including malformed messages and rejected public equations. -/
theorem publicCheck_correct (context : Context) (probe : Probe K productionShape) :
    publicCheck running context probe = true ↔
      probe.FixedWidthAccepted extensionOps K.embed
        ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) 9 := Iff.rfl

variable
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (contexts : PMF Context)
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)))
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (sourceCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9)
    (sourceProgram context) (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context)))

include lowNorm correct bounded sourceCorrect in
/-- Every probability term uses the same reachable continuation. The right
side counts the source values that the checked projection actually returns.
There is no finite-work requirement on an impossible private state. -/
theorem returned_source_success_bound :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (PaperCompositionProbability.disagreementProbability contexts
        (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running))
        (none : InteractiveComposition.Endpoint relation ajtai)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
        (InteractiveComposition.consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) +
          IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation program sourceProgram contexts := by
  dsimp only
  rw [InteractiveOutput.returnedSourceProbability_eq relation ajtai running fresh
    originalFirstPhase (publicCheck running) _ program sourceProgram sourceCorrect contexts]
  exact InteractiveComposition.source_success_bound relation ajtai running fresh
    originalFirstPhase (publicCheck running) _ program (publicCheck_correct relation ajtai running fresh)
    (PaperExtractionAlgebra.extractionAlgebra ajtai)
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm)
    correct bounds bounded contexts

include lowNorm correct bounded sourceCorrect in
/-- The selected interactive loss is the weak retry loss plus the strong
test loss and the measured binding event of two actual executions. A bound
on that event from public-seed MSIS hardness remains a separate contract. -/
theorem returned_source_bound_with_binding :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (InteractiveAgreement.bindingProbability relation ajtai running fresh
        originalFirstPhase (publicCheck running) continuation program contexts +
        IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation program sourceProgram contexts := by
  dsimp only
  have source := returned_source_success_bound relation ajtai running fresh contexts
    originalFirstPhase abortTape provider program sourceProgram lowNorm
    correct bounds bounded sourceCorrect
  have agreement := InteractiveAgreement.disagreement_le_bindingProbability relation ajtai running fresh
    originalFirstPhase (publicCheck running)
    (SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider)
    program (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct contexts
  exact (sub_le_sub_left (Real.sqrt_le_sqrt
    (_root_.add_le_add agreement (le_refl (IndependentExecution.testError productionShape 9)))) _).trans source

include lowNorm correct bounded sourceCorrect in
/-- The existing source-success theorem now consumes the actual same-key
short-kernel output probability. Its arity loss comes from the reduction's
uniform coordinate choice. No numerical MSIS or Fiat-Shamir premise is added. -/
theorem returned_source_bound_with_msis :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (BindingProbability.successProbability ajtai program relation running fresh
        originalFirstPhase (publicCheck running) continuation (fun context => (sourceProgram context).access)
        contexts * PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation program sourceProgram contexts := by
  dsimp only
  have source := returned_source_bound_with_binding relation ajtai running fresh contexts
    originalFirstPhase abortTape provider program sourceProgram lowNorm
    correct bounds bounded sourceCorrect
  have reduction := BindingProbability.binding_le_success ajtai program relation running fresh
    originalFirstPhase (publicCheck running)
    (SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider)
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct
    (fun context => (sourceProgram context).access) (fun context => (sourceCorrect context).access) contexts
  exact (sub_le_sub_left (Real.sqrt_le_sqrt
    (_root_.add_le_add reduction (le_refl (IndependentExecution.testError productionShape 9)))) _).trans source

include lowNorm correct bounded sourceCorrect in
/-- The v1.2 additive source bound uses the actual adaptive MSIS reduction
under the original context law. Both selected calls pass the executable
relaxed check; zero-success contexts remain in the experiment. Hardness and
resource applicability remain separate from this probability inequality. -/
theorem returned_source_bound_with_adaptive_msis :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation) - InteractiveComposition.weakLoss relation ajtai -
      IndependentExecution.testError productionShape 9 -
      AdaptiveBindingProbability.successProbability relation ajtai program running fresh
        originalFirstPhase (publicCheck running) continuation sourceProgram contexts * PaperProfile.arity.total ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation program sourceProgram contexts := by
  dsimp only
  let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
  rw [InteractiveOutput.returnedSourceProbability_eq relation ajtai running fresh
    originalFirstPhase (publicCheck running) continuation program sourceProgram sourceCorrect contexts]
  have source := InteractiveComposition.source_success_retry_bound relation ajtai running fresh
    originalFirstPhase (publicCheck running) continuation program (publicCheck_correct relation ajtai running fresh)
    (PaperExtractionAlgebra.extractionAlgebra ajtai)
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct bounds bounded contexts
  have reduction := AdaptiveBindingProbability.retryDisagreement_le_success relation ajtai program running fresh
    originalFirstPhase (publicCheck running) continuation sourceProgram
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct contexts
    (fun context => (sourceCorrect context).access) sourceCorrect
  exact (sub_le_sub_left reduction _).trans source

include lowNorm correct bounded sourceCorrect in
/-- The same checked execution has both the selected source-return bound
and polynomial expected work. The first conjunct connects its actual prefix
clock to the receipt used by the probability law. All moments are global;
there is no uniform time bound on individual contexts or private calls. -/
theorem probability_and_expected_work
    (call : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value = InteractivePrefix.run
        (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running) context) alpha gamma point)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)
    (securityParameter : Nat) (basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
    let base := InteractiveWork.baseClock relation ajtai running fresh continuation call program sourceProgram
    let total := InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean contexts base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∀ context alpha gamma point,
      total context alpha gamma point = ((call context alpha gamma point).work : ℝ) +
        (match InteractivePrefix.run
          (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running) context) alpha gamma point with
        | none => 0
        | some receipt =>
            PiRLC.CoordinateExtraction.expectedTotalWork (ProductionKey.key relation ajtai).piRlcAlgebra
              (InteractiveWork.law relation ajtai running fresh continuation context receipt)
              (InteractiveWork.parentChecker relation ajtai running fresh continuation context receipt) program +
            PaperCompositionWork.finishMean (ProductionKey.key relation ajtai).piRlcAlgebra
              (InteractiveWork.law relation ajtai running fresh continuation context receipt)
              (InteractiveWork.parentChecker relation ajtai running fresh continuation context receipt) program
              (sourceProgram context) (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) receipt.1) + 1) ∧
    (StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (InteractiveAgreement.bindingProbability relation ajtai running fresh
        originalFirstPhase (publicCheck running) continuation program contexts +
        IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation program sourceProgram contexts) ∧
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (total context)) ∧
    StrongProbability.clockMean contexts total ≤
      (Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
        Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
        Polynomial.C (productionShape.freshCount : ℝ) *
          (Polynomial.C (WitnessProjection.privateWidth (FullShape logicalWidth publicFits) : ℝ) *
            (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
        Polynomial.C (productionShape.runningCount : ℝ) *
          (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
            (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
        Polynomial.C 13).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT
  let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
  refine ⟨?_, ?_, ?_⟩
  · intro context alpha gamma point
    have clock := InteractiveWork.totalClock_on_checked_prefix relation ajtai running fresh continuation call program
      sourceProgram originalFirstPhase (publicCheck running) callCorrect context alpha gamma point
    refine clock.trans ?_
    dsimp only [InteractiveWork.firstPhase, InteractiveComposition.firstPhase]
    cases InteractivePrefix.run
      (InteractivePrefix.checked (originalFirstPhase context) ((publicCheck running) context)) alpha gamma point <;> rfl
  · exact returned_source_bound_with_binding relation ajtai running fresh contexts originalFirstPhase
      abortTape provider program sourceProgram lowNorm correct bounds bounded sourceCorrect
  · exact InteractiveWork.expected_work_polynomial_bound relation ajtai running fresh continuation call program
      sourceProgram (PaperExtractionAlgebra.extractionAlgebra ajtai)
      (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct bounds bounded
      accessBound accessBounded contexts baseSummable securityParameter basePolynomial primitivePolynomial
      accessPolynomial basePPT primitivePPT accessPPT

include lowNorm correct bounded sourceCorrect in
/-- The original preparation call generates the context once for both
source runs. Probability uses its actual returned context, and work includes
its full preparation/preprocessing clock. The checked prefix and reachable
continuation remain the existing selected NIFS execution. -/
theorem msis_probability_and_expected_work {SetupTape : Type*}
    (setupTapes : PMF SetupTape) (prepare : SetupTape → Result Context)
    (preparedContexts : contexts = ContextPreparation.contexts setupTapes prepare)
    (preparationSummable : Summable fun tape => (setupTapes tape).toReal * (prepare tape).work)
    (call : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value = InteractivePrefix.run
        (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running) context) alpha gamma point)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)
    (securityParameter : Nat)
    (preparationPolynomial basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
    let base := InteractiveWork.baseClock relation ajtai running fresh continuation call program sourceProgram
    let sourceTotal := InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram
    let bindingTotal := BindingWork.totalClock ajtai program relation running fresh originalFirstPhase (publicCheck running)
      continuation call sourceProgram
    let total := ContextPreparation.clock prepare
      (fun context => StrongProbability.verifierMean (bindingTotal context))
    let sourcePolynomial := Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
      Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
      Polynomial.C (productionShape.freshCount : ℝ) *
        (Polynomial.C (WitnessProjection.privateWidth (FullShape logicalWidth publicFits) : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
      Polynomial.C (productionShape.runningCount : ℝ) *
        (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) + Polynomial.C 13
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean contexts base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∑' tape, (setupTapes tape).toReal * (prepare tape).work) ≤
      preparationPolynomial.eval (securityParameter : ℝ) →
    (∀ context alpha gamma point,
      sourceTotal context alpha gamma point = ((call context alpha gamma point).work : ℝ) +
        (match InteractivePrefix.run
          (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running) context) alpha gamma point with
        | none => 0
        | some receipt =>
            PiRLC.CoordinateExtraction.expectedTotalWork (ProductionKey.key relation ajtai).piRlcAlgebra
              (InteractiveWork.law relation ajtai running fresh continuation context receipt)
              (InteractiveWork.parentChecker relation ajtai running fresh continuation context receipt) program +
            PaperCompositionWork.finishMean (ProductionKey.key relation ajtai).piRlcAlgebra
              (InteractiveWork.law relation ajtai running fresh continuation context receipt)
              (InteractiveWork.parentChecker relation ajtai running fresh continuation context receipt) program
              (sourceProgram context) (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) receipt.1) + 1) ∧
    (StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt ((∑' tape, (setupTapes tape).toReal * BindingProbability.localSuccessProbability ajtai program
        (sourceProgram (prepare tape).value).access relation running fresh originalFirstPhase (publicCheck running)
        continuation (prepare tape).value) * PaperProfile.arity.total +
          IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (publicCheck running) continuation program sourceProgram contexts) ∧
    Summable (fun tape => (setupTapes tape).toReal * total tape) ∧
    (∑' tape, (setupTapes tape).toReal * total tape) ≤
      (preparationPolynomial + Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 13).eval (securityParameter : ℝ) := by
  subst contexts
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT preparationPPT
  let contexts := ContextPreparation.contexts setupTapes prepare
  let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase (publicCheck running)) abortTape provider
  have source := probability_and_expected_work relation ajtai running fresh contexts originalFirstPhase
    abortTape provider program sourceProgram lowNorm correct bounds bounded sourceCorrect
    call callCorrect accessBound accessBounded securityParameter basePolynomial primitivePolynomial accessPolynomial
    baseSummable basePPT primitivePPT accessPPT
  refine ⟨source.1, ?_, ?_⟩
  · have success := returned_source_bound_with_msis relation ajtai running fresh contexts originalFirstPhase
      abortTape provider program sourceProgram lowNorm correct bounds bounded sourceCorrect
    dsimp only [contexts] at success
    rw [BindingProbability.prepared_successProbability_eq] at success
    exact success
  · exact BindingWork.prepared_expected_work_polynomial_bound ajtai program relation running fresh originalFirstPhase
      (publicCheck running) continuation call sourceProgram bounds bounded accessBound accessBounded setupTapes prepare
      preparationSummable source.2.2.1 securityParameter preparationPolynomial _ primitivePolynomial accessPolynomial
      preparationPPT source.2.2.2 primitivePPT accessPPT

end NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction
