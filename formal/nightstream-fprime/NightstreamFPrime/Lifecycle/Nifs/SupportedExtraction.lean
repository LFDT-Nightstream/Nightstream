import NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput
import NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement
import NightstreamFPrime.Lifecycle.Nifs.BindingProbability
import NightstreamFPrime.Lifecycle.Nifs.BindingWork
import NightstreamFPrime.Lifecycle.Nifs.SupportedContinuation
import NightstreamFPrime.Lifecycle.Nifs.InteractiveWork

/-!
Selected interactive extraction from the actual checked prefix and its
reachable continuations. The ring algebra is fixed by the production key;
low-norm invertibility and correctness of costed primitives remain explicit.
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
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (contexts : PMF Context)
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck))
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))
  (checkCorrect : ∀ context probe, publicCheck context probe = true ↔
    probe.FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) 9)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (sourceCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9)
    (sourceProgram context) (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context)))

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- Every probability term uses the same reachable continuation. The right
side counts the source values that the checked projection actually returns.
There is no finite-work requirement on an impossible private state. -/
theorem returned_source_success_bound :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (PaperCompositionProbability.disagreementProbability contexts
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck)
        (none : InteractiveComposition.Endpoint relation ajtai)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
        (InteractiveComposition.consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) +
          IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts := by
  dsimp only
  rw [InteractiveOutput.returnedSourceProbability_eq relation ajtai running fresh
    originalFirstPhase publicCheck _ program sourceProgram sourceCorrect contexts]
  exact InteractiveComposition.source_success_bound relation ajtai running fresh
    originalFirstPhase publicCheck _ program checkCorrect
    (PaperExtractionAlgebra.extractionAlgebra ajtai)
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm)
    correct bounds bounded contexts

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- The selected interactive loss is the weak retry loss plus the strong
test loss and the measured binding event of two actual executions. A bound
on that event from public-seed MSIS hardness remains a separate contract. -/
theorem returned_source_bound_with_binding :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (InteractiveAgreement.bindingProbability relation ajtai running fresh
        originalFirstPhase publicCheck continuation program contexts +
        IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts := by
  dsimp only
  have source := returned_source_success_bound relation ajtai running fresh contexts
    originalFirstPhase publicCheck abortTape provider program sourceProgram checkCorrect lowNorm
    correct bounds bounded sourceCorrect
  have agreement := InteractiveAgreement.disagreement_le_bindingProbability relation ajtai running fresh
    originalFirstPhase publicCheck
    (SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider)
    program (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct contexts
  exact (sub_le_sub_left (Real.sqrt_le_sqrt
    (_root_.add_le_add agreement (le_refl (IndependentExecution.testError productionShape 9)))) _).trans source

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- The existing source-success theorem now consumes the actual same-key
short-kernel output probability. Its arity loss comes from the reduction's
uniform coordinate choice. No numerical MSIS or Fiat-Shamir premise is added. -/
theorem returned_source_bound_with_msis :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (BindingProbability.successProbability ajtai program relation running fresh
        originalFirstPhase publicCheck continuation (fun context => (sourceProgram context).access)
        contexts * PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts := by
  dsimp only
  have source := returned_source_bound_with_binding relation ajtai running fresh contexts
    originalFirstPhase publicCheck abortTape provider program sourceProgram checkCorrect lowNorm
    correct bounds bounded sourceCorrect
  have reduction := BindingProbability.binding_le_success ajtai program relation running fresh
    originalFirstPhase publicCheck
    (SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider)
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct
    (fun context => (sourceProgram context).access) (fun context => (sourceCorrect context).access) contexts
  exact (sub_le_sub_left (Real.sqrt_le_sqrt
    (_root_.add_le_add reduction (le_refl (IndependentExecution.testError productionShape 9)))) _).trans source

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- The same checked execution has both the selected source-return bound
and polynomial expected work. The first conjunct connects its actual prefix
clock to the receipt used by the probability law. All moments are global;
there is no uniform time bound on individual contexts or private calls. -/
theorem probability_and_expected_work
    (call : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value = InteractivePrefix.run
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context) alpha gamma point)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)
    (securityParameter : Nat) (basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    let base := InteractiveWork.baseClock relation ajtai running fresh continuation call program sourceProgram
    let total := InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean contexts base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∀ context alpha gamma point,
      total context alpha gamma point = ((call context alpha gamma point).work : ℝ) +
        (match InteractivePrefix.run
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context) alpha gamma point with
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
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (InteractiveAgreement.bindingProbability relation ajtai running fresh
        originalFirstPhase publicCheck continuation program contexts +
        IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts) ∧
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (total context)) ∧
    StrongProbability.clockMean contexts total ≤
      (Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
        Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
        Polynomial.C (productionShape.freshCount : ℝ) *
          (Polynomial.C (WitnessProjection.privateWidth (FullShape logicalWidth publicFits) : ℝ) *
            (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
        Polynomial.C (productionShape.runningCount : ℝ) *
          (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
            (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
        Polynomial.C 9).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT
  let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
  refine ⟨?_, ?_, ?_⟩
  · intro context alpha gamma point
    have clock := InteractiveWork.totalClock_on_checked_prefix relation ajtai running fresh continuation call program
      sourceProgram originalFirstPhase publicCheck callCorrect context alpha gamma point
    refine clock.trans ?_
    dsimp only [InteractiveWork.firstPhase, InteractiveComposition.firstPhase]
    cases InteractivePrefix.run
      (InteractivePrefix.checked (originalFirstPhase context) (publicCheck context)) alpha gamma point <;> rfl
  · exact returned_source_bound_with_binding relation ajtai running fresh contexts originalFirstPhase
      publicCheck abortTape provider program sourceProgram checkCorrect lowNorm correct bounds bounded sourceCorrect
  · exact InteractiveWork.expected_work_polynomial_bound relation ajtai running fresh continuation call program
      sourceProgram (PaperExtractionAlgebra.extractionAlgebra ajtai)
      (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct bounds bounded
      accessBound accessBounded contexts baseSummable securityParameter basePolynomial primitivePolynomial
      accessPolynomial basePPT primitivePPT accessPPT

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- The selected source theorem and the binding reduction share
the checked prefix, supported continuation, and actual work premises. The
source loss uses the emitted same-key MSIS vector probability. -/
theorem msis_probability_and_expected_work
    (call : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value = InteractivePrefix.run
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context) alpha gamma point)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)
    (securityParameter : Nat) (basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    let base := InteractiveWork.baseClock relation ajtai running fresh continuation call program sourceProgram
    let sourceTotal := InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram
    let bindingTotal := BindingWork.totalClock ajtai program relation running fresh originalFirstPhase publicCheck
      continuation call sourceProgram
    let sourcePolynomial := Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
      Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
      Polynomial.C (productionShape.freshCount : ℝ) *
        (Polynomial.C (WitnessProjection.privateWidth (FullShape logicalWidth publicFits) : ℝ) *
          (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) +
      Polynomial.C (productionShape.runningCount : ℝ) *
        (Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 1) + Polynomial.C 2) + Polynomial.C 9
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean contexts base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∀ context alpha gamma point,
      sourceTotal context alpha gamma point = ((call context alpha gamma point).work : ℝ) +
        (match InteractivePrefix.run
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context) alpha gamma point with
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
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (BindingProbability.successProbability ajtai program relation running fresh
        originalFirstPhase publicCheck continuation (fun context => (sourceProgram context).access)
        contexts * PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts) ∧
    Summable (fun context => (contexts context).toReal * StrongProbability.verifierMean (bindingTotal context)) ∧
    StrongProbability.clockMean contexts bindingTotal ≤
      (Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 12).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT
  let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
  have source := probability_and_expected_work relation ajtai running fresh contexts originalFirstPhase
    publicCheck abortTape provider program sourceProgram checkCorrect lowNorm correct bounds bounded sourceCorrect
    call callCorrect accessBound accessBounded securityParameter basePolynomial primitivePolynomial accessPolynomial
    baseSummable basePPT primitivePPT accessPPT
  refine ⟨source.1, ?_, ?_⟩
  · exact returned_source_bound_with_msis relation ajtai running fresh contexts originalFirstPhase
      publicCheck abortTape provider program sourceProgram checkCorrect lowNorm correct bounds bounded sourceCorrect
  · exact BindingWork.expected_work_polynomial_bound ajtai program relation running fresh originalFirstPhase
      publicCheck continuation call sourceProgram bounds bounded accessBound accessBounded contexts source.2.2.1
      securityParameter _ primitivePolynomial accessPolynomial source.2.2.2 primitivePPT accessPPT

end NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction
