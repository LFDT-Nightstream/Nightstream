import NightstreamFPrime.Export.Stage1.NifsFiatShamir
import NightstreamFPrime.Export.Stage1.NifsExtractionProvider
import NightstreamFPrime.Export.Stage1.HyperNovaSourceLaw

/-!
Selected NIFS source return under its constructed output PMF. The
provider uses the concrete suffix and parent checks; the prefix call runs the
existing selected public checker. Preparation returns its supplied context.
Thus no provider correctness, call-value equality or preparation-law equality
is a premise of the final consumer.

The external FS transfer, low-norm hypothesis, caller clocks and their bounds
and moments remain explicit. Identity preparation does not implement an
adversary translation or an efficient sampler for the context law. Its clock
is the supplied experiment label, not a claim about arbitrary preprocessing.
The conclusion concerns declared expected work, not machine runtime.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsClosure

open scoped BigOperators

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier CheckedWitnessExtraction
open _root_.NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Lifecycle.Nifs
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)

/-- The context has already been supplied by this experiment's PMF. The
caller labels this preparation with its declared context-dependent work. -/
def prepare {Context : Type*} (clock : Context → Nat) (context : Context) : Result Context :=
  ⟨context, clock context⟩

/-- Execute the given checked causal prefix, retaining its exact receipt.
The supplied clock covers this whole call in the declared experiment. -/
def prefixCall {Context State : Type*}
    (firstPhase : Context → InteractivePrefix.Prover State productionShape 9)
    (clock : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Nat)
    (context : Context) (alpha : CubePoint K productionShape.cubeVariables)
    (gamma : K) (point : CubePoint K productionShape.cubeVariables) :
    Result (Option (Probe K productionShape × State)) :=
  ⟨InteractivePrefix.run (firstPhase context) alpha gamma point,
    clock context alpha gamma point⟩

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  (law : PMF (Context × Option (FiatShamirTransfer.RealOutput relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (tapes : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      context coins output state → PMF Tape)
  (rawCall : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      context coins output state →
        PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) NifsExtractionProvider.rlc)
  (suffixCheckClock : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      context coins output state → NifsExtractionProvider.CheckClock)
  (storageClock : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      context coins output state → NifsExtractionProvider.StorageClock)
  (parentClock : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      context coins output state → NifsExtractionProvider.ParentClock)
  (storageBound : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      context coins output state → Nat)
  (storageBounded : ∀ context coins output state support assignments,
    storageClock context coins output state support assignments ≤
      storageBound context coins output state support)
  (suffixSummable : ∀ context coins output state support vector, Summable fun tape =>
    (tapes context coins output state support tape).toReal *
      (PaperWeakOracle.baseWork NifsExtractionProvider.rlc
        (NifsExtractionProvider.suffixProgram
          (NifsExtractionProvider.batchAt inputs context coins output)
          (suffixCheckClock context coins output state support)
          (storageClock context coins output state support))
        (rawCall context coins output state support) vector tape : ℝ))
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    law originalFirstPhase abortTape
    (NifsExtractionProvider.provider inputs (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      tapes rawCall suffixCheckClock
      storageClock parentClock storageBound storageBounded suffixSummable) g deltaFS Q)
  (scalarSubClock : RingF → RingF → Nat)
  (inverseAdapterClock : RingF → Nat)
  (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
  (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
  (checkClock : Context → PiCCSStoredSourceProbability.CheckClock)
  (accessClock : Context → PiCCSStoredSourceProbability.AccessClock)
  (preparationClock : Context → Nat)
  (prefixClock : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Nat)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
    (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds)

include model lowNorm bounded in
/-- Probability-only consumer on the actual selected source PMF. The provider
and all value checks are constructed here. Averaged work hypotheses are not
needed for this inequality; the algorithm's per-call admissibility and the
exact FS model remain explicit. This is the per-visit security interface. -/
theorem source_probability_bound :
    let running := fun context => PiCCSInputCheck.running (inputs context)
    let fresh := fun context => PiCCSInputCheck.fresh (inputs context)
    let contexts := FiatShamirTransfer.contextLaw relation law
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let sourceProgram := fun context => PiCCSStoredSourceProbability.sourceProgram
      (inputs context) (checkClock context) (accessClock context)
    let continuation := SupportedContinuation.extension relation productionAjtaiKey running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape
      (NifsExtractionProvider.provider inputs contexts
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        tapes rawCall suffixCheckClock storageClock parentClock storageBound storageBounded suffixSummable)
    g Q (FiatShamirTransfer.realSuccessProbability relation productionAjtaiKey running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation productionAjtaiKey -
      Real.sqrt (BindingProbability.successProbability productionAjtaiKey program relation running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation
        (fun context => (sourceProgram context).access) contexts * PaperProfile.arity.total +
          IndependentExecution.testError productionShape 9) ≤
      ((HyperNovaSourceLaw.law inputs contexts originalFirstPhase continuation program).toOuterMeasure
        {sample | SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
          (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2}).toReal := by
  dsimp only
  let contexts := FiatShamirTransfer.contextLaw relation law
  let running := fun context => PiCCSInputCheck.running (inputs context)
  let fresh := fun context => PiCCSInputCheck.fresh (inputs context)
  let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let provider := NifsExtractionProvider.provider inputs contexts
    (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
    tapes rawCall suffixCheckClock storageClock parentClock storageBound storageBounded suffixSummable
  let continuation := SupportedContinuation.extension relation productionAjtaiKey running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
    abortTape provider
  have lower := FiatShamirTransfer.returned_source_bound_with_msis relation productionAjtaiKey
    running fresh law originalFirstPhase abortTape provider g deltaFS Q model program
    (fun context => PiCCSStoredSourceProbability.sourceProgram (inputs context)
      (checkClock context) (accessClock context)) lowNorm
    (PiRLCExtractionPrimitives.program_correct scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds bounded
    (fun context => PiCCSStoredSourceProbability.sourceProgram_correct
      (inputs context) (checkClock context) (accessClock context))
  have storedEvent := PiCCSStoredSourceProbability.returnedSourceProbability_eq_finishValue
    inputs contexts originalFirstPhase (SupportedExtraction.publicCheck running)
    continuation program checkClock accessClock
  exact lower.trans (le_of_eq (storedEvent.trans
    (HyperNovaSourceLaw.source_event_mass_eq inputs contexts originalFirstPhase continuation program).symm))

include model lowNorm bounded in
/-- The actual stored source event and the prepared reduction use the same
constructed provider and exact checked prefix. The remaining premises are
external transfer/invertibility and explicit declared-clock moment bounds.
The constructed source PMF realizes the exact existing sequential event law.
No efficient translation, runtime bound, or numerical MSIS estimate follows. -/
theorem finishValue_probability_and_expected_work
    (preparationSummable : Summable fun context =>
      (FiatShamirTransfer.contextLaw relation law context).toReal * preparationClock context)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded
      (PiCCSStoredSourceProbability.sourceProgram (inputs context)
        (checkClock context) (accessClock context)).access accessBound)
    (securityParameter : Nat)
    (preparationPolynomial basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let running := fun context => PiCCSInputCheck.running (inputs context)
    let fresh := fun context => PiCCSInputCheck.fresh (inputs context)
    let contexts := FiatShamirTransfer.contextLaw relation law
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let sourceProgram := fun context => PiCCSStoredSourceProbability.sourceProgram
      (inputs context) (checkClock context) (accessClock context)
    let continuation := SupportedContinuation.extension relation productionAjtaiKey running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape
      (NifsExtractionProvider.provider inputs contexts
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        tapes rawCall suffixCheckClock
        storageClock parentClock storageBound storageBounded suffixSummable)
    let call := prefixCall
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)) prefixClock
    let base := InteractiveWork.baseClock relation productionAjtaiKey running fresh
      continuation call program sourceProgram
    let bindingTotal := BindingWork.totalClock productionAjtaiKey program relation running fresh
      originalFirstPhase (SupportedExtraction.publicCheck running) continuation call sourceProgram
    let total := ContextPreparation.clock (prepare preparationClock)
      (fun context => StrongProbability.verifierMean (bindingTotal context))
    let sourcePolynomial := Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
      Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
      Polynomial.C (productionShape.freshCount : ℝ) *
        (Polynomial.C (WitnessProjection.privateWidth PiCCSStoredWitnessCheck.carrier : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
      Polynomial.C (productionShape.runningCount : ℝ) *
        (Polynomial.C (PiCCSStoredWitnessCheck.carrier.carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) + Polynomial.C 13
    Summable (fun context => (contexts context).toReal *
      StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean contexts base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∑' context, (contexts context).toReal * preparationClock context) ≤
      preparationPolynomial.eval (securityParameter : ℝ) →
    (g Q (FiatShamirTransfer.realSuccessProbability relation productionAjtaiKey running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation productionAjtaiKey -
      Real.sqrt ((∑' context, (contexts context).toReal * BindingProbability.localSuccessProbability
        productionAjtaiKey program (sourceProgram context).access relation running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation context) *
          PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      ((HyperNovaSourceLaw.law inputs contexts originalFirstPhase continuation program).toOuterMeasure
        {sample | SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
          (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2}).toReal) ∧
    Summable (fun context => (contexts context).toReal * total context) ∧
    (∑' context, (contexts context).toReal * total context) ≤
      (preparationPolynomial + Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C (PiCCSStoredWitnessCheck.carrier.carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 13).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT preparationPPT
  have preparedContexts : FiatShamirTransfer.contextLaw relation law =
      ContextPreparation.contexts (FiatShamirTransfer.contextLaw relation law) (prepare preparationClock) := by
    simpa only [ContextPreparation.contexts, prepare] using
      (PMF.map_id (FiatShamirTransfer.contextLaw relation law)).symm
  have checked := NifsFiatShamir.finishValue_probability_and_expected_work
    inputs law originalFirstPhase abortTape
    (NifsExtractionProvider.provider inputs (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      tapes rawCall suffixCheckClock
      storageClock parentClock storageBound storageBounded suffixSummable) g deltaFS Q model
    scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock checkClock accessClock
    lowNorm bounds bounded (FiatShamirTransfer.contextLaw relation law) (prepare preparationClock) preparedContexts
    preparationSummable
    (prefixCall (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context)))) prefixClock)
    (fun _ _ _ _ => rfl) accessBound accessBounded securityParameter
    preparationPolynomial basePolynomial primitivePolynomial accessPolynomial
    baseSummable basePPT primitivePPT accessPPT preparationPPT
  simpa only [prepare, HyperNovaSourceLaw.source_event_mass_eq] using checked

end NightstreamFPrime.Export.Stage1.NifsClosure
