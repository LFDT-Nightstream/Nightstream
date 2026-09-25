import NightstreamFPrime.Export.Stage1.SecurityInstance
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

variable (inst : SecurityInstance)

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
  [Fintype (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  (law : PMF (Context × Option (FiatShamirTransfer.RealOutput inst.relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (tapes : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      context coins output state → PMF Tape)
  (rawCall : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      context coins output state →
        PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) (NifsExtractionProvider.rlc inst))
  (suffixCheckClock : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      context coins output state → NifsExtractionProvider.CheckClock inst)
  (storageClock : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      context coins output state → NifsExtractionProvider.StorageClock inst)
  (parentClock : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      context coins output state → NifsExtractionProvider.ParentClock inst)
  (storageBound : ∀ context coins output state,
    SupportedContinuation.Supported (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      context coins output state → Nat)
  (storageBounded : ∀ context coins output state support assignments,
    storageClock context coins output state support assignments ≤
      storageBound context coins output state support)
  (suffixSummable : ∀ context coins output state support vector, Summable fun tape =>
    (tapes context coins output state support tape).toReal *
      (PaperWeakOracle.baseWork (NifsExtractionProvider.rlc inst)
        (NifsExtractionProvider.suffixProgram inst
          (NifsExtractionProvider.batchAt inst inputs context coins output)
          (suffixCheckClock context coins output state support)
          (storageClock context coins output state support))
        (rawCall context coins output state support) vector tape : ℝ))
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : WideFiatShamir.FiatShamirModel inst.relation inst.ajtai
    (fun context => inst.running (inputs context))
    (fun context => inst.fresh (inputs context))
    law originalFirstPhase abortTape
    (NifsExtractionProvider.provider inst inputs (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      tapes rawCall suffixCheckClock
      storageClock parentClock storageBound storageBounded suffixSummable) g deltaFS Q)
  (scalarSubClock : RingF → RingF → Nat)
  (inverseAdapterClock : RingF → Nat)
  (assignmentSubClock : PiRLCExtractionPrimitives.Assignment inst →
      PiRLCExtractionPrimitives.Assignment inst → Nat)
  (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment inst → Nat)
  (checkClock : Context → PiCCSStoredSourceProbability.CheckClock inst)
  (accessClock : Context → PiCCSStoredSourceProbability.AccessClock inst)
  (preparationClock : Context → Nat)
  (prefixClock : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Nat)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra inst.ajtai).ring
    (PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds)

include model lowNorm bounded in
/-- The same selected source PMF satisfies the additive retry bound. The
MSIS term measures the actual adaptive reduction with this source program. -/
theorem source_probability_linear_bound :
    let running := fun context => inst.running (inputs context)
    let fresh := fun context => inst.fresh (inputs context)
    let contexts := FiatShamirTransfer.contextLaw inst.relation law
    let program := PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let sourceProgram := fun context => PiCCSStoredSourceProbability.sourceProgram inst
      (inputs context) (checkClock context) (accessClock context)
    let continuation := SupportedContinuation.extension inst.relation inst.ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape
      (NifsExtractionProvider.provider inst inputs contexts
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        tapes rawCall suffixCheckClock storageClock parentClock storageBound storageBounded suffixSummable)
    g Q (WideFiatShamir.realSuccessProbability inst.relation inst.ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss inst.relation inst.ajtai -
      IndependentExecution.testError productionShape 9 -
      AdaptiveBindingProbability.successProbability inst.relation inst.ajtai program running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation sourceProgram contexts *
          PaperProfile.arity.total ≤
      ((HyperNovaSourceLaw.law inst inputs contexts originalFirstPhase continuation program).toOuterMeasure
        {sample | SourceReturned (PiCCSStoredWitnessCheck.commit inst) productionGlobalParams
          (PiCCSStoredWitnessCheck.statement inst (inputs sample.1)) sample.2}).toReal := by
  dsimp only
  let contexts := FiatShamirTransfer.contextLaw inst.relation law
  let running := fun context => inst.running (inputs context)
  let fresh := fun context => inst.fresh (inputs context)
  let program := PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let provider := NifsExtractionProvider.provider inst inputs contexts
    (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
    tapes rawCall suffixCheckClock storageClock parentClock storageBound storageBounded suffixSummable
  let continuation := SupportedContinuation.extension inst.relation inst.ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
    abortTape provider
  have lower := WideFiatShamir.returned_source_bound_with_adaptive_msis inst.relation inst.ajtai
    running fresh law originalFirstPhase abortTape provider g deltaFS Q model program
    (fun context => PiCCSStoredSourceProbability.sourceProgram inst (inputs context)
      (checkClock context) (accessClock context)) lowNorm
    (PiRLCExtractionPrimitives.program_correct inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds bounded
    (fun context => PiCCSStoredSourceProbability.sourceProgram_correct inst
      (inputs context) (checkClock context) (accessClock context))
  have storedEvent := PiCCSStoredSourceProbability.returnedSourceProbability_eq_finishValue inst
    inputs contexts originalFirstPhase (SupportedExtraction.publicCheck running)
    continuation program checkClock accessClock
  exact lower.trans (le_of_eq (storedEvent.trans
    (HyperNovaSourceLaw.source_event_mass_eq inst inputs contexts originalFirstPhase continuation
        program).symm))

include model lowNorm bounded in
/-- The actual stored source event and the prepared reduction use the same
constructed provider and exact checked prefix. The remaining premises are
external transfer/invertibility and explicit declared-clock moment bounds.
The constructed source PMF realizes the exact existing sequential event law.
No efficient translation, runtime bound, or numerical MSIS estimate follows. -/
theorem finishValue_probability_and_expected_work
    (preparationSummable : Summable fun context =>
      (FiatShamirTransfer.contextLaw inst.relation law context).toReal * preparationClock context)
    (accessBound : Nat)
    (accessBounded : ∀ context, CostedWitnessProjection.Bounded
      (PiCCSStoredSourceProbability.sourceProgram inst (inputs context)
        (checkClock context) (accessClock context)).access accessBound)
    (securityParameter : Nat)
    (preparationPolynomial basePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ) :
    let running := fun context => inst.running (inputs context)
    let fresh := fun context => inst.fresh (inputs context)
    let contexts := FiatShamirTransfer.contextLaw inst.relation law
    let program := PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let sourceProgram := fun context => PiCCSStoredSourceProbability.sourceProgram inst
      (inputs context) (checkClock context) (accessClock context)
    let continuation := SupportedContinuation.extension inst.relation inst.ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape
      (NifsExtractionProvider.provider inst inputs contexts
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        tapes rawCall suffixCheckClock
        storageClock parentClock storageBound storageBounded suffixSummable)
    let call := prefixCall
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)) prefixClock
    let base := InteractiveWork.baseClock inst.relation inst.ajtai running fresh
      continuation call program sourceProgram
    let bindingTotal := BindingWork.totalClock inst.ajtai program inst.relation running fresh
      originalFirstPhase (SupportedExtraction.publicCheck running) continuation call sourceProgram
    let total := ContextPreparation.clock (prepare preparationClock)
      (fun context => StrongProbability.verifierMean (bindingTotal context))
    let sourcePolynomial := Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * basePolynomial +
      Polynomial.C (PaperProfile.arity.total : ℝ) * (primitivePolynomial + Polynomial.C 3) +
      Polynomial.C (productionShape.freshCount : ℝ) *
        (Polynomial.C (WitnessProjection.privateWidth (PiCCSStoredWitnessCheck.carrier inst) : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) +
      Polynomial.C (productionShape.runningCount : ℝ) *
        (Polynomial.C ((PiCCSStoredWitnessCheck.carrier inst).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 9) + Polynomial.C 13
    Summable (fun context => (contexts context).toReal *
      StrongProbability.verifierMean (base context)) →
    StrongProbability.clockMean contexts base ≤ basePolynomial.eval (securityParameter : ℝ) →
    (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ) →
    (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ) →
    (∑' context, (contexts context).toReal * preparationClock context) ≤
      preparationPolynomial.eval (securityParameter : ℝ) →
    (g Q (WideFiatShamir.realSuccessProbability inst.relation inst.ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss inst.relation inst.ajtai -
      Real.sqrt ((∑' context, (contexts context).toReal * BindingProbability.localSuccessProbability
        inst.ajtai program (sourceProgram context).access inst.relation running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation context) *
          PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      ((HyperNovaSourceLaw.law inst inputs contexts originalFirstPhase continuation program).toOuterMeasure
        {sample | SourceReturned (PiCCSStoredWitnessCheck.commit inst) productionGlobalParams
          (PiCCSStoredWitnessCheck.statement inst (inputs sample.1)) sample.2}).toReal) ∧
    Summable (fun context => (contexts context).toReal * total context) ∧
    (∑' context, (contexts context).toReal * total context) ≤
      (preparationPolynomial + Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C ((PiCCSStoredWitnessCheck.carrier inst).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 13).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT preparationPPT
  have preparedContexts : FiatShamirTransfer.contextLaw inst.relation law =
      ContextPreparation.contexts (FiatShamirTransfer.contextLaw inst.relation law)
          (prepare preparationClock) := by
    simpa only [ContextPreparation.contexts, prepare] using!
      (PMF.map_id (FiatShamirTransfer.contextLaw inst.relation law)).symm
  have checked := NifsFiatShamir.finishValue_probability_and_expected_work inst
    inputs law originalFirstPhase abortTape
    (NifsExtractionProvider.provider inst inputs (FiatShamirTransfer.contextLaw inst.relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
      tapes rawCall suffixCheckClock
      storageClock parentClock storageBound storageBounded suffixSummable) g deltaFS Q model
    scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock checkClock accessClock
    lowNorm bounds bounded (FiatShamirTransfer.contextLaw inst.relation law)
        (prepare preparationClock) preparedContexts
    preparationSummable
    (prefixCall (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => inst.running (inputs context)))) prefixClock)
    (fun _ _ _ _ => rfl) accessBound accessBounded securityParameter
    preparationPolynomial basePolynomial primitivePolynomial accessPolynomial
    baseSummable basePPT primitivePPT accessPPT preparationPPT
  simpa only [prepare, (HyperNovaSourceLaw.source_event_mass_eq inst)] using checked

end NightstreamFPrime.Export.Stage1.NifsClosure
