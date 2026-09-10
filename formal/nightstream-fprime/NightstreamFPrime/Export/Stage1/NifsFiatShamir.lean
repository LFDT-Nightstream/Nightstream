import NightstreamFPrime.Export.Stage1.PiCCSStoredSourceProbability
import NightstreamFPrime.Export.Stage1.PiRLCExtractionPrimitives
import NightstreamFPrime.Lifecycle.Nifs.FiatShamirTransfer

/-!
Selected source-return and declared expected-work consumer. The input
projections, fixed Ajtai key, primitive program and stored checked return
are the existing production owners. Their value correctness is proved here
by their existing contracts, with no free checker or primitive premise.

FiatShamirModel remains an explicit, unapproved game-transfer hypothesis.
The real event is the actual verifier's acceptance with witnesses for its
exact children. Its law and the interactive context marginal are shared.
No adversary translation, query bound, or FS model instance is constructed.
Low-norm invertibility, declared clock bounds, preparation/call refinement,
and moment bounds remain separate hypotheses. The work conclusion is for
the existing prepared extractor clock, not machine or simulator time.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsFiatShamir

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

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  (law : PMF (Context × Option (FiatShamirTransfer.RealOutput relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    (FiatShamirTransfer.contextLaw relation law)
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context)))))
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    law originalFirstPhase abortTape provider g deltaFS Q)
  (scalarSubClock : RingF → RingF → Nat)
  (inverseAdapterClock : RingF → Nat)
  (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
  (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
  (checkClock : Context → PiCCSStoredSourceProbability.CheckClock)
  (accessClock : Context → PiCCSStoredSourceProbability.AccessClock)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
    (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds)

include model lowNorm bounded in
/-- The selected stored source event has the conditional FS/MSIS lower bound,
and the same prepared reduction has the supplied polynomial declared-work
bound. Both local Correct obligations are discharged by the selected owners.
The MSIS term is the existing reduction's success mass at this fixed key;
no numerical hardness estimate or model approval is asserted. -/
theorem finishValue_probability_and_expected_work {SetupTape : Type*}
    (setupTapes : PMF SetupTape) (prepare : SetupTape → Result Context)
    (preparedContexts : FiatShamirTransfer.contextLaw relation law =
      ContextPreparation.contexts setupTapes prepare)
    (preparationSummable : Summable fun tape => (setupTapes tape).toReal * (prepare tape).work)
    (call : Context → CubePoint K productionShape.cubeVariables → K →
      CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
    (callCorrect : ∀ context alpha gamma point,
      (call context alpha gamma point).value = InteractivePrefix.run
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck
            (fun context => PiCCSInputCheck.running (inputs context))) context) alpha gamma point)
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
      abortTape provider
    let base := InteractiveWork.baseClock relation productionAjtaiKey running fresh
      continuation call program sourceProgram
    let bindingTotal := BindingWork.totalClock productionAjtaiKey program relation running fresh
      originalFirstPhase (SupportedExtraction.publicCheck running) continuation call sourceProgram
    let total := ContextPreparation.clock prepare
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
    (∑' tape, (setupTapes tape).toReal * (prepare tape).work) ≤
      preparationPolynomial.eval (securityParameter : ℝ) →
    (g Q (FiatShamirTransfer.realSuccessProbability relation productionAjtaiKey running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation productionAjtaiKey -
      Real.sqrt ((∑' tape, (setupTapes tape).toReal * BindingProbability.localSuccessProbability
        productionAjtaiKey program (sourceProgram (prepare tape).value).access relation running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation (prepare tape).value) *
          PaperProfile.arity.total + IndependentExecution.testError productionShape 9) ≤
      PaperCompositionProbability.eventProbability contexts
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        (InteractiveComposition.suffixLaw relation productionAjtaiKey running fresh continuation)
        (InteractiveComposition.consume relation productionAjtaiKey program)
        (fun context outcome => SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
          (PiCCSStoredWitnessCheck.statement (inputs context))
          (PiCCSStoredWitnessCheck.finishValue (inputs context)
            (PiCCSStoredSourceProbability.storeOutcome outcome)))) ∧
    Summable (fun tape => (setupTapes tape).toReal * total tape) ∧
    (∑' tape, (setupTapes tape).toReal * total tape) ≤
      (preparationPolynomial + Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C (PiCCSStoredWitnessCheck.carrier.carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 13).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT primitivePPT accessPPT preparationPPT
  have checked := FiatShamirTransfer.prepared_probability_and_expected_work
    relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    law originalFirstPhase abortTape provider g deltaFS Q model
    (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock)
    (fun context => PiCCSStoredSourceProbability.sourceProgram
      (inputs context) (checkClock context) (accessClock context))
    lowNorm
    (PiRLCExtractionPrimitives.program_correct scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock)
    bounds bounded
    (fun context => PiCCSStoredSourceProbability.sourceProgram_correct
      (inputs context) (checkClock context) (accessClock context))
    setupTapes prepare preparedContexts preparationSummable call callCorrect accessBound accessBounded
    securityParameter preparationPolynomial basePolynomial primitivePolynomial accessPolynomial
    baseSummable basePPT primitivePPT accessPPT preparationPPT
  have storedEvent := PiCCSStoredSourceProbability.returnedSourceProbability_eq_finishValue
    inputs (FiatShamirTransfer.contextLaw relation law) originalFirstPhase
    (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context)))
    (SupportedContinuation.extension relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context))
      (FiatShamirTransfer.contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
      abortTape provider)
    (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) checkClock accessClock
  exact ⟨checked.1.trans (le_of_eq storedEvent), checked.2⟩

end NightstreamFPrime.Export.Stage1.NifsFiatShamir
