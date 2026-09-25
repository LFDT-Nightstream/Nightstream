import NightstreamFPrime.Export.Stage1.SecurityInstance
import NightstreamFPrime.Export.Stage1.NifsExtractionProvider
import NightstreamFPrime.Export.Stage1.HyperNovaSourceLaw
import NightstreamFPrime.Export.Stage1.NifsClosure

/-!
The selected checked continuation can be fixed before choosing a context law.
Restricting its same raw calls, tapes and clocks to supported receipts gives
the provider consumed by NifsClosure. Its abort extension has exactly the
same source-result law. No equality of experiments is left as a premise.

The supplied total raw-call family has the existing finite per-call moments
needed to construct its algorithms. This module supplies no averaged work
bound, efficient context sampler, or Fiat--Shamir model for a new law.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsProviderLaw

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nifs
open PiRLC.PaperForkExtractionWork (Primitives)
open PiRLC.CoordinateForkLaw (Challenge)

variable (inst : SecurityInstance)

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  (tapes : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
  (rawCall : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State →
      PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) (NifsExtractionProvider.rlc inst))
  (checkClock : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock inst)
  (storageClock : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock inst)
  (parentClock : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock inst)
  (storageBound : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → Nat)
  (storageBounded : ∀ context coins output state assignments,
    storageClock context coins output state assignments ≤ storageBound context coins output state)
  (baseSummable : ∀ context coins output state vector, Summable fun tape =>
    (tapes context coins output state tape).toReal *
      (PaperWeakOracle.baseWork (NifsExtractionProvider.rlc inst)
        (NifsExtractionProvider.suffixProgram inst
            (NifsExtractionProvider.batchAt inst inputs context coins output)
          (checkClock context coins output state) (storageClock context coins output state))
        (rawCall context coins output state) vector tape : ℝ))

/-- Fix the operational continuation using the selected suffix and parent
checks. Its data and per-call moment proofs do not depend on a visited law. -/
def continuation (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State) :
    WeakExtraction.Continuation Tape inst.relation inst.ajtai
      (inst.running (inputs context)) (inst.fresh (inputs context)) coins output :=
  (NifsExtractionProvider.continuationAt inst) inputs context coins output
    (tapes context coins output state) (rawCall context coins output state)
    (checkClock context coins output state) (storageClock context coins output state)
    (parentClock context coins output state) (storageBound context coins output state)
    (storageBounded context coins output state) (baseSummable context coins output state)

/-- Restrict the same fixed data to receipts supported under this law.
Only the proof argument changes; it is ignored by every data accessor. -/
def supportedProvider (contexts : PMF Context)
    (firstPhase : Context → InteractivePrefix.Prover State productionShape 9) :
    SupportedContinuation.Provider Tape inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) contexts firstPhase :=
  (NifsExtractionProvider.provider inst) inputs contexts firstPhase
    (fun context coins output state _ => tapes context coins output state)
    (fun context coins output state _ => rawCall context coins output state)
    (fun context coins output state _ => checkClock context coins output state)
    (fun context coins output state _ => storageClock context coins output state)
    (fun context coins output state _ => parentClock context coins output state)
    (fun context coins output state _ => storageBound context coins output state)
    (fun context coins output state _ => storageBounded context coins output state)
    (fun context coins output state _ => baseSummable context coins output state)

variable [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]

/-- The fixed operational source law equals the actual supported-provider
extension used by NifsClosure under this exact context law. The proof derives
receipt support from positive mass and preserves the same call, tape, checks
and clocks. It neither conditions the law nor assumes a provider agreement. -/
theorem source_law_eq_supported_extension
    (contexts : PMF Context)
    (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape)
    (primitives : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := inst.logicalWidth)
        (publicFits := inst.publicFits))) :
    let running := fun context => inst.running (inputs context)
    let fresh := fun context => inst.fresh (inputs context)
    let checked := InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)
    let provider := supportedProvider inst inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable contexts checked
    (HyperNovaSourceLaw.law inst) inputs contexts originalFirstPhase
        (continuation inst inputs tapes rawCall checkClock storageClock parentClock storageBound
          storageBounded baseSummable) primitives =
      (HyperNovaSourceLaw.law inst) inputs contexts originalFirstPhase
        (SupportedContinuation.extension inst.relation inst.ajtai running fresh contexts checked
          abortTape provider) primitives := by
  dsimp only
  apply (HyperNovaSourceLaw.law_eq_of_continuation_eq_on_return inst)
  intro context supported alpha gamma point receipt returned
  have positive : 0 < (contexts context).toReal :=
    ENNReal.toReal_pos ((contexts.mem_support_iff context).mp supported) (contexts.apply_ne_top context)
  rw [SupportedContinuation.extension_eq_on_return inst.relation inst.ajtai
    (fun context => inst.running (inputs context))
    (fun context => inst.fresh (inputs context)) contexts
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
    abortTape _ context positive alpha gamma point receipt returned]
  rfl

/-- Apply the additive retry bound to the same fixed operational source
law through its exact supported-provider equality. -/
theorem source_probability_linear_bound
    (law : PMF (Context × Option (FiatShamirTransfer.RealOutput inst.relation)))
    (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
    (model : WideFiatShamir.FiatShamirModel inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) law originalFirstPhase abortTape
      (supportedProvider inst inputs tapes rawCall checkClock storageClock parentClock storageBound
        storageBounded baseSummable (FiatShamirTransfer.contextLaw inst.relation law)
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck (fun context => inst.running (inputs context)))))
      g deltaFS Q)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment inst →
        PiRLCExtractionPrimitives.Assignment inst → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment inst → Nat)
    (sourceCheckClock : Context → PiCCSStoredSourceProbability.CheckClock inst)
    (accessClock : Context → PiCCSStoredSourceProbability.AccessClock inst)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra inst.ajtai).ring
      (PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds) :
    let running := fun context => inst.running (inputs context)
    let fresh := fun context => inst.fresh (inputs context)
    let contexts := FiatShamirTransfer.contextLaw inst.relation law
    let program := PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let checked := InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)
    let provider := supportedProvider inst inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable contexts checked
    let extended := SupportedContinuation.extension inst.relation inst.ajtai running fresh contexts checked
      abortTape provider
    g Q (WideFiatShamir.realSuccessProbability inst.relation inst.ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss inst.relation inst.ajtai -
      IndependentExecution.testError productionShape 9 -
      AdaptiveBindingProbability.successProbability inst.relation inst.ajtai program running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) extended
        (fun context => PiCCSStoredSourceProbability.sourceProgram inst (inputs context)
          (sourceCheckClock context) (accessClock context)) contexts * PaperProfile.arity.total ≤
      ((HyperNovaSourceLaw.law inst inputs contexts originalFirstPhase
        (continuation inst inputs tapes rawCall checkClock storageClock parentClock storageBound
          storageBounded baseSummable) program).toOuterMeasure
        {sample | CheckedWitnessExtraction.SourceReturned (PiCCSStoredWitnessCheck.commit inst)
            productionGlobalParams
          (PiCCSStoredWitnessCheck.statement inst (inputs sample.1)) sample.2}).toReal := by
  dsimp only
  rw [source_law_eq_supported_extension inst inputs tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded baseSummable (FiatShamirTransfer.contextLaw inst.relation law)
    originalFirstPhase abortTape]
  exact NifsClosure.source_probability_linear_bound inst inputs law originalFirstPhase abortTape
    (fun context coins output state _ => tapes context coins output state)
    (fun context coins output state _ => rawCall context coins output state)
    (fun context coins output state _ => checkClock context coins output state)
    (fun context coins output state _ => storageClock context coins output state)
    (fun context coins output state _ => parentClock context coins output state)
    (fun context coins output state _ => storageBound context coins output state)
    (fun context coins output state _ => storageBounded context coins output state)
    (fun context coins output state _ => baseSummable context coins output state)
    g deltaFS Q model scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock
    sourceCheckClock accessClock lowNorm bounds bounded

end NightstreamFPrime.Export.Stage1.NifsProviderLaw
