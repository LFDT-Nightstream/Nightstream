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
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  (tapes : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
  (rawCall : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State →
      PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) NifsExtractionProvider.rlc)
  (checkClock : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock)
  (storageClock : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock)
  (parentClock : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock)
  (storageBound : Context → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → Nat)
  (storageBounded : ∀ context coins output state assignments,
    storageClock context coins output state assignments ≤ storageBound context coins output state)
  (baseSummable : ∀ context coins output state vector, Summable fun tape =>
    (tapes context coins output state tape).toReal *
      (PaperWeakOracle.baseWork NifsExtractionProvider.rlc
        (NifsExtractionProvider.suffixProgram (NifsExtractionProvider.batchAt inputs context coins output)
          (checkClock context coins output state) (storageClock context coins output state))
        (rawCall context coins output state) vector tape : ℝ))

/-- Fix the operational continuation using the selected suffix and parent
checks. Its data and per-call moment proofs do not depend on a visited law. -/
def continuation (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State) :
    WeakExtraction.Continuation Tape relation productionAjtaiKey
      (PiCCSInputCheck.running (inputs context)) (PiCCSInputCheck.fresh (inputs context)) coins output :=
  NifsExtractionProvider.continuationAt inputs context coins output
    (tapes context coins output state) (rawCall context coins output state)
    (checkClock context coins output state) (storageClock context coins output state)
    (parentClock context coins output state) (storageBound context coins output state)
    (storageBounded context coins output state) (baseSummable context coins output state)

/-- Restrict the same fixed data to receipts supported under this law.
Only the proof argument changes; it is ignored by every data accessor. -/
def supportedProvider (contexts : PMF Context)
    (firstPhase : Context → InteractivePrefix.Prover State productionShape 9) :
    SupportedContinuation.Provider Tape relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) contexts firstPhase :=
  NifsExtractionProvider.provider inputs contexts firstPhase
    (fun context coins output state _ => tapes context coins output state)
    (fun context coins output state _ => rawCall context coins output state)
    (fun context coins output state _ => checkClock context coins output state)
    (fun context coins output state _ => storageClock context coins output state)
    (fun context coins output state _ => parentClock context coins output state)
    (fun context coins output state _ => storageBound context coins output state)
    (fun context coins output state _ => storageBounded context coins output state)
    (fun context coins output state _ => baseSummable context coins output state)

variable [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]

/-- The fixed operational source law equals the actual supported-provider
extension used by NifsClosure under this exact context law. The proof derives
receipt support from positive mass and preserves the same call, tape, checks
and clocks. It neither conditions the law nor assumes a provider agreement. -/
theorem source_law_eq_supported_extension
    (contexts : PMF Context)
    (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape)
    (primitives : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := PiDECInputCheck.logicalWidth)
        (publicFits := PiDECInputCheck.publicFits))) :
    let running := fun context => PiCCSInputCheck.running (inputs context)
    let fresh := fun context => PiCCSInputCheck.fresh (inputs context)
    let checked := InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)
    let provider := supportedProvider inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable contexts checked
    HyperNovaSourceLaw.law inputs contexts originalFirstPhase
        (continuation inputs tapes rawCall checkClock storageClock parentClock storageBound
          storageBounded baseSummable) primitives =
      HyperNovaSourceLaw.law inputs contexts originalFirstPhase
        (SupportedContinuation.extension relation productionAjtaiKey running fresh contexts checked
          abortTape provider) primitives := by
  dsimp only
  apply HyperNovaSourceLaw.law_eq_of_continuation_eq_on_return
  intro context supported alpha gamma point receipt returned
  have positive : 0 < (contexts context).toReal :=
    ENNReal.toReal_pos ((contexts.mem_support_iff context).mp supported) (contexts.apply_ne_top context)
  rw [SupportedContinuation.extension_eq_on_return relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context)) contexts
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
    abortTape _ context positive alpha gamma point receipt returned]
  rfl

/-- Apply the existing selected FS/MSIS bound to the fixed operational
source law. The model is for exactly this context law and its restricted
provider. The MSIS term names that same reduction; no numerical hardness or
transfer bound is inferred, and no source-law equality is assumed. -/
theorem source_probability_bound
    (law : PMF (Context × Option (FiatShamirTransfer.RealOutput relation)))
    (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
    (model : FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) law originalFirstPhase abortTape
      (supportedProvider inputs tapes rawCall checkClock storageClock parentClock storageBound
        storageBounded baseSummable (FiatShamirTransfer.contextLaw relation law)
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context)))))
      g deltaFS Q)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
    (sourceCheckClock : Context → PiCCSStoredSourceProbability.CheckClock)
    (accessClock : Context → PiCCSStoredSourceProbability.AccessClock)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds) :
    let running := fun context => PiCCSInputCheck.running (inputs context)
    let fresh := fun context => PiCCSInputCheck.fresh (inputs context)
    let contexts := FiatShamirTransfer.contextLaw relation law
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let checked := InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)
    let provider := supportedProvider inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable contexts checked
    let extended := SupportedContinuation.extension relation productionAjtaiKey running fresh contexts checked
      abortTape provider
    g Q (FiatShamirTransfer.realSuccessProbability relation productionAjtaiKey running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation productionAjtaiKey -
      Real.sqrt (BindingProbability.successProbability productionAjtaiKey program relation running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) extended
        (fun context => (PiCCSStoredSourceProbability.sourceProgram (inputs context)
          (sourceCheckClock context) (accessClock context)).access) contexts * PaperProfile.arity.total +
          IndependentExecution.testError productionShape 9) ≤
      ((HyperNovaSourceLaw.law inputs contexts originalFirstPhase
        (continuation inputs tapes rawCall checkClock storageClock parentClock storageBound
          storageBounded baseSummable) program).toOuterMeasure
        {sample | CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
          (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2}).toReal := by
  dsimp only
  rw [source_law_eq_supported_extension inputs tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded baseSummable (FiatShamirTransfer.contextLaw relation law)
    originalFirstPhase abortTape]
  exact NifsClosure.source_probability_bound inputs law originalFirstPhase abortTape
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
