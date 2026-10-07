import NightstreamFPrime.Export.Stage1.NifsExtractionProvider

/-!
Owns the fixed operational continuation of the interactive source extractor:
the selected suffix and parent checks applied to the supplied raw calls,
tapes and clocks. Its data and per-call moment proofs do not depend on a
visited law. `HyperNovaSourceWork` charges its work on the operational
history.

Does not own: a context law, an averaged work bound, or a security bound.
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

end NightstreamFPrime.Export.Stage1.NifsProviderLaw
