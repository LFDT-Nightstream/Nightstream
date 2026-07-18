import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.Rows

/-!
Public theorem-shape regressions for the active PiRLC sampler row boundary.

Assurance tier: model-level conditional projections plus artifact-derived
source-row extents. No Rust row-list identity or physical ownership partition
is asserted here.
-/

namespace NightstreamTests.FPrimeRecursivePiRlcChallengeSamplerRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.Rows

#check CanonicalRowsEmbedded
#check SamplerRowsEmbedded
#check EmbeddedRowsSatisfied
#check accepted_canonicalLane_refines
#check accepted_laneRows
#check accepted_tailRows
#check accepted_initialCount_zero
#check accepted_readableTail

#check chunkAcceptanceSourceRowExtent
#check chunkMod5SourceRowExtent
#check chunkSymbolPrefixSourceRowExtent
#check acceptanceBoundSourceRowExtent
#check selectionInitializationSourceRowExtent
#check selectionOneHotSourceRowExtent
#check selectionProductsSourceRowExtent
#check selectionBindingsSourceRowExtent

example :
    canonicalTranscriptSourceRowExtent = 16560 ∧
    samplerInitializationSourceRowExtent = 15 ∧
    chunkAcceptanceSourceRowExtent = 3840 ∧
    chunkMod5SourceRowExtent = 19200 ∧
    chunkSymbolPrefixSourceRowExtent = 1920 ∧
    samplerResidualLaneSourceRowExtent = 24960 ∧
    acceptanceBoundSourceRowExtent = 90 ∧
    selectionInitializationSourceRowExtent = 15 ∧
    selectionOneHotSourceRowExtent = 9720 ∧
    selectionProductsSourceRowExtent = 26730 ∧
    selectionBindingsSourceRowExtent = 2430 ∧
    samplerTailSourceRowExtent = 38985 ∧
    samplerSourceRowExtent = 63960 :=
  sourceRowExtentTable

example
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin 15) (block lane : Fin 4) :
    Satisfies (laneRows rho block lane) assignment :=
  accepted_laneRows accepted rho block lane

end NightstreamTests.FPrimeRecursivePiRlcChallengeSamplerRows
