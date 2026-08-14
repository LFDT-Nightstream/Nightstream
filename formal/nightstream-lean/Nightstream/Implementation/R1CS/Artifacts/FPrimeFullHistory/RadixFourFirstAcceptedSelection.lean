import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourFirstAcceptedSelection

/-!
Stable facade for the production-width radix-four first-accepted selection
schedule.

Owns: one handwritten import boundary over the generated eight-sampler
schedule.

Does not own: artifact validation, source-row semantics, final low-norm gate
semantics, one-hotness, or permission to remove another row family.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourFirstAcceptedSelection

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourFirstAcceptedSelection
  (profileId rawCoverage)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourFirstAcceptedSelection
