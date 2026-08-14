import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourCenteredDomainRows

/-!
Stable facade for two exact production radix-four centered-domain rows.

Owns: one handwritten import boundary over the generated pair and odd-tail
rows materialized from the final production recursive-arm matrices.

Does not own: decoding, arithmetic semantics, all centered rows, source-column
meaning, selector dispatch, constraint necessity, or row removal.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourCenteredDomainRows

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourCenteredDomainRows
  (rawPairRow rawTailRow)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourCenteredDomainRows
