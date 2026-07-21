import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.QuarticEmbedding

/-!
Composition boundary from one decoded production round to one independent
quartic combined-NC round.

Owns: exact-row production acceptance and its generic quartic claimed-round
view.

Does not own: generated round-map truth, satisfaction of any production row,
transcript ordering, terminal semantics, raw-child authority, costs, or row
removal.

Emits constraints: none.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.QuarticEmbedding

/-- A satisfying decoded 30-row round yields the independent quartic round
equations for its exact five assignment-derived coefficients. -/
theorem roundMapHolds_implies_quarticAccepted
    (round : DecodedRoundMap)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RoundMapHolds round assignment) :
    ClaimedRoundAccepted
      (ProductionRound.coefficientValues
        (Relabel.assignment round.raw.columnMap assignment))
      (ProductionRound.claimInValue
        (Relabel.assignment round.raw.columnMap assignment))
      (ProductionRound.challengeValue
        (Relabel.assignment round.raw.columnMap assignment))
      (ProductionRound.claimOutValue
        (Relabel.assignment round.raw.columnMap assignment)) := by
  have productionAccepted := roundMapAccepted_of_holds round
    canonical one holds
  exact (productionAccepted_iff_claimedRoundAccepted
    (Relabel.assignment round.raw.columnMap assignment)).1
    productionAccepted

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundSemantics
