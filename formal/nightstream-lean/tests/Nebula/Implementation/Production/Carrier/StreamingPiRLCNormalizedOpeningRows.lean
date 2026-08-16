import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOpeningRows

namespace Nightstream.Tests.Nebula.Implementation.Production.Carrier.StreamingPiRLCNormalizedOpeningRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOpeningRows.Normalized

#check openingChunkSchedule
#check canonicalAssignment_encodedLt
#check accepted_implies_canonicalOpening
#check accepted_implies_activeDigitExact
#check accepted_implies_bodySourceColumnsExact
#check accepted_implies_bodyPhaseBindingPlaced

example : activeDigitCount = 33210 /\ centeredRowCount = 16605 := by
  exact ⟨activeDigitCount_exact, centeredRowCount_exact⟩

end Nightstream.Tests.Nebula.Implementation.Production.Carrier.StreamingPiRLCNormalizedOpeningRows
