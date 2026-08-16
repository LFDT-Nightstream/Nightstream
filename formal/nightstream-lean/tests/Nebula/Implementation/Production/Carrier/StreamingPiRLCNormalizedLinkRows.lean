import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedLinkRows

namespace Nightstream.Tests.Nebula.Implementation.Production.Carrier.StreamingPiRLCNormalizedLinkRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedLinkRows.Normalized

#check accepted_implies_fieldLinksHold
#check overlaySourceColumnsExact_of_links
#check bodyPhaseBindingPlaced_of_links
#check accepted_implies_bodyPhaseBindingPlaced

example :
    BodyFinalColumns = 2484972 /\ OverlayFinalColumns = 35856 := by
  constructor <;> native_decide

example :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit.totalLinkCount =
      3669490 := by
  exact receipt_geometry_exact.2.2.2.2

end Nightstream.Tests.Nebula.Implementation.Production.Carrier.StreamingPiRLCNormalizedLinkRows
