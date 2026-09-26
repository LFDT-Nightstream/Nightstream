import NightstreamFPrime.Export.Stage1.Wide.ApplicationSupport
import NightstreamFPrime.Export.Stage1.Wide.PilotSupport
import NightstreamFPrime.Export.Stage1.Wide.PiDECSupport
import NightstreamFPrime.Export.Stage1.Wide.RunningSupport

/-! Full read-support certificates for the reused Stage 1 phases. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint

theorem prefixPlan (program : Program)
    {width : Nat} {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (relation : Lifecycle.ProductionKey.LogicalRelation width publicFits)
    (geometry : PiDECRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    CommonPlans program (DirectPiDECPrefixPlan.piCcsCompletePlan relation geometry) := by
  apply append
  · apply append
    · apply append
      · apply append
        · apply append
          · exact pilot_poseidon program _
          · exact piCcs_poseidon program _ _
        · exact piCcs_ordinary program relation _
      · exact pilot_ordinary program _
    · exact pilot_binding program _
  · exact piCcs_pins program _ _

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
