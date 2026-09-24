import NightstreamFPrime.Export.Stage1.Wide.InputSupport
import NightstreamFPrime.Export.Stage1.Wide.Stage1Plan

namespace NightstreamFPrime.Export.Stage1.Wide.InputSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

theorem inputsBefore (program : RetainedLayout.Program) :
    PiRLCWitness.InputsBefore (Stage1Plan.piRlcInterface program) := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · exact common_before program
      (ApplicationRetainedGeometry.oneColumn (Stage1Plan.referenceGeometry program))
      (Or.inl (by
        rw [(RetainedLayout.boundaries program).1]
        change 0 < 113904174
        decide))
  · intro lane
    exact renamed_before program _ (piCcsOutput program (Stage1Plan.poseidonGeometry program) _ lane)
  · intro ring lane
    exact renamed_before program _ (location program (Stage1Plan.piCcsGeometry program)
      (PiRLCValueWiring.located (PiRLCProductRingSchedule.laneInvocation ring lane)).location)


end NightstreamFPrime.Export.Stage1.Wide.InputSupport
