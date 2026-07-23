import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc.Step
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection

/-! Compatibility import for the relocated protocol-owned adjacent delayed
projection step. New protocol code imports its deviation owner directly. -/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionStep

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc.Step
  (parentProjection rawPackedParent ProducerBetaBadRoot
    accepted_next_implies_rawProjection_or_badEvent
    accepted_next_of_rawRecomposition_implies_previous_packedYZcolBound_or_badEvent
    accepted_next_implies_previous_packedYZcolBound_or_parentBindingEvent
    accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent
    accepted_next_implies_previous_packedYZcolBound_or_badEvent)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionStep
