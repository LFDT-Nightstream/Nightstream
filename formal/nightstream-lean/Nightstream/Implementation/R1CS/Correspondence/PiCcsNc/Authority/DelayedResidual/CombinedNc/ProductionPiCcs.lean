import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.MessageTerminal
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection

/-! Compatibility import for the relocated protocol-owned production
combined-NC acceptance. -/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
  (fePoint ncPoint ncTranscriptState ncPoint_eq_transcriptPoint
    rawInitial rawPolynomial messageTerminal
    NcMessageAccepted OutputBindingFailure NcAccepted
    ncMessageAccepted_implies_ncAccepted_or_outputBindingFailure
    MessageAccepted Accepted
    messageAccepted_implies_accepted_or_outputBindingFailure
    accepted_of_messageAccepted_and_packed YRingUnbound YRingBound BadEvent
    ncAccepted_implies_truth_or_badEvent
    accepted_implies_paper_or_yRingUnbound_or_badEvent
    accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs
