import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.MessageTerminal
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance

/-! Compatibility import for the relocated protocol-owned message terminal. -/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.MessageTerminal

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.MessageTerminal
  (runningValueFromMessage runningValueFromMessage_eq_authoritative_of_bound
    delayedFromMessage verifierTerminal
    verifierTerminal_eq_combinedAtPoint_of_bound
    verifierTerminal_eq_sumcheckPolynomial_of_bound AcceptedFromMessage
    acceptedFromMessage_implies_rawAccepted_or_outputBindingFailure
    TranscriptAcceptedFromMessage
    transcriptAcceptedFromMessage_implies_rawAccepted_or_outputBindingFailure)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.MessageTerminal
