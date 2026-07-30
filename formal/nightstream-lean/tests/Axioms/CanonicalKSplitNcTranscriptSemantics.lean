import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcTranscriptSemantics

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics.decoded_preSumcheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcTranscriptSemantics.decoded_preSumcheck

end NightstreamTests.Axioms.CanonicalKSplitNcTranscriptSemantics
