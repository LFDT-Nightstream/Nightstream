import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcTranscriptPhases

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases.decoded_fe' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcTranscriptPhases.decoded_fe

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases.decoded_nc' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcTranscriptPhases.decoded_nc

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases.decoded_output' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcTranscriptPhases.decoded_output

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases.feAgrees' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcTranscriptPhases.feAgrees

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases.ncAgrees' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcTranscriptPhases.ncAgrees

end NightstreamTests.Axioms.CanonicalKSplitNcTranscriptPhases
