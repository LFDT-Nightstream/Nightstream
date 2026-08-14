import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.SliceComposition
import tests.Axioms.Support

namespace NightstreamTests.Axioms.Implementation.Poseidon2TranscriptSliceComposition

open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.SliceComposition

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.SliceComposition.semanticExecuteSlice_append' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms semanticExecuteSlice_append

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.SliceComposition.semanticExecuteSlices_eq_flatten' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms semanticExecuteSlices_eq_flatten

end NightstreamTests.Axioms.Implementation.Poseidon2TranscriptSliceComposition
