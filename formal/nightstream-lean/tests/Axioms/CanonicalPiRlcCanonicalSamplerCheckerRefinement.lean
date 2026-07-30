import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPiRlcCanonicalSamplerCheckerRefinement

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.chunkValue_eq_bitsValue_slice' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.chunkValue_eq_bitsValue_slice

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.semanticCandidate_eq_checkerStream' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.semanticCandidate_eq_checkerStream

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.semanticCandidates_eq_candidatePrefix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.semanticCandidates_eq_candidatePrefix

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.sampleChallenge?_eq_some_semanticChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.sampleChallenge?_eq_some_semanticChallenge

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge_coordinate_eq_outputColumn' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge_coordinate_eq_outputColumn

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.samplerRows_sampleChallenge?_eq_some' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement.samplerRows_sampleChallenge?_eq_some

end NightstreamTests.Axioms.CanonicalPiRlcCanonicalSamplerCheckerRefinement
