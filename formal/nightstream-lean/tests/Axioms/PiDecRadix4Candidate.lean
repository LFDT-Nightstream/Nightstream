import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecRadix4Candidate
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the model-level radix-four PiDEC row refinement.
-/

namespace NightstreamTests.Axioms.PiDecRadix4Candidate

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.R1CS.PiDecRadix4Candidate.digitMagnitude_lt_four' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecRadix4Candidate.digitMagnitude_lt_four

/-- info: 'Nightstream.Implementation.R1CS.PiDecRadix4Candidate.rows_force_canonical_split' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecRadix4Candidate.rows_force_canonical_split

end NightstreamTests.Axioms.PiDecRadix4Candidate
