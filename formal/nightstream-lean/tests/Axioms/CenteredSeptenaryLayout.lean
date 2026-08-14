import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryLayout
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the centered-septenary assignment-layout bridge.
-/

namespace NightstreamTests.Axioms.CenteredSeptenaryLayout

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.CenteredSeptenaryLayout

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLayout.word_coordinate_lt' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms word_coordinate_lt

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLayout.every_word_has_septenary_alphabet' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms every_word_has_septenary_alphabet

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLayout.decodedAssignment_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedAssignment_canonical

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLayout.accepted_reconstructs_canonical_source' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_reconstructs_canonical_source

end NightstreamTests.Axioms.CenteredSeptenaryLayout
