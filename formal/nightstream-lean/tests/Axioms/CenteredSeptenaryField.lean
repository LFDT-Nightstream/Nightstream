import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenary
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the model-level centered-septenary field encoding.
-/

namespace NightstreamTests.Axioms.CenteredSeptenaryField

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.CenteredSeptenaryField

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryField.width_boundary' does not depend on any axioms -/
#guard_msgs in
#audit_axioms width_boundary

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryField.alphabetWord_low_norm' does not depend on any axioms -/
#guard_msgs in
#audit_axioms alphabetWord_low_norm

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryField.decode_encodeDigit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decode_encodeDigit

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryField.augmented_exists_iff_semantic_exists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms augmented_exists_iff_semantic_exists

end NightstreamTests.Axioms.CenteredSeptenaryField
