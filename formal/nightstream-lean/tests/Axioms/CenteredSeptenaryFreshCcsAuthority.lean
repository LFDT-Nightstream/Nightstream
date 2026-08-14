import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryFreshCcsAuthority
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the fresh radix-four CCS norm transfer.
-/

namespace NightstreamTests.Axioms.CenteredSeptenaryFreshCcsAuthority

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority.wordCoordinate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms wordCoordinate

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority.every_word_has_septenary_alphabet_of_norm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms every_word_has_septenary_alphabet_of_norm

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority.norm_four_of_fresh_ccsHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms norm_four_of_fresh_ccsHolds

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority.radixFourCandidate_every_word_has_septenary_alphabet' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms radixFourCandidate_every_word_has_septenary_alphabet

end NightstreamTests.Axioms.CenteredSeptenaryFreshCcsAuthority
