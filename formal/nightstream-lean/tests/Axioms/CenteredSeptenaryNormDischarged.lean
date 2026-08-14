import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryNormDischarged
import tests.Axioms.Support

/-!
Fail-closed axiom guard for radix-four norm discharge.
-/

namespace NightstreamTests.Axioms.CenteredSeptenaryNormDischarged

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.normBoundFour_iff_centeredResidue' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms normBoundFour_iff_centeredResidue

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.concrete_normBounded_four_implies_centered' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms concrete_normBounded_four_implies_centered

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.finiteWordOfField_alphabet_of_outer_norm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finiteWordOfField_alphabet_of_outer_norm

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged.reconstructed_source_exists_of_outer_norm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms reconstructed_source_exists_of_outer_norm

end NightstreamTests.Axioms.CenteredSeptenaryNormDischarged
