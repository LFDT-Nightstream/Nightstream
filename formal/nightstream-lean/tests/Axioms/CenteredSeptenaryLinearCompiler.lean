import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryLinearCompiler
import tests.Axioms.Support

/-!
Fail-closed axiom guard for radix-four linear substitution.
-/

namespace NightstreamTests.Axioms.CenteredSeptenaryLinearCompiler

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler.decodedPrivateColumn' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedPrivateColumn

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler.loweredRows_iff_sourceRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms loweredRows_iff_sourceRows

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler.loweredRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms loweredRows_sound

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler.privateNorm_of_fresh_ccsHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms privateNorm_of_fresh_ccsHolds

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryLinearCompiler.freshCcs_loweredRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms freshCcs_loweredRows_sound

end NightstreamTests.Axioms.CenteredSeptenaryLinearCompiler
