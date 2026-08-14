import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.RadixFourCanonicalX
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the exact Rust/Lean radix-four PiDEC row check.
-/

namespace NightstreamTests.Axioms.PiDecRadixFourCanonicalXRustConformance

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.generated_rows_match_independent_compiler' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_rows_match_independent_compiler

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.checked_partition_count' does not depend on any axioms -/
#guard_msgs in
#audit_axioms checked_partition_count

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.candidate_geometry_matches_model' does not depend on any axioms -/
#guard_msgs in
#audit_axioms candidate_geometry_matches_model

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.generated_coordinate_count' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_coordinate_count

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.generated_row_count' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_row_count

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.generated_coordinate_rows_match_model' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_coordinate_rows_match_model

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX.generated_rows_force_canonical_split' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_rows_force_canonical_split

end NightstreamTests.Axioms.PiDecRadixFourCanonicalXRustConformance
