import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.CanonicalX
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the exact Rust/Lean PiDEC canonical-X row check.
-/

namespace NightstreamTests.Axioms.PiDecCanonicalXRustConformance

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX.generated_rows_match_independent_compiler' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_rows_match_independent_compiler

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX.checked_partition_count' does not depend on any axioms -/
#guard_msgs in
#audit_axioms checked_partition_count

end NightstreamTests.Axioms.PiDecCanonicalXRustConformance
