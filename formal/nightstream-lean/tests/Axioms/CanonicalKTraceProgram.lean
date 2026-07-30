import Nightstream.Implementation.R1CS.Canonical.KTraceProgram
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKTraceProgram

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.carriedValue_decodePoint' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceProgram.carriedValue_decodePoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.traceAccepts_of_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgram.traceAccepts_of_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.traceRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgram.traceRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgram.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.batchAccepted_of_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgram.batchAccepted_of_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.batchExact_or_badRoot_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgram.batchExact_or_badRoot_of_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.Occurrence.exact_or_badRoot' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgram.Occurrence.exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.Occurrence.badRoot_is_bound' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KTraceProgram.Occurrence.badRoot_is_bound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgram.Occurrence.exact_excludes_badRoot' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KTraceProgram.Occurrence.exact_excludes_badRoot

end NightstreamTests.Axioms.CanonicalKTraceProgram
