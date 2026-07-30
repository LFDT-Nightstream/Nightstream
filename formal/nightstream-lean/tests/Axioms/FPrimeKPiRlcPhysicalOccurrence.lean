import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.FPrimeKPiRlcPhysicalOccurrence

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcPhysicalOccurrence.rows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.row_ids_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcPhysicalOccurrence.row_ids_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.rows_supported' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcPhysicalOccurrence.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.satisfies_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcPhysicalOccurrence.satisfies_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence.PhysicalOccurrence.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcPhysicalOccurrence.PhysicalOccurrence.rows_honest

end NightstreamTests.Axioms.FPrimeKPiRlcPhysicalOccurrence
