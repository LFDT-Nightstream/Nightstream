import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.FPrimeKFixedPhasePhysicalOccurrence

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.rows_cost' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.rows_cost

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.row_ids_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.row_ids_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.numeric_support_below_end' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.numeric_support_below_end

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.rows_supported' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.satisfies_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.satisfies_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.PhysicalOccurrence.rows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.PhysicalOccurrence.rows_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence.PhysicalOccurrence.rows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhasePhysicalOccurrence.PhysicalOccurrence.rows_honest

end NightstreamTests.Axioms.FPrimeKFixedPhasePhysicalOccurrence
