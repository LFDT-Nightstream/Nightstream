import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.FPrimeKFixedPhaseSemanticOccurrence

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.decodeCarried_carried' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFixedPhaseSemanticOccurrence.decodeCarried_carried

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.SourceColumns.rows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSemanticOccurrence.SourceColumns.rows_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence.SourceColumns.rows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSemanticOccurrence.SourceColumns.rows_honest

end NightstreamTests.Axioms.FPrimeKFixedPhaseSemanticOccurrence
