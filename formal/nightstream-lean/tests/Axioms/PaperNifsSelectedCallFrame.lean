import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsSelectedCallFrame

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.running_decodes_of_frame_decodes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.running_decodes_of_frame_decodes

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.decodedVerifierInput_eq_statement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.decodedVerifierInput_eq_statement

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.rows_imply_tableTruth_or_paperBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsSelectedCallFrame.rows_imply_tableTruth_or_paperBadEvent

end NightstreamTests.Axioms.PaperNifsSelectedCallFrame
