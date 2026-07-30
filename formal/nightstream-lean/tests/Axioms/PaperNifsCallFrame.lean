import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsCallFrame

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame.decodes_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame.decodes_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame.encodes_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame.encodes_iff

end NightstreamTests.Axioms.PaperNifsCallFrame
