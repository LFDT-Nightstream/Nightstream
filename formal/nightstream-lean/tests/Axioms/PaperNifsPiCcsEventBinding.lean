import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsPiCcsEventBinding

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding.rows_imply_tableTruth_or_paperBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsEventBinding.rows_imply_tableTruth_or_paperBadEvent

end NightstreamTests.Axioms.PaperNifsPiCcsEventBinding
