import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsCodecProjection

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.ColumnBundle.values_eq_encode_of_decodes' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Goldilocks.ColumnBundle.values_eq_encode_of_decodes

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection.KView.value_eq_of_bundle_decodes' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection.KView.value_eq_of_bundle_decodes

end NightstreamTests.Axioms.PaperNifsCodecProjection
