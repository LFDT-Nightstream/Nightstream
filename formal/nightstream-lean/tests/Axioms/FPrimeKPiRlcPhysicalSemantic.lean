import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalSemantic
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.FPrimeKPiRlcPhysicalSemantic

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalSemantic.equations_or_badRoot_of_typed_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcPhysicalSemantic.equations_or_badRoot_of_typed_rows

end NightstreamTests.Axioms.FPrimeKPiRlcPhysicalSemantic
