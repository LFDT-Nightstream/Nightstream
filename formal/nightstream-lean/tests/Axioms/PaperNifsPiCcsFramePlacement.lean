import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsPiCcsFramePlacement

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement.fromFrame' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement.fromFrame

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement.decodedVerifierInput_eq_statement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsFramePlacement.decodedVerifierInput_eq_statement

end NightstreamTests.Axioms.PaperNifsPiCcsFramePlacement
