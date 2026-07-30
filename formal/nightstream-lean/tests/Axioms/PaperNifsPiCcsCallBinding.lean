import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsPiCcsCallBinding

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap.KLocation.numeric_value_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap.KLocation.numeric_value_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding.RunningPlacement.decodedVerifierInput_eq_physical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding.RunningPlacement.decodedVerifierInput_eq_physical

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding.RunningPlacement.decodedVerifierInput_eq_statement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding.RunningPlacement.decodedVerifierInput_eq_statement

end NightstreamTests.Axioms.PaperNifsPiCcsCallBinding
