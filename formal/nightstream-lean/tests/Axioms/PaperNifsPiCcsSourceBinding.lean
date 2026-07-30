import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.PaperNifsPiCcsSourceBinding

open NightstreamTests.Axioms

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding.RunningViews.physicalPriorPoint_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding.RunningViews.physicalPriorPoint_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding.RunningViews.physicalClaimedCoefficient_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding.RunningViews.physicalClaimedCoefficient_eq

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding.RunningViews.physicalVerifierInput_eq_statement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding.RunningViews.physicalVerifierInput_eq_statement

end NightstreamTests.Axioms.PaperNifsPiCcsSourceBinding
