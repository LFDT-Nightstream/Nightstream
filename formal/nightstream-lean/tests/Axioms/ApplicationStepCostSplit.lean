import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ApplicationStepCostSplit

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit.callReceipt_cost_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationStepCostSplit.callReceipt_cost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepCost_eq_callCost' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepCost_eq_callCost

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit.CompleteApplicationCertification.stepCost_eq_fixedProtocol_add_application' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationStepCostSplit.CompleteApplicationCertification.stepCost_eq_fixedProtocol_add_application

end NightstreamTests.Axioms.ApplicationStepCostSplit
