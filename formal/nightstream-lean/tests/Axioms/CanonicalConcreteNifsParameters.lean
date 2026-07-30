import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalConcreteNifsParameters

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters.nifsVerifier_eq_some_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsParameters.nifsVerifier_eq_some_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters.selected_setup_nifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsParameters.selected_setup_nifs

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters.callEval_nifsVerify' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsParameters.callEval_nifsVerify

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters.callEval_nifsVerify_eq_some_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsParameters.callEval_nifsVerify_eq_some_iff

end NightstreamTests.Axioms.CanonicalConcreteNifsParameters
