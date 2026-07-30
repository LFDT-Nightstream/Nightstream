import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsCompleteApplication

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication.allRecipes_nifsVerify' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsCompleteApplication.allRecipes_nifsVerify

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication.allRecipes_step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsCompleteApplication.allRecipes_step

end NightstreamTests.Axioms.ConcreteNifsCompleteApplication
