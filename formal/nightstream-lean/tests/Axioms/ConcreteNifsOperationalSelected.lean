import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsOperationalSelected

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected.selectedPiCcsAccepted_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelected.selectedPiCcsAccepted_of_rows

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected.selectedPiRlcInitialState_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelected.selectedPiRlcInitialState_eq

end NightstreamTests.Axioms.ConcreteNifsOperationalSelected
