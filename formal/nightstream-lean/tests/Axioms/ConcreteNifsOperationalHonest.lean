import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsOperationalHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalHonest.rows_honest_of_semantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalHonest.rows_honest_of_semantics

end NightstreamTests.Axioms.ConcreteNifsOperationalHonest
