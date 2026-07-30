import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsRawHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawHonest.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawHonest.rows_honest

end NightstreamTests.Axioms.ConcreteNifsRawHonest
