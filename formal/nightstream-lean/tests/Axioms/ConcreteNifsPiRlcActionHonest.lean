import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsPiRlcActionHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionHonest.columns_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsPiRlcActionHonest.columns_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionHonest.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsPiRlcActionHonest.rows_honest

end NightstreamTests.Axioms.ConcreteNifsPiRlcActionHonest
