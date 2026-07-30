import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsOperationalSamplerHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonestRefinement.honestAssignment_output_eq_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  PiRlcCanonicalSamplerHonestRefinement.honestAssignment_output_eq_bound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerHonest.actionBase_le_orderedLength' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSamplerHonest.actionBase_le_orderedLength

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerHonest.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSamplerHonest.rows_honest

end NightstreamTests.Axioms.ConcreteNifsOperationalSamplerHonest
