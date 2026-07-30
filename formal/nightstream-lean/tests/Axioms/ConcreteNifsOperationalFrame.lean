import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsOperationalFrame

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame.decodedAuthority_of_frame_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalFrame.decodedAuthority_of_frame_decodes

end NightstreamTests.Axioms.ConcreteNifsOperationalFrame
