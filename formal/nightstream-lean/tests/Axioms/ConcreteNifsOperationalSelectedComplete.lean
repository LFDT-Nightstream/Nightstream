import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsOperationalSelectedComplete

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest.decoded_eq_of_preserved' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  KSplitNcEndpointsSemanticHonest.decoded_eq_of_preserved

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsEndpointConservation.proofView_below_transcriptBase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsEndpointConservation.proofView_below_transcriptBase

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedHonest.retargetedTranscript_inPrefix' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelectedHonest.retargetedTranscript_inPrefix

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedHonest.selectedTranscript_valid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelectedHonest.selectedTranscript_valid

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedHonest.selectedChains_afterTranscript' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelectedHonest.selectedChains_afterTranscript

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete.selectedEndpoints_afterNumeric' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelectedComplete.selectedEndpoints_afterNumeric

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete.selectedAuthority_afterNumeric' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelectedComplete.selectedAuthority_afterNumeric

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete.selectedRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsOperationalSelectedComplete.selectedRows_honest

end NightstreamTests.Axioms.ConcreteNifsOperationalSelectedComplete
