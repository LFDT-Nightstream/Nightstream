import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStepPaperRefinement
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsStepPaperRefinement

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStepPaperRefinement.recursiveNifs_refinesPaper_or_boundEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  ConcreteNifsStepPaperRefinement.recursiveNifs_refinesPaper_or_boundEvent

end NightstreamTests.Axioms.ConcreteNifsStepPaperRefinement
