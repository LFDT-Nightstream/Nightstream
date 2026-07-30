import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsPaperRefinement

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement.no_outcome_at_wrong_output' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPaperRefinement.no_outcome_at_wrong_output

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement.accepted_refinesPaper_or_boundEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPaperRefinement.accepted_refinesPaper_or_boundEvent

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement.rawRows_refinePaper_or_boundEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPaperRefinement.rawRows_refinePaper_or_boundEvent

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement.selectedNifs_refinesPaper_or_boundEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPaperRefinement.selectedNifs_refinesPaper_or_boundEvent

end NightstreamTests.Axioms.ConcreteNifsPaperRefinement
