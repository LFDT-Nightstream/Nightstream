import Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection.selected_verify_block_eq_false_of_sampler_rejected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selected_verify_block_eq_false_of_sampler_rejected

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection.verifyClaim_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms verifyClaim_sound

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection.verifyClaim_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms verifyClaim_complete
