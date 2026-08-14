import Nightstream.Implementation.Nebula.NIFS.Core.PaperSelection
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductPaperNifsSelection

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperNifsSelection.selected_verify_block_eq_false_of_sampler_rejected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selected_verify_block_eq_false_of_sampler_rejected

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperNifsSelection.verifyClaim_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms verifyClaim_sound

/-- info: 'Nightstream.Implementation.Nebula.ProductPaperNifsSelection.verifyClaim_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms verifyClaim_complete
