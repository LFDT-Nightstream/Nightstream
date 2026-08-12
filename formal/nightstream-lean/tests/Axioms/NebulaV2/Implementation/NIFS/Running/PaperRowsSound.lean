import Nightstream.Implementation.NebulaV2.NIFS.Running.PaperRowsSound
import tests.Axioms.Support

/-! Dependency audit for the exact row-derived V2 paper NIFS result. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.parentBundle_decode_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.parentBundle_decode_eq

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.parentEvaluation_decode_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.parentEvaluation_decode_eq

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.piDecPlacement_of_parentFields' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.piDecPlacement_of_parentFields

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.rows_imply_exact_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductNifsPaperRowsSound.rows_imply_exact_result
