import Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor.sections_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sections_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor.runningCoordinate_surjective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCoordinate_surjective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor.runningCodecFor_point_getD' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCodecFor_point_getD

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor.runningCodecFor_commitment_getD' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCodecFor_commitment_getD

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor.runningCodecFor_publicInput_getD' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCodecFor_publicInput_getD

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor.runningCodecFor_evaluation_getD' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCodecFor_evaluation_getD
