import Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed challenge schedule. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor.base_open_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor.base_open_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor.continuation_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor.continuation_exact
