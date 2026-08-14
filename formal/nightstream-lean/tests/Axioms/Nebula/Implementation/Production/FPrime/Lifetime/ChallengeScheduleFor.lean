import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ChallengeScheduleFor
import tests.Axioms.Support

/-! Dependency audit for the exponent-indexed challenge schedule. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperChallengeScheduleFor.base_open_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperChallengeScheduleFor.base_open_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperChallengeScheduleFor.continuation_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperChallengeScheduleFor.continuation_exact
