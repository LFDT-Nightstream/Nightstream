import Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor
import tests.Axioms.Support

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Lifetime.exact_schedule

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Lifetime.construct_exact_schedule

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Lifetime.construct_has_terminal

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Schedule.consumerInvocationIndices_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Schedule.consumerInvocationIndices_length
