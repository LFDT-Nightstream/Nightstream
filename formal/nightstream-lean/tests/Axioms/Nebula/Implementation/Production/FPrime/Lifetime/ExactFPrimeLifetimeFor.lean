import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ExactFPrimeLifetimeFor
import tests.Axioms.Support

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor.Lifetime.exact_schedule

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor.Lifetime.construct_exact_schedule

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor.Lifetime.construct_has_terminal

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor.Schedule.consumerInvocationIndices_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor.Schedule.consumerInvocationIndices_length
