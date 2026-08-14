import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import tests.Axioms.Support

/-! Dependency audit for the verifier-owned streaming F-prime program. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram.Phase.code_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram.Phase.code_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram.complete_run_steps_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram.complete_run_steps_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram.production_complete_run_steps_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram.production_complete_run_steps_exact
