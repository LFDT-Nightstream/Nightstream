import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution
import tests.Axioms.Support

/-! Kernel dependency report for the active Carrier270 public-write trace. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace.productionTrace_certificate' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionPublicWriteTrace.productionTrace_certificate

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.ProductionPublicWriteTrace.production_projectPhysical270_execute_eq_projectPublicInput' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionPublicWriteTrace.production_projectPhysical270_execute_eq_projectPublicInput

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution.execution_activePublicWritesBound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution.execution_activePublicWritesBound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution.execution_normalizedPublicInput_eq_projectPublicInput' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution.execution_normalizedPublicInput_eq_projectPublicInput
