import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
import tests.Axioms.Support

/-! Kernel dependency report for the bounded fresh public-`X` decoder. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

/-- info: 'Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Exact.records_all_wellFormed' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Exact.records_all_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Exact.sourceColumn_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Exact.sourceColumn_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Refinement.coordinateValueBindings_and_dataflow_imply_freshPublicInput' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Refinement.coordinateValueBindings_and_dataflow_imply_freshPublicInput
