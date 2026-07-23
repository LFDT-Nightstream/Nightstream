import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition
import tests.Axioms.Support

/-! Fail-closed dependency gate for fixed-profile production NC acceptance. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open SelectedRowsActiveComposition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge.computed_finalSum_eq_messageTerminal' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionTerminalBridge.computed_finalSum_eq_messageTerminal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionColumnBindings.columnBindings_imply_exactDataflow' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProductionColumnBindings.columnBindings_imply_exactDataflow

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition.generatedEmittedAssignmentSatisfies_implies_claimsAccepted' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedEmittedAssignmentSatisfies_implies_claimsAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition.generatedEmittedAssignmentSatisfies_implies_claimsCheck' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedEmittedAssignmentSatisfies_implies_claimsCheck

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition.generatedEmittedAssignmentPair_of_nextPacked_implies_previousPacked_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedEmittedAssignmentPair_of_nextPacked_implies_previousPacked_or_namedFailure
