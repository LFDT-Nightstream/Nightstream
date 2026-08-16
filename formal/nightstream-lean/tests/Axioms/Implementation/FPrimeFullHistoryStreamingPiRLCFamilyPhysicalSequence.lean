import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPhysicalSequence
import tests.Axioms.Support

/-! Fail-closed axiom guard for the 110-arm physical PiRLC sequence. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence.AcceptedPhysicalRun.adjacent_state_or_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedPhysicalRun.adjacent_state_or_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence.AcceptedPhysicalRun.semanticRun' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedPhysicalRun.semanticRun

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence.AcceptedPhysicalRun.semanticRun_or_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedPhysicalRun.semanticRun_or_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence.AcceptedPhysicalRun.start_finish_recovers_inputs_or_failure_or_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedPhysicalRun.start_finish_recovers_inputs_or_failure_or_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence.AcceptedPhysicalRun.outputs_exact_or_failure_or_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedPhysicalRun.outputs_exact_or_failure_or_collision
