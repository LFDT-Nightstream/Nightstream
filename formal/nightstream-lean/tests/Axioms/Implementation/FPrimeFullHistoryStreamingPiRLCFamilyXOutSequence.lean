import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutSequence
import tests.Axioms.Support

/-! Fail-closed axiom guard for the full-`x_out` PiRLC family sequence. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.adjacent_state_or_failure' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedFullStateRun.adjacent_state_or_failure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.semanticRun' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedFullStateRun.semanticRun

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.semanticRun_or_failure' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedFullStateRun.semanticRun_or_failure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.start_finish_recovers_inputs_or_failure' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedFullStateRun.start_finish_recovers_inputs_or_failure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.outputs_exact_or_failure' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedFullStateRun.outputs_exact_or_failure
