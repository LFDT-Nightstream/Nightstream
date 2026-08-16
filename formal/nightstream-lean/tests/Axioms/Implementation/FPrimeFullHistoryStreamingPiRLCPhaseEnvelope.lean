import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for the exact PiRLC carry-phase envelope. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.artifact_valid' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.hash_rows_refine_phase_envelope' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms hash_rows_refine_phase_envelope

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.x_out_semantic_refines_phase_envelope' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms x_out_semantic_refines_phase_envelope
