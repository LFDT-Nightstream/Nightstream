import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyReplayArtifact
import tests.Axioms.Support

/-! Dependency audit for the production PiRLC family replay artifact. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact.artifact_valid' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact.exact_shape' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact.execution' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms execution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact.family_replays_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms family_replays_exact
