import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPublicArtifact
import tests.Axioms.Support

/-! Dependency audit for the PiRLC family public-suffix artifact. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact.artifact_valid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact.exact_suffix_owner_chain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_suffix_owner_chain

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact.canonical_call_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonical_call_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact.poseidon2_call_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms poseidon2_call_refines
