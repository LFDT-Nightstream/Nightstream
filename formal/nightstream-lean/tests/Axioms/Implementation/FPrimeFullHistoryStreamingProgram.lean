import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingFPrimeProgramArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for the Rust streaming F-prime program. -/

namespace NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingProgram

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.artifact_valid' does not depend on any axioms -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_program_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rust_program_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_program_length_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rust_program_length_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_lifecycle_group_map_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rust_lifecycle_group_map_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_circuit_kind_map_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rust_circuit_kind_map_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_claim_coordinate_overlay_kind_map_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rust_claim_coordinate_overlay_kind_map_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_claim_coordinate_overlay_link_runs_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms rust_claim_coordinate_overlay_link_runs_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_claim_coordinate_overlay_link_census_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rust_claim_coordinate_overlay_link_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_pi_rlc_family_overlay_kind_map_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms rust_pi_rlc_family_overlay_kind_map_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_combined_overlay_kind_map_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rust_combined_overlay_kind_map_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_pi_rlc_family_overlay_link_runs_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms rust_pi_rlc_family_overlay_link_runs_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_pi_rlc_family_overlay_link_census_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms rust_pi_rlc_family_overlay_link_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_pi_rlc_family_physical_link_contract_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms rust_pi_rlc_family_physical_link_contract_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_pi_rlc_family_body_overlay_rows_exact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rust_pi_rlc_family_body_overlay_rows_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_phase_public_layout_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms rust_phase_public_layout_exact

end NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingProgram
