//! Focused drift gate for the emitted three-matrix diagnostic PiRLC artifacts.
//!
//! Owns: one production recursive-program build shared by the exact parent
//! `y_zcol` output owners and complete identities, beta ladder, and
//! rho-evaluation generators.
//!
//! Does not own: deprecated paper certificates, semantic authority, encoded
//! lowering, cost estimates, or row removal.
//!
//! Emits constraints: no.

use neo_fold_clean::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;

use super::build_recursive_program_with_output;

#[test]
fn active_pi_rlc_projection_artifacts_match_production_trace() {
    let (builder, output) = build_recursive_program_with_output();
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    validate_projection_identity_traces(&source, trace).expect("exact production projection trace");
    super::active_rho_challenge_wiring::check_generated_artifact(trace);
    super::active_sampler_layout::check_generated_artifact(&source, trace);
    super::active_y_zcol_output_owners::check_generated_artifact(&source, trace, &output);
    super::active_beta_ladder::check_generated_artifact(&source, trace, builder.projection_ladder_audits());
    super::active_rho_evaluations::check_generated_artifact(&source, trace);
    super::active_y_zcol_identities::check_generated_artifact(&source, trace, builder.projection_identity_audits());
}
