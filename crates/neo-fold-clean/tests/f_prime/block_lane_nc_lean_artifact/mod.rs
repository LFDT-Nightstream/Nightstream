//! Lean rendering for the bounded production combined-NC certificate.
//!
//! Payloads are split by data family. Every generated list shard contains at
//! most 128 proof-free records (64 for the line-heavy thirteen-port rows).

mod active_pins;
mod execution;
mod execution_format;
mod execution_projection;
mod pi_dec_paper_shape;
mod production_placement;
mod provenance;
mod render;
mod rows;
mod terminal_projection;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcBlockLaneNcSelectiveRowsAudit;

pub(super) use execution::ExecutionCertificate;
pub(super) use production_placement::ProductionPlacementCertificate;
pub(super) use terminal_projection::TerminalProjectionFixture;

pub(super) const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/DelayedProjection/CombinedNc/Generated";
pub(super) const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema";
pub(super) const NAMESPACE_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated";

pub(super) struct GeneratedLeanFile {
    pub(super) relative_path: String,
    pub(super) contents: String,
}

#[derive(Clone, Copy)]
pub(super) struct TinyFixtureScope {
    pub(super) parameter_constraint_count: usize,
    pub(super) commitment_width: usize,
    pub(super) security_bits: usize,
    pub(super) application_row_count: usize,
    pub(super) application_column_count: usize,
    pub(super) application_public_input_count: usize,
}

pub(super) fn generated_files(
    audit: &R1csIvcBlockLaneNcSelectiveRowsAudit,
    fixture: TinyFixtureScope,
    execution: &ExecutionCertificate,
    production_placement: &ProductionPlacementCertificate,
    terminal_projection: &TerminalProjectionFixture,
) -> Vec<GeneratedLeanFile> {
    let mut files = vec![render::metadata(audit, fixture), render::round_maps(audit)];
    files.extend(active_pins::render(audit));
    files.extend(rows::source_rows(audit));
    files.extend(rows::emitted_rows(audit));
    files.extend(provenance::render(audit.projected_rows()));
    files.extend(execution::render(execution, audit));
    files.push(production_placement::render(production_placement));
    files.extend(terminal_projection::render(terminal_projection));
    files
}

pub(super) fn assert_execution_mutations_fail(
    execution: &ExecutionCertificate,
    audit: &R1csIvcBlockLaneNcSelectiveRowsAudit,
) {
    execution::assert_mutations_fail(execution, audit)
}

pub(super) fn assert_pi_dec_paper_shape_contract(
    audit: &neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcPostPiDecExecutionAudit,
) {
    pi_dec_paper_shape::assert_live_contract(audit)
}

pub(super) fn focused_raw_old_block_projection_contract(
    execution: &ExecutionCertificate,
    placement: &ProductionPlacementCertificate,
) -> Vec<GeneratedLeanFile> {
    let mut files = execution::focused_raw_old_block_projection_contract(execution);
    files.push(production_placement::render(placement));
    files
}

pub(super) fn focused_public_write_files(execution: &ExecutionCertificate) -> Vec<GeneratedLeanFile> {
    execution::focused_public_write_files(execution)
}

pub(super) fn assert_public_write_mutations_fail(execution: &ExecutionCertificate) {
    execution::assert_public_write_mutations_fail(execution)
}

pub(super) fn terminal_projection_row_files(terminal_projection: &TerminalProjectionFixture) -> Vec<GeneratedLeanFile> {
    terminal_projection::row_chunks(terminal_projection)
}

pub(super) fn generated_header(owner: &str) -> String {
    format!(
        "/-\nGenerated file: production combined-NC artifact; do not hand-edit.\n\nOwns: {owner}.\n\nDoes not own: decoding, row satisfaction, transcript authority, commitment\nbinding, semantic acceptance, costs, or permission to remove rows.\n\nEmits constraints: no.\n\n| Stable stage path | Obligation | Authority class |\n|---|---|---|\n| `f_prime.pi_ccs_nc.delayed.combined.generated` | The generated payload named by `Owns` above | computed artifact |\n-/\n\n"
    )
}

pub(super) fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn lean_option(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("some {value}"))
}
