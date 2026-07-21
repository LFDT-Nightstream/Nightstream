//! Lean rendering for the bounded production combined-NC certificate.
//!
//! Payloads are split by data family. Every generated list shard contains at
//! most 128 proof-free records (64 for the line-heavy thirteen-port rows).

mod provenance;
mod render;
mod rows;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcBlockLaneNcSelectiveRowsAudit;

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
) -> Vec<GeneratedLeanFile> {
    let mut files = vec![render::metadata(audit, fixture), render::round_maps(audit)];
    files.extend(rows::source_rows(audit));
    files.extend(rows::emitted_rows(audit));
    files.extend(provenance::render(audit.projected_rows()));
    files
}

pub(super) fn generated_header(owner: &str) -> String {
    format!(
        "/-\nGenerated file: production combined-NC artifact; do not hand-edit.\n\nOwns: {owner}.\n\nDoes not own: decoding, row satisfaction, transcript authority, commitment\nbinding, semantic acceptance, costs, or permission to remove rows.\n\nEmits constraints: no.\n-/\n\n"
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
