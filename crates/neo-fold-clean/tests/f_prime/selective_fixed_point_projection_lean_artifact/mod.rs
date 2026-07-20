//! Rendering support for the bounded tiny-fixture projection drift gate.
//!
//! Owns: the small in-memory representation of one generated Lean file and
//! separation between metadata rendering and exact sparse-row sharding.
//!
//! Does not own: fixed-point construction, filesystem mutation, Lean proofs,
//! or any protocol authority.
//!
//! Emits constraints: no.
//!
//! | Child | Responsibility |
//! |---|---|
//! | `render` | typed tiny-fixture trace and producer metadata |
//! | `selective_rows` | compact 14-leaf source-to-selective interval join |
//! | `stage_paths` | exact 14-leaf stable Rust stage vocabulary |
//! | `rows` | exact source-row sparse partition |
//! | `source_columns` | reachable source slots, definitions, and rewrite provenance |
//! | `selective_matrix_rows` | exact compact rows from the shared final emitter |
//! | `fresh_source` | exact 270-coordinate fresh public-X source decoder |

mod carrier_decoder;
mod carrier_private_padding;
mod carrier_public_padding;
mod carrier_selectors;
mod fresh_source;
mod packed_witness_decoder;
mod raw_running_assignments;
mod render;
mod rows;
mod selective_matrix_rows;
mod selective_rows;
mod source_columns;
mod stage_paths;
mod width_census;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::PiCcsOutputYZcolProjectionAudit;
use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcRawRunningAssignmentAudit;
use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcYZcolSelectiveRowsAudit;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedRowsAudit;
use neo_fold_clean::paper::params::Params;

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
    projection: &PiCcsOutputYZcolProjectionAudit,
    params: &Params,
    fixture: TinyFixtureScope,
    projected: &SelectiveProjectedRowsAudit,
    raw_running_assignments: &[R1csIvcRawRunningAssignmentAudit],
    audit: &R1csIvcYZcolSelectiveRowsAudit,
) -> Vec<GeneratedLeanFile> {
    let mut files = vec![
        render::metadata(projection, fixture),
        stage_paths::render(),
        selective_rows::render(projection.selective_rows()),
    ];
    files.extend(carrier_decoder::render(projected));
    files.push(carrier_public_padding::render(projected));
    files.push(carrier_private_padding::render(projected));
    files.push(carrier_selectors::render(projected));
    files.extend(raw_running_assignments::render(projected, raw_running_assignments));
    files.extend(fresh_source::render(projected, audit));
    files.push(packed_witness_decoder::render(
        projected,
        raw_running_assignments,
        params,
    ));
    files.push(width_census::render(audit.fixed_point(), projected));
    files.extend(rows::row_shards(projection));
    files.extend(source_columns::render(projected));
    files.extend(selective_matrix_rows::render(projection.selective_rows(), projected));
    files
}
