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
//! | `rows` | exact 12-way sparse-row partition |

mod render;
mod rows;
mod selective_rows;
mod stage_paths;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::PiCcsOutputYZcolProjectionAudit;

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
    fixture: TinyFixtureScope,
) -> Vec<GeneratedLeanFile> {
    let mut files = vec![
        render::metadata(projection, fixture),
        stage_paths::render(),
        selective_rows::render(projection.selective_rows()),
    ];
    files.extend(rows::row_shards(projection));
    files
}
