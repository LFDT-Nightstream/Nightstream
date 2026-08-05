//! Fail-closed consumer for a Lean-owned F′ plus Nebula CCS manifest.
//!
//! Owns: schema-v4 decoding, reuse of the validated four-matrix native Step
//! core, validation of the nineteen-role polynomial and combined placement,
//! and exact construction of the combined sparse CCS relation.
//!
//! Does not own: manifest generation, witness generation, setup selection,
//! recursive proving, Spartan, WHIR, or application semantics.
//!
//! Emits constraints: yes. It places the exact Lean Step rows first and the
//! exact Lean Nebula rows second. Rust does not regenerate either program.

use std::collections::HashMap;

use neo_ccs::{check_ccs_rowwise_zero, CcsMatrix, CscMat, SparsePoly, Term};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use thiserror::Error;

use crate::paper::relations::Structure;

use super::lean_manifest::{invalid, ColumnId, LeanManifestError, ManifestCost, ManifestTerm, GOLDILOCKS_MODULUS};
use super::lean_native_ccs_manifest::{
    LeanNativeCcsManifest, TerminalR1csDescriptor, LEAN_NATIVE_CCS_MANIFEST_FORMAT,
    LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION,
};

/// Schema selected by the Lean combined-manifest encoder.
pub const LEAN_NEBULA_COMBINED_MANIFEST_SCHEMA_VERSION: u64 = 4;
/// Stable format selected by the Lean combined-manifest encoder.
pub const LEAN_NEBULA_COMBINED_MANIFEST_FORMAT: &str = "nightstream/fprime-nebula-combined-manifest";

const NATIVE_MATRIX_COUNT: usize = 4;
const NEBULA_MATRIX_COUNT: usize = 15;
const COMBINED_MATRIX_COUNT: usize = NATIVE_MATRIX_COUNT + NEBULA_MATRIX_COUNT;
const COMBINED_STRICT_DEGREE_BOUND: usize = 5;
const NATIVE_PUBLIC_WIDTH: usize = 257;
const PHI81_RING_DEGREE: usize = 54;
const TERMINAL_COMMITMENT_ROWS: usize = 18;
const FIXED_PUBLIC_PADDING: usize = 4;

#[derive(Debug, Error)]
pub enum LeanNebulaCombinedEmissionError {
    /// A native Step column has no caller-supplied value.
    #[error("missing value for native Step column {column:?}")]
    MissingNativeValue { column: ColumnId },
    /// The supplied combined public input has the wrong width.
    #[error("combined public input has {found} values; expected {expected}")]
    PublicWidth { found: usize, expected: usize },
    /// The supplied Nebula private witness has the wrong width.
    #[error("Nebula private witness has {found} values; expected {expected}")]
    NebulaPrivateWidth { found: usize, expected: usize },
    /// Two views of one shared coordinate disagree.
    #[error("shared assignment differs at combined column {column}")]
    SharedValue { column: usize },
    /// A validated row refers to a column absent from the validated core.
    #[error("validated native row refers to unknown column {column:?}")]
    UnknownNativeColumn { column: ColumnId },
    /// The exact validated data cannot form one CCS structure.
    #[error("cannot construct the Lean combined relation: {0}")]
    InvalidStructure(String),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CombinedPolynomialTerm {
    coefficient: u64,
    exponents: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum NebulaFamily {
    Filler,
    OperationBit,
    OperationCount,
    ReadWrite,
    TimestampOrder,
    RomWrite,
    RomRange,
    Padding,
    ReadProduct,
    WriteProduct,
    InitialScanBit,
    FinalScanBit,
    InitialScanProduct,
    FinalScanProduct,
    BoundaryTimestamp,
    BoundaryProduct,
}

impl NebulaFamily {
    pub(super) fn is_extension(self) -> bool {
        matches!(
            self,
            Self::ReadProduct | Self::WriteProduct | Self::InitialScanProduct | Self::FinalScanProduct
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct NebulaRowId {
    family: NebulaFamily,
    slot: usize,
    component: usize,
    ordinal: usize,
    position: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct NebulaTerm {
    pub(super) column: usize,
    pub(super) coefficient: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct NebulaImages {
    pub(super) bit: Vec<NebulaTerm>,
    pub(super) product_left: Vec<NebulaTerm>,
    pub(super) product_right: Vec<NebulaTerm>,
    pub(super) linear_left: Vec<NebulaTerm>,
    pub(super) linear_right: Vec<NebulaTerm>,
    pub(super) output: Vec<NebulaTerm>,
    pub(super) extension_a: Vec<NebulaTerm>,
    pub(super) extension_b: Vec<NebulaTerm>,
    pub(super) pad: Vec<NebulaTerm>,
    pub(super) active: Vec<NebulaTerm>,
    pub(super) fingerprint_a: Vec<NebulaTerm>,
    pub(super) fingerprint_b: Vec<NebulaTerm>,
    pub(super) value_a: Vec<NebulaTerm>,
    pub(super) value_b: Vec<NebulaTerm>,
    pub(super) value: Vec<NebulaTerm>,
}

impl NebulaImages {
    fn combinations(&self) -> [&[NebulaTerm]; NEBULA_MATRIX_COUNT] {
        [
            &self.bit,
            &self.product_left,
            &self.product_right,
            &self.linear_left,
            &self.linear_right,
            &self.output,
            &self.extension_a,
            &self.extension_b,
            &self.pad,
            &self.active,
            &self.fingerprint_a,
            &self.fingerprint_b,
            &self.value_a,
            &self.value_b,
            &self.value,
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct NebulaRow {
    id: NebulaRowId,
    pub(super) images: NebulaImages,
}

impl NebulaRow {
    pub(super) fn family(&self) -> NebulaFamily {
        self.id.family
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct NebulaProgram {
    matrix_count: usize,
    strict_degree_bound: usize,
    column_count: usize,
    public_end: usize,
    pub(super) rows: Vec<NebulaRow>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct CombinedLayout {
    pub(super) row_variables: usize,
    pub(super) native_logical_width: usize,
    pub(super) native_rows: usize,
    pub(super) native_public_width: usize,
    pub(super) combined_logical_width: usize,
    pub(super) combined_public_width: usize,
    pub(super) nebula_column_count: usize,
    pub(super) nebula_public_end: usize,
    pub(super) nebula_private_width: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CombinedRelation {
    matrix_count: usize,
    strict_degree_bound: usize,
    fresh_source_count: usize,
    running_source_count: usize,
    polynomial: Vec<CombinedPolynomialTerm>,
    layout: CombinedLayout,
    application: NebulaProgram,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CombinedManifestWire {
    schema: u64,
    format: String,
    goldilocks_modulus: u64,
    ajtai_setup: Value,
    core: Value,
    relation: CombinedRelation,
    terminal_r1cs: TerminalR1csDescriptor,
}

/// One schema-v4 manifest after all structural checks pass.
#[derive(Debug)]
pub struct LeanNebulaCombinedManifest {
    core: LeanNativeCcsManifest,
    wire: CombinedManifestWire,
}

/// Exact nineteen-matrix relation and one assignment emitted from a
/// validated Lean manifest.
pub struct NebulaCombinedEmission {
    structure: Structure,
    assignment: Vec<F>,
    native_columns: HashMap<ColumnId, usize>,
    logical_width: usize,
    public_width: usize,
}

impl NebulaCombinedEmission {
    /// Read the exact combined CCS structure.
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    /// Read the public-plus-witness assignment in combined column order.
    pub fn assignment(&self) -> &[F] {
        &self.assignment
    }

    /// Read the logical width before final Phi81 block completion.
    pub fn logical_width(&self) -> usize {
        self.logical_width
    }

    /// Read the complete public carrier width.
    pub fn public_width(&self) -> usize {
        self.public_width
    }

    /// Locate one native Step column in the combined assignment.
    pub fn native_column_index(&self, column: &ColumnId) -> Option<usize> {
        self.native_columns.get(column).copied()
    }

    /// Check the exact combined relation against this assignment.
    pub fn is_satisfied(&self) -> bool {
        check_ccs_rowwise_zero(
            &self.structure,
            &self.assignment[..self.public_width],
            &self.assignment[self.public_width..],
        )
        .is_ok()
    }
}

impl LeanNebulaCombinedManifest {
    /// Decode and validate one complete Lean-owned combined manifest.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, LeanManifestError> {
        let wire: CombinedManifestWire = serde_json::from_slice(bytes)?;
        if wire.schema != LEAN_NEBULA_COMBINED_MANIFEST_SCHEMA_VERSION {
            return Err(LeanManifestError::UnsupportedSchema {
                found: wire.schema,
                expected: LEAN_NEBULA_COMBINED_MANIFEST_SCHEMA_VERSION,
            });
        }
        if wire.format != LEAN_NEBULA_COMBINED_MANIFEST_FORMAT {
            return Err(invalid("format", format!("unsupported format {:?}", wire.format)));
        }
        if wire.goldilocks_modulus != GOLDILOCKS_MODULUS {
            return Err(invalid(
                "goldilocks_modulus",
                format!("got {}, expected {GOLDILOCKS_MODULUS}", wire.goldilocks_modulus),
            ));
        }
        let core = validate_core(&wire)?;
        validate_relation(&wire, &core)?;
        Ok(Self { core, wire })
    }

    /// Read the validated four-matrix native Step core.
    pub fn core(&self) -> &LeanNativeCcsManifest {
        &self.core
    }

    /// Read the exact combined matrix arity.
    pub fn matrix_count(&self) -> usize {
        self.wire.relation.matrix_count
    }

    /// Read the exact strict polynomial degree bound.
    pub fn strict_degree_bound(&self) -> usize {
        self.wire.relation.strict_degree_bound
    }

    /// Read the complete combined public carrier width.
    pub fn public_carrier_width(&self) -> usize {
        self.wire.relation.layout.combined_public_width
    }

    /// Read the exact application-owned private witness width.
    pub fn nebula_private_width(&self) -> usize {
        self.wire.relation.layout.nebula_private_width
    }

    pub(super) fn running_claim_count(&self) -> usize {
        self.wire.relation.running_source_count
    }

    pub(super) fn fresh_claim_count(&self) -> usize {
        self.wire.relation.fresh_source_count
    }

    /// Read the setup seed selected by the Lean manifest.
    pub fn ajtai_setup_seed(&self) -> [u8; 32] {
        self.core.ajtai_setup_seed()
    }

    /// Read the direct terminal relation descriptor.
    pub fn terminal_r1cs(&self) -> TerminalR1csDescriptor {
        self.wire.terminal_r1cs
    }

    pub(super) fn application_rows(&self) -> &[NebulaRow] {
        &self.wire.relation.application.rows
    }

    pub(super) fn combined_layout(&self) -> CombinedLayout {
        self.wire.relation.layout
    }

    pub(super) fn terminal_structure(&self) -> Result<Structure, LeanNebulaCombinedEmissionError> {
        let (native_columns, _) = native_column_order(&self.core)?;
        build_structure(&self.core, &self.wire.relation, &native_columns)
    }

    /// Emit the exact combined relation and one assignment.
    ///
    /// `public_values` is the complete combined public carrier. The native
    /// callback supplies every non-constant native Step column. Shared native
    /// link coordinates must equal the matching public prefix. The Nebula
    /// private slice starts at its exact private source coordinate.
    pub fn emit(
        &self,
        public_values: &[F],
        mut native_values: impl FnMut(&ColumnId) -> Option<F>,
        nebula_private: &[F],
    ) -> Result<NebulaCombinedEmission, LeanNebulaCombinedEmissionError> {
        let layout = self.wire.relation.layout;
        if public_values.len() != layout.combined_public_width {
            return Err(LeanNebulaCombinedEmissionError::PublicWidth {
                found: public_values.len(),
                expected: layout.combined_public_width,
            });
        }
        if nebula_private.len() != layout.nebula_private_width {
            return Err(LeanNebulaCombinedEmissionError::NebulaPrivateWidth {
                found: nebula_private.len(),
                expected: layout.nebula_private_width,
            });
        }
        if public_values.first() != Some(&F::ONE) {
            return Err(LeanNebulaCombinedEmissionError::SharedValue { column: 0 });
        }
        let fixed_start = layout
            .combined_public_width
            .checked_sub(FIXED_PUBLIC_PADDING)
            .ok_or_else(|| LeanNebulaCombinedEmissionError::InvalidStructure("public padding underflow".into()))?;
        if public_values[fixed_start..]
            .iter()
            .any(|value| *value != F::ZERO)
        {
            return Err(LeanNebulaCombinedEmissionError::SharedValue { column: fixed_start });
        }

        let (native_columns, native_order) = native_column_order(&self.core)?;
        let carrier_width = phi81_carrier_width(layout.combined_logical_width)
            .map_err(|error| LeanNebulaCombinedEmissionError::InvalidStructure(error.to_string()))?;
        let mut assignment = vec![F::ZERO; carrier_width];
        assignment[..layout.combined_public_width].copy_from_slice(public_values);
        for (native_index, column) in native_order.iter().enumerate() {
            let combined_index = map_native_index(layout, native_index);
            let value = if *column == &self.core.step_program().one {
                F::ONE
            } else {
                native_values(column).ok_or_else(|| LeanNebulaCombinedEmissionError::MissingNativeValue {
                    column: (*column).clone(),
                })?
            };
            if native_index < layout.native_public_width && assignment[combined_index] != value {
                return Err(LeanNebulaCombinedEmissionError::SharedValue { column: combined_index });
            }
            assignment[combined_index] = value;
        }
        let private_start = nebula_private_start(layout);
        assignment[private_start..private_start + nebula_private.len()].copy_from_slice(nebula_private);

        let structure = build_structure(&self.core, &self.wire.relation, &native_columns)?;
        Ok(NebulaCombinedEmission {
            structure,
            assignment,
            native_columns: native_columns
                .into_iter()
                .map(|(column, index)| (column, map_native_index(layout, index)))
                .collect(),
            logical_width: layout.combined_logical_width,
            public_width: layout.combined_public_width,
        })
    }
}

fn validate_core(wire: &CombinedManifestWire) -> Result<LeanNativeCcsManifest, LeanManifestError> {
    let core = wire
        .core
        .as_object()
        .ok_or_else(|| invalid("core", "must be an object"))?;
    let (native_rows, native_columns) = core_shape(core)?;
    let row_variables = minimal_row_variables(native_rows)?;
    let fresh_relation_rows = checked_mul("terminal_r1cs.fresh_relation_rows", 2, native_rows)?;
    let cost = terminal_cost_values(
        native_columns,
        NATIVE_MATRIX_COUNT,
        5,
        14,
        1,
        fresh_relation_rows,
        native_rows,
    )?;
    let mut synthetic = core.clone();
    synthetic.insert("schema".into(), json!(LEAN_NATIVE_CCS_MANIFEST_SCHEMA_VERSION));
    synthetic.insert("format".into(), json!(LEAN_NATIVE_CCS_MANIFEST_FORMAT));
    synthetic.insert("goldilocks_modulus".into(), json!(GOLDILOCKS_MODULUS));
    synthetic.insert("ajtai_setup".into(), wire.ajtai_setup.clone());
    synthetic.insert(
        "profile".into(),
        json!({
            "name": "fixed_one_plain_270",
            "matrix_count": 4,
            "fresh_source_count": 1,
            "running_source_count": 14,
            "public_carrier_width": 270,
            "fresh_legacy_width": 257,
            "fresh_completion_width": 13,
            "running_carrier_width": 270,
            "poseidon_width": 8,
            "poseidon_rate": 4,
            "poseidon_capacity": 4,
            "poseidon_digest_width": 4,
            "binding_preimage_width": 23,
            "decomposition_base": 2,
            "decomposition_children": 14
        }),
    );
    synthetic.insert(
        "terminal_r1cs".into(),
        json!({
            "row_variables": row_variables,
            "logical_width": native_columns,
            "recursive_rows": native_rows,
            "fresh_relation_rows": fresh_relation_rows,
            "fresh_relation_auxiliary_columns": native_rows,
            "matrix_count": 4,
            "public_ring_columns": 5,
            "verifier_rows": TERMINAL_COMMITMENT_ROWS,
            "cost": cost_json(cost)
        }),
    );
    LeanNativeCcsManifest::from_json_slice(&serde_json::to_vec(&Value::Object(synthetic))?)
}

fn validate_relation(wire: &CombinedManifestWire, core: &LeanNativeCcsManifest) -> Result<(), LeanManifestError> {
    let relation = &wire.relation;
    let layout = relation.layout;
    if relation.matrix_count != COMBINED_MATRIX_COUNT {
        return Err(invalid("relation.matrix_count", "must equal nineteen"));
    }
    if relation.strict_degree_bound != COMBINED_STRICT_DEGREE_BOUND {
        return Err(invalid("relation.strict_degree_bound", "must equal five"));
    }
    if relation.fresh_source_count != 1 || relation.running_source_count != 14 {
        return Err(invalid(
            "relation",
            "must select one fresh and fourteen running sources",
        ));
    }
    if relation.polynomial != expected_polynomial() {
        return Err(invalid(
            "relation.polynomial",
            "does not equal the Lean nineteen-role polynomial",
        ));
    }
    validate_application(&relation.application)?;
    let (_, native_columns) = core_shape(
        wire.core
            .as_object()
            .ok_or_else(|| invalid("core", "must be an object"))?,
    )?;
    let native_rows = core.step_program().row_count();
    if layout.native_logical_width != native_columns || layout.native_rows != native_rows {
        return Err(invalid(
            "relation.layout",
            "native dimensions do not match the validated Step program",
        ));
    }
    if layout.native_public_width != NATIVE_PUBLIC_WIDTH {
        return Err(invalid(
            "relation.layout.native_public_width",
            "must equal the 257-coordinate F-prime link",
        ));
    }
    if layout.nebula_column_count != relation.application.column_count
        || layout.nebula_public_end != relation.application.public_end
    {
        return Err(invalid(
            "relation.layout",
            "Nebula dimensions do not match the application program",
        ));
    }
    let private_width = relation
        .application
        .column_count
        .checked_sub(relation.application.public_end)
        .ok_or_else(|| invalid("relation.application.public_end", "exceeds column_count"))?;
    if layout.nebula_private_width != private_width {
        return Err(invalid(
            "relation.layout.nebula_private_width",
            "does not match the application private suffix",
        ));
    }
    let native_private = layout
        .native_logical_width
        .checked_sub(layout.native_public_width)
        .ok_or_else(|| invalid("relation.layout.native_public_width", "exceeds native width"))?;
    let expected_logical = checked_add(
        "relation.layout.combined_logical_width",
        checked_add(
            "relation.layout.combined_logical_width",
            layout.combined_public_width,
            native_private,
        )?,
        layout.nebula_private_width,
    )?;
    if layout.combined_logical_width != expected_logical {
        return Err(invalid(
            "relation.layout.combined_logical_width",
            "does not equal public plus both private suffixes",
        ));
    }
    let nebula_public_end = checked_add(
        "relation.layout.combined_public_width",
        layout.native_public_width,
        relation.application.public_end.saturating_sub(1),
    )?;
    if nebula_public_end > layout.combined_public_width
        || layout.combined_public_width % PHI81_RING_DEGREE != 0
        || layout.combined_public_width < FIXED_PUBLIC_PADDING
    {
        return Err(invalid(
            "relation.layout.combined_public_width",
            "does not contain the aligned application public carrier",
        ));
    }
    let recursive_rows = checked_add(
        "terminal_r1cs.recursive_rows",
        layout.native_rows,
        relation.application.rows.len(),
    )?;
    validate_minimal_row_domain(layout.row_variables, recursive_rows)?;
    validate_terminal_descriptor(wire.terminal_r1cs, relation, recursive_rows)?;
    Ok(())
}

fn validate_application(program: &NebulaProgram) -> Result<(), LeanManifestError> {
    if program.matrix_count != NEBULA_MATRIX_COUNT {
        return Err(invalid("relation.application.matrix_count", "must equal fifteen"));
    }
    if program.strict_degree_bound != COMBINED_STRICT_DEGREE_BOUND {
        return Err(invalid("relation.application.strict_degree_bound", "must equal five"));
    }
    if program.public_end == 0 || program.public_end > program.column_count {
        return Err(invalid(
            "relation.application.public_end",
            "must select a nonempty prefix within column_count",
        ));
    }
    for (position, row) in program.rows.iter().enumerate() {
        if row.id.position != position {
            return Err(invalid(
                format!("relation.application.rows[{position}].id.position"),
                "does not equal the exact emitted row position",
            ));
        }
        for (matrix, combination) in row.images.combinations().iter().enumerate() {
            for (term_index, term) in combination.iter().enumerate() {
                if term.column >= program.column_count {
                    return Err(invalid(
                        format!("relation.application.rows[{position}].images[{matrix}][{term_index}].column"),
                        "is outside column_count",
                    ));
                }
                if term.coefficient == 0 || term.coefficient >= GOLDILOCKS_MODULUS {
                    return Err(invalid(
                        format!("relation.application.rows[{position}].images[{matrix}][{term_index}].coefficient"),
                        "must be a nonzero canonical Goldilocks representative",
                    ));
                }
            }
        }
        validate_family_shape(row, position)?;
    }
    Ok(())
}

fn validate_family_shape(row: &NebulaRow, position: usize) -> Result<(), LeanManifestError> {
    let images = &row.images;
    let valid = match row.id.family {
        NebulaFamily::OperationBit | NebulaFamily::InitialScanBit | NebulaFamily::FinalScanBit => {
            images.bit.len() == 1
                && images.bit[0].coefficient == 1
                && images.product_left.is_empty()
                && images.product_right.is_empty()
                && images.linear_left.is_empty()
                && images.linear_right.is_empty()
                && extension_images_empty(images)
        }
        NebulaFamily::ReadWrite
        | NebulaFamily::TimestampOrder
        | NebulaFamily::RomWrite
        | NebulaFamily::RomRange
        | NebulaFamily::Padding => {
            images.bit.is_empty()
                && images.linear_left.is_empty()
                && images.linear_right.is_empty()
                && extension_images_empty(images)
        }
        NebulaFamily::Filler
        | NebulaFamily::OperationCount
        | NebulaFamily::BoundaryTimestamp
        | NebulaFamily::BoundaryProduct => {
            images.bit.is_empty()
                && images.product_left.is_empty()
                && images.product_right.is_empty()
                && extension_images_empty(images)
        }
        family if family.is_extension() => {
            images.bit.is_empty()
                && images.product_left.is_empty()
                && images.product_right.is_empty()
                && images.linear_left.is_empty()
                && images.linear_right.is_empty()
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(
            format!("relation.application.rows[{position}].images"),
            "contains a matrix role outside its declared Nebula row family",
        ))
    }
}

fn extension_images_empty(images: &NebulaImages) -> bool {
    images.output.is_empty()
        && images.extension_a.is_empty()
        && images.extension_b.is_empty()
        && images.pad.is_empty()
        && images.active.is_empty()
        && images.fingerprint_a.is_empty()
        && images.fingerprint_b.is_empty()
        && images.value_a.is_empty()
        && images.value_b.is_empty()
        && images.value.is_empty()
}

fn application_terminal_counts(program: &NebulaProgram) -> Result<(usize, usize), LeanManifestError> {
    let extension_rows = program
        .rows
        .iter()
        .filter(|row| row.id.family.is_extension())
        .count();
    let added = checked_mul("terminal_r1cs.fresh_relation_rows", 5, extension_rows)?;
    Ok((
        checked_add("terminal_r1cs.fresh_relation_rows", program.rows.len(), added)?,
        added,
    ))
}

fn validate_terminal_descriptor(
    descriptor: TerminalR1csDescriptor,
    relation: &CombinedRelation,
    recursive_rows: usize,
) -> Result<(), LeanManifestError> {
    let layout = relation.layout;
    if descriptor.row_variables() != layout.row_variables
        || descriptor.logical_width() != layout.combined_logical_width
        || descriptor.recursive_rows() != recursive_rows
        || descriptor.matrix_count() != relation.matrix_count
        || descriptor.public_ring_columns() * PHI81_RING_DEGREE != layout.combined_public_width
        || descriptor.verifier_rows() != TERMINAL_COMMITMENT_ROWS
    {
        return Err(invalid(
            "terminal_r1cs",
            "does not match the combined relation dimensions",
        ));
    }
    let (application_rows, application_auxiliary) = application_terminal_counts(&relation.application)?;
    let native_rows = checked_mul("terminal_r1cs.fresh_relation_rows", 2, layout.native_rows)?;
    let expected_fresh_rows = checked_add("terminal_r1cs.fresh_relation_rows", native_rows, application_rows)?;
    let expected_fresh_auxiliary = checked_add(
        "terminal_r1cs.fresh_relation_auxiliary_columns",
        layout.native_rows,
        application_auxiliary,
    )?;
    if descriptor.fresh_relation_rows() != expected_fresh_rows
        || descriptor.fresh_relation_auxiliary_columns() != expected_fresh_auxiliary
    {
        return Err(invalid(
            "terminal_r1cs",
            "does not match the exact native-plus-Nebula terminal lowering",
        ));
    }
    let expected = terminal_cost_values(
        layout.combined_logical_width,
        relation.matrix_count,
        descriptor.public_ring_columns(),
        relation.running_source_count,
        relation.fresh_source_count,
        expected_fresh_rows,
        expected_fresh_auxiliary,
    )?;
    let actual = descriptor.cost();
    if cost_tuple(actual) != expected {
        return Err(invalid(
            "terminal_r1cs.cost",
            "does not match the Lean terminal cost formula",
        ));
    }
    Ok(())
}

fn build_structure(
    core: &LeanNativeCcsManifest,
    relation: &CombinedRelation,
    native_columns: &HashMap<ColumnId, usize>,
) -> Result<Structure, LeanNebulaCombinedEmissionError> {
    let layout = relation.layout;
    let row_domain = row_domain(layout.row_variables)
        .map_err(|error| LeanNebulaCombinedEmissionError::InvalidStructure(error.to_string()))?;
    let carrier_width = phi81_carrier_width(layout.combined_logical_width)
        .map_err(|error| LeanNebulaCombinedEmissionError::InvalidStructure(error.to_string()))?;
    let mut triplets: [Vec<(usize, usize, F)>; COMBINED_MATRIX_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut row_index = 0usize;
    for receipt in &core.step_program().receipts {
        let selector_index = *native_columns.get(&receipt.selector).ok_or_else(|| {
            LeanNebulaCombinedEmissionError::UnknownNativeColumn {
                column: receipt.selector.clone(),
            }
        })?;
        let selector_index = map_native_index(layout, selector_index);
        for row in &receipt.rows {
            append_native_terms(&mut triplets[0], row_index, &row.a, native_columns, layout)?;
            append_native_terms(&mut triplets[1], row_index, &row.b, native_columns, layout)?;
            append_native_terms(&mut triplets[2], row_index, &row.c, native_columns, layout)?;
            triplets[3].push((row_index, selector_index, F::ONE));
            row_index += 1;
        }
    }
    if row_index != layout.native_rows {
        return Err(LeanNebulaCombinedEmissionError::InvalidStructure(
            "native row count changed after validation".into(),
        ));
    }
    for row in &relation.application.rows {
        let combined_row = layout.native_rows + row.id.position;
        for (matrix, combination) in row.images.combinations().iter().enumerate() {
            for term in *combination {
                triplets[NATIVE_MATRIX_COUNT + matrix].push((
                    combined_row,
                    map_nebula_index(layout, term.column),
                    F::from_u64(term.coefficient),
                ));
            }
        }
    }
    let matrices = triplets
        .into_iter()
        .map(|entries| CcsMatrix::Csc(CscMat::from_counted_triplets(entries, row_domain, carrier_width)))
        .collect();
    let polynomial = SparsePoly::new(
        COMBINED_MATRIX_COUNT,
        relation
            .polynomial
            .iter()
            .map(|term| Term {
                coeff: F::from_u64(term.coefficient),
                exps: term.exponents.clone(),
            })
            .collect(),
    );
    Structure::new_sparse(matrices, polynomial)
        .map_err(|error| LeanNebulaCombinedEmissionError::InvalidStructure(error.to_string()))
}

fn append_native_terms(
    target: &mut Vec<(usize, usize, F)>,
    row: usize,
    terms: &[ManifestTerm],
    native_columns: &HashMap<ColumnId, usize>,
    layout: CombinedLayout,
) -> Result<(), LeanNebulaCombinedEmissionError> {
    for term in terms {
        let native_index =
            *native_columns
                .get(&term.column)
                .ok_or_else(|| LeanNebulaCombinedEmissionError::UnknownNativeColumn {
                    column: term.column.clone(),
                })?;
        target.push((
            row,
            map_native_index(layout, native_index),
            F::from_u64(term.coefficient),
        ));
    }
    Ok(())
}

fn native_column_order(
    core: &LeanNativeCcsManifest,
) -> Result<(HashMap<ColumnId, usize>, Vec<&ColumnId>), LeanNebulaCombinedEmissionError> {
    let mut indices = HashMap::new();
    let mut order = Vec::new();
    for allocation in core
        .step_program()
        .receipts
        .iter()
        .flat_map(|receipt| &receipt.allocations)
    {
        let index = order.len();
        indices.insert(allocation.id.clone(), index);
        order.push(&allocation.id);
    }
    if indices.get(&core.step_program().one) != Some(&0) {
        return Err(LeanNebulaCombinedEmissionError::InvalidStructure(
            "native constant one is not logical column zero".into(),
        ));
    }
    Ok((indices, order))
}

pub(super) fn map_native_index(layout: CombinedLayout, column: usize) -> usize {
    if column < layout.native_public_width {
        column
    } else {
        layout.combined_public_width + (column - layout.native_public_width)
    }
}

fn nebula_private_start(layout: CombinedLayout) -> usize {
    layout.combined_public_width + (layout.native_logical_width - layout.native_public_width)
}

pub(super) fn map_nebula_index(layout: CombinedLayout, column: usize) -> usize {
    if column == 0 {
        0
    } else if column < layout.nebula_public_end {
        layout.native_public_width + (column - 1)
    } else {
        nebula_private_start(layout) + (column - layout.nebula_public_end)
    }
}

fn core_shape(core: &Map<String, Value>) -> Result<(usize, usize), LeanManifestError> {
    let receipts = core
        .get("step_program")
        .and_then(|program| program.get("receipts"))
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("core.step_program.receipts", "must be an array"))?;
    let mut rows = 0usize;
    let mut columns = 0usize;
    for (index, receipt) in receipts.iter().enumerate() {
        let allocations = receipt
            .get("allocations")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid(
                    format!("core.step_program.receipts[{index}].allocations"),
                    "must be an array",
                )
            })?;
        let receipt_rows = receipt
            .get("rows")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("core.step_program.receipts[{index}].rows"), "must be an array"))?;
        columns = checked_add("core.step_program.allocations", columns, allocations.len())?;
        rows = checked_add("core.step_program.rows", rows, receipt_rows.len())?;
    }
    Ok((rows, columns))
}

fn expected_polynomial() -> Vec<CombinedPolynomialTerm> {
    let minus_one = GOLDILOCKS_MODULUS - 1;
    let term = |coefficient: u64, powers: &[(usize, u32)]| {
        let mut exponents = vec![0u32; COMBINED_MATRIX_COUNT];
        for &(index, exponent) in powers {
            exponents[index] = exponent;
        }
        CombinedPolynomialTerm { coefficient, exponents }
    };
    vec![
        term(1, &[(0, 1), (1, 1), (3, 1)]),
        term(minus_one, &[(2, 1), (3, 1)]),
        term(1, &[(4, 2)]),
        term(minus_one, &[(4, 1)]),
        term(1, &[(5, 1), (6, 1)]),
        term(1, &[(7, 1)]),
        term(minus_one, &[(8, 1)]),
        term(minus_one, &[(9, 1)]),
        term(1, &[(10, 1), (12, 1)]),
        term(1, &[(10, 1), (13, 1), (14, 1)]),
        term(minus_one, &[(10, 1), (13, 1), (16, 1), (18, 1)]),
        term(1, &[(11, 1), (13, 1), (15, 1)]),
        term(minus_one, &[(11, 1), (13, 1), (17, 1), (18, 1)]),
    ]
}

fn checked_add(path: &str, left: usize, right: usize) -> Result<usize, LeanManifestError> {
    left.checked_add(right)
        .ok_or_else(|| invalid(path, "count overflow"))
}

fn checked_mul(path: &str, left: usize, right: usize) -> Result<usize, LeanManifestError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(path, "count overflow"))
}

fn row_domain(row_variables: usize) -> Result<usize, LeanManifestError> {
    let shift = u32::try_from(row_variables).map_err(|_| invalid("row_variables", "row domain overflow"))?;
    1usize
        .checked_shl(shift)
        .ok_or_else(|| invalid("row_variables", "row domain overflow"))
}

fn minimal_row_variables(rows: usize) -> Result<usize, LeanManifestError> {
    if rows <= 1 {
        return Ok(0);
    }
    let variables = usize::BITS as usize - (rows - 1).leading_zeros() as usize;
    row_domain(variables)?;
    Ok(variables)
}

fn validate_minimal_row_domain(row_variables: usize, rows: usize) -> Result<(), LeanManifestError> {
    let capacity = row_domain(row_variables)?;
    if rows > capacity || (row_variables > 0 && rows <= capacity / 2) {
        return Err(invalid(
            "relation.layout.row_variables",
            "is not the least power-of-two domain for the combined rows",
        ));
    }
    Ok(())
}

fn phi81_carrier_width(logical_width: usize) -> Result<usize, LeanManifestError> {
    let blocks = checked_add("logical_width", logical_width, PHI81_RING_DEGREE - 1)? / PHI81_RING_DEGREE;
    checked_mul("logical_width", blocks, PHI81_RING_DEGREE)
}

type CostValues = (usize, usize, usize, usize);

fn terminal_cost_values(
    logical_width: usize,
    matrix_count: usize,
    public_ring_columns: usize,
    running_claims: usize,
    fresh_claims: usize,
    fresh_relation_rows: usize,
    fresh_relation_auxiliary_columns: usize,
) -> Result<CostValues, LeanManifestError> {
    if fresh_claims != 1 {
        return Err(invalid(
            "fresh_source_count",
            "terminal relation requires one fresh source",
        ));
    }
    let carrier = phi81_carrier_width(logical_width)?;
    let public_width = checked_mul("terminal_cost", public_ring_columns, PHI81_RING_DEGREE)?;
    let verifier_width = checked_mul("terminal_cost", TERMINAL_COMMITMENT_ROWS, PHI81_RING_DEGREE)?;
    let evaluations = checked_mul("terminal_cost", matrix_count, PHI81_RING_DEGREE)?;
    let two_carriers = checked_mul("terminal_cost", 2, carrier)?;
    let running_statement = checked_add(
        "terminal_cost",
        checked_add("terminal_cost", verifier_width, public_width)?,
        checked_mul("terminal_cost", 2, evaluations)?,
    )?;
    let running_rows = checked_add("terminal_cost", running_statement, two_carriers)?;
    let fresh_statement = checked_add("terminal_cost", verifier_width, public_width)?;
    let fresh_rows = checked_add(
        "terminal_cost",
        checked_add("terminal_cost", fresh_statement, two_carriers)?,
        fresh_relation_rows,
    )?;
    let claims = checked_add("terminal_cost", running_claims, fresh_claims)?;
    Ok((
        checked_add(
            "terminal_cost",
            checked_mul("terminal_cost", running_claims, running_rows)?,
            fresh_rows,
        )?,
        checked_mul("terminal_cost", claims, carrier)?,
        checked_add(
            "terminal_cost",
            checked_add(
                "terminal_cost",
                1,
                checked_mul("terminal_cost", running_claims, running_statement)?,
            )?,
            fresh_statement,
        )?,
        checked_add(
            "terminal_cost",
            checked_mul("terminal_cost", claims, carrier)?,
            fresh_relation_auxiliary_columns,
        )?,
    ))
}

fn cost_tuple(cost: ManifestCost) -> CostValues {
    (
        cost.recurring_rows(),
        cost.committed_columns(),
        cost.public_columns(),
        cost.auxiliary_columns(),
    )
}

fn cost_json(cost: CostValues) -> Value {
    json!({
        "recurring_rows": cost.0,
        "committed_columns": cost.1,
        "public_columns": cost.2,
        "auxiliary_columns": cost.3
    })
}
