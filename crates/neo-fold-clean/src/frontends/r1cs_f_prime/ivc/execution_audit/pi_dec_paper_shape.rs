//! Live strict-PiDEC public-shape evidence for the paper NIFS boundary.
//!
//! This module joins only the 270 active public-X coordinates of the PiRLC
//! parent and each of the fourteen ordered PiDEC children.  It deliberately
//! does not walk the private assignment, child `y_zcol` sidecars, or digests.
//! The semantic endpoint is the independent `UniformXAccepted` predicate:
//! 270 radix-recomposition rows plus 270 sixteen-row sign/digit blocks.

use std::collections::BTreeSet;
use std::ops::Range;

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::{normalized_target_column, R1csIvcBranch, R1csIvcSelectorWriteAudit};
use crate::engine::r1cs_circuit::{PiDecCanonicalXReceipt, R1csBuilder, Var};
use crate::frontends::r1cs_f_prime::lowering::normalized_source_column;
use crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use crate::paper::nifs::NifsProof;
use crate::paper::relations::CeClaim;

pub const PI_DEC_PAPER_CHILD_COUNT: usize = 14;
pub const PI_DEC_PAPER_PUBLIC_COORDINATES: usize = 270;
pub const PI_DEC_PAPER_ACTIVE_X_COLUMNS: usize = PI_DEC_PAPER_PUBLIC_COORDINATES / D;
pub const PI_DEC_PAPER_EVALUATION_ARITY: usize = 13;
pub const PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE: usize = PI_DEC_PAPER_CHILD_COUNT + 2;

/// The one production profile for which this execution join is claimed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcPiDecPaperShapeProfile {
    ActiveFPrimeRadix2,
}

/// Native source of one strict public-X pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcPiDecPaperXOwner {
    PiRlcParent,
    PiDecChild(usize),
}

/// One public-X coordinate joined across native, builder, and normalized
/// source-assignment surfaces.
///
/// `public_column` uses the 270-coordinate public-vector order.  The strict
/// gadget stores X row-major, so its active index is
/// `x_row * active_x_columns + x_active_column`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcPiDecPaperXPinAudit {
    owner: R1csIvcPiDecPaperXOwner,
    public_column: usize,
    x_row: usize,
    x_active_column: usize,
    x_active_index: usize,
    builder_column: usize,
    normalized_column: usize,
    native_value: F,
    builder_value: F,
    normalized_value: F,
}

/// One generated trace column joined to the normalized source assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcPiDecPaperTraceColumnAudit {
    builder_column: usize,
    normalized_column: usize,
    expected_value: F,
    builder_value: F,
    normalized_value: F,
}

impl R1csIvcPiDecPaperTraceColumnAudit {
    pub fn builder_column(&self) -> usize {
        self.builder_column
    }

    pub fn normalized_column(&self) -> usize {
        self.normalized_column
    }

    pub fn expected_value(&self) -> F {
        self.expected_value
    }

    pub fn builder_value(&self) -> F {
        self.builder_value
    }

    pub fn normalized_value(&self) -> F {
        self.normalized_value
    }
}

/// Exact source-row and trace-column ownership for one active parent-X
/// coordinate.  `x_row` is a matrix row; the two row fields are the actual
/// strict R1CS source rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcPiDecCanonicalXCoordinateAudit {
    public_column: usize,
    x_row: usize,
    x_active_column: usize,
    x_active_index: usize,
    recomposition_source_row: usize,
    canonicality_source_rows: Range<usize>,
    sign: R1csIvcPiDecPaperTraceColumnAudit,
    sign_output: R1csIvcPiDecPaperTraceColumnAudit,
}

impl R1csIvcPiDecCanonicalXCoordinateAudit {
    pub fn public_column(&self) -> usize {
        self.public_column
    }

    pub fn x_coordinate(&self) -> (usize, usize) {
        (self.x_row, self.x_active_column)
    }

    pub fn x_active_index(&self) -> usize {
        self.x_active_index
    }

    pub fn recomposition_source_row(&self) -> usize {
        self.recomposition_source_row
    }

    pub fn canonicality_source_rows(&self) -> Range<usize> {
        self.canonicality_source_rows.clone()
    }

    pub fn sign(&self) -> R1csIvcPiDecPaperTraceColumnAudit {
        self.sign
    }

    pub fn sign_output(&self) -> R1csIvcPiDecPaperTraceColumnAudit {
        self.sign_output
    }
}

impl R1csIvcPiDecPaperXPinAudit {
    pub fn owner(&self) -> R1csIvcPiDecPaperXOwner {
        self.owner
    }

    pub fn public_column(&self) -> usize {
        self.public_column
    }

    pub fn x_coordinate(&self) -> (usize, usize) {
        (self.x_row, self.x_active_column)
    }

    pub fn x_active_index(&self) -> usize {
        self.x_active_index
    }

    pub fn builder_column(&self) -> usize {
        self.builder_column
    }

    /// Column of the normalized source assignment passed to the selective
    /// lowering.  The final low-norm disposition is intentionally a separate
    /// compiler-provenance join.
    pub fn normalized_column(&self) -> usize {
        self.normalized_column
    }

    pub fn native_value(&self) -> F {
        self.native_value
    }

    pub fn builder_value(&self) -> F {
        self.builder_value
    }

    pub fn normalized_value(&self) -> F {
        self.normalized_value
    }
}

/// Exact live execution evidence needed by the compact paper-output shape
/// bridge.  The child vector is in native PiDEC proof order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcPiDecPaperShapeExecutionAudit {
    profile: R1csIvcPiDecPaperShapeProfile,
    strict_rows: Range<usize>,
    x_recomposition_rows: Range<usize>,
    x_canonicality_rows: Range<usize>,
    recursive_selector_logical_column: usize,
    recursive_selector_value: F,
    parent_pins: Vec<R1csIvcPiDecPaperXPinAudit>,
    child_pins: Vec<Vec<R1csIvcPiDecPaperXPinAudit>>,
    canonical_x_coordinates: Vec<R1csIvcPiDecCanonicalXCoordinateAudit>,
    child_evaluation_arities: Vec<usize>,
}

impl R1csIvcPiDecPaperShapeExecutionAudit {
    pub fn profile(&self) -> R1csIvcPiDecPaperShapeProfile {
        self.profile
    }

    pub fn strict_rows(&self) -> Range<usize> {
        self.strict_rows.clone()
    }

    pub fn x_recomposition_rows(&self) -> Range<usize> {
        self.x_recomposition_rows.clone()
    }

    pub fn x_canonicality_rows(&self) -> Range<usize> {
        self.x_canonicality_rows.clone()
    }

    pub fn recursive_selector_logical_column(&self) -> usize {
        self.recursive_selector_logical_column
    }

    pub fn recursive_selector_value(&self) -> F {
        self.recursive_selector_value
    }

    pub fn parent_pins(&self) -> &[R1csIvcPiDecPaperXPinAudit] {
        &self.parent_pins
    }

    pub fn child_pins(&self) -> &[Vec<R1csIvcPiDecPaperXPinAudit>] {
        &self.child_pins
    }

    pub fn canonical_x_coordinates(&self) -> &[R1csIvcPiDecCanonicalXCoordinateAudit] {
        &self.canonical_x_coordinates
    }

    pub fn child_evaluation_arities(&self) -> &[usize] {
        &self.child_evaluation_arities
    }

    /// Recheck all proof-free cardinality, order, and cross-surface value
    /// invariants without consulting a digest or sidecar.
    pub fn validate(&self) -> Result<(), String> {
        if self.profile != R1csIvcPiDecPaperShapeProfile::ActiveFPrimeRadix2
            || self.strict_rows.is_empty()
            || self.x_recomposition_rows.start < self.strict_rows.start
            || self.x_recomposition_rows.end > self.strict_rows.end
            || self.x_recomposition_rows.len() != PI_DEC_PAPER_PUBLIC_COORDINATES
            || self.x_canonicality_rows.start < self.strict_rows.start
            || self.x_canonicality_rows.end > self.strict_rows.end
            || self.x_canonicality_rows.len()
                != PI_DEC_PAPER_PUBLIC_COORDINATES * PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE
            || self.recursive_selector_value != F::ONE
            || self.parent_pins.len() != PI_DEC_PAPER_PUBLIC_COORDINATES
            || self.child_pins.len() != PI_DEC_PAPER_CHILD_COUNT
            || self
                .child_pins
                .iter()
                .any(|pins| pins.len() != PI_DEC_PAPER_PUBLIC_COORDINATES)
            || self.canonical_x_coordinates.len() != PI_DEC_PAPER_PUBLIC_COORDINATES
            || self.child_evaluation_arities.len() != PI_DEC_PAPER_CHILD_COUNT
            || self
                .child_evaluation_arities
                .iter()
                .any(|&arity| arity != PI_DEC_PAPER_EVALUATION_ARITY)
        {
            return Err("live PiDEC paper-shape header drift".into());
        }

        let mut builder_columns = BTreeSet::new();
        let mut normalized_columns = BTreeSet::new();
        validate_pin_family(
            &self.parent_pins,
            R1csIvcPiDecPaperXOwner::PiRlcParent,
            &mut builder_columns,
            &mut normalized_columns,
        )?;
        for (child, pins) in self.child_pins.iter().enumerate() {
            validate_pin_family(
                pins,
                R1csIvcPiDecPaperXOwner::PiDecChild(child),
                &mut builder_columns,
                &mut normalized_columns,
            )?;
        }
        validate_canonical_x_coordinates(self, &mut builder_columns, &mut normalized_columns)?;
        let expected_pins = (PI_DEC_PAPER_CHILD_COUNT + 1) * PI_DEC_PAPER_PUBLIC_COORDINATES;
        let expected_columns = expected_pins + 2 * PI_DEC_PAPER_PUBLIC_COORDINATES;
        if builder_columns.len() != expected_columns || normalized_columns.len() != expected_columns {
            return Err("live PiDEC public-X columns are not one-to-one".into());
        }
        Ok(())
    }
}

fn validate_canonical_x_coordinates(
    audit: &R1csIvcPiDecPaperShapeExecutionAudit,
    builder_columns: &mut BTreeSet<usize>,
    normalized_columns: &mut BTreeSet<usize>,
) -> Result<(), String> {
    for (public_column, coordinate) in audit.canonical_x_coordinates.iter().enumerate() {
        let x_row = public_column % D;
        let x_active_column = public_column / D;
        let x_active_index = x_row * PI_DEC_PAPER_ACTIVE_X_COLUMNS + x_active_column;
        let canonical_start =
            audit.x_canonicality_rows.start + x_active_index * PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE;
        if coordinate.public_column != public_column
            || coordinate.x_row != x_row
            || coordinate.x_active_column != x_active_column
            || coordinate.x_active_index != x_active_index
            || coordinate.recomposition_source_row != audit.x_recomposition_rows.start + x_active_index
            || coordinate.canonicality_source_rows
                != (canonical_start..canonical_start + PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE)
        {
            return Err(format!(
                "live PiDEC canonical-X coordinate {public_column} row ownership drift"
            ));
        }
        for trace in [coordinate.sign, coordinate.sign_output] {
            if trace.expected_value != trace.builder_value
                || trace.expected_value != trace.normalized_value
                || !builder_columns.insert(trace.builder_column)
                || !normalized_columns.insert(trace.normalized_column)
            {
                return Err(format!(
                    "live PiDEC canonical-X coordinate {public_column} trace mapping drift"
                ));
            }
        }
    }
    Ok(())
}

fn validate_pin_family(
    pins: &[R1csIvcPiDecPaperXPinAudit],
    owner: R1csIvcPiDecPaperXOwner,
    builder_columns: &mut BTreeSet<usize>,
    normalized_columns: &mut BTreeSet<usize>,
) -> Result<(), String> {
    for (public_column, pin) in pins.iter().enumerate() {
        let x_row = public_column % D;
        let x_active_column = public_column / D;
        let x_active_index = x_row * PI_DEC_PAPER_ACTIVE_X_COLUMNS + x_active_column;
        if pin.owner != owner
            || pin.public_column != public_column
            || pin.x_row != x_row
            || pin.x_active_column != x_active_column
            || pin.x_active_index != x_active_index
            || pin.native_value != pin.builder_value
            || pin.native_value != pin.normalized_value
            || !builder_columns.insert(pin.builder_column)
            || !normalized_columns.insert(pin.normalized_column)
        {
            return Err(format!(
                "live PiDEC public-X pin {owner:?}/{public_column} mapping drift"
            ));
        }
    }
    Ok(())
}

pub(super) fn capture_and_validate(
    builder: &R1csBuilder,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    selector_writes: &[R1csIvcSelectorWriteAudit],
    receipt: Option<&PiDecCanonicalXReceipt>,
    nifs: &NifsProof,
) -> Result<R1csIvcPiDecPaperShapeExecutionAudit, String> {
    let receipt = receipt
        .ok_or_else(|| "active recursive execution is missing the outer PiDEC canonical-X receipt".to_string())?;
    validate_receipt_profile(builder, receipt, &nifs.pi_rlc.combined, &nifs.pi_dec.children)?;
    let program = receipt.program();
    let x_recomposition_rows = receipt.recomposition_rows();
    let x_canonicality_rows = receipt.canonicality_rows();
    let recursive_selector = validate_recursive_selector(selector_writes)?;

    let parent_columns = (0..PI_DEC_PAPER_PUBLIC_COORDINATES)
        .map(|active_index| {
            let canonical = program.parent_canonical_column(active_index)?;
            receipt.columns().actual_column(canonical)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "live PiDEC receipt omits a parent public-X column".to_string())?;
    let parent_pins = capture_claim_pins(
        builder,
        public_outputs,
        normalized_assignment,
        &parent_columns,
        &nifs.pi_rlc.combined,
        R1csIvcPiDecPaperXOwner::PiRlcParent,
    )?;
    let mut child_pins = Vec::with_capacity(PI_DEC_PAPER_CHILD_COUNT);
    for (child, native_child) in nifs.pi_dec.children.iter().enumerate() {
        let columns = (0..PI_DEC_PAPER_PUBLIC_COORDINATES)
            .map(|active_index| {
                let canonical = program.child_canonical_column(child, active_index)?;
                receipt.columns().actual_column(canonical)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("live PiDEC receipt omits child {child} public-X columns"))?;
        child_pins.push(capture_claim_pins(
            builder,
            public_outputs,
            normalized_assignment,
            &columns,
            native_child,
            R1csIvcPiDecPaperXOwner::PiDecChild(child),
        )?);
    }
    let canonical_x_coordinates = capture_canonical_x_coordinates(
        builder,
        public_outputs,
        normalized_assignment,
        receipt,
        &nifs.pi_dec.children,
        &x_recomposition_rows,
        &x_canonicality_rows,
    )?;

    let audit = R1csIvcPiDecPaperShapeExecutionAudit {
        profile: R1csIvcPiDecPaperShapeProfile::ActiveFPrimeRadix2,
        strict_rows: receipt.strict_rows(),
        x_recomposition_rows,
        x_canonicality_rows,
        recursive_selector_logical_column: recursive_selector.logical_column(),
        recursive_selector_value: recursive_selector.value(),
        parent_pins,
        child_pins,
        canonical_x_coordinates,
        child_evaluation_arities: nifs
            .pi_dec
            .children
            .iter()
            .map(|child| child.y_ring.len())
            .collect(),
    };
    audit.validate()?;
    Ok(audit)
}

fn validate_receipt_profile(
    builder: &R1csBuilder,
    receipt: &PiDecCanonicalXReceipt,
    parent: &CeClaim,
    children: &[CeClaim],
) -> Result<(), String> {
    let plan = receipt.program().plan();
    if plan.x_rows() != D
        || plan.active_columns() != PI_DEC_PAPER_ACTIVE_X_COLUMNS
        || plan.child_count() != PI_DEC_PAPER_CHILD_COUNT
        || plan.logical_coordinates() != PI_DEC_PAPER_PUBLIC_COORDINATES
        || plan.recomposition_rows() != PI_DEC_PAPER_PUBLIC_COORDINATES
        || plan.canonicality_rows() != PI_DEC_PAPER_PUBLIC_COORDINATES * PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE
        || receipt.strict_rows().is_empty()
        || receipt.strict_rows().end > builder.rows()
        || receipt
            .columns()
            .canonical_to_actual()
            .iter()
            .any(|&column| column >= builder.cols())
        || children.len() != PI_DEC_PAPER_CHILD_COUNT
    {
        return Err("live strict PiDEC receipt is not the active paper-shape profile".into());
    }
    validate_native_claim_shape(parent, "parent")?;
    for (child, native_child) in children.iter().enumerate() {
        validate_native_claim_shape(native_child, &format!("child {child}"))?;
    }
    Ok(())
}

fn validate_native_claim_shape(native: &CeClaim, owner: &str) -> Result<(), String> {
    if native.X.rows() != D
        || native.X.cols() != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN
        || native.m_in != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN
        || native.y_ring.len() != PI_DEC_PAPER_EVALUATION_ARITY
    {
        return Err(format!(
            "live strict PiDEC {owner} is not 54x270 with thirteen evaluations"
        ));
    }
    Ok(())
}

fn validate_recursive_selector(
    selector_writes: &[R1csIvcSelectorWriteAudit],
) -> Result<&R1csIvcSelectorWriteAudit, String> {
    if selector_writes.len() != 3 {
        return Err(format!(
            "active execution has {} branch selectors, expected three",
            selector_writes.len()
        ));
    }
    let mut recursive = None;
    let mut logical_columns = BTreeSet::new();
    for write in selector_writes {
        let expected = if write.arm() == R1csIvcBranch::Recursive {
            F::ONE
        } else {
            F::ZERO
        };
        if write.value() != expected || !logical_columns.insert(write.logical_column()) {
            return Err("active execution selector/profile association drift".into());
        }
        if write.arm() == R1csIvcBranch::Recursive {
            if recursive.replace(write).is_some() {
                return Err("active execution repeats the recursive selector".into());
            }
        }
    }
    recursive.ok_or_else(|| "active execution omits the recursive selector".into())
}

#[allow(clippy::too_many_arguments)]
fn capture_canonical_x_coordinates(
    builder: &R1csBuilder,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    receipt: &PiDecCanonicalXReceipt,
    children: &[CeClaim],
    x_recomposition_rows: &Range<usize>,
    x_canonicality_rows: &Range<usize>,
) -> Result<Vec<R1csIvcPiDecCanonicalXCoordinateAudit>, String> {
    let mut coordinates = Vec::with_capacity(PI_DEC_PAPER_PUBLIC_COORDINATES);
    let negative_one = F::ZERO - F::ONE;
    let program = receipt.program();
    for public_column in 0..PI_DEC_PAPER_PUBLIC_COORDINATES {
        let x_row = public_column % D;
        let x_active_column = public_column / D;
        let x_active_index = x_row * PI_DEC_PAPER_ACTIVE_X_COLUMNS + x_active_column;
        let sign_column = receipt
            .columns()
            .actual_column(
                program
                    .sign_canonical_column(x_active_index)
                    .ok_or_else(|| format!("PiDEC public column {public_column} has no sign column"))?,
            )
            .ok_or_else(|| format!("PiDEC public column {public_column} sign column is not mapped"))?;
        let sign_output_column = receipt
            .columns()
            .actual_column(
                program
                    .product_canonical_column(x_active_index)
                    .ok_or_else(|| format!("PiDEC public column {public_column} has no sign-product column"))?,
            )
            .ok_or_else(|| format!("PiDEC public column {public_column} sign-product column is not mapped"))?;
        let expected_sign = children
            .iter()
            .map(|child| child.X[(x_row, x_active_column)])
            .find(|value| *value != F::ZERO)
            .filter(|value| *value == F::ONE || *value == negative_one)
            .unwrap_or(F::ZERO);
        let expected_sign_output = (expected_sign + F::ONE) * expected_sign;
        let canonical_start =
            x_canonicality_rows.start + x_active_index * PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE;
        coordinates.push(R1csIvcPiDecCanonicalXCoordinateAudit {
            public_column,
            x_row,
            x_active_column,
            x_active_index,
            recomposition_source_row: x_recomposition_rows.start + x_active_index,
            canonicality_source_rows: canonical_start..canonical_start + PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE,
            sign: capture_trace_column(
                builder,
                public_outputs,
                normalized_assignment,
                sign_column,
                expected_sign,
                public_column,
                "sign",
            )?,
            sign_output: capture_trace_column(
                builder,
                public_outputs,
                normalized_assignment,
                sign_output_column,
                expected_sign_output,
                public_column,
                "sign output",
            )?,
        });
    }
    Ok(coordinates)
}

fn capture_trace_column(
    builder: &R1csBuilder,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    builder_column: usize,
    expected_value: F,
    public_column: usize,
    label: &str,
) -> Result<R1csIvcPiDecPaperTraceColumnAudit, String> {
    let normalized_column = normalized_target_column(builder.cols(), public_outputs, builder_column)
        .ok_or_else(|| format!("strict PiDEC public column {public_column} {label} does not normalize"))?;
    if normalized_source_column(builder.cols(), public_outputs, normalized_column) != Some(builder_column) {
        return Err(format!(
            "strict PiDEC public column {public_column} {label} fails normalization round trip"
        ));
    }
    let builder_value = builder
        .witness()
        .get(builder_column)
        .copied()
        .ok_or_else(|| format!("strict PiDEC public column {public_column} {label} escapes builder"))?;
    let normalized_value = normalized_assignment
        .get(normalized_column)
        .copied()
        .ok_or_else(|| format!("strict PiDEC public column {public_column} {label} escapes assignment"))?;
    if expected_value != builder_value || expected_value != normalized_value {
        return Err(format!(
            "strict PiDEC public column {public_column} {label} disagrees across expected, builder, and normalized assignment"
        ));
    }
    Ok(R1csIvcPiDecPaperTraceColumnAudit {
        builder_column,
        normalized_column,
        expected_value,
        builder_value,
        normalized_value,
    })
}

fn capture_claim_pins(
    builder: &R1csBuilder,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    builder_columns: &[usize],
    native: &CeClaim,
    owner: R1csIvcPiDecPaperXOwner,
) -> Result<Vec<R1csIvcPiDecPaperXPinAudit>, String> {
    if builder_columns.len() != PI_DEC_PAPER_PUBLIC_COORDINATES {
        return Err(format!(
            "strict PiDEC {owner:?} receipt has {} public-X columns, expected {PI_DEC_PAPER_PUBLIC_COORDINATES}",
            builder_columns.len()
        ));
    }
    let mut pins = Vec::with_capacity(PI_DEC_PAPER_PUBLIC_COORDINATES);
    for public_column in 0..PI_DEC_PAPER_PUBLIC_COORDINATES {
        let x_row = public_column % D;
        let x_active_column = public_column / D;
        let x_active_index = x_row * PI_DEC_PAPER_ACTIVE_X_COLUMNS + x_active_column;
        let builder_column = builder_columns
            .get(x_active_index)
            .copied()
            .ok_or_else(|| format!("strict PiDEC {owner:?} public column {public_column} has no X wire"))?;
        let normalized_column = normalized_target_column(builder.cols(), public_outputs, builder_column)
            .ok_or_else(|| format!("strict PiDEC {owner:?} builder column {builder_column} does not normalize"))?;
        if normalized_source_column(builder.cols(), public_outputs, normalized_column) != Some(builder_column) {
            return Err(format!(
                "strict PiDEC {owner:?} public column {public_column} fails normalization round trip"
            ));
        }
        let builder_value = builder
            .witness()
            .get(builder_column)
            .copied()
            .ok_or_else(|| format!("strict PiDEC {owner:?} X wire escapes the builder"))?;
        let normalized_value = normalized_assignment
            .get(normalized_column)
            .copied()
            .ok_or_else(|| format!("strict PiDEC {owner:?} X wire escapes the normalized assignment"))?;
        let native_value = native.X[(x_row, x_active_column)];
        if native_value != builder_value || native_value != normalized_value {
            return Err(format!(
                "strict PiDEC {owner:?} public column {public_column} disagrees across native, builder, and normalized assignment"
            ));
        }
        pins.push(R1csIvcPiDecPaperXPinAudit {
            owner,
            public_column,
            x_row,
            x_active_column,
            x_active_index,
            builder_column,
            normalized_column,
            native_value,
            builder_value,
            normalized_value,
        });
    }
    Ok(pins)
}
