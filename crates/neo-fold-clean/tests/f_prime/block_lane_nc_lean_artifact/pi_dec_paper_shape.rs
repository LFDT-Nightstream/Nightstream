//! Fail-closed checks for the compact live strict-PiDEC paper-shape seam.

use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    R1csIvcPiDecCanonicalXCoordinateAudit, R1csIvcPiDecPaperShapeExecutionAudit, R1csIvcPiDecPaperShapeProfile,
    R1csIvcPiDecPaperTraceColumnAudit, R1csIvcPiDecPaperXOwner, R1csIvcPiDecPaperXPinAudit,
    R1csIvcPostPiDecExecutionAudit, PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE, PI_DEC_PAPER_CHILD_COUNT,
    PI_DEC_PAPER_EVALUATION_ARITY, PI_DEC_PAPER_PUBLIC_COORDINATES,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Pin {
    owner: R1csIvcPiDecPaperXOwner,
    public_column: usize,
    x_coordinate: (usize, usize),
    x_active_index: usize,
    builder_column: usize,
    normalized_column: usize,
    native_value: F,
    builder_value: F,
    normalized_value: F,
}

impl From<&R1csIvcPiDecPaperXPinAudit> for Pin {
    fn from(pin: &R1csIvcPiDecPaperXPinAudit) -> Self {
        Self {
            owner: pin.owner(),
            public_column: pin.public_column(),
            x_coordinate: pin.x_coordinate(),
            x_active_index: pin.x_active_index(),
            builder_column: pin.builder_column(),
            normalized_column: pin.normalized_column(),
            native_value: pin.native_value(),
            builder_value: pin.builder_value(),
            normalized_value: pin.normalized_value(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Trace {
    builder_column: usize,
    normalized_column: usize,
    expected_value: F,
    builder_value: F,
    normalized_value: F,
}

impl From<R1csIvcPiDecPaperTraceColumnAudit> for Trace {
    fn from(trace: R1csIvcPiDecPaperTraceColumnAudit) -> Self {
        Self {
            builder_column: trace.builder_column(),
            normalized_column: trace.normalized_column(),
            expected_value: trace.expected_value(),
            builder_value: trace.builder_value(),
            normalized_value: trace.normalized_value(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Coordinate {
    public_column: usize,
    x_coordinate: (usize, usize),
    x_active_index: usize,
    recomposition_source_row: usize,
    canonicality_source_rows: Range<usize>,
    sign: Trace,
    sign_output: Trace,
}

impl From<&R1csIvcPiDecCanonicalXCoordinateAudit> for Coordinate {
    fn from(coordinate: &R1csIvcPiDecCanonicalXCoordinateAudit) -> Self {
        Self {
            public_column: coordinate.public_column(),
            x_coordinate: coordinate.x_coordinate(),
            x_active_index: coordinate.x_active_index(),
            recomposition_source_row: coordinate.recomposition_source_row(),
            canonicality_source_rows: coordinate.canonicality_source_rows(),
            sign: coordinate.sign().into(),
            sign_output: coordinate.sign_output().into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Snapshot {
    profile_tag: usize,
    strict_rows: Range<usize>,
    x_recomposition_rows: Range<usize>,
    x_canonicality_rows: Range<usize>,
    recursive_selector_logical_column: usize,
    recursive_selector_value: F,
    parent: Vec<Pin>,
    children: Vec<Vec<Pin>>,
    coordinates: Vec<Coordinate>,
    child_evaluation_arities: Vec<usize>,
}

impl Snapshot {
    fn capture(audit: &R1csIvcPiDecPaperShapeExecutionAudit) -> Self {
        Self {
            profile_tag: match audit.profile() {
                R1csIvcPiDecPaperShapeProfile::ActiveFPrimeRadix2 => 0,
            },
            strict_rows: audit.strict_rows(),
            x_recomposition_rows: audit.x_recomposition_rows(),
            x_canonicality_rows: audit.x_canonicality_rows(),
            recursive_selector_logical_column: audit.recursive_selector_logical_column(),
            recursive_selector_value: audit.recursive_selector_value(),
            parent: audit.parent_pins().iter().map(Pin::from).collect(),
            children: audit
                .child_pins()
                .iter()
                .map(|pins| pins.iter().map(Pin::from).collect())
                .collect(),
            coordinates: audit
                .canonical_x_coordinates()
                .iter()
                .map(Coordinate::from)
                .collect(),
            child_evaluation_arities: audit.child_evaluation_arities().to_vec(),
        }
    }

    fn validate_against(&self, live: &R1csIvcPiDecPaperShapeExecutionAudit) -> Result<(), String> {
        live.validate()?;
        if self != &Self::capture(live) {
            return Err("PiDEC paper-shape snapshot differs from the live execution join".into());
        }
        if self.profile_tag != 0
            || self.parent.len() != PI_DEC_PAPER_PUBLIC_COORDINATES
            || self.children.len() != PI_DEC_PAPER_CHILD_COUNT
            || self
                .children
                .iter()
                .any(|pins| pins.len() != PI_DEC_PAPER_PUBLIC_COORDINATES)
            || self.coordinates.len() != PI_DEC_PAPER_PUBLIC_COORDINATES
            || self.child_evaluation_arities != vec![PI_DEC_PAPER_EVALUATION_ARITY; PI_DEC_PAPER_CHILD_COUNT]
            || self.x_recomposition_rows.len() != PI_DEC_PAPER_PUBLIC_COORDINATES
            || self.x_canonicality_rows.len()
                != PI_DEC_PAPER_PUBLIC_COORDINATES * PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE
        {
            return Err("PiDEC paper-shape snapshot cardinality drift".into());
        }
        Ok(())
    }
}

pub(super) fn assert_live_contract(post: &R1csIvcPostPiDecExecutionAudit) {
    let live = post.pi_dec_paper_shape();
    let certificate = Snapshot::capture(live);
    certificate
        .validate_against(live)
        .expect("live strict-PiDEC paper-shape execution contract");
    let reject = |mutated: &Snapshot, label: &str| {
        assert!(mutated.validate_against(live).is_err(), "{label} must fail closed");
    };

    let mut changed = certificate.clone();
    changed.parent[0].normalized_column += 1;
    reject(&changed, "parent normalized-column mutation");

    let mut changed = certificate.clone();
    changed.children.swap(0, 1);
    reject(&changed, "ordered child mutation");

    let mut changed = certificate.clone();
    changed.children[0][0].native_value += F::ONE;
    reject(&changed, "authoritative child-X value mutation");

    let mut changed = certificate.clone();
    changed.child_evaluation_arities[0] += 1;
    reject(&changed, "child evaluation-arity mutation");

    let mut changed = certificate.clone();
    changed.recursive_selector_value = F::ZERO;
    reject(&changed, "recursive selector mutation");

    let mut changed = certificate.clone();
    changed.profile_tag += 1;
    reject(&changed, "profile mutation");

    let mut changed = certificate.clone();
    changed.coordinates[0].recomposition_source_row += 1;
    reject(&changed, "X recomposition-row mutation");

    let mut changed = certificate.clone();
    changed.coordinates[0].canonicality_source_rows.start += 1;
    reject(&changed, "X canonicality-row mutation");

    let mut changed = certificate;
    changed.coordinates[0].sign.normalized_column += 1;
    reject(&changed, "sign-trace normalized-column mutation");
}
