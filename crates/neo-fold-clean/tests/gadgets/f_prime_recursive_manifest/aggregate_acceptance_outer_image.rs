//! Exact sparse outer-image audit for recursive aggregate acceptance.
//!
//! Owns: the fixed recursive branch invocation of the production symbolic
//! lowering planner and its derived census. It does not materialize the full
//! encoded witness or CCS matrices.
//!
//! Does not own: the nine-row leaf equations, Lean semantics, artifact
//! serialization, or permission to remove any row.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the source program and trace are production evidence;
//! independent Lean semantics remain authoritative for obligation retention.
//!
//! | Audit branch | Exact evidence | Semantic owner |
//! |---|---|---|
//! | chunk census | every recursive sampler chunk has one 16-bit/9-row image | aggregate outer-image refinement |
//! | decoder census | singleton and transitive sparse images are classified by width | linear-substitution refinement |
//! | owner census | every input bit names one physical Boolean row | coordinate/fallback refinement |
//! | row census | source and physical sparse rows are deduplicated by exact index | complete placement refinement |

use std::collections::BTreeMap;

use neo_fold_clean::frontends::f_prime::gadget_native::{
    audit_r1cs_gadget_native_aggregate_acceptance_outer_image, AggregateAcceptanceBooleanRowOwner,
    AggregateAcceptanceDecodedImage, AggregateAcceptanceOuterImageAudit,
};

use super::{build_recursive_program, repo_root};

#[path = "aggregate_acceptance_outer_image/artifact.rs"]
mod artifact;

const CHUNKS: usize = 960;
const INPUTS_PER_CHUNK: usize = 16;
const ACTIVE_ROWS_PER_CHUNK: usize = 9;

fn build_outer_image() -> AggregateAcceptanceOuterImageAudit {
    let builder = build_recursive_program();
    let source = builder.snapshot();
    audit_r1cs_gadget_native_aggregate_acceptance_outer_image(&source, builder.encoding_trace(), &[])
        .expect("fixed recursive aggregate-acceptance outer image")
}

#[test]
fn recursive_aggregate_acceptance_sparse_outer_image_is_complete() {
    let audit = build_outer_image();

    assert_eq!(audit.chunks.len(), CHUNKS);
    assert_eq!(audit.matrix_arity, 56);
    assert_eq!(
        audit
            .chunks
            .iter()
            .map(|chunk| chunk.bits.len())
            .sum::<usize>(),
        CHUNKS * INPUTS_PER_CHUNK,
    );
    assert!(audit
        .chunks
        .iter()
        .all(|chunk| chunk.active_rows.len() == ACTIVE_ROWS_PER_CHUNK));

    let mut decoder_widths = BTreeMap::<usize, usize>::new();
    let mut singleton = 0usize;
    let mut linear = 0usize;
    let mut coordinate_pair_left = 0usize;
    let mut coordinate_pair_right = 0usize;
    let mut coordinate_tail = 0usize;
    let mut translated_source = 0usize;
    for bit in audit.chunks.iter().flat_map(|chunk| &chunk.bits) {
        match &bit.decoded {
            AggregateAcceptanceDecodedImage::Singleton { .. } => singleton += 1,
            AggregateAcceptanceDecodedImage::SparseLinear { terms } => {
                linear += 1;
                *decoder_widths.entry(terms.len()).or_default() += 1;
            }
        }
        match bit.boolean_owner {
            AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { .. } => coordinate_pair_left += 1,
            AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { .. } => coordinate_pair_right += 1,
            AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. } => coordinate_tail += 1,
            AggregateAcceptanceBooleanRowOwner::TranslatedSource { .. } => translated_source += 1,
        }
    }
    assert_eq!(singleton + linear, CHUNKS * INPUTS_PER_CHUNK);
    assert_eq!(
        coordinate_pair_left + coordinate_pair_right + coordinate_tail + translated_source,
        CHUNKS * INPUTS_PER_CHUNK,
    );
    assert!(audit
        .source_rows
        .windows(2)
        .all(|rows| rows[0].row < rows[1].row));
    assert!(audit
        .physical_rows
        .windows(2)
        .all(|rows| rows[0].row < rows[1].row));

    eprintln!(
        "recursive aggregate acceptance: source={}x{}, encoded={}x{}, definitions={}, source_rows={}, physical_rows={}, singleton={}, linear={}, decoder_widths={decoder_widths:?}, pair_left={}, pair_right={}, tail={}, translated={}",
        audit.source_row_count,
        audit.source_columns,
        audit.encoded_rows,
        audit.encoded_columns,
        audit.linear_definitions.len(),
        audit.source_rows.len(),
        audit.physical_rows.len(),
        singleton,
        linear,
        coordinate_pair_left,
        coordinate_pair_right,
        coordinate_tail,
        translated_source,
    );
}

#[test]
fn recursive_aggregate_acceptance_lean_outer_image_matches_sparse_production() {
    let audit = build_outer_image();
    let mut drifted = Vec::new();
    for file in artifact::render(&audit) {
        let path = repo_root().join(&file.relative_path);
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        if committed != file.contents {
            let expected = path.with_extension("lean.expected");
            std::fs::create_dir_all(expected.parent().expect("outer-image artifact parent"))
                .expect("create outer-image artifact directory");
            std::fs::write(&expected, &file.contents).expect("write outer-image artifact candidate");
            drifted.push(file.relative_path);
        }
    }
    assert!(
        drifted.is_empty(),
        "recursive aggregate-acceptance Lean artifact drifted; inspect and deliberately promote: {drifted:?}",
    );
}
