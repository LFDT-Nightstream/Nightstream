//! Exact structural row conformance for the production linked overlay.

use std::collections::BTreeMap;
use std::ops::Range;

use neo_ccs::{CcsMatrix, GeometricRowRun};
use neo_fold_clean::frontends::r1cs_f_prime::{LinkedOverlayLowNormR1cs, OverlayKindLinks};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const SELECTIVE_ARITY: usize = 13;
const GENERAL_SELECTOR: usize = 1;
const A: usize = 2;
const B: usize = 3;
const C: usize = 4;

#[derive(Clone, Copy)]
enum ComponentEmbedding {
    Base,
    Overlay { private_start: usize },
}

impl ComponentEmbedding {
    fn map(self, column: usize) -> usize {
        match self {
            Self::Base => column,
            Self::Overlay { private_start } => {
                if column == 0 {
                    0
                } else {
                    private_start + column - 1
                }
            }
        }
    }
}

fn assert_component_embedding_exact(
    owner: &str,
    source_rows: usize,
    source_columns: usize,
    source_matrices: &[CcsMatrix<F>],
    final_matrices: &[CcsMatrix<F>],
    final_rows: Range<usize>,
    embedding: ComponentEmbedding,
) {
    assert_eq!(source_rows, final_rows.len(), "{owner} row extent");
    assert_eq!(source_matrices.len(), SELECTIVE_ARITY, "{owner} source arity");
    assert_eq!(final_matrices.len(), SELECTIVE_ARITY, "{owner} final arity");

    for (port, (source, final_matrix)) in source_matrices.iter().zip(final_matrices).enumerate() {
        let source_csc = source
            .sparse_component()
            .expect("production component has an ordinary sparse part");
        let final_csc = final_matrix
            .sparse_component()
            .expect("production linked relation has an ordinary sparse part");
        assert_eq!(source_csc.nrows, source_rows, "{owner} source rows at port {port}");
        assert_eq!(
            source_csc.ncols, source_columns,
            "{owner} source columns at port {port}"
        );

        for source_column in 0..source_columns {
            let final_column = embedding.map(source_column);
            let mut actual = final_csc
                .column_range(final_column)
                .filter(|&entry| final_rows.contains(&final_csc.row_index(entry)));
            for source_entry in source_csc.column_range(source_column) {
                let final_entry = actual.next().unwrap_or_else(|| {
                    panic!(
                        "{owner} port {port}, source column {source_column} lost row {}",
                        source_csc.row_index(source_entry)
                    )
                });
                assert_eq!(
                    final_csc.row_index(final_entry),
                    final_rows.start + source_csc.row_index(source_entry),
                    "{owner} embedded row at port {port}, source column {source_column}",
                );
                assert_eq!(
                    final_csc.vals[final_entry], source_csc.vals[source_entry],
                    "{owner} embedded coefficient at port {port}, source column {source_column}",
                );
            }
            assert!(
                actual.next().is_none(),
                "{owner} port {port}, source column {source_column} gained a sparse row"
            );
        }

        let actual_sparse_count = final_csc
            .row_idx
            .iter()
            .filter(|&&row| final_rows.contains(&(row as usize)))
            .count();
        assert_eq!(
            actual_sparse_count,
            source_csc.vals.len(),
            "{owner} sparse row ownership at port {port}",
        );

        let expected_blocks = source
            .seeded_phi81_blocks()
            .iter()
            .map(|block| {
                block
                    .with_geometry(
                        final_rows.start + block.row_start(),
                        block
                            .word_starts()
                            .iter()
                            .map(|&column| embedding.map(column))
                            .collect(),
                    )
                    .expect("embedded production compact block geometry")
            })
            .collect::<Vec<_>>();
        let actual_blocks = final_matrix
            .seeded_phi81_blocks()
            .iter()
            .filter(|block| final_rows.contains(&block.row_start()))
            .cloned()
            .collect::<Vec<_>>();
        for block in final_matrix.seeded_phi81_blocks() {
            assert!(
                block.row_end() <= final_rows.start
                    || final_rows.end <= block.row_start()
                    || (final_rows.contains(&block.row_start()) && block.row_end() <= final_rows.end),
                "{owner} compact block crosses the component row boundary at port {port}",
            );
        }
        assert_eq!(actual_blocks, expected_blocks, "{owner} compact blocks at port {port}");

        let expected_geometric = source
            .geometric_runs()
            .iter()
            .map(|run| {
                GeometricRowRun::new(
                    final_rows.start + run.row(),
                    embedding.map(run.column_start()),
                    run.len(),
                    *run.initial(),
                    *run.ratio(),
                )
            })
            .collect::<Vec<_>>();
        let actual_geometric = final_matrix
            .geometric_runs()
            .iter()
            .filter(|run| final_rows.contains(&run.row()))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            actual_geometric, expected_geometric,
            "{owner} geometric rows at port {port}",
        );
    }
}

fn normalized_terms(terms: Vec<(usize, F)>) -> BTreeMap<usize, F> {
    let mut normalized = BTreeMap::new();
    for (column, coefficient) in terms {
        *normalized.entry(column).or_insert(F::ZERO) += coefficient;
    }
    normalized.retain(|_, coefficient| *coefficient != F::ZERO);
    normalized
}

fn empty_row() -> [Vec<(usize, F)>; SELECTIVE_ARITY] {
    std::array::from_fn(|_| Vec::new())
}

fn assert_sparse_entry(matrix: &CcsMatrix<F>, row: usize, column: usize, expected: F, label: &str) {
    let csc = matrix
        .sparse_component()
        .expect("production linked relation has an ordinary sparse part");
    let entries = csc.column_range(column);
    let rows = &csc.row_idx[entries.clone()];
    let offset = rows
        .binary_search(&(row as u32))
        .unwrap_or_else(|_| panic!("{label}: missing row {row}, column {column}"));
    assert_eq!(csc.vals[entries.start + offset], expected, "{label}: coefficient");
}

fn assert_row_exact(
    matrices: &[CcsMatrix<F>],
    row: usize,
    expected: [Vec<(usize, F)>; SELECTIVE_ARITY],
    expected_counts: &mut [usize; SELECTIVE_ARITY],
    label: &str,
) {
    for (port, terms) in expected.into_iter().enumerate() {
        let terms = normalized_terms(terms);
        expected_counts[port] += terms.len();
        for (column, coefficient) in terms {
            assert_sparse_entry(&matrices[port], row, column, coefficient, label);
        }
    }
}

fn assert_composer_rows_exact(relation: &LinkedOverlayLowNormR1cs, links: &[OverlayKindLinks]) {
    let layout = relation.layout();
    let matrices = &relation.structure().matrices;
    let base_selectors = layout.base_selector_columns();
    let overlay_selectors = layout.overlay_selector_columns();
    let base_of = layout.overlay_base_kinds();
    let mut expected_counts = [0usize; SELECTIVE_ARITY];

    for base_kind in 0..base_selectors.len() {
        let row = layout.base_kind_equality_rows().start + base_kind;
        let mut expected = empty_row();
        expected[GENERAL_SELECTOR].push((0, F::ONE));
        expected[C].push((base_selectors[base_kind], F::ONE));
        for (overlay_kind, &owner) in base_of.iter().enumerate() {
            if owner == base_kind {
                expected[C].push((overlay_selectors[overlay_kind], -F::ONE));
            }
        }
        assert_row_exact(matrices, row, expected, &mut expected_counts, "base selector equality");
    }

    for overlay_kind in 0..overlay_selectors.len() {
        let row = layout.overlay_activation_rows().start + overlay_kind;
        let mut expected = empty_row();
        expected[GENERAL_SELECTOR].push((0, F::ONE));
        expected[A].push((overlay_selectors[overlay_kind], F::ONE));
        expected[B].push((base_selectors[base_of[overlay_kind]], F::ONE));
        expected[C].push((overlay_selectors[overlay_kind], F::ONE));
        assert_row_exact(matrices, row, expected, &mut expected_counts, "overlay activation");
    }

    assert_eq!(links.len(), overlay_selectors.len(), "complete overlay link contracts");
    for overlay_kind in 0..overlay_selectors.len() {
        let contract = &links[overlay_kind];
        assert_eq!(contract.overlay_kind, overlay_kind, "overlay link order");
        let base_kind = base_of[overlay_kind];
        let link_rows = layout
            .field_link_rows_for_kind(overlay_kind)
            .expect("field-link row ownership");
        assert_eq!(link_rows.len(), contract.fields.len(), "field-link row count");
        for (offset, link) in contract.fields.iter().enumerate() {
            let row = link_rows.start + offset;
            let mut expected = empty_row();
            expected[GENERAL_SELECTOR].push((0, F::ONE));
            expected[A].push((overlay_selectors[overlay_kind], F::ONE));
            expected[B].extend(
                relation
                    .base_field_decoding_terms(base_kind, link.phase_field)
                    .expect("base field decoder"),
            );
            expected[B].extend(
                relation
                    .overlay_field_decoding_terms(overlay_kind, link.overlay_field)
                    .expect("overlay field decoder")
                    .into_iter()
                    .map(|(column, coefficient)| (column, -coefficient)),
            );
            assert_row_exact(matrices, row, expected, &mut expected_counts, "field link");
        }

        let pin_rows = layout
            .base_field_pin_rows_for_kind(overlay_kind)
            .expect("base-pin row ownership");
        assert_eq!(pin_rows.len(), contract.base_pins.len(), "base-pin row count");
        for (offset, pin) in contract.base_pins.iter().enumerate() {
            let row = pin_rows.start + offset;
            let mut expected = empty_row();
            expected[GENERAL_SELECTOR].push((0, F::ONE));
            expected[A].push((overlay_selectors[overlay_kind], F::ONE));
            expected[B].extend(
                relation
                    .base_field_decoding_terms(base_kind, pin.phase_field)
                    .expect("base pin decoder"),
            );
            expected[B].push((0, -pin.value));
            assert_row_exact(matrices, row, expected, &mut expected_counts, "base field pin");
        }
    }

    for (row, column) in layout
        .ring_padding_rows()
        .zip(layout.ring_padding_columns())
    {
        let mut expected = empty_row();
        expected[GENERAL_SELECTOR].push((0, F::ONE));
        expected[C].push((column, F::ONE));
        assert_row_exact(matrices, row, expected, &mut expected_counts, "ring padding");
    }

    let composer_rows = layout.base_kind_equality_rows().start..layout.ring_padding_rows().end;
    for (port, matrix) in matrices.iter().enumerate() {
        let csc = matrix
            .sparse_component()
            .expect("production linked relation has an ordinary sparse part");
        let actual = csc
            .row_idx
            .iter()
            .filter(|&&row| composer_rows.contains(&(row as usize)))
            .count();
        assert_eq!(
            actual, expected_counts[port],
            "composer sparse ownership at port {port}"
        );
        assert!(
            matrix
                .seeded_phi81_blocks()
                .iter()
                .all(|block| block.row_end() <= composer_rows.start || composer_rows.end <= block.row_start()),
            "composer rows do not contain seeded blocks at port {port}",
        );
        assert!(
            matrix
                .geometric_runs()
                .iter()
                .all(|run| !composer_rows.contains(&run.row())),
            "composer rows do not contain geometric runs at port {port}",
        );
    }
}

pub(super) fn assert_exact_final_row_embedding(relation: &LinkedOverlayLowNormR1cs, links: &[OverlayKindLinks]) {
    let final_structure = relation.structure();
    let layout = relation.layout();
    let base = relation.base_relation().structure();
    let overlay = relation.overlay_relation().structure();

    assert_component_embedding_exact(
        "base",
        base.n,
        base.m,
        &base.matrices,
        &final_structure.matrices,
        layout.base_rows(),
        ComponentEmbedding::Base,
    );
    assert_eq!(
        layout.overlay_private_columns().len() + 1,
        overlay.m,
        "overlay private-column embedding",
    );
    assert_component_embedding_exact(
        "overlay",
        overlay.n,
        overlay.m,
        &overlay.matrices,
        &final_structure.matrices,
        layout.overlay_rows(),
        ComponentEmbedding::Overlay {
            private_start: layout.overlay_private_columns().start,
        },
    );
    assert_composer_rows_exact(relation, links);
}
