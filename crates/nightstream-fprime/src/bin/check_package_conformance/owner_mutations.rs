//! Owner-led matrix mutations for the exact Stage 1 conformance gate.

use nightstream_fprime::PackageSparseMatrix;

use super::{
    actual_row, canonicalize, changed_word, exact_row_accepts, ColumnOwnerSpan, ReferenceLayout, COLUMN_OWNER_SPANS,
    ROW_OWNER_SPANS,
};

fn pilot_source_to_spartan(column: usize) -> usize {
    if column < 45_937 {
        column
    } else if column < 46_207 {
        13_692_351 + (column - 45_937)
    } else if column < 92_144 {
        45_937 + (column - 46_207)
    } else if column < 92_148 {
        13_692_621 + (column - 92_144)
    } else {
        91_874 + (column - 92_148)
    }
}

fn lift_pilot_column(column: usize) -> usize {
    if column < 91_874 {
        column
    } else if column < 13_692_350 {
        column + 29_072
    } else {
        27_695_694 + (column - 13_692_350)
    }
}

fn source_to_spartan(column: usize) -> usize {
    assert!(column < 27_695_972, "Stage 1 source column");
    if column < 13_692_624 {
        lift_pilot_column(pilot_source_to_spartan(column))
    } else if column < 13_692_628 {
        27_695_969 + (column - 13_692_624)
    } else if column < 13_721_700 {
        91_874 + (column - 13_692_628)
    } else {
        13_721_422 + (column - 13_721_700)
    }
}

fn pilot_spartan_to_source(column: usize) -> Option<usize> {
    if column < 45_937 {
        Some(column)
    } else if column < 91_874 {
        Some(46_207 + (column - 45_937))
    } else if column < 13_692_350 {
        Some(92_148 + (column - 91_874))
    } else if column == 13_692_350 {
        None
    } else if column < 13_692_621 {
        Some(45_937 + (column - 13_692_351))
    } else if column < 13_692_625 {
        Some(92_144 + (column - 13_692_621))
    } else {
        None
    }
}

fn spartan_to_source(column: usize) -> Option<usize> {
    if column < 91_874 {
        pilot_spartan_to_source(column)
    } else if column < 120_946 {
        Some(13_692_628 + (column - 91_874))
    } else if column < 13_721_422 {
        pilot_spartan_to_source(column - 29_072)
    } else if column < 27_695_694 {
        Some(13_721_700 + (column - 13_721_422))
    } else if column == 27_695_694 {
        None
    } else if column < 27_695_969 {
        pilot_spartan_to_source(13_692_350 + (column - 27_695_694))
    } else if column < 27_695_973 {
        Some(13_692_624 + (column - 27_695_969))
    } else {
        None
    }
}

fn final_to_spartan(column: usize, layout: &ReferenceLayout) -> Option<usize> {
    if column < layout.unpadded_constant {
        Some(column)
    } else if column < layout.domain_size {
        None
    } else if column < layout.final_columns {
        Some(layout.unpadded_constant + (column - layout.domain_size))
    } else {
        None
    }
}

pub(super) fn row_owner_mutation_checks(sides: &[(&str, &PackageSparseMatrix)], rows: usize) -> usize {
    assert_eq!(ROW_OWNER_SPANS.first().map(|span| span.start), Some(0));
    assert_eq!(ROW_OWNER_SPANS.last().map(|span| span.end), Some(rows));
    for adjacent in ROW_OWNER_SPANS.windows(2) {
        assert_eq!(adjacent[0].end, adjacent[1].start, "row-owner coverage");
    }

    let mut checks = 0;
    for owner in ROW_OWNER_SPANS {
        let mut applicable_sides = 0;
        for &(side, matrix) in sides {
            let selected =
                (owner.start..owner.end).find(|&row| matrix.row_offsets()[row] < matrix.row_offsets()[row + 1]);
            let Some(row) = selected else { continue };
            applicable_sides += 1;
            let actual = actual_row(matrix, row);

            let mut deleted = actual.clone();
            deleted.remove(0);
            assert!(
                !exact_row_accepts(matrix, row, &deleted),
                "exact comparator accepted {} {side} row deletion",
                owner.name
            );
            checks += 1;

            let mut coefficient = actual.clone();
            coefficient[0].1 = changed_word(coefficient[0].1);
            coefficient = canonicalize(coefficient);
            assert!(
                !exact_row_accepts(matrix, row, &coefficient),
                "exact comparator accepted {} {side} coefficient mutation",
                owner.name
            );
            checks += 1;
        }
        assert!(applicable_sides > 0, "{} has no applicable matrix side", owner.name);
    }
    checks
}

fn find_owned_term(
    matrix: &PackageSparseMatrix,
    owner: ColumnOwnerSpan,
    layout: &ReferenceLayout,
) -> Option<(usize, usize, usize)> {
    for row in owner.rows.start..owner.rows.end {
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        for term in start..end {
            let final_column = matrix.column_indices()[term];
            let Some(spartan_column) = final_to_spartan(final_column, layout) else {
                continue;
            };
            let Some(source_column) = spartan_to_source(spartan_column) else {
                continue;
            };
            if owner.columns.start <= source_column && source_column < owner.columns.end {
                assert_eq!(
                    source_to_spartan(source_column),
                    spartan_column,
                    "{} round trip",
                    owner.name
                );
                return Some((row, term - start, source_column));
            }
        }
    }
    None
}

pub(super) fn column_owner_mutation_checks(sides: &[(&str, &PackageSparseMatrix)], layout: &ReferenceLayout) -> usize {
    let mut checks = 0;
    for &owner in COLUMN_OWNER_SPANS {
        assert!(owner.rows.start < owner.rows.end, "{} row interval", owner.name);
        assert!(
            owner.columns.start < owner.columns.end,
            "{} column interval",
            owner.name
        );
        let mut applicable_sides = 0;
        for &(side, matrix) in sides {
            let Some((row, term, source_column)) = find_owned_term(matrix, owner, layout) else {
                continue;
            };
            applicable_sides += 1;
            let target_source = if source_column + 1 < owner.columns.end {
                source_column + 1
            } else {
                source_column - 1
            };
            let target_column = layout.map_column(source_to_spartan(target_source));
            let actual = actual_row(matrix, row);
            let mut changed = actual.clone();
            changed[term].0 = target_column;
            changed = canonicalize(changed);
            assert!(
                !exact_row_accepts(matrix, row, &changed),
                "exact comparator accepted {} {side} column mutation",
                owner.name
            );
            checks += 1;
        }
        assert!(applicable_sides > 0, "{} has no applicable matrix side", owner.name);
    }

    for &(side, matrix) in sides {
        let selected = (0..layout.unpadded_rows).find_map(|row| {
            let actual = actual_row(matrix, row);
            actual
                .iter()
                .position(|term| term.0 == layout.constant_column())
                .map(|term| (row, term, actual))
        });
        let Some((row, term, actual)) = selected else { continue };
        let mut changed = actual.clone();
        changed[term].0 = 0;
        changed = canonicalize(changed);
        assert!(
            !exact_row_accepts(matrix, row, &changed),
            "exact comparator accepted constant {side} column mutation at row {row}"
        );
        checks += 1;
    }
    checks
}
