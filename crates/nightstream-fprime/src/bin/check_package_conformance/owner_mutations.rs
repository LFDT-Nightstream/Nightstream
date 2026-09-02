//! Owner-led matrix mutations for the exact Stage 1 conformance gate.

use nightstream_fprime::PackageSparseMatrix;

use super::{
    actual_row, canonicalize, changed_word, exact_row_accepts, ColumnOwnerSpan, ReferenceLayout, COLUMN_OWNER_SPANS,
    FINAL_COLUMN_OWNER_SPANS, ROW_OWNER_SPANS,
};

fn pilot_source_to_spartan(column: usize) -> usize {
    if column < 49_393 {
        column
    } else if column < 49_663 {
        14_722_239 + (column - 49_393)
    } else if column < 99_056 {
        49_393 + (column - 49_663)
    } else if column < 99_060 {
        14_722_509 + (column - 99_056)
    } else {
        98_786 + (column - 99_060)
    }
}

fn lift_pilot_column(column: usize) -> usize {
    if column < 98_786 {
        column
    } else if column < 14_722_238 {
        column + 29_288
    } else {
        29_336_446 + (column - 14_722_238)
    }
}

fn source_to_spartan(column: usize, layout: &ReferenceLayout) -> usize {
    assert!(column < 29_336_724, "Stage 1 source column");
    let prefix_column = if column < 14_722_512 {
        lift_pilot_column(pilot_source_to_spartan(column))
    } else if column < 14_722_516 {
        29_336_721 + (column - 14_722_512)
    } else if column < 14_751_804 {
        98_786 + (column - 14_722_516)
    } else {
        14_751_526 + (column - 14_751_804)
    };
    if prefix_column < 29_336_446 {
        prefix_column
    } else {
        prefix_column + (layout.unpadded_constant - 29_336_446)
    }
}

fn pilot_spartan_to_source(column: usize) -> Option<usize> {
    if column < 49_393 {
        Some(column)
    } else if column < 98_786 {
        Some(49_663 + (column - 49_393))
    } else if column < 14_722_238 {
        Some(99_060 + (column - 98_786))
    } else if column == 14_722_238 {
        None
    } else if column < 14_722_509 {
        Some(49_393 + (column - 14_722_239))
    } else if column < 14_722_513 {
        Some(99_056 + (column - 14_722_509))
    } else {
        None
    }
}

fn spartan_to_source(column: usize, layout: &ReferenceLayout) -> Option<usize> {
    let column = if column < 29_336_446 {
        column
    } else if column < layout.unpadded_constant {
        return None;
    } else {
        29_336_446 + (column - layout.unpadded_constant)
    };
    if column < 98_786 {
        pilot_spartan_to_source(column)
    } else if column < 128_074 {
        Some(14_722_516 + (column - 98_786))
    } else if column < 14_751_526 {
        pilot_spartan_to_source(column - 29_288)
    } else if column < 29_336_446 {
        Some(14_751_804 + (column - 14_751_526))
    } else if column == 29_336_446 {
        None
    } else if column < 29_336_721 {
        pilot_spartan_to_source(14_722_238 + (column - 29_336_446))
    } else if column < 29_336_725 {
        Some(14_722_512 + (column - 29_336_721))
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
            let Some(source_column) = spartan_to_source(spartan_column, layout) else {
                continue;
            };
            if owner.columns.start <= source_column && source_column < owner.columns.end {
                assert_eq!(
                    source_to_spartan(source_column, layout),
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
            let target_column = layout.map_column(source_to_spartan(target_source, layout));
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

    for &owner in FINAL_COLUMN_OWNER_SPANS {
        assert!(owner.rows.start < owner.rows.end, "{} row interval", owner.name);
        assert!(
            owner.columns.start < owner.columns.end,
            "{} column interval",
            owner.name
        );
        let mut applicable_sides = 0;
        for &(side, matrix) in sides {
            let selected = (owner.rows.start..owner.rows.end).find_map(|row| {
                actual_row(matrix, row)
                    .iter()
                    .position(|term| owner.columns.start <= term.0 && term.0 < owner.columns.end)
                    .map(|term| (row, term))
            });
            let Some((row, term)) = selected else { continue };
            applicable_sides += 1;
            let actual = actual_row(matrix, row);
            let target = if actual[term].0 + 1 < owner.columns.end {
                actual[term].0 + 1
            } else {
                actual[term].0 - 1
            };
            let mut changed = actual.clone();
            changed[term].0 = target;
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
