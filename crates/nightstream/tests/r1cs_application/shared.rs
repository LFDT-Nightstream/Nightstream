use super::*;
use nightstream::nebula::{Application, SharedColumns};

fn shared_application(rows: usize, columns: Vec<usize>) -> Result<R1csApplication<u32>, R1csApplicationError> {
    let count = NonZeroUsize::new(rows).unwrap();
    let mut source = RowRelation::of(&relation()).unwrap();
    source.width = 7;
    source
        .rows
        .push([vec![(4, F::ONE)], vec![(4, F::ONE)], vec![(6, F::ONE)]]);
    source
        .rows
        .push([vec![(VALUE, F::ONE)], vec![(4, F::ONE)], vec![(5, F::ONE)]]);
    let catalog = catalog();
    let (regions, _) = pack_regions(
        catalog
            .entries()
            .iter()
            .map(|memory| (memory.id, memory.kind, vec![2])),
    )
    .unwrap();
    R1csApplication::with_shared(
        source,
        &continuity(),
        &catalog,
        Size {
            geometry: Geometry::new(2, 2, 2 * rows, 1, 1).unwrap(),
            rows: count,
            regions,
            slots: batched_slots(&catalog, &[vec![0], vec![1]], count).unwrap(),
        },
        SharedColumns {
            columns,
            rows: NonZeroUsize::new(2).unwrap(),
        },
    )
}

#[test]
fn shared_columns_keep_local_columns_and_a_short_final_group_distinct() {
    let app = shared_application(3, vec![4, 6]).unwrap();
    let shared = SharedColumns {
        columns: vec![4, 6],
        rows: NonZeroUsize::new(2).unwrap(),
    };
    let rows = (0..3)
        .map(|row| {
            let value = F::from_u32(TABLE[row]);
            let x = F::from_usize(2 + row / 2);
            vec![
                F::ONE,
                F::from_usize(row),
                F::from_usize(row + 1),
                value,
                x,
                value * x,
                x * x,
            ]
        })
        .collect::<Vec<_>>();
    let z = shared
        .step_witness(7, ONE, &rows.iter().map(Vec::as_slice).collect::<Vec<_>>(), |_| 0)
        .unwrap();
    assert_eq!(z.len(), 19);
    assert_eq!(app.aux(), 19);
    assert_eq!(app.relation_rows(), 10);
    app.check(&z).unwrap();
    for row in 0..3 {
        for column in 0..7 {
            assert_eq!(z[app.witness_column(row, column).unwrap()], rows[row][column]);
        }
    }
    assert_eq!(app.witness_column(0, 4), app.witness_column(1, 4));
    assert_ne!(app.witness_column(1, 4), app.witness_column(2, 4));
    assert_ne!(app.witness_column(0, 5), app.witness_column(1, 5));
    assert_eq!(app.witness_column(3, 0), None);
    assert_eq!(app.witness_column(0, 7), None);
    for column in 0..z.len() {
        let mut forged = z.clone();
        forged[column] += F::ONE;
        if [5, 10].contains(&column) {
            continue;
        }
        assert!(app.check(&forged).is_err(), "changed coordinate {column}");
    }
}

#[test]
fn sharing_rejects_interfaces_duplicates_and_out_of_bounds_columns() {
    for columns in [vec![ONE], vec![BEFORE], vec![AFTER], vec![VALUE], vec![4, 4], vec![7]] {
        assert!(matches!(
            shared_application(3, columns),
            Err(R1csApplicationError::SharedColumn)
        ));
    }
    let app = application(2);
    for row in 0..2 {
        for column in 0..COLUMNS {
            assert_eq!(
                app.witness_column(row, column),
                Some(if column == ONE { ONE } else { row * COLUMNS + column })
            );
        }
    }
}
