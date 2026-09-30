use super::*;
use crate::superneo_eval::MatrixShape;

/// Unit, dense and geometric runs, an empty matrix, and a matrix that is
/// empty on odd rows.
struct MixedRows {
    rows: usize,
}

impl MatrixRows for MixedRows {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            columns: 3 * D,
            matrices: 3,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        for row in rows {
            sink.push_run(row, 0, GeometricRowRun::new(row, row % (3 * D), 1, F::ONE, F::ONE))?;
            if row % 3 != 0 {
                sink.push_run(
                    row,
                    0,
                    GeometricRowRun::new(row, (row * 7) % (3 * D), 1, F::from_usize(row + 2), F::ONE),
                )?;
            }
            sink.push_run(
                row,
                0,
                GeometricRowRun::new(row, D - 2, 5, F::from_usize(row + 1), F::from_u64(3)),
            )?;
            if sink.finish_matrix_row(row, 0)?.is_break() || sink.finish_matrix_row(row, 1)?.is_break() {
                return Ok(());
            }
            if row % 2 == 0 {
                sink.push_run(
                    row,
                    2,
                    GeometricRowRun::new(row, D + row % D, 2, -F::ONE, F::from_u64(5)),
                )?;
            }
            if sink.finish_matrix_row(row, 2)?.is_break() {
                return Ok(());
            }
        }
        Ok(())
    }
}

#[test]
fn chunked_window_equals_the_serial_prefix_window() {
    let source = MixedRows { rows: 17 };
    for (requested, chunk_rows) in [(0..17, 1), (0..17, 3), (0..17, 16), (0..17, 40), (5..16, 4)] {
        let chunked = load_complete(&source, requested.clone(), usize::MAX, 8, chunk_rows)
            .unwrap()
            .expect("the complete range fits");
        let serial = MatrixWindow::load_prefix(&source, requested.clone(), usize::MAX, 8).unwrap();
        assert_eq!(chunked.rows(), requested);
        assert_eq!(serial.rows(), requested);
        assert_eq!(chunked.storage_bytes(), serial.storage_bytes());
        assert_eq!(
            format!("{:?}", chunked.cache()),
            format!("{:?}", serial.cache()),
            "chunk rows {chunk_rows}"
        );
    }
}

#[test]
fn chunked_window_declines_a_range_that_does_not_fit() {
    let source = MixedRows { rows: 17 };
    let required = MatrixWindow::required_workspace(&source, 0..17, 0).unwrap();
    assert!(load_complete(&source, 0..17, required / 2, 0, 4)
        .unwrap()
        .is_none());
}
