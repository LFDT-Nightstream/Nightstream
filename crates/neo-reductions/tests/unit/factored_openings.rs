use std::sync::atomic::{AtomicUsize, Ordering};

use neo_ccs::Mat;
use neo_math::F;

use super::*;
use crate::superneo_eval::{
    eval_real_v1_2_openings_from_rows, eval_real_v1_2_openings_from_rows_reusing, FactoredEntries, FactoredSlots,
    MatrixRowSink, MatrixRows, MatrixShape,
};
use crate::PiCcsError;

struct Rows {
    factored: FactoredRows<'static>,
    visits: AtomicUsize,
}

impl MatrixRows for Rows {
    fn shape(&self) -> MatrixShape {
        self.factored.shape()
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.visits.fetch_add(1, Ordering::Relaxed);
        self.factored.visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.factored))
    }
}

#[test]
fn factored_openings_match_expanded_windows_and_skip_their_construction() {
    const COLUMNS: usize = BAND_COLUMNS + 2 * D;
    const STARTS: &[u32] = &[
        0,
        0,
        (D - 2) as u32,
        (BAND_COLUMNS - 3) as u32,
        (BAND_COLUMNS + 9) as u32,
    ];
    const OVERLAP: &[u32] = &[0, 0, (D - 2) as u32, (D - 1) as u32, (BAND_COLUMNS + 9) as u32];
    const WIDTHS: &[u8] = &[1, 0, 6, 7, 2];
    let entries = FactoredEntries {
        offsets: &[0, 2, 2, 3, 5, 6, 7, 10, 11],
        slots: &[0, 2, 3, 2, 4, 2, 3, 4, 4, 3, 3],
        coefficients: &[
            F::ONE,
            F::NEG_ONE,
            F::ONE,
            F::NEG_ONE,
            F::ONE,
            F::ONE,
            F::ONE,
            F::ONE,
            F::ONE,
            F::NEG_ONE,
            F::ONE,
        ],
    };
    let mut positive = vec![0; COLUMNS / D];
    let mut negative = vec![0; COLUMNS / D];
    for block in [0, 1, BAND_COLUMNS / D - 1, BAND_COLUMNS / D] {
        positive[block] = 0xf00000000000001f;
        negative[block] = 1 << 9;
    }
    let packed = Mat::compact_signed_unit_from_column_masks(D, COLUMNS / D, &positive, &negative).unwrap();
    let zero = Mat::virtual_constant(D, COLUMNS / D, F::ZERO);
    let dense = {
        let mut matrix = Mat::zero(D, COLUMNS / D, F::ZERO);
        for block in [0, 1, BAND_COLUMNS / D - 1, BAND_COLUMNS / D, BAND_COLUMNS / D + 1] {
            for lane in 0..D {
                matrix[(lane, block)] = F::from_usize(lane + 3);
            }
        }
        matrix
    };
    let blocks: Vec<_> = [&packed, &zero, &dense]
        .into_iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, COLUMNS).unwrap())
        .collect();
    let point: Vec<_> = (0..COLUMNS.next_power_of_two().ilog2())
        .map(|bit| K::from_coeffs([F::from_u32(bit + 4), F::from_u32(bit + 7)]))
        .collect();
    for (starts, expects_fast_path) in [(STARTS, true), (OVERLAP, false)] {
        let rows = Rows {
            factored: FactoredRows::new(
                MatrixShape {
                    rows: 4,
                    columns: COLUMNS,
                    matrices: 2,
                },
                entries,
                FactoredSlots {
                    starts,
                    widths: WIDTHS,
                    ratio: F::from_u64(3),
                },
            )
            .unwrap(),
            visits: AtomicUsize::new(0),
        };
        let expected = eval_real_v1_2_openings_from_rows(&rows, &point, &blocks, usize::MAX).unwrap();
        rows.visits.store(0, Ordering::Relaxed);
        let actual =
            eval_real_v1_2_openings_from_rows_reusing(&rows, &point, &blocks, usize::MAX, vec![K::ONE; COLUMNS], None)
                .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(rows.visits.load(Ordering::Relaxed) == 0, expects_fast_path);
    }
}
