use std::sync::atomic::{AtomicUsize, Ordering};

use neo_ccs::Mat;
use neo_math::F;

use super::*;
use crate::superneo_eval::{
    eval_real_v1_2_openings_from_rows, eval_real_v1_2_openings_from_rows_reusing, FactoredEntries, FactoredSlots,
    MatrixRowSink, MatrixRows, MatrixShape, OpeningColumns,
};
use crate::PiCcsError;

struct Rows<'a> {
    factored: FactoredRows<'a>,
    visits: AtomicUsize,
}

impl MatrixRows for Rows<'_> {
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
fn cached_opening_columns_match_fallback_for_distinct_points_and_many_sources() {
    let columns = 3 * D;
    let cache = OpeningColumns::default();
    let coefficients = [
        F::ONE,
        F::NEG_ONE,
        F::from_u8(91),
        F::from_u8(22),
        F::ONE,
        F::from_u8(12),
        F::NEG_ONE,
        F::from_u8(11),
        F::ONE,
        F::ZERO,
        F::NEG_ONE,
        F::from_u8(22),
        F::ONE,
    ];
    let rows = Rows {
        factored: FactoredRows::new(
            MatrixShape {
                rows: 3,
                columns,
                matrices: 2,
            },
            FactoredEntries {
                ports: neo_ccs::sparse::FactoredPorts::Offsets(&[0, 3, 5, 7, 8, 11, 13]),
                slots: &[0, 1, 1, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1],
                coefficients: &coefficients,
            },
            FactoredSlots {
                starts: &[0, 59, 130],
                widths: &[1, 46, 40],
                ratio: F::from_u8(3),
            },
        )
        .unwrap()
        .with_opening_columns(&cache),
        visits: AtomicUsize::new(0),
    };
    let witnesses = (0..17)
        .map(|i| {
            let witness = if i % 3 == 0 {
                Mat::compact_signed_unit_from_column_masks(D, 3, &[1 << 63, 0x1010, 1], &[1, 2, 4]).unwrap()
            } else if i % 3 == 1 {
                Mat::virtual_constant(D, 3, F::ZERO)
            } else {
                let mut matrix = Mat::zero(D, 3, F::ZERO);
                for block in 0..3 {
                    matrix[((3 * i + block) % D, block)] = if block == 0 {
                        -F::from_u64(2049)
                    } else {
                        F::from_usize(20 + i)
                    };
                }
                matrix
            };
            SuperneoZBlocks::from_witness_mat(&witness, columns).unwrap()
        })
        .collect::<Vec<_>>();
    for repetition in 0..3 {
        let point = (0..8)
            .map(|i| K::from_coeffs([F::from_usize(i + repetition), F::from_usize((i + 2) * repetition)]))
            .collect::<Vec<_>>();
        let baseline = eval_real_v1_2_openings_from_rows(&rows, &point, &witnesses, usize::MAX).unwrap();
        rows.visits.store(0, Ordering::Relaxed);
        let actual =
            eval_real_v1_2_openings_from_rows_reusing(&rows, &point, &witnesses, usize::MAX, Vec::new(), None, None)
                .unwrap();
        assert_eq!(actual, baseline);
        assert!(rows.factored.opening_columns(usize::MAX).is_some());
        assert_eq!(rows.visits.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn too_many_column_factors_falls_back_to_the_row_transpose() {
    let cache = OpeningColumns::default();
    let factors: Vec<_> = (2..=65_536).map(F::from_usize).collect();
    let quotient = vec![0; factors.len()];
    let starts = [0];
    let widths = [1];
    let offsets = [0, factors.len()];
    let rows = Rows {
        factored: FactoredRows::new(
            MatrixShape {
                rows: 1,
                columns: D,
                matrices: 1,
            },
            FactoredEntries {
                ports: neo_ccs::sparse::FactoredPorts::Offsets(&offsets),
                slots: &quotient,
                coefficients: &factors,
            },
            FactoredSlots {
                starts: &starts,
                widths: &widths,
                ratio: F::NEG_ONE,
            },
        )
        .unwrap()
        .with_opening_columns(&cache),
        visits: AtomicUsize::new(0),
    };
    let matrix = Mat::virtual_constant(D, 1, F::NEG_ONE);
    let witness = [SuperneoZBlocks::from_witness_mat(&matrix, D).unwrap()];
    let point = [K::ZERO; 6];
    let baseline = eval_real_v1_2_openings_from_rows(&rows, &point, &witness, usize::MAX).unwrap();
    let actual =
        eval_real_v1_2_openings_from_rows_reusing(&rows, &point, &witness, usize::MAX, Vec::new(), None, None).unwrap();
    assert!(rows.factored.opening_columns(usize::MAX).is_none());
    assert_eq!(actual, baseline);
}

#[test]
fn budget_fallback_does_not_materialize_or_disable_the_index() {
    let cache = OpeningColumns::default();
    let rows = Rows {
        factored: FactoredRows::new(
            MatrixShape {
                rows: 1,
                columns: D,
                matrices: 1,
            },
            FactoredEntries {
                ports: neo_ccs::sparse::FactoredPorts::Offsets(&[0, 1]),
                slots: &[0],
                coefficients: &[F::NEG_ONE],
            },
            FactoredSlots {
                starts: &[0],
                widths: &[41],
                ratio: F::from_u8(3),
            },
        )
        .unwrap()
        .with_opening_columns(&cache),
        visits: AtomicUsize::new(0),
    };
    let matrix = Mat::virtual_constant(D, 1, F::ONE);
    let witness = [SuperneoZBlocks::from_witness_mat(&matrix, D).unwrap()];
    let point = [K::ZERO; 6];
    let baseline = eval_real_v1_2_openings_from_rows(&rows, &point, &witness, usize::MAX).unwrap();
    let work = size_of::<K>()
        + size_of::<Range<usize>>()
        + size_of::<&SuperneoZBlocks>()
        + TaskSums::working_bytes(1, 1).unwrap();
    assert!(OpeningBands::new(rows.factored, work - 1, 1, 0).is_none());
    let bands = OpeningBands::new(rows.factored, work, 1, 0).unwrap();
    assert!(bands.index_budget.is_none() && !bands.parallel_transpose);
    rows.visits.store(0, Ordering::Relaxed);
    let actual =
        eval_real_v1_2_openings_from_rows_reusing(&rows, &point, &witness, work, Vec::new(), None, None).unwrap();
    assert_eq!(actual, baseline);
    assert_eq!(rows.visits.load(Ordering::Relaxed), 0);
    assert!(rows.factored.opening_columns(0).is_none());
    assert!(rows.factored.opening_columns(usize::MAX).is_some());
    assert!(rows.factored.opening_columns(0).is_none());
}

#[test]
fn four_matrix_bands_match_separate_openings_for_mixed_witnesses() {
    let cache = OpeningColumns::default();
    let coefficients = std::array::from_fn::<_, 36, _>(|i| F::from_usize(i + 2));
    let columns = 3 * D;
    let indexes = std::array::from_fn::<_, 36, _>(|i| (i % 9) as u32);
    let rows = FactoredRows::new(
        MatrixShape {
            rows: 1,
            columns,
            matrices: 4,
        },
        FactoredEntries {
            ports: neo_ccs::sparse::FactoredPorts::Offsets(&[0, 9, 18, 27, 36]),
            slots: &indexes,
            coefficients: &coefficients,
        },
        FactoredSlots {
            starts: &[0, 2, 4, 6, 8, 10, 12, 61, 130],
            widths: &[1, 1, 1, 1, 1, 1, 1, 41, 2],
            ratio: F::from_u8(3),
        },
    )
    .unwrap()
    .with_opening_columns(&cache);
    let packed = Mat::compact_signed_unit_from_column_masks(D, 3, &[1 << 63, 0xa, 0x1001], &[1, 0x100, 4]).unwrap();
    let mut dense = Mat::zero(D, 3, F::ZERO);
    dense[(63, 0)] = F::from_u64(2048);
    dense[(17, 1)] = -F::from_u64(2049);
    dense[(2, 2)] = -F::from_u64(3);
    let blocks = [&packed, &dense].map(|m| SuperneoZBlocks::from_witness_mat(m, columns).unwrap());
    let sources = blocks.each_ref();
    let weights = EqualityWeights::new(&[K::from_coeffs([F::from_u64(19), F::from_u64(17)]); 8]);
    let evaluate = |separate| {
        let mut out = (0..2)
            .map(|_| V1_2Evaluations {
                eval_k: vec![K::ZERO; D],
                eval_a: vec![vec![K::ZERO; D]; 4],
            })
            .collect::<Vec<_>>();
        let mut bands = OpeningBands::new(rows, usize::MAX, 2, 4096).unwrap();
        bands.separate_matrices = separate;
        bands.evaluate(&weights, &sources, &[0, 1], &mut out, Vec::with_capacity(4096), None);
        out
    };
    assert_eq!(evaluate(false), evaluate(true));
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
    const WIDTHS: &[u8] = &[1, 0, 126, 7, 2];
    let entries = FactoredEntries {
        ports: neo_ccs::sparse::FactoredPorts::Offsets(&[0, 2, 2, 3, 5, 6, 7, 10, 11]),
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
    for block in [0, 1, 2, BAND_COLUMNS / D - 1, BAND_COLUMNS / D] {
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
    for ratio in [F::ZERO, F::ONE, F::NEG_ONE, F::from_u64(3)] {
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
                        ratio,
                    },
                )
                .unwrap(),
                visits: AtomicUsize::new(0),
            };
            for witnesses in [&blocks[..2], &blocks[..]] {
                let expected = eval_real_v1_2_openings_from_rows(&rows, &point, witnesses, usize::MAX).unwrap();
                rows.visits.store(0, Ordering::Relaxed);
                let actual = eval_real_v1_2_openings_from_rows_reusing(
                    &rows,
                    &point,
                    witnesses,
                    usize::MAX,
                    vec![K::ONE; COLUMNS],
                    None,
                    None,
                )
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(rows.visits.load(Ordering::Relaxed) == 0, expects_fast_path);
            }
        }
    }
}
