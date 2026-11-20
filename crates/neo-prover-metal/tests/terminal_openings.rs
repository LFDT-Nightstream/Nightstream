#![cfg(feature = "metal")]

use std::{cell::Cell, ops::Range};

use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_prover_metal::MetalRowProver;
use neo_reductions::superneo_eval::{
    eval_real_v1_2_openings_from_rows, eval_real_v1_2_openings_from_rows_reusing, FactoredEntries, FactoredRows,
    FactoredSlots, MatrixRowSink, MatrixRows, MatrixShape, OpeningColumns, OpeningMatrix, OpeningWitness,
    SuperneoZBlocks, TerminalOpeningDevice,
};
use neo_reductions::PiCcsError;
use p3_field::PrimeCharacteristicRing;

struct Rows<'a>(FactoredRows<'a>);
impl MatrixRows for Rows<'_> {
    fn shape(&self) -> MatrixShape {
        self.0.shape()
    }
    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.0))
    }
    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.0.visit_rows(rows, sink)
    }
}

struct Limited<'a> {
    metal: &'a MetalRowProver,
    calls: Cell<usize>,
}
impl TerminalOpeningDevice for Limited<'_> {
    fn supports(&self, blocks: usize, witnesses: usize) -> bool {
        self.metal.supports(blocks, witnesses)
    }
    fn evaluate(
        &self,
        matrix: &OpeningMatrix<'_>,
        weights: &[K],
        witnesses: &[OpeningWitness<'_>],
        budget: usize,
        out: &mut [[K; D]],
    ) -> bool {
        let remaining = if self.calls.replace(self.calls.get() + 1) == 0 {
            budget
        } else {
            0
        };
        self.metal
            .evaluate(matrix, weights, witnesses, remaining, out)
    }
}

#[test]
fn first_and_later_terminal_openings_match_all_cpu_coordinates_and_recover_from_resource_limits() {
    let metal = MetalRowProver::new().unwrap();
    for blocks in [17, 2055] {
        let columns = blocks * D;
        let shape = MatrixShape {
            rows: 273,
            columns,
            matrices: 4,
        };
        let mut offsets = vec![0];
        let mut slots = Vec::new();
        let mut coefficients = Vec::new();
        for row in 0..shape.rows {
            for matrix in 0..4 {
                for slot in [0, (row + matrix) % 4] {
                    slots.push(slot as u32);
                    coefficients.push(match (row + matrix + slot) % 5 {
                        0 => F::ONE,
                        1 => -F::ONE,
                        _ => F::from_usize(3 + row + matrix),
                    });
                }
                offsets.push(slots.len());
            }
        }
        let starts = [0, 61, 2 * D as u32 + 7, (columns - D + 47) as u32];
        let cache = OpeningColumns::default();
        let relation = Rows(
            FactoredRows::new(
                shape,
                FactoredEntries {
                    offsets: &offsets,
                    slots: &slots,
                    coefficients: &coefficients,
                },
                FactoredSlots {
                    starts: &starts,
                    widths: &[3, 18, 67, 13],
                    ratio: F::from_u8(7),
                },
            )
            .unwrap()
            .with_opening_columns(&cache),
        );
        let positive = (0..blocks).map(|i| 1u64 << (i % 64)).collect::<Vec<_>>();
        let negative = (0..blocks)
            .map(|i| 1u64 << ((i + 19) % 64))
            .collect::<Vec<_>>();
        let signed = Mat::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap();
        let mut dense = Mat::zero(D, blocks, F::ZERO);
        for i in 0..blocks {
            dense[(i % D, i)] = F::from_usize(i + 2);
            dense[((i + 9) % D, i)] = -F::from_usize(i + 3);
        }
        let witnesses = [signed, dense].map(|w| SuperneoZBlocks::from_witness_mat(&w, columns).unwrap());
        for count in [1, 2] {
            let point = (0..columns.next_power_of_two().ilog2() as usize)
                .map(|i| K::from_coeffs([F::from_usize(i + 3 + count), F::from_usize(i * 5 + 4)]))
                .collect::<Vec<_>>();
            let source = &witnesses[..count];
            let expected = eval_real_v1_2_openings_from_rows(&relation, &point, source, 64 << 20).unwrap();
            let evaluate = |device: Option<&dyn TerminalOpeningDevice>| {
                eval_real_v1_2_openings_from_rows_reusing(&relation, &point, source, 64 << 20, Vec::new(), None, device)
                    .unwrap()
            };
            let before = metal.activity();
            assert_eq!(evaluate(None), expected);
            assert_eq!(metal.activity(), before);
            assert_eq!(evaluate(Some(&metal)), expected);
            let after = metal.activity();
            if blocks == 17 {
                assert_eq!(after, before);
            } else {
                assert_eq!(after.command_buffers - before.command_buffers, 4);
                assert_eq!(after.host_waits - before.host_waits, 4);
                assert_eq!(
                    after.downloaded_bytes - before.downloaded_bytes,
                    (4 * count * D * size_of::<K>()) as u64
                );
                let limited = Limited {
                    metal: &metal,
                    calls: Cell::new(0),
                };
                assert_eq!(evaluate(Some(&limited)), expected);
                assert_eq!(limited.calls.get(), 2);
                assert_eq!(metal.activity().host_waits - after.host_waits, 1);
            }
        }
    }
}
