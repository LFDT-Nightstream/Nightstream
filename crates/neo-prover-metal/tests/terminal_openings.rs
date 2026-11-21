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
        matrices: &[OpeningMatrix<'_>],
        weights: &[K],
        witnesses: &[OpeningWitness<'_>],
        budget: usize,
        out: &mut [[K; D]],
    ) -> bool {
        self.calls.set(self.calls.get() + 1);
        let mut one = vec![[K::ZERO; D]; witnesses.len()];
        let mut low = 0;
        let mut high = budget;
        assert!(self
            .metal
            .evaluate(&matrices[..1], weights, witnesses, high, &mut one));
        while high - low > 4096 {
            let middle = low + (high - low) / 2;
            if self
                .metal
                .evaluate(&matrices[..1], weights, witnesses, middle, &mut one)
            {
                high = middle;
            } else {
                low = middle;
            }
        }
        out.fill([K::ONE; D]);
        let before = self.metal.activity();
        assert!(!self.metal.evaluate(matrices, weights, witnesses, high, out));
        let after = self.metal.activity();
        assert_eq!(after.command_buffers, before.command_buffers + 1);
        assert_eq!(after.host_waits, before.host_waits);
        assert!(out.iter().flatten().all(|&v| v == K::ONE));
        false
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
                    ports: neo_ccs::sparse::FactoredPorts::Offsets(&offsets),
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
        let mut small = Mat::zero(D, blocks, F::ZERO);
        for i in 0..blocks {
            small[(i % D, i)] = F::from_i8(i8::MIN);
            small[((i + 9) % D, i)] = F::from_i8(i8::MAX);
        }
        let witnesses = [signed, dense, small].map(|w| SuperneoZBlocks::from_witness_mat(&w, columns).unwrap());
        for count in 1..=witnesses.len() {
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
                assert_eq!(after.command_buffers - before.command_buffers, 1);
                assert_eq!(after.host_waits - before.host_waits, 1);
                assert_eq!(after.dispatches - before.dispatches, (4 * (3 + 3 * count)) as u64);
                assert_eq!(
                    after.downloaded_bytes - before.downloaded_bytes,
                    (4 * count * D * size_of::<K>()) as u64
                );
                let limited = Limited {
                    metal: &metal,
                    calls: Cell::new(0),
                };
                assert_eq!(evaluate(Some(&limited)), expected);
                assert_eq!(limited.calls.get(), 1);
            }
        }
    }
}

#[test]
fn translated_short_and_heavy_columns_match_the_direct_openings_on_metal() {
    struct Translated<'a>(&'a MetalRowProver);
    impl TerminalOpeningDevice for Translated<'_> {
        fn supports(&self, blocks: usize, witnesses: usize) -> bool {
            self.0.supports(blocks, witnesses)
        }
        fn evaluate(
            &self,
            matrices: &[OpeningMatrix<'_>],
            weights: &[K],
            witnesses: &[OpeningWitness<'_>],
            budget: usize,
            out: &mut [[K; D]],
        ) -> bool {
            assert!(matrices
                .iter()
                .all(|m| m.translated && !m.tiles.heavy.is_empty()));
            assert!(self.0.evaluate(matrices, weights, witnesses, budget, out));
            true
        }
    }
    let blocks = 2055;
    let shape = MatrixShape {
        rows: 4700,
        columns: blocks * D,
        matrices: 4,
    };
    let mut row_terms = vec![Vec::new(); shape.rows];
    for slot in 0..4105 {
        let count = if slot % 251 == 0 { 513 } else { 8 };
        for j in 0..count {
            row_terms[slot + j].push((slot as u32, F::from_usize(j % 7 + 1)));
        }
    }
    let mut offsets = vec![0];
    let mut slots = Vec::new();
    let mut coefficients = Vec::new();
    for terms in row_terms {
        for _ in 0..4 {
            slots.extend(terms.iter().map(|&(slot, _)| slot));
            coefficients.extend(terms.iter().map(|&(_, coefficient)| coefficient));
            offsets.push(slots.len());
        }
    }
    let starts: Vec<_> = (0..65536).map(|slot| slot * 2).collect();
    let widths = vec![1u8; starts.len()];
    let rows = FactoredRows::new(
        shape,
        FactoredEntries {
            ports: neo_ccs::sparse::FactoredPorts::Offsets(&offsets),
            slots: &slots,
            coefficients: &coefficients,
        },
        FactoredSlots {
            starts: &starts,
            widths: &widths,
            ratio: F::from_u8(3),
        },
    )
    .unwrap();
    let positive: Vec<_> = (0..blocks).map(|i| 1u64 << (i % D)).collect();
    let signed = Mat::<F>::compact_signed_unit_from_column_masks(D, blocks, &positive, &vec![0; blocks]).unwrap();
    let witnesses = [SuperneoZBlocks::from_witness_mat(&signed, shape.columns).unwrap()];
    let point: Vec<_> = (0..shape.columns.next_power_of_two().ilog2())
        .map(|i| K::from_coeffs([F::from_u32(i + 3), F::from_u32(2 * i + 5)]))
        .collect();
    let expected = eval_real_v1_2_openings_from_rows(&Rows(rows), &point, &witnesses, 64 << 20).unwrap();
    let cache = OpeningColumns::default();
    let metal = MetalRowProver::new().unwrap();
    let before = metal.activity();
    let actual = eval_real_v1_2_openings_from_rows_reusing(
        &Rows(rows.with_opening_columns(&cache)),
        &point,
        &witnesses,
        64 << 20,
        Vec::new(),
        None,
        Some(&Translated(&metal)),
    )
    .unwrap();
    assert_eq!(actual, expected);
    let after = metal.activity();
    assert_eq!(after.command_buffers, before.command_buffers + 1);
    assert_eq!(after.host_waits, before.host_waits + 1);
}
