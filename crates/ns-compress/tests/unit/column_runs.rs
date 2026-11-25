use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::SlotRuns;
use p3_field::PrimeCharacteristicRing;

use super::{project, windows, write};
use crate::ring::{dot, LaneTransform, Point};

#[test]
fn geometric_projections_match_expanded_columns_across_tile_and_block_boundaries() {
    let ratios = [F::ZERO, F::ONE, F::new(3)];
    let mut starts = Vec::new();
    let mut widths = Vec::new();
    let mut cols = 0u32;
    for width in [1u8, 3, 0, 41, 64, 65, 1, 129, 255] {
        for _ in 0..33 {
            starts.push(if width == 0 { 0 } else { cols });
            widths.push(width);
            cols += u32::from(width);
        }
    }
    let runs = SlotRuns::new(&starts, &widths).unwrap();
    let weights: Vec<K> = (1..=starts.len() as u64)
        .map(|i| K::from_coeffs([F::new(i * i), F::new(i + 8)]))
        .collect();
    let form: [K; D] = weights[..D].try_into().unwrap();
    for ratio in ratios {
        let mut powers = [F::ONE; 256];
        for i in 1..powers.len() {
            powers[i] = powers[i - 1] * ratio;
        }
        let table = windows(&runs, &form, &powers);
        let mut expanded = vec![K::ZERO; (cols as usize + D).next_multiple_of(D)];
        for (i, &w) in widths.iter().enumerate() {
            let start = starts[i] as usize;
            let mut term = weights[i];
            for dest in &mut expanded[start..][..usize::from(w)] {
                *dest += term;
                term = term.scale_base(ratio);
            }
        }
        for (block, reference) in expanded.chunks(D).enumerate() {
            let mut written = [K::ONE; D];
            write(&runs, &weights, block * D, &mut written, &powers);
            assert_eq!(written, reference);
            let mut value = [K::ONE];
            project(&runs, &weights, block * D, &mut value, &table);
            assert_eq!(value[0], dot(reference, &form));
        }
    }
}

#[test]
fn normalized_rotation_evaluations_match_the_definition() {
    let element = std::array::from_fn(|i| K::from_coeffs([F::new(101 * i as u64), F::new((i + 1) as u64)]));
    for delta in [K::ZERO, K::ONE, K::NEG_ONE, K::from_coeffs([F::new(718), F::new(3)])] {
        let point = Point::new(delta);
        assert_eq!(LaneTransform::new(&point).lanes(&element), point.lanes(&element));
    }
}
