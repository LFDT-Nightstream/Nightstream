use super::*;
use crate::superneo_eval::fill_geometric_projection;

#[test]
fn contracted_slot_intervals_match_expanded_projection() {
    let mut digit = vec![F::ZERO; D * 9];
    let mut seed = 42_163u64;
    for slot in &mut digit[..D * 6 + 37] {
        seed = seed.wrapping_mul(479_851_321_590_327).wrapping_add(123);
        *slot = if seed.is_multiple_of(8) {
            F::ZERO
        } else {
            F::from_u64(seed)
        };
    }
    digit[D..2 * D].fill(F::ZERO);
    let witness = SuperneoZBlocks::from_base_row_f(&digit);
    let starts = [0, 19, 62, 65, 131, 256, 371, 19, 501, 37, 511, 0];
    let widths = [0, 2, 9, 41, 123, 1, 64, 35, 12, 0, 64, 4];
    for g in [F::ZERO, F::ONE, F::from_u64(3), -F::ONE] {
        let slots = FactoredSlots {
            starts: &starts,
            widths: &widths,
            ratio: g,
        };
        for ratio in [K::from_coeffs([F::from_u64(13), F::from_u64(7)]), -K::ONE, K::ONE] {
            let mut projection = vec![K::ZERO; digit.len()];
            fill_geometric_projection(std::slice::from_ref(&witness), &[K::ONE], ratio, &mut projection);
            let expected = starts
                .iter()
                .zip(widths)
                .map(|(&start, len)| {
                    let start = start as usize;
                    projection[start..start + usize::from(len)]
                        .iter()
                        .rev()
                        .fold(K::ZERO, |value, &digit| value.scale_base(g) + digit)
                })
                .collect::<Vec<_>>();
            if ratio == K::from(g) {
                assert!(project(&witness, slots, ratio).is_none());
            } else {
                assert_eq!(project(&witness, slots, ratio).unwrap(), expected);
            }
        }
        assert!(project(&witness, slots, K::ZERO).is_none());
    }
}
