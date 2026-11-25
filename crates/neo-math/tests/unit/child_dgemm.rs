use super::*;
use crate::signed_sums::{SignedShiftSums, SplitRing};

#[test]
fn batched_toom4_matches_integer_shifts_across_panels_and_groups() {
    let count = GROUP + 5;
    let keys: Vec<_> = (0..count * D)
        .map(|i| {
            (i as u64)
                .wrapping_mul(0xca63_7e51_14c9_cca5)
                .rotate_left((i / D) as u32)
        })
        .collect();
    let planes: Vec<_> = (0..6)
        .map(|p| {
            let a = (0..count)
                .map(|i| 0x29a1_54c1_82a3_239f_u64.rotate_left((p + i) as u32))
                .collect::<Vec<_>>();
            let b = (0..count)
                .map(|i| 0x4837_68ab_ee84_d51a_u64.rotate_left((p * 7 + i / 5) as u32))
                .collect::<Vec<_>>();
            (a, b)
        })
        .collect();
    for (n, p) in [(0, 1), (1, 3), (TILE - 1, 4), (TILE + 1, 6), (GROUP + 1, 5)] {
        let indices: Vec<u32> = (0..n as u32).rev().collect();
        let mut out = vec![[F::ZERO; D]; p];
        let borrow: Vec<_> = planes[..p]
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        assert!(compute(&keys, &indices, &borrow, &mut out));
        for ((a, b), v) in planes.iter().zip(out) {
            let mut expected = SignedShiftSums::zero();
            for i in 0..n {
                let ring = SplitRing::from_u64(keys[i * D..][..D].try_into().unwrap());
                expected.add_signed_units(&ring, a[i], b[i]);
            }
            assert_eq!(expected.reduce(), v);
        }
    }
}
