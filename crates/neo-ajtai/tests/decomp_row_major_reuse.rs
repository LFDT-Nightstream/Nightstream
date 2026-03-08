use neo_ajtai::{decomp_b_row_major, decomp_b_row_major_into, DecompStyle};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

fn sample_vec() -> Vec<Fq> {
    vec![
        Fq::from_u64(0),
        Fq::from_u64(1),
        Fq::from_u64(2),
        Fq::ZERO - Fq::from_u64(1),
        Fq::ZERO - Fq::from_u64(5),
        Fq::from_u64(17),
        Fq::from_u64(123456),
    ]
}

#[test]
fn decomp_b_row_major_into_matches_allocating_path() {
    let z = sample_vec();
    let mut scratch = vec![Fq::from_u64(999); 3];
    for &base in &[2u32, 3, 11] {
        for &style in &[DecompStyle::Balanced, DecompStyle::NonNegative] {
            let expected = decomp_b_row_major(&z, base, /*d=*/ 4, style);
            decomp_b_row_major_into(&z, base, /*d=*/ 4, style, &mut scratch);
            assert_eq!(scratch, expected, "base={base} style={style:?}");
        }
    }
}

#[test]
fn decomp_b_row_major_into_reuses_buffer_across_calls() {
    let z0 = sample_vec();
    let z1 = vec![
        Fq::from_u64(7),
        Fq::ZERO - Fq::from_u64(9),
        Fq::from_u64(11),
        Fq::from_u64(13),
    ];

    let mut scratch = vec![Fq::from_u64(42); 64];
    decomp_b_row_major_into(&z0, /*b=*/ 3, /*d=*/ 4, DecompStyle::Balanced, &mut scratch);
    let first = scratch.clone();
    assert_eq!(first, decomp_b_row_major(&z0, 3, 4, DecompStyle::Balanced));

    decomp_b_row_major_into(&z1, /*b=*/ 11, /*d=*/ 4, DecompStyle::Balanced, &mut scratch);
    assert_eq!(scratch, decomp_b_row_major(&z1, 11, 4, DecompStyle::Balanced));
}
