use neo_ajtai::{s_mul, scale_commitment, scale_commitment_add_inplace, Commitment};
use neo_math::ring::Rq as RqEl;
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

fn sample_commitment() -> Commitment {
    let mut data = Vec::with_capacity(D * 3);
    for idx in 0..(D * 3) {
        data.push(match idx % 5 {
            0 => -Fq::ONE,
            1 => Fq::ZERO,
            2 => Fq::ONE,
            3 => Fq::from_u64(7),
            _ => Fq::from_u64(11),
        });
    }
    Commitment::new(3, data).expect("commitment shape")
}

#[test]
fn scalar_commitment_scaling_matches_constant_ring_action() {
    let commitment = sample_commitment();
    for scalar in [Fq::ZERO, Fq::ONE, -Fq::ONE, Fq::from_u64(7), Fq::from_u64(123_456)] {
        let expected = s_mul(&RqEl::from_field_scalar(scalar), &commitment);
        let observed = scale_commitment(scalar, &commitment);
        assert_eq!(observed, expected);
    }
}

#[test]
fn scale_commitment_add_inplace_matches_explicit_addition() {
    let lhs = sample_commitment();
    let rhs = sample_commitment();
    let scalar = Fq::from_u64(19);

    let mut observed = lhs.clone();
    scale_commitment_add_inplace(&mut observed, scalar, &rhs);

    let mut expected = lhs.clone();
    expected.add_inplace(&scale_commitment(scalar, &rhs));

    assert_eq!(observed, expected);
}
