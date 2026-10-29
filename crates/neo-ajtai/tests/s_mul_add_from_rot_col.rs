use neo_ajtai::{s_mul, s_mul_add, s_mul_add_from_rot_col, Commitment};
use neo_math::ring::{cf, Rq as RqEl};
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

fn sample_commitment_with_kappa(kappa: usize) -> Commitment {
    let mut data = Vec::with_capacity(D * kappa);
    for idx in 0..(D * kappa) {
        data.push(match idx % 7 {
            0 => -Fq::ONE,
            1 => Fq::ZERO,
            2 => Fq::ONE,
            3 => Fq::from_u64(5),
            4 => Fq::from_u64(11),
            5 => Fq::from_u64(19),
            _ => Fq::from_u64(23),
        });
    }
    Commitment::new(kappa, data).expect("commitment shape")
}

fn sample_commitment() -> Commitment {
    sample_commitment_with_kappa(3)
}

fn sample_rho() -> RqEl {
    let mut coeffs = [Fq::ZERO; D];
    for (idx, coeff) in coeffs.iter_mut().enumerate() {
        *coeff = match idx % 5 {
            0 => -Fq::ONE,
            1 => Fq::ZERO,
            2 => Fq::ONE,
            3 => Fq::from_u64(7),
            _ => Fq::from_u64(13),
        };
    }
    RqEl(coeffs)
}

#[test]
fn s_mul_add_from_rot_col_matches_ring_action_from_zero() {
    let commitment = sample_commitment();
    let rho = sample_rho();
    let rot_col = cf(rho);

    let mut observed = Commitment::zeros(commitment.kappa());
    s_mul_add_from_rot_col(&mut observed, &rot_col, &commitment);

    let expected = s_mul(&rho, &commitment);
    assert_eq!(observed, expected);
}

#[test]
fn s_mul_add_from_rot_col_matches_existing_accumulate_path() {
    let lhs = sample_commitment();
    let rhs = sample_commitment();
    let rho = sample_rho();
    let rot_col = cf(rho);

    let mut observed = lhs.clone();
    s_mul_add_from_rot_col(&mut observed, &rot_col, &rhs);

    let mut expected = lhs.clone();
    s_mul_add(&mut expected, &rho, &rhs);

    assert_eq!(observed, expected);
}
