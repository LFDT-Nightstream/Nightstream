use neo_math::ring::test_reduce_mod_phi_81;
use neo_math::{Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rq<F>(mut f: F) -> Rq
where
    F: FnMut(usize) -> Fq,
{
    let mut coeffs = [Fq::ZERO; D];
    for i in 0..D {
        coeffs[i] = f(i);
    }
    Rq(coeffs)
}

fn schoolbook_mul_reference(lhs: &Rq, rhs: &Rq) -> Rq {
    let mut tmp = [Fq::ZERO; 2 * D - 1];
    for i in 0..D {
        let ai = lhs.0[i];
        for j in 0..D {
            tmp[i + j] += ai * rhs.0[j];
        }
    }
    test_reduce_mod_phi_81(&mut tmp);
    let mut out = [Fq::ZERO; D];
    out.copy_from_slice(&tmp[..D]);
    Rq(out)
}

fn adversarial_cases() -> Vec<Rq> {
    let mut cases = Vec::new();
    cases.push(Rq::zero());
    cases.push(Rq::one());
    cases.push(make_rq(|_| Fq::ONE));
    cases.push(make_rq(|i| if i % 2 == 0 { Fq::ONE } else { -Fq::ONE }));
    cases.push(make_rq(|i| match i % 4 {
        0 => Fq::ZERO,
        1 => Fq::ONE,
        2 => -Fq::ONE,
        _ => Fq::from_u64(2),
    }));
    cases.push(make_rq(|i| {
        if i == 0 || i == D / 2 || i == D - 1 {
            -Fq::from_u64(2)
        } else {
            Fq::ZERO
        }
    }));
    cases.push(make_rq(|i| {
        if i < 18 {
            Fq::from_u64((i as u64) + 1)
        } else if i < 36 {
            -Fq::from_u64((i as u64) - 17)
        } else {
            Fq::from_u64((i as u64) * 3 + 7)
        }
    }));
    cases.push(make_rq(|i| {
        if i % 3 == 0 {
            Fq::from_u64(u64::MAX)
        } else if i % 3 == 1 {
            Fq::from_u64((1u64 << 63) - 1)
        } else {
            Fq::from_u64((1u64 << 62) + 9)
        }
    }));
    cases
}

#[test]
fn rq_mul_matches_reference_randomized() {
    let mut rng = StdRng::seed_from_u64(0x9e37_79b9_7f4a_7c15);
    for _ in 0..256 {
        let lhs = Rq::random_uniform(&mut rng);
        let rhs = Rq::random_uniform(&mut rng);
        let got = lhs.mul(&rhs);
        let want = schoolbook_mul_reference(&lhs, &rhs);
        assert_eq!(got, want);
    }
}

#[test]
fn rq_mul_matches_reference_adversarial() {
    let cases = adversarial_cases();
    for lhs in &cases {
        for rhs in &cases {
            let got = lhs.mul(rhs);
            let want = schoolbook_mul_reference(lhs, rhs);
            assert_eq!(got, want, "mismatch for lhs={lhs:?}, rhs={rhs:?}");
        }
    }
}

#[test]
fn rq_mul_matches_reference_monomials() {
    for i in 0..D {
        let lhs = make_rq(|j| if j == i { Fq::ONE } else { Fq::ZERO });
        for j in 0..D {
            let rhs = make_rq(|k| if k == j { Fq::ONE } else { Fq::ZERO });
            let got = lhs.mul(&rhs);
            let want = schoolbook_mul_reference(&lhs, &rhs);
            assert_eq!(got, want, "basis mismatch for X^{i} * X^{j}");
        }
    }
}
