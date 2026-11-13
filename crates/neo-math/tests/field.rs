//! `F_q` with `q = 2^64 − 59` and `K = F_q[u]/(u² − 7)`, against integer
//! references (`decisions/field-2p64-59-ring-x64.md`).

use neo_math::{Fq, KExtensions, K};
use p3_field::extension::BinomiallyExtendable;
use p3_field::integers::QuotientMap;
use p3_field::{Field, PermutationMonomial, PrimeCharacteristicRing, PrimeField, PrimeField64};
use rand_chacha::{rand_core::RngCore, rand_core::SeedableRng, ChaCha8Rng};

const Q: u64 = 18_446_744_073_709_551_557;
const Q_MINUS_1_FACTORS: [(u64, u32); 5] = [(2, 2), (11, 1), (137, 1), (547, 1), (5_594_472_617_641, 1)];
const Q_PLUS_1_FACTORS: [(u64, u32); 4] = [(2, 1), (3, 1), (7, 1), (439_208_192_231_179_799, 1)];

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(m)) as u64
}

fn pow_mod(mut base: u64, mut exponent: u64, m: u64) -> u64 {
    let mut acc = 1;
    while exponent > 0 {
        if exponent & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        base = mul_mod(base, base, m);
        exponent >>= 1;
    }
    acc
}

/// Miller–Rabin with the first twelve primes as bases, which is exact below 2^64.
fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    if let Some(&p) = BASES.iter().find(|&&p| n.is_multiple_of(p)) {
        return n == p;
    }
    let shift = (n - 1).trailing_zeros();
    BASES.iter().all(|&base| {
        let mut x = pow_mod(base, (n - 1) >> shift, n);
        x == 1
            || (0..shift).any(|_| {
                let hit = x == n - 1;
                x = mul_mod(x, x, n);
                hit
            })
    })
}

fn samples() -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(0x9e37_79b9);
    let mut values = vec![0, 1, 2, 58, 59, 60, 7, 1 << 32, (1 << 32) - 1, Q / 2, Q / 2 + 1];
    values.extend([Q - 2, Q - 1, Q, Q + 1, u64::MAX - 1, u64::MAX]);
    values.extend((0..48).map(|_| rng.next_u64()));
    values
}

fn product(factors: &[(u64, u32)]) -> u128 {
    factors.iter().map(|&(p, e)| u128::from(p).pow(e)).product()
}

#[test]
fn modulus_is_the_prime_two_to_the_64_minus_59() {
    assert_eq!(Fq::ORDER_U64, Q);
    assert_eq!(u128::from(Q), (1u128 << 64) - 59);
    assert!(is_prime(Q));
    assert_eq!(Fq::order(), Q.into());
}

#[test]
fn values_are_canonical_and_reduce_from_any_u64() {
    for x in samples() {
        let element = Fq::from_u64(x);
        assert_eq!(element.as_canonical_u64(), x % Q);
        assert_eq!(Fq::new(x), element);
        assert_eq!(element.as_canonical_biguint(), (x % Q).into());
        assert_eq!(element.to_string(), (x % Q).to_string());
    }
    assert_eq!(Fq::from_u64(Q), Fq::ZERO);
    assert_eq!(Fq::NEG_ONE.as_canonical_u64(), Q - 1);
}

#[test]
fn arithmetic_matches_the_u128_reference() {
    let q = u128::from(Q);
    for a in samples() {
        for b in samples() {
            let (x, y) = (Fq::from_u64(a), Fq::from_u64(b));
            let (a, b) = (u128::from(a) % q, u128::from(b) % q);
            assert_eq!(u128::from((x + y).as_canonical_u64()), (a + b) % q);
            assert_eq!(u128::from((x - y).as_canonical_u64()), (a + q - b) % q);
            assert_eq!(u128::from((x * y).as_canonical_u64()), a * b % q);
            assert_eq!(u128::from((-x).as_canonical_u64()), (q - a) % q);
            assert_eq!(x.halve() + x.halve(), x);
            assert_eq!(x.double(), x + x);
            assert_eq!(x.square(), x * x);
        }
    }
}

#[test]
fn field_axioms_hold() {
    let values = samples();
    for (i, &a) in values.iter().enumerate() {
        let a = Fq::from_u64(a);
        let b = Fq::from_u64(values[(i + 7) % values.len()]);
        let c = Fq::from_u64(values[(i + 19) % values.len()]);
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a + Fq::ZERO, a);
        assert_eq!(a * Fq::ONE, a);
        assert_eq!(a - a, Fq::ZERO);
    }
}

#[test]
fn every_nonzero_element_has_an_inverse() {
    assert_eq!(Fq::ZERO.try_inverse(), None);
    for x in samples().into_iter().filter(|x| x % Q != 0) {
        let x = Fq::from_u64(x);
        assert_eq!(x * x.inverse(), Fq::ONE);
        assert_eq!(Fq::ONE / x, x.inverse());
    }
    assert_eq!(Fq::NEG_ONE.inverse(), Fq::NEG_ONE);
}

#[test]
fn generator_has_order_q_minus_1() {
    assert_eq!(product(&Q_MINUS_1_FACTORS), u128::from(Q - 1));
    assert_eq!(Fq::GENERATOR.exp_u64(Q - 1), Fq::ONE);
    for (p, _) in Q_MINUS_1_FACTORS {
        assert!(is_prime(p), "{p}");
        assert_ne!(Fq::GENERATOR.exp_u64((Q - 1) / p), Fq::ONE, "order divides (q − 1)/{p}");
    }
}

#[test]
fn seventh_power_is_a_permutation_with_the_given_root() {
    assert!(Q_MINUS_1_FACTORS.iter().all(|&(p, _)| p != 7));
    for x in samples() {
        let x = Fq::from_u64(x);
        assert_eq!(x.exp_u64(7).injective_exp_root_n(), x);
    }
}

#[test]
fn integer_conversions_match_euclidean_reduction() {
    let q = i128::from(Q);
    let reference = |x: i128| Fq::from_u64(x.rem_euclid(q) as u64);
    for x in [
        i64::MIN,
        i64::MIN + 1,
        -(Q as i128 / 2) as i64,
        -59,
        -1,
        0,
        1,
        59,
        i64::MAX,
    ] {
        assert_eq!(Fq::from_i64(x), reference(x.into()));
        assert_eq!(Fq::from_i128(x.into()), reference(x.into()));
    }
    for x in [i128::MIN, i128::MIN + 1, -q, -q - 1, q, q + 1, i128::MAX] {
        assert_eq!(Fq::from_i128(x), reference(x));
    }
    for x in [0, u128::from(Q), u128::from(u64::MAX) + 1, u128::MAX - 1, u128::MAX] {
        assert_eq!(Fq::from_u128(x).as_canonical_u64(), (x % u128::from(Q)) as u64);
    }
    assert_eq!(Fq::from_u8(255) + Fq::from_i8(-128), Fq::from_u32(127));
    assert_eq!(Fq::from_i32(i32::MIN), -Fq::from_u64(1 << 31));
}

#[test]
fn checked_conversions_accept_only_canonical_integers() {
    assert_eq!(Fq::from_canonical_checked(Q - 1), Some(Fq::NEG_ONE));
    assert_eq!(Fq::from_canonical_checked(Q), None);
    assert_eq!(Fq::from_canonical_checked(u64::MAX), None);
    assert_eq!(Fq::from_canonical_checked(u128::from(Q)), None);
    assert_eq!(Fq::from_canonical_checked(-1i64), Some(Fq::NEG_ONE));
    let half = (Q / 2) as i128;
    assert_eq!(Fq::from_canonical_checked(half), Some(Fq::from_u64(Q / 2)));
    assert_eq!(Fq::from_canonical_checked(-half), Some(Fq::from_u64(Q / 2 + 1)));
    assert_eq!(Fq::from_canonical_checked(half + 1), None);
    assert_eq!(Fq::from_canonical_checked(-half - 1), None);
}

#[test]
fn sums_of_products_match_the_reference_at_the_extremes() {
    let reference = |pairs: &[(Fq, Fq)]| pairs.iter().fold(Fq::ZERO, |acc, &(a, b)| acc + a * b);

    // (q − 1)² is above 2^127, so the 128-bit accumulator wraps on every second term.
    let extreme = vec![(Fq::NEG_ONE, Fq::NEG_ONE); 1_000];
    assert_eq!(Fq::sum_of_products(extreme.iter().copied()), Fq::from_u64(1_000));
    assert_eq!(Fq::sum_of_products([]), Fq::ZERO);

    let values = samples();
    let pairs: Vec<(Fq, Fq)> = values
        .iter()
        .zip(values.iter().rev())
        .map(|(&a, &b)| (Fq::from_u64(a), Fq::from_u64(b)))
        .collect();
    assert_eq!(Fq::sum_of_products(pairs.iter().copied()), reference(&pairs));

    let lhs: [Fq; 5] = core::array::from_fn(|i| pairs[i].0);
    let rhs: [Fq; 5] = core::array::from_fn(|i| pairs[i].1);
    assert_eq!(Fq::dot_product(&lhs, &rhs), reference(&pairs[..5]));
    assert_eq!(Fq::dot_product(&[Fq::NEG_ONE; 3], &[Fq::NEG_ONE; 3]), Fq::from_u64(3));

    let sum: Fq = vec![Fq::NEG_ONE; 1_000].into_iter().sum();
    assert_eq!(sum, -Fq::from_u64(1_000));
    let product: Fq = [2u64, 3, 5].into_iter().map(Fq::from_u64).product();
    assert_eq!(product, Fq::from_u64(30));
}

#[test]
fn serde_round_trips_and_rejects_values_at_or_above_the_modulus() {
    for x in samples() {
        let element = Fq::from_u64(x);
        let json = serde_json::to_string(&element).unwrap();
        assert_eq!(json, (x % Q).to_string());
        assert_eq!(serde_json::from_str::<Fq>(&json).unwrap(), element);
    }
    for bad in [
        Q.to_string(),
        u64::MAX.to_string(),
        "-1".into(),
        "18446744073709551616".into(),
    ] {
        assert!(serde_json::from_str::<Fq>(&bad).is_err(), "{bad}");
    }
    let k = K::from_coeffs([Fq::from_u64(5), Fq::NEG_ONE]);
    assert_eq!(
        serde_json::from_str::<K>(&serde_json::to_string(&k).unwrap()).unwrap(),
        k
    );
}

#[test]
fn seven_is_not_a_square() {
    assert_eq!(pow_mod(7, (Q - 1) / 2, Q), Q - 1);
    assert_eq!(<Fq as BinomiallyExtendable<2>>::W, Fq::from_u64(7));
    assert_eq!(Fq::from_u64(7).exp_u64((Q - 1) / 2), Fq::NEG_ONE);
    assert_eq!(<Fq as BinomiallyExtendable<2>>::DTH_ROOT, Fq::NEG_ONE);
}

#[test]
fn k_is_the_field_with_u_squared_equal_to_7() {
    let u = K::from_coeffs([Fq::ZERO, Fq::ONE]);
    assert_eq!(u * u, K::from(Fq::from_u64(7)));

    let values = samples();
    for (i, &re) in values.iter().enumerate() {
        let a = K::from_coeffs([Fq::from_u64(re), Fq::from_u64(values[(i + 5) % values.len()])]);
        let b = K::from_coeffs([Fq::from_u64(values[(i + 11) % values.len()]), Fq::from_u64(re)]);
        let [a0, a1] = a.as_coeffs();
        let [b0, b1] = b.as_coeffs();
        let seven = Fq::from_u64(7);
        assert_eq!((a * b).as_coeffs(), [a0 * b0 + seven * a1 * b1, a0 * b1 + a1 * b0]);
        assert_eq!(a.conj().conj(), a);
        assert_eq!((a * a.conj()).as_coeffs(), [a0 * a0 - seven * a1 * a1, Fq::ZERO]);
        if a != K::ZERO {
            assert_eq!(a * a.inv(), K::ONE);
        }
        assert_eq!(a.scale_base(b0), a * K::from(b0));
        assert_eq!(a.scale_base_k(b), a * b);
        assert_eq!(a.to_limbs_u64(), (a0.as_canonical_u64(), a1.as_canonical_u64()));
    }
    assert_eq!(K::ZERO.try_inverse(), None);
}

#[test]
fn k_generator_has_order_q_squared_minus_1() {
    assert_eq!(product(&Q_PLUS_1_FACTORS), u128::from(Q) + 1);
    let generator = K::GENERATOR;
    assert_eq!(
        generator,
        K::from_coeffs(<Fq as BinomiallyExtendable<2>>::EXT_GENERATOR)
    );
    assert_eq!(generator.exp_u64(Q - 1).exp_u64(Q + 1), K::ONE);
    for (p, _) in Q_MINUS_1_FACTORS {
        assert_ne!(generator.exp_u64((Q - 1) / p).exp_u64(Q + 1), K::ONE, "{p}");
    }
    for (p, _) in Q_PLUS_1_FACTORS {
        assert!(is_prime(p), "{p}");
        let cofactor = ((u128::from(Q) + 1) / u128::from(p)) as u64;
        assert_ne!(generator.exp_u64(Q - 1).exp_u64(cofactor), K::ONE, "{p}");
    }
}
