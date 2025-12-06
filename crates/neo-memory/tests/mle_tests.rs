use neo_math::K;
use neo_memory::mle::{build_chi_table, lt_eval, mle_eval};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn bits(num: usize, ell: usize) -> Vec<Goldilocks> {
    (0..ell)
        .map(|i| {
            if (num >> i) & 1 == 1 {
                Goldilocks::ONE
            } else {
                Goldilocks::ZERO
            }
        })
        .collect()
}

#[test]
fn lt_matches_integer_order() {
    let ell = 3;
    for a in 0..(1 << ell) {
        for b in 0..(1 << ell) {
            let j_prime = bits(a, ell);
            let j = bits(b, ell);
            let lt = lt_eval(&j_prime, &j);
            let expected = if a < b { Goldilocks::ONE } else { Goldilocks::ZERO };
            assert_eq!(lt, expected, "lt_eval({}, {})", a, b);
        }
    }
}

#[test]
fn chi_table_is_partition_of_unity() {
    let r = vec![Goldilocks::from_u64(5), Goldilocks::from_u64(7)];
    let chi = build_chi_table(&r);
    let sum: Goldilocks = chi.iter().copied().sum();
    assert_eq!(sum, Goldilocks::ONE);
}

#[test]
fn mle_eval_matches_manual_sum() {
    let v = vec![
        Goldilocks::from_u64(1),
        Goldilocks::from_u64(2),
        Goldilocks::from_u64(3),
        Goldilocks::from_u64(4),
    ];
    let r = vec![K::from(Goldilocks::from_u64(9)), K::from(Goldilocks::from_u64(11))];
    let chi = build_chi_table(&r);
    let expected = v
        .iter()
        .zip(chi.iter())
        .fold(K::ZERO, |acc, (val, weight)| acc + K::from(*val) * *weight);
    let eval = mle_eval(&v, &r);
    assert_eq!(eval, expected);
}
