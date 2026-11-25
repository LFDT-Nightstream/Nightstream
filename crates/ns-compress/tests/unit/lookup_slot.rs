use neo_math::{F, K};
use p3_field::{Field, PrimeCharacteristicRing};

use super::{equality, EqualityWeights, LookupPolynomial, Slice, Slot};
use crate::lookup::code;

#[test]
fn the_first_lookup_round_and_bind_use_each_witness_value() {
    let slice = Slice { offset: 0, vars: 14 };
    let bound = 512;
    let poles = [1001, 1003, 1007, 1009].map(F::from_u64);
    let polynomial = LookupPolynomial::new(&poles, &[K::ONE; 4]);
    let inverses: Vec<_> = (0..=2 * bound)
        .map(|i| {
            let z = F::from_u64(i) - F::from_u64(bound);
            poles.iter().map(|&t| t - z).product::<F>().inverse()
        })
        .collect();
    let r = neo_math::from_complex(F::from_u8(7), F::from_u8(9));
    let beta = vec![r; slice.vars];
    let eq = EqualityWeights::new(&beta[1..]);
    for last in [4, 260, 513] {
        let mut witness = vec![F::ZERO; slice.len()];
        for (i, v) in witness[..8192].iter_mut().enumerate() {
            *v = F::from_usize(i % 7) - F::from_u8(3);
        }
        witness[8192] = F::from_u64(last);
        let codes: Vec<_> = witness.iter().map(|&v| code(v, bound)).collect();
        let mut slot = Slot::new(slice, bound, &codes, &inverses, [r, r.square()], &witness, &beta);
        let inverse = |i: usize| {
            inverses
                .get(usize::from(codes[i]))
                .copied()
                .unwrap_or(F::ZERO)
        };
        let pair = |i: usize, t: K| {
            let k = 2 * i;
            (
                t * (witness[k + 1] - witness[k]) + witness[k],
                t * (inverse(k + 1) - inverse(k)) + inverse(k),
            )
        };
        let expected = std::array::from_fn(|j| {
            let t = K::from_usize(j);
            let scale = r * equality(beta[0], t);
            (0..slice.len() / 2)
                .map(|i| {
                    let (z, u) = pair(i, t);
                    let (den, num) = polynomial.at(z);
                    u * (scale * eq.at(i) * den + r.square() * num)
                })
                .sum()
        });
        assert_eq!(slot.round(0, &beta, K::ONE, &polynomial), expected);
        slot.bind(r);
        for i in 0..slice.len() / 2 {
            let cell = slot.get(i);
            assert_eq!((cell.value, cell.inverse), pair(i, r));
        }
    }
}
