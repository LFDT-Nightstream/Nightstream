//! The succinct verifier's closed forms against brute-force sums, each with
//! a negative control that must fail.

use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing as _;
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing};

use super::word;
use crate::field::{eq_table, Ext, Gl, Kx};
use crate::mle::{chi, eval_k, identity, lane_split, less_than, scaled_eq_point, step};
use crate::sumcheck;

fn ext(seed: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, i as u64)))
}

fn k(seed: u64) -> K {
    K::from_coeffs([F::from_u64(word(seed, 0)), F::from_u64(word(seed, 1))])
}

fn points(seed: u64, count: usize) -> Vec<Ext> {
    (0..count).map(|i| ext(seed * 1000 + i as u64)).collect()
}

/// `χ_r(c)` in `K`, low bit first, bits beyond `r` must be zero.
fn chi_at(r: &[K], c: usize) -> K {
    r.iter()
        .enumerate()
        .map(|(t, &value)| if (c >> t) & 1 == 1 { value } else { K::ONE - value })
        .product()
}

#[test]
fn scaled_eq_point_matches_power_weights() {
    let f: Vec<Ext> = (0..64)
        .map(|i| if i < D { ext(10 + i as u64) } else { Ext::ZERO })
        .collect();
    let c = ext(3);
    let direct: Ext = (0..64u64).map(|i| c.exp_u64(i) * f[i as usize]).sum();
    let (kappa, point) = scaled_eq_point(c, 6).unwrap();
    assert_eq!(direct, kappa * sumcheck::evaluate(&f, &point));
    // Control: the unscaled point (c, c^2, ...) is not the right one.
    let unscaled: Vec<Ext> = (0..6).map(|t| c.exp_u64(1 << t)).collect();
    assert_ne!(direct, kappa * sumcheck::evaluate(&f, &unscaled));
}

#[test]
fn intervals_steps_and_identity_match_their_tables() {
    let n = 8;
    let point = points(4, n);
    let eq = eq_table(&point);
    for bound in [0u64, 1, 77, 200, 255, 256] {
        let direct: Ext = eq.iter().take(bound as usize).copied().sum();
        assert_eq!(less_than(&point, bound), direct, "bound {bound}");
    }
    let starts = [0u64, 9, 40, 41, 130, 199];
    let values: Vec<Ext> = (0..starts.len()).map(|s| ext(50 + s as u64)).collect();
    let end = 230;
    let table: Vec<Ext> = (0..256u64)
        .map(|x| match starts.iter().rposition(|&a| a <= x) {
            Some(s) if x < end => values[s],
            _ => Ext::ZERO,
        })
        .collect();
    assert_eq!(step(&point, &starts, &values, end), sumcheck::evaluate(&table, &point));
    assert_ne!(
        step(&point, &starts, &values, end + 1),
        sumcheck::evaluate(&table, &point)
    );
    let identity_table: Vec<Ext> = (0..256u64).map(|x| Ext::from(Gl::from_u64(x))).collect();
    assert_eq!(identity(&point), sumcheck::evaluate(&identity_table, &point));
}

#[test]
fn chi_at_an_ext_point_matches_the_k_table() {
    let r: Vec<K> = (0..9).map(|t| k(70 + t)).collect();
    let point = points(5, 6);
    let eq = eq_table(&point);
    let direct = (0..64).fold(Kx::ZERO, |acc, x| acc.add(Kx::from_k(chi_at(&r, x)).scale(eq[x])));
    assert_eq!(chi(&r, &point), direct);
}

#[test]
fn lane_split_matches_every_run_start() {
    let tau: [Ext; D] = std::array::from_fn(|l| ext(100 + l as u64));
    let blocks = 9;
    let block_weight: Vec<Ext> = (0..=blocks)
        .map(|b| if b < blocks { ext(300 + b as u64) } else { Ext::ZERO })
        .collect();
    let g = |c: usize| block_weight[c / D] * tau[c % D];
    for (length, ratio) in [(41, 3u64), (1, 1), (54, 5), (17, 2)] {
        let ratio = Gl::from_u64(ratio);
        let [head, tail] = lane_split(&tau, length, ratio);
        for c in 0..=D * blocks - length {
            let (b, l) = (c / D, c % D);
            let direct: Ext = (0..length)
                .map(|k| g(c + k) * ratio.exp_u64(k as u64))
                .sum();
            assert_eq!(
                direct,
                block_weight[b] * head[l] + block_weight[b + 1] * tail[l],
                "len {length} c {c}"
            );
        }
    }
    // Control: a tail exponent of 53 - l instead of 54 - l breaks the split.
    let ratio = Gl::from_u64(3);
    let [head, tail] = lane_split(&tau, 41, ratio);
    let c = D * 2 + 30;
    let direct: Ext = (0..41).map(|k| g(c + k) * ratio.exp_u64(k as u64)).sum();
    let wrong_tail = tail[30] * ratio.inverse();
    assert_ne!(direct, block_weight[2] * head[30] + block_weight[3] * wrong_tail);
}

#[test]
fn eval_k_division_matches_brute_force() {
    let block_bits = 6;
    let s = points(6, block_bits);
    let eq = eq_table(&s);
    let tau: [Ext; D] = std::array::from_fn(|l| ext(400 + l as u64));
    let r: Vec<K> = (0..14).map(|t| k(500 + t)).collect();
    for blocks in [37, 50, 64] {
        let direct = (0..blocks).fold(Kx::ZERO, |acc, b| {
            (0..D).fold(acc, |acc, l| {
                acc.add(Kx::from_k(chi_at(&r, D * b + l)).scale(eq[b] * tau[l]))
            })
        });
        assert_eq!(eval_k(&r, &s, &tau, blocks), direct, "{blocks} blocks");
        // Control: one block too many.
        if blocks < 64 {
            assert_ne!(eval_k(&r, &s, &tau, blocks + 1), direct);
        }
    }
}
