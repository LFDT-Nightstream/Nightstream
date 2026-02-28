use neo_math::{Fq, Rq, D};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

const CHALLENGE_BOUND: u128 = 2;
const P5_RING_MUL_BOUND: u128 = 3 * (D as u128) * CHALLENGE_BOUND * CHALLENGE_BOUND;

fn fq_from_i64(v: i64) -> Fq {
    if v >= 0 {
        Fq::from_u64(v as u64)
    } else {
        -Fq::from_u64((-v) as u64)
    }
}

fn centered_abs_fq(x: Fq) -> u128 {
    let q: u128 = Fq::ORDER_U64 as u128;
    let half = (q - 1) / 2;
    let xv = x.as_canonical_u64() as u128;
    if xv <= half { xv } else { q - xv }
}

fn challenge_values() -> [Fq; 5] {
    [
        fq_from_i64(-2),
        fq_from_i64(-1),
        fq_from_i64(0),
        fq_from_i64(1),
        fq_from_i64(2),
    ]
}

fn sample_challenge_poly(rng: &mut ChaCha20Rng) -> Rq {
    let vals = challenge_values();
    let mut coeffs = [Fq::ZERO; D];
    for c in coeffs.iter_mut() {
        let idx = (rng.next_u32() as usize) % vals.len();
        *c = vals[idx];
    }
    Rq(coeffs)
}

#[test]
fn m5_scalar_bounds_hold_for_challenge_values() {
    let vals = challenge_values();
    for x in vals {
        for y in vals {
            let nx = centered_abs_fq(x);
            let ny = centered_abs_fq(y);
            let nadd = centered_abs_fq(x + y);
            let nsub = centered_abs_fq(x - y);
            let nmul = centered_abs_fq(x * y);

            assert!(
                nadd <= nx + ny,
                "add triangle failed: |x+y|={nadd} > |x|+|y|={} for x={}, y={}",
                nx + ny,
                x.as_canonical_u64(),
                y.as_canonical_u64()
            );
            assert!(
                nsub <= nx + ny,
                "sub triangle failed: |x-y|={nsub} > |x|+|y|={} for x={}, y={}",
                nx + ny,
                x.as_canonical_u64(),
                y.as_canonical_u64()
            );
            assert!(
                nmul <= nx * ny,
                "mul bound failed: |x*y|={nmul} > |x|*|y|={} for x={}, y={}",
                nx * ny,
                x.as_canonical_u64(),
                y.as_canonical_u64()
            );
            assert!(
                nmul <= 4,
                "challenge-product bound failed: |x*y|={nmul} > 4 for x={}, y={}",
                x.as_canonical_u64(),
                y.as_canonical_u64()
            );
        }
    }
}

#[test]
fn p5_challenge_operand_norms_are_small() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x505f_6d35_736d_616c);
    for _ in 0..128 {
        let a = sample_challenge_poly(&mut rng);
        let b = sample_challenge_poly(&mut rng);
        assert!(a.norm_inf() as u128 <= CHALLENGE_BOUND);
        assert!(b.norm_inf() as u128 <= CHALLENGE_BOUND);
    }
}

#[test]
fn p5_ring_mul_bound_holds_for_challenge_operands() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x505f_6d35_6d75_6c31);
    for _ in 0..256 {
        let a = sample_challenge_poly(&mut rng);
        let b = sample_challenge_poly(&mut rng);
        let c = a.mul(&b);
        let nc = c.norm_inf() as u128;

        assert!(
            nc <= P5_RING_MUL_BOUND,
            "P5 ring bound failed: ||a*b||_inf={nc} > {} (D={}, ||a||<=2, ||b||<=2)",
            P5_RING_MUL_BOUND,
            D
        );
    }
}

#[test]
fn p5_ring_mul_bound_holds_for_extreme_challenge_patterns() {
    let mut plus = [Fq::ZERO; D];
    let mut minus = [Fq::ZERO; D];
    plus.fill(fq_from_i64(2));
    minus.fill(fq_from_i64(-2));

    let cases = [Rq(plus), Rq(minus)];
    for a in cases {
        for b in cases {
            let c = a.mul(&b);
            assert!(
                (c.norm_inf() as u128) <= P5_RING_MUL_BOUND,
                "P5 ring bound failed on extreme pattern"
            );
        }
    }
}
