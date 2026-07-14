//! Host-side parity of `kernels::goldilocks` against `neo_math` (no GPU).

use neo_math::{from_complex, KExtensions, F, K};
use neo_prover_cuda::kernels::goldilocks::{Gl, Kx, GOLDILOCKS_MODULUS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{Rng, SeedableRng};

fn k_from_words(c0: u64, c1: u64) -> K {
    from_complex(F::from_u64(c0), F::from_u64(c1))
}

#[test]
fn gl_ops_match_neo_math() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for _ in 0..10_000 {
        let (a, b) = (
            rng.random::<u64>() % GOLDILOCKS_MODULUS,
            rng.random::<u64>() % GOLDILOCKS_MODULUS,
        );
        let (fa, fb) = (F::from_u64(a), F::from_u64(b));
        let (ga, gb) = (Gl::from_u64(a), Gl::from_u64(b));
        assert_eq!((ga + gb).as_canonical_u64(), (fa + fb).as_canonical_u64());
        assert_eq!((ga - gb).as_canonical_u64(), (fa - fb).as_canonical_u64());
        assert_eq!((ga * gb).as_canonical_u64(), (fa * fb).as_canonical_u64());
        assert_eq!((-ga).as_canonical_u64(), (-fa).as_canonical_u64());
    }
}

#[test]
fn gl_handles_noncanonical_and_boundary_inputs() {
    for a in [0, 1, GOLDILOCKS_MODULUS - 1, GOLDILOCKS_MODULUS, u64::MAX] {
        let expected = a % GOLDILOCKS_MODULUS;
        assert_eq!(Gl::from_u64(a).as_canonical_u64(), expected);
    }
}

#[test]
fn kx_ops_match_neo_math() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    for _ in 0..10_000 {
        let w: [u64; 4] = std::array::from_fn(|_| rng.random::<u64>() % GOLDILOCKS_MODULUS);
        let (ka, kb) = (k_from_words(w[0], w[1]), k_from_words(w[2], w[3]));
        let (ga, gb) = (Kx::from_words(w[0], w[1]), Kx::from_words(w[2], w[3]));
        let mul = (ka * kb).to_limbs_u64();
        assert_eq!((ga * gb).as_words(), [mul.0, mul.1]);
        let add = (ka + kb).to_limbs_u64();
        assert_eq!((ga + gb).as_words(), [add.0, add.1]);
        let sub = (ka - kb).to_limbs_u64();
        assert_eq!((ga - gb).as_words(), [sub.0, sub.1]);
        let neg = (-ka).to_limbs_u64();
        assert_eq!((-ga).as_words(), [neg.0, neg.1]);
        let scaled = ka.scale_base(F::from_u64(w[2])).to_limbs_u64();
        assert_eq!(ga.scale_base(Gl::from_u64(w[2])).as_words(), [scaled.0, scaled.1]);
    }
}
