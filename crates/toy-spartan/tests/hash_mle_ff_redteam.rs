//! Comprehensive red-team tests for FF/Keccak Hash-MLE PCS implementation
//! Tests both valid functionality and adversarial tampering scenarios

use ff::Field;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use toy_spartan::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  provider::{
    PallasMerkleMleEngine as E, keccak::Keccak256Transcript, pcs::merkle_mle_pc::HashMlePCS,
  },
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
};

type F = <E as Engine>::Scalar;
type PCS = HashMlePCS<E>;

fn rand_poly(m: usize, seed: u64) -> Vec<F> {
  let n = 1usize << m;
  let mut rng = ChaCha8Rng::seed_from_u64(seed);
  (0..n).map(|_| F::from(rng.random::<u64>())).collect()
}

fn rand_point(m: usize, seed: u64) -> Vec<F> {
  let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x55AA_77CC);
  // general field points (not just bits)
  (0..m).map(|_| F::from(rng.random::<u64>())).collect()
}

fn prove_once(
  m: usize,
  seed: u64,
) -> (
  toy_spartan::provider::pcs::merkle_mle_pc::HashMleCommitment<E>,
  toy_spartan::provider::pcs::merkle_mle_pc::HashMleEvaluationArgument<E>,
  F,
  Vec<F>,
) {
  let poly = rand_poly(m, seed);
  let point = rand_point(m, seed);

  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  // compute expected eval independently
  let mle = MultilinearPolynomial::new(poly.clone());
  let expected = mle.evaluate(&point);

  // prove and verify
  let mut tp = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval, arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();
  assert_eq!(eval, expected);

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap();

  (comm, arg, eval, point)
}

#[test]
fn ff_valid_random_small_dims() {
  for m in 0..=6 {
    let _ = prove_once(m, 1000 + m as u64);
  }
}

#[test]
fn ff_non_power_of_two_vector_rejected() {
  let (ck, _vk) = PCS::setup(b"ff-keccak", 0);
  let v = vec![F::from(1u64); 14]; // not power of two
  let r = PCS::blind(&ck, v.len());
  let err = PCS::commit(&ck, &v, &r, false).unwrap_err();
  match err {
    SpartanError::InvalidInputLength { .. } => {}
    e => panic!("unexpected error: {:?}", e),
  }
}

#[test]
fn ff_wrong_poly_len_in_prove_rejected() {
  let m = 5usize;
  let (ck, _vk) = PCS::setup(b"ff-keccak", 1 << m);
  let mut v = rand_poly(m, 7);
  v.pop(); // break len = 2^m - 1
  let r = PCS::blind(&ck, v.len());

  // Create a properly sized vector for commitment
  let mut v_padded = v.clone();
  v_padded.resize(v.len().next_power_of_two(), F::ZERO);
  let comm = PCS::commit(&ck, &v_padded, &r, false).unwrap();
  let point = rand_point(m, 7);

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::prove(&ck, &mut t, &comm, &v, &r, &point).unwrap_err();
  match err {
    SpartanError::InvalidInputLength { .. } => {}
    e => panic!("unexpected error: {:?}", e),
  }
}

// ---------------- Red-team (tampering) ----------------

#[test]
fn ff_tamper_layer_root_fails() {
  let m = 6usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let blind = PCS::blind(&ck, 1 << m);
  let poly = rand_poly(m, 22);
  let point = rand_point(m, 22);
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut tp = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval, mut arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();

  // tamper first root (base layer)
  arg.layer_roots[0].0[0] ^= 0x01;

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

#[test]
fn ff_swap_paths_a_b_fails() {
  let m = 5usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let poly = rand_poly(m, 33);
  let point = rand_point(m, 33);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval, mut arg) = PCS::prove(&ck, &mut t, &comm, &poly, &blind, &point).unwrap();

  // swap paths only (indices/depths will mismatch)
  let r0 = &mut arg.rounds[0];
  std::mem::swap(&mut r0.path_a, &mut r0.path_b);

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

#[test]
fn ff_tamper_next_fold_value_fails() {
  let m = 5usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let poly = rand_poly(m, 41);
  let point = rand_point(m, 41);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval, mut arg) = PCS::prove(&ck, &mut t, &comm, &poly, &blind, &point).unwrap();

  arg.rounds[0].next = arg.rounds[0].a; // wrong folded value

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

#[test]
fn ff_tamper_eval_fails() {
  let m = 4usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let poly = rand_poly(m, 51);
  let point = rand_point(m, 51);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (mut eval, arg) = PCS::prove(&ck, &mut t, &comm, &poly, &blind, &point).unwrap();

  // change the claimed eval but keep correct paths
  eval = eval + F::ONE;

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

#[test]
fn ff_mix_commit_and_proof_from_different_polys_fails() {
  let m = 5usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);

  let poly1 = rand_poly(m, 61);
  let poly2 = rand_poly(m, 62);
  let point = rand_point(m, 61);

  let r1 = PCS::blind(&ck, poly1.len());
  let c1 = PCS::commit(&ck, &poly1, &r1, false).unwrap();

  let r2 = PCS::blind(&ck, poly2.len());
  let c2 = PCS::commit(&ck, &poly2, &r2, false).unwrap();

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval2, arg2) = PCS::prove(&ck, &mut t, &c2, &poly2, &r2, &point).unwrap();

  // verify proof from poly2 against commitment to poly1
  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &c1, &point, &eval2, &arg2).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

#[test]
fn ff_depth_guard_fails_when_path_truncated() {
  let m = 5usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let poly = rand_poly(m, 71);
  let point = rand_point(m, 71);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval, mut arg) = PCS::prove(&ck, &mut t, &comm, &poly, &blind, &point).unwrap();

  // remove one sibling from next-path depth at round 0
  arg.rounds[0].path_next.siblings.pop();

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

#[test]
fn ff_index_guard_fails_if_a_index_not_zero() {
  let m = 4usize;
  let (ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
  let poly = rand_poly(m, 81);
  let point = rand_point(m, 81);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut t = Keccak256Transcript::<E>::new(b"ff-keccak");
  let (eval, mut arg) = PCS::prove(&ck, &mut t, &comm, &poly, &blind, &point).unwrap();

  arg.rounds[0].path_a.leaf_index = 1; // should be 0

  let mut tv = Keccak256Transcript::<E>::new(b"ff-keccak");
  let err = PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap_err();
  assert!(matches!(err, SpartanError::InvalidPCS));
}

// ---------------- Proptest: randomized dimensions/points ----------------

use proptest::prelude::*;

proptest! {
    #[test]
    fn ff_prop_randomized(m in 1usize..=6, seed in 0u64..10_000) {
        let (comm, arg, eval, point) = prove_once(m, seed);

        // Shuffle a copy of layer_roots as a negative test
        let mut arg_bad = arg.clone();
        if arg_bad.layer_roots.len() > 2 {
            arg_bad.layer_roots.swap(1, 2);
        } else if arg_bad.layer_roots.len() == 2 {
            // For small cases, swap the two available roots
            arg_bad.layer_roots.swap(0, 1);
        } else {
            // If there's only one root, we can't meaningfully tamper, so skip this test case
            return Ok(());
        }

        let (_ck, vk) = PCS::setup(b"ff-keccak", 1 << m);
        let mut tv1 = Keccak256Transcript::<E>::new(b"ff-keccak");
        PCS::verify(&vk, &mut tv1, &comm, &point, &eval, &arg).unwrap();

        let mut tv2 = Keccak256Transcript::<E>::new(b"ff-keccak");
        prop_assert!(PCS::verify(&vk, &mut tv2, &comm, &point, &eval, &arg_bad).is_err());
    }
}
