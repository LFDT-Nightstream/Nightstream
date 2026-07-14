//! Comprehensive red-team tests for p3/Goldilocks Hash-MLE PCS implementation
//! Tests both valid functionality and adversarial tampering scenarios

#![cfg(feature = "p3_backend")]

use ff::Field;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use toy_spartan::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  provider::{
    GoldilocksMerkleMleEngine as KeccakEngine, GoldilocksP3MerkleMleEngine as E,
    keccak::Keccak256Transcript,
    pcs::{merkle_mle_pc::HashMlePCS as KeccakPCS, merkle_mle_pc_p3::HashMlePcsP3},
  },
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
};

type F = <E as Engine>::Scalar;
type P3Transcript = <E as Engine>::TE;
type PCS = HashMlePcsP3<E>;

fn rand_poly(m: usize, seed: u64) -> Vec<F> {
  let n = 1usize << m;
  let mut rng = ChaCha8Rng::seed_from_u64(seed);
  (0..n).map(|_| F::from(rng.random::<u64>())).collect()
}

fn rand_point(m: usize, seed: u64) -> Vec<F> {
  let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xDEAD_BEEF);
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

  let (ck, vk) = PCS::setup(b"p3-poseidon2", 1 << m);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  // compute expected eval independently
  let mle = MultilinearPolynomial::new(poly.clone());
  let expected = mle.evaluate(&point);

  // prove and verify
  let mut tp = P3Transcript::new(b"p3-poseidon2");
  let (eval, arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();
  assert_eq!(eval, expected);

  let mut tv = P3Transcript::new(b"p3-poseidon2");
  PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).unwrap();

  (comm, arg, eval, point)
}

#[test]
fn p3_valid_random() {
  for m in 0..=8 {
    let _ = prove_once(m, 123 + m as u64);
  }
}

#[test]
fn p3_tamper_root_fails() {
  let m = 6usize;
  let (ck, vk) = PCS::setup(b"p3-poseidon2", 1 << m);
  let poly = rand_poly(m, 11);
  let point = rand_point(m, 11);
  let blind = PCS::blind(&ck, poly.len());

  let mut comm = PCS::commit(&ck, &poly, &blind, false).unwrap();
  let mut tp = P3Transcript::new(b"p3-poseidon2");
  let (eval, mut arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();

  // tamper the base root
  comm.base_root.0[0] ^= 1;
  let mut tv1 = P3Transcript::new(b"p3-poseidon2");
  assert!(PCS::verify(&vk, &mut tv1, &comm, &point, &eval, &arg).is_err());

  // restore commitment, tamper layer roots
  let comm2 = PCS::commit(&ck, &poly, &blind, false).unwrap();
  arg.layer_roots[0].0[0] ^= 1;
  let mut tv2 = P3Transcript::new(b"p3-poseidon2");
  assert!(PCS::verify(&vk, &mut tv2, &comm2, &point, &eval, &arg).is_err());
}

#[test]
fn p3_tamper_paths_and_next_fails() {
  let m = 5usize;
  let (ck, vk) = PCS::setup(b"p3-poseidon2", 1 << m);
  let poly = rand_poly(m, 88);
  let point = rand_point(m, 88);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut tp = P3Transcript::new(b"p3-poseidon2");
  let (eval, mut arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();

  // swap A/B paths in round 0
  if !arg.rounds.is_empty()
    && !arg.rounds[0].path_a.siblings.is_empty()
    && !arg.rounds[0].path_b.siblings.is_empty()
  {
    let len = arg.rounds[0].path_a.siblings.len();
    arg.rounds[0].path_a.siblings.swap(0, len - 1);
    let mut tv1 = P3Transcript::new(b"p3-poseidon2");
    assert!(PCS::verify(&vk, &mut tv1, &comm, &point, &eval, &arg).is_err());
  }

  // revert and set wrong next
  let mut tp2 = P3Transcript::new(b"p3-poseidon2");
  let (_eval2, mut arg2) = PCS::prove(&ck, &mut tp2, &comm, &poly, &blind, &point).unwrap();
  if !arg2.rounds.is_empty() {
    arg2.rounds[0].next = arg2.rounds[0].a;
    let mut tv2 = P3Transcript::new(b"p3-poseidon2");
    assert!(PCS::verify(&vk, &mut tv2, &comm, &point, &eval, &arg2).is_err());
  }
}

#[test]
fn p3_eval_mismatch_fails() {
  let m = 4usize;
  let (ck, vk) = PCS::setup(b"p3-poseidon2", 1 << m);
  let poly = rand_poly(m, 99);
  let point = rand_point(m, 99);
  let blind = PCS::blind(&ck, poly.len());
  let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let mut tp = P3Transcript::new(b"p3-poseidon2");
  let (mut eval, arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();
  eval = eval + F::ONE; // wrong

  let mut tv = P3Transcript::new(b"p3-poseidon2");
  assert!(PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).is_err());
}

#[test]
fn p3_compare_keccak_vs_poseidon2() {
  let m = 3usize;
  let n = 1usize << m;
  let poly = (0..n).map(|i| F::from((i + 1) as u64)).collect::<Vec<_>>();
  let point = (0..m).map(|i| F::from(i as u64)).collect::<Vec<_>>();

  // Test with Keccak backend
  let (ck_k, vk_k) = KeccakPCS::<KeccakEngine>::setup(b"compare", n);
  let blind_k = KeccakPCS::<KeccakEngine>::blind(&ck_k, n);
  let com_k = KeccakPCS::<KeccakEngine>::commit(&ck_k, &poly, &blind_k, false).unwrap();
  let mut tr_k = Keccak256Transcript::<KeccakEngine>::new(b"compare");
  let (eval_k, arg_k) =
    KeccakPCS::<KeccakEngine>::prove(&ck_k, &mut tr_k, &com_k, &poly, &blind_k, &point).unwrap();

  // Test with Poseidon2 backend
  let (ck_p, vk_p) = PCS::setup(b"compare", n);
  let blind_p = PCS::blind(&ck_p, n);
  let com_p = PCS::commit(&ck_p, &poly, &blind_p, false).unwrap();
  let mut tr_p = P3Transcript::new(b"compare");
  let (eval_p, arg_p) = PCS::prove(&ck_p, &mut tr_p, &com_p, &poly, &blind_p, &point).unwrap();

  // Evaluations should be the same (same polynomial, same point)
  assert_eq!(eval_k, eval_p);

  // Commitments will be different (different hash functions)
  assert_ne!(com_k.base_root.0, com_p.base_root.0);

  // Both should verify
  let mut tr_k_v = Keccak256Transcript::<KeccakEngine>::new(b"compare");
  KeccakPCS::<KeccakEngine>::verify(&vk_k, &mut tr_k_v, &com_k, &point, &eval_k, &arg_k).unwrap();

  let mut tr_p_v = P3Transcript::new(b"compare");
  PCS::verify(&vk_p, &mut tr_p_v, &com_p, &point, &eval_p, &arg_p).unwrap();
}

#[test]
fn p3_non_power_of_two_vector_rejected() {
  let (ck, _vk) = PCS::setup(b"p3-poseidon2", 0);
  let v = vec![F::from(1u64); 14]; // not power of two
  let r = PCS::blind(&ck, v.len());
  let err = PCS::commit(&ck, &v, &r, false).unwrap_err();
  match err {
    SpartanError::InvalidInputLength { .. } => {}
    e => panic!("unexpected error: {:?}", e),
  }
}

#[test]
fn p3_commitment_consistency() {
  let m = 4usize;
  let n = 1usize << m;
  let poly = (0..n)
    .map(|i| F::from((i * i + 7) as u64))
    .collect::<Vec<_>>();

  let (ck, _vk) = PCS::setup(b"test_consistency_p3", n);
  let blind = PCS::blind(&ck, n);

  // Commit to the same polynomial twice
  let com1 = PCS::commit(&ck, &poly, &blind, false).unwrap();
  let com2 = PCS::commit(&ck, &poly, &blind, false).unwrap();

  // Commitments should be identical
  assert_eq!(com1, com2);
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn p3_prop_fuzz(m in 1usize..=7, seed in 0u64..10_000) {
        let poly = rand_poly(m, seed);
        let point = rand_point(m, seed ^ 0xA5A5);

        let (ck, vk) = PCS::setup(b"p3-fuzz", 1 << m);
        let blind = PCS::blind(&ck, poly.len());
        let comm = PCS::commit(&ck, &poly, &blind, false).unwrap();

        let mut tp = P3Transcript::new(b"p3-fuzz");
        let (eval, mut arg) = PCS::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();

        let mut tv = P3Transcript::new(b"p3-fuzz");
        prop_assert!(PCS::verify(&vk, &mut tv, &comm, &point, &eval, &arg).is_ok());

        // negative by truncating next-path depth (round 0)
        if !arg.rounds.is_empty() && !arg.rounds[0].path_next.siblings.is_empty() {
            arg.rounds[0].path_next.siblings.pop();
            let mut tv2 = P3Transcript::new(b"p3-fuzz");
            prop_assert!(PCS::verify(&vk, &mut tv2, &comm, &point, &eval, &arg).is_err());
        }
    }
}

#[test]
#[cfg(feature = "p3_backend")]
fn p3_tamper_samples_fail() {
  use toy_spartan::{
    provider::{GoldilocksP3MerkleMleEngine as E, pcs::merkle_mle_pc_p3::HashMlePcsP3},
    traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
  };
  type F = <E as Engine>::Scalar;
  type P3Transcript = <E as Engine>::TE;
  let m = 5usize;
  let n = 1usize << m;
  let poly: Vec<F> = (0..n).map(|i| F::from(i as u64)).collect();
  let point: Vec<F> = (0..m).map(|_| F::from(1u64)).collect(); // All 1s to ensure tampering is detected

  let (ck, vk) = HashMlePcsP3::<E>::setup(b"p3-sample-tamper", n);
  let blind = HashMlePcsP3::<E>::blind(&ck, n);
  let comm = HashMlePcsP3::<E>::commit(&ck, &poly, &blind, false).unwrap();

  let mut tp = P3Transcript::new(b"p3-sample-tamper");
  let (eval, mut arg) =
    HashMlePcsP3::<E>::prove(&ck, &mut tp, &comm, &poly, &blind, &point).unwrap();

  // tamper one sample's next value
  arg.samples[0][0].next = arg.samples[0][0].a;
  let mut tv = P3Transcript::new(b"p3-sample-tamper");
  assert!(HashMlePcsP3::<E>::verify(&vk, &mut tv, &comm, &point, &eval, &arg).is_err());
}
