use ff::Field;
use neo_params::poseidon2_goldilocks::{SEED, WIDTH};
use wip_spartan::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  provider::{
    GoldilocksWhirEngine as E,
    pcs::whir_pc::{
      WHIR_EXTENSION_DEGREE, WHIR_POW_BITS, WHIR_SECURITY_LEVEL, WhirBlind, WhirCommitment,
      WhirCommitmentKey, WhirEvaluationArgument, WhirPcsP3, WhirVerifierKey,
      validate_whir_configuration,
    },
  },
  traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
};

type F = <E as Engine>::Scalar;
type PCS = WhirPcsP3<E>;

fn polynomial(num_variables: usize) -> Vec<F> {
  (0..1usize << num_variables)
    .map(|i| F::from((i * i + 3 * i + 11) as u64))
    .collect()
}

fn point(num_variables: usize) -> Vec<F> {
  (0..num_variables)
    .map(|i| F::from((7 * i + 5) as u64))
    .collect()
}

#[test]
fn whir_field_and_poseidon2_match_superneo_across_p3_versions() {
  use p3_field::{PrimeCharacteristicRing as _, PrimeField64 as _};
  use p3_field_whir::{PrimeCharacteristicRing as _, PrimeField64 as _};
  use p3_symmetric::Permutation as _;
  use p3_symmetric_whir::Permutation as _;
  use rand_chacha_p3::{ChaCha8Rng, rand_core::SeedableRng};

  type SuperNeoBase = p3_goldilocks::Goldilocks;
  type SuperNeoExtension = p3_field::extension::BinomialExtensionField<SuperNeoBase, 2>;
  type WhirBase = p3_goldilocks_whir::Goldilocks;
  type WhirExtension = p3_field_whir::extension::BinomialExtensionField<WhirBase, 2>;

  let superneo_a = SuperNeoExtension::new([SuperNeoBase::from_u64(3), SuperNeoBase::from_u64(5)]);
  let superneo_b = SuperNeoExtension::new([SuperNeoBase::from_u64(7), SuperNeoBase::from_u64(11)]);
  let whir_a = WhirExtension::new([WhirBase::from_u64(3), WhirBase::from_u64(5)]);
  let whir_b = WhirExtension::new([WhirBase::from_u64(7), WhirBase::from_u64(11)]);
  let superneo_product = superneo_a * superneo_b;
  let whir_product = whir_a * whir_b;
  let superneo_coefficients: &[SuperNeoBase] =
    p3_field::BasedVectorSpace::as_basis_coefficients_slice(&superneo_product);
  let whir_coefficients: &[WhirBase] =
    p3_field_whir::BasedVectorSpace::as_basis_coefficients_slice(&whir_product);
  assert_eq!(
    superneo_coefficients
      .iter()
      .map(|value| value.as_canonical_u64())
      .collect::<Vec<_>>(),
    whir_coefficients
      .iter()
      .map(|value| value.as_canonical_u64())
      .collect::<Vec<_>>()
  );

  let mut superneo_rng = ChaCha8Rng::from_seed(SEED);
  let mut whir_rng = ChaCha8Rng::from_seed(SEED);
  let superneo_permutation =
    p3_goldilocks::Poseidon2Goldilocks::<WIDTH>::new_from_rng_128(&mut superneo_rng);
  let whir_permutation =
    p3_goldilocks_whir::Poseidon2Goldilocks::<WIDTH>::new_from_rng_128(&mut whir_rng);
  let superneo_state = core::array::from_fn(|i| SuperNeoBase::from_u64((i * i + 1) as u64));
  let whir_state = core::array::from_fn(|i| WhirBase::from_u64((i * i + 1) as u64));
  let superneo_output = superneo_permutation.permute(superneo_state);
  let whir_output = whir_permutation.permute(whir_state);
  assert_eq!(
    superneo_output.map(|value| value.as_canonical_u64()),
    whir_output.map(|value| value.as_canonical_u64())
  );
}

#[test]
fn whir_roundtrip_uses_superneo_profile_and_survives_serialization() {
  assert_eq!(WHIR_SECURITY_LEVEL, 125);
  assert_eq!(WHIR_POW_BITS, 18);
  assert_eq!(WHIR_EXTENSION_DEGREE, 3);

  let num_variables = 4;
  let poly = polynomial(num_variables);
  let point = point(num_variables);
  let (ck, vk) = PCS::setup(b"whir-roundtrip", poly.len());
  assert_eq!(ck.num_variables(), num_variables);
  assert_eq!(vk.num_variables(), num_variables);

  let blind = PCS::blind(&ck, poly.len());
  let commitment = PCS::commit(&ck, &poly, &blind, false).unwrap();

  let ck: WhirCommitmentKey = bincode::deserialize(&bincode::serialize(&ck).unwrap()).unwrap();
  let vk: WhirVerifierKey = bincode::deserialize(&bincode::serialize(&vk).unwrap()).unwrap();
  let commitment: WhirCommitment =
    bincode::deserialize(&bincode::serialize(&commitment).unwrap()).unwrap();
  let blind: WhirBlind = bincode::deserialize(&bincode::serialize(&blind).unwrap()).unwrap();

  let mut prover_transcript = <E as Engine>::TE::new(b"whir-roundtrip");
  let (eval, argument) = PCS::prove(
    &ck,
    &mut prover_transcript,
    &commitment,
    &poly,
    &blind,
    &point,
  )
  .unwrap();
  assert_eq!(eval, MultilinearPolynomial::new(poly).evaluate(&point));

  let argument: WhirEvaluationArgument =
    bincode::deserialize(&bincode::serialize(&argument).unwrap()).unwrap();
  let mut verifier_transcript = <E as Engine>::TE::new(b"whir-roundtrip");
  PCS::verify(
    &vk,
    &mut verifier_transcript,
    &commitment,
    &point,
    &eval,
    &argument,
  )
  .unwrap();
}

#[test]
fn whir_configures_the_stage1_pilot_within_the_profile_budget() {
  validate_whir_configuration(24).unwrap();
}

#[test]
fn whir_roundtrip_exercises_an_intermediate_stir_round() {
  let num_variables = 12;
  let poly = polynomial(num_variables);
  let point = point(num_variables);
  let (ck, vk) = PCS::setup(b"whir-intermediate-round", poly.len());
  let blind = PCS::blind(&ck, poly.len());
  let commitment = PCS::commit(&ck, &poly, &blind, false).unwrap();
  let mut prover_transcript = <E as Engine>::TE::new(b"whir-intermediate-round");
  let (eval, argument) = PCS::prove(
    &ck,
    &mut prover_transcript,
    &commitment,
    &poly,
    &blind,
    &point,
  )
  .unwrap();
  assert!(argument.intermediate_rounds() > 0);

  let mut verifier_transcript = <E as Engine>::TE::new(b"whir-intermediate-round");
  PCS::verify(
    &vk,
    &mut verifier_transcript,
    &commitment,
    &point,
    &eval,
    &argument,
  )
  .unwrap();
}

#[test]
fn whir_rejects_wrong_claims_and_sparse_polynomial_corruption() {
  let num_variables = 3;
  let poly = polynomial(num_variables);
  let point = point(num_variables);
  let (ck, vk) = PCS::setup(b"whir-redteam", poly.len());
  let blind = PCS::blind(&ck, poly.len());
  let commitment = PCS::commit(&ck, &poly, &blind, false).unwrap();
  let mut prover_transcript = <E as Engine>::TE::new(b"whir-redteam");
  let (eval, argument) = PCS::prove(
    &ck,
    &mut prover_transcript,
    &commitment,
    &poly,
    &blind,
    &point,
  )
  .unwrap();

  let mut wrong_eval_transcript = <E as Engine>::TE::new(b"whir-redteam");
  assert!(
    PCS::verify(
      &vk,
      &mut wrong_eval_transcript,
      &commitment,
      &point,
      &(eval + F::ONE),
      &argument,
    )
    .is_err()
  );

  let mut wrong_point = point.clone();
  wrong_point[0] += F::ONE;
  let mut wrong_point_transcript = <E as Engine>::TE::new(b"whir-redteam");
  assert!(
    PCS::verify(
      &vk,
      &mut wrong_point_transcript,
      &commitment,
      &wrong_point,
      &eval,
      &argument,
    )
    .is_err()
  );

  let mut other_poly = poly.clone();
  other_poly[1] += F::ONE;
  let other_blind = PCS::blind(&ck, other_poly.len());
  let other_commitment = PCS::commit(&ck, &other_poly, &other_blind, false).unwrap();
  let mut wrong_commitment_transcript = <E as Engine>::TE::new(b"whir-redteam");
  assert!(
    PCS::verify(
      &vk,
      &mut wrong_commitment_transcript,
      &other_commitment,
      &point,
      &eval,
      &argument,
    )
    .is_err()
  );

  let empty_cache_blind: WhirBlind =
    bincode::deserialize(&bincode::serialize(&blind).unwrap()).unwrap();
  let mut corrupted_poly = poly;
  corrupted_poly[5] += F::ONE;
  let mut corruption_transcript = <E as Engine>::TE::new(b"whir-redteam");
  assert_eq!(
    PCS::prove(
      &ck,
      &mut corruption_transcript,
      &commitment,
      &corrupted_poly,
      &empty_cache_blind,
      &point,
    )
    .unwrap_err(),
    SpartanError::InvalidPCS
  );
}

#[test]
fn whir_explicitly_rejects_multiple_partial_commitments() {
  let poly = polynomial(2);
  let (ck, _vk) = PCS::setup(b"whir-partials", poly.len());
  let blind = PCS::blind(&ck, poly.len());
  let (commitment, blind) = PCS::commit_partial(&ck, &poly, &blind, false).unwrap();

  assert!(PCS::combine_partial(&[commitment.clone(), commitment]).is_err());
  assert!(PCS::combine_blinds(&[blind.clone(), blind]).is_err());
}
