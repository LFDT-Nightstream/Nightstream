//! Standalone WHIR PCS for Toy Spartan's Goldilocks engine.
//!
//! This module owns the conversion between Toy Spartan's `ff` Goldilocks
//! values and Plonky3's quadratic-extension WHIR implementation. It uses the
//! canonical SuperNeo Poseidon2 parameters and opens only caller-prescribed
//! multilinear points. It does not own or alter the SuperNeo folding lifecycle.
//!
//! The commitment is binding but not hiding. Consequently, selecting this PCS
//! does not by itself make the surrounding Toy Spartan proof zero knowledge.
//! The 125-bit target applies to this PCS; Toy Spartan's outer sum-check still
//! samples base-field challenges and has its own soundness bound.

use crate::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  provider::goldi::F as SpartanGoldilocks,
  traits::{
    Engine, Group,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use neo_params::{
  goldilocks_paper_b2::{EXTENSION_DEGREE, KAPPA, LAMBDA},
  poseidon2_goldilocks::{DIGEST_LEN, RATE, SEED, WIDTH},
};
use p3_challenger_whir::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit_whir::MultilinearPcs;
use p3_dft_whir::Radix2DFTSmallBatch;
use p3_field_whir::{Field as P3Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks_whir::{Goldilocks as WhirBase, Poseidon2Goldilocks};
use p3_matrix_whir::dense::RowMajorMatrix;
use p3_merkle_tree_whir::MerkleTreeMmcs;
use p3_multilinear_util_whir::point::Point;
use p3_sumcheck_whir::{
  OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec,
  layout::{Layout, SuffixProver, Table, Witness},
};
use p3_symmetric_whir::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};
use p3_whir::{
  DomainSeparator, FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver,
};
use rand_chacha_p3::{ChaCha8Rng, rand_core::SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
  fmt::{Debug, Formatter},
  marker::PhantomData,
  sync::{Arc, Mutex},
};

type WhirExtension = p3_field_whir::extension::BinomialExtensionField<WhirBase, 2>;
type WhirPermutation = Poseidon2Goldilocks<WIDTH>;
type WhirHash = PaddingFreeSponge<WhirPermutation, WIDTH, RATE, DIGEST_LEN>;
type WhirCompress = TruncatedPermutation<WhirPermutation, 2, DIGEST_LEN, WIDTH>;
type WhirChallenger = DuplexChallenger<WhirBase, WhirPermutation, WIDTH, RATE>;
type PackedWhirBase = <WhirBase as P3Field>::Packing;
type WhirMmcs =
  MerkleTreeMmcs<PackedWhirBase, PackedWhirBase, WhirHash, WhirCompress, 2, DIGEST_LEN>;
type WhirDft = Radix2DFTSmallBatch<WhirBase>;
type WhirLayout = SuffixProver<WhirBase, WhirExtension>;
type WhirRuntime =
  WhirProver<WhirExtension, WhirBase, WhirDft, WhirMmcs, WhirChallenger, WhirLayout>;
type RawCommitment = MerkleCap<WhirBase, [WhirBase; DIGEST_LEN]>;
type RawProverData = p3_whir::WhirProverData<WhirBase, WhirExtension, WhirMmcs, WhirLayout>;
type RawProof = p3_whir::PcsProof<WhirBase, WhirExtension, WhirMmcs>;

const OUTER_DOMAIN: &[u8] = b"toy-spartan/whir-pcs/v1";
const INTERNAL_DOMAIN: &[u8] = b"toy-spartan/whir-pcs/internal/v1";

const _: () = assert!(EXTENSION_DEGREE == 2);
const _: () = assert!(WIDTH == RATE + neo_params::poseidon2_goldilocks::CAPACITY);

/// Target soundness of the standalone WHIR PCS, inherited from SuperNeo's profile.
pub const WHIR_SECURITY_LEVEL: usize = LAMBDA as usize;

/// Maximum grinding budget of the standalone WHIR PCS, inherited from SuperNeo's profile.
pub const WHIR_POW_BITS: usize = KAPPA as usize;

/// Extension degree used for WHIR challenges and sum-checks.
pub const WHIR_EXTENSION_DEGREE: usize = EXTENSION_DEGREE as usize;

/// Serializable setup material for WHIR commitments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirCommitmentKey {
  num_evaluations: usize,
  label: Vec<u8>,
}

impl WhirCommitmentKey {
  /// Number of Boolean-hypercube evaluations accepted by this key.
  pub const fn num_evaluations(&self) -> usize {
    self.num_evaluations
  }

  /// Number of multilinear variables accepted by this key.
  pub fn num_variables(&self) -> usize {
    self.num_evaluations.ilog2() as usize
  }
}

/// Serializable verifier setup material for WHIR openings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifierKey {
  num_evaluations: usize,
  label: Vec<u8>,
}

impl WhirVerifierKey {
  /// Number of Boolean-hypercube evaluations accepted by this key.
  pub const fn num_evaluations(&self) -> usize {
    self.num_evaluations
  }

  /// Number of multilinear variables accepted by this key.
  pub fn num_variables(&self) -> usize {
    self.num_evaluations.ilog2() as usize
  }
}

/// A Poseidon2 Merkle-cap commitment to one multilinear evaluation table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirCommitment {
  num_evaluations: usize,
  root: RawCommitment,
}

impl WhirCommitment {
  /// Number of Boolean-hypercube evaluations committed by this root.
  pub const fn num_evaluations(&self) -> usize {
    self.num_evaluations
  }
}

impl<G: Group> TranscriptReprTrait<G> for WhirCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(24 + DIGEST_LEN * 8);
    bytes.extend_from_slice(OUTER_DOMAIN);
    bytes.extend_from_slice(&(self.num_evaluations as u64).to_le_bytes());
    bytes.extend_from_slice(&(self.root.num_roots() as u64).to_le_bytes());
    for digest in self.root.roots() {
      for value in digest {
        bytes.extend_from_slice(&value.as_canonical_u64().to_le_bytes());
      }
    }
    bytes
  }
}

impl<E> CommitmentTrait<E> for WhirCommitment where E: Engine<Scalar = SpartanGoldilocks> {}

#[derive(Clone)]
struct CachedProverData {
  num_evaluations: usize,
  root: RawCommitment,
  data: RawProverData,
}

/// Non-hiding WHIR blind plus an optional prover-only commitment cache.
///
/// The cache is skipped by serialization. A deserialized blind remains valid;
/// proving simply reconstructs and checks the committed codeword once.
#[derive(Clone, Serialize, Deserialize)]
pub struct WhirBlind {
  #[serde(skip)]
  cache: Arc<Mutex<Option<CachedProverData>>>,
}

impl Default for WhirBlind {
  fn default() -> Self {
    Self {
      cache: Arc::new(Mutex::new(None)),
    }
  }
}

impl Debug for WhirBlind {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    let cached = self
      .cache
      .lock()
      .map(|guard| guard.is_some())
      .unwrap_or(false);
    f.debug_struct("WhirBlind")
      .field("prover_data_cached", &cached)
      .finish()
  }
}

impl PartialEq for WhirBlind {
  fn eq(&self, _other: &Self) -> bool {
    true
  }
}

impl Eq for WhirBlind {}

/// A prescribed-point WHIR opening proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct WhirEvaluationArgument {
  proof: RawProof,
}

impl WhirEvaluationArgument {
  /// Number of intermediate STIR rounds carried by this proof.
  pub fn intermediate_rounds(&self) -> usize {
    self.proof.whir.rounds.len()
  }
}

impl Debug for WhirEvaluationArgument {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("WhirEvaluationArgument")
      .finish_non_exhaustive()
  }
}

/// WHIR polynomial commitment engine for Toy Spartan's Goldilocks scalar field.
#[derive(Clone, Debug)]
pub struct WhirPcsP3<E: Engine> {
  _engine: PhantomData<E>,
}

fn invalid_input(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: reason.into(),
  }
}

fn internal_error(reason: impl Into<String>) -> SpartanError {
  SpartanError::InternalError {
    reason: reason.into(),
  }
}

fn verification_error(reason: impl Into<String>) -> SpartanError {
  SpartanError::ProofVerifyError {
    reason: reason.into(),
  }
}

fn whir_permutation() -> WhirPermutation {
  let mut rng = ChaCha8Rng::from_seed(SEED);
  WhirPermutation::new_from_rng_128(&mut rng)
}

fn folding_factor(num_variables: usize) -> usize {
  num_variables.saturating_sub(1).clamp(1, 5)
}

fn build_runtime(num_variables: usize) -> Result<WhirRuntime, SpartanError> {
  if num_variables == 0 {
    return Err(invalid_input(
      "WHIR requires at least one multilinear variable",
    ));
  }

  let params = ProtocolParameters {
    starting_log_inv_rate: 1,
    round_log_inv_rates: Vec::new(),
    folding_factor: FoldingFactor::Constant(folding_factor(num_variables)),
    soundness_type: SecurityAssumption::UniqueDecoding,
    security_level: WHIR_SECURITY_LEVEL,
    pow_bits: WHIR_POW_BITS,
  };
  let config = WhirConfig::<WhirExtension, WhirBase, WhirChallenger>::new(num_variables, params)
    .map_err(|err| internal_error(format!("invalid WHIR configuration: {err}")))?;
  let max_fft_size = 1usize
    .checked_shl(config.max_fft_size() as u32)
    .ok_or_else(|| internal_error("WHIR FFT size does not fit usize"))?;

  let permutation = whir_permutation();
  let mmcs = WhirMmcs::new(
    WhirHash::new(permutation.clone()),
    WhirCompress::new(permutation),
    0,
  );
  Ok(WhirRuntime::new(config, WhirDft::new(max_fft_size), mmcs))
}

fn observe_bytes(challenger: &mut WhirChallenger, bytes: &[u8]) {
  challenger.observe(WhirBase::from_usize(bytes.len()));
  for chunk in bytes.chunks(7) {
    let mut limb = [0u8; 8];
    limb[..chunk.len()].copy_from_slice(chunk);
    challenger.observe(WhirBase::from_u64(u64::from_le_bytes(limb)));
  }
}

fn initial_challenger(runtime: &WhirRuntime, label: &[u8]) -> WhirChallenger {
  let mut challenger = WhirChallenger::new(whir_permutation());
  let mut separator = DomainSeparator::new(Vec::new());
  runtime.add_domain_separator::<DIGEST_LEN>(&mut separator);
  separator.observe_domain_separator(&mut challenger);
  observe_bytes(&mut challenger, INTERNAL_DOMAIN);
  observe_bytes(&mut challenger, label);
  challenger
}

fn to_whir_base(value: SpartanGoldilocks) -> WhirBase {
  WhirBase::from_u64(value.to_canonical_u64())
}

fn to_whir_extension(value: SpartanGoldilocks) -> WhirExtension {
  WhirExtension::from(to_whir_base(value))
}

fn to_whir_point(point: &[SpartanGoldilocks]) -> Point<WhirExtension> {
  Point::new(point.iter().copied().map(to_whir_extension).collect())
}

fn opening_protocol(num_variables: usize) -> OpeningProtocol {
  OpeningProtocol::new(vec![TableSpec::new(
    TableShape::new(num_variables, 1),
    vec![OpeningBatch::new(vec![0], Vec::new())],
  )])
}

fn witness(runtime: &WhirRuntime, values: &[SpartanGoldilocks]) -> Witness<WhirBase> {
  let evaluations = values.iter().copied().map(to_whir_base).collect();
  let table = Table::new(RowMajorMatrix::new(evaluations, values.len()));
  WhirLayout::new_witness(vec![table], runtime.round_folding_factor(0))
}

fn bind_outer_transcript<E>(
  transcript: &mut E::TE,
  commitment: &WhirCommitment,
  point: &[SpartanGoldilocks],
  eval: &SpartanGoldilocks,
) -> Result<[SpartanGoldilocks; DIGEST_LEN], SpartanError>
where
  E: Engine<Scalar = SpartanGoldilocks>,
{
  transcript.dom_sep(OUTER_DOMAIN);
  transcript.absorb(b"whir/commitment", commitment);
  transcript.absorb(b"whir/point", &point);
  transcript.absorb(b"whir/evaluation", eval);

  let anchors = (0..DIGEST_LEN)
    .map(|_| transcript.squeeze(b"whir/anchor"))
    .collect::<Result<Vec<_>, _>>()?;
  anchors
    .try_into()
    .map_err(|_| internal_error("invalid WHIR transcript anchor length"))
}

fn bind_internal_claim(
  challenger: &mut WhirChallenger,
  anchors: &[SpartanGoldilocks; DIGEST_LEN],
  point: &Point<WhirExtension>,
  eval: WhirExtension,
) {
  observe_bytes(challenger, b"outer-transcript-anchor");
  challenger.observe_slice(
    &anchors
      .iter()
      .copied()
      .map(to_whir_base)
      .collect::<Vec<_>>(),
  );
  observe_bytes(challenger, b"prescribed-point");
  challenger.observe(WhirBase::from_usize(point.num_variables()));
  challenger.observe_algebra_slice(point.as_slice());
  observe_bytes(challenger, b"claimed-evaluation");
  challenger.observe_algebra_element(eval);
}

fn cached_data(
  blind: &WhirBlind,
  commitment: &WhirCommitment,
) -> Result<Option<RawProverData>, SpartanError> {
  let cache = blind
    .cache
    .lock()
    .map_err(|_| internal_error("WHIR prover-data cache is poisoned"))?;
  Ok(cache.as_ref().and_then(|cached| {
    (cached.num_evaluations == commitment.num_evaluations && cached.root == commitment.root)
      .then(|| cached.data.clone())
  }))
}

fn store_cached_data(
  blind: &WhirBlind,
  commitment: &WhirCommitment,
  data: RawProverData,
) -> Result<(), SpartanError> {
  let mut cache = blind
    .cache
    .lock()
    .map_err(|_| internal_error("WHIR prover-data cache is poisoned"))?;
  *cache = Some(CachedProverData {
    num_evaluations: commitment.num_evaluations,
    root: commitment.root.clone(),
    data,
  });
  Ok(())
}

fn claimed_proof_evaluation(proof: &RawProof) -> Option<WhirExtension> {
  proof
    .evals
    .first()
    .and_then(|batch| batch.current().first())
    .copied()
}

impl<E> PCSEngineTrait<E> for WhirPcsP3<E>
where
  E: Engine<Scalar = SpartanGoldilocks>,
{
  type CommitmentKey = WhirCommitmentKey;
  type VerifierKey = WhirVerifierKey;
  type Commitment = WhirCommitment;
  type PartialCommitment = WhirCommitment;
  type Blind = WhirBlind;
  type EvaluationArgument = WhirEvaluationArgument;

  fn setup(label: &'static [u8], n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    assert!(
      n >= 2 && n.is_power_of_two(),
      "WHIR setup length must be a power of two and at least two"
    );
    let label = label.to_vec();
    (
      WhirCommitmentKey {
        num_evaluations: n,
        label: label.clone(),
      },
      WhirVerifierKey {
        num_evaluations: n,
        label,
      },
    )
  }

  fn width() -> usize {
    2
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    WhirBlind::default()
  }

  fn commit(
    ck: &Self::CommitmentKey,
    values: &[E::Scalar],
    blind: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    if values.len() != ck.num_evaluations {
      return Err(invalid_input(format!(
        "WHIR commitment expected {} evaluations, got {}",
        ck.num_evaluations,
        values.len()
      )));
    }

    let runtime = build_runtime(ck.num_variables())?;
    let mut challenger = initial_challenger(&runtime, &ck.label);
    let (root, data) = runtime.commit(witness(&runtime, values), &mut challenger);
    let commitment = WhirCommitment {
      num_evaluations: values.len(),
      root,
    };
    store_cached_data(blind, &commitment, data)?;
    Ok(commitment)
  }

  fn commit_partial(
    ck: &Self::CommitmentKey,
    values: &[E::Scalar],
    blind: &Self::Blind,
    is_small: bool,
  ) -> Result<(Self::PartialCommitment, Self::Blind), SpartanError> {
    let commitment = Self::commit(ck, values, blind, is_small)?;
    Ok((commitment, blind.clone()))
  }

  fn check_partial(commitment: &Self::PartialCommitment, n: usize) -> Result<(), SpartanError> {
    if commitment.num_evaluations != n {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: format!(
          "WHIR commitment contains {} evaluations, expected {n}",
          commitment.num_evaluations
        ),
      });
    }
    Ok(())
  }

  fn combine_partial(
    commitments: &[Self::PartialCommitment],
  ) -> Result<Self::Commitment, SpartanError> {
    if commitments.len() != 1 {
      return Err(invalid_input(
        "WHIR is non-homomorphic and requires exactly one witness commitment",
      ));
    }
    Ok(commitments[0].clone())
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.len() != 1 {
      return Err(invalid_input(
        "WHIR is non-homomorphic and requires exactly one witness blind",
      ));
    }
    Ok(blinds[0].clone())
  }

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    commitment: &Self::Commitment,
    poly: &[E::Scalar],
    blind: &Self::Blind,
    point: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError> {
    if commitment.num_evaluations != ck.num_evaluations || poly.len() != ck.num_evaluations {
      return Err(invalid_input(
        "WHIR key, commitment, and polynomial lengths do not agree",
      ));
    }
    if point.len() != ck.num_variables() {
      return Err(invalid_input(format!(
        "WHIR opening expected a {}-coordinate point, got {}",
        ck.num_variables(),
        point.len()
      )));
    }

    let eval = MultilinearPolynomial::evaluate_with(poly, point);
    let anchors = bind_outer_transcript::<E>(transcript, commitment, point, &eval)?;
    let runtime = build_runtime(ck.num_variables())?;
    let mut challenger = initial_challenger(&runtime, &ck.label);

    let data = if let Some(data) = cached_data(blind, commitment)? {
      challenger.observe(commitment.root.clone());
      data
    } else {
      let (recomputed_root, data) = runtime.commit(witness(&runtime, poly), &mut challenger);
      if recomputed_root != commitment.root {
        return Err(SpartanError::InvalidPCS);
      }
      store_cached_data(blind, commitment, data.clone())?;
      data
    };

    let whir_point = to_whir_point(point);
    let whir_eval = to_whir_extension(eval);
    bind_internal_claim(&mut challenger, &anchors, &whir_point, whir_eval);
    let protocol = opening_protocol(ck.num_variables());
    let proof = runtime.open_at(data, &protocol, &[whir_point], &mut challenger);
    if claimed_proof_evaluation(&proof) != Some(whir_eval) {
      return Err(internal_error(
        "WHIR returned an evaluation inconsistent with Toy Spartan",
      ));
    }

    Ok((eval, WhirEvaluationArgument { proof }))
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    commitment: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    argument: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    if commitment.num_evaluations != vk.num_evaluations {
      return Err(verification_error(
        "WHIR commitment length does not match the verifier key",
      ));
    }
    if point.len() != vk.num_variables() {
      return Err(verification_error(format!(
        "WHIR opening expected a {}-coordinate point, got {}",
        vk.num_variables(),
        point.len()
      )));
    }

    let anchors = bind_outer_transcript::<E>(transcript, commitment, point, eval)?;
    let runtime = build_runtime(vk.num_variables())?;
    let mut challenger = initial_challenger(&runtime, &vk.label);
    challenger.observe(commitment.root.clone());

    let whir_point = to_whir_point(point);
    let whir_eval = to_whir_extension(*eval);
    bind_internal_claim(&mut challenger, &anchors, &whir_point, whir_eval);
    let protocol = opening_protocol(vk.num_variables());
    let openings = runtime
      .verify_at(
        &commitment.root,
        &argument.proof,
        &protocol,
        &[whir_point],
        &mut challenger,
      )
      .map_err(|err| verification_error(format!("WHIR verification failed: {err:?}")))?;

    let opened_eval = openings
      .first()
      .and_then(|batch| batch.current().first())
      .copied()
      .ok_or_else(|| verification_error("WHIR proof omitted the requested evaluation"))?;
    if opened_eval != whir_eval {
      return Err(verification_error(
        "WHIR proof evaluation does not match the claimed value",
      ));
    }
    Ok(())
  }
}
