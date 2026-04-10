//! Owns the Phase 1 claim-reduction bucket surface and its canonical digests.
//!
//! It owns:
//! - deterministic Phase 1 bucketing from the Phase 0 claim bundle
//! - canonical bucket/proof/result digest formulas
//! - witness-backed same-schema point unification and verifier replay
//! - structural validation of unified-point reduction outputs
//!
//! It does not own:
//! - Phase 0 claim emission or witness reconstruction
//! - Phase 2 same-point collapse

use std::collections::BTreeMap;
use std::mem;

use neo_math::{from_complex, KExtensions, F, K};
use neo_reductions::sumcheck::{interpolate_from_evals, poly_eval_k};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::chip8::poly::{build_eq_table, eq_eval_le};
use crate::opening::OpeningDomain;

use super::opening_eval_claim_witness::{phase0_binding_digest, FamilyEvalClaimWitness};
use super::opening_eval_claims::{
    canonical_claim_cmp, phase0_family_order, CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalClaimId,
    FamilyEvalSchemaId, PackedColumnEval, Rv64imEvalClaimBundle,
};
use super::opening_phase0_binding_surface::Rv64imPhase0BindingSurface;
use super::opening_point_derivation::derive_phase0_point;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QuadraticRoundPoly {
    pub a0: K,
    pub a1: K,
    pub a2: K,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClaimReductionBucket {
    pub commitment_context: CommitmentContextId,
    pub schema: FamilyEvalSchemaId,
    pub claims: Vec<FamilyEvalClaim>,
}

impl ClaimReductionBucket {
    pub fn new(claims: Vec<FamilyEvalClaim>) -> Result<Self, ClaimReductionError> {
        Self::new_with_claim_validation(claims, true)
    }

    fn new_trusted_local(claims: Vec<FamilyEvalClaim>) -> Result<Self, ClaimReductionError> {
        Self::new_with_claim_validation(claims, false)
    }

    fn from_canonical_claims_trusted_local(claims: Vec<FamilyEvalClaim>) -> Result<Self, ClaimReductionError> {
        Self::new_with_canonical_claims(claims, false)
    }

    fn new_with_claim_validation(
        mut claims: Vec<FamilyEvalClaim>,
        validate_claims: bool,
    ) -> Result<Self, ClaimReductionError> {
        claims.sort_by(canonical_claim_cmp);
        Self::new_with_canonical_claims(claims, validate_claims)
    }

    fn new_with_canonical_claims(
        claims: Vec<FamilyEvalClaim>,
        validate_claims: bool,
    ) -> Result<Self, ClaimReductionError> {
        if claims.is_empty() {
            return Err(ClaimReductionError::EmptyBucket);
        }

        let first = &claims[0];
        let commitment_context = first.commitment_context;
        let schema = first.payload.schema;
        let point_arity = first.point.len();
        let payload_width = first.payload.column_evals.len();
        let opening_domain = domain_for_schema(schema);

        for (index, claim) in claims.iter().enumerate() {
            if validate_claims {
                claim
                    .validate()
                    .map_err(|source| ClaimReductionError::InvalidPhase0Claim { index, source })?;
            }

            if claim.commitment_context != commitment_context {
                return Err(ClaimReductionError::MixedCommitmentContext { index });
            }
            if claim.payload.schema != schema {
                return Err(ClaimReductionError::MixedSchema {
                    expected: schema,
                    actual: claim.payload.schema,
                    index,
                });
            }

            let actual_point_arity = claim.point.len();
            if actual_point_arity != point_arity {
                return Err(ClaimReductionError::MixedPointArity {
                    expected: point_arity,
                    actual: actual_point_arity,
                    index,
                });
            }

            let actual_payload_width = claim.payload.column_evals.len();
            if actual_payload_width != payload_width {
                return Err(ClaimReductionError::MixedPayloadWidth {
                    expected: payload_width,
                    actual: actual_payload_width,
                    index,
                });
            }

            let actual_domain = domain_for_schema(claim.payload.schema);
            if actual_domain != opening_domain {
                return Err(ClaimReductionError::MixedOpeningDomain {
                    expected: opening_domain,
                    actual: actual_domain,
                    index,
                });
            }
        }

        Ok(Self {
            commitment_context,
            schema,
            claims,
        })
    }

    pub fn validate(&self) -> Result<(), ClaimReductionError> {
        Self::new(self.claims.clone()).map(|_| ())
    }

    pub fn opening_domain(&self) -> OpeningDomain {
        domain_for_schema(self.schema)
    }

    pub fn point_arity(&self) -> usize {
        self.claims
            .first()
            .map(|claim| claim.point.len())
            .unwrap_or_default()
    }

    pub fn payload_width(&self) -> usize {
        self.claims
            .first()
            .map(|claim| claim.payload.column_evals.len())
            .unwrap_or_default()
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase1/bucket");
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase1/bucket/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase1/bucket/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/phase1/bucket/meta",
            &[
                self.schema.tag(),
                opening_domain_tag(self.opening_domain()),
                self.claims.len() as u64,
            ],
        );
        for claim in &self.claims {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/phase1/bucket/claim_digest",
                &phase1_claim_digest(claim),
            );
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClaimReductionProof {
    pub bucket_digest: [u8; 32],
    pub eta: K,
    pub gamma: Option<K>,
    pub rho: K,
    pub round_polys: Vec<QuadraticRoundPoly>,
    pub scalar_evals_at_r_star: Vec<K>,
    pub digest: [u8; 32],
}

impl ClaimReductionProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase1/proof");
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase1/proof/bucket_digest",
            &self.bucket_digest,
        );
        append_k(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase1/proof/eta",
            &self.eta,
        );
        append_k(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase1/proof/gamma",
            &self.gamma.unwrap_or(K::ZERO),
        );
        append_k(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase1/proof/rho",
            &self.rho,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/phase1/proof/round_poly_count",
            &[self.round_polys.len() as u64],
        );
        for poly in &self.round_polys {
            append_quadratic_round_poly(
                &mut tr,
                b"neo.fold.next/rv64im/opening_convergence/phase1/proof/round_poly",
                poly,
            );
        }
        append_k_vec(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase1/proof/scalar_evals_at_r_star",
            &self.scalar_evals_at_r_star,
        );
        tr.digest32()
    }

    pub fn validate_for_bucket(&self, bucket: &ClaimReductionBucket) -> Result<(), ClaimReductionError> {
        let expected_bucket_digest = bucket.expected_digest();
        if self.bucket_digest != expected_bucket_digest {
            return Err(ClaimReductionError::BucketDigestMismatch {
                expected: expected_bucket_digest,
                actual: self.bucket_digest,
            });
        }

        let expected_gamma = bucket.payload_width() > 1;
        match (expected_gamma, self.gamma) {
            (true, None) => {
                return Err(ClaimReductionError::MissingGamma {
                    payload_width: bucket.payload_width(),
                });
            }
            (false, Some(_)) => {
                return Err(ClaimReductionError::UnexpectedGamma {
                    payload_width: bucket.payload_width(),
                });
            }
            _ => {}
        }

        let expected_scalar_eval_count = bucket.claims.len();
        let actual_scalar_eval_count = self.scalar_evals_at_r_star.len();
        if actual_scalar_eval_count != expected_scalar_eval_count {
            return Err(ClaimReductionError::ScalarEvalCountMismatch {
                expected: expected_scalar_eval_count,
                actual: actual_scalar_eval_count,
            });
        }

        let expected_digest = self.expected_digest();
        if self.digest != expected_digest {
            return Err(ClaimReductionError::ProofDigestMismatch {
                expected: expected_digest,
                actual: self.digest,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClaimReductionResult {
    pub bucket: ClaimReductionBucket,
    pub unified_point: Vec<K>,
    pub unified_claims: Vec<FamilyEvalClaim>,
    pub proof: ClaimReductionProof,
}

impl ClaimReductionResult {
    pub fn validate(&self) -> Result<(), ClaimReductionError> {
        self.bucket.validate()?;
        self.proof.validate_for_bucket(&self.bucket)?;

        let expected_unified_claim_count = self.bucket.claims.len();
        let actual_unified_claim_count = self.unified_claims.len();
        if actual_unified_claim_count != expected_unified_claim_count {
            return Err(ClaimReductionError::UnifiedClaimCountMismatch {
                expected: expected_unified_claim_count,
                actual: actual_unified_claim_count,
            });
        }

        let expected_point_arity = self.bucket.point_arity();
        let actual_point_arity = self.unified_point.len();
        if actual_point_arity != expected_point_arity {
            return Err(ClaimReductionError::UnifiedPointArityMismatch {
                expected: expected_point_arity,
                actual: actual_point_arity,
            });
        }

        let actual_round_count = self.proof.round_polys.len();
        if actual_round_count != actual_point_arity {
            return Err(ClaimReductionError::RoundCountMismatch {
                expected: actual_point_arity,
                actual: actual_round_count,
            });
        }

        let mut payloads_by_object = BTreeMap::<[u8; 32], &Vec<PackedColumnEval>>::new();
        for (index, (source_claim, unified_claim)) in self
            .bucket
            .claims
            .iter()
            .zip(&self.unified_claims)
            .enumerate()
        {
            unified_claim
                .validate()
                .map_err(|source| ClaimReductionError::InvalidUnifiedClaim { index, source })?;

            if unified_claim.point != self.unified_point {
                return Err(ClaimReductionError::UnifiedPointMismatch { index });
            }
            if unified_claim.opened_object != source_claim.opened_object {
                return Err(ClaimReductionError::UnifiedOpenedObjectMismatch { index });
            }
            if unified_claim.id != source_claim.id {
                return Err(ClaimReductionError::UnifiedClaimIdMismatch { index });
            }
            if unified_claim.commitment_context != source_claim.commitment_context {
                return Err(ClaimReductionError::UnifiedCommitmentContextMismatch { index });
            }
            if unified_claim.binding_digest != source_claim.binding_digest {
                return Err(ClaimReductionError::UnifiedBindingDigestMismatch { index });
            }
            if unified_claim.payload.schema != source_claim.payload.schema {
                return Err(ClaimReductionError::UnifiedSchemaMismatch {
                    expected: source_claim.payload.schema,
                    actual: unified_claim.payload.schema,
                    index,
                });
            }

            match payloads_by_object.get(&unified_claim.opened_object.digest) {
                Some(expected_payload) if *expected_payload != &unified_claim.payload.column_evals => {
                    return Err(ClaimReductionError::SameObjectPayloadMismatch {
                        opened_object_digest: unified_claim.opened_object.digest,
                    });
                }
                Some(_) => {}
                None => {
                    payloads_by_object.insert(unified_claim.opened_object.digest, &unified_claim.payload.column_evals);
                }
            }
        }

        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        claim_reduction_result_digest_from_parts(
            self.bucket.expected_digest(),
            self.proof.gamma,
            self.proof.rho,
            &self.unified_point,
            &self.bucket.claims,
            &self.unified_claims,
            self.proof.expected_digest(),
        )
    }

    pub(crate) fn trusted_local_digest(&self) -> [u8; 32] {
        claim_reduction_result_digest_from_parts(
            self.proof.bucket_digest,
            self.proof.gamma,
            self.proof.rho,
            &self.unified_point,
            &self.bucket.claims,
            &self.unified_claims,
            self.proof.digest,
        )
    }
}

fn claim_reduction_result_digest_from_parts(
    bucket_digest: [u8; 32],
    gamma: Option<K>,
    rho: K,
    unified_point: &[K],
    source_claims: &[FamilyEvalClaim],
    unified_claims: &[FamilyEvalClaim],
    proof_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase1/result");
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/bucket_digest",
        &bucket_digest,
    );
    append_k(
        &mut tr,
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/gamma",
        &gamma.unwrap_or(K::ZERO),
    );
    append_k(
        &mut tr,
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/rho",
        &rho,
    );
    append_k_vec(
        &mut tr,
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/unified_point",
        unified_point,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/proof_digest",
        &proof_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/source_claim_count",
        &[source_claims.len() as u64],
    );
    for claim in source_claims {
        append_claim_id(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase1/result/source_claim_id",
            &claim.id,
        );
    }
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase1/result/unified_claim_count",
        &[unified_claims.len() as u64],
    );
    for claim in unified_claims {
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase1/result/unified_claim_digest",
            &phase1_unified_claim_digest(claim),
        );
    }
    tr.digest32()
}

pub fn build_claim_reduction_buckets(
    bundle: &Rv64imEvalClaimBundle,
) -> Result<Vec<ClaimReductionBucket>, ClaimReductionError> {
    let expected_bundle_digest = bundle.expected_digest();
    if bundle.digest != expected_bundle_digest {
        return Err(ClaimReductionError::EvalClaimBundleDigestMismatch {
            expected: expected_bundle_digest,
            actual: bundle.digest,
        });
    }

    build_claim_reduction_buckets_from_trusted_local_bundle(bundle, true)
}

fn build_claim_reduction_buckets_from_trusted_local_bundle(
    bundle: &Rv64imEvalClaimBundle,
    validate_claims: bool,
) -> Result<Vec<ClaimReductionBucket>, ClaimReductionError> {
    if !validate_claims {
        return build_claim_reduction_buckets_from_canonical_trusted_local_bundle(bundle);
    }

    let mut claims_by_schema = BTreeMap::<FamilyEvalSchemaId, Vec<FamilyEvalClaim>>::new();
    for claim in &bundle.claims {
        claims_by_schema
            .entry(claim.payload.schema)
            .or_default()
            .push(claim.clone());
    }

    claims_by_schema
        .into_values()
        .map(|claims| {
            if validate_claims {
                ClaimReductionBucket::new(claims)
            } else {
                ClaimReductionBucket::new_trusted_local(claims)
            }
        })
        .collect()
}

fn build_claim_reduction_buckets_from_canonical_trusted_local_bundle(
    bundle: &Rv64imEvalClaimBundle,
) -> Result<Vec<ClaimReductionBucket>, ClaimReductionError> {
    let mut buckets = Vec::new();
    let mut current_schema = None;
    let mut current_claims = Vec::new();

    for claim in &bundle.claims {
        if current_schema != Some(claim.payload.schema) && !current_claims.is_empty() {
            buckets.push(ClaimReductionBucket::from_canonical_claims_trusted_local(mem::take(
                &mut current_claims,
            ))?);
        }
        current_schema = Some(claim.payload.schema);
        current_claims.push(claim.clone());
    }

    if !current_claims.is_empty() {
        buckets.push(ClaimReductionBucket::from_canonical_claims_trusted_local(
            current_claims,
        )?);
    }

    Ok(buckets)
}

pub fn build_claim_reduction_results_from_witnesses(
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<ClaimReductionResult>, ClaimReductionError> {
    let bundle = Rv64imEvalClaimBundle::new(claim_witnesses.iter().map(|w| w.claim.clone()).collect())
        .map_err(ClaimReductionError::WitnessClaimBundleBuildFailed)?;
    build_claim_reduction_results_from_bundle_and_witnesses(&bundle, claim_witnesses)
}

pub(crate) fn build_claim_reduction_results_from_bundle_and_witnesses(
    bundle: &Rv64imEvalClaimBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<ClaimReductionResult>, ClaimReductionError> {
    let buckets = build_claim_reduction_buckets(bundle)?;
    let witness_map = build_claim_witness_map(claim_witnesses)?;

    buckets
        .iter()
        .map(|bucket| build_claim_reduction_result_from_bucket(bucket, &witness_map))
        .collect()
}

pub(crate) fn build_claim_reduction_results_from_bundle_and_witnesses_trusted_local(
    bundle: &Rv64imEvalClaimBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<ClaimReductionResult>, ClaimReductionError> {
    let buckets = build_claim_reduction_buckets_from_trusted_local_bundle(bundle, false)?;
    let witness_map = build_claim_witness_map(claim_witnesses)?;

    buckets
        .into_iter()
        .map(|bucket| build_claim_reduction_result_from_owned_trusted_bucket_witnesses(bucket, &witness_map))
        .collect()
}

pub fn verify_claim_reduction_result_with_binding_surface(
    result: &ClaimReductionResult,
    claim_witnesses: &[FamilyEvalClaimWitness],
    phase0_binding_surface: &Rv64imPhase0BindingSurface,
) -> Result<(), ClaimReductionError> {
    validate_phase0_binding_surface(phase0_binding_surface)?;
    verify_phase0_claim_bindings_against_surface(claim_witnesses, phase0_binding_surface)?;
    validate_claim_reduction_result_with_witnesses(result, claim_witnesses)
}

pub fn verify_claim_reduction_results_with_binding_surface(
    results: &[ClaimReductionResult],
    claim_witnesses: &[FamilyEvalClaimWitness],
    phase0_binding_surface: &Rv64imPhase0BindingSurface,
) -> Result<(), ClaimReductionError> {
    validate_phase0_binding_surface(phase0_binding_surface)?;
    verify_phase0_claim_bindings_against_surface(claim_witnesses, phase0_binding_surface)?;
    validate_claim_reduction_results_with_witnesses(results, claim_witnesses)
}

pub(crate) fn validate_claim_reduction_result_with_witnesses(
    result: &ClaimReductionResult,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), ClaimReductionError> {
    result.validate()?;
    let witness_map = build_claim_witness_map(claim_witnesses)?;
    let bucket_witnesses = claim_witnesses_for_bucket(&result.bucket, &witness_map)?;

    let mut transcript = phase1_transcript(result.bucket.expected_digest());
    let eta = sample_k(&mut transcript, b"neo.fold.next/rv64im/opening_convergence/phase1/eta");
    if eta != result.proof.eta {
        return Err(ClaimReductionError::EtaMismatch {
            expected: eta,
            actual: result.proof.eta,
        });
    }

    let gamma = if result.bucket.payload_width() > 1 {
        let gamma = sample_k(
            &mut transcript,
            b"neo.fold.next/rv64im/opening_convergence/phase1/gamma",
        );
        if Some(gamma) != result.proof.gamma {
            return Err(ClaimReductionError::GammaMismatch {
                expected: Some(gamma),
                actual: result.proof.gamma,
            });
        }
        Some(gamma)
    } else {
        if result.proof.gamma.is_some() {
            return Err(ClaimReductionError::UnexpectedGamma {
                payload_width: result.bucket.payload_width(),
            });
        }
        None
    };

    let rho = sample_k(&mut transcript, b"neo.fold.next/rv64im/opening_convergence/phase1/rho");
    if rho != result.proof.rho {
        return Err(ClaimReductionError::RhoMismatch {
            expected: rho,
            actual: result.proof.rho,
        });
    }

    let initial_sum = claimed_sum_for_bucket(&result.bucket, eta, gamma, rho);
    let (unified_point, terminal_sumcheck_value) =
        replay_phase1_sumcheck(&mut transcript, initial_sum, &result.proof.round_polys)?;
    if unified_point != result.unified_point {
        return Err(ClaimReductionError::UnifiedPointTranscriptMismatch);
    }

    append_k_vec(
        &mut transcript,
        b"neo.fold.next/rv64im/opening_convergence/phase1/scalar_evals_at_r_star",
        &result.proof.scalar_evals_at_r_star,
    );

    let mut payload_cache = BTreeMap::<[u8; 32], (Vec<PackedColumnEval>, K)>::new();
    let mut combined_scalar = K::ZERO;
    for (index, (claim_witness, unified_claim)) in bucket_witnesses
        .iter()
        .copied()
        .zip(&result.unified_claims)
        .enumerate()
    {
        let (expected_payload, expected_scalar_eval) =
            if let Some((payload, scalar_eval)) = payload_cache.get(&claim_witness.claim.opened_object.digest) {
                (payload.clone(), *scalar_eval)
            } else {
                let payload = claim_witness
                    .witness
                    .evaluate_payload(&result.unified_point)
                    .map_err(|source| ClaimReductionError::WitnessPayloadEvaluationFailed { index, source })?;
                let scalar_eval = scalarize_column_evals(&payload, eta, gamma);
                payload_cache.insert(claim_witness.claim.opened_object.digest, (payload.clone(), scalar_eval));
                (payload, scalar_eval)
            };
        if unified_claim.payload.column_evals != expected_payload {
            return Err(ClaimReductionError::UnifiedPayloadDoesNotMatchWitness { index });
        }

        if result.proof.scalar_evals_at_r_star[index] != expected_scalar_eval {
            return Err(ClaimReductionError::ScalarEvalMismatch {
                index,
                expected: expected_scalar_eval,
                actual: result.proof.scalar_evals_at_r_star[index],
            });
        }

        combined_scalar += rho_power(rho, index)
            * eq_eval_le(&claim_witness.claim.point, &result.unified_point)
            * expected_scalar_eval;
    }

    if terminal_sumcheck_value != combined_scalar {
        return Err(ClaimReductionError::CombinedScalarMismatch {
            expected: combined_scalar,
            actual: terminal_sumcheck_value,
        });
    }

    Ok(())
}

pub(crate) fn validate_claim_reduction_results_with_witnesses(
    results: &[ClaimReductionResult],
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), ClaimReductionError> {
    let expected_buckets = build_claim_reduction_buckets(
        &Rv64imEvalClaimBundle::new(claim_witnesses.iter().map(|w| w.claim.clone()).collect())
            .map_err(ClaimReductionError::WitnessClaimBundleBuildFailed)?,
    )?;

    if results.len() != expected_buckets.len() {
        return Err(ClaimReductionError::ResultCountMismatch {
            expected: expected_buckets.len(),
            actual: results.len(),
        });
    }

    for (index, (result, expected_bucket)) in results.iter().zip(expected_buckets.iter()).enumerate() {
        if &result.bucket != expected_bucket {
            return Err(ClaimReductionError::UnexpectedBucketAtIndex { index });
        }
        validate_claim_reduction_result_with_witnesses(result, claim_witnesses)?;
    }

    Ok(())
}

pub fn phase1_claim_digest(claim: &FamilyEvalClaim) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase1/claim");
    append_phase1_claim_body(&mut tr, claim);
    tr.digest32()
}

pub fn phase1_unified_claim_digest(claim: &FamilyEvalClaim) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase1/unified_claim");
    append_phase1_claim_body(&mut tr, claim);
    tr.digest32()
}

pub fn domain_for_schema(schema: FamilyEvalSchemaId) -> OpeningDomain {
    schema.opening_domain()
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ClaimReductionError {
    #[error("phase1 claim-reduction bucket cannot be empty")]
    EmptyBucket,
    #[error("phase1 eval-claim bundle digest mismatch: expected {expected:?}, got {actual:?}")]
    EvalClaimBundleDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 could not build a canonical claim bundle from witnesses: {0}")]
    WitnessClaimBundleBuildFailed(EvalClaimError),
    #[error("phase1 bucket contains invalid phase0 claim at index {index}: {source}")]
    InvalidPhase0Claim {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase1 witness set contains duplicate claim id {claim_id:?}")]
    DuplicateWitnessClaimId { claim_id: FamilyEvalClaimId },
    #[error("phase1 witness set is missing claim witness for id {claim_id:?}")]
    MissingWitnessForClaimId { claim_id: FamilyEvalClaimId },
    #[error("phase1 witness claim for id {claim_id:?} does not match the bucket claim")]
    WitnessClaimMismatch { claim_id: FamilyEvalClaimId },
    #[error("phase1 binding surface target count mismatch: expected {expected}, got {actual}")]
    BindingSurfaceTargetCountMismatch { expected: usize, actual: usize },
    #[error("phase1 binding surface schema mismatch at index {index}: expected {expected:?}, got {actual:?}")]
    BindingSurfaceSchemaMismatch {
        index: usize,
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
    },
    #[error("phase1 binding surface is missing target for {schema:?}")]
    BindingSurfaceTargetMissing { schema: FamilyEvalSchemaId },
    #[error("phase1 binding surface target digest mismatch at index {index}: expected {expected:?}, got {actual:?}")]
    BindingSurfaceTargetDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 binding surface digest mismatch: expected {expected:?}, got {actual:?}")]
    BindingSurfaceDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 witness claim {claim_id:?} binding digest mismatch: expected {expected:?}, got {actual:?}")]
    WitnessClaimBindingDigestMismatch {
        claim_id: FamilyEvalClaimId,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 witness claim {claim_id:?} point does not match its canonical binding-derived point")]
    WitnessClaimPointBindingMismatch { claim_id: FamilyEvalClaimId },
    #[error("phase1 bucket mixes commitment contexts at claim {index}")]
    MixedCommitmentContext { index: usize },
    #[error("phase1 bucket mixes schemas: expected {expected:?}, got {actual:?} at claim {index}")]
    MixedSchema {
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
        index: usize,
    },
    #[error("phase1 bucket mixes point arities: expected {expected}, got {actual} at claim {index}")]
    MixedPointArity {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("phase1 bucket mixes payload widths: expected {expected}, got {actual} at claim {index}")]
    MixedPayloadWidth {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("phase1 bucket mixes opening domains: expected {expected:?}, got {actual:?} at claim {index}")]
    MixedOpeningDomain {
        expected: OpeningDomain,
        actual: OpeningDomain,
        index: usize,
    },
    #[error("phase1 proof bucket digest mismatch: expected {expected:?}, got {actual:?}")]
    BucketDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 proof for payload width {payload_width} requires gamma")]
    MissingGamma { payload_width: usize },
    #[error("phase1 proof for payload width {payload_width} must not carry gamma")]
    UnexpectedGamma { payload_width: usize },
    #[error("phase1 scalar eval count mismatch: expected {expected}, got {actual}")]
    ScalarEvalCountMismatch { expected: usize, actual: usize },
    #[error("phase1 proof digest mismatch: expected {expected:?}, got {actual:?}")]
    ProofDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 eta mismatch: expected {expected:?}, got {actual:?}")]
    EtaMismatch { expected: K, actual: K },
    #[error("phase1 gamma mismatch: expected {expected:?}, got {actual:?}")]
    GammaMismatch {
        expected: Option<K>,
        actual: Option<K>,
    },
    #[error("phase1 rho mismatch: expected {expected:?}, got {actual:?}")]
    RhoMismatch { expected: K, actual: K },
    #[error("phase1 result unified-claim count mismatch: expected {expected}, got {actual}")]
    UnifiedClaimCountMismatch { expected: usize, actual: usize },
    #[error("phase1 result unified-point arity mismatch: expected {expected}, got {actual}")]
    UnifiedPointArityMismatch { expected: usize, actual: usize },
    #[error("phase1 result round count mismatch: expected {expected}, got {actual}")]
    RoundCountMismatch { expected: usize, actual: usize },
    #[error("phase1 result contains invalid unified claim at index {index}: {source}")]
    InvalidUnifiedClaim {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase1 result unified claim {index} does not carry the result unified point")]
    UnifiedPointMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed opened object")]
    UnifiedOpenedObjectMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed claim id")]
    UnifiedClaimIdMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed commitment context")]
    UnifiedCommitmentContextMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed binding digest")]
    UnifiedBindingDigestMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed schema: expected {expected:?}, got {actual:?}")]
    UnifiedSchemaMismatch {
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
        index: usize,
    },
    #[error(
        "phase1 result carries different payloads for opened object {opened_object_digest:?} at one unified point"
    )]
    SameObjectPayloadMismatch { opened_object_digest: [u8; 32] },
    #[error("phase1 sumcheck round {round} invariant failed: expected {expected:?}, got {actual:?}")]
    SumcheckInvariantFailed {
        round: usize,
        expected: K,
        actual: K,
    },
    #[error("phase1 result unified point does not match the transcript-derived r*")]
    UnifiedPointTranscriptMismatch,
    #[error("phase1 witness payload evaluation failed for unified claim {index}: {source}")]
    WitnessPayloadEvaluationFailed {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase1 unified claim {index} payload does not match its real witness at r*")]
    UnifiedPayloadDoesNotMatchWitness { index: usize },
    #[error("phase1 scalar eval mismatch for unified claim {index}: expected {expected:?}, got {actual:?}")]
    ScalarEvalMismatch {
        index: usize,
        expected: K,
        actual: K,
    },
    #[error("phase1 combined scalar check failed: expected {expected:?}, got {actual:?}")]
    CombinedScalarMismatch { expected: K, actual: K },
    #[error("phase1 result count mismatch: expected {expected}, got {actual}")]
    ResultCountMismatch { expected: usize, actual: usize },
    #[error("phase1 result at index {index} carries an unexpected bucket")]
    UnexpectedBucketAtIndex { index: usize },
}

#[derive(Clone)]
struct ClaimReductionOracleTerm {
    rho_power: K,
    eq_values: Vec<K>,
    g_values: Vec<K>,
}

struct ClaimReductionOracle {
    terms: Vec<ClaimReductionOracleTerm>,
    rounds: usize,
}

impl ClaimReductionOracle {
    fn new(
        bucket: &ClaimReductionBucket,
        witnesses: &[&FamilyEvalClaimWitness],
        eta: K,
        gamma: Option<K>,
        rho: K,
    ) -> Result<Self, ClaimReductionError> {
        let mut scalarized_tables = BTreeMap::<[u8; 32], Vec<K>>::new();
        let terms = bucket
            .claims
            .iter()
            .zip(witnesses.iter())
            .enumerate()
            .map(|(index, (claim, witness))| {
                let eq_values = build_eq_table(&claim.point);
                let g_values = if let Some(values) = scalarized_tables.get(&claim.opened_object.digest) {
                    values.clone()
                } else {
                    let values = scalarized_claim_table(witness, eta, gamma)?;
                    scalarized_tables.insert(claim.opened_object.digest, values.clone());
                    values
                };
                if eq_values.len() != g_values.len() {
                    return Err(ClaimReductionError::WitnessPayloadEvaluationFailed {
                        index,
                        source: EvalClaimError::WitnessRowDomainLengthMismatch {
                            column_index: 0,
                            expected: eq_values.len(),
                            actual: g_values.len(),
                        },
                    });
                }
                Ok(ClaimReductionOracleTerm {
                    rho_power: rho_power(rho, index),
                    eq_values,
                    g_values,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            terms,
            rounds: bucket.point_arity(),
        })
    }

    fn eval_round_at(&self, point: K) -> K {
        self.terms
            .iter()
            .fold(K::ZERO, |acc, term| acc + term.rho_power * eval_term_round(term, point))
    }

    fn fold(&mut self, challenge: K) {
        for term in &mut self.terms {
            term.eq_values = fold_values(&term.eq_values, challenge);
            term.g_values = fold_values(&term.g_values, challenge);
        }
    }
}

fn build_claim_witness_map<'a>(
    claim_witnesses: &'a [FamilyEvalClaimWitness],
) -> Result<BTreeMap<FamilyEvalClaimId, &'a FamilyEvalClaimWitness>, ClaimReductionError> {
    let mut map = BTreeMap::new();
    for claim_witness in claim_witnesses {
        let claim_id = claim_witness.claim.id;
        if map.insert(claim_id, claim_witness).is_some() {
            return Err(ClaimReductionError::DuplicateWitnessClaimId { claim_id });
        }
    }
    Ok(map)
}

fn claim_witnesses_for_bucket<'a>(
    bucket: &ClaimReductionBucket,
    witness_map: &'a BTreeMap<FamilyEvalClaimId, &'a FamilyEvalClaimWitness>,
) -> Result<Vec<&'a FamilyEvalClaimWitness>, ClaimReductionError> {
    bucket
        .claims
        .iter()
        .map(|claim| {
            let witness = witness_map
                .get(&claim.id)
                .copied()
                .ok_or(ClaimReductionError::MissingWitnessForClaimId { claim_id: claim.id })?;
            if witness.claim != *claim {
                return Err(ClaimReductionError::WitnessClaimMismatch { claim_id: claim.id });
            }
            Ok(witness)
        })
        .collect()
}

fn build_claim_reduction_result_from_bucket(
    bucket: &ClaimReductionBucket,
    witness_map: &BTreeMap<FamilyEvalClaimId, &FamilyEvalClaimWitness>,
) -> Result<ClaimReductionResult, ClaimReductionError> {
    let result = build_claim_reduction_result_from_trusted_bucket_witnesses(bucket, witness_map)?;
    result.validate()?;
    Ok(result)
}

fn build_claim_reduction_result_from_trusted_bucket_witnesses(
    bucket: &ClaimReductionBucket,
    witness_map: &BTreeMap<FamilyEvalClaimId, &FamilyEvalClaimWitness>,
) -> Result<ClaimReductionResult, ClaimReductionError> {
    let bucket_witnesses = claim_witnesses_for_bucket(bucket, witness_map)?;
    let bucket_digest = bucket.expected_digest();
    build_claim_reduction_result_from_owned_trusted_bucket_witnesses_with_materialized_witnesses(
        bucket.clone(),
        &bucket_witnesses,
        bucket_digest,
    )
}

fn build_claim_reduction_result_from_owned_trusted_bucket_witnesses(
    bucket: ClaimReductionBucket,
    witness_map: &BTreeMap<FamilyEvalClaimId, &FamilyEvalClaimWitness>,
) -> Result<ClaimReductionResult, ClaimReductionError> {
    let bucket_witnesses = claim_witnesses_for_bucket(&bucket, witness_map)?;
    let bucket_digest = bucket.expected_digest();
    build_claim_reduction_result_from_owned_trusted_bucket_witnesses_with_materialized_witnesses(
        bucket,
        &bucket_witnesses,
        bucket_digest,
    )
}

fn build_claim_reduction_result_from_owned_trusted_bucket_witnesses_with_materialized_witnesses(
    bucket: ClaimReductionBucket,
    bucket_witnesses: &[&FamilyEvalClaimWitness],
    bucket_digest: [u8; 32],
) -> Result<ClaimReductionResult, ClaimReductionError> {
    let mut transcript = phase1_transcript(bucket_digest);

    let eta = sample_k(&mut transcript, b"neo.fold.next/rv64im/opening_convergence/phase1/eta");
    let gamma = if bucket.payload_width() > 1 {
        Some(sample_k(
            &mut transcript,
            b"neo.fold.next/rv64im/opening_convergence/phase1/gamma",
        ))
    } else {
        None
    };
    let rho = sample_k(&mut transcript, b"neo.fold.next/rv64im/opening_convergence/phase1/rho");
    let initial_sum = claimed_sum_for_bucket(&bucket, eta, gamma, rho);

    let mut oracle = ClaimReductionOracle::new(&bucket, &bucket_witnesses, eta, gamma, rho)?;
    let (round_polys, unified_point) = run_phase1_sumcheck_prover(&mut transcript, &mut oracle, initial_sum)?;

    let mut payload_cache = BTreeMap::<[u8; 32], Vec<PackedColumnEval>>::new();
    let mut unified_claims = Vec::with_capacity(bucket_witnesses.len());
    let mut scalar_evals_at_r_star = Vec::with_capacity(bucket_witnesses.len());
    for (index, claim_witness) in bucket_witnesses.iter().copied().enumerate() {
        let payload = if let Some(payload) = payload_cache.get(&claim_witness.claim.opened_object.digest) {
            payload.clone()
        } else {
            let payload = claim_witness
                .witness
                .evaluate_payload(&unified_point)
                .map_err(|source| ClaimReductionError::WitnessPayloadEvaluationFailed { index, source })?;
            payload_cache.insert(claim_witness.claim.opened_object.digest, payload.clone());
            payload
        };
        let unified_payload =
            super::opening_eval_claims::FamilyEvalPayload::new(claim_witness.claim.payload.schema, payload.clone())
                .map_err(|source| ClaimReductionError::InvalidUnifiedClaim { index, source })?;
        unified_claims.push(
            FamilyEvalClaim::new(
                claim_witness.claim.opened_object.clone(),
                claim_witness.claim.id.slot,
                claim_witness.claim.commitment_context,
                unified_point.clone(),
                unified_payload,
                claim_witness.claim.binding_digest,
            )
            .map_err(|source| ClaimReductionError::InvalidUnifiedClaim { index, source })?,
        );
        scalar_evals_at_r_star.push(scalarize_column_evals(&payload, eta, gamma));
    }
    append_k_vec(
        &mut transcript,
        b"neo.fold.next/rv64im/opening_convergence/phase1/scalar_evals_at_r_star",
        &scalar_evals_at_r_star,
    );

    let mut proof = ClaimReductionProof {
        bucket_digest,
        eta,
        gamma,
        rho,
        round_polys,
        scalar_evals_at_r_star,
        digest: [0; 32],
    };
    proof.digest = proof.expected_digest();

    let result = ClaimReductionResult {
        bucket,
        unified_point,
        unified_claims,
        proof,
    };
    Ok(result)
}

fn phase1_transcript(bucket_digest: [u8; 32]) -> Poseidon2Transcript {
    let mut transcript = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase1");
    transcript.append_message(
        b"neo.fold.next/rv64im/opening_convergence/phase1/bucket_digest",
        &bucket_digest,
    );
    transcript
}

fn validate_phase0_binding_surface(surface: &Rv64imPhase0BindingSurface) -> Result<(), ClaimReductionError> {
    let expected_order = phase0_family_order();
    if surface.targets.len() != expected_order.len() {
        return Err(ClaimReductionError::BindingSurfaceTargetCountMismatch {
            expected: expected_order.len(),
            actual: surface.targets.len(),
        });
    }
    for (index, (target, expected_schema)) in surface
        .targets
        .iter()
        .zip(expected_order.iter())
        .enumerate()
    {
        if target.schema != *expected_schema {
            return Err(ClaimReductionError::BindingSurfaceSchemaMismatch {
                index,
                expected: *expected_schema,
                actual: target.schema,
            });
        }
        let expected_digest = target.expected_digest();
        if target.digest != expected_digest {
            return Err(ClaimReductionError::BindingSurfaceTargetDigestMismatch {
                index,
                expected: expected_digest,
                actual: target.digest,
            });
        }
    }
    let expected_digest = surface.expected_digest();
    if surface.digest != expected_digest {
        return Err(ClaimReductionError::BindingSurfaceDigestMismatch {
            expected: expected_digest,
            actual: surface.digest,
        });
    }
    Ok(())
}

fn verify_phase0_claim_bindings_against_surface(
    claim_witnesses: &[FamilyEvalClaimWitness],
    surface: &Rv64imPhase0BindingSurface,
) -> Result<(), ClaimReductionError> {
    for claim_witness in claim_witnesses {
        let claim = &claim_witness.claim;
        let target = surface
            .targets
            .iter()
            .find(|target| target.schema == claim.payload.schema)
            .ok_or(ClaimReductionError::BindingSurfaceTargetMissing {
                schema: claim.payload.schema,
            })?;
        let expected_binding_digest = phase0_binding_digest(
            &claim.opened_object,
            claim.payload.schema,
            claim.id.slot,
            target.family_binding_anchor_digest,
            target.stage_proof_binding_digest,
        );
        if claim.binding_digest != expected_binding_digest {
            return Err(ClaimReductionError::WitnessClaimBindingDigestMismatch {
                claim_id: claim.id,
                expected: expected_binding_digest,
                actual: claim.binding_digest,
            });
        }
        let expected_point = derive_phase0_point(
            &claim.opened_object,
            &claim.commitment_context,
            claim.payload.schema,
            claim.id.slot,
            claim.binding_digest,
        );
        if claim.point != expected_point {
            return Err(ClaimReductionError::WitnessClaimPointBindingMismatch { claim_id: claim.id });
        }
    }
    Ok(())
}

fn sample_k(tr: &mut Poseidon2Transcript, label: &'static [u8]) -> K {
    let challenge = tr.challenge_fields(label, 2);
    from_complex(challenge[0], challenge[1])
}

fn run_phase1_sumcheck_prover(
    transcript: &mut Poseidon2Transcript,
    oracle: &mut ClaimReductionOracle,
    initial_sum: K,
) -> Result<(Vec<QuadraticRoundPoly>, Vec<K>), ClaimReductionError> {
    let xs = [K::ZERO, K::ONE, k_from_u64(2)];
    let mut running_sum = initial_sum;
    let mut round_polys = Vec::with_capacity(oracle.rounds);
    let mut unified_point = Vec::with_capacity(oracle.rounds);

    for round in 0..oracle.rounds {
        let ys = xs.map(|x| oracle.eval_round_at(x));
        let actual = ys[0] + ys[1];
        if actual != running_sum {
            return Err(ClaimReductionError::SumcheckInvariantFailed {
                round,
                expected: running_sum,
                actual,
            });
        }

        let coeffs = interpolate_from_evals(&xs, &ys);
        let round_poly = QuadraticRoundPoly {
            a0: coeffs.first().copied().unwrap_or(K::ZERO),
            a1: coeffs.get(1).copied().unwrap_or(K::ZERO),
            a2: coeffs.get(2).copied().unwrap_or(K::ZERO),
        };
        append_quadratic_round_poly(
            transcript,
            b"neo.fold.next/rv64im/opening_convergence/phase1/round_poly",
            &round_poly,
        );
        let challenge = sample_k(
            transcript,
            b"neo.fold.next/rv64im/opening_convergence/phase1/r_star_round",
        );
        running_sum = poly_eval_k(&[round_poly.a0, round_poly.a1, round_poly.a2], challenge);
        oracle.fold(challenge);
        round_polys.push(round_poly);
        unified_point.push(challenge);
    }

    Ok((round_polys, unified_point))
}

fn replay_phase1_sumcheck(
    transcript: &mut Poseidon2Transcript,
    initial_sum: K,
    round_polys: &[QuadraticRoundPoly],
) -> Result<(Vec<K>, K), ClaimReductionError> {
    let mut running_sum = initial_sum;
    let mut unified_point = Vec::with_capacity(round_polys.len());

    for (round, round_poly) in round_polys.iter().enumerate() {
        let coeffs = [round_poly.a0, round_poly.a1, round_poly.a2];
        let actual = poly_eval_k(&coeffs, K::ZERO) + poly_eval_k(&coeffs, K::ONE);
        if actual != running_sum {
            return Err(ClaimReductionError::SumcheckInvariantFailed {
                round,
                expected: running_sum,
                actual,
            });
        }
        append_quadratic_round_poly(
            transcript,
            b"neo.fold.next/rv64im/opening_convergence/phase1/round_poly",
            round_poly,
        );
        let challenge = sample_k(
            transcript,
            b"neo.fold.next/rv64im/opening_convergence/phase1/r_star_round",
        );
        running_sum = poly_eval_k(&coeffs, challenge);
        unified_point.push(challenge);
    }

    Ok((unified_point, running_sum))
}

fn claimed_sum_for_bucket(bucket: &ClaimReductionBucket, eta: K, gamma: Option<K>, rho: K) -> K {
    bucket
        .claims
        .iter()
        .enumerate()
        .fold(K::ZERO, |acc, (index, claim)| {
            acc + rho_power(rho, index) * scalarize_payload(&claim.payload, eta, gamma)
        })
}

fn scalarized_claim_table(
    claim_witness: &FamilyEvalClaimWitness,
    eta: K,
    gamma: Option<K>,
) -> Result<Vec<K>, ClaimReductionError> {
    let time_len = claim_witness
        .witness
        .packed_columns
        .first()
        .map(|column| column.rows.len())
        .unwrap_or_default();

    (0..time_len)
        .map(|row_index| scalarize_witness_row(&claim_witness.witness, row_index, eta, gamma))
        .collect()
}

fn scalarize_witness_row(
    witness: &super::opening_eval_claim_witness::OpenedAjtaiObjectWitness,
    row_index: usize,
    eta: K,
    gamma: Option<K>,
) -> Result<K, ClaimReductionError> {
    let column_scalars = witness
        .packed_columns
        .iter()
        .map(|column| {
            let row =
                column
                    .rows
                    .get(row_index)
                    .copied()
                    .ok_or(ClaimReductionError::WitnessPayloadEvaluationFailed {
                        index: row_index,
                        source: EvalClaimError::WitnessRowDomainLengthMismatch {
                            column_index: column.column_index as usize,
                            expected: row_index + 1,
                            actual: column.rows.len(),
                        },
                    })?;
            Ok(coeff_linearize_row(&row, eta))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(gamma_linearize_scalars(&column_scalars, gamma))
}

fn coeff_linearize_row(row: &[F; neo_math::D], eta: K) -> K {
    let mut eta_power = K::ONE;
    row.iter().fold(K::ZERO, |acc, coeff| {
        let next = acc + eta_power * K::from(*coeff);
        eta_power *= eta;
        next
    })
}

fn scalarize_payload(payload: &super::opening_eval_claims::FamilyEvalPayload, eta: K, gamma: Option<K>) -> K {
    scalarize_column_evals(&payload.column_evals, eta, gamma)
}

fn scalarize_column_evals(column_evals: &[PackedColumnEval], eta: K, gamma: Option<K>) -> K {
    let column_scalars = column_evals
        .iter()
        .map(|column_eval| {
            let mut eta_power = K::ONE;
            column_eval.coeffs.iter().fold(K::ZERO, |acc, coeff| {
                let next = acc + eta_power * *coeff;
                eta_power *= eta;
                next
            })
        })
        .collect::<Vec<_>>();
    gamma_linearize_scalars(&column_scalars, gamma)
}

fn gamma_linearize_scalars(values: &[K], gamma: Option<K>) -> K {
    match gamma {
        Some(gamma) => values
            .iter()
            .enumerate()
            .fold(K::ZERO, |acc, (index, value)| acc + gamma_power(gamma, index) * *value),
        None => values.first().copied().unwrap_or(K::ZERO),
    }
}

fn gamma_power(gamma: K, index: usize) -> K {
    rho_power(gamma, index)
}

fn rho_power(rho: K, index: usize) -> K {
    let mut base = rho;
    let mut exp = (index as u64) + 1;
    let mut acc = K::ONE;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc
}

fn eval_term_round(term: &ClaimReductionOracleTerm, point: K) -> K {
    debug_assert_eq!(term.eq_values.len(), term.g_values.len());
    if term.eq_values.len() == 1 {
        return term.eq_values[0] * term.g_values[0];
    }

    let mut acc = K::ZERO;
    for pair in 0..(term.eq_values.len() / 2) {
        let eq0 = term.eq_values[2 * pair];
        let eq1 = term.eq_values[2 * pair + 1];
        let g0 = term.g_values[2 * pair];
        let g1 = term.g_values[2 * pair + 1];
        let eq_z = eq0 + point * (eq1 - eq0);
        let g_z = g0 + point * (g1 - g0);
        acc += eq_z * g_z;
    }
    acc
}

fn fold_values(values: &[K], challenge: K) -> Vec<K> {
    if values.len() == 1 {
        return values.to_vec();
    }
    (0..(values.len() / 2))
        .map(|pair| {
            let low = values[2 * pair];
            let high = values[2 * pair + 1];
            low + challenge * (high - low)
        })
        .collect()
}

fn k_from_u64(value: u64) -> K {
    K::from(F::from_u64(value))
}

fn append_phase1_claim_body(tr: &mut Poseidon2Transcript, claim: &FamilyEvalClaim) {
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/phase1/claim/opened_object_digest",
        &claim.opened_object.digest,
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase1/claim/meta",
        &[claim.id.slot as u64, claim.payload.schema.tag()],
    );
    append_k_vec(
        tr,
        b"neo.fold.next/rv64im/opening_convergence/phase1/claim/point",
        &claim.point,
    );
    append_packed_column_evals(
        tr,
        b"neo.fold.next/rv64im/opening_convergence/phase1/claim/payload",
        &claim.payload.column_evals,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/opening_convergence/phase1/claim/binding_digest",
        &claim.binding_digest,
    );
}

fn opening_domain_tag(domain: OpeningDomain) -> u64 {
    match domain {
        OpeningDomain::Cpu => 1,
        OpeningDomain::Mem => 2,
    }
}

fn append_k(tr: &mut Poseidon2Transcript, label: &'static [u8], value: &K) {
    tr.append_fields_iter(label, value.as_coeffs().len(), value.as_coeffs());
}

fn append_k_vec(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase1/k_vec_len",
        &[values.len() as u64],
    );
    let coeffs_per_elem = values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    tr.append_fields_iter(
        label,
        values.len().saturating_mul(coeffs_per_elem),
        values.iter().flat_map(|value| value.as_coeffs()),
    );
}

fn append_packed_column_evals(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[PackedColumnEval]) {
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase1/column_eval_count",
        &[values.len() as u64],
    );
    tr.append_fields_iter(
        label,
        values
            .iter()
            .map(|value| value.coeffs.len())
            .sum::<usize>()
            .saturating_mul(2),
        values
            .iter()
            .flat_map(|value| value.coeffs.iter().flat_map(|coeff| coeff.as_coeffs())),
    );
}

fn append_quadratic_round_poly(tr: &mut Poseidon2Transcript, label: &'static [u8], poly: &QuadraticRoundPoly) {
    tr.append_fields_iter(
        label,
        poly.a0.as_coeffs().len() + poly.a1.as_coeffs().len() + poly.a2.as_coeffs().len(),
        poly.a0
            .as_coeffs()
            .into_iter()
            .chain(poly.a1.as_coeffs())
            .chain(poly.a2.as_coeffs()),
    );
}

fn append_claim_id(tr: &mut Poseidon2Transcript, label: &'static [u8], id: &FamilyEvalClaimId) {
    tr.append_message(label, &id.opened_object_digest);
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase1/claim_id/slot",
        &[id.slot as u64],
    );
}
