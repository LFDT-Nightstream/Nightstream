//! Retained red-team regressions for native/recursive sumcheck parity.
//!
//! | Regression | Mathematical obligation | Current gap |
//! |---|---|---|
//! | zero NC projection | An NC proof accepted for a fresh claim must certify the committed witness's centered norm bound | output `y_zcol` is not yet bound back to the committed witness |

#[path = "../support/mod.rs"]
mod support;

use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::engine::r1cs_circuit::{KVar, R1csBuilder, TranscriptGadget};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs::{self, NifsProof};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_fe_claimed_initial, enforce_split_nc_pi_ccs_v, sample_engine_challenges, FeClaimedInitialInputs,
    SplitNcPiCcsVConfig, SplitNcPiCcsVMessages,
};
use neo_fold_clean::paper::reductions::{pi_ccs, pi_dec, pi_rlc};
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance, CcsWitness};
use neo_fold_clean::{config, preprocess};
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_reductions::optimized_engine::oracle::{NcColSnapshot, RowPhaseSnapshot};
use neo_reductions::optimized_engine::{
    BackendTranscriptMode, FeRowRoundSummary, FeRowRoundTrace, FeSumcheckBackend, NcColRoundTrace, NcFinalizedColState,
    NcSumcheckBackend,
};
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript as _;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

fn noncanonical_alias(value: K) -> Option<K> {
    let [real, imag] = value.as_coeffs();
    for limb in 0..2 {
        let canonical = [real, imag][limb].as_canonical_u64();
        let Some(raw) = F::ORDER_U64.checked_add(canonical) else {
            continue;
        };
        let encoded = bincode::serialize(&raw).expect("serialize raw Goldilocks word");
        let alias: F = bincode::deserialize(&encoded).expect("p3-goldilocks 0.5 serde accepts a noncanonical raw word");
        let candidate = if limb == 0 {
            from_complex(alias, imag)
        } else {
            from_complex(real, alias)
        };
        if candidate == value && bincode::serialize(&candidate).ok() != bincode::serialize(&value).ok() {
            return Some(candidate);
        }
    }
    None
}

fn alias_one_verified_round_coefficient(rounds: &mut [Vec<K>]) -> bool {
    for round in rounds {
        for coefficient in round {
            if let Some(alias) = noncanonical_alias(*coefficient) {
                *coefficient = alias;
                return true;
            }
        }
    }
    false
}

struct EmptyZeroRowBackend;

impl FeSumcheckBackend for EmptyZeroRowBackend {
    fn start(&mut self, _snapshot: &RowPhaseSnapshot<'_>) -> bool {
        true
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        Vec::new()
    }

    fn fold(&mut self, _r: K) {}

    fn row_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<FeRowRoundTrace> {
        let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_state, transcript_absorbed);
        let mut challenges = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            transcript.append_fields_raw(&[]);
            let lanes = transcript.challenge_fields_raw(2);
            challenges.push(neo_math::from_complex(lanes[0], lanes[1]));
        }
        Some(FeRowRoundTrace {
            coeffs: vec![Vec::new(); rounds],
            challenges,
            transcript_after: Some((transcript.state(), transcript.absorbed())),
            ajtai_y_eval: None,
        })
    }
}

struct OverlongZeroSuffixRowBackend {
    coeffs: Vec<K>,
}

impl FeSumcheckBackend for OverlongZeroSuffixRowBackend {
    fn start(&mut self, _snapshot: &RowPhaseSnapshot<'_>) -> bool {
        true
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.coeffs.clone()
    }

    fn fold(&mut self, _r: K) {}

    fn row_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<FeRowRoundTrace> {
        assert_eq!(rounds, 1, "the toy fixture must have one FE row-domain round");
        let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_state, transcript_absorbed);
        transcript.append_fields_raw(&neo_reductions::sumcheck::round_coeff_fields(&self.coeffs));
        let lanes = transcript.challenge_fields_raw(2);
        Some(FeRowRoundTrace {
            coeffs: vec![self.coeffs.clone()],
            challenges: vec![neo_math::from_complex(lanes[0], lanes[1])],
            transcript_after: Some((transcript.state(), transcript.absorbed())),
            ajtai_y_eval: None,
        })
    }
}

struct OverlongZeroSuffixNcBackend {
    coeffs: Vec<K>,
    eq_beta_m: Vec<K>,
    witness_count: usize,
}

/// Adversarial prover backend that replaces the NC column projection with
/// zero while preserving the exact accepted round shape and transcript flow.
struct ZeroNcProjectionBackend {
    d_sc: usize,
    eq_beta_m: Vec<K>,
    witness_count: usize,
}

struct InconsistentDeferredRowLogBackend {
    summary_coeffs: Vec<K>,
    exported_coeffs: Vec<K>,
}

impl FeSumcheckBackend for InconsistentDeferredRowLogBackend {
    fn start(&mut self, _snapshot: &RowPhaseSnapshot<'_>) -> bool {
        true
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        panic!("deferred-summary fixture must not request online coefficients")
    }

    fn fold(&mut self, _r: K) {
        panic!("deferred-summary fixture must not request an online fold")
    }

    fn row_round_summary_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
        initial_sum: K,
    ) -> Option<FeRowRoundSummary> {
        assert_eq!(rounds, 1, "the toy fixture must have one FE row-domain round");
        assert_eq!(
            neo_reductions::sumcheck::poly_eval_k(&self.summary_coeffs, K::ZERO)
                + neo_reductions::sumcheck::poly_eval_k(&self.summary_coeffs, K::ONE),
            initial_sum,
            "summary coefficients must be an honest first round"
        );

        let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_state, transcript_absorbed);
        transcript.append_fields_raw(&neo_reductions::sumcheck::round_coeff_fields(&self.summary_coeffs));
        let lanes = transcript.challenge_fields_raw(2);
        let challenge = neo_math::from_complex(lanes[0], lanes[1]);
        Some(FeRowRoundSummary {
            challenges: vec![challenge],
            sumcheck_final: neo_reductions::sumcheck::poly_eval_k(&self.summary_coeffs, challenge),
            transcript_after: Some((transcript.state(), transcript.absorbed())),
        })
    }

    fn export_row_rounds(&mut self) -> Option<Vec<Vec<K>>> {
        Some(vec![self.exported_coeffs.clone()])
    }
}

impl NcSumcheckBackend for OverlongZeroSuffixNcBackend {
    fn start(&mut self, snapshot: &NcColSnapshot<'_>) -> bool {
        self.eq_beta_m = snapshot.eq_beta_m_tbl.to_vec();
        self.witness_count = snapshot.weights.len();
        true
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.coeffs.clone()
    }

    fn fold(&mut self, _r: K) {}

    fn finalized_col_state(&mut self) -> NcFinalizedColState {
        panic!("the NC trace path returns its finalized state directly")
    }

    fn col_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<NcColRoundTrace> {
        assert_eq!(rounds, 1, "the toy fixture must have one NC column round");
        let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_state, transcript_absorbed);
        transcript.append_fields_raw(&neo_reductions::sumcheck::round_coeff_fields(&self.coeffs));
        let lanes = transcript.challenge_fields_raw(2);
        let challenge = neo_math::from_complex(lanes[0], lanes[1]);
        let eq_beta_m0 = self.eq_beta_m[0] + (self.eq_beta_m[1] - self.eq_beta_m[0]) * challenge;
        Some(NcColRoundTrace {
            coeffs: vec![self.coeffs.clone()],
            challenges: vec![challenge],
            transcript_after: Some((transcript.state(), transcript.absorbed())),
            finalized: NcFinalizedColState {
                digit_rows: vec![[K::ZERO; D]; self.witness_count],
                eq_beta_m0,
            },
        })
    }
}

impl NcSumcheckBackend for ZeroNcProjectionBackend {
    fn start(&mut self, snapshot: &NcColSnapshot<'_>) -> bool {
        self.eq_beta_m = if snapshot.eq_beta_m_tbl.is_empty() {
            neo_ccs::utils::tensor_point_parallel::<K>(snapshot.beta_m)
        } else {
            snapshot.eq_beta_m_tbl.to_vec()
        };
        self.witness_count = snapshot.weights.len();
        true
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        vec![K::ZERO; self.d_sc + 1]
    }

    fn fold(&mut self, challenge: K) {
        assert!(
            self.eq_beta_m.len() >= 2 && self.eq_beta_m.len() % 2 == 0,
            "NC equality table must have an even active length"
        );
        let half = self.eq_beta_m.len() / 2;
        for index in 0..half {
            let lo = self.eq_beta_m[2 * index];
            let hi = self.eq_beta_m[2 * index + 1];
            self.eq_beta_m[index] = lo + (hi - lo) * challenge;
        }
        self.eq_beta_m.truncate(half);
    }

    fn finalized_col_state(&mut self) -> NcFinalizedColState {
        assert_eq!(
            self.eq_beta_m.len(),
            1,
            "all NC column rounds must fold eq(beta_m) to one value"
        );
        NcFinalizedColState {
            digit_rows: vec![[K::ZERO; D]; self.witness_count],
            eq_beta_m0: self.eq_beta_m[0],
        }
    }
}

fn split_nc_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'a> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params reconstruction");
    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure()).expect("engine dims");
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &mat_digest,
    )
    .expect("header bundle");
    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure().into(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

fn pi_ccs_verifier_acceptance(
    prep: &neo_fold_clean::Preprocessing,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &pi_ccs::Proof,
) -> (bool, bool, bool) {
    let mut raw_transcript = Poseidon2Transcript::new(SESSION_LABEL);
    let raw_optimized = neo_fold_clean::engine::optimized::verify_pi_ccs(
        &mut raw_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        fresh_claims,
        running,
        &proof.outputs,
        &proof.sumcheck,
    )
    .unwrap_or(false);

    let mut clean_transcript = Transcript::session();
    let clean_native = pi_ccs::verify(
        &mut clean_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        fresh_claims,
        running,
        proof,
    )
    .is_ok();

    let recursive = catch_unwind(AssertUnwindSafe(|| {
        let mut builder = R1csBuilder::new();
        let mut transcript = TranscriptGadget::new(&mut builder, SESSION_LABEL);
        let cfg = split_nc_config(prep);
        enforce_split_nc_pi_ccs_v(
            &mut builder,
            &mut transcript,
            &cfg,
            &SplitNcPiCcsVMessages {
                fresh: fresh_claims,
                running: &running.claims,
                running_parent_authority: running.parent_authority.as_ref(),
                running_pending_projection: running.pending_projection(),
                variant: proof.sumcheck.variant,
                outputs: &proof.outputs,
                outputs_digest: proof.outputs_digest,
                sc_initial_sum: proof.sumcheck.sc_initial_sum,
                sumcheck_rounds_fe: &proof.sumcheck.sumcheck_rounds,
                sumcheck_rounds_nc: &proof.sumcheck.sumcheck_rounds_nc,
                header_digest: &proof.sumcheck.header_digest,
            },
        )
        .is_ok()
            && builder.is_satisfied()
    }))
    .unwrap_or(false);

    (raw_optimized, clean_native, recursive)
}

fn high_norm_fresh_instance() -> (neo_fold_clean::Preprocessing, CcsInstance) {
    let structure =
        CcsStructure::new(vec![Mat::identity(2)], SparsePoly::new(1, vec![])).expect("high-norm regression structure");
    let params = config::r1cs_params(structure.n, structure.m).expect("high-norm regression params");
    support::install_ajtai_module(&params, &structure);
    let prep = preprocess(params, structure, Some(1)).expect("high-norm regression preprocessing");

    let mut z = Mat::zero(D, prep.structure().m.div_ceil(D), F::ZERO);
    z[(1, 0)] = F::from_u64(prep.params.b() as u64);
    let instance = CcsInstance {
        claim: CcsClaim {
            adv: None,
            c: prep.log.commit(&z),
            x: vec![z[(0, 0)]],
            m_in: 1,
        },
        witness: CcsWitness {
            w: vec![z[(1, 0)]],
            Z: z,
        },
    };
    (prep, instance)
}

/// Fail-closed regression for NC projection laundering across a complete NIFS
/// fold.
///
/// The witness is honestly committed and satisfies the fixture's vacuous CCS
/// polynomial, but one private coordinate equals `b` and is therefore outside
/// the centered CE(b) alphabet. A prover-controlled backend replaces the
/// Pi_CCS NC projection and every NC round with zero. The public Pi_RLC and
/// Pi_DEC provers can then turn the same committed witness into honest
/// low-norm digit children whose terminal CE relations, including `y_zcol`,
/// are individually satisfied. The NIFS boundary must not both preserve those
/// child claims and accept their terminal witness authority: that would erase
/// the failed fresh-witness relation instead of folding it.
#[test]
#[ignore = "known NC y_zcol projection-authority gap"]
fn nifs_rejects_zero_nc_projection_laundered_into_valid_terminal_children() {
    let (prep, fresh) = high_norm_fresh_instance();
    assert_eq!(
        fresh.witness.Z[(1, 0)],
        F::from_u64(prep.params.b() as u64),
        "fixture coordinate must sit exactly outside the centered CE(b) alphabet"
    );

    let fresh_claims = vec![fresh.claim];
    let rlc_witnesses = vec![fresh.witness.Z.clone()];
    let fresh_witnesses = vec![fresh.witness];
    let running = RunningInstance::default();
    let mut backend = ZeroNcProjectionBackend {
        d_sc: split_nc_config(&prep).d_sc,
        eq_beta_m: Vec::new(),
        witness_count: 0,
    };
    let mut prover_transcript = Transcript::session();
    let Ok(pi_ccs_proof) = pi_ccs::prove_from_parts_with_backends(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        None,
        Some(&mut backend),
    ) else {
        return;
    };

    assert!(
        pi_ccs_proof
            .sumcheck
            .sumcheck_rounds_nc
            .iter()
            .flatten()
            .all(|&coefficient| coefficient == K::ZERO),
        "malicious fixture must emit the all-zero NC proof"
    );
    assert!(
        pi_ccs_proof
            .outputs
            .iter()
            .flat_map(|output| &output.y_zcol)
            .all(|&value| value == K::ZERO),
        "malicious fixture must replace every output NC projection with zero"
    );

    let Ok((rlc_out, pi_rlc_proof)) = pi_rlc::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &pi_ccs_proof.outputs,
        &rlc_witnesses,
    ) else {
        return;
    };
    let Ok((dec_out, pi_dec_proof)) = pi_dec::prove(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.combine_b_pows(),
        &rlc_out.claim,
        &rlc_out.witness,
    ) else {
        return;
    };

    assert!(
        dec_out
            .witnesses
            .iter()
            .flat_map(Mat::to_dense_vec)
            .all(|value| neo_math::balanced::within_nc_bound(value, prep.params.b())),
        "Pi_DEC must produce honest low-norm digit witnesses"
    );
    assert!(
        dec_out.claims.iter().all(|claim| !claim.y_zcol.is_empty())
            && dec_out
                .claims
                .iter()
                .flat_map(|claim| &claim.y_zcol)
                .any(|&value| value != K::ZERO),
        "terminal child y_zcol checks must be present and non-vacuous"
    );

    let next_running = RunningInstance {
        claims: dec_out.claims,
        witnesses: dec_out.witnesses,
        parent_authority: Some(rlc_out.claim),
    };
    let proof = NifsProof {
        pi_ccs: pi_ccs_proof,
        pi_rlc: pi_rlc_proof,
        pi_dec: pi_dec_proof,
    };

    let mut verifier_transcript = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    );
    let nifs_relation_preserved = verified.as_ref().is_ok_and(|verifier_next| {
        verifier_next.claims == next_running.claims && verifier_next.parent_authority == next_running.parent_authority
    });
    let terminal_authority_accepted =
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &next_running).is_ok();

    assert!(
        !(nifs_relation_preserved && terminal_authority_accepted),
        "known NC projection-authority gap: native NIFS preserved children laundered from an invalid fresh witness, and every honest low-norm child passed terminal CE authority including y_zcol"
    );
}

/// The native verifier defines an empty coefficient vector as the zero
/// polynomial. Exercise that encoding through an actual Π_CCS proof. The custom
/// backend emits the native verifier's accepted encoding of the zero
/// polynomial, so the recursive verifier must not reject or panic merely
/// because that encoding uses zero coefficients.
#[test]
fn recursive_pi_ccs_accepts_native_verified_empty_zero_round() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 1);
    let fresh_claims = vec![fresh.claim.clone()];
    let fresh_witnesses = vec![fresh.witness.clone()];
    let running = RunningInstance::default();

    let mut backend = EmptyZeroRowBackend;
    let mut prover_transcript = Transcript::session();
    let proof = pi_ccs::prove_from_parts_with_backends(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        Some(&mut backend),
        None,
    )
    .expect("construct Π_CCS proof with empty zero row");
    assert!(proof.sumcheck.sumcheck_rounds[0].is_empty());

    let mut native_transcript = Transcript::session();
    pi_ccs::verify(
        &mut native_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("native Π_CCS verifier accepts the proof");

    let circuit_result = catch_unwind(AssertUnwindSafe(|| {
        let mut builder = R1csBuilder::new();
        let mut transcript = TranscriptGadget::new(&mut builder, SESSION_LABEL);
        let cfg = split_nc_config(&prep);
        enforce_split_nc_pi_ccs_v(
            &mut builder,
            &mut transcript,
            &cfg,
            &SplitNcPiCcsVMessages {
                fresh: &fresh_claims,
                running: &running.claims,
                running_parent_authority: running.parent_authority.as_ref(),
                running_pending_projection: running.pending_projection(),
                variant: proof.sumcheck.variant,
                outputs: &proof.outputs,
                outputs_digest: proof.outputs_digest,
                sc_initial_sum: proof.sumcheck.sc_initial_sum,
                sumcheck_rounds_fe: &proof.sumcheck.sumcheck_rounds,
                sumcheck_rounds_nc: &proof.sumcheck.sumcheck_rounds_nc,
                header_digest: &proof.sumcheck.header_digest,
            },
        )
        .expect("recursive Π_CCS verifier should recognize the native proof");
        assert!(builder.is_satisfied(), "recursive Π_CCS verifier is unsatisfied");
    }));

    assert!(
        circuit_result.is_ok(),
        "completeness failure: recursive Π_CCS verification panicked on a proof accepted by native Π_CCS verification"
    );
}

/// Backend traces cross an implementation trust boundary but remain prover
/// inputs, not proof authority. Replay mode must reject FE and NC coefficient
/// vectors longer than the verifier's `d_sc + 1` limit before emitting a
/// proof. A trailing zero leaves the polynomial and every sumcheck identity
/// unchanged, so this specifically checks that prover and verifier agree on
/// the accepted coefficient-vector encoding.
#[test]
fn pi_ccs_prover_rejects_backend_round_above_coefficient_limit() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 91);
    let fresh_claims = vec![fresh.claim.clone()];
    let fresh_witnesses = vec![fresh.witness.clone()];
    let running = RunningInstance::default();

    let mut honest_transcript = Transcript::session();
    let honest = pi_ccs::prove_from_parts_with_backends(
        &mut honest_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        None,
        None,
    )
    .expect("honest Pi_CCS proof");

    let d_sc = split_nc_config(&prep).d_sc;
    let mut overlong_fe_coeffs = honest.sumcheck.sumcheck_rounds[0].clone();
    assert_eq!(
        overlong_fe_coeffs.len(),
        d_sc + 1,
        "canonical prover must expose the verifier's maximum coefficient count"
    );
    overlong_fe_coeffs.push(K::ZERO);

    let mut fe_backend = OverlongZeroSuffixRowBackend {
        coeffs: overlong_fe_coeffs,
    };
    let mut fe_prover_transcript = Transcript::session();
    let fe_produced = pi_ccs::prove_from_parts_with_backends(
        &mut fe_prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        Some(&mut fe_backend),
        None,
    );

    let mut overlong_nc_coeffs = honest.sumcheck.sumcheck_rounds_nc[0].clone();
    assert_eq!(
        overlong_nc_coeffs.len(),
        d_sc + 1,
        "canonical prover must expose the verifier's maximum NC coefficient count"
    );
    overlong_nc_coeffs.push(K::ZERO);
    let mut nc_backend = OverlongZeroSuffixNcBackend {
        coeffs: overlong_nc_coeffs,
        eq_beta_m: Vec::new(),
        witness_count: 0,
    };
    let mut nc_prover_transcript = Transcript::session();
    let nc_produced = pi_ccs::prove_from_parts_with_backends(
        &mut nc_prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        None,
        Some(&mut nc_backend),
    );

    let mut invalid_emissions = Vec::new();
    if let Ok(proof) = fe_produced {
        assert_eq!(proof.sumcheck.sumcheck_rounds[0].len(), d_sc + 2);
        let (raw, clean, recursive) = pi_ccs_verifier_acceptance(&prep, &fresh_claims, &running, &proof);
        if !(raw && clean && recursive) {
            invalid_emissions.push(format!(
                "FE trace: raw optimized={raw}, clean native={clean}, recursive={recursive}"
            ));
        }
    }
    if let Ok(proof) = nc_produced {
        assert_eq!(proof.sumcheck.sumcheck_rounds_nc[0].len(), d_sc + 2);
        let (raw, clean, recursive) = pi_ccs_verifier_acceptance(&prep, &fresh_claims, &running, &proof);
        if !(raw && clean && recursive) {
            invalid_emissions.push(format!(
                "NC trace: raw optimized={raw}, clean native={clean}, recursive={recursive}"
            ));
        }
    }

    assert!(
        invalid_emissions.is_empty(),
        "completeness failure: replay-mode Pi_CCS prover emitted proof(s) outside the verifier coefficient-length language: {}",
        invalid_emissions.join("; ")
    );
}

/// A deferred backend summary and its later proof-log export are two views of
/// one Fiat-Shamir execution. The prover must reject if the exported round no
/// longer reproduces the challenge, terminal sum, and transcript snapshot it
/// adopted from the summary. Otherwise the supported fast path can return a
/// proof that every verifier rejects even though both backend messages are
/// individually well-shaped.
#[test]
fn pi_ccs_deferred_prover_rejects_round_log_inconsistent_with_adopted_summary() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 97);
    let fresh_claims = vec![fresh.claim.clone()];
    let fresh_witnesses = vec![fresh.witness.clone()];
    let running = RunningInstance::default();

    let mut honest_transcript = Transcript::session();
    let honest = pi_ccs::prove_from_parts_with_backends(
        &mut honest_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        None,
        None,
    )
    .expect("honest Pi_CCS proof");
    let summary_coeffs = honest.sumcheck.sumcheck_rounds[0].clone();
    let mut exported_coeffs = summary_coeffs.clone();
    exported_coeffs[0] += K::ONE;

    let mut backend = InconsistentDeferredRowLogBackend {
        summary_coeffs,
        exported_coeffs,
    };
    let mut prover_transcript = Transcript::session();
    let deferred = pi_ccs::defer_from_parts_with_device_backends_and_transcript_mode(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        &fresh_claims,
        &fresh_witnesses,
        &running,
        &mut backend,
        None,
        BackendTranscriptMode::DeviceSnapshot,
        None,
        None,
    )
    .expect("summary-driven deferred Pi_CCS prove");
    let emitted = deferred.finish_with_fe_backend(&mut backend);

    let verifier_rejection = match emitted {
        Err(_) => None,
        Ok(proof) => {
            let (raw, clean, recursive) = pi_ccs_verifier_acceptance(&prep, &fresh_claims, &running, &proof);
            (!(raw && clean && recursive)).then_some((raw, clean, recursive))
        }
    };
    assert!(
        verifier_rejection.is_none(),
        "completeness failure: deferred Pi_CCS proof assembly accepted a round log that does not reproduce the adopted summary transcript; verifier acceptance (raw optimized, clean native, recursive) = {verifier_rejection:?}"
    );
}

/// Every serialized field in a proof must either be verified or absent from
/// the wire type. In particular, contradictory cached challenges and claimed
/// sumcheck results cannot be accepted as part of an otherwise valid proof:
/// downstream callers and alternate verifiers would observe a different
/// transcript than the production verifier actually checked.
#[test]
fn native_pi_ccs_rejects_contradictory_ignored_proof_fields() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 2);
    let fresh_claims = vec![fresh.claim.clone()];
    let running = RunningInstance::default();

    let mut prover_transcript = Transcript::session();
    let mut proof = pi_ccs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        vec![fresh],
        &running,
    )
    .expect("honest Pi_CCS proof");

    proof.sumcheck.sumcheck_challenges = vec![K::ONE];
    proof.sumcheck.sumcheck_challenges_nc = vec![K::ONE, K::ONE];
    proof.sumcheck.sc_initial_sum_nc = Some(K::ONE);
    proof.sumcheck.challenges_public.alpha = vec![K::ONE];
    proof.sumcheck.challenges_public.beta_a = vec![K::ONE];
    proof.sumcheck.challenges_public.beta_r = vec![K::ONE];
    proof.sumcheck.challenges_public.beta_m = vec![K::ONE];
    proof.sumcheck.challenges_public.gamma = K::ONE;
    proof.sumcheck.sumcheck_final = K::ONE;
    proof.sumcheck.sumcheck_final_nc = K::ONE;
    proof.sumcheck._extra = Some(vec![0x52, 0x45, 0x44]);

    let mut verifier_transcript = Transcript::session();
    let result = pi_ccs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof,
    );

    assert!(
        result.is_err(),
        "proof-language failure: native Pi_CCS accepted contradictory serialized challenges, claimed sums, and opaque extra bytes that it never verified"
    );
}

/// A proof crossing a serde boundary must have one encoding per algebraic
/// proof. `p3-goldilocks` 0.5 deserializes every raw `u64`, including `p+x`,
/// while field operations and transcript absorption treat it as `x`. Mutate a
/// coefficient that the verifier actually evaluates and absorbs, then require
/// rejection of the alternate byte encoding.
#[test]
fn native_pi_ccs_rejects_noncanonical_serialized_field_alias() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 3);
    let fresh_claims = vec![fresh.claim.clone()];
    let running = RunningInstance::default();

    let mut prover_transcript = Transcript::session();
    let mut proof = pi_ccs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        vec![fresh],
        &running,
    )
    .expect("honest Pi_CCS proof");

    let canonical_wire = bincode::serialize(&proof.sumcheck).expect("serialize canonical proof");
    let mutated = alias_one_verified_round_coefficient(&mut proof.sumcheck.sumcheck_rounds)
        || alias_one_verified_round_coefficient(&mut proof.sumcheck.sumcheck_rounds_nc);
    assert!(mutated, "toy proof needs one coefficient with a raw Goldilocks alias");

    let alternate_wire = bincode::serialize(&proof.sumcheck).expect("serialize alternate proof");
    assert_ne!(
        alternate_wire, canonical_wire,
        "the mutation must produce a distinct serialized proof"
    );
    proof.sumcheck = bincode::deserialize(&alternate_wire)
        .expect("the public proof deserializer currently accepts noncanonical field words");

    let mut verifier_transcript = Transcript::session();
    let result = pi_ccs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof,
    );

    assert!(
        result.is_err(),
        "proof-malleability failure: native Pi_CCS accepted a distinct serialized proof obtained by replacing a verified Goldilocks limb x with p+x"
    );
}

#[test]
fn recursive_challenge_sampler_rejects_domain_length_overflow_without_panicking() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut builder = R1csBuilder::new();
        let mut transcript = TranscriptGadget::new(&mut builder, b"redteam/recursive-challenge-domain-overflow");
        let _ = sample_engine_challenges(&mut builder, &mut transcript, usize::MAX, 1);
    }));

    assert!(
        result.is_ok(),
        "invalid recursive challenge dimensions must be rejected without panicking"
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn recursive_fe_claimed_initial_rejects_gamma_table_size_overflow_without_panicking() {
    let mut builder = R1csBuilder::new();
    let gamma = KVar::alloc(&mut builder, F::ONE, F::ZERO);
    let result = catch_unwind(AssertUnwindSafe(|| {
        enforce_fe_claimed_initial(
            &mut builder,
            &FeClaimedInitialInputs {
                k_mcs: 2,
                t: 1usize << 63,
                ell_d: 0,
                gamma,
                alpha: &[],
                running_y_ring: &[],
            },
        )
    }));

    assert!(
        result.is_ok(),
        "invalid recursive FE dimensions must be rejected without panicking"
    );
    assert!(
        result.unwrap().is_err(),
        "overflowing recursive FE dimensions must return a shape error"
    );
}
