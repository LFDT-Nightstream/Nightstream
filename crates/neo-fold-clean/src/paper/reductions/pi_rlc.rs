//! Native Pi_RLC reduction from CE inputs to one combined CE parent.
//!
//! Owns: input/combined validation, rho transcript scheduling, projection
//! binding data, and prover/verifier orchestration.
//!
//! Does not own: the RLC norm-bound theorem, engine ring mixing, Pi_DEC, or
//! in-circuit verification.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the prover-supplied combined claim is checked against
//! verifier-derived rho values and the authenticated inputs before Pi_DEC uses
//! it; projection digests are compression, not authority.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Rho schedule | [`begin_rho_sampling`], [`derive_rhos_for_inputs`] | no | Transcript-bound Pi_CCS outputs |
//! | Combined claim | [`validate_combined`] | no | Verifier recomputation from authenticated inputs |
//! | Prove/verify | [`prove`], [`verify`] | no | Engine Pi_RLC relation and checked transcript |

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_math::field::KExtensions;
use neo_math::{D, F, K};
use thiserror::Error;

use crate::engine::optimized as engine;
use crate::engine::r1cs_circuit::ring_action::{projection_quotient, PROJECTION_QUOTIENT_LEN};
use crate::engine::transcript::{Poseidon2TranscriptSnapshot, Transcript};
use crate::paper::digest;
use crate::paper::params::Params;
use crate::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest as sis_accumulator_digest, SisAccumulatorError, PI_RLC_PROJECTION_SIS_CONFIG,
};
use crate::paper::reductions::pi_rlc_circuit::rlc_projection_quotients;
use crate::paper::relations::{superneo_inactive_x_zero, validate_adv_shape, CeClaim, RlcMixer};
use crate::paper::sampling::check_rlc_bound;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

pub const PI_RLC_INPUT_CLAIMS_DIGEST_LABEL: &[u8] = b"pi_rlc/input_claims_digest";
pub(crate) const PI_RLC_PROJECTION_COMBINED_C_LABEL: &[u8] = b"pi_rlc/projection_combined_c";
pub(crate) const PI_RLC_PROJECTION_QUOTIENTS_LABEL: &[u8] = b"pi_rlc/projection_quotients";
pub(crate) const PI_RLC_PROJECTION_COMBINED_ADV_LABEL: &[u8] = b"pi_rlc/projection_combined_adv";
pub(crate) const PI_RLC_PROJECTION_ADV_QUOTIENTS_LABEL: &[u8] = b"pi_rlc/projection_adv_quotients";
pub(crate) const PI_RLC_PROJECTION_COMBINED_X_LABEL: &[u8] = b"pi_rlc/projection_combined_x";
pub(crate) const PI_RLC_PROJECTION_X_QUOTIENTS_LABEL: &[u8] = b"pi_rlc/projection_x_quotients";
pub(crate) const PI_RLC_PROJECTION_COMBINED_Y_RING_LABEL: &[u8] = b"pi_rlc/projection_combined_y_ring";
pub(crate) const PI_RLC_PROJECTION_Y_RING_QUOTIENTS_LABEL: &[u8] = b"pi_rlc/projection_y_ring_quotients";
pub(crate) const PI_RLC_PROJECTION_COMBINED_Y_ZCOL_LABEL: &[u8] = b"pi_rlc/projection_combined_y_zcol";
pub(crate) const PI_RLC_PROJECTION_Y_ZCOL_QUOTIENTS_LABEL: &[u8] = b"pi_rlc/projection_y_zcol_quotients";
pub(crate) const PI_RLC_PROJECTION_BINDING_DOMAIN: &[u8] = b"neo.fold.clean/pi_rlc/projection_binding/v1";
pub(crate) const PI_RLC_PROJECTION_BINDING_DIGEST_LABEL: &[u8] = b"pi_rlc/projection_binding_digest";
pub(crate) const PI_RLC_PROJECTION_BETA_LABEL: &[u8] = b"pi_rlc/projection_beta";

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_RLC: input claims must be \u{2265} 1 (got 0)")]
    Shape,
    #[error("\u{03A0}_RLC: |claims| ({claims}) \u{2260} |witnesses| ({witnesses})")]
    WitnessMismatch { claims: usize, witnesses: usize },
    #[error("\u{03A0}_RLC: verifier rejected the prover's combined CE claim")]
    VerifyRejected,
    #[error("\u{03A0}_RLC: combined fold_digest must match every input fold_digest")]
    FoldDigest,
    #[error("\u{03A0}_RLC: noncanonical fold_digest byte limb in {owner} at lane {lane}")]
    FoldDigestCanonicality { owner: &'static str, lane: usize },
    #[error("\u{03A0}_RLC: inactive X columns must be zero in {0}")]
    InactiveX(&'static str),
    #[error("\u{03A0}_RLC: r length must match the SplitNc row point in {0}")]
    RShape(&'static str),
    #[error("\u{03A0}_RLC: combined r must match every input r")]
    RConsistency,
    #[error("\u{03A0}_RLC: combined s_col must match every input s_col")]
    SColConsistency,
    #[error("\u{03A0}_RLC: s_col length must match the SplitNc column point in {0}")]
    SColShape(&'static str),
    #[error("\u{03A0}_RLC: combined y_zcol must equal the RLC of input y_zcol values")]
    YZcolConsistency,
    #[error("\u{03A0}_RLC: adv presence must be all-or-nothing across inputs ({present}/{total} present)")]
    AdvPresence { present: usize, total: usize },
    #[error("\u{03A0}_RLC: combined adv must equal the component-wise RLC of input adv tuples")]
    AdvConsistency,
    #[error("\u{03A0}_RLC: invalid product-commitment shape: {0}")]
    AdvShape(String),
    #[error("\u{03A0}_RLC: y_zcol padding lanes must be zero in {0}")]
    YZcolPadding(&'static str),
    #[error("\u{03A0}_RLC: cached ct must equal the constant term of y_ring in {0}")]
    CtConsistency(&'static str),
    #[error("\u{03A0}_RLC: y_ring shape must match the padded SplitNc ring shape in {0}")]
    YRingShape(&'static str),
    #[error("\u{03A0}_RLC: y_ring padding lanes must be zero in {0}")]
    YRingPadding(&'static str),
    #[error("\u{03A0}_RLC: unsupported sidecar field {field} in {owner}")]
    UnsupportedSidecar {
        owner: &'static str,
        field: &'static str,
    },
    #[error(
        "\u{03A0}_RLC: projection schedule — combined commitment lane {lane} is not the ring-action mix of the inputs"
    )]
    ProjectionMixDrift { lane: usize },
    #[error("Pi_RLC: projection schedule - combined adv.{coordinate} lane {lane} is not the ring-action mix")]
    AdvProjectionMixDrift {
        coordinate: &'static str,
        lane: usize,
    },
    #[error("Pi_RLC: projection schedule - combined {client} identity {identity} is not the ring-action mix")]
    AuxiliaryProjectionMixDrift {
        client: &'static str,
        identity: usize,
    },
    #[error("Pi_RLC: accelerator failed to compute the canonical projection SIS digest")]
    BackendProjectionDigest,
    #[error(transparent)]
    Projection(#[from] crate::paper::reductions::pi_rlc_circuit::Error),
    #[error(transparent)]
    SisAccumulator(#[from] SisAccumulatorError),
    #[error(transparent)]
    Sampling(#[from] crate::paper::sampling::SamplingError),
    #[error(transparent)]
    Engine(#[from] engine::Error),
}

/// Output of one Π_RLC step — a single CE claim of norm B plus its
/// combined witness `Z_mix = Σρ_i Z_i`. Witness is prover-only; the
/// verifier reconstructs only the claim.
#[derive(Clone, Debug)]
pub struct Output {
    pub claim: CeClaim,
    pub witness: Mat<F>,
    /// The Lemma 5 β schedule this fold ran (Road A candidate E) —
    /// prover-side plumbing for the F' image's projection regions.
    pub projection: ProjectionSchedule,
}

/// Post-mix β schedule for the projection-checked commitment
/// combination (encoding.md candidate E; security-note Lemma 5 §4b).
/// Nothing here rides the wire: the verifier recomputes every field
/// from ρ and the input commitments, so a carried value can never
/// out-vote the transcript.
#[derive(Clone, Debug)]
pub struct ProjectionSchedule {
    /// ρ_i ring elements (rotation-matrix first columns), fold order.
    pub rhos: Vec<[F; D]>,
    /// Per-κ-lane division quotients `q_lane` with
    /// `Σ_i ρ_i(X)·c_{i,lane}(X) = q_lane(X)·Φ(X) + combined_lane(X)`.
    pub q_lanes: Vec<[F; PROJECTION_QUOTIENT_LEN]>,
    /// Projection quotients for the `(ops, is, fs)` coordinates of `L+`.
    pub adv_q_lanes: Option<LaneCommitments<Vec<[F; PROJECTION_QUOTIENT_LEN]>>>,
    /// One quotient per active X ring column.
    pub x_q_lanes: Vec<[F; PROJECTION_QUOTIENT_LEN]>,
    /// Two quotients (c0, c1) per y_ring row.
    pub y_ring_q_lanes: Vec<[[F; PROJECTION_QUOTIENT_LEN]; 2]>,
    /// Two quotients (c0, c1) for y_zcol.
    pub y_zcol_q_lanes: [[F; PROJECTION_QUOTIENT_LEN]; 2],
    /// The evaluation challenge, squeezed after c* and every `q_lane`
    /// are on the transcript — the order is the soundness (Lemma 5).
    pub beta: K,
}

/// Wire-format proof: the prover's combined CE claim of norm B.
///
/// The ρ-rotation challenges are not serialized here; prover and verifier
/// both resample them from the Fiat-Shamir transcript at this phase.
#[derive(Clone, Debug)]
pub struct Proof {
    pub combined: CeClaim,
}

// ──────────────────────────────────────────────────────────────────────────
// Prover  (§7.4 step order)
// ──────────────────────────────────────────────────────────────────────────

pub fn prove(
    tr: &mut Transcript,
    pp: &Params,
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    witnesses: &[Mat<F>],
) -> Result<(Output, Proof), Error> {
    let witness_refs: Vec<&Mat<F>> = witnesses.iter().collect();
    prove_refs(tr, pp, s, mix, claims, &witness_refs)
}

pub(crate) fn prove_refs(
    tr: &mut Transcript,
    pp: &Params,
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    witnesses: &[&Mat<F>],
) -> Result<(Output, Proof), Error> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let prepare_started = std::time::Instant::now();
    validate_input_shape(claims, witnesses)?;
    validate_inputs_before_rho(s, claims)?;
    let rhos = derive_rhos_for_inputs(tr, pp, claims)?;
    #[cfg(feature = "perf-timers")]
    let prepare_elapsed = prepare_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let mix_started = std::time::Instant::now();
    let (mut combined, z_mix) = engine::prove_pi_rlc_refs(pp, s, &rhos, claims, witnesses, |zs, cs| mix(zs, cs))?;
    #[cfg(feature = "perf-timers")]
    let mix_elapsed = mix_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let sidecars_started = std::time::Instant::now();
    combined.adv = mixed_adv(mix, &rhos, claims)?;
    validate_nc_sidecars(s, mix, &rhos, claims, &combined)?;
    #[cfg(feature = "perf-timers")]
    let sidecars_elapsed = sidecars_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let projection_started = std::time::Instant::now();
    let projection = projection_schedule(tr, &rhos, claims, &combined)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-rlc/prove] prepare={:.3}s mix={:.3}s sidecars={:.3}s projection={:.3}s total={:.3}s inputs={} c_lanes={} adv={} X={} y={} yz={}",
        prepare_elapsed.as_secs_f64(),
        mix_elapsed.as_secs_f64(),
        sidecars_elapsed.as_secs_f64(),
        projection_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
        claims.len(),
        combined.c.kappa,
        combined.adv.is_some(),
        combined.X.cols(),
        combined.y_ring.len(),
        combined.y_zcol.len(),
    );
    Ok((
        Output {
            claim: combined.clone(),
            witness: z_mix,
            projection,
        },
        Proof { combined },
    ))
}

/// The prover's pre-ρ input validations (`prove_refs` line one), exposed for
/// NIFS backends that run Π_RLC's claim algebra outside `prove`. Call before
/// deriving ρ so a malformed input fails with the CPU error surface and an
/// untouched transcript.
pub fn validate_inputs(s: &crate::paper::relations::Structure, claims: &[CeClaim]) -> Result<(), Error> {
    validate_inputs_before_rho(s, claims)
}

/// The prover's post-combine claim validations, for the same backends.
/// Transcript-neutral; touches only claims, never witnesses.
pub fn validate_combined(
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    rhos: &[neo_reductions::common::RotRho],
    claims: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    validate_nc_sidecars(s, mix, rhos, claims, combined)
}

// ──────────────────────────────────────────────────────────────────────────
// Verifier (§7.4)
// ──────────────────────────────────────────────────────────────────────────

/// Verify the prover's combined CE claim against `Σρ_i · u_i` recomputed
/// from `(transcript, claims)`. Returns the verified parent for Π_DEC.
pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    proof: &Proof,
) -> Result<CeClaim, Error> {
    validate_inputs_before_rho(s, claims)?;
    let rhos = derive_rhos_for_inputs(tr, pp, claims)?;
    validate_nc_sidecars(s, mix, &rhos, claims, &proof.combined)?;
    let ok = engine::verify_pi_rlc(pp, s, &rhos, claims, &proof.combined, |zs, cs| mix(zs, cs))?;
    if !ok {
        return Err(Error::VerifyRejected);
    }
    projection_schedule(tr, &rhos, claims, &proof.combined)?;
    Ok(proof.combined.clone())
}

/// Bind the Π_CCS output claims and derive Π_RLC rotation challenges.
///
/// The caller must pass a transcript that is already at the Π_RLC phase, i.e.
/// after Π_CCS has produced and bound its output claims.
pub fn derive_rhos_for_inputs(
    tr: &mut Transcript,
    pp: &Params,
    claims: &[CeClaim],
) -> Result<Vec<neo_reductions::common::RotRho>, Error> {
    let (rhos, _) = derive_rhos_for_inputs_with_sampling_start(tr, pp, claims)?;
    Ok(rhos)
}

/// Bind Π_CCS output claims, capture the exact rho-sampling transcript start,
/// then derive Π_RLC rotation challenges.
///
/// CUDA backends use the returned snapshot to reproduce the same rho matrices
/// on device without treating copied rho buffers as protocol authority.
pub fn derive_rhos_for_inputs_with_sampling_start(
    tr: &mut Transcript,
    pp: &Params,
    claims: &[CeClaim],
) -> Result<(Vec<neo_reductions::common::RotRho>, Poseidon2TranscriptSnapshot), Error> {
    let sampling_start = begin_rho_sampling(tr, pp, claims)?;
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    Ok((rhos, sampling_start))
}

/// Bind the Π_CCS output claims and return the transcript snapshot from
/// which Π_RLC rho sampling starts.
///
/// CUDA backends use this to let the device transcript own the rho sampling
/// loop while the host transcript remains authoritative for the public claim
/// digest bind.
pub fn begin_rho_sampling(
    tr: &mut Transcript,
    pp: &Params,
    claims: &[CeClaim],
) -> Result<Poseidon2TranscriptSnapshot, Error> {
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    begin_rho_sampling_from_outputs_digest(tr, pp, claims.len(), digest::pi_ccs_outputs_digest(claims))
}

/// Bind an already-computed Π_CCS output digest and return the transcript
/// snapshot from which Π_RLC rho sampling starts.
///
/// This is the compact handoff used by GPU-oriented provers: Π_CCS owns
/// producing the output surface and its digest; Π_RLC owns sampling `ρ` from
/// that canonical digest. The digest is never verifier authority — it is
/// recomputed from `pi_ccs::Proof::outputs` before verification accepts.
pub fn begin_rho_sampling_from_outputs_digest(
    tr: &mut Transcript,
    pp: &Params,
    claim_count: usize,
    outputs_digest: [F; 4],
) -> Result<Poseidon2TranscriptSnapshot, Error> {
    validate_rho_sampling_count(pp, claim_count)?;
    bind_outputs_digest_for_rho(tr, outputs_digest);
    Ok(tr.snapshot())
}

/// Validate the public Π_RLC sampling shape without mutating a transcript.
/// Device-owned fold transcripts call this before binding a resident Π_CCS
/// output digest with the same canonical label and field count.
pub fn validate_rho_sampling_count(pp: &Params, claim_count: usize) -> Result<(), Error> {
    if claim_count == 0 {
        return Err(Error::Shape);
    }
    enforce_rlc_bound(pp, claim_count)
}

fn bind_outputs_digest_for_rho(tr: &mut Transcript, outputs_digest: [F; 4]) {
    tr.append_fields(PI_RLC_INPUT_CLAIMS_DIGEST_LABEL, &outputs_digest);
}

/// Lemma 5 transcript schedule, shared verbatim by `prove` and
/// `verify`: with ρ sampled and the mix fixed, recompute the per-lane
/// quotients from the input commitments, absorb the combined
/// commitment and every quotient, then squeeze β. Also discharges the
/// wire-identity obligation (Lemma 5 audit item 1): the mix the
/// quotients divide against must BE the combined commitment, lane for
/// lane — so the projection algebra can never drift from the mixer the
/// fold actually used.
fn projection_schedule(
    tr: &mut Transcript,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<ProjectionSchedule, Error> {
    projection_schedule_with_digest(tr, rhos, inputs, combined, |preimage| {
        Ok(sis_accumulator_digest(PI_RLC_PROJECTION_SIS_CONFIG, preimage)?)
    })
}

fn projection_schedule_with_digest(
    tr: &mut Transcript,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
    compute_digest: impl FnOnce(&[F]) -> Result<[F; 4], Error>,
) -> Result<ProjectionSchedule, Error> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    let mut binding_preimage = digest::pack_bytes_as_fields(PI_RLC_PROJECTION_BINDING_DOMAIN);
    let rho_coeffs: Vec<[F; D]> = rhos
        .iter()
        .map(|rho| {
            let mat = rho.as_mat();
            core::array::from_fn(|row| mat[(row, 0)])
        })
        .collect();
    let input_cs: Vec<Commitment> = inputs.iter().map(|claim| claim.c.clone()).collect();
    #[cfg(feature = "perf-timers")]
    let commitment_started = std::time::Instant::now();
    let lanes = checked_projection_lanes(&rho_coeffs, &input_cs, &combined.c, None)?;
    append_projection_binding(
        &mut binding_preimage,
        PI_RLC_PROJECTION_COMBINED_C_LABEL,
        &combined.c.data,
    );
    for lane in &lanes {
        append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_QUOTIENTS_LABEL, &lane.q);
    }
    #[cfg(feature = "perf-timers")]
    let commitment_elapsed = commitment_started.elapsed();

    #[cfg(feature = "perf-timers")]
    let adv_started = std::time::Instant::now();
    let present = inputs.iter().filter(|claim| claim.adv.is_some()).count();
    let adv_lanes = match (&combined.adv, present) {
        (None, 0) => None,
        (Some(_), 0) | (None, _) => return Err(Error::AdvConsistency),
        (Some(combined_adv), count) if count == inputs.len() => {
            let input_advs: Vec<&LaneCommitments<Commitment>> = inputs
                .iter()
                .map(|claim| claim.adv.as_ref().unwrap())
                .collect();
            let coordinate =
                |select: fn(&LaneCommitments<Commitment>) -> &Commitment,
                 combined_coordinate: &Commitment,
                 name: &'static str|
                 -> Result<Vec<crate::paper::reductions::pi_rlc_circuit::RlcLaneProjection>, Error> {
                    let commitments: Vec<Commitment> = input_advs.iter().map(|adv| select(adv).clone()).collect();
                    checked_projection_lanes(&rho_coeffs, &commitments, combined_coordinate, Some(name))
                };
            let ops = coordinate(|adv| &adv.ops, &combined_adv.ops, "ops")?;
            let is = coordinate(|adv| &adv.is, &combined_adv.is, "is")?;
            let fs = coordinate(|adv| &adv.fs, &combined_adv.fs, "fs")?;

            for leaf in digest::nebula_lane_leaf_digests(combined_adv) {
                append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_COMBINED_ADV_LABEL, &leaf);
            }
            for lane in ops.iter().chain(is.iter()).chain(fs.iter()) {
                append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_ADV_QUOTIENTS_LABEL, &lane.q);
            }
            Some(LaneCommitments { ops, is, fs })
        }
        (Some(_), count) => {
            return Err(Error::AdvPresence {
                present: count,
                total: inputs.len(),
            })
        }
    };
    #[cfg(feature = "perf-timers")]
    let adv_elapsed = adv_started.elapsed();

    #[cfg(feature = "perf-timers")]
    let x_started = std::time::Instant::now();
    let active_x_cols = crate::paper::relations::superneo_public_x_cols(combined.m_in);
    let mut x_lanes = Vec::with_capacity(active_x_cols);
    for col in 0..active_x_cols {
        let input_coeffs: Vec<[F; D]> = inputs
            .iter()
            .map(|claim| core::array::from_fn(|row| claim.X[(row, col)]))
            .collect();
        let output = core::array::from_fn(|row| combined.X[(row, col)]);
        let lane = checked_auxiliary_projection(&rho_coeffs, &input_coeffs, output, "X", col)?;
        append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_COMBINED_X_LABEL, &output);
        append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_X_QUOTIENTS_LABEL, &lane.q);
        x_lanes.push(lane);
    }
    #[cfg(feature = "perf-timers")]
    let x_elapsed = x_started.elapsed();

    #[cfg(feature = "perf-timers")]
    let y_ring_started = std::time::Instant::now();
    let mut y_ring_lanes = Vec::with_capacity(combined.y_ring.len());
    for row in 0..combined.y_ring.len() {
        let input_rows: Vec<&[K]> = inputs
            .iter()
            .map(|claim| claim.y_ring[row].as_slice())
            .collect();
        let lanes = checked_k_vector_projection(&rho_coeffs, &input_rows, &combined.y_ring[row], "y_ring", 2 * row)?;
        for lane in &lanes {
            append_projection_binding(
                &mut binding_preimage,
                PI_RLC_PROJECTION_COMBINED_Y_RING_LABEL,
                &lane.out,
            );
            append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_Y_RING_QUOTIENTS_LABEL, &lane.q);
        }
        y_ring_lanes.push(lanes);
    }
    #[cfg(feature = "perf-timers")]
    let y_ring_elapsed = y_ring_started.elapsed();

    #[cfg(feature = "perf-timers")]
    let y_zcol_started = std::time::Instant::now();
    let input_y_zcols: Vec<&[K]> = inputs.iter().map(|claim| claim.y_zcol.as_slice()).collect();
    let y_zcol_lanes = checked_k_vector_projection(&rho_coeffs, &input_y_zcols, &combined.y_zcol, "y_zcol", 0)?;
    for lane in &y_zcol_lanes {
        append_projection_binding(
            &mut binding_preimage,
            PI_RLC_PROJECTION_COMBINED_Y_ZCOL_LABEL,
            &lane.out,
        );
        append_projection_binding(&mut binding_preimage, PI_RLC_PROJECTION_Y_ZCOL_QUOTIENTS_LABEL, &lane.q);
    }
    #[cfg(feature = "perf-timers")]
    let y_zcol_elapsed = y_zcol_started.elapsed();

    #[cfg(feature = "perf-timers")]
    let binding_started = std::time::Instant::now();
    let binding_digest = compute_digest(&binding_preimage)?;
    tr.append_fields(PI_RLC_PROJECTION_BINDING_DIGEST_LABEL, &binding_digest);
    let beta = tr.challenge_fields(PI_RLC_PROJECTION_BETA_LABEL, 2);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-rlc/projection] c={:.3}s adv={:.3}s X={:.3}s y={:.3}s yz={:.3}s sis+beta={:.3}s total={:.3}s identities=c:{} adv:{} X:{} y:{} yz:{} preimage_fields={}",
        commitment_elapsed.as_secs_f64(),
        adv_elapsed.as_secs_f64(),
        x_elapsed.as_secs_f64(),
        y_ring_elapsed.as_secs_f64(),
        y_zcol_elapsed.as_secs_f64(),
        binding_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
        lanes.len(),
        adv_lanes.as_ref().map_or(0, |adv| adv.ops.len() + adv.is.len() + adv.fs.len()),
        x_lanes.len(),
        y_ring_lanes.len() * 2,
        y_zcol_lanes.len(),
        binding_preimage.len(),
    );
    Ok(ProjectionSchedule {
        rhos: rho_coeffs,
        q_lanes: lanes.into_iter().map(|lane| lane.q).collect(),
        adv_q_lanes: adv_lanes.map(|lanes| LaneCommitments {
            ops: lanes.ops.into_iter().map(|lane| lane.q).collect(),
            is: lanes.is.into_iter().map(|lane| lane.q).collect(),
            fs: lanes.fs.into_iter().map(|lane| lane.q).collect(),
        }),
        x_q_lanes: x_lanes.into_iter().map(|lane| lane.q).collect(),
        y_ring_q_lanes: y_ring_lanes
            .into_iter()
            .map(|lanes| lanes.map(|lane| lane.q))
            .collect(),
        y_zcol_q_lanes: y_zcol_lanes.map(|lane| lane.q),
        beta: K::from_coeffs([beta[0], beta[1]]),
    })
}

/// Replay the post-rho projection binding for a backend-produced combined
/// claim. The returned schedule is prover metadata; verifiers recompute it
/// from the public inputs and transcript.
pub fn bind_backend_projection_schedule(
    tr: &mut Transcript,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<ProjectionSchedule, Error> {
    projection_schedule(tr, rhos, inputs, combined)
}

/// Replay the canonical projection checks while delegating only the final
/// fixed-seed SIS compression to an accelerator. The callback sees the exact
/// preimage used by the native prover; the verifier independently rebuilds
/// and compresses it before accepting the proof.
#[doc(hidden)]
pub fn bind_backend_projection_schedule_with_digest(
    tr: &mut Transcript,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
    compute_digest: impl FnOnce(&[F]) -> Result<[F; 4], Error>,
) -> Result<ProjectionSchedule, Error> {
    projection_schedule_with_digest(tr, rhos, inputs, combined, compute_digest)
}

fn append_projection_binding(preimage: &mut Vec<F>, label: &[u8], fields: &[F]) {
    preimage.extend(digest::pack_bytes_as_fields(label));
    preimage.push(F::from_u64(fields.len() as u64));
    preimage.extend_from_slice(fields);
}

fn checked_auxiliary_projection(
    rho_coeffs: &[[F; D]],
    inputs: &[[F; D]],
    combined: [F; D],
    client: &'static str,
    identity: usize,
) -> Result<crate::paper::reductions::pi_rlc_circuit::RlcLaneProjection, Error> {
    if rho_coeffs.len() != inputs.len() {
        return Err(Error::Shape);
    }
    let pairs: Vec<([F; D], [F; D])> = rho_coeffs
        .iter()
        .copied()
        .zip(inputs.iter().copied())
        .collect();
    let (out, q) = projection_quotient(&pairs);
    if out != combined {
        return Err(Error::AuxiliaryProjectionMixDrift { client, identity });
    }
    Ok(crate::paper::reductions::pi_rlc_circuit::RlcLaneProjection { out, q })
}

fn checked_k_vector_projection(
    rho_coeffs: &[[F; D]],
    inputs: &[&[K]],
    combined: &[K],
    client: &'static str,
    identity_start: usize,
) -> Result<[crate::paper::reductions::pi_rlc_circuit::RlcLaneProjection; 2], Error> {
    if combined.len() < D || inputs.iter().any(|input| input.len() < D) {
        return Err(Error::Shape);
    }
    let input_c0: Vec<[F; D]> = inputs
        .iter()
        .map(|input| core::array::from_fn(|i| input[i].as_coeffs()[0]))
        .collect();
    let input_c1: Vec<[F; D]> = inputs
        .iter()
        .map(|input| core::array::from_fn(|i| input[i].as_coeffs()[1]))
        .collect();
    let combined_c0 = core::array::from_fn(|i| combined[i].as_coeffs()[0]);
    let combined_c1 = core::array::from_fn(|i| combined[i].as_coeffs()[1]);
    Ok([
        checked_auxiliary_projection(rho_coeffs, &input_c0, combined_c0, client, identity_start)?,
        checked_auxiliary_projection(rho_coeffs, &input_c1, combined_c1, client, identity_start + 1)?,
    ])
}

fn checked_projection_lanes(
    rho_coeffs: &[[F; D]],
    inputs: &[Commitment],
    combined: &Commitment,
    coordinate: Option<&'static str>,
) -> Result<Vec<crate::paper::reductions::pi_rlc_circuit::RlcLaneProjection>, Error> {
    let lanes = rlc_projection_quotients(rho_coeffs, inputs)?;
    for (lane_idx, lane) in lanes.iter().enumerate() {
        if combined.data.get(lane_idx * D..(lane_idx + 1) * D) != Some(&lane.out[..]) {
            return match coordinate {
                None => Err(Error::ProjectionMixDrift { lane: lane_idx }),
                Some(coordinate) => Err(Error::AdvProjectionMixDrift {
                    coordinate,
                    lane: lane_idx,
                }),
            };
        }
    }
    Ok(lanes)
}

// ──────────────────────────────────────────────────────────────────────────
// Step bodies
// ──────────────────────────────────────────────────────────────────────────

fn validate_inputs_before_rho(s: &crate::paper::relations::Structure, inputs: &[CeClaim]) -> Result<(), Error> {
    for input in inputs {
        validate_adv_shape(input.adv.as_ref(), input.c.d, input.c.kappa, "input").map_err(Error::AdvShape)?;
        validate_fold_digest_canonical("input", input)?;
        validate_inactive_x_zero_one("input", input)?;
        validate_r_shape_one("input", s, input)?;
        validate_y_ring_shape_one("input", s, input)?;
        validate_y_ring_padding_zero_one("input", input)?;
        validate_ct_consistency_one("input", input)?;
        validate_s_col_shape_one("input", s, input)?;
        validate_y_zcol_shape_padding_one("input", input)?;
        validate_supported_sidecars_one("input", input)?;
    }
    Ok(())
}

fn validate_input_shape(claims: &[CeClaim], witnesses: &[&Mat<F>]) -> Result<(), Error> {
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    if claims.len() != witnesses.len() {
        return Err(Error::WitnessMismatch {
            claims: claims.len(),
            witnesses: witnesses.len(),
        });
    }
    Ok(())
}

/// Definition 14 norm bound: `count · T · (b−1) < B`. Fails loudly here
/// so the caller cannot reach the engine with a count that violates it.
fn enforce_rlc_bound(pp: &Params, count: usize) -> Result<(), Error> {
    check_rlc_bound(pp, count, pp.T() as u128).map_err(Into::into)
}

fn validate_nc_sidecars(
    s: &crate::paper::relations::Structure,
    mix: RlcMixer,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    validate_adv_shape(combined.adv.as_ref(), combined.c.d, combined.c.kappa, "combined").map_err(Error::AdvShape)?;
    validate_fold_digest_canonical("combined", combined)?;
    validate_inactive_x_zero(inputs, combined)?;
    validate_r_shape(s, inputs, combined)?;
    validate_r_consistency(inputs, combined)?;
    validate_y_ring_shape(s, inputs, combined)?;
    validate_y_ring_padding_zero(inputs, combined)?;
    validate_ct_consistency(inputs, combined)?;
    validate_s_col_shape(s, inputs, combined)?;
    validate_s_col_consistency(inputs, combined)?;
    validate_y_zcol_combination(rhos, inputs, combined)?;
    validate_adv_combination(mix, rhos, inputs, combined)?;
    validate_fold_digest_consistency(inputs, combined)?;
    validate_supported_sidecars(inputs, combined)?;
    Ok(())
}

/// Spec §5.2 R2 (Π_RLC side): the combined claim's `adv` must equal the
/// component-wise ρ-mix of the input tuples — the same public arithmetic
/// that combines `c`, recomputed here on both prove and verify paths.
fn validate_adv_combination(
    mix: RlcMixer,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    let expected = mixed_adv(mix, rhos, inputs)?;
    if combined.adv != expected {
        return Err(Error::AdvConsistency);
    }
    Ok(())
}

/// Component-wise ρ-mix of the inputs' `adv` tuples (`None` for a plain
/// fold; all-or-nothing presence enforced).
fn mixed_adv(
    mix: RlcMixer,
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
) -> Result<Option<neo_ccs::LaneCommitments<neo_ajtai::Commitment>>, Error> {
    let rho_mats: Vec<Mat<F>> = rhos.iter().map(|rho| rho.as_mat().clone()).collect();
    let advs: Vec<_> = inputs.iter().map(|claim| claim.adv.clone()).collect();
    crate::paper::relations::mix_adv(mix, &rho_mats, &advs).map_err(|e| Error::AdvPresence {
        present: e.present,
        total: e.total,
    })
}

fn validate_inactive_x_zero(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_inactive_x_zero_one("input", input)?;
    }
    validate_inactive_x_zero_one("combined", combined)?;
    Ok(())
}

fn validate_inactive_x_zero_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if !superneo_inactive_x_zero(&claim.X, claim.m_in) {
        return Err(Error::InactiveX(owner));
    }
    Ok(())
}

fn validate_r_shape(
    s: &crate::paper::relations::Structure,
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    for input in inputs {
        validate_r_shape_one("input", s, input)?;
    }
    validate_r_shape_one("combined", s, combined)?;
    Ok(())
}

fn validate_r_shape_one(
    owner: &'static str,
    s: &crate::paper::relations::Structure,
    claim: &CeClaim,
) -> Result<(), Error> {
    let expected = s.n.next_power_of_two().max(2).trailing_zeros() as usize;
    if claim.r.len() != expected {
        return Err(Error::RShape(owner));
    }
    Ok(())
}

fn validate_r_consistency(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        if input.r != combined.r {
            return Err(Error::RConsistency);
        }
    }
    Ok(())
}

fn validate_y_ring_shape(
    s: &crate::paper::relations::Structure,
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    for input in inputs {
        validate_y_ring_shape_one("input", s, input)?;
    }
    validate_y_ring_shape_one("combined", s, combined)?;
    Ok(())
}

fn validate_y_ring_shape_one(
    owner: &'static str,
    s: &crate::paper::relations::Structure,
    claim: &CeClaim,
) -> Result<(), Error> {
    if claim.y_ring.len() != s.t() {
        return Err(Error::YRingShape(owner));
    }
    let expected_lanes = D.next_power_of_two();
    if claim.y_ring.iter().any(|row| row.len() != expected_lanes) {
        return Err(Error::YRingShape(owner));
    }
    Ok(())
}

fn validate_y_ring_padding_zero(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_y_ring_padding_zero_one("input", input)?;
    }
    validate_y_ring_padding_zero_one("combined", combined)?;
    Ok(())
}

fn validate_y_ring_padding_zero_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    for row in &claim.y_ring {
        if row.iter().skip(D).any(|&lane| lane != K::default()) {
            return Err(Error::YRingPadding(owner));
        }
    }
    Ok(())
}

fn validate_ct_consistency(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_ct_consistency_one("input", input)?;
    }
    validate_ct_consistency_one("combined", combined)?;
    Ok(())
}

fn validate_ct_consistency_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(Error::CtConsistency(owner));
    }
    for (ct, row) in claim.ct.iter().zip(&claim.y_ring) {
        let Some(&constant_term) = row.first() else {
            return Err(Error::CtConsistency(owner));
        };
        if *ct != constant_term {
            return Err(Error::CtConsistency(owner));
        }
    }
    Ok(())
}

fn validate_s_col_shape(
    s: &crate::paper::relations::Structure,
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    for input in inputs {
        validate_s_col_shape_one("input", s, input)?;
    }
    validate_s_col_shape_one("combined", s, combined)?;
    Ok(())
}

fn validate_s_col_shape_one(
    owner: &'static str,
    s: &crate::paper::relations::Structure,
    claim: &CeClaim,
) -> Result<(), Error> {
    let expected = s.m.next_power_of_two().max(2).trailing_zeros() as usize;
    if claim.s_col.len() != expected {
        return Err(Error::SColShape(owner));
    }
    Ok(())
}

fn validate_s_col_consistency(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        if input.s_col != combined.s_col {
            return Err(Error::SColConsistency);
        }
    }
    Ok(())
}

fn validate_y_zcol_combination(
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    for input in inputs {
        validate_y_zcol_shape_padding_one("input", input)?;
    }
    validate_y_zcol_shape_padding_one("combined", combined)?;

    let d_pad = D.next_power_of_two();
    let mut expected = vec![K::default(); d_pad];
    for (rho, input) in rhos.iter().zip(inputs.iter()) {
        let rho = rho.as_mat();
        for k in 0..D {
            let yk = input.y_zcol[k];
            if yk == K::default() {
                continue;
            }
            for r in 0..D {
                expected[r] += K::from(rho[(r, k)]) * yk;
            }
        }
    }
    if expected != combined.y_zcol {
        return Err(Error::YZcolConsistency);
    }
    Ok(())
}

fn validate_y_zcol_shape_padding_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    let d_pad = D.next_power_of_two();
    if claim.y_zcol.len() != d_pad {
        return Err(Error::YZcolConsistency);
    }
    if claim
        .y_zcol
        .iter()
        .skip(D)
        .any(|&lane| lane != K::default())
    {
        return Err(Error::YZcolPadding(owner));
    }
    Ok(())
}

fn validate_fold_digest_consistency(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        if input.fold_digest != combined.fold_digest {
            return Err(Error::FoldDigest);
        }
    }
    Ok(())
}

fn validate_fold_digest_canonical(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    for (lane, chunk) in claim.fold_digest.chunks_exact(8).enumerate() {
        let value = u64::from_le_bytes(chunk.try_into().expect("fold_digest lanes are 8 bytes"));
        if value >= F::ORDER_U64 {
            return Err(Error::FoldDigestCanonicality { owner, lane });
        }
    }
    Ok(())
}

fn validate_supported_sidecars(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_supported_sidecars_one("input", input)?;
    }
    validate_supported_sidecars_one("combined", combined)?;
    Ok(())
}

fn validate_supported_sidecars_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if !claim.aux_openings.is_empty() {
        return Err(Error::UnsupportedSidecar {
            owner,
            field: "aux_openings",
        });
    }
    if !claim.c_step_coords.is_empty() {
        return Err(Error::UnsupportedSidecar {
            owner,
            field: "c_step_coords",
        });
    }
    if claim.u_offset != 0 {
        return Err(Error::UnsupportedSidecar {
            owner,
            field: "u_offset",
        });
    }
    if claim.u_len != 0 {
        return Err(Error::UnsupportedSidecar { owner, field: "u_len" });
    }
    Ok(())
}
