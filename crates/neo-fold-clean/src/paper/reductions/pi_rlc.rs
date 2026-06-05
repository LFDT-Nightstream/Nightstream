//! Π_RLC — SuperNeo §7.4. Random Linear Combination.
//!
//! Reduction:  CE(b, ℒ)^{K+k}   →   CE(B, ℒ)   where B = b^k
//!
//! Soundness: **weak** wrt φ projecting commitments (Lemma 4, proof in §D.5).
//! Composes with Π_CCS (strong wrt the same φ) via Theorem 6.
//!
//! ## What this file owns
//!
//! - The `prove` and `verify` step-down flows in paper-step order.
//! - The shape contract: K+k input CE claims, K+k witnesses (prover-only).
//!
//! ## What this file does *not* own
//!
//! - The bound check `(K+k)·T·(b−1) < B` — that's in `paper/sampling.rs`.
//! - The actual Σρ_i mix — that's `engine::optimized::prove_pi_rlc`
//!   (prover) and `engine::optimized::verify_pi_rlc` (verifier).
//!
//! ## Why `combined` is on the wire
//!
//! In principle the verifier can recompute `combined = Σ ρ_i · u_i` from
//! public-coin ρ and the K+k Π_CCS outputs. We mirror `neo-fold-prototype`'s
//! contract and have the prover send its parent: the verifier *checks* the
//! recomputation matches before feeding Π_DEC. Why not just recompute and
//! drop the field?
//!
//! Π_DEC's children are committed against the prover's exact `parent`. The
//! parent's CE claim is built by the engine and then patched with
//! `out.c = mix_commits(rhos, &inputs_c)`. Recomputing on the verifier
//! side could produce an equivalent CE claim but not necessarily a
//! bit-identical one (e.g., on the `c` field if the mix closure on the
//! verifier side has a subtle implementation drift). Comparing the
//! prover-supplied parent against the verifier's recomputation is the safer
//! contract; the wire cost is one CE claim per IVC step.

use neo_ccs::Mat;
use neo_math::{D, F, K};
use thiserror::Error;

use crate::engine::optimized as engine;
use crate::engine::transcript::Transcript;
use crate::paper::digest;
use crate::paper::params::Params;
use crate::paper::relations::{superneo_inactive_x_zero, CeClaim, RlcMixer};
use crate::paper::sampling::check_rlc_bound;
use p3_field::PrimeField64;

pub(crate) const PI_RLC_INPUT_CLAIMS_DIGEST_LABEL: &[u8] = b"pi_rlc/input_claims_digest";

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
    validate_input_shape(claims, witnesses)?;
    enforce_rlc_bound(pp, claims.len())?;
    validate_inputs_before_rho(s, claims)?;
    bind_input_claims_for_rho(tr, claims);
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    let (combined, z_mix) = engine::prove_pi_rlc_refs(pp, s, &rhos, claims, witnesses, |zs, cs| mix(zs, cs))?;
    validate_nc_sidecars(s, &rhos, claims, &combined)?;
    Ok((
        Output {
            claim: combined.clone(),
            witness: z_mix,
        },
        Proof { combined },
    ))
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
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    enforce_rlc_bound(pp, claims.len())?;
    validate_inputs_before_rho(s, claims)?;
    bind_input_claims_for_rho(tr, claims);
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    validate_nc_sidecars(s, &rhos, claims, &proof.combined)?;
    let ok = engine::verify_pi_rlc(pp, s, &rhos, claims, &proof.combined, |zs, cs| mix(zs, cs))?;
    if !ok {
        return Err(Error::VerifyRejected);
    }
    Ok(proof.combined.clone())
}

fn bind_input_claims_for_rho(tr: &mut Transcript, claims: &[CeClaim]) {
    let input_claims_digest = digest::pi_ccs_outputs_digest(claims);
    tr.append_fields(PI_RLC_INPUT_CLAIMS_DIGEST_LABEL, &input_claims_digest);
}

// ──────────────────────────────────────────────────────────────────────────
// Step bodies
// ──────────────────────────────────────────────────────────────────────────

fn validate_inputs_before_rho(s: &crate::paper::relations::Structure, inputs: &[CeClaim]) -> Result<(), Error> {
    for input in inputs {
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
    rhos: &[neo_reductions::common::RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
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
    validate_fold_digest_consistency(inputs, combined)?;
    validate_supported_sidecars(inputs, combined)?;
    Ok(())
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
