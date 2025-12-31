//! Shared helpers for Twist/Shout.
//!
//! This module is intentionally small and mechanical: it centralizes the
//! duplicated transcript sampling, ME-opening construction, CCS padding, and
//! Ajtai decode+pad utilities used by both protocols.

use crate::mle::compute_me_y_for_ccs;
use neo_ajtai::Commitment as AjtaiCmt;
use neo_ccs::{matrix::Mat, relations::MeInstance, CcsStructure};
use neo_math::{from_complex, F as BaseField, K as KElem};
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript, TranscriptProtocol};
use p3_field::{PrimeCharacteristicRing, PrimeField};

use crate::ajtai::decode_vector as ajtai_decode_vector;

// ============================================================================
// Transcript sampling helpers
// ============================================================================

pub fn sample_ext_point(
    tr: &mut Poseidon2Transcript,
    label: &'static [u8],
    coord0_label: &'static [u8],
    coord1_label: &'static [u8],
    len: usize,
) -> Vec<KElem> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(label, &i.to_le_bytes());
        let c0 = tr.challenge_field(coord0_label);
        let c1 = tr.challenge_field(coord1_label);
        out.push(from_complex(c0, c1));
    }
    out
}

pub fn sample_base_addr_point(
    tr: &mut Poseidon2Transcript,
    label: &'static [u8],
    coord0_label: &'static [u8],
    len: usize,
) -> Vec<KElem> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(label, &i.to_le_bytes());
        let c0 = tr.challenge_field(coord0_label);
        out.push(from_complex(c0, BaseField::ZERO));
    }
    out
}

// ============================================================================
// Transcript binding helpers
// ============================================================================

pub fn absorb_ajtai_commitments(
    tr: &mut Poseidon2Transcript,
    count_label: &'static [u8],
    idx_label: &'static [u8],
    comms: &[AjtaiCmt],
) {
    tr.append_message(count_label, &(comms.len() as u64).to_le_bytes());
    for (i, comm) in comms.iter().enumerate() {
        tr.append_message(idx_label, &(i as u64).to_le_bytes());
        tr.absorb_commit_coords(&comm.data);
    }
}

// ============================================================================
// CCS padding + ME opening
// ============================================================================

pub fn pad_mat_to_ccs_width(mat: &Mat<BaseField>, target_cols: usize) -> Result<Mat<BaseField>, PiCcsError> {
    if mat.cols() > target_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "pad_mat_to_ccs_width: matrix width ({}) exceeds CCS width ({})",
            mat.cols(),
            target_cols
        )));
    }
    if mat.cols() == target_cols {
        return Ok(mat.clone());
    }
    let d = mat.rows();
    let mut out = Mat::zero(d, target_cols, BaseField::ZERO);
    for r in 0..d {
        let row = mat.row(r);
        for c in 0..mat.cols() {
            out.set(r, c, row[c]);
        }
    }
    Ok(out)
}

/// Shared ME-opening constructor.
///
/// `digest_label` must remain domain-separated (e.g. `b"twist/me_digest"`, `b"shout/me_digest"`).
pub fn mk_me_opening_with_ccs<Cmt, KOut>(
    tr: &Poseidon2Transcript,
    digest_label: &'static [u8],
    params: &NeoParams,
    s: &CcsStructure<BaseField>,
    comm: &Cmt,
    mat: &Mat<BaseField>,
    r: &[KElem],
    m_in: usize,
) -> Result<MeInstance<Cmt, BaseField, KOut>, PiCcsError>
where
    KOut: From<KElem> + Clone,
    Cmt: Clone,
{
    let d = params.d as usize;
    let t = s.t();
    let y_pad = d.next_power_of_two();

    // Pad witness to CCS width
    let z_padded = pad_mat_to_ccs_width(mat, s.m)?;

    // X = L_x(Z)
    let x_mat = crate::mle::compute_me_x(&z_padded, m_in);

    // CCS-aware y_j / y_scalar_j
    let (mut y_vecs_k, mut y_scalars_k) = compute_me_y_for_ccs(s, &z_padded, r, params.b as u64);

    // Ensure canonical shapes (matches neo-fold normalization).
    for row in y_vecs_k.iter_mut() {
        row.resize(y_pad, KElem::ZERO);
    }
    y_vecs_k.resize_with(t, || vec![KElem::ZERO; y_pad]);
    y_scalars_k.resize(t, KElem::ZERO);

    let y: Vec<Vec<KOut>> = y_vecs_k
        .into_iter()
        .map(|yj| yj.into_iter().map(KOut::from).collect())
        .collect();

    let y_scalars: Vec<KOut> = y_scalars_k.into_iter().map(KOut::from).collect();

    let fold_digest = {
        let mut fork = tr.fork(digest_label);
        fork.digest32()
    };

    Ok(MeInstance {
        c: comm.clone(),
        X: x_mat,
        r: r.iter().copied().map(KOut::from).collect(),
        y,
        y_scalars,
        m_in,
        fold_digest,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    })
}

// ============================================================================
// Decode helpers
// ============================================================================

/// Decode an Ajtai-encoded column vector and pad to a power-of-two domain length.
pub fn decode_mat_to_k_padded(params: &NeoParams, mat: &Mat<BaseField>, pow2_len: usize) -> Vec<KElem> {
    let v = ajtai_decode_vector(params, mat);
    let mut out: Vec<KElem> = v.into_iter().map(Into::into).collect();
    out.resize(pow2_len, KElem::ZERO);
    out
}

/// Decode many Ajtai matrices into padded `KElem` vectors (common for bit-columns).
pub fn decode_mats_to_k_padded(params: &NeoParams, mats: &[Mat<BaseField>], pow2_len: usize) -> Vec<Vec<KElem>> {
    mats.iter()
        .map(|m| decode_mat_to_k_padded(params, m, pow2_len))
        .collect()
}

/// Decode address bits into flattened addresses (shared by semantic checkers).
pub fn decode_addrs_from_bits<F: PrimeField>(
    params: &NeoParams,
    addr_bit_mats: &[Mat<F>],
    d: usize,
    ell: usize,
    n_side: usize,
    steps: usize,
) -> Vec<u64> {
    let decoded: Vec<Vec<F>> = addr_bit_mats
        .iter()
        .map(|m| ajtai_decode_vector(params, m))
        .collect();

    let mut addrs = vec![0u64; steps];
    for dim in 0..d {
        let base = dim * ell;
        let stride = (n_side as u64)
            .checked_pow(dim as u32)
            .expect("decode_addrs_from_bits: stride overflow");
        for b in 0..ell {
            let col = &decoded[base + b];
            let bit_weight = 1u64
                .checked_shl(b as u32)
                .expect("decode_addrs_from_bits: bit_weight overflow");
            for j in 0..steps.min(col.len()) {
                if col[j] == F::ONE {
                    let delta = bit_weight
                        .checked_mul(stride)
                        .expect("decode_addrs_from_bits: address contribution overflow");
                    addrs[j] = addrs[j]
                        .checked_add(delta)
                        .expect("decode_addrs_from_bits: address overflow");
                }
            }
        }
    }
    addrs
}

// ============================================================================
// Convenience helpers
// ============================================================================

pub fn emit_me_claims_for_mats<Cmt>(
    tr: &Poseidon2Transcript,
    digest_label: &'static [u8],
    params: &NeoParams,
    s: &CcsStructure<BaseField>,
    comms: &[Cmt],
    mats: &[Mat<BaseField>],
    r: &[KElem],
    m_in: usize,
) -> Result<Vec<MeInstance<Cmt, BaseField, KElem>>, PiCcsError>
where
    Cmt: Clone,
{
    if comms.len() < mats.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "emit_me_claims_for_mats: comms.len()={} < mats.len()={}",
            comms.len(),
            mats.len()
        )));
    }

    let mut out = Vec::with_capacity(mats.len());
    for (i, mat) in mats.iter().enumerate() {
        out.push(mk_me_opening_with_ccs::<Cmt, KElem>(
            tr,
            digest_label,
            params,
            s,
            &comms[i],
            mat,
            r,
            m_in,
        )?);
    }
    Ok(out)
}
