//! Outputs module: Build ME(b,L)^k output instances
//!
//! # Paper Reference
//! Section 4.4, Step 3: Send y' values
//!
//! For each instance i ∈ [k]:
//! - y'_{(i,j)} = Z_i·M_j^T·r̂' for all j ∈ [t]
//! - y'_scalars[j] = Σ_ℓ b^{ℓ-1}·y'_{(i,j),ℓ} (Ajtai recomposition)

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::pi_ccs::precompute::Inst;
use crate::pi_ccs::eq_weights::{HalfTableEq, spmv_csr_t_weighted_fk};
use crate::pi_ccs::sparse_matrix::Csr;
use neo_ccs::{CcsStructure, MeInstance, MatRef, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

/// Build ME(b,L)^k output instances from sum-check result
///
/// # Paper Reference
/// Section 4.4, Step 3:
/// P: For all i ∈ [k] and j ∈ [t], send y'_{(i,j)} := Z_i·M_j^T·r̂'
///
/// Where:
/// - First mcs_list.len() outputs come from MCS instances
/// - Remaining me_inputs.len() outputs come from ME input instances
/// - All use the same r' from sum-check
pub fn build_me_outputs<L>(
    tr: &mut Poseidon2Transcript,
    s: &CcsStructure<F>,
    params: &NeoParams,
    mats_csr: &[Csr<F>],
    insts: &[Inst],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    r_prime: &[K],
    l: &L,
) -> Result<Vec<MeInstance<Cmt, F, K>>, PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    let fold_digest = tr.digest32();

    let w = HalfTableEq::new(r_prime);
    let vjs: Vec<Vec<K>> = mats_csr
        .par_iter()
        .map(|csr| spmv_csr_t_weighted_fk(csr, &w))
        .collect();

    let base_f = K::from(F::from_u64(params.b as u64));
    let mut pow_cache = vec![K::ONE; D];
    for i in 1..D {
        pow_cache[i] = pow_cache[i - 1] * base_f;
    }
    let recompose = |y: &[K]| -> K {
        y.iter()
            .zip(&pow_cache)
            .fold(K::ZERO, |acc, (&v, &p)| acc + v * p)
    };

    let mut out_me = Vec::with_capacity(insts.len() + me_witnesses.len());

    for inst in insts.iter() {
        let X = l.project_x(inst.Z, inst.m_in);

        let mut y = Vec::with_capacity(s.t());
        let z_ref = MatRef::from_mat(inst.Z);
        for vj in &vjs {
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F, K>(z_ref.data, z_ref.rows, z_ref.cols, vj);
            y.push(yj);
        }

        let y_scalars: Vec<K> = y.iter().map(|yj| recompose(yj)).collect();

        out_me.push(MeInstance {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: inst.c.clone(),
            X,
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inst.m_in,
            fold_digest,
        });
    }

    for (inp, zi) in me_inputs.iter().zip(me_witnesses.iter()) {
        let mut y = Vec::with_capacity(s.t());
        let z_ref = MatRef::from_mat(zi);
        for vj in &vjs {
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F, K>(z_ref.data, z_ref.rows, z_ref.cols, vj);
            y.push(yj);
        }
        let y_scalars: Vec<K> = y.iter().map(|yj| recompose(yj)).collect();

        out_me.push(MeInstance {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: inp.c.clone(),
            X: inp.X.clone(),
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inp.m_in,
            fold_digest,
        });
    }

    Ok(out_me)
}

