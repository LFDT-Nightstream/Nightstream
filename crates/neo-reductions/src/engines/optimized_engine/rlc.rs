//! Optimized Π_RLC mixing owns the witness-side random linear combination path.
//!
//! It does not own paper-exact cross-check formulas or DEC.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

fn left_mul_acc_optimized<Ff>(acc: &mut Mat<Ff>, rho: &Mat<Ff>, a: &Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    debug_assert_eq!(rho.rows(), D);
    debug_assert_eq!(rho.cols(), D);
    debug_assert_eq!(a.rows(), D);
    debug_assert_eq!(acc.rows(), D);
    debug_assert_eq!(a.cols(), acc.cols());

    let m = acc.cols();
    let rho_data = rho.as_slice();
    let a_data = a.as_slice();

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_num_threads() > 1 {
            let acc_data = acc.as_mut_slice();
            const BLOCK_COLS: usize = 1024;
            acc_data
                .par_chunks_exact_mut(m)
                .enumerate()
                .for_each(|(rr, row_out)| {
                    for col0 in (0..m).step_by(BLOCK_COLS) {
                        let len = core::cmp::min(BLOCK_COLS, m - col0);
                        for kk in 0..D {
                            let coeff = rho_data[rr * D + kk];
                            if coeff == Ff::ZERO {
                                continue;
                            }
                            let in_off = kk * m + col0;
                            for t in 0..len {
                                row_out[col0 + t] += coeff * a_data[in_off + t];
                            }
                        }
                    }
                });
            return;
        }
    }

    let acc_data = acc.as_mut_slice();
    const BLOCK_COLS: usize = 1024;
    for rr in 0..D {
        let row_out = &mut acc_data[rr * m..(rr + 1) * m];
        for col0 in (0..m).step_by(BLOCK_COLS) {
            let len = core::cmp::min(BLOCK_COLS, m - col0);
            for kk in 0..D {
                let coeff = rho_data[rr * D + kk];
                if coeff == Ff::ZERO {
                    continue;
                }
                let in_off = kk * m + col0;
                for t in 0..len {
                    row_out[col0 + t] += coeff * a_data[in_off + t];
                }
            }
        }
    }
}

fn rlc_reduction_optimized_from_refs<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!me_inputs.is_empty(), "Π_RLC(optimized): need at least one input");
    let k1 = me_inputs.len();
    assert_eq!(rhos.len(), k1, "Π_RLC: |rhos| must equal |inputs|");
    assert_eq!(Zs.len(), k1, "Π_RLC: |Zs| must equal |inputs|");
    crate::common::validate_rhos_are_rotation_matrices(params, rhos, "Π_RLC(optimized): rhos")
        .unwrap_or_else(|e| panic!("Π_RLC(optimized): invalid rho set: {e}"));
    let z_cols = Zs[0].cols();
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_superneo_witness_mat(*z, s.m)
            .unwrap_or_else(|e| panic!("Π_RLC(optimized): invalid witness shape at input {idx}: {e}"));
        assert_eq!(
            z.cols(),
            z_cols,
            "Π_RLC(optimized): all witness mats must share packed width"
        );
    }

    let d_pad = 1usize << ell_d;
    let t_core = s.t();
    let m_in = me_inputs[0].m_in;
    let r = me_inputs[0].r.clone();
    let aux_len = me_inputs[0].aux_openings.len();
    for (idx, inst) in me_inputs.iter().enumerate() {
        assert_eq!(
            inst.aux_openings.len(),
            aux_len,
            "Π_RLC: aux_openings.len mismatch at input {idx}"
        );
    }

    let mut y_ring: Vec<Vec<K>> = Vec::with_capacity(t_core);
    for j in 0..t_core {
        let mut yj_acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            let yi = &me_inputs[i].y_ring[j];
            debug_assert!(yi.len() >= D, "ME.y_ring[{j}] must have length >= D");
            let rho = &rhos[i];
            for rr in 0..D {
                let mut acc_rr = K::ZERO;
                for kk in 0..D {
                    acc_rr += K::from(rho[(rr, kk)]) * yi[kk];
                }
                yj_acc[rr] += acc_rr;
            }
        }
        y_ring.push(yj_acc);
    }

    let wants_nc_channel = !(me_inputs[0].s_col.is_empty() && me_inputs[0].y_zcol.is_empty());
    if wants_nc_channel {
        assert!(
            !me_inputs[0].s_col.is_empty() && !me_inputs[0].y_zcol.is_empty(),
            "Π_RLC: incomplete NC channel on input 0 (expected both s_col and y_zcol)"
        );
        for (idx, inst) in me_inputs.iter().enumerate() {
            assert_eq!(inst.s_col, me_inputs[0].s_col, "Π_RLC: s_col mismatch at input {idx}");
            assert_eq!(
                inst.y_zcol.len(),
                d_pad,
                "Π_RLC: y_zcol len mismatch at input {idx} (expected {d_pad}, got {})",
                inst.y_zcol.len()
            );
        }
    }

    let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);

    let mut aux_openings = vec![K::ZERO; aux_len];
    for (rho, inst) in rhos.iter().zip(me_inputs.iter()) {
        let w = K::from(rho[(0, 0)]);
        for (dst, src) in aux_openings.iter_mut().zip(inst.aux_openings.iter()) {
            *dst += w * *src;
        }
    }

    let mut Z = Mat::zero(D, z_cols, Ff::ZERO);
    for (rho, z_in) in rhos.iter().zip(Zs.iter()) {
        left_mul_acc_optimized(&mut Z, rho, z_in);
    }

    let mut X = Mat::zero(D, m_in, Ff::ZERO);
    for (rho, inst) in rhos.iter().zip(me_inputs.iter()) {
        left_mul_acc_optimized(&mut X, rho, &inst.X);
    }

    let y_zcol = if wants_nc_channel {
        let mut acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            for rr in 0..D {
                let mut sum = K::ZERO;
                for kk in 0..D {
                    sum += K::from(rhos[i][(rr, kk)]) * me_inputs[i].y_zcol[kk];
                }
                acc[rr] += sum;
            }
        }
        acc
    } else {
        Vec::new()
    };

    let out = CeClaim::<Cmt, Ff, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: me_inputs[0].c.clone(),
        X,
        r,
        s_col: me_inputs[0].s_col.clone(),
        y_ring,
        ct,
        aux_openings,
        y_zcol,
        m_in,
        fold_digest: me_inputs[0].fold_digest,
    };

    (out, Z)
}

pub fn rlc_reduction_optimized<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let z_refs: Vec<&Mat<Ff>> = Zs.iter().collect();
    rlc_reduction_optimized_from_refs::<Ff>(s, params, rhos, me_inputs, &z_refs, ell_d)
}

pub fn rlc_reduction_optimized_with_commit_mix<Ff, Comb>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
    combine_commit: Comb,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    Comb: Fn(&[Mat<Ff>], &[Cmt]) -> Cmt,
{
    let (mut out, Z) = rlc_reduction_optimized_from_refs::<Ff>(s, params, rhos, me_inputs, Zs, ell_d);
    let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
    out.c = combine_commit(rhos, &inputs_c);
    (out, Z)
}
