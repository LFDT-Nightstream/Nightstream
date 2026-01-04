//! Minimal protocol utilities for paper-exact implementation
//!
//! Contains only the essential functions needed by prove and verify.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsMatrix, CcsStructure, McsInstance, MeInstance, SparsePoly};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_transcript::{labels as tr_labels, Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

/// Computed dimensions for the CCS reduction
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub ell_d: usize,
    pub ell_n: usize,
    pub ell: usize,
    pub d_sc: usize,
}

pub use crate::optimized_engine::Challenges;

/// Build dimensions and validate extension field security policy
pub fn build_dims_and_policy(params: &NeoParams, s: &CcsStructure<F>) -> Result<Dims, PiCcsError> {
    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }

    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;

    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;

    let ell = ell_d + ell_n;

    let d_sc = core::cmp::max(
        s.max_degree() as usize + 1,
        core::cmp::max(2, 2 * (params.b as usize) + 2),
    );

    let ext = params
        .extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;

    if ext.slack_bits < 0 {
        return Err(PiCcsError::ExtensionPolicyFailed(format!(
            "Insufficient security slack: {} bits",
            ext.slack_bits
        )));
    }

    Ok(Dims {
        ell_d,
        ell_n,
        ell,
        d_sc,
    })
}

/// Bind header and MCS instances to transcript
pub fn bind_header_and_instances(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    ell: usize,
    d_sc: usize,
    _slack_bits: i32,
) -> Result<(), PiCcsError> {
    let digest = digest_ccs_matrices(s);
    bind_header_and_instances_with_digest(tr, params, s, mcs_list, ell, d_sc, &digest)
}

/// Bind CCS header and MCS instances to transcript, using a precomputed CCS matrix digest.
///
/// This is performance-critical in shard folding, where the same `s` is reused across many steps.
pub fn bind_header_and_instances_with_digest(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    ell: usize,
    d_sc: usize,
    mat_digest: &[Goldilocks],
) -> Result<(), PiCcsError> {
    tr.append_message(tr_labels::PI_CCS, b"");

    let ext = params
        .extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;

    tr.append_message(b"neo/ccs/header/v1", b"");

    tr.append_u64s(
        b"ccs/header",
        &[
            64,
            ext.s_supported as u64,
            params.lambda as u64,
            ell as u64,
            d_sc as u64,
            ext.slack_bits.unsigned_abs() as u64,
        ],
    );
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 { 1 } else { 0 }]);

    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);

    if mat_digest.len() != 4 {
        return Err(PiCcsError::InvalidInput(format!(
            "CCS matrix digest must have len 4, got {}",
            mat_digest.len()
        )));
    }
    for &digest_elem in mat_digest {
        tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]);
    }

    absorb_sparse_polynomial(tr, &s.f);

    for inst in mcs_list.iter() {
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        tr.append_fields(b"c_data", &inst.c.data);
    }

    Ok(())
}

/// Bind ME inputs to transcript
pub fn bind_me_inputs(tr: &mut Poseidon2Transcript, me_inputs: &[MeInstance<Cmt, F, K>]) -> Result<(), PiCcsError> {
    // v2 batches (r, y) coefficient absorption under a single label+len framing for performance.
    // This is NOT transcript-equivalent to the previous per-limb/per-element `append_fields` loop.
    tr.append_message(b"neo/ccs/me_inputs/v2", b"");
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);

    let k_coeffs_len = K::ONE.as_coeffs().len();

    for me in me_inputs {
        tr.append_fields(b"c_data_in", &me.c.data);
        tr.append_u64s(b"m_in_in", &[me.m_in as u64]);

        let r_field_len = me
            .r
            .len()
            .checked_mul(k_coeffs_len)
            .ok_or_else(|| PiCcsError::InvalidInput("ME.r length overflow".into()))?;
        tr.append_fields_iter(
            b"r_in",
            r_field_len,
            me.r.iter().flat_map(|limb| limb.as_coeffs().into_iter()),
        );

        let y_elem_count: usize = me
            .y
            .iter()
            .try_fold(0usize, |acc, yj| {
                acc.checked_add(yj.len())
                    .ok_or_else(|| PiCcsError::InvalidInput("ME.y length overflow".into()))
            })?;
        let y_field_len = y_elem_count
            .checked_mul(k_coeffs_len)
            .ok_or_else(|| PiCcsError::InvalidInput("ME.y length overflow".into()))?;
        tr.append_fields_iter(
            b"y_elem",
            y_field_len,
            me.y
                .iter()
                .flat_map(|yj| yj.iter().flat_map(|y_elem| y_elem.as_coeffs().into_iter())),
        );
    }

    Ok(())
}

/// Sample challenges α, β, γ from transcript
pub fn sample_challenges(tr: &mut Poseidon2Transcript, ell_d: usize, ell: usize) -> Result<Challenges, PiCcsError> {
    tr.append_message(b"neo/ccs/chals/v1", b"");

    let alpha: Vec<K> = (0..ell_d)
        .map(|_| {
            let c = tr.challenge_fields(b"chal/k", 2);
            neo_math::from_complex(c[0], c[1])
        })
        .collect();

    let beta: Vec<K> = (0..ell)
        .map(|_| {
            let c = tr.challenge_fields(b"chal/k", 2);
            neo_math::from_complex(c[0], c[1])
        })
        .collect();

    let (beta_a, beta_r) = beta.split_at(ell_d);

    let g = tr.challenge_fields(b"chal/k", 2);
    let gamma = neo_math::from_complex(g[0], g[1]);

    Ok(Challenges {
        alpha,
        beta_a: beta_a.to_vec(),
        beta_r: beta_r.to_vec(),
        gamma,
    })
}

pub fn digest_ccs_matrices<F: Field + PrimeField64>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

    const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154;
    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);

    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;

    const DOMAIN_STRING: &[u8] = b"neo/ccs/matrices/v1";
    for &byte in DOMAIN_STRING {
        if absorbed >= 15 {
            poseidon2.permute_mut(&mut state);
            absorbed = 0;
        }
        state[absorbed] = Goldilocks::from_u32(byte as u32);
        absorbed += 1;
    }

    if absorbed + 3 >= 16 {
        poseidon2.permute_mut(&mut state);
        absorbed = 0;
    }
    state[absorbed] = Goldilocks::from_u64(s.n as u64);
    state[absorbed + 1] = Goldilocks::from_u64(s.m as u64);
    state[absorbed + 2] = Goldilocks::from_u64(s.t() as u64);

    poseidon2.permute_mut(&mut state);

    for (j, matrix) in s.matrices.iter().enumerate() {
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(j as u64);
        absorbed += 1;

        let mut emit = |row: usize, col: usize, val_u64: u64| {
            if absorbed + 3 > 15 {
                poseidon2.permute_mut(&mut state);
                absorbed = 0;
            }
            state[absorbed] = Goldilocks::from_u64(row as u64);
            state[absorbed + 1] = Goldilocks::from_u64(col as u64);
            state[absorbed + 2] = Goldilocks::from_u64(val_u64);
            absorbed += 3;
        };

        match matrix {
            CcsMatrix::Identity { n } => {
                debug_assert_eq!(*n, s.n);
                debug_assert_eq!(*n, s.m);
                let one_u = F::ONE.as_canonical_u64();
                for row in 0..s.n {
                    emit(row, row, one_u);
                }
            }
            CcsMatrix::Csc(csc) => {
                // Enumerate non-zeros in row-major order (matches dense scan) without allocating
                // a `Vec<Vec<_>>` of length `nrows` (which is massive for large circuits).
                //
                // Strategy: build CSR-style row segments in one contiguous allocation.
                let nrows = csc.nrows;
                let nnz = csc.vals.len();
                debug_assert_eq!(csc.row_idx.len(), nnz);

                // 1) Count entries per row.
                let mut row_counts = vec![0u32; nrows];
                for &r in csc.row_idx.iter() {
                    row_counts[r] += 1;
                }

                // 2) Prefix sums to get row offsets.
                let mut row_offsets = vec![0usize; nrows + 1];
                for r in 0..nrows {
                    row_offsets[r + 1] = row_offsets[r] + (row_counts[r] as usize);
                }
                debug_assert_eq!(row_offsets[nrows], nnz);

                // 3) Fill (col,val) pairs into per-row segments while scanning columns in order.
                let mut write_pos = row_offsets[..nrows].to_vec();
                let mut entries = vec![(0usize, 0u64); nnz];

                for col in 0..csc.ncols {
                    let s0 = csc.col_ptr[col];
                    let e0 = csc.col_ptr[col + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        let idx = write_pos[row];
                        write_pos[row] = idx + 1;
                        entries[idx] = (col, csc.vals[k].as_canonical_u64());
                    }
                }

                // 4) Emit in row-major order.
                for row in 0..nrows {
                    let start = row_offsets[row];
                    let end = row_offsets[row + 1];
                    for &(col, val_u64) in &entries[start..end] {
                        emit(row, col, val_u64);
                    }
                }
            }
        }

        poseidon2.permute_mut(&mut state);
    }

    state[0..4].to_vec()
}

/// Compute the CCS matrix digest, optionally using a prebuilt sparse cache to avoid scanning dense zeros.
///
/// When `sparse` is provided, this function matches `digest_ccs_matrices` exactly (same digest),
/// but enumerates non-zeros from the cache and sorts them into row-major order.
pub fn digest_ccs_matrices_with_sparse_cache<Ff: Field + PrimeField64>(
    s: &CcsStructure<Ff>,
    _sparse: Option<&crate::engines::optimized_engine::oracle::SparseCache<Ff>>,
) -> Vec<Goldilocks> {
    digest_ccs_matrices(s)
}

fn absorb_sparse_polynomial(tr: &mut Poseidon2Transcript, f: &SparsePoly<F>) {
    tr.append_message(b"neo/ccs/poly", b"");
    tr.append_u64s(b"arity", &[f.arity() as u64]);
    tr.append_u64s(b"terms_len", &[f.terms().len() as u64]);

    let mut terms: Vec<_> = f.terms().iter().collect();
    terms.sort_by_key(|term| &term.exps);

    for term in terms {
        tr.append_fields(b"coeff", &[term.coeff]);
        let exps: Vec<u64> = term.exps.iter().map(|&e| e as u64).collect();
        tr.append_u64s(b"exps", &exps);
    }
}
