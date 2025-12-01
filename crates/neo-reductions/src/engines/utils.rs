//! Minimal protocol utilities for paper-exact implementation
//! 
//! Contains only the essential functions needed by prove and verify.

#![allow(non_snake_case)]

use neo_transcript::{Transcript, Poseidon2Transcript, labels as tr_labels};
use neo_ccs::{CcsStructure, McsInstance, MeInstance, SparsePoly};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{F, K, D, KExtensions};
use p3_field::{Field, PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use crate::error::PiCcsError;

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
pub fn build_dims_and_policy(
    params: &NeoParams,
    s: &CcsStructure<F>,
) -> Result<Dims, PiCcsError> {
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

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;

    if ext.slack_bits < 0 {
        return Err(PiCcsError::ExtensionPolicyFailed(format!(
            "Insufficient security slack: {} bits", ext.slack_bits
        )));
    }

    Ok(Dims { ell_d, ell_n, ell, d_sc })
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
    tr.append_message(tr_labels::PI_CCS, b"");
    
    let ext = params
        .extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    
    tr.append_message(b"neo/ccs/header/v1", b"");
    
    tr.append_u64s(b"ccs/header", &[
        64,
        ext.s_supported as u64,
        params.lambda as u64,
        ell as u64,
        d_sc as u64,
        ext.slack_bits.unsigned_abs() as u64,
    ]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 { 1 } else { 0 }]);

    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);

    for &digest_elem in &digest_ccs_matrices(s) {
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
pub fn bind_me_inputs(
    tr: &mut Poseidon2Transcript,
    me_inputs: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    tr.append_message(b"neo/ccs/me_inputs", b"");
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);

    for me in me_inputs {
        tr.append_fields(b"c_data_in", &me.c.data);
        tr.append_u64s(b"m_in_in", &[me.m_in as u64]);
        for limb in &me.r {
            tr.append_fields(b"r_in", &limb.as_coeffs());
        }
        for yj in &me.y {
            for &y_elem in yj {
                tr.append_fields(b"y_elem", &y_elem.as_coeffs());
            }
        }
    }

    Ok(())
}

/// Sample challenges α, β, γ from transcript
pub fn sample_challenges(
    tr: &mut Poseidon2Transcript,
    ell_d: usize,
    ell: usize,
) -> Result<Challenges, PiCcsError> {
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

fn digest_ccs_matrices<F: Field + PrimeField64>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
    use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};

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

    // Use sparse matrices for efficiency
    for (j, matrix) in s.sparse_matrices.iter().enumerate() {
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(j as u64);
        absorbed += 1;

        // Iterate non-zero elements efficiently using CSR format
        // CSR: row_ptrs[i]..row_ptrs[i+1] are elements of row i
        for row in 0..s.n {
            let start = matrix.row_ptrs[row];
            let end = matrix.row_ptrs[row + 1];
            for k in start..end {
                let col = matrix.col_indices[k];
                let val = matrix.values[k];

                if val != F::ZERO {
                    if absorbed + 3 > 15 {
                        poseidon2.permute_mut(&mut state);
                        absorbed = 0;
                    }

                    state[absorbed] = Goldilocks::from_u64(row as u64);
                    state[absorbed + 1] = Goldilocks::from_u64(col as u64);
                    state[absorbed + 2] = Goldilocks::from_u64(val.as_canonical_u64());
                    absorbed += 3;
                }
            }
        }

        poseidon2.permute_mut(&mut state);
    }

    state[0..4].to_vec()
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

