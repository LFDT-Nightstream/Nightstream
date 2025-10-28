//! Transcript module: Binding header/instances and sampling challenges
//!
//! # Paper Reference  
//! Section 4.4, Step 0-1:
//! - Step 0: Bind all public data to transcript before sampling randomness
//! - Step 1: Sample challenges α, β, γ from transcript
//!
//! This ensures Fiat-Shamir security: all public data is committed before
//! any randomness is derived, preventing selective failure attacks.

#![allow(non_snake_case)]

use neo_transcript::{Transcript, Poseidon2Transcript, labels as tr_labels};
use neo_ccs::{CcsStructure, McsInstance, MeInstance, MatRef, SparsePoly};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{F, K, KExtensions};
use p3_field::{Field, PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use crate::error::PiCcsError;

/// Challenges sampled in Step 1 of the protocol
#[derive(Debug, Clone)]
pub struct Challenges {
    /// α ∈ K^{log d} - for Ajtai dimension
    pub alpha: Vec<K>,
    /// β = (β_a, β_r) ∈ K^{log(dn)} split into Ajtai and row parts
    pub beta_a: Vec<K>,
    pub beta_r: Vec<K>,
    /// γ ∈ K - random linear combination weight
    pub gamma: K,
}

/// Bind header and MCS instances to transcript (Step 0)
///
/// # Paper Reference
/// Before sampling any challenges, we must bind:
/// - Protocol version and security parameters
/// - CCS structure (n, m, t, matrices, polynomial f)
/// - All MCS instances (commitments c, public inputs x, m_in values)
pub fn bind_header_and_instances(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    ell: usize,
    d_sc: usize,
    slack_bits: i32,
) -> Result<(), PiCcsError> {
    tr.append_message(tr_labels::PI_CCS, b"");
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[
        64,
        s.n as u64,  // Use dims directly rather than params.s_supported()
        params.lambda as u64,
        ell as u64,
        d_sc as u64,
        slack_bits.unsigned_abs() as u64,
    ]);
    tr.append_message(b"ccs/slack_sign", &[if slack_bits >= 0 { 1 } else { 0 }]);

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

/// Bind ME inputs to transcript (Step 0)
///
/// For k > 1 (folding with prior ME instances), bind all input ME instances
/// before sampling challenges.
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

/// Sample challenges α, β, γ from transcript (Step 1)
///
/// # Paper Reference
/// Section 4.4, Step 1:
/// - α ← K^{log d}: Ajtai randomness
/// - β ← K^{log(dn)}: Split into (β_a, β_r) for two-axis folding
/// - γ ← K: Random linear combination weight
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

/// Deterministic digest of CCS matrices for transcript binding
///
/// Uses Poseidon2 sponge to absorb matrix structure and non-zero entries
/// in canonical order. This ensures identical CCS structures produce
/// identical digests regardless of internal representation.
pub(crate) fn digest_ccs_matrices<F: Field + PrimeField64>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
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

    for (j, matrix) in s.matrices.iter().enumerate() {
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(j as u64);
        absorbed += 1;

        let mat_ref = MatRef::from_mat(matrix);

        for row in 0..s.n {
            let row_slice = mat_ref.row(row);
            for (col, &val) in row_slice.iter().enumerate() {
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

/// Absorb sparse polynomial f into transcript
///
/// Binds the CCS polynomial structure and all terms in deterministic order
pub(crate) fn absorb_sparse_polynomial(tr: &mut Poseidon2Transcript, f: &SparsePoly<F>) {
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

