//! Transcript Replay Utilities
//!
//! Helper functions for replaying the Π-CCS transcript to extract verification data.
//! These are used for debugging and testing, allowing extraction of intermediate
//! challenge values without full verification.

#![allow(non_snake_case)]

use neo_transcript::{Transcript, Poseidon2Transcript, labels as tr_labels};
use neo_ccs::{CcsStructure, McsInstance, MeInstance};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K, KExtensions};
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use crate::error::PiCcsError;
use crate::pi_ccs::{PiCcsProof, digest_ccs_matrices, absorb_sparse_polynomial};
use crate::sumcheck::verify_sumcheck_rounds;

/// Data derived from the Π-CCS transcript tail used by the verifier.
#[derive(Debug, Clone)]
pub struct TranscriptTail {
    pub _wr: K,
    pub r: Vec<K>,
    pub alphas: Vec<K>,
    pub running_sum: K,
    /// The claimed sum over the hypercube (T in the paper), used to verify satisfiability
    pub initial_sum: K,
}

/// Replay the Π-CCS transcript to derive the tail (wr, r, alphas).
///
/// This is primarily used for debugging and testing. It replays the transcript
/// to extract intermediate values without performing full verification.
///
/// # Note
/// This currently assumes k=1 (no ME inputs). For k>1, use the full verifier.
pub fn pi_ccs_derive_transcript_tail(
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    proof: &PiCcsProof,
) -> Result<TranscriptTail, PiCcsError> {
    let mut tr = Poseidon2Transcript::new(b"neo/fold");
    tr.append_message(tr_labels::PI_CCS, b"");
    
    // Header (same as in pi_ccs_verify)
    // Use shared helper to ensure transcript replay matches prover/verifier dimensions
    let crate::pi_ccs::context::Dims { ell_d, ell_n: _, ell, d_sc } = 
        crate::pi_ccs::context::build_dims_and_policy(params, s)?;
    
    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"ccs/slack_sign", &[if ext.slack_bits >= 0 {1} else {0}]);

    // Instances
    tr.append_message(b"neo/ccs/instances", b"");
    tr.append_u64s(b"dims", &[s.n as u64, s.m as u64, s.t() as u64]);
    let matrix_digest = digest_ccs_matrices(s);
    for &digest_elem in &matrix_digest { 
        tr.append_fields(b"mat_digest", &[F::from_u64(digest_elem.as_canonical_u64())]); 
    }
    absorb_sparse_polynomial(&mut tr, &s.f);
    for inst in mcs_list.iter() {
        tr.append_fields(b"x", &inst.x);
        tr.append_u64s(b"m_in", &[inst.m_in as u64]);
        tr.append_fields(b"c_data", &inst.c.data);
    }

    // NOTE: ME inputs not absorbed here since this is a simplified helper
    // In real usage, this should match prove/verify transcript exactly
    // For now, assume no ME inputs (empty slice) for initial fold
    tr.append_message(b"neo/ccs/me_inputs", b"");
    tr.append_u64s(b"me_count", &[0u64]); // Initial fold: k=1, no ME inputs

    // Sample challenges (mirror prove/verify)
    tr.append_message(b"neo/ccs/chals/v1", b"");
    let _alpha_vec: Vec<K> = (0..ell_d)
        .map(|_| { 
            let ch = tr.challenge_fields(b"chal/k", 2); 
            neo_math::from_complex(ch[0], ch[1]) 
        })
        .collect();
    let _beta: Vec<K> = (0..ell)
        .map(|_| { 
            let ch = tr.challenge_fields(b"chal/k", 2); 
            neo_math::from_complex(ch[0], ch[1]) 
        })
        .collect();
    let _gamma: K = {
        let ch_g = tr.challenge_fields(b"chal/k", 2);
        neo_math::from_complex(ch_g[0], ch_g[1])
    };

    // Derive r by verifying rounds (structure only)
    let d_round = d_sc;
    
    // Use the prover-carried initial sum when present; else derive from round 0
    let claimed_initial = match proof.sc_initial_sum {
        Some(s) => s,
        None => {
            if let Some(round0) = proof.sumcheck_rounds.get(0) {
                use crate::sumcheck::poly_eval_k;
                poly_eval_k(round0, K::ZERO) + poly_eval_k(round0, K::ONE)
            } else {
                K::ZERO
            }
        }
    };

    // Bind initial_sum BEFORE rounds to match prover/verifier transcript layout
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    let (r, running_sum, ok_rounds) = verify_sumcheck_rounds(
        &mut tr, 
        d_round, 
        claimed_initial, 
        &proof.sumcheck_rounds
    );
    
    if !ok_rounds {
        #[cfg(feature = "debug-logs")]
        eprintln!(
            "[pi-ccs] rounds invalid: expected degree ≤ {}, got {} rounds", 
            d_round, 
            proof.sumcheck_rounds.len()
        );
        return Err(PiCcsError::SumcheckError("rounds invalid".into()));
    }

    // Keep transcript layout; wr no longer used by verifier semantics
    let _wr = K::ONE;
    
    #[cfg(feature = "debug-logs")]
    eprintln!(
        "[pi-ccs] derive_tail: s.n={}, ell={}, d_sc={}, outputs={}, rounds={}", 
        s.n, ell, d_sc, mcs_list.len(), proof.sumcheck_rounds.len()
    );
    
    Ok(TranscriptTail { 
        _wr, 
        r, 
        alphas: Vec::new(), 
        running_sum, 
        initial_sum: claimed_initial 
    })
}

/// Compute the terminal claim from Π_CCS outputs given wr or generic CCS terminal.
///
/// This computes the expected Q(r) value from the ME outputs, which can be compared
/// against the running_sum from sum-check for verification.
///
/// # Note
/// This is a legacy function that ignores `_wr` and always uses generic CCS semantics.
pub fn pi_ccs_compute_terminal_claim_r1cs_or_ccs(
    s: &CcsStructure<F>,
    _wr: K,
    alphas: &[K],
    out_me: &[MeInstance<Cmt, F, K>],
) -> K {
    // Unified semantics: ignore wr; always compute generic CCS terminal
    let mut expected_q_r = K::ZERO;
    for (inst_idx, me_inst) in out_me.iter().enumerate() {
        let f_eval = s.f.eval_in_ext::<K>(&me_inst.y_scalars);
        expected_q_r += alphas[inst_idx] * f_eval;
    }
    expected_q_r
}

