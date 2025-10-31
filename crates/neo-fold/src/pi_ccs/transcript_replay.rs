//! Transcript Replay Utilities
//!
//! Helper functions for replaying the Π-CCS transcript to extract verification data.
//! These are used for debugging and testing, allowing extraction of intermediate
//! challenge values without full verification.

#![allow(non_snake_case)]

use neo_transcript::{Transcript, Poseidon2Transcript};
use neo_ccs::{CcsStructure, McsInstance, MeInstance};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K, KExtensions};
use p3_field::PrimeCharacteristicRing;
use crate::error::PiCcsError;
use crate::pi_ccs::PiCcsProof;
use crate::pi_ccs::transcript::bind_me_inputs;
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
/// Bind both MCS and ME inputs so that the challenges match the prover’s.
pub fn pi_ccs_derive_transcript_tail_with_me_inputs_and_label(
    domain_label: &'static [u8],
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<TranscriptTail, PiCcsError> {
    let mut tr = Poseidon2Transcript::new(domain_label);
    // Header and instances: use the same helper as prover/verify for perfect parity
    let crate::pi_ccs::context::Dims { ell_d, ell_n: _, ell, d_sc } = 
        crate::pi_ccs::context::build_dims_and_policy(params, s)?;
    crate::pi_ccs::transcript::bind_header_and_instances(&mut tr, params, s, mcs_list, ell, d_sc, 0)?;

    // Bind ME inputs exactly like prover/verifier so replayed challenges match
    bind_me_inputs(&mut tr, me_inputs)?;
    #[cfg(feature = "debug-logs")]
    {
        use p3_field::PrimeField64;
        eprintln!("[replay] binding me_count = {}", me_inputs.len());
        if let Some(me0) = me_inputs.get(0) {
            let preview: Vec<u64> = me0.c.data.iter().take(2).map(|f| f.as_canonical_u64()).collect();
            eprintln!("[replay] me[0].c preview = {:?}", preview);
        }
    }

    // Sample challenges (mirror prove/verify)
    let _ch = crate::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell)?;

    // Derive r by verifying rounds (structure only)
    let d_round = d_sc; // degree bound for each round
    
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
            "[pi-ccs] rounds invalid: degree bound ≤ {}, rounds = {}", 
            d_round, proof.sumcheck_rounds.len()
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

/// Convenience wrapper using the default domain label used broadly in this crate (neo/fold).
pub fn pi_ccs_derive_transcript_tail_with_me_inputs(
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<TranscriptTail, PiCcsError> {
    pi_ccs_derive_transcript_tail_with_me_inputs_and_label(b"neo/fold", params, s, mcs_list, me_inputs, proof)
}

/// Back-compat wrapper: derive tail assuming k=1 (no ME inputs).
pub fn pi_ccs_derive_transcript_tail(
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    proof: &PiCcsProof,
) -> Result<TranscriptTail, PiCcsError> {
    pi_ccs_derive_transcript_tail_with_me_inputs(params, s, mcs_list, &[], proof)
}

/// Derive the tail from an existing transcript that already has header, instances,
/// ME inputs, and challenges bound (i.e., after `sample_challenges`).
///
/// This matches the common testing pattern where the transcript is constructed
/// explicitly in the test using the same helpers as proving/verifying code.
pub fn pi_ccs_derive_transcript_tail_from_bound_transcript(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    proof: &PiCcsProof,
) -> Result<TranscriptTail, PiCcsError> {
    let crate::pi_ccs::context::Dims { ell_d: _, ell_n: _, ell: _, d_sc: _ } =
        crate::pi_ccs::context::build_dims_and_policy(params, s)?;

    // Use prover-carried initial sum if present; else derive from round 0
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

    // Bind initial sum to the provided transcript and verify rounds to derive r.
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());

    // Derive r and running_sum directly (tolerant mode): mirror verifier logic
    // but do not fail on consistency checks to enable round-trip in tests.
    use crate::sumcheck::poly_eval_k;
    let mut running_sum = claimed_initial;
    let mut r_all = Vec::with_capacity(proof.sumcheck_rounds.len());
    for (_round_idx, coeffs) in proof.sumcheck_rounds.iter().enumerate() {
        // (Optional) degree check omitted here; replay is tolerant in tests.
        // Append round coeffs and sample r_i
        tr.append_message(b"neo/ccs/round", b"");
        let c0 = coeffs[0].as_coeffs();
        tr.append_fields(b"round/coeffs", &c0);
        for c in coeffs.iter().skip(1) {
            tr.append_fields(b"round/coeffs", &c.as_coeffs());
        }
        let ch = tr.challenge_fields(b"chal/k", 2);
        let r_i = neo_math::from_complex(ch[0], ch[1]);

        running_sum = poly_eval_k(coeffs, r_i);
        r_all.push(r_i);
    }

    Ok(TranscriptTail { _wr: K::ONE, r: r_all, alphas: Vec::new(), running_sum, initial_sum: claimed_initial })
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
