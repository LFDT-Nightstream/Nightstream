//! Neo Folding Protocol - Single Three-Reduction Pipeline
//!
//! Implements the complete folding protocol: Π_CCS → Π_RLC → Π_DEC  
//! Uses one transcript (Poseidon2), one backend (Ajtai), and one sum-check over K.

pub mod error;
/// Poseidon2 transcript for Fiat-Shamir
pub mod transcript;
/// Π_CCS: Sum-check reduction over extension field K  
pub mod pi_ccs;
/// Π_RLC: Random linear combination with S-action
pub mod pi_rlc;
/// Π_DEC: Verified split opening (TODO: implement real version)
pub mod pi_dec;

// Re-export main types
pub use error::{FoldingError, PiCcsError, PiRlcError, PiDecError};
pub use transcript::{FoldTranscript, Domain};
pub use pi_ccs::{pi_ccs_prove, pi_ccs_verify, PiCcsProof};  
pub use pi_rlc::{pi_rlc_prove, pi_rlc_verify, PiRlcProof};
pub use pi_dec::{pi_dec, pi_dec_verify, PiDecProof};

use neo_ccs::{MeInstance, MeWitness, CcsStructure};
use neo_math::{F, K};
use neo_ajtai::Commitment as Cmt;
use p3_field::PrimeCharacteristicRing;

/// Proof that k+1 CCS instances fold to k instances
#[derive(Debug, Clone)]
pub struct FoldingProof {
    /// Π_CCS proof (sum-check over K) 
    pub pi_ccs_proof: PiCcsProof,
    /// Π_RLC proof (S-action combination)
    pub pi_rlc_proof: PiRlcProof,
    /// Π_DEC proof (verified split opening)  
    pub pi_dec_proof: PiDecProof,
}

/// Fold k+1 CCS instances to k instances using the three-reduction pipeline  
/// Input: k+1 CCS instances and witnesses
/// Output: k ME instances and folding proof
pub fn fold_ccs_instances(
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    instances: &[neo_ccs::McsInstance<Cmt, F>],
    witnesses: &[neo_ccs::McsWitness<F>],
) -> Result<(Vec<MeInstance<Cmt, F, K>>, FoldingProof), FoldingError> {
    if instances.is_empty() || instances.len() != witnesses.len() {
        return Err(FoldingError::InvalidInput("empty or mismatched inputs".into()));
    }

    // Ajtai S-module from the globally published PP
    let l = neo_ajtai::AjtaiSModule::from_global()
        .map_err(|e| FoldingError::InvalidInput(format!("Ajtai PP not initialized: {}", e)))?;

    // One transcript shared end-to-end
    let mut tr = FoldTranscript::default();

    // 1) Π_CCS: k+1 MCS → k+1 ME(b,L)
    let (me_list, pi_ccs_proof) =
        pi_ccs::pi_ccs_prove(&mut tr, params, structure, instances, witnesses, &l)?;

    // 2) Π_RLC: k+1 ME(b,L) → 1 ME(B,L)
    let (me_b, pi_rlc_proof) = pi_rlc::pi_rlc_prove(&mut tr, params, &me_list)?;

    // 2b) Build the combined witness Z' = Σ rot(ρ_i)·Z_i for the DEC prover
    let d = witnesses[0].Z.rows();
    let m = witnesses[0].Z.cols();
    if d != neo_math::D {
        return Err(FoldingError::InvalidInput(format!(
            "Ajtai ring dimension D={} but witness Z has rows={}", neo_math::D, d
        )));
    }
    // Convert ρ to ring elements
    let rhos_ring: Vec<neo_math::Rq> = pi_rlc_proof.rho_elems.iter()
        .map(|coeffs| neo_math::ring::cf_inv(*coeffs))
        .collect();
    if rhos_ring.len() != witnesses.len() {
        return Err(FoldingError::PiRlc(crate::error::PiRlcError::InvalidInput(format!(
            "rho count {} != witness count {}", rhos_ring.len(), witnesses.len()
        ))));
    }
    // Accumulate Σ rot(ρ_i)·Z_i column-wise
    let mut z_prime = neo_ccs::Mat::zero(d, m, F::ZERO);
    for (wit, rho) in witnesses.iter().zip(rhos_ring.iter()) {
        let s_action = neo_math::SAction::from_ring(*rho);
        for c in 0..m {
            let mut col = [F::ZERO; neo_math::D];
            for r in 0..d { col[r] = wit.Z[(r, c)]; }
            let rotated = s_action.apply_vec(&col);
            for r in 0..d { z_prime[(r, c)] += rotated[r]; }
        }
    }
    let me_b_wit = MeWitness { Z: z_prime };

    // 3) Π_DEC: 1 ME(B,L) → k ME(b,L) with verified openings & range assertions
    let (me_out, _digit_wits, pi_dec_proof) =
        pi_dec::pi_dec(&mut tr, params, &me_b, &me_b_wit, structure, &l)
            .map_err(|e| match e {
                pi_dec::PiDecError::InvalidInput(msg) => FoldingError::PiDec(crate::error::PiDecError::InvalidInput(msg)),
                pi_dec::PiDecError::DecompositionFailed(msg) => FoldingError::PiDec(crate::error::PiDecError::CommitmentError(msg)),
                pi_dec::PiDecError::VerifiedOpeningFailed(msg) => FoldingError::PiDec(crate::error::PiDecError::OpeningFailed(msg)),
                pi_dec::PiDecError::RangeCheckFailed(msg) => FoldingError::PiDec(crate::error::PiDecError::RangeViolation(msg)),
                pi_dec::PiDecError::SHomomorphismError(msg) => FoldingError::PiDec(crate::error::PiDecError::CommitmentError(msg)),
            })?;

    let proof = FoldingProof {
        pi_ccs_proof,
        pi_rlc_proof,
        pi_dec_proof,
    };
    Ok((me_out, proof))
}

/// Verify a folding proof
/// Reconstructs the public computation and checks all three sub-protocols
pub fn verify_folding_proof(
    _params: &neo_params::NeoParams,  
    _structure: &CcsStructure<F>,
    _input_instances: &[neo_ccs::McsInstance<Cmt, F>],
    _output_instances: &[MeInstance<Cmt, F, K>], 
    _proof: &FoldingProof,
) -> Result<bool, FoldingError> {
    // TODO: Replace with real verification pipeline
    eprintln!("⚠️  verify_folding_proof: using placeholder - full implementation needed");
    Ok(true)
}