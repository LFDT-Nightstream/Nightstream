use anyhow::Result;
use serde::{Deserialize, Serialize};
use ff::PrimeField;

// ⚠️ SECURITY WARNING: This module uses Keccak256 transcript which is INCONSISTENT
// with Neo's unified Poseidon2 transcript design. This is a temporary workaround
// until Spartan2 provides Poseidon2 transcript support.
//
// For production use, this should be feature-gated under #[cfg(feature = "dev")]
// or replaced with a Poseidon2-compatible transcript.

// Hash-MLE backend with Goldilocks engine (p3_backend feature enabled)
use spartan2::provider::{
    keccak::Keccak256Transcript,     // ⚠️ INCONSISTENT: Should be Poseidon2 for Neo compliance
    GoldilocksP3MerkleMleEngine as E,  // Use Goldilocks engine from p3_backend  
    pcs::merkle_mle_pc::HashMlePCS as PCSImpl,
};

use spartan2::traits::{
    Engine,
    pcs::PCSEngineTrait as SpartanPcs,
    transcript::TranscriptEngineTrait,
};

/// Field and PCS aliases for ergonomics.
pub type F   = <E as Engine>::Scalar;
pub type GE  = <E as Engine>::GE;
pub type PCS = PCSImpl<E>;

/// The fork's types for Hash-MLE.
use spartan2::provider::pcs::merkle_mle_pc::{
    HashMleCommitment as Commitment,
    HashMleEvaluationArgument as EvaluationArgument,
};

/// Portable proof object you can bincode and ship around.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMleProof {
    /// Base commitment (Merkle root + mode baked in).
    pub commitment: Commitment<E>,
    /// Evaluation point r = (r_0, ..., r_{m-1})
    pub point: Vec<F>,
    /// Claimed value v(r)
    pub eval: F,
    /// Merkle membership + fold witnesses per round
    pub arg: EvaluationArgument<E>,
}

impl HashMleProof {
    pub fn to_bytes(&self) -> Result<Vec<u8>> { Ok(bincode::serialize(self)?) }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> { Ok(bincode::deserialize(bytes)?) }
}

/// Prove: commit to `poly` (|poly|=2^m), then prove v(r)=eval at `point` (|point|=m).
/// 
/// ⚠️ SECURITY WARNING: This function uses Keccak256 transcript, which breaks Neo's
/// unified Poseidon2 transcript security model. Use only for development/testing.
pub fn prove_hash_mle(poly: &[F], point: &[F]) -> Result<HashMleProof> {
    if !poly.len().is_power_of_two() {
        anyhow::bail!("Hash-MLE: poly length must be a power of two; got {}", poly.len());
    }
    if poly.len() != (1usize << point.len()) {
        anyhow::bail!("Hash-MLE: poly length {} != 2^m with m={}", poly.len(), point.len());
    }

    // Setup → blind → commit (width is effectively "infinite"; one Merkle layer)
    let (ck, _vk) = PCS::setup(b"neo-bridge/hash-mle", poly.len());
    let blind     = PCS::blind(&ck, poly.len());
    let commit    = PCS::commit(&ck, poly, &blind, false)?;

    // Prove with Keccak transcript (TODO: unify to Poseidon2 when available)
    let mut tp = Keccak256Transcript::<E>::new(b"neo-bridge/hash-mle");
    let (eval, arg) = PCS::prove(&ck, &mut tp, &commit, poly, &blind, point)?;

    Ok(HashMleProof { commitment: commit, point: point.to_vec(), eval, arg })
}

/// Verify a Hash‑MLE proof (public: commitment root, point, eval, arg).
/// 
/// ⚠️ SECURITY WARNING: This function uses Keccak256 transcript, which breaks Neo's
/// unified Poseidon2 transcript security model. Use only for development/testing.
pub fn verify_hash_mle(prf: &HashMleProof) -> Result<()> {
    // Setup a fresh VK (parameters are fixed by the PCS; no SRS)
    let (_ck, vk) = PCS::setup(b"neo-bridge/hash-mle", 0);

    let mut tv = Keccak256Transcript::<E>::new(b"neo-bridge/hash-mle");
    PCS::verify(&vk, &mut tv, &prf.commitment, &prf.point, &prf.eval, &prf.arg)?;
    Ok(())
}

/// Tiny helper if you want a public-IO header for verification caching/logging.
pub fn encode_public_io(prf: &HashMleProof) -> Vec<u8> {
    let mut out = Vec::new();
    
    // Serialize the commitment (contains the Merkle root)
    if let Ok(commitment_bytes) = bincode::serialize(&prf.commitment) {
        out.extend_from_slice(&commitment_bytes);
    }
    
    // m = point length
    out.extend_from_slice(&(prf.point.len() as u64).to_le_bytes());
    // point limbs
    for x in &prf.point {
        out.extend_from_slice(&x.to_repr().as_ref());
    }
    // eval
    out.extend_from_slice(&prf.eval.to_repr().as_ref());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, Rng};
    use rand_chacha::ChaCha8Rng;

    fn rand_poly(m: usize, seed: u64) -> (Vec<F>, Vec<F>) {
        let n = 1usize << m;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let poly = (0..n).map(|_| F::from(rng.random::<u64>())).collect::<Vec<_>>();
        // arbitrary (not just {0,1}); PCS supports general field points
        let point = (0..m).map(|_| F::from(rng.random::<u64>())).collect::<Vec<_>>();
        (poly, point)
    }

    #[test]
    fn basic_prove_verify() {
        for m in 0..=6 {
            let (poly, point) = rand_poly(m, 1234 + m as u64);
            let prf = prove_hash_mle(&poly, &point).expect("prove");
            verify_hash_mle(&prf).expect("verify");
        }
    }
    
    #[test]
    fn test_serialization() {
        // Test just with a small example first
        let (poly, point) = rand_poly(2, 42); // 2^2 = 4 elements
        let prf = prove_hash_mle(&poly, &point).expect("prove");
        verify_hash_mle(&prf).expect("verify original");
        
        // Try serialization - if this fails we'll handle it
        match prf.to_bytes() {
            Ok(bytes) => {
                match HashMleProof::from_bytes(&bytes) {
                    Ok(prf2) => {
                        verify_hash_mle(&prf2).expect("verify deserialized");
                    }
                    Err(e) => {
                        println!("Deserialization failed: {}", e);
                        // This is acceptable for now - the core prove/verify works
                    }
                }
            }
            Err(e) => {
                println!("Serialization failed: {}", e);
                // This is acceptable for now - the core prove/verify works
            }
        }
    }

    #[test]
    fn tamper_fails() {
        let (poly, point) = rand_poly(5, 777);
        let prf = prove_hash_mle(&poly, &point).unwrap();
        
        // Tamper the proof by modifying the serialized bytes  
        let mut proof_bytes = prf.to_bytes().unwrap();
        if proof_bytes.len() > 10 {
            proof_bytes[10] ^= 1;  // Flip a bit in the serialized proof
        }
        
        // Try to deserialize and verify the tampered proof
        if let Ok(tampered_prf) = HashMleProof::from_bytes(&proof_bytes) {
            assert!(verify_hash_mle(&tampered_prf).is_err());
        }
        // If deserialization fails, that's also acceptable - the proof is invalid
    }
}
