// crates/neo-spartan-bridge/src/hash_mle.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};
use ff::PrimeField;
use spartan2::traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait};
use spartan2::provider::{
    GoldilocksP3MerkleMleEngine as E,
    pcs::merkle_mle_pc_p3::HashMlePcsP3 as PCSImpl,
};

pub type F = <E as Engine>::Scalar;
pub type PCS = PCSImpl<E>;

use spartan2::provider::pcs::merkle_mle_pc::{
  HashMleCommitment as Commitment, 
  HashMleEvaluationArgument as EvaluationArgument,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMleProof { 
    pub commitment: Commitment<E>, 
    pub point: Vec<F>, 
    pub eval: F, 
    pub arg: EvaluationArgument<E> 
}

impl HashMleProof {
    pub fn to_bytes(&self) -> Result<Vec<u8>> { 
        Ok(bincode::serialize(self)?) 
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> { 
        Ok(bincode::deserialize(bytes)?) 
    }
}

pub fn prove_hash_mle(poly: &[F], point: &[F]) -> Result<HashMleProof> {
    if !poly.len().is_power_of_two() { 
        anyhow::bail!("poly len must be power of two"); 
    }
    if poly.len() != (1usize << point.len()) { 
        anyhow::bail!("poly len {} != 2^{}", poly.len(), point.len()); 
    }
    
    let (ck, _vk) = PCS::setup(b"neo/hash-mle/poseidon2", poly.len());
    let r = PCS::blind(&ck, poly.len());
    let c = PCS::commit(&ck, poly, &r, true)?;
    
    let mut t = <E as Engine>::TE::new(b"neo/hash-mle/poseidon2");
    
    // SECURITY: Bind commitment and point into Fiat-Shamir transcript 
    // This prevents floating public input attacks by making the challenges depend on the statement
    t.absorb(b"neo-hash-mle-commitment", &c);
    
    // Use proper transcript representation for point coordinates  
    for x in point { 
        t.absorb(b"neo-hash-mle-point-coord", x);
    }
    
    let (eval, arg) = PCS::prove(&ck, &mut t, &c, poly, &r, point)?;
    
    Ok(HashMleProof { commitment: c, point: point.to_vec(), eval, arg })
}

pub fn verify_hash_mle(prf: &HashMleProof) -> Result<()> {
    let (_ck, vk) = PCS::setup(b"neo/hash-mle/poseidon2", 0);
    let mut t = <E as Engine>::TE::new(b"neo/hash-mle/poseidon2");
    
    // SECURITY: Bind commitment and point into Fiat-Shamir transcript 
    // This must match the prover-side transcript binding exactly
    t.absorb(b"neo-hash-mle-commitment", &prf.commitment);
    
    // Use proper transcript representation for point coordinates  
    for x in &prf.point { 
        t.absorb(b"neo-hash-mle-point-coord", x);
    }
    
    PCS::verify(&vk, &mut t, &prf.commitment, &prf.point, &prf.eval, &prf.arg)
        .map_err(|e| anyhow::anyhow!("Hash-MLE verification failed: {:?}", e))
}

pub fn encode_public_io(prf: &HashMleProof) -> Vec<u8> {
    let mut out = Vec::new();
    let cb = bincode::serialize(&prf.commitment).expect("serialize");
    out.extend_from_slice(&(cb.len() as u64).to_le_bytes()); 
    out.extend_from_slice(&cb);
    out.extend_from_slice(&(prf.point.len() as u64).to_le_bytes());
    for x in &prf.point { 
        out.extend_from_slice(&x.to_repr().as_ref()); 
    }
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
        
        let bytes = prf.to_bytes().expect("serialize");
        let prf2 = HashMleProof::from_bytes(&bytes).expect("deserialize");
        verify_hash_mle(&prf2).expect("verify deserialized");
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