//! Shared digest and commitment utilities
//!
//! These functions handle hashing and serialization for accumulators and commitments.

use crate::F;
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use neo_ccs::crypto::poseidon2_goldilocks as p2;

use super::types::Accumulator;

/// Compute digest of commitment coordinates using Poseidon2
#[allow(dead_code)]
pub fn digest_commit_coords(coords: &[F]) -> [u8; 32] {
    let p = p2::permutation();

    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0usize;
    const RATE: usize = p2::RATE;

    // Domain separation
    for &b in b"neo/commitment-digest/v1" {
        if absorbed == RATE { 
            st = p.permute(st); 
            absorbed = 0; 
        }
        st[absorbed] = Goldilocks::from_u64(b as u64); 
        absorbed += 1;
    }
    
    // Absorb commitment coordinates
    for &x in coords {
        if absorbed == RATE { 
            st = p.permute(st); 
            absorbed = 0; 
        }
        st[absorbed] = Goldilocks::from_u64(x.as_canonical_u64()); 
        absorbed += 1;
    }
    
    // Final permutation and pad
    if absorbed < RATE {
        st[absorbed] = Goldilocks::ONE; // domain separator  
    }
    st = p.permute(st);

    // Extract digest bytes (first 4 limbs = 32 bytes)
    let mut digest = [0u8; 32];
    for (i, elem) in st[..4].iter().enumerate() {
        digest[i*8..(i+1)*8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
    }
    digest
}

/// Serialize accumulator for commitment binding
fn serialize_accumulator_for_commitment(accumulator: &Accumulator) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    
    // Step counter (8 bytes)
    bytes.extend_from_slice(&accumulator.step.to_le_bytes());
    
    // c_z_digest (32 bytes)
    bytes.extend_from_slice(&accumulator.c_z_digest);
    
    // y_compact length + elements
    bytes.extend_from_slice(&(accumulator.y_compact.len() as u64).to_le_bytes());
    for &y in &accumulator.y_compact {
        bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes());
    }
    
    Ok(bytes)
}

/// Compute accumulator digest as field elements for in-circuit use
pub fn compute_accumulator_digest_fields(acc: &Accumulator) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    // Reuse existing serializer for exact byte encoding
    let bytes = serialize_accumulator_for_commitment(acc)?;
    // Hash to 4 field elements (32 bytes) and return them as F limbs
    let digest_felts = p2::poseidon2_hash_packed_bytes(&bytes);
    let mut out = Vec::with_capacity(p2::DIGEST_LEN);
    for x in digest_felts { out.push(F::from_u64(x.as_canonical_u64())); }
    Ok(out)
}

