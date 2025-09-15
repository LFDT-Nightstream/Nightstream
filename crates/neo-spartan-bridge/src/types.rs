use serde::{Serialize, Deserialize};
use std::fmt;

/// NEW: Lean proof structure without the huge VK
#[derive(Serialize, Deserialize)]
pub struct Proof {
    /// Version for future compatibility
    pub version: u8,
    /// Circuit fingerprint for VK lookup
    pub circuit_key: [u8; 32],
    /// VK digest for binding verification 
    pub vk_digest: [u8; 32],
    /// Public IO you expect verifiers to re-encode identically (bridge header + public inputs).
    pub public_io_bytes: Vec<u8>,
    /// ONLY the Spartan2 proof bytes (no VK!)
    pub proof_bytes: Vec<u8>,
}

/// DEPRECATED: Old bundle structure (kept for compatibility during transition)
#[derive(Serialize, Deserialize)]
pub struct ProofBundle {
    /// Spartan2 proof bytes (includes the Hash‑MLE PCS proof inside Spartan2's structure).
    pub proof: Vec<u8>,
    /// Verifier key (serialized) - THIS IS THE 51MB MONSTER
    pub vk: Vec<u8>,
    /// Public IO you expect verifiers to re-encode identically (bridge header + public inputs).
    pub public_io_bytes: Vec<u8>,
}

// Custom Debug implementations to avoid dumping massive binary data
impl fmt::Debug for Proof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Proof")
            .field("version", &self.version)
            .field("circuit_key", &format!("[{} bytes]", self.circuit_key.len()))
            .field("vk_digest", &format!("[{} bytes]", self.vk_digest.len()))
            .field("public_io_bytes", &format!("[{} bytes]", self.public_io_bytes.len()))
            .field("proof_bytes", &format!("[{} bytes]", self.proof_bytes.len()))
            .finish()
    }
}

impl fmt::Debug for ProofBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProofBundle")
            .field("proof", &format!("[{} bytes]", self.proof.len()))
            .field("vk", &format!("[{} bytes]", self.vk.len()))
            .field("public_io_bytes", &format!("[{} bytes]", self.public_io_bytes.len()))
            .finish()
    }
}

impl Proof {
    pub fn new(
        circuit_key: [u8; 32],
        vk_digest: [u8; 32], 
        public_io_bytes: Vec<u8>,
        proof_bytes: Vec<u8>
    ) -> Self {
        Self {
            version: 1,
            circuit_key,
            vk_digest,
            public_io_bytes,
            proof_bytes,
        }
    }
    
    /// Total size of the lean proof (no VK!)
    pub fn total_size(&self) -> usize {
        1 + // version
        32 + // circuit_key
        32 + // vk_digest  
        self.public_io_bytes.len() + 
        self.proof_bytes.len()
    }
}

impl ProofBundle {
    pub fn new_with_vk(proof: Vec<u8>, vk: Vec<u8>, public_io_bytes: Vec<u8>) -> Self {
        Self { proof, vk, public_io_bytes }
    }

    pub fn total_size(&self) -> usize {
        self.proof.len() + self.vk.len() + self.public_io_bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_sizes() {
        let b = ProofBundle::new_with_vk(vec![1,2], vec![3], vec![4,5,6]);
        assert_eq!(b.total_size(), 2+1+3);
    }
}