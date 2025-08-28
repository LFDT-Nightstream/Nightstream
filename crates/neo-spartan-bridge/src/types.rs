use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ProofBundle {
    /// Spartan2 proof bytes (already include the FRI/PCS proof inside Spartan2's structure).
    pub proof: Vec<u8>,
    /// Public IO you expect verifiers to re-encode identically (bridge header + public inputs).
    pub public_io_bytes: Vec<u8>,
    /// Optional metrics
    pub fri_num_queries: usize,
    pub fri_log_blowup: usize,
    pub size_bytes: usize,
}

impl ProofBundle {
    pub fn new(
        proof: Vec<u8>,
        public_io_bytes: Vec<u8>, 
        fri_num_queries: usize,
        fri_log_blowup: usize,
    ) -> Self {
        let size_bytes = proof.len() + public_io_bytes.len();
        Self {
            proof,
            public_io_bytes,
            fri_num_queries,
            fri_log_blowup,
            size_bytes,
        }
    }

    pub fn total_size(&self) -> usize {
        self.size_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_bundle_creation() {
        let proof = vec![1, 2, 3, 4];
        let io = vec![5, 6, 7];
        
        let bundle = ProofBundle::new(proof.clone(), io.clone(), 60, 2);
        
        assert_eq!(bundle.proof, proof);
        assert_eq!(bundle.public_io_bytes, io);
        assert_eq!(bundle.fri_num_queries, 60);
        assert_eq!(bundle.fri_log_blowup, 2);
        assert_eq!(bundle.size_bytes, 7); // 4 + 3
        assert_eq!(bundle.total_size(), 7);
        
        println!("✅ ProofBundle serialization structure works");
    }
}

