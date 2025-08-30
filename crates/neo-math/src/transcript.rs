//! Simple Transcript implementation for backward compatibility
//! 
//! Note: According to STRUCTURE.md, transcript logic should live in neo-fold.
//! This is a minimal compatibility shim until the proper migration is complete.
//! 
//! ⚠️ DEPRECATED: This transcript uses DefaultHasher which is NOT cryptographically secure.
//! Use neo_fold::transcript::FoldTranscript for all cryptographic applications.

use std::collections::HashMap;
use p3_field::PrimeCharacteristicRing;

/// Simple transcript for Fiat-Shamir (backward compatibility only)
/// 
/// ⚠️ DEPRECATED: This uses DefaultHasher which is NOT cryptographically secure.
/// Use `neo_fold::transcript::FoldTranscript` for all production code.
#[deprecated(since = "0.1.0", note = "Use neo_fold::transcript::FoldTranscript instead")]
#[derive(Clone, Debug)]
pub struct Transcript {
    state: Vec<u8>,
    challenges: HashMap<String, Vec<u8>>,
}

#[allow(deprecated)]
impl Transcript {
    /// Create new transcript with protocol name
    pub fn new(protocol: &str) -> Self {
        let mut state = Vec::new();
        state.extend_from_slice(protocol.as_bytes());
        Self {
            state,
            challenges: HashMap::new(),
        }
    }

    /// Absorb bytes into transcript
    pub fn absorb_bytes(&mut self, label: &str, data: &[u8]) {
        self.state.extend_from_slice(label.as_bytes());
        self.state.extend_from_slice(data);
    }

    /// Absorb tag into transcript
    pub fn absorb_tag(&mut self, tag: &str) {
        self.state.extend_from_slice(tag.as_bytes());
    }

    /// Challenge bytes (simplified)
    pub fn challenge_bytes(&mut self, label: &str, output: &mut [u8]) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Simple hash-based challenge generation (not cryptographically secure)
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        label.hash(&mut hasher);
        let hash = hasher.finish();
        
        for (i, byte) in output.iter_mut().enumerate() {
            *byte = ((hash >> (8 * (i % 8))) & 0xFF) as u8;
        }
        
        // Store for reproducibility
        self.challenges.insert(label.to_string(), output.to_vec());
    }

    /// Challenge base field element (backward compatibility)
    pub fn challenge_base(&mut self, label: &str) -> crate::Fq {
        let mut bytes = [0u8; 8];
        self.challenge_bytes(label, &mut bytes);
        crate::Fq::from_u64(u64::from_le_bytes(bytes))
    }
}
