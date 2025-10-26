//! Witness disclosure policies

use serde::{Serialize, Deserialize};
use super::types::{WitnessInfo, GradientContribution, SparseRow};
use neo_math::F;
use p3_field::PrimeField64;

/// Witness disclosure policy (leak-aware)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessPolicy {
    /// Only indices touching the failing row (minimal disclosure)
    RowSupportOnly,
    
    /// Indices with large |∂r/∂z_i| (smart disclosure)
    GradientSlice {
        threshold: String,
        top_k: usize,
    },
    
    /// Full witness (use with caution!)
    FullWitness,
    
    /// Full witness with private elements redacted (BLAKE3 hashes)
    RedactedFull,
}

impl WitnessInfo {
    /// Create witness info based on policy (LEGACY - for backward compat)
    /// 
    /// NOTE: This is kept for compatibility but the new `capture_full_witness`
    /// function in capture.rs should be used for full storage mode.
    pub fn from_policy(
        witness: &[F],
        _rows: &[SparseRow],
        gradient_blame: Vec<GradientContribution>,
        policy: WitnessPolicy,
        m_in: usize,
    ) -> Self {
        // For full storage, always store everything
        let z_full: Vec<String> = witness.iter()
            .map(|f| field_to_canonical_hex(*f))
            .collect();
        
        let x_public: Vec<String> = witness[..m_in.min(witness.len())].iter()
            .map(|f| field_to_canonical_hex(*f))
            .collect();
        
        let w_private: Vec<String> = witness[m_in.min(witness.len())..].iter()
            .map(|f| field_to_canonical_hex(*f))
            .collect();
        
        WitnessInfo {
            policy,
            z_full,
            x_public,
            w_private,
            z_decomposition: None,
            gradient_blame,
            size: witness.len(),
        }
    }
}

/// Parse witness policy from environment
pub fn policy_from_env() -> WitnessPolicy {
    match std::env::var("NEO_DIAGNOSTIC_WITNESS_POLICY").as_deref() {
        Ok("row_support") => WitnessPolicy::RowSupportOnly,
        Ok("gradient") => {
            let threshold = std::env::var("NEO_DIAGNOSTIC_GRADIENT_THRESHOLD")
                .unwrap_or_else(|_| "1000".to_string());
            let top_k = std::env::var("NEO_DIAGNOSTIC_GRADIENT_TOP_K")
                .unwrap_or_else(|_| "20".to_string())
                .parse()
                .unwrap_or(20);
            WitnessPolicy::GradientSlice { threshold, top_k }
        }
        Ok("full") => WitnessPolicy::FullWitness,
        _ => WitnessPolicy::RowSupportOnly,  // Default: minimal leak
    }
}

/// Convert field element to canonical hex (LE bytes)
pub fn field_to_canonical_hex(f: F) -> String {
    let bytes = f.as_canonical_u64().to_le_bytes();
    hex::encode(bytes)
}

/// Convert field element to signed string for readability
pub fn field_to_signed_string(f: F) -> String {
    field_to_signed_i128(f).to_string()
}

/// Convert field element to signed i128 representation
pub fn field_to_signed_i128(f: F) -> i128 {
    let canonical = f.as_canonical_u64();
    let modulus = F::ORDER_U64;
    let half = modulus / 2;
    
    if canonical > half {
        -((modulus - canonical) as i128)
    } else {
        canonical as i128
    }
}
