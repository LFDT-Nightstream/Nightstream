//! Witness disclosure policies

use serde::{Serialize, Deserialize};
use super::types::{WitnessInfo, SparseVector, GradientContribution, SparseRow};
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
}

impl WitnessInfo {
    /// Create witness info based on policy
    pub fn from_policy(
        witness: &[F],
        rows: &[SparseRow],
        gradient_blame: Vec<GradientContribution>,
        policy: WitnessPolicy,
    ) -> Self {
        let z_sparse = match &policy {
            WitnessPolicy::RowSupportOnly => {
                // Only indices that appear in failing row
                let mut indices: std::collections::HashSet<usize> = 
                    std::collections::HashSet::new();
                for row in rows {
                    for &idx in &row.idx {
                        indices.insert(idx);
                    }
                }
                let mut idx_vec: Vec<_> = indices.into_iter().collect();
                idx_vec.sort();
                
                SparseVector {
                    idx: idx_vec.clone(),
                    val: idx_vec.iter()
                        .map(|&i| field_to_canonical_hex(witness[i]))
                        .collect(),
                }
            }
            
            WitnessPolicy::GradientSlice { threshold, top_k } => {
                // Only indices with large |∂r/∂z_i|
                let threshold_val = parse_canonical_u64(threshold);
                let indices: Vec<usize> = gradient_blame.iter()
                    .take(*top_k)
                    .filter(|g| {
                        parse_canonical_u64(&g.gradient_abs) >= threshold_val
                    })
                    .map(|g| g.i)
                    .collect();
                
                SparseVector {
                    idx: indices.clone(),
                    val: indices.iter()
                        .map(|&i| field_to_canonical_hex(witness[i]))
                        .collect(),
                }
            }
            
            WitnessPolicy::FullWitness => {
                // Full witness - check safety guard
                if std::env::var("NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS").is_err() {
                    eprintln!("⚠️  Warning: Full witness disclosure requires NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS=1");
                    eprintln!("⚠️  Falling back to RowSupportOnly policy for safety");
                    
                    // Fallback to row support
                    return Self::from_policy(
                        witness,
                        rows,
                        gradient_blame.clone(),
                        WitnessPolicy::RowSupportOnly,
                    );
                }
                
                SparseVector {
                    idx: (0..witness.len()).collect(),
                    val: witness.iter()
                        .map(|f| field_to_canonical_hex(*f))
                        .collect(),
                }
            }
        };
        
        WitnessInfo {
            policy,
            z_sparse,
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

/// Parse canonical u64 from string
fn parse_canonical_u64(s: &str) -> u64 {
    s.parse().unwrap_or(0)
}

