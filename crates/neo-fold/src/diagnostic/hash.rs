//! Canonical structure hashing (stable across builds)

use blake3::Hasher;
use neo_ccs::{CcsStructure, SparsePoly};
use neo_math::F;
use p3_field::PrimeField64;
use super::types::{PolyCanonical, PolyTerm};

/// Compute stable canonical hash of CCS structure
/// Version 1: domain-separated BLAKE3 with canonical LE bytes
pub fn compute_structure_hash_v1(structure: &CcsStructure<F>) -> String {
    let mut hasher = Hasher::new();
    
    // Domain separation
    hasher.update(b"neo.ccs.structure.v1");
    
    // Dimensions (fixed-width LE)
    hasher.update(&structure.n.to_le_bytes());
    hasher.update(&structure.m.to_le_bytes());
    hasher.update(&structure.t().to_le_bytes());
    
    // Hash each matrix (canonical LE bytes, row-major order)
    for (j, mat) in structure.matrices.iter().enumerate() {
        hasher.update(format!("matrix.{}", j).as_bytes());
        hasher.update(&mat.rows().to_le_bytes());
        hasher.update(&mat.cols().to_le_bytes());
        
        // Row-major order, canonical LE bytes
        for row in 0..mat.rows() {
            for col in 0..mat.cols() {
                let val = mat[(row, col)];
                hasher.update(&val.as_canonical_u64().to_le_bytes());
            }
        }
    }
    
    // Hash polynomial (canonical representation)
    hasher.update(b"polynomial");
    let poly_canonical = serialize_poly_canonical(&structure.f);
    hasher.update(&poly_canonical);
    
    format!("blake3v1:{}", hex::encode(hasher.finalize().as_bytes()))
}

/// Serialize polynomial in canonical binary form
fn serialize_poly_canonical(poly: &SparsePoly<F>) -> Vec<u8> {
    let mut buf = Vec::new();
    
    // Number of terms
    let num_terms = poly.terms().len();
    buf.extend_from_slice(&num_terms.to_le_bytes());
    
    // Collect and sort terms for determinism
    let mut terms: Vec<_> = poly.terms().to_vec();
    terms.sort_by_key(|term| {
        (term.exps.clone(), term.coeff.as_canonical_u64())
    });
    
    for term in &terms {
        // Coefficient (canonical LE)
        buf.extend_from_slice(&term.coeff.as_canonical_u64().to_le_bytes());
        
        // Exponents
        let exps = &term.exps;
        buf.extend_from_slice(&exps.len().to_le_bytes());
        
        // Exponent values
        for &exp in exps {
            buf.extend_from_slice(&exp.to_le_bytes());
        }
    }
    
    buf
}

/// Convert polynomial to canonical JSON representation
pub fn poly_to_canonical_json(poly: &SparsePoly<F>) -> PolyCanonical {
    let mut terms: Vec<_> = poly.terms()
        .iter()
        .map(|term| {
            // Convert exponents to (var_index, exponent) pairs (only non-zero exponents)
            // This preserves full polynomial information for non-multilinear cases
            let vars: Vec<(usize, u32)> = term.exps.iter()
                .enumerate()
                .filter(|(_, &exp)| exp > 0)
                .map(|(idx, &exp)| (idx, exp))
                .collect();
            
            PolyTerm {
                coeff: hex::encode(term.coeff.as_canonical_u64().to_le_bytes()),
                vars,
            }
        })
        .collect();
    
    // Sort for determinism
    terms.sort_by(|a, b| {
        match a.vars.cmp(&b.vars) {
            std::cmp::Ordering::Equal => a.coeff.cmp(&b.coeff),
            other => other,
        }
    });
    
    PolyCanonical { terms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_ccs::{r1cs_to_ccs, Mat};
    use p3_field::PrimeCharacteristicRing;
    
    #[test]
    fn test_hash_stability() {
        // Create simple R1CS
        let a = Mat::from_row_major(2, 3, vec![
            F::ONE, F::ZERO, F::ZERO,
            F::ZERO, F::ONE, F::ZERO,
        ]);
        let b = Mat::from_row_major(2, 3, vec![
            F::ZERO, F::ONE, F::ZERO,
            F::ZERO, F::ZERO, F::ONE,
        ]);
        let c = Mat::from_row_major(2, 3, vec![
            F::ZERO, F::ZERO, F::ONE,
            F::ONE, F::ZERO, F::ZERO,
        ]);
        
        let ccs = r1cs_to_ccs(a, b, c);
        
        // Hash should be stable
        let hash1 = compute_structure_hash_v1(&ccs);
        let hash2 = compute_structure_hash_v1(&ccs);
        
        assert_eq!(hash1, hash2);
        assert!(hash1.starts_with("blake3v1:"));
    }
}

