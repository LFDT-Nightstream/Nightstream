#![deny(missing_docs)]
//! CCS frontend for Neo: structures, relations (MCS/ME), and row-wise checks.
//!
//! Implements the MUST and SHOULD in the Neo spec, matching the paper's §4.1 relations
//! (MCS & ME), the row-wise CCS check, and the consistency equalities used by Π_CCS/Π_RLC/Π_DEC.

// New audit-ready core modules
/// Error types for CCS operations.
pub mod error;
/// Matrix types and operations.
pub mod matrix;
/// Polynomial types and evaluation.
pub mod poly;
/// R1CS to CCS conversion utilities.
pub mod r1cs;
/// Core CCS relations and consistency checks.
pub mod relations;
/// Traits for commitment scheme integration.
pub mod traits;
/// Utility functions for tensor products and matrix operations.
pub mod utils;

// Legacy compatibility modules (preserved during migration)
/// CCS sumcheck prover and verifier (legacy).
#[cfg(feature = "legacy-sumcheck")]
#[deprecated(note = "Sum-check lives in neo-fold. This stub will be removed.")]
pub mod ccs_sumcheck;
/// Format conversion utilities (legacy).
#[cfg(feature = "legacy-compat")]
pub mod converters;
/// Integration utilities for Spartan2 compatibility (legacy).
#[cfg(feature = "legacy-compat")]
pub mod integration;
/// Bridge adapter for neo-spartan-bridge integration.
pub mod bridge_adapter;

// Tests module
#[cfg(test)]
mod tests;

// Import field types for bridge adapters
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// Re-export new core types
pub use error::{CcsError, DimMismatch, RelationError};
pub use matrix::{Mat, MatRef};
pub use poly::{SparsePoly, Term};
pub use r1cs::r1cs_to_ccs;
// Main CCS types and functions (audit-ready)
pub use relations::{
    CcsStructure, McsInstance, McsWitness, MeInstance, MeWitness,
    check_mcs_opening, check_me_consistency, check_ccs_rowwise_zero,
    check_ccs_rowwise_relaxed,
};
pub use traits::SModuleHomomorphism;
pub use utils::{tensor_point, mat_vec_mul_fk, mat_vec_mul_ff, validate_power_of_two};

// Re-export legacy compatibility types and functions (gated)
#[cfg(feature = "legacy-sumcheck")]
#[allow(deprecated)]
pub use ccs_sumcheck::{ccs_sumcheck_prover, ccs_sumcheck_verifier};

// Legacy compatibility exports (gated)
#[cfg(feature = "legacy-compat")]
pub use legacy::{CcsStructure as LegacyCcsStructure, CcsInstance, CcsWitness, mv_poly, Multivariate};

#[cfg(feature = "legacy-compat")]
pub use converters::{
    ccs_to_r1cs_format, ccs_instance_to_r1cs_format, ccs_witness_to_r1cs_format,
};
#[cfg(feature = "spartan2-compat")]
pub use converters::field_conversion;

// Legacy compatibility types (gated for migration period)
#[cfg(feature = "legacy-compat")] 
pub mod legacy {
    use neo_math::{embed_base_to_ext, ExtF, F};
    use neo_math::RingElement;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use std::sync::Arc;

    /// Legacy trait for querying the total degree of a polynomial.
    pub trait Degree {
        /// Get the total degree.
        fn degree(&self) -> usize;
    }

    /// Legacy trait representing a multivariate polynomial over `ExtF`.
    pub trait MvPolynomial: Send + Sync + Degree {
        /// Evaluate at inputs.
        fn evaluate(&self, inputs: &[ExtF]) -> ExtF;
        /// Maximum degree of any individual variable.
        fn max_individual_degree(&self) -> usize {
            self.degree()
        }
    }

    /// Legacy convenience wrapper for closures with an associated degree.
    pub struct ClosureMv<Fn>
    where
        Fn: std::ops::Fn(&[ExtF]) -> ExtF + Send + Sync,
    {
        func: Fn,
        deg: usize,
    }

    impl<Fn> Degree for ClosureMv<Fn>
    where
        Fn: std::ops::Fn(&[ExtF]) -> ExtF + Send + Sync,
    {
        fn degree(&self) -> usize {
            self.deg
        }
    }

    impl<Fn> MvPolynomial for ClosureMv<Fn>
    where
        Fn: std::ops::Fn(&[ExtF]) -> ExtF + Send + Sync,
    {
        fn evaluate(&self, inputs: &[ExtF]) -> ExtF {
            (self.func)(inputs)
        }

        fn max_individual_degree(&self) -> usize {
            self.deg
        }
    }

    /// Legacy function to construct a multivariate polynomial from a closure and its degree.
    pub fn mv_poly<Fn>(f: Fn, deg: usize) -> Multivariate
    where
        Fn: std::ops::Fn(&[ExtF]) -> ExtF + Send + Sync + 'static,
    {
        Arc::new(ClosureMv { func: f, deg })
    }

    /// Legacy multivariate polynomial handle.
    pub type Multivariate = Arc<dyn MvPolynomial>;

    /// Legacy CCS structure for backward compatibility.
    #[derive(Clone)]
    pub struct CcsStructure {
        /// List of constraint matrices M_j (s matrices)
        pub mats: Vec<RowMajorMatrix<ExtF>>, 
        /// Constraint polynomial f over s vars
        pub f: Multivariate,                 
        /// Size of matrices (n rows)
        pub num_constraints: usize,          
        /// m columns
        pub witness_size: usize,             
        /// Maximum total degree of f
        pub max_deg: usize,                  
    }

    impl CcsStructure {
        /// Legacy constructor
        pub fn new(mats: Vec<RowMajorMatrix<F>>, f: Multivariate) -> Self {
            let lifted_mats: Vec<RowMajorMatrix<ExtF>> = mats
                .into_iter()
                .map(|mat| {
                    let data: Vec<ExtF> = mat.values.iter().map(|&v| embed_base_to_ext(v)).collect();
                    RowMajorMatrix::new(data, mat.width())
                })
                .collect();
            let (num_constraints, witness_size) = if lifted_mats.is_empty() {
                (0, 0)
            } else {
                (lifted_mats[0].height(), lifted_mats[0].width())
            };
            let max_deg = f.degree();
            let mid = f.max_individual_degree();
            assert!(
                mid <= 1,
                "CcsStructure: f must be multilinear (max individual degree ≤ 1) for soundness"
            );
            Self {
                mats: lifted_mats,
                f,
                num_constraints,
                witness_size,
                max_deg,
            }
        }
    }

    /// Legacy CCS instance.
    #[derive(Clone)]
    pub struct CcsInstance {
        /// Commitment to witness z
        pub commitment: Vec<RingElement>, 
        /// x (public part of instance)
        pub public_input: Vec<F>,         
        /// Relaxation scalar
        pub u: F,                         
        /// Relaxation offset
        pub e: F,                         
    }

    /// Legacy CCS witness.
    #[derive(Clone)]
    pub struct CcsWitness {
        /// Private witness vector (does not include public inputs)
        pub z: Vec<ExtF>,
    }
}

/// Legacy relaxed satisfiability check function.
#[cfg(feature = "legacy-compat")]
pub fn check_relaxed_satisfiability(
    structure: &legacy::CcsStructure,
    instance: &legacy::CcsInstance,
    witness: &legacy::CcsWitness,
    u: F,
    e: F,
) -> bool {
    let mut full_z: Vec<ExtF> = instance
        .public_input
        .iter()
        .map(|&x| embed_base_to_ext(x))
        .collect();
    full_z.extend_from_slice(&witness.z);
    if full_z.len() != structure.witness_size {
        return false;
    }

    // NOTE: Do not add extra invariants here; the CCS relation is enforced by f(Mz) row-wise.

    let s = structure.mats.len();
    // Relaxed CCS: f(Mz) = e * u (Def. 19)
    let right = embed_base_to_ext(u) * embed_base_to_ext(e);
    for row in 0..structure.num_constraints {
        let mut inputs = vec![ExtF::ZERO; s];
        for (input, mat) in inputs.iter_mut().zip(structure.mats.iter()) {
            let mut sum = ExtF::ZERO;
            for col in 0..structure.witness_size {
                sum += mat.get(row, col).unwrap_or(ExtF::ZERO) * full_z[col];
            }
            *input = sum;
        }
        if structure.f.evaluate(&inputs) != right {
            return false;
        }
    }
    true
}

/// Legacy standard satisfiability check (Def. 17).
#[cfg(feature = "legacy-compat")]
pub fn check_satisfiability(
    structure: &legacy::CcsStructure,
    instance: &legacy::CcsInstance,
    witness: &legacy::CcsWitness,
) -> bool {
    check_relaxed_satisfiability(structure, instance, witness, instance.u, instance.e)
}

/// Legacy verifier CCS structure.
#[cfg(feature = "legacy-compat")]
pub fn verifier_ccs() -> legacy::CcsStructure {
    use neo_math::{ExtF, F};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    let mats = vec![
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO], 4),
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO], 4),
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO], 4),
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE], 4),
    ];

    let f: Multivariate = mv_poly(move |inputs: &[ExtF]| {
        if inputs.len() != 4 {
            ExtF::ZERO
        } else {
            inputs[0] + inputs[1] - inputs[3]
        }
    }, 1);

    legacy::CcsStructure::new(mats, f)
}

/// Legacy function to add lookups.
#[cfg(feature = "legacy-compat")]
pub fn add_lookups(structure: &mut legacy::CcsStructure, table: Vec<F>) {
    use neo_math::{embed_base_to_ext, ExtF, F};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    let data: Vec<ExtF> = table.into_iter().map(|v| embed_base_to_ext(v)).collect();
    let lookup_mat = RowMajorMatrix::new(data, 1);
    structure.mats.push(lookup_mat);
    let orig_f = structure.f.clone();
    let orig_deg = orig_f.degree();
    let f_lookup = legacy::mv_poly(
        |inputs: &[ExtF]| {
            if let (Some(first), Some(last)) = (inputs.first(), inputs.last()) {
                if first == last {
                    ExtF::ONE
                } else {
                    ExtF::ZERO
                }
            } else {
                ExtF::ZERO
            }
        },
        1,
    );
    structure.f = legacy::mv_poly(
        move |ins: &[ExtF]| orig_f.evaluate(ins) * f_lookup.evaluate(ins),
        orig_deg.max(1),
    );
    structure.max_deg = structure.f.degree();
}

// Legacy ME types for final folding outputs (preserved)
/// Legacy Matrix Evaluation instance - the final claim after folding
#[derive(Clone, Debug)]
pub struct MEInstance {
    /// Ajtai commitment coordinates c ∈ F_q^{d×κ}
    pub c_coords: Vec<F>, 
    /// ME outputs y_j = ⟨M_j^T r^b, Z⟩ for each matrix j
    pub y_outputs: Vec<F>, 
    /// Public random point r^b from sum-check 
    pub r_point: Vec<F>, 
    /// Base parameter for range constraints
    pub base_b: u64,
    /// Transcript header digest for binding to neo-fold
    pub header_digest: [u8; 32],
}

/// Legacy Matrix Evaluation witness - the final witness after folding  
#[derive(Clone, Debug)]
pub struct MEWitness {
    /// Witness digits Z in base b: |Z|_∞ < b
    pub z_digits: Vec<i64>, 
    /// Weight vectors v_j = M_j^T r^b for computing ⟨v_j, Z⟩ = y_j
    pub weight_vectors: Vec<Vec<F>>, 
    /// Optional Ajtai linear map rows L for c = L(Z) verification
    pub ajtai_rows: Option<Vec<Vec<F>>>, 
}

impl MEInstance {
    /// Create a new ME instance
    pub fn new(
        c_coords: Vec<F>,
        y_outputs: Vec<F>, 
        r_point: Vec<F>,
        base_b: u64,
        header_digest: [u8; 32],
    ) -> Self {
        Self {
            c_coords,
            y_outputs,
            r_point,
            base_b,
            header_digest,
        }
    }
    
    /// Number of ME outputs
    pub fn num_outputs(&self) -> usize {
        self.y_outputs.len()
    }
    
    /// Witness dimension 
    pub fn witness_dim(&self) -> usize {
        self.r_point.len()
    }
}

impl MEWitness {
    /// Create a new ME witness
    pub fn new(
        z_digits: Vec<i64>,
        weight_vectors: Vec<Vec<F>>,
        ajtai_rows: Option<Vec<Vec<F>>>,
    ) -> Self {
        Self {
            z_digits,
            weight_vectors, 
            ajtai_rows,
        }
    }
    
    /// Verify consistency: all weight vectors have same length as z_digits
    pub fn check_consistency(&self) -> bool {
        let z_len = self.z_digits.len();
        
        if !self.weight_vectors.iter().all(|w| w.len() == z_len) {
            return false;
        }
        
        if let Some(ref rows) = self.ajtai_rows {
            if !rows.iter().all(|row| row.len() == z_len) {
                return false;
            }
        }
        
        true
    }
    
    /// Verify that ME equations hold: ⟨v_j, Z⟩ = y_j
    pub fn verify_me_equations(&self, instance: &MEInstance) -> bool {
        if self.weight_vectors.len() != instance.y_outputs.len() {
            return false;
        }
        
        for (_j, (weights, &expected_y)) in 
            self.weight_vectors.iter().zip(instance.y_outputs.iter()).enumerate() 
        {
            if weights.len() != self.z_digits.len() {
                return false;
            }
            
            let mut actual_y = F::ZERO;
            for (&w, &z) in weights.iter().zip(self.z_digits.iter()) {
                let z_field = if z >= 0 { 
                    F::from_u64(z as u64) 
                } else { 
                    -F::from_u64((-z) as u64) 
                };
                actual_y += w * z_field;
            }
            
            if actual_y != expected_y {
                return false;
            }
        }
        
        true
    }
    
    /// Verify Ajtai commitment if present: c = L(Z)
    pub fn verify_ajtai_commitment(&self, instance: &MEInstance) -> bool {
        if let Some(ref ajtai_rows) = self.ajtai_rows {
            if ajtai_rows.len() != instance.c_coords.len() {
                return false;
            }
            
            for (_t, (row, &expected_c)) in 
                ajtai_rows.iter().zip(instance.c_coords.iter()).enumerate()
            {
                if row.len() != self.z_digits.len() {
                    return false;
                }
                
                let mut actual_c = F::ZERO;
                for (&l, &z) in row.iter().zip(self.z_digits.iter()) {
                    let z_field = if z >= 0 { 
                        F::from_u64(z as u64) 
                    } else { 
                        -F::from_u64((-z) as u64) 
                    };
                    actual_c += l * z_field;
                }
                
                if actual_c != expected_c {
                    return false;
                }
            }
        }
        
        true
    }
}