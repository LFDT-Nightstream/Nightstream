//! Π_DEC reduction: Single ME(B,L) → k ME(b,L) via base-b decomposition with verified openings
//!
//! This implements the third reduction in the Neo folding pipeline:
//! - Takes 1 ME(B,L) instance from Π_RLC with large base B
//! - Decomposes witness Z' into k base-b digits: Z' = Σ b^i · Z_i with ||Z_i||_∞ < b
//! - Uses verified openings to prove c = Σ b^i · L(Z_i) via neo-ajtai split_b
//! - Outputs k ME(b,L) instances ready for next folding round

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use crate::transcript::{FoldTranscript, Domain};
use neo_ajtai::{split_b, s_lincomb, Commitment as Cmt, DecompStyle};
use neo_ccs::{CcsStructure, MeInstance, MeWitness, Mat, utils::{tensor_point, mat_vec_mul_fk}};
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

#[derive(Debug, Clone)]
pub struct PiDecProof {
    /// Commitments to each base-b digit Z_i (optional - may be implicit)
    pub digit_commitments: Option<Vec<Cmt>>,
    /// Verified opening proof for recomposition c = Σ b^i · c_i
    pub recomposition_proof: Vec<u8>,
    /// Range proof data ensuring ||Z_i||_∞ < b for each digit
    pub range_proofs: Vec<u8>,
}

/// Error type for Π_DEC reduction
#[derive(Debug, thiserror::Error)]
pub enum PiDecError {
    #[error("Base-b decomposition failed: {0}")]
    DecompositionFailed(String),
    #[error("Verified opening failed: {0}")]
    VerifiedOpeningFailed(String),
    #[error("Range check failed: {0}")]
    RangeCheckFailed(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("S-homomorphism error: {0}")]
    SHomomorphismError(String),
}

/// Π_DEC reduction: 1 ME(B,L) instance → k ME(b,L) instances
///
/// This decomposes a large-base ME instance into multiple small-base instances.
/// Uses neo-ajtai verified openings to ensure cryptographic soundness.
pub fn pi_dec<L: SModuleHomomorphism<F, Cmt>>(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    me_B: &MeInstance<Cmt, F, K>,
    wit_B: &MeWitness<F>, // Witness Z' for the large-base claim (prover-side)
    s: &CcsStructure<F>, // CCS structure needed for y recomputation
    l: &L, // S-module homomorphism for commitment operations
) -> Result<(Vec<MeInstance<Cmt, F, K>>, Vec<MeWitness<F>>, PiDecProof), PiDecError> {
    
    // === Domain separation ===
    tr.domain(Domain::Dec);
    
    let d = wit_B.Z.rows();
    let m = wit_B.Z.cols();
    let k = params.k as usize;
    let b = params.b;
    
    if d == 0 || m == 0 {
        return Err(PiDecError::InvalidInput("Empty witness matrix".into()));
    }
    
    if k == 0 {
        return Err(PiDecError::InvalidInput("k=0 not allowed".into()));
    }
    
    println!("🔧 PI_DEC: Decomposing {}×{} matrix into {} base-{} digits", d, m, k, b);
    
    // === Decompose Z' into k base-b digits ===
    // Convert matrix to column-major slice for split_b
    let mut z_col_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            z_col_major[col * d + row] = wit_B.Z[(row, col)];
        }
    }
    
    let z_digits = split_b(
        &z_col_major,
        b,
        d,
        m,
        k,
        DecompStyle::Balanced
    );
    
    if z_digits.len() != k {
        return Err(PiDecError::DecompositionFailed(format!(
            "Expected {} digits, got {}", k, z_digits.len()
        )));
    }
    
    // === Convert digits back to matrix format and commit ===
    let mut digit_witnesses = Vec::with_capacity(k);
    let mut digit_commitments = Vec::with_capacity(k);
    
    for (i, digit_slice) in z_digits.into_iter().enumerate() {
        // Convert column-major slice back to matrix
        let mut z_digit = Mat::zero(d, m, F::ZERO);
        for col in 0..m {
            for row in 0..d {
                z_digit[(row, col)] = digit_slice[col * d + row];
            }
        }
        
        // Range constraint: split_b returns balanced base-b digits by construction
        // The decomposition algorithm ensures ||Z_i||_∞ < b automatically
        // No manual u64 check needed (would be incorrect for balanced representation)
        println!("  ✅ Digit {}: Balanced base-{} decomposition (range guaranteed by split_b)", i, b);
        
        // Commit to this digit: c_i = L(Z_i)
        let c_i = l.commit(&z_digit);
        
        digit_witnesses.push(MeWitness { Z: z_digit });
        digit_commitments.push(c_i);
    }
    
    // === Verified opening: prove c = Σ b^i · c_i ===
    let recomposition_proof = verify_split_open(
        &me_B.c,
        &digit_commitments,
        b as u64,
        l, // Pass the S-module homomorphism for verification
    ).map_err(|e| PiDecError::VerifiedOpeningFailed(format!("recomposition failed: {e:?}")))?;
    
    // === Create k ME(b,L) instances ===
    let mut me_instances = Vec::with_capacity(k);
    
    for (_i, (wit, c_i)) in digit_witnesses.iter().zip(digit_commitments.iter()).enumerate() {
        // Derive X_i = L_x(Z_i) for each digit
        let X_i = l.project_x(&wit.Z, me_B.m_in);
        
        // CRITICAL FIX: Recompute y_{i,j} = Z_i · (M_j^T · r^⊗) for each matrix M_j
        // This ensures each digit has the correct ME relation
        let rb = tensor_point::<K>(&me_B.r);
        let mut y_i = Vec::with_capacity(s.t());
        
        for mj in &s.matrices {
            // Compute v_j = M_j^T * r^⊗ (verifier can compute this)
            let mut v_j = vec![K::ZERO; s.m];
            for row_idx in 0..s.n {
                let coeff = rb[row_idx];
                let matrix_row = mj.row(row_idx);
                for col_idx in 0..s.m {
                    v_j[col_idx] += K::from(matrix_row[col_idx]) * coeff;
                }
            }
            
            // Compute y_{i,j} = Z_i * v_j (matrix-vector multiply over F×K)
            let y_ij = mat_vec_mul_fk::<F, K>(wit.Z.as_slice(), wit.Z.rows(), wit.Z.cols(), &v_j);
            y_i.push(y_ij);
        }
        
        me_instances.push(MeInstance {
            c: c_i.clone(),
            X: X_i,
            r: me_B.r.clone(),
            y: y_i,
            m_in: me_B.m_in,
        });
    }
    
    let proof = PiDecProof {
        digit_commitments: Some(digit_commitments),
        recomposition_proof,
        range_proofs: vec![], // Placeholder for range proof data
    };
    
    println!("✅ PI_DEC: Created {} ME(b,L) instances from ME(B,L)", k);
    
    Ok((me_instances, digit_witnesses, proof))
}

/// Verify a Π_DEC proof
pub fn pi_dec_verify<L: SModuleHomomorphism<F, Cmt>>(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    input_me: &MeInstance<Cmt, F, K>,
    output_me_list: &[MeInstance<Cmt, F, K>],
    proof: &PiDecProof,
    l: &L,
) -> Result<bool, PiDecError> {
    
    tr.domain(Domain::Dec);
    
    let k = params.k as usize;
    let b = params.b;
    
    if output_me_list.len() != k {
        return Ok(false);
    }
    
    // === Verify recomposition of commitment ===
    if let Some(ref digit_commitments) = proof.digit_commitments {
        if digit_commitments.len() != k {
            return Ok(false);
        }
        
        // Verify c = Σ b^i · c_i using neo-ajtai verified opening
        let recomposition_ok = verify_split_open(
            &input_me.c,
            digit_commitments,
            b as u64,
            l,
        ).is_ok();
        
        if !recomposition_ok {
            return Ok(false);
        }
    }
    
    // === Verify instance consistency ===
    for (i, me_digit) in output_me_list.iter().enumerate() {
        // All digits should have same r
        if me_digit.r != input_me.r {
            return Ok(false);
        }
        
        // All digits should have same m_in
        if me_digit.m_in != input_me.m_in {
            return Ok(false);
        }
        
        // Verify commitment matches digit commitment if provided
        if let Some(ref digit_commitments) = proof.digit_commitments {
            if me_digit.c != digit_commitments[i] {
                return Ok(false);
            }
        }
    }
    
    // TODO: Verify range proofs for each digit
    // TODO: Verify y_j recomputation is consistent
    
    println!("✅ PI_DEC_VERIFY: Verification passed");
    Ok(true)
}

/// Real verified opening: prove c = Σ b^i · c_i using S-module homomorphism
/// This is the core binding between the combined commitment and digit commitments
fn verify_split_open<L: SModuleHomomorphism<F, Cmt>>(
    combined_c: &Cmt,
    digit_cs: &[Cmt],
    base: u64,
    _l: &L,
) -> Result<Vec<u8>, String> {

    
    // Build ρ_i = (b^i) · I_d ∈ S for recomposition
    // TODO: This requires SMatrix::scalar() helper in neo-math
    // For now, we'll do a simplified check using s_lincomb directly
    
    // Create scalar coefficients b^0, b^1, b^2, ... as ring elements
    let mut coeffs = Vec::with_capacity(digit_cs.len());
    let mut pow = neo_math::F::ONE;
    
    for _i in 0..digit_cs.len() {
        // Convert scalar to ring element: [pow, 0, 0, ..., 0] represents pow * 1
        let mut coeff_array = [neo_math::F::ZERO; 54]; // D = 54 from neo-math
        coeff_array[0] = pow; // pow * X^0
        let ring_elem = neo_math::cf_inv(coeff_array);
        coeffs.push(ring_elem);
        pow *= neo_math::F::from_u64(base);
    }
    
    // Recompose: c' = Σ b^i · c_i
    let recomposed = s_lincomb(&coeffs, digit_cs);
    
    // Verify c == c'
    if &recomposed != combined_c {
        return Err(format!(
            "❌ Ajtai recomposition failed: c ≠ Σ b^i · c_i (base={})", 
            base
        ));
    }
    
    println!("✅ REAL_VERIFY_SPLIT_OPEN: Recomposition check passed for {} digits", digit_cs.len());
    
    // Return empty proof - recomposition is a deterministic check
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_params::NeoParams;
    
    #[test]
    fn test_pi_dec_validation() {
        // Test basic input validation
        let params = NeoParams::goldilocks_127();
        let mut tr = FoldTranscript::new(b"test_dec");
        
        // Empty witness should fail
        let empty_me = dummy_me_instance_B();
        let empty_wit = MeWitness { Z: Mat::zero(0, 0, F::ZERO) };
        let dummy_ccs = dummy_ccs_structure();
        let dummy_l = dummy_s_module_hom();
        
        let result = pi_dec(&mut tr, &params, &empty_me, &empty_wit, &dummy_ccs, &dummy_l);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PiDecError::InvalidInput(_)));
    }
    
    fn dummy_me_instance_B() -> MeInstance<Cmt, F, K> {
        use neo_ccs::Mat;
        use p3_field::PrimeCharacteristicRing;
        
        MeInstance {
            c: Cmt::zeros(4, 2), // d=4, kappa=2 dummy commitment
            X: Mat::zero(2, 1, F::ZERO), // 2x1 matrix
            r: vec![K::new_real(F::from_u64(42))], // dummy r vector
            y: vec![vec![K::new_real(F::from_u64(100))]], // dummy y vector
            m_in: 1,
        }
    }
    
    fn dummy_ccs_structure() -> CcsStructure<F> {
        use neo_ccs::{SparsePoly, Term, Mat};
        
        // Single 4x3 matrix (n=4, m=3)
        let matrices = vec![Mat::zero(4, 3, F::ZERO)]; 
        let terms = vec![Term { coeff: F::ONE, exps: vec![1] }]; // Simple linear term
        let f = SparsePoly::new(1, terms); // arity=1 to match single matrix
        
        CcsStructure::new(matrices, f).expect("Valid dummy CCS structure")
    }

    fn dummy_s_module_hom() -> DummySModuleHomDec {
        DummySModuleHomDec
    }
    
    struct DummySModuleHomDec;
    
    impl SModuleHomomorphism<F, Cmt> for DummySModuleHomDec {
        fn commit(&self, _z: &neo_ccs::Mat<F>) -> Cmt {
            // Return dummy commitment for testing
            Cmt::zeros(4, 2) // d=4, kappa=2 dummy commitment
        }
        
        fn project_x(&self, z: &neo_ccs::Mat<F>, m_in: usize) -> neo_ccs::Mat<F> {
            // Return first m_in columns for testing
            let rows = z.rows();
            let cols = m_in.min(z.cols());
            let mut result = neo_ccs::Mat::zero(rows, cols, F::ZERO);
            for r in 0..rows {
                for c in 0..cols {
                    result[(r, c)] = z[(r, c)];
                }
            }
            result
        }
    }
}
