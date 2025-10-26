//! Diagnostic capture functionality - FULL STORAGE MODE
//!
//! Captures EVERYTHING: all matrices, full witness, all intermediate values

use super::types::*;
use super::blame::compute_gradient_blame;
use super::hash::{compute_structure_hash_v1, poly_to_canonical_json};
use super::witness::{field_to_canonical_hex, field_to_signed_string};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use super::SCHEMA_VERSION;

/// Capture a COMPLETE diagnostic snapshot with EVERYTHING
/// 
/// This captures:
/// - All n×m matrices (complete structure)
/// - Full witness z, split into x (public) and w (private)
/// - All n constraint evaluations (not just failing one)
/// - Detailed matrix-vector product terms
/// - Instance information (commitments, etc.)
pub fn capture_diagnostic(
    structure: &CcsStructure<F>,
    row_index: usize,
    witness: &[F],
    context: ContextInfo,
    folding_context: FoldingContext,
    transcript_challenges: Option<TranscriptInfo>,
) -> ConstraintDiagnostic {
    capture_diagnostic_with_instance(
        structure,
        row_index,
        witness,
        context,
        folding_context,
        transcript_challenges,
        None,  // instance - will be None for test cases
        0,     // m_in - default to 0 (all witness is private)
    )
}

/// Generate a cryptographically secure random salt for witness redaction
fn generate_redaction_salt() -> [u8; 32] {
    let mut salt = [0u8; 32];
    // Use modern rand API for random salt generation
    use rand::RngCore;
    rand::rng().fill_bytes(&mut salt);
    salt
}

/// Capture diagnostic with full instance information
pub fn capture_diagnostic_with_instance(
    structure: &CcsStructure<F>,
    row_index: usize,
    witness: &[F],
    context: ContextInfo,
    folding_context: FoldingContext,
    transcript_challenges: Option<TranscriptInfo>,
    _instance_opt: Option<&dyn std::any::Any>,  // Will be McsInstance in real usage
    m_in: usize,
) -> ConstraintDiagnostic {
    let m = structure.m;
    let n = structure.n;
    let t = structure.t();
    
    // === SECURITY: Generate random salt for witness redaction ===
    let redaction_salt = generate_redaction_salt();
    let redaction_salt_hex = Some(hex::encode(redaction_salt));
    
    // === FULL MATRIX STORAGE ===
    let full_matrices = capture_full_matrices(structure);
    
    // === SPARSE FAILING ROW (for backward compat) ===
    let sparse_rows: Vec<SparseRow> = structure.matrices.iter()
        .enumerate()
        .map(|(j, mat)| extract_sparse_row(mat, row_index, j))
        .collect();
    
    // === FULL WITNESS STORAGE (with secure redaction) ===
    let witness_info = capture_full_witness(witness, m_in, structure, row_index, &sparse_rows, &redaction_salt);
    
    // === ALL CONSTRAINT EVALUATIONS ===
    let all_constraints = evaluate_all_constraints(structure, witness);
    
    // === DETAILED MATRIX PRODUCTS FOR FAILING ROW ===
    let matrix_products = compute_detailed_products(structure, witness, row_index);
    
    // Compute the actual constraint evaluation for failing row
    let y_vals: Vec<F> = sparse_rows.iter().map(|sparse_row| {
        sparse_row.idx.iter().zip(&sparse_row.val).fold(F::ZERO, |acc, (&col_idx, v_hex)| {
            let bytes = hex::decode(v_hex).unwrap();
            let u64_val = u64::from_le_bytes(bytes.try_into().unwrap());
            let mat_val = F::from_u64(u64_val);
            acc + mat_val * witness[col_idx]
        })
    }).collect();
    
    // Evaluate the polynomial f(y_1, ..., y_t)
    let actual = evaluate_poly(&structure.f, &y_vals);
    let expected = F::ZERO;
    let residual = actual - expected;
    
    // For R1CS (t=3), compute az, bz, cz products
    let r1cs_products = if t == 3 && y_vals.len() == 3 {
        Some(R1csProducts {
            az: field_to_canonical_hex(y_vals[0]),
            bz: field_to_canonical_hex(y_vals[1]),
            cz: field_to_canonical_hex(y_vals[2]),
        })
    } else {
        None
    };
    
    // === INSTANCE INFORMATION ===
    let instance_info = None;  // TODO: Extract from instance_opt when available
    
    ConstraintDiagnostic {
        schema: SCHEMA_VERSION.to_string(),
        redaction_salt: redaction_salt_hex,
        
        field: FieldConfig {
            q: F::ORDER_U64.to_string(),
            name: "goldilocks".to_string(),
            ext_deg: 1,
            encoding: "canonical_le_bytes_hex".to_string(),
            ext_modulus: None,
        },
        
        folding: folding_context,
        
        structure: StructureInfo {
            hash: compute_structure_hash_v1(structure),
            t,
            n,
            m,
            m_in,
            row_index,
            rows: sparse_rows,
            full_matrices,
            f_canonical: poly_to_canonical_json(&structure.f),
        },
        
        instance: instance_info,
        
        eval: EvaluationInfo {
            r: vec![],  // Filled by caller if applicable
            y_at_r: vec![],  // Filled by caller if applicable
            matrix_products,
            r1cs_products,
            all_constraints,
            expected: field_to_canonical_hex(expected),
            actual: field_to_canonical_hex(actual),
            delta: field_to_canonical_hex(residual),
            delta_signed: field_to_signed_string(residual),
        },
        
        witness: witness_info,
        
        transcript: transcript_challenges.unwrap_or_else(|| TranscriptInfo {
            transcript_hash: "unavailable".to_string(),
            domain_labels: vec![],
            sumcheck: SumcheckChallenges {
                alpha: vec![],
                beta: vec![],
                gamma: String::new(),
                alpha_prime: vec![],
                r_prime: vec![],
            },
            rlc: None,
            dec: None,
        }),
        
        context,
        
        symbols: load_symbol_table_from_env(),
    }
}

/// Capture ALL matrices (complete n×m structure)
fn capture_full_matrices(structure: &CcsStructure<F>) -> Vec<FullMatrix> {
    structure.matrices.iter().enumerate().map(|(j, mat)| {
        let n = mat.rows();
        let m = mat.cols();
        
        // Extract all rows as sparse vectors
        let mut rows = Vec::with_capacity(n);
        let mut total_nnz = 0;
        
        for row_idx in 0..n {
            let sparse_row = extract_sparse_row(mat, row_idx, j);
            total_nnz += sparse_row.idx.len();
            rows.push(sparse_row);
        }
        
        let density = (total_nnz as f64) / ((n * m) as f64) * 100.0;
        
        FullMatrix {
            matrix_j: j,
            shape: (n, m),
            rows,
            density,
            nnz: total_nnz,
        }
    }).collect()
}

/// Capture FULL witness with public/private split
fn capture_full_witness(
    witness: &[F],
    m_in: usize,
    structure: &CcsStructure<F>,
    row_index: usize,
    _sparse_rows: &[SparseRow],
    redaction_salt: &[u8; 32],
) -> WitnessInfo {
    let m = witness.len();
    
    // Check if full witness disclosure is allowed
    let allow_full = std::env::var("NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    
    if !allow_full {
        eprintln!("⚠️  Warning: Full witness disclosure requires NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS=1");
        eprintln!("⚠️  This is DANGEROUS in lattice settings (exposes Ajtai/SIS preimages)");
        eprintln!("⚠️  Redacting private witness values with keyed BLAKE3...");
    }
    
    // Derive keying material from salt for BLAKE3 keyed mode
    let key = blake3::derive_key("neo/diagnostic/redaction/v1", redaction_salt);
    
    // Split into public (x) and private (w)
    let x_public: Vec<String> = witness[..m_in.min(m)].iter()
        .map(|&v| field_to_canonical_hex(v))
        .collect();
    
    let w_private: Vec<String> = if allow_full {
        witness[m_in.min(m)..].iter()
            .map(|&v| field_to_canonical_hex(v))
            .collect()
    } else {
        // SECURITY FIX: Use keyed BLAKE3 with 128+ bit output to prevent preimage attacks
        // Previous vulnerability: unkeyed hash with only 64-bit output was vulnerable to
        // brute-force attacks over Goldilocks' 64-bit canonical space
        witness[m_in.min(m)..].iter()
            .map(|&v| {
                let bytes = v.as_canonical_u64().to_le_bytes();
                let digest = blake3::keyed_hash(&key, &bytes);
                // Use 16 bytes (128 bits) minimum for security
                format!("REDACTED:{}", hex::encode(&digest.as_bytes()[..16]))
            })
            .collect()
    };
    
    // Full witness z = x || w (respect redaction)
    let z_full: Vec<String> = if allow_full {
        witness.iter().map(|&v| field_to_canonical_hex(v)).collect()
    } else {
        // For redacted: show public x, then redact private w
        witness.iter().enumerate().map(|(i, &v)| {
            if i < m_in {
                field_to_canonical_hex(v) // keep x public
            } else {
                let bytes = v.as_canonical_u64().to_le_bytes();
                let digest = blake3::keyed_hash(&key, &bytes);
                format!("REDACTED:{}", hex::encode(&digest.as_bytes()[..16]))
            }
        }).collect()
    };
    
    // Compute gradient blame
    let gradient_top_k = std::env::var("NEO_DIAGNOSTIC_GRADIENT_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    
    let gradient_blame = compute_gradient_blame(
        structure,
        row_index,
        witness,
        gradient_top_k,
    );
    
    // Determine actual policy based on redaction
    let policy = if allow_full {
        super::witness::WitnessPolicy::FullWitness
    } else {
        super::witness::WitnessPolicy::RedactedFull
    };
    
    WitnessInfo {
        policy,
        z_full,
        x_public,
        w_private,
        z_decomposition: None,  // TODO: Add when decomposition is available
        gradient_blame,
        size: m,
    }
}

/// Evaluate ALL n constraints (not just failing one)
fn evaluate_all_constraints(structure: &CcsStructure<F>, witness: &[F]) -> Vec<ConstraintEval> {
    let n = structure.n;
    
    let mut results = Vec::with_capacity(n);
    
    for row_idx in 0..n {
        // Compute y_j = M_j[row_idx] · z for each matrix j
        let y_vals: Vec<F> = structure.matrices.iter().map(|mat| {
            let mut sum = F::ZERO;
            for col_idx in 0..mat.cols() {
                let mat_val = mat[(row_idx, col_idx)];
                if mat_val != F::ZERO {
                    sum += mat_val * witness[col_idx];
                }
            }
            sum
        }).collect();
        
        // Evaluate constraint polynomial
        let value = evaluate_poly(&structure.f, &y_vals);
        let passed = value == F::ZERO;
        
        results.push(ConstraintEval {
            row: row_idx,
            passed,
            value: field_to_canonical_hex(value),
            y_values: y_vals.iter().map(|&v| field_to_canonical_hex(v)).collect(),
        });
    }
    
    results
}

/// Compute detailed matrix-vector products with all terms
fn compute_detailed_products(
    structure: &CcsStructure<F>,
    witness: &[F],
    row_index: usize,
) -> Vec<MatrixProduct> {
    structure.matrices.iter().enumerate().map(|(j, mat)| {
        let mut terms = Vec::new();
        let mut sum = F::ZERO;
        
        for col_idx in 0..mat.cols() {
            let mat_val = mat[(row_index, col_idx)];
            if mat_val != F::ZERO {
                let wit_val = witness[col_idx];
                let prod = mat_val * wit_val;
                sum += prod;
                
                terms.push(ProductTerm {
                    col_idx,
                    matrix_value: field_to_canonical_hex(mat_val),
                    witness_value: field_to_canonical_hex(wit_val),
                    product: field_to_canonical_hex(prod),
                });
            }
        }
        
        MatrixProduct {
            matrix_j: j,
            value: field_to_canonical_hex(sum),
            terms,
        }
    }).collect()
}

/// Extract sparse row from matrix
fn extract_sparse_row(mat: &Mat<F>, row: usize, matrix_j: usize) -> SparseRow {
    let mut idx = Vec::new();
    let mut val = Vec::new();
    
    for col in 0..mat.cols() {
        let v = mat[(row, col)];
        if v != F::ZERO {
            idx.push(col);
            val.push(field_to_canonical_hex(v));
        }
    }
    
    SparseRow {
        matrix_j,
        idx,
        val,
    }
}

/// Load symbol table from environment (if available)
fn load_symbol_table_from_env() -> Option<SymbolTable> {
    if let Ok(path) = std::env::var("NEO_SYMBOL_TABLE") {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(table) = serde_json::from_str(&content) {
                return Some(table);
            }
        }
    }
    None
}

/// Fast exponentiation for field elements
fn pow_u32(mut base: F, mut e: u32) -> F {
    let mut acc = F::ONE;
    while e > 0 {
        if e & 1 == 1 { acc *= base; }
        base *= base;
        e >>= 1;
    }
    acc
}

/// Evaluate a sparse polynomial at a given point
fn evaluate_poly(poly: &SparsePoly<F>, y: &[F]) -> F {
    let mut result = F::ZERO;
    
    for term in poly.terms() {
        let mut prod = term.coeff;
        for (j, &exp) in term.exps.iter().enumerate() {
            if exp != 0 {
                prod *= pow_u32(y[j], exp as u32);
            }
        }
        result += prod;
    }
    
    result
}
