//! Diagnostic capture functionality

use super::types::*;
use super::blame::compute_gradient_blame;
use super::hash::{compute_structure_hash_v1, poly_to_canonical_json};
use super::witness::{policy_from_env, field_to_canonical_hex, field_to_signed_string};
use super::export::{export_diagnostic, DiagnosticFormat};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use super::SCHEMA_VERSION;

/// Capture a diagnostic snapshot when a constraint fails
/// 
/// Computes the constraint residual and captures all relevant debug information.
/// The expected value is always zero (constraints should evaluate to zero).
pub fn capture_diagnostic(
    structure: &CcsStructure<F>,
    row_index: usize,
    witness: &[F],
    context: ContextInfo,
    folding_context: FoldingContext,
    transcript_challenges: Option<TranscriptInfo>,
) -> ConstraintDiagnostic {
    // Extract sparse rows for failing constraint
    let sparse_rows: Vec<SparseRow> = structure.matrices.iter()
        .enumerate()
        .map(|(j, mat)| extract_sparse_row(mat, row_index, j))
        .collect();
    
    // Compute gradient-based blame
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
    
    // Create witness info based on policy
    let policy = policy_from_env();
    let witness_info = WitnessInfo::from_policy(
        witness,
        &sparse_rows,
        gradient_blame,
        policy,
    );
    
    // Compute the actual constraint evaluation
    // For CCS: evaluate f(y_1, ..., y_t) where y_j = <M_j[row,:], z>
    let y_vals: Vec<F> = sparse_rows.iter().map(|sparse_row| {
        sparse_row.idx.iter().zip(&sparse_row.val).fold(F::ZERO, |acc, (&col_idx, v_hex)| {
            // Parse hex value back to field element
            let bytes = hex::decode(v_hex).unwrap();
            let u64_val = u64::from_le_bytes(bytes.try_into().unwrap());
            let mat_val = F::from_u64(u64_val);
            acc + mat_val * witness[col_idx]
        })
    }).collect();
    
    // Evaluate the polynomial f(y_1, ..., y_t)
    let actual = evaluate_poly(&structure.f, &y_vals);
    
    // Expected is always zero (constraints should be satisfied)
    let expected = F::ZERO;
    let residual = actual - expected;
    
    ConstraintDiagnostic {
        schema: SCHEMA_VERSION.to_string(),
        
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
            t: structure.t(),
            n: structure.n,
            m: structure.m,
            row_index,
            rows: sparse_rows,
            f_canonical: poly_to_canonical_json(&structure.f),
        },
        
        eval: EvaluationInfo {
            r: vec![],  // Filled by caller if applicable
            y_at_r: vec![],  // Filled by caller if applicable
            r1cs_products: None,  // Filled by caller if R1CS
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

/// Capture and export diagnostic (convenience function)
pub fn capture_and_export_diagnostic(
    structure: &CcsStructure<F>,
    row_index: usize,
    witness: &[F],
    context: ContextInfo,
    folding_context: FoldingContext,
    transcript_challenges: Option<TranscriptInfo>,
) -> Result<std::path::PathBuf, std::io::Error> {
    let diagnostic = capture_diagnostic(
        structure,
        row_index,
        witness,
        context,
        folding_context,
        transcript_challenges,
    );
    
    let format = DiagnosticFormat::from_env();
    export_diagnostic(&diagnostic, format)
}

/// Evaluate a sparse polynomial at a given point
fn evaluate_poly(poly: &SparsePoly<F>, y: &[F]) -> F {
    let mut result = F::ZERO;
    
    for term in poly.terms() {
        let mut prod = term.coeff;
        for (var_idx, &exp) in term.exps.iter().enumerate() {
            if exp > 0 {
                // Compute y[var_idx]^exp
                let base = y[var_idx];
                for _ in 1..exp {
                    prod *= base;
                }
            }
        }
        result += prod;
    }
    
    result
}

