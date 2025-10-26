//! CCS constraint checking with diagnostic capture
//!
//! Provides diagnostic-aware wrappers for CCS constraint checking that
//! automatically capture detailed debug information on constraint failures.

use neo_ccs::{CcsStructure, CcsError};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use super::types::{ContextInfo, FoldingContext};
use super::capture::capture_diagnostic;
use super::simple_printer::save_and_print_diagnostic;

/// Check CCS constraints row-wise with automatic diagnostic capture on failure
/// 
/// This wraps `neo_ccs::check_ccs_rowwise_zero` and captures diagnostics when enabled
/// via the `NEO_DIAGNOSTICS` environment variable.
pub fn check_ccs_with_diagnostics(
    s: &CcsStructure<F>,
    x: &[F],
    w: &[F],
    context: ContextInfo,
) -> Result<(), CcsError> {
    // Check if diagnostics are enabled
    let diagnostics_enabled = std::env::var("NEO_DIAGNOSTICS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    
    if !diagnostics_enabled {
        // Fast path: no diagnostics
        return neo_ccs::check_ccs_rowwise_zero(s, x, w);
    }
    
    // Diagnostic path: check row-by-row to capture failure info
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len{ 
            context: "z = x||w length", 
            expected: s.m, 
            got: x.len()+w.len() 
        });
    }
    
    let mut z = x.to_vec(); 
    z.extend_from_slice(w);
    
    // Compute M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let v = neo_ccs::mat_vec_mul_ff::<F>(mj.as_slice(), s.n, s.m, &z);
        mz.push(v);
    }
    
    // Row-wise: for each i, evaluate f( (M_1 z)[i], ..., (M_t z)[i] ) == 0
    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() { 
            point.push(mz[j][i]); 
        }
        let val = s.f.eval(&point);
        
        if val != F::ZERO {
            // Capture and save diagnostic
            let folding_context = extract_folding_context();
            let diagnostic = capture_diagnostic(
                s,
                i,  // failing row
                &z,
                context.clone(),
                folding_context,
                None,  // transcript_challenges
            );
            
            // Save to file and print summary
            if let Err(e) = save_and_print_diagnostic(&diagnostic) {
                eprintln!("⚠️  Failed to save diagnostic: {}", e);
            }
            
            return Err(CcsError::RowFail{ row: i });
        }
    }
    
    Ok(())
}

/// Extract folding context from environment or use defaults
fn extract_folding_context() -> FoldingContext {
    // Try to get from environment, otherwise use sensible defaults
    let b = std::env::var("NEO_PARAM_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    
    let k = std::env::var("NEO_PARAM_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    
    FoldingContext {
        phase: "CCS_CHECK".to_string(),
        b,
        k,
        norm_bound: b.pow(k as u32),
        expansion_factor: 2,
        challenge_set: "neo@v1".to_string(),
    }
}
