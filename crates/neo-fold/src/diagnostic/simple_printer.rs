//! Diagnostic file writer with simple console output
//! 
//! Writes comprehensive JSON diagnostic files and prints summary to console

use super::types::{ConstraintDiagnostic, SparseRow};
use neo_math::F;
use p3_field::PrimeField64;
use std::path::{Path, PathBuf};
use std::io::Write;

/// Save diagnostic to file and print summary
/// 
/// Returns the path to the saved file
pub fn save_and_print_diagnostic(diagnostic: &ConstraintDiagnostic) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Get diagnostic output directory from environment or use default
    let diag_dir = std::env::var("NEO_DIAGNOSTICS_DIR")
        .unwrap_or_else(|_| "./neo_diagnostics".to_string());
    
    let diag_path = Path::new(&diag_dir);
    std::fs::create_dir_all(diag_path)?;
    
    // Generate filename with timestamp
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!(
        "constraint_failure_step{}_row{}_{}.json",
        diagnostic.context.step_idx,
        diagnostic.structure.row_index,
        timestamp
    );
    
    let file_path = diag_path.join(&filename);
    
    // Write full JSON diagnostic with restricted permissions (0600 on Unix)
    use std::fs::OpenOptions;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    
    let json = serde_json::to_string_pretty(diagnostic)?;
    
    let mut opts = OpenOptions::new();
    opts.create(true).write(true).truncate(true);
    #[cfg(unix)]
    opts.mode(0o600);  // Owner read/write only - contains sensitive witness data!
    
    let mut file = opts.open(&file_path)?;
    file.write_all(json.as_bytes())?;
    
    // Also write a human-readable summary
    let summary_path = diag_path.join(format!("summary_{}", filename.replace(".json", ".txt")));
    let mut summary = std::fs::File::create(&summary_path)?;
    write_summary(&mut summary, diagnostic)?;
    
    // Print to console (gated by NEO_DIAGNOSTIC_STDERR to prevent CI log leakage)
    // Default: OFF to avoid accidentally leaking witness data to CI logs (CWE-532)
    let print_to_stderr = std::env::var("NEO_DIAGNOSTIC_STDERR")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    
    if print_to_stderr {
        print_console_summary(diagnostic, &file_path);
    } else {
        eprintln!("📊 Constraint diagnostic saved to: {}", file_path.display());
        eprintln!("   (Set NEO_DIAGNOSTIC_STDERR=1 to print full summary to console)");
    }
    
    Ok(file_path)
}

/// Print compact summary to console
fn print_console_summary(diagnostic: &ConstraintDiagnostic, saved_to: &Path) {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("❌ CONSTRAINT FAILURE DETECTED");
    eprintln!("{}", "=".repeat(80));
    
    // Context info
    eprintln!("\n📋 Context:");
    eprintln!("  Test: {}", diagnostic.context.test);
    eprintln!("  Step: {}", diagnostic.context.step_idx);
    eprintln!("  Instruction: {}", diagnostic.context.instruction);
    eprintln!("  Row: {}", diagnostic.structure.row_index);
    if let Some(reason) = &diagnostic.context.failure_reason {
        eprintln!("  Reason: {}", reason);
    }
    
    // Print sparse rows (for R1CS: A, B, C)
    eprintln!("\n📊 Constraint Rows:");
    for sparse_row in &diagnostic.structure.rows {
        let row_name = match sparse_row.matrix_j {
            0 => "row_a",
            1 => "row_b",
            2 => "row_c",
            j => {
                eprintln!("  row_{}[{}]={}", j, diagnostic.structure.row_index, 
                    format_sparse_row(sparse_row, diagnostic.structure.m));
                continue;
            }
        };
        
        eprintln!("  {}[{}]={}", row_name, diagnostic.structure.row_index, 
            format_sparse_row(sparse_row, diagnostic.structure.m));
    }
    
    // Print witness (respect redaction)
    eprintln!("\n🔍 Witness:");
    let is_redacted = diagnostic.witness.w_private.iter().any(|s| s.starts_with("REDACTED:"));
    if is_redacted {
        eprintln!("  x (public) = \"{}\"", format_witness_full(&diagnostic.witness.x_public));
        let w_len = diagnostic.witness.size.saturating_sub(diagnostic.witness.x_public.len());
        eprintln!("  w (private) = [REDACTED {} elements]", w_len);
        eprintln!("  💡 Set NEO_DIAGNOSTIC_ALLOW_FULL_WITNESS=1 to see full witness (DANGEROUS!)");
    } else {
        eprintln!("  z (full) = \"{}\"", format_witness_full(&diagnostic.witness.z_full));
    }
    
    // Print evaluation results (for R1CS)
    if let Some(ref r1cs) = diagnostic.eval.r1cs_products {
        eprintln!("\n📈 Products:");
        eprintln!("  az = {}", parse_and_format_as_decimal(&r1cs.az));
        eprintln!("  bz = {}", parse_and_format_as_decimal(&r1cs.bz));
        eprintln!("  cz = {}", parse_and_format_as_decimal(&r1cs.cz));
    }
    
    // Print residual
    eprintln!("\n⚠️  Residual:");
    eprintln!("  Expected: {}", diagnostic.eval.expected);
    eprintln!("  Actual:   {}", diagnostic.eval.actual);
    eprintln!("  Delta:    {} (signed: {})", diagnostic.eval.delta, diagnostic.eval.delta_signed);
    
    // Print top gradient blame
    if !diagnostic.witness.gradient_blame.is_empty() {
        eprintln!("\n🎯 Top Contributors (Gradient Blame):");
        for (rank, contrib) in diagnostic.witness.gradient_blame.iter().take(5).enumerate() {
            let name = contrib.name.as_ref()
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            eprintln!("  {}. z[{}]{}: gradient={}, contribution={}", 
                rank + 1, contrib.i, name, 
                parse_and_format_as_decimal(&contrib.gradient),
                parse_and_format_as_decimal(&contrib.contribution));
        }
    }
    
    eprintln!("\n📁 Full diagnostic saved to:");
    eprintln!("   {}", saved_to.display());
    eprintln!("{}", "=".repeat(80));
}

/// Write detailed summary to file
fn write_summary(writer: &mut dyn Write, diagnostic: &ConstraintDiagnostic) -> std::io::Result<()> {
    writeln!(writer, "CONSTRAINT FAILURE DIAGNOSTIC")?;
    writeln!(writer, "{}", "=".repeat(80))?;
    writeln!(writer)?;
    
    writeln!(writer, "Context:")?;
    writeln!(writer, "  Test: {}", diagnostic.context.test)?;
    writeln!(writer, "  Step: {}", diagnostic.context.step_idx)?;
    writeln!(writer, "  Instruction: {}", diagnostic.context.instruction)?;
    writeln!(writer, "  Row: {}", diagnostic.structure.row_index)?;
    if let Some(reason) = &diagnostic.context.failure_reason {
        writeln!(writer, "  Reason: {}", reason)?;
    }
    writeln!(writer)?;
    
    writeln!(writer, "CCS Structure:")?;
    writeln!(writer, "  Hash: {}", diagnostic.structure.hash)?;
    writeln!(writer, "  Matrices (t): {}", diagnostic.structure.t)?;
    writeln!(writer, "  Constraints (n): {}", diagnostic.structure.n)?;
    writeln!(writer, "  Variables (m): {}", diagnostic.structure.m)?;
    writeln!(writer)?;
    
    writeln!(writer, "Constraint Rows (dense format):")?;
    for sparse_row in &diagnostic.structure.rows {
        writeln!(writer, "  row_{}[{}] = {}", 
            sparse_row.matrix_j, 
            diagnostic.structure.row_index,
            format_sparse_row(sparse_row, diagnostic.structure.m))?;
    }
    writeln!(writer)?;
    
    writeln!(writer, "Witness (z):")?;
    writeln!(writer, "  z = {}", format_witness_full(&diagnostic.witness.z_full))?;
    writeln!(writer)?;
    
    if let Some(ref r1cs) = diagnostic.eval.r1cs_products {
        writeln!(writer, "Evaluation Products:")?;
        writeln!(writer, "  az = {}", parse_and_format_as_decimal(&r1cs.az))?;
        writeln!(writer, "  bz = {}", parse_and_format_as_decimal(&r1cs.bz))?;
        writeln!(writer, "  cz = {}", parse_and_format_as_decimal(&r1cs.cz))?;
        writeln!(writer)?;
    }
    
    writeln!(writer, "Residual:")?;
    writeln!(writer, "  Expected: {}", diagnostic.eval.expected)?;
    writeln!(writer, "  Actual:   {}", diagnostic.eval.actual)?;
    writeln!(writer, "  Delta:    {} (signed: {})", diagnostic.eval.delta, diagnostic.eval.delta_signed)?;
    writeln!(writer)?;
    
    if !diagnostic.witness.gradient_blame.is_empty() {
        writeln!(writer, "Top Contributors (Gradient Blame):")?;
        for (rank, contrib) in diagnostic.witness.gradient_blame.iter().take(10).enumerate() {
            let name = contrib.name.as_ref()
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            writeln!(writer, "  {}. z[{}]{}", rank + 1, contrib.i, name)?;
            writeln!(writer, "     gradient = {}", parse_and_format_as_decimal(&contrib.gradient))?;
            writeln!(writer, "     contribution = {}", parse_and_format_as_decimal(&contrib.contribution))?;
        }
    }
    
    Ok(())
}

/// Format sparse row as comma-separated dense vector
fn format_sparse_row(sparse_row: &SparseRow, m: usize) -> String {
    let mut dense = vec![0u64; m];
    
    for (&idx, val_hex) in sparse_row.idx.iter().zip(&sparse_row.val) {
        if idx < m {
            if let Ok(bytes) = hex::decode(val_hex) {
                if bytes.len() == 8 {
                    if let Ok(bytes_array) = <[u8; 8]>::try_from(bytes) {
                        dense[idx] = u64::from_le_bytes(bytes_array);
                    }
                }
            }
        }
    }
    
    dense.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Format full witness as comma-separated decimal values
fn format_witness_full(z_full: &[String]) -> String {
    z_full.iter()
        .map(|hex| parse_and_format_as_decimal(hex))
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse hex canonical value and format as signed decimal
fn parse_and_format_as_decimal(hex_str: &str) -> String {
    // Handle redacted values
    if hex_str.starts_with("REDACTED:") {
        return hex_str.to_string();
    }
    
    if let Ok(bytes) = hex::decode(hex_str) {
        if bytes.len() == 8 {
            if let Ok(bytes_array) = <[u8; 8]>::try_from(bytes) {
                let canonical = u64::from_le_bytes(bytes_array);
                let modulus = F::ORDER_U64;
                let half = modulus / 2;
                
                // If value is in upper half, represent as negative
                if canonical > half {
                    return format!("-{}", modulus - canonical);
                } else {
                    return canonical.to_string();
                }
            }
        }
    }
    hex_str.to_string()
}

/// Print diagnostic in simple developer format (for backward compatibility)
pub fn print_simple_diagnostic(diagnostic: &ConstraintDiagnostic) {
    if let Err(e) = save_and_print_diagnostic(diagnostic) {
        eprintln!("⚠️  Failed to save diagnostic file: {}", e);
        // Fall back to console-only output
        print_console_summary(diagnostic, Path::new("(not saved)"));
    }
}
