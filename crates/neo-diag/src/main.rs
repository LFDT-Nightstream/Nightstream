//! Neo Constraint Diagnostic Replayer
//!
//! Tool for replaying and analyzing constraint diagnostics captured during proving.

use clap::Parser;
use std::path::PathBuf;
use neo_fold::diagnostic::{load_diagnostic, ConstraintDiagnostic, PolyCanonical};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[derive(Parser)]
#[clap(name = "neo-diag")]
#[clap(about = "Neo Constraint Diagnostic Replayer", long_about = None)]
struct Args {
    /// Path to diagnostic file (.json, .json.gz, or .cbor)
    #[clap(value_parser)]
    diagnostic: PathBuf,
    
    /// Verbose output
    #[clap(short, long)]
    verbose: bool,
    
    /// Generate regression test
    #[clap(short = 't', long)]
    generate_test: Option<PathBuf>,
    
    /// Show witness values
    #[clap(short = 'w', long)]
    show_witness: bool,
    
    /// Show gradient blame details
    #[clap(short = 'g', long)]
    show_gradient: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    // Load diagnostic
    println!("📂 Loading diagnostic from: {}", args.diagnostic.display());
    let diagnostic = load_diagnostic(&args.diagnostic)?;
    
    // Display header
    println!("\n{}", "=".repeat(70));
    println!("🔍 CONSTRAINT DIAGNOSTIC");
    println!("{}", "=".repeat(70));
    
    // Basic info
    println!("\n📋 Context:");
    println!("   Test:        {}", diagnostic.context.test);
    println!("   Step:        {}", diagnostic.context.step_idx);
    println!("   Instruction: {}", diagnostic.context.instruction);
    println!("   Phase:       {}", diagnostic.folding.phase);
    println!("   Constraint:  {} (row in CCS)", diagnostic.structure.row_index);
    
    if let Some(reason) = &diagnostic.context.failure_reason {
        println!("   Reason:      {}", reason);
    }
    
    // Structure info
    println!("\n📐 CCS Structure:");
    println!("   Hash:       {}", diagnostic.structure.hash);
    println!("   Matrices:   {} (t)", diagnostic.structure.t);
    println!("   Constraints: {} (n)", diagnostic.structure.n);
    println!("   Variables:   {} (m)", diagnostic.structure.m);
    
    // Field info
    println!("\n🔢 Field:");
    println!("   Name:     {}", diagnostic.field.name);
    println!("   Modulus:  {}", diagnostic.field.q);
    println!("   Ext deg:  {}", diagnostic.field.ext_deg);
    
    // Evaluation info
    println!("\n📊 Evaluation:");
    println!("   Expected:     {}", diagnostic.eval.expected);
    println!("   Actual:       {}", diagnostic.eval.actual);
    println!("   Delta:        {} (canonical)", diagnostic.eval.delta);
    println!("   Delta signed: {}", diagnostic.eval.delta_signed);
    
    if args.verbose {
        println!("\n🔬 Verbose Details:");
        
        // Sparse rows
        println!("\n   Sparse Rows (non-zero entries):");
        for row in &diagnostic.structure.rows {
            println!("     Matrix M{}: {} non-zero entries", 
                row.matrix_j, row.idx.len());
            if row.idx.len() < 10 {
                for (idx, val) in row.idx.iter().zip(row.val.iter()) {
                    println!("       [{}] = {}", idx, val);
                }
            }
        }
        
        // Witness policy
        println!("\n   Witness Policy: {:?}", diagnostic.witness.policy);
        println!("   Witness Disclosed: {} / {} entries",
            diagnostic.witness.z_sparse.idx.len(),
            diagnostic.witness.size);
    }
    
    // Gradient blame
    if !diagnostic.witness.gradient_blame.is_empty() {
        println!("\n🎯 Gradient Blame Analysis (Top Contributors):");
        let show_count = if args.show_gradient { 
            diagnostic.witness.gradient_blame.len() 
        } else { 
            diagnostic.witness.gradient_blame.len().min(5) 
        };
        
        for (rank, contrib) in diagnostic.witness.gradient_blame.iter().take(show_count).enumerate() {
            let name = contrib.name.as_ref()
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            
            println!("   {}. z[{}]{}", rank + 1, contrib.i, name);
            println!("      ∂r/∂z[{}] = {} (abs: {})",
                contrib.i, contrib.gradient, contrib.gradient_abs);
            println!("      z[{}] = {}", contrib.i, contrib.z_value);
            println!("      contribution = {}", contrib.contribution);
        }
        
        if !args.show_gradient && diagnostic.witness.gradient_blame.len() > 5 {
            println!("   ... {} more (use --show-gradient)", 
                diagnostic.witness.gradient_blame.len() - 5);
        }
    }
    
    // Witness values
    if args.show_witness {
        println!("\n📝 Witness Values (disclosed):");
        for (idx, val) in diagnostic.witness.z_sparse.idx.iter()
            .zip(diagnostic.witness.z_sparse.val.iter())
            .take(20)
        {
            println!("   z[{}] = {}", idx, val);
        }
        if diagnostic.witness.z_sparse.idx.len() > 20 {
            println!("   ... {} more values", diagnostic.witness.z_sparse.idx.len() - 20);
        }
    }
    
    // Replay constraint
    println!("\n🔄 Replaying Constraint...");
    let replay_result = replay_constraint_minimal(&diagnostic)?;
    
    println!("\n✅ Replay Results:");
    println!("   Expected:     {}", replay_result.expected);
    println!("   Computed:     {}", replay_result.actual);
    println!("   Delta:        {}", replay_result.delta);
    println!("   Delta signed: {}", replay_result.delta_signed);
    
    if replay_result.matches_dump {
        println!("\n✅ SUCCESS: Replay matches diagnostic (residual verified)");
    } else {
        println!("\n⚠️  WARNING: Replay mismatch detected!");
        println!("   This may indicate:");
        println!("   - Non-deterministic transcript");
        println!("   - Incomplete witness disclosure");
        println!("   - Different field arithmetic");
    }
    
    // Generate regression test if requested
    if let Some(test_path) = args.generate_test {
        println!("\n📝 Generating regression test...");
        generate_regression_test(&diagnostic, &test_path)?;
        println!("✅ Test generated: {}", test_path.display());
    }
    
    println!("\n{}", "=".repeat(70));
    
    // Exit code based on match
    if replay_result.matches_dump {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

struct ReplayResult {
    expected: String,
    actual: String,
    delta: String,
    delta_signed: String,
    matches_dump: bool,
}

/// Replay constraint evaluation (minimal, no full matrix reconstruction)
fn replay_constraint_minimal(
    diagnostic: &ConstraintDiagnostic,
) -> anyhow::Result<ReplayResult> {
    // Reconstruct sparse witness
    let mut witness = vec![F::ZERO; diagnostic.witness.size];
    for (&idx, val_str) in diagnostic.witness.z_sparse.idx.iter()
        .zip(diagnostic.witness.z_sparse.val.iter())
    {
        witness[idx] = parse_canonical_hex(val_str)?;
    }
    
    // Compute y_j = M_j[row, :] · z for each matrix (from sparse rows)
    let mut y_vals = Vec::new();
    for sparse_row in &diagnostic.structure.rows {
        let mut y_j = F::ZERO;
        for (&col_idx, val_str) in sparse_row.idx.iter().zip(sparse_row.val.iter()) {
            let m_ij = parse_canonical_hex(val_str)?;
            y_j += m_ij * witness[col_idx];
        }
        y_vals.push(y_j);
    }
    
    // Evaluate constraint polynomial f(y_1, ..., y_t)
    let residual = evaluate_poly_canonical(
        &diagnostic.structure.f_canonical,
        &y_vals,
    )?;
    
    let delta_str = field_to_canonical_hex(residual);
    let matches = delta_str == diagnostic.eval.delta;
    
    Ok(ReplayResult {
        expected: diagnostic.eval.expected.clone(),
        actual: field_to_canonical_hex(residual),
        delta: delta_str,
        delta_signed: field_to_signed_string(residual),
        matches_dump: matches,
    })
}

/// Evaluate polynomial from canonical representation
fn evaluate_poly_canonical(
    poly: &PolyCanonical,
    y: &[F],
) -> anyhow::Result<F> {
    let mut result = F::ZERO;
    
    for term in &poly.terms {
        // Decode coefficient
        let coeff_bytes = hex::decode(&term.coeff)?;
        let coeff_u64 = u64::from_le_bytes(coeff_bytes.try_into()
            .map_err(|v: Vec<u8>| anyhow::anyhow!("Expected 8 bytes, got {}", v.len()))?);
        let coeff = F::from_u64(coeff_u64);
        
        // Compute product of variables with exponents
        let mut prod = coeff;
        for &(var_idx, exp) in &term.vars {
            prod *= pow_u32(y[var_idx], exp);
        }
        
        result += prod;
    }
    
    Ok(result)
}

/// Compute base^exp using square-and-multiply
fn pow_u32(mut base: F, mut exp: u32) -> F {
    if exp == 0 { return F::ONE; }
    let mut acc = F::ONE;
    while exp > 0 {
        if exp & 1 == 1 { acc *= base; }
        base *= base;
        exp >>= 1;
    }
    acc
}

/// Parse field element from canonical hex (LE bytes)
fn parse_canonical_hex(hex_str: &str) -> anyhow::Result<F> {
    let bytes = hex::decode(hex_str)?;
    let u64_val = u64::from_le_bytes(bytes.try_into()
        .map_err(|v: Vec<u8>| anyhow::anyhow!("Expected 8 bytes, got {}", v.len()))?);
    Ok(F::from_u64(u64_val))
}

/// Convert field element to canonical hex
fn field_to_canonical_hex(f: F) -> String {
    hex::encode(f.as_canonical_u64().to_le_bytes())
}

/// Convert field element to signed string
fn field_to_signed_string(f: F) -> String {
    let canonical = f.as_canonical_u64();
    let modulus = F::ORDER_U64;
    let half = modulus / 2;
    
    if canonical > half {
        format!("-{}", modulus - canonical)
    } else {
        canonical.to_string()
    }
}

/// Generate regression test from diagnostic
fn generate_regression_test(
    diagnostic: &ConstraintDiagnostic,
    out_path: &std::path::Path,
) -> anyhow::Result<()> {
    let test_code = format!(r#"
//! Auto-generated regression test from constraint diagnostic
//!
//! Schema: {}
//! Test: {} / Step: {} / Constraint: {}
//! Phase: {}

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_constraint_{}_regression() {{
    // Sparse witness (only disclosed values)
    let mut z = vec![F::ZERO; {}];
{}
    
    // Sparse rows for failing constraint
{}
    
    // Compute y_j = M_j[row,:] * z for each matrix
    let mut y_vals = Vec::new();
{}
    
    // Evaluate constraint polynomial
    // For R1CS: residual = y[0] * y[1] - y[2]
    // For general CCS: evaluate f(y_1, ..., y_t)
    let residual = {}; // Simplified for R1CS
    
    assert_eq!(residual, F::ZERO,
        "Constraint {} violated: residual = {{}} (signed: {{}})",
        residual.as_canonical_u64(),
        field_to_signed(residual)
    );
}}

fn field_to_signed(f: F) -> String {{
    let canonical = f.as_canonical_u64();
    let modulus = F::ORDER_U64;
    let half = modulus / 2;
    if canonical > half {{
        format!("-{{}}", modulus - canonical)
    }} else {{
        canonical.to_string()
    }}
}}
"#,
        diagnostic.schema,
        diagnostic.context.test,
        diagnostic.context.step_idx,
        diagnostic.structure.row_index,
        diagnostic.folding.phase,
        diagnostic.structure.row_index,
        diagnostic.witness.size,
        generate_witness_init(diagnostic),
        generate_sparse_rows_init(diagnostic),
        generate_y_computation(diagnostic),
        generate_residual_computation(diagnostic),
        diagnostic.structure.row_index,
    );
    
    std::fs::write(out_path, test_code)?;
    Ok(())
}

fn generate_witness_init(diagnostic: &ConstraintDiagnostic) -> String {
    diagnostic.witness.z_sparse.idx.iter()
        .zip(diagnostic.witness.z_sparse.val.iter())
        .map(|(idx, val)| {
            format!("    z[{}] = F::from_canonical_u64(u64::from_le_bytes(hex::decode(\"{}\").unwrap().try_into().unwrap()));", idx, val)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_sparse_rows_init(diagnostic: &ConstraintDiagnostic) -> String {
    let mut lines = Vec::new();
    for row in &diagnostic.structure.rows {
        lines.push(format!("    // Matrix M{}", row.matrix_j));
        lines.push(format!("    let mut row_{} = vec![(0usize, F::ZERO); {}];", row.matrix_j, row.idx.len()));
        for (i, (idx, val)) in row.idx.iter().zip(row.val.iter()).enumerate() {
            lines.push(format!(
                "    row_{}[{}] = ({}, F::from_canonical_u64(u64::from_le_bytes(hex::decode(\"{}\").unwrap().try_into().unwrap())));",
                row.matrix_j, i, idx, val
            ));
        }
    }
    lines.join("\n")
}

fn generate_y_computation(diagnostic: &ConstraintDiagnostic) -> String {
    let mut lines = Vec::new();
    for j in 0..diagnostic.structure.t {
        lines.push(format!("    let mut y_{} = F::ZERO;", j));
        lines.push(format!("    for (col, val) in &row_{} {{", j));
        lines.push(format!("        y_{} += *val * z[*col];", j));
        lines.push("    }".to_string());
        lines.push(format!("    y_vals.push(y_{});", j));
    }
    lines.join("\n")
}

fn generate_residual_computation(diagnostic: &ConstraintDiagnostic) -> String {
    // Simple R1CS case: y[0] * y[1] - y[2]
    if diagnostic.structure.t == 3 {
        "y_vals[0] * y_vals[1] - y_vals[2]".to_string()
    } else {
        // Generic CCS would require parsing f_canonical
        "/* evaluate f_canonical here */".to_string()
    }
}

