//! Logging helpers for crosscheck engine diagnostics.
//!
//! This module contains all the diagnostic logging functions used by the
//! crosscheck engine to report validation mismatches and progress.

#![allow(dead_code)]

use neo_math::{F, K};

pub fn log_per_round_header(total_rounds: usize) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║           CROSSCHECK: Validating Per-Round Polynomials                   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Total rounds: {}", total_rounds);
    eprintln!();
}

pub fn log_per_round_progress(round_idx: usize, total_rounds: usize, num_evals: usize) {
    eprintln!(
        "  Round {}/{}: Checking {} evaluation points...",
        round_idx + 1,
        total_rounds,
        num_evals
    );
}

pub fn log_per_round_success(total_rounds: usize) {
    eprintln!("  ✓ All {} rounds validated successfully", total_rounds);
    eprintln!();
}

pub fn log_per_round_mismatch(round_idx: usize, total_rounds: usize, eval_idx: usize, actual: K, expected: K) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: Per-Round Polynomial Mismatch                   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Round:        {}/{}", round_idx + 1, total_rounds);
    eprintln!("Evaluation:   p({})", eval_idx);
    eprintln!();
    eprintln!("Optimized:    {:?}", actual);
    eprintln!("Paper-exact:  {:?}", expected);
    eprintln!();
}

pub fn log_terminal_header() {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║           CROSSCHECK: Computing Terminal Evaluation Q(α', r')            ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

pub fn log_terminal_optimized_header() {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                    OPTIMIZED ENGINE (from ME outputs)                      ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
}

pub fn log_terminal_optimized_result(rhs_opt: K) {
    eprintln!("  [Optimized] Result: {:?}", rhs_opt);
}

pub fn log_terminal_paper_exact_header() {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                 PAPER-EXACT ENGINE (from witnesses)                        ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
}

pub fn log_terminal_paper_exact_result(lhs_exact: K) {
    eprintln!("  [Paper-exact] Result: {:?}", lhs_exact);
}

pub fn log_terminal_comparison(running_sum_prover: K, rhs_opt: K, lhs_exact: K) {
    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                           COMPARISON                                       ║");
    eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
    eprintln!(
        "  Sumcheck final value (from optimized proof): {:?}",
        running_sum_prover
    );
    eprintln!("  Optimized Q(α', r'):                         {:?}", rhs_opt);
    eprintln!("  Paper-exact Q(α', r'):                       {:?}", lhs_exact);
    eprintln!("  Match: {}", rhs_opt == lhs_exact && rhs_opt == running_sum_prover);
    eprintln!();
}

pub fn log_terminal_mismatch(rhs_opt: K, lhs_exact: K, running_sum_prover: K) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: Terminal Evaluation Claim Mismatch               ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Background:");
    eprintln!("  The optimized engine ran and produced a proof via sumcheck.");
    eprintln!("  Now we're verifying that Q(α', r') can be computed correctly.");
    eprintln!();
    eprintln!("Three ways to compute/check Q(α', r'):");
    eprintln!();
    eprintln!("  1. Optimized terminal formula (from output ME instances):");
    eprintln!("     → {:?}", rhs_opt);
    eprintln!();
    eprintln!("  2. Paper-exact direct evaluation (from witnesses):");
    eprintln!("     → {:?}", lhs_exact);
    eprintln!();
    eprintln!("  3. Sumcheck final value (from optimized engine's proof):");
    eprintln!("     → {:?}", running_sum_prover);
    eprintln!();
    eprintln!("Comparisons:");
    eprintln!("  Optimized terminal == Paper-exact:   {}", rhs_opt == lhs_exact);
    eprintln!(
        "  Optimized terminal == Sumcheck final: {}",
        rhs_opt == running_sum_prover
    );
    eprintln!(
        "  Paper-exact == Sumcheck final:        {}",
        lhs_exact == running_sum_prover
    );
    eprintln!();
    eprintln!("Expected: All three should match.");
    eprintln!("Actual:   Mismatch detected!");
    eprintln!();
    if rhs_opt == running_sum_prover && lhs_exact != running_sum_prover {
        eprintln!("Diagnosis: Optimized engine is self-consistent, but paper-exact");
        eprintln!("           formula produces a different value.");
    } else if lhs_exact == running_sum_prover && rhs_opt != running_sum_prover {
        eprintln!("Diagnosis: Paper-exact matches sumcheck, but optimized terminal");
        eprintln!("           formula produces a different value.");
    } else {
        eprintln!("Diagnosis: Complex mismatch - neither matches sumcheck final value.");
    }
    eprintln!();
    eprintln!("Tip: Set NEO_CROSSCHECK_DETAIL=1 for detailed step-by-step computation logs.");
    eprintln!("═══════════════════════════════════════════════════════════════════════════\n");
}

pub fn log_outputs_length_mismatch(paper_len: usize, optimized_len: usize) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: Output Length Mismatch                           ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Paper-exact outputs: {}", paper_len);
    eprintln!("  Optimized outputs:   {}", optimized_len);
    eprintln!();
}

pub fn log_outputs_metadata_mismatch(
    idx: usize,
    total_len: usize,
    mismatches: &[String],
    m_in_match: bool,
    r_match: bool,
    c_data_match: bool,
    m_in: usize,
    r_len: usize,
    c_data_len: usize,
    fold_digest: &[u8; 32],
    mcs_list_len: usize,
) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!(
        "║              CROSSCHECK: Output Metadata Mismatch at Index {}            ║",
        idx
    );
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Output instance {}/{}", idx + 1, total_len);
    eprintln!();
    eprintln!("Fields that MATCH:");
    if m_in_match {
        eprintln!("  ✓ m_in: {}", m_in);
    }
    if r_match {
        eprintln!("  ✓ r: length {} (all elements match)", r_len);
    }
    if c_data_match {
        eprintln!("  ✓ c.data: length {} (all elements match)", c_data_len);
    }
    eprintln!();
    eprintln!("Fields that MISMATCH:");
    for mismatch in mismatches {
        eprintln!("  ✗ {}", mismatch);
    }
    eprintln!();
    eprintln!("Additional context:");
    eprintln!("  fold_digest (first 8 bytes): {:?}", &fold_digest[..8]);
    eprintln!(
        "  Instance is from: {}",
        if idx < mcs_list_len {
            format!("MCS instance {}", idx)
        } else {
            format!("ME input {}", idx - mcs_list_len)
        }
    );
    eprintln!();
}

pub fn log_outputs_y_row_length_mismatch(idx: usize, j: usize, paper_len: usize, optimized_len: usize) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: y Row Length Mismatch                            ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Output instance: {}", idx);
    eprintln!("  y row index:     {}", j);
    eprintln!("  Paper-exact len: {}", paper_len);
    eprintln!("  Optimized len:   {}", optimized_len);
    eprintln!();
}

pub fn log_outputs_y_row_content_mismatch(
    idx: usize,
    j: usize,
    match_count: usize,
    total_len: usize,
    first_mismatch_idx: usize,
    paper_val: &K,
    opt_val: &K,
) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: y Row Content Mismatch                           ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Output instance:   {}", idx);
    eprintln!("  y row index:       {}", j);
    eprintln!("  Matching elements: {}/{}", match_count, total_len);
    eprintln!();
    eprintln!("  First mismatch at index {}:", first_mismatch_idx);
    eprintln!("    Paper-exact: {:?}", paper_val);
    eprintln!("    Optimized:   {:?}", opt_val);
    eprintln!();
}

pub fn log_outputs_y_scalars_mismatch(idx: usize, match_count: usize, total_len: usize, mismatches: &[(usize, K, K)]) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: y_scalars Mismatch                               ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Output instance:   {}", idx);
    eprintln!("  Matching scalars:  {}/{}", match_count, total_len);
    eprintln!();
    for (k, a_val, b_val) in mismatches {
        eprintln!("  Mismatch at scalar {}:", k);
        eprintln!("    Paper-exact: {:?}", a_val);
        eprintln!("    Optimized:   {:?}", b_val);
    }
    eprintln!();
}

pub fn log_outputs_x_dimension_mismatch(
    idx: usize,
    paper_rows: usize,
    paper_cols: usize,
    opt_rows: usize,
    opt_cols: usize,
) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: X Matrix Dimension Mismatch                      ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Output instance: {}", idx);
    eprintln!("  Paper-exact X:   {} × {}", paper_rows, paper_cols);
    eprintln!("  Optimized X:     {} × {}", opt_rows, opt_cols);
    eprintln!();
}

pub fn log_outputs_x_element_mismatch(idx: usize, r: usize, c: usize, paper_val: &F, opt_val: &F) {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║              CROSSCHECK: X Matrix Element Mismatch                        ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
    eprintln!("  Output instance: {}", idx);
    eprintln!("  Position:        ({}, {})", r, c);
    eprintln!("  Paper-exact:     {:?}", paper_val);
    eprintln!("  Optimized:       {:?}", opt_val);
    eprintln!();
}

pub fn log_paper_exact_feature_warning() {
    eprintln!("WARNING: per_round crosscheck requested but paper-exact feature not enabled");
}
