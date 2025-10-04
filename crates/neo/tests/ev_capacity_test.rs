//! Capacity test: build an augmented CCS with a very large `y_len` and
//! ensure the builder succeeds without a hard D-row cap.
//!
//! EV rows are 2*y_len. With y_len=500 this implies 1000 EV rows
//! before any step rows or binders. Since the augmented CCS no longer
//! enforces a hard D=54 cap, this should succeed. A future EV
//! aggregation can further reduce EV rows to 2, but is not required
//! for this test to pass.

use neo::{F};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};

/// Build a trivial CCS with `rows` identical R1CS constraints of 1*1=1
/// and `m = 1 + y_len` columns where column 0 is the const-1 witness.
fn trivial_step_ccs_rows(y_len: usize, rows: usize) -> CcsStructure<F> {
    let m = 1 + y_len; // const-1 + y_step slots
    let n = rows.max(1);
    let mut a = vec![F::ZERO; n * m];
    let mut b = vec![F::ZERO; n * m];
    let mut c = vec![F::ZERO; n * m];
    for r in 0..n { a[r * m] = F::ONE; b[r * m] = F::ONE; c[r * m] = F::ONE; }
    let a_mat = Mat::from_row_major(n, m, a);
    let b_mat = Mat::from_row_major(n, m, b);
    let c_mat = Mat::from_row_major(n, m, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn large_y_len_augmented_ccs_should_eventually_fit_after_ev_aggregation() {
    // Set a very large y_len so that current EV rows = 2*y_len >> D.
    // "Equivalent to 1000 rows" today means y_len=500 ⇒ 2*y_len=1000 EV rows.
    let y_len = 500usize;

    // Keep the step computation tiny (1 row) so EV dominates the row budget.
    let step_ccs = trivial_step_ccs_rows(y_len, 1);

    // Binding: put y_step in witness indices [1..=y_len]; const1 at 0.
    let y_step_offsets: Vec<usize> = (1..=y_len).collect();
    let y_prev_witness_indices: Vec<usize> = vec![]; // no y_prev binder for this test
    let app_input_witness_indices: Vec<usize> = vec![]; // no app-input binder
    let const1_idx = 0usize;
    let step_x_len = 0usize; // empty step_x for this test

    // Attempt to build the augmented CCS. With the D-cap removed, this should succeed
    // even though EV rows dominate. (Future EV aggregation reduces EV to 2 rows.)
    let augmented = neo::ivc::build_augmented_ccs_linked_with_rlc(
        &step_ccs,
        step_x_len,
        &y_step_offsets,
        &y_prev_witness_indices,
        &app_input_witness_indices,
        y_len,
        const1_idx,
        None,
    );

    assert!(
        augmented.is_ok(),
        "Expected large-y_len augmented CCS to build after EV aggregation; got error: {:?}",
        augmented.err()
    );
}
