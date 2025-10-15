use neo::{F};
use neo_ccs::{Mat, CcsStructure, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::PrimeCharacteristicRing;

/// Construct a minimal augmented CCS with an RLC binder where the intended equation
///   <G, z_witness> = rhs_diff (non-zero)
/// holds, and verify that the system correctly implements linear equality encoding.
/// This test FAILS if the system uses multiplicative encoding (<G, z_witness> * rhs_diff = 0)
/// instead of the correct linear equality.
#[test]
fn rlc_binder_linear_equality_correctness() {
    // --- Step CCS (trivial) ---
    // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2); A=B=C=0 so the step portion imposes no constraints.
    let rows = 4usize;
    let cols = 2usize; // witness has at least 2 entries: [const1, y_step]
    let zero = vec![F::ZERO; rows * cols];
    let a = Mat::from_row_major(rows, cols, zero.clone());
    let b = Mat::from_row_major(rows, cols, zero.clone());
    let c = Mat::from_row_major(rows, cols, zero);
    let step_ccs: CcsStructure<F> = r1cs_to_ccs(a, b, c);

    // --- Parameters for augmentation ---
    let y_len = 1usize;         // single y component
    let step_x_len = 1usize;    // one public app input
    let const1_witness_index = 0usize; // step_witness[0] = 1
    let y_step_offset = 1usize;       // step_witness[1] holds y_step

    // No extra binders for x or y_prev in this focused test
    let x_witness_indices: Vec<usize> = vec![];
    let y_prev_witness_indices: Vec<usize> = vec![];

    // EV semantics: u = y_step, y_next = u (full output, not delta)
    // Note: ρ is only used for folding commitments, not for state evolution
    let rho = F::from_u64(3);            // irrelevant for EV
    let y_step = F::from_u64(5);
    let u = y_step;                      // 5
    let y_prev = F::from_u64(7);
    let y_next = u;                      // 5 (= y_step, not y_prev + u)
    let step_x = F::from_u64(42);

    // Witness block: [step_witness || u] where step_witness = [1, y_step]
    // total_wit_cols = step_ccs.m (2) + y_len (1) = 3
    let step_witness = vec![F::ONE, y_step];
    let u_vec = vec![u];

    // Aggregated Ajtai row G over witness columns: pick G = [0, 0, 1] so <G, z_wit> = u
    let total_wit_cols = step_ccs.m + y_len; // 2 + 1 = 3
    let mut g = vec![F::ZERO; total_wit_cols];
    g[2] = F::ONE; // only weigh the u position
    let rhs_diff = u; // Intended: <G, z_wit> = rhs_diff ≠ 0

    // Build augmented CCS with the current RLC binder encoding
    let augmented = neo::build_augmented_ccs_linked_with_rlc(
        &step_ccs,
        step_x_len,
        &[y_step_offset],
        &y_prev_witness_indices,
        &x_witness_indices,
        y_len,
        const1_witness_index,
        Some((g.clone(), rhs_diff)),
    ).expect("failed to build augmented CCS");

    // Public input: [step_x || rho || y_prev || y_next]
    let mut x_pub = Vec::with_capacity(step_x_len + 1 + 2 * y_len);
    x_pub.push(step_x);
    x_pub.push(rho);
    x_pub.push(y_prev);
    x_pub.push(y_next);

    // Witness: [step_witness || u]
    let mut w_priv = step_witness.clone();
    w_priv.extend_from_slice(&u_vec);

    // Under the intended linear RLC equation, this assignment is valid.
    // With the current multiplicative encoding (<G,z> * rhs = 0), it should fail.
    let check = check_ccs_rowwise_zero(&augmented, &x_pub, &w_priv);
    assert!(check.is_ok(), "RLC binder vulnerability detected: system using multiplicative encoding instead of linear equality");
}

/// Regression: the RLC binder must reject assignments where <G, z_witness> != rhs_diff.
/// This guards against accidentally reintroducing the multiplicative encoding.
#[test]
fn rlc_binder_rejects_mismatch() {
    // --- Step CCS (trivial) ---
    let rows = 4usize;  // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 2usize;
    let zero = vec![F::ZERO; rows * cols];
    let a = Mat::from_row_major(rows, cols, zero.clone());
    let b = Mat::from_row_major(rows, cols, zero.clone());
    let c = Mat::from_row_major(rows, cols, zero);
    let step_ccs: CcsStructure<F> = r1cs_to_ccs(a, b, c);

    // --- Parameters for augmentation ---
    let y_len = 1usize;
    let step_x_len = 1usize;
    let const1_witness_index = 0usize; // step_witness[0] = 1
    let y_step_offset = 1usize;        // step_witness[1] holds y_step

    let x_witness_indices: Vec<usize> = vec![];
    let y_prev_witness_indices: Vec<usize> = vec![];

    // EV semantics
    let rho = F::from_u64(3);
    let y_step = F::from_u64(5);
    let u = rho * y_step;              // 15
    let y_prev = F::from_u64(7);
    let y_next = y_prev + u;           // 22
    let step_x = F::from_u64(42);

    // Witness block: [step_witness || u]
    let step_witness = vec![F::ONE, y_step];
    let u_vec = vec![u];

    // Aggregated Ajtai row G over witness columns: pick G = [0, 0, 1] so <G, z_wit> = u
    let total_wit_cols = step_ccs.m + y_len; // 2 + 1 = 3
    let mut g = vec![F::ZERO; total_wit_cols];
    g[2] = F::ONE;
    // Choose rhs different from <G,z>
    let rhs_diff = u + F::from_u64(1);

    let augmented = neo::build_augmented_ccs_linked_with_rlc(
        &step_ccs,
        step_x_len,
        &[y_step_offset],
        &y_prev_witness_indices,
        &x_witness_indices,
        y_len,
        const1_witness_index,
        Some((g.clone(), rhs_diff)),
    ).expect("failed to build augmented CCS");

    // Public input: [step_x || rho || y_prev || y_next]
    let mut x_pub = Vec::with_capacity(step_x_len + 1 + 2 * y_len);
    x_pub.push(step_x);
    x_pub.push(rho);
    x_pub.push(y_prev);
    x_pub.push(y_next);

    // Witness: [step_witness || u]
    let mut w_priv = step_witness.clone();
    w_priv.extend_from_slice(&u_vec);

    let check = check_ccs_rowwise_zero(&augmented, &x_pub, &w_priv);
    assert!(check.is_err(), "RLC binder accepted mismatched equality: expected failure, got success");
}
