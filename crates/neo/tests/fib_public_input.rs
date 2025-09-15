use anyhow::Result;
use neo::{prove, ProveInput, NeoParams, CcsStructure, F, claim_z_eq, verify_and_extract_exact};
use neo_ccs::{r1cs_to_ccs, Mat};
use p3_field::PrimeCharacteristicRing;

/// Build CCS for Fibonacci with a public input p: enforce z0 = p, z1 = 1, z_{i+2} = z_{i+1} + z_i
/// Column layout: [p (public), const1, z0, z1, z2, ..., z_n]
fn fibonacci_ccs_with_public_seed(n: usize) -> CcsStructure<F> {
    assert!(n >= 1, "n must be >= 1");

    let rows = n + 1;        // 2 seed rows + (n-1) recurrence rows
    let cols = n + 3;        // [p, 1, z0, z1, ..., z_n] → total n+3 columns

    // Dense row-major allocation helpers
    fn zero_mat(rows: usize, cols: usize) -> Vec<F> { vec![F::ZERO; rows * cols] }
    fn set(m: &mut [F], _rows: usize, cols: usize, r: usize, c: usize, v: F) { m[r * cols + c] = v; }

    let mut a = zero_mat(rows, cols);
    let mut b = zero_mat(rows, cols);
    let c = zero_mat(rows, cols); // always zero

    // Row 0: z0 - p = 0  => A: +1*z0 + (-1)*p; B: *1
    set(&mut a, rows, cols, 0, 2, F::ONE);     // z0 is at col 2
    set(&mut a, rows, cols, 0, 0, -F::ONE);    // p is at col 0
    set(&mut b, rows, cols, 0, 1, F::ONE);     // const 1 at col 1

    // Row 1: z1 - 1 = 0  => A: +1*z1 + (-1)*const1; B: *1
    set(&mut a, rows, cols, 1, 3, F::ONE);     // z1 at col 3
    set(&mut a, rows, cols, 1, 1, -F::ONE);    // -1*const1 at col 1
    set(&mut b, rows, cols, 1, 1, F::ONE);     // const 1 at col 1

    // Recurrence rows: for i in 0..(n-1)
    // Row (2+i): z[i+2] - z[i+1] - z[i] = 0
    // z[i] starts at col 2, so z[i] is col 2+i
    for i in 0..(n - 1) {
        let r = 2 + i;
        set(&mut a, rows, cols, r, 4 + i,  F::ONE);  // z[i+2]
        set(&mut a, rows, cols, r, 3 + i, -F::ONE);  // -z[i+1]
        set(&mut a, rows, cols, r, 2 + i, -F::ONE);  // -z[i]
        set(&mut b, rows, cols, r, 1, F::ONE);       // *1 via const column
    }

    let a = Mat::from_row_major(rows, cols, a);
    let b = Mat::from_row_major(rows, cols, b);
    let c = Mat::from_row_major(rows, cols, c);

    r1cs_to_ccs(a, b, c)
}

/// Build witness vector for given n and public seed p: returns witness (private) part only
/// Witness layout expected by ProveInput: witness corresponds to columns after public inputs
/// Here: witness = [1, z0, z1, ..., z_n] (length n+2), public_input = [p]
fn generate_fib_witness_with_public_seed(n: usize, p: F) -> Vec<F> {
    assert!(n >= 1);
    let mut w = Vec::with_capacity(n + 2);
    w.push(F::ONE);  // const1 at col 1
    w.push(p);       // z0 equals public p enforced by constraints, but included in witness layout after public slice
    w.push(F::ONE);  // z1 = 1
    while w.len() < n + 2 {
        let len = w.len();
        let next = w[len - 1] + w[len - 2];
        w.push(next);
    }
    w
}

#[test]
fn prove_and_verify_fibonacci_with_public_seed() -> Result<()> {
    // Problem size
    let n = 12usize;

    // Build CCS that uses a public input p for z0
    let ccs = fibonacci_ccs_with_public_seed(n);

    // Parameters
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Choose public seed p and construct private witness accordingly
    let p = F::from_u64(7);
    let public_input = vec![p];
    let witness = generate_fib_witness_with_public_seed(n, p);

    // Expose final value z_{n} as an output claim
    // Overall column layout is [p | 1, z0, z1, ..., z_n], but claim_z_eq expects index in the full z
    // Prover composes z = [public | witness], so index of z_n is: 1 (const) + n (z0..z_{n-1}) + 1 (for z_n) = n+2 overall,
    // but zero-based and including public slice means idx = (public.len()) + (1 /*const*/ + n /*z0..z_{n-1}*/)
    let idx_zn = public_input.len() + (1 + n);
    let final_value = witness.last().copied().unwrap();
    let claim = claim_z_eq(&params, ccs.m, idx_zn, final_value);

    // Prove
    let proof = prove(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[claim],
    })?;

    // Verify success with correct public input
    let outputs = verify_and_extract_exact(&ccs, &public_input, &proof, 1)?;
    assert_eq!(outputs[0], final_value, "extracted output must match claimed final Fibonacci value");

    // Verify failure with wrong public input (binds the instance to public p)
    let wrong_public = vec![F::from_u64(8)];
    let bad = verify_and_extract_exact(&ccs, &wrong_public, &proof, 1);
    assert!(bad.is_err(), "verification must fail for wrong public input");

    Ok(())
}


