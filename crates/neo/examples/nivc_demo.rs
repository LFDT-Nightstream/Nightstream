//! Minimal NIVC demo: two heterogeneous step types, à‑la‑carte steps.
//!
//! Run with:
//!   NEO_DETERMINISTIC=1 cargo run -p neo --example nivc_demo

use neo::{self, F, CcsStructure, NeoParams};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn trivial_step_ccs_rows(y_len: usize, rows: usize) -> CcsStructure<F> {
    // Very small CCS derived from a trivial R1CS (1*1=1) repeated `rows` times.
    let m = 1 + y_len; // const-1 + y_step slots
    let n = rows.max(1);
    let mut a = vec![F::ZERO; n * m];
    let mut b = vec![F::ZERO; n * m];
    let mut c = vec![F::ZERO; n * m];
    for r in 0..n { a[r * m] = F::ONE; b[r * m] = F::ONE; c[r * m] = F::ONE; }
    let a_mat = neo_ccs::Mat::from_row_major(n, m, a);
    let b_mat = neo_ccs::Mat::from_row_major(n, m, b);
    let c_mat = neo_ccs::Mat::from_row_major(n, m, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

fn main() -> anyhow::Result<()> {
    // Make runs reproducible in examples
    std::env::set_var("NEO_DETERMINISTIC", "1");

    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 2usize;

    // Two different CCS shapes (same m, different n)
    let ccs_a = trivial_step_ccs_rows(y_len, 1);
    let ccs_b = trivial_step_ccs_rows(y_len, 2);

    // Binding: y_step lives at witness indices [1..=y_len]; const1 at 0
    let binding = neo::ivc::StepBindingSpec {
        y_step_offsets: (1..=y_len).collect(),
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    let program = neo::NivcProgram::new(vec![
        neo::NivcStepSpec { ccs: ccs_a.clone(), binding: binding.clone() },
        neo::NivcStepSpec { ccs: ccs_b.clone(), binding: binding.clone() },
    ]);

    let y0 = vec![F::from_u64(1), F::from_u64(2)];
    let mut st = neo::NivcState::new(params.clone(), program.clone(), y0.clone())?;

    // Step schedule: A (type 0) → B (type 1) → A (type 0)
    // Witness layout: [1, y_step0, y_step1]
    let w0 = vec![F::ONE, F::from_u64(3),  F::from_u64(5)];  // type 0
    let w1 = vec![F::ONE, F::from_u64(7),  F::from_u64(11)]; // type 1
    let w2 = vec![F::ONE, F::from_u64(13), F::from_u64(17)]; // type 0

    let sp0 = st.step(0, &[], &w0)?; // A
    println!("step0: which_type={}, y_next={:?}", sp0.which_type, sp0.inner.next_accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());

    let sp1 = st.step(1, &[], &w1)?; // B
    println!("step1: which_type={}, y_next={:?}", sp1.which_type, sp1.inner.next_accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());

    let sp2 = st.step(0, &[], &w2)?; // A
    println!("step2: which_type={}, y_next={:?}", sp2.which_type, sp2.inner.next_accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());

    // Build chain proof and verify
    let chain = st.into_proof();
    let ok = neo::verify_nivc_chain(&program, &params, &chain, &y0)?;
    println!("verify_nivc_chain: {} (steps={})", ok, chain.steps.len());
    anyhow::ensure!(ok, "NIVC verification failed");

    Ok(())
}
