//! Poseidon2-in-circuit gadget byte-for-byte parity test against
//! `neo_ccs::crypto::poseidon2_goldilocks::PERM`.
//!
//! If this passes, the in-circuit permutation produces identical output to the
//! native permutation for every input.

use neo_ccs::crypto::poseidon2_goldilocks::{permute_state, poseidon2_hash};
use neo_fold_clean::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, enforce_poseidon2_permutation};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const WIDTH: usize = 8;

fn alloc_input(b: &mut R1csBuilder, vals: &[F; WIDTH]) -> [Var; WIDTH] {
    let mut out = [Var::ONE; WIDTH];
    for (slot, &v) in out.iter_mut().zip(vals.iter()) {
        *slot = b.alloc(v);
    }
    out
}

fn run_gadget_and_extract_output(input: [F; WIDTH]) -> [F; WIDTH] {
    let mut b = R1csBuilder::new();
    let input_vars = alloc_input(&mut b, &input);
    let output_vars = enforce_poseidon2_permutation(&mut b, &input_vars);
    assert!(
        b.is_satisfied(),
        "gadget unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let mut out = [F::ZERO; WIDTH];
    for (slot, var) in out.iter_mut().zip(output_vars.iter()) {
        *slot = b.witness()[var.col()];
    }
    out
}

#[test]
fn poseidon2_gadget_matches_native_on_zero_input() {
    let input = [F::ZERO; WIDTH];
    let expected = permute_state(input);
    let got = run_gadget_and_extract_output(input);
    assert_eq!(got, expected, "gadget output diverges from native on zero input");
}

#[test]
fn poseidon2_gadget_matches_native_on_basis_vectors() {
    for i in 0..WIDTH {
        let mut input = [F::ZERO; WIDTH];
        input[i] = F::ONE;
        let expected = permute_state(input);
        let got = run_gadget_and_extract_output(input);
        assert_eq!(got, expected, "gadget output diverges on basis vector e_{i}");
    }
}

#[test]
fn poseidon2_gadget_matches_native_on_random_inputs() {
    let inputs: [[u64; WIDTH]; 5] = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0, 1, 2, 3, 4],
        [
            u64::MAX - 7,
            u64::MAX - 6,
            u64::MAX - 5,
            u64::MAX - 4,
            u64::MAX - 3,
            u64::MAX - 2,
            u64::MAX - 1,
            u64::MAX,
        ],
        // Note: Goldilocks values are < q ≈ 2^64, so we mask high bits.
        [0xFFFF_FFFE_FFFF_FFFF, 0, 0, 0, 0, 0, 0, 0],
        [42; WIDTH],
    ];
    for (case_idx, raw) in inputs.iter().enumerate() {
        let input: [F; WIDTH] = std::array::from_fn(|i| {
            // Clamp into the Goldilocks range: q = 2^64 - 2^32 + 1.
            let q: u64 = 0xFFFF_FFFF_0000_0001;
            F::from_u64(raw[i] % q)
        });
        let expected = permute_state(input);
        let got = run_gadget_and_extract_output(input);
        assert_eq!(got, expected, "gadget diverges from native on case {case_idx}");
    }
}

#[test]
fn poseidon2_gadget_rejects_tampered_output() {
    let input = [F::from_u64(7); WIDTH];

    let mut b = R1csBuilder::new();
    let input_vars = alloc_input(&mut b, &input);
    let output_vars = enforce_poseidon2_permutation(&mut b, &input_vars);
    assert!(b.is_satisfied(), "baseline");

    let target = output_vars[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "gadget accepted a tampered output[0]");
}

#[test]
fn poseidon2_gadget_rejects_tampered_input() {
    let input = [F::from_u64(11); WIDTH];

    let mut b = R1csBuilder::new();
    let input_vars = alloc_input(&mut b, &input);
    let _output_vars = enforce_poseidon2_permutation(&mut b, &input_vars);
    assert!(b.is_satisfied(), "baseline");

    // Tamper the first input lane — every downstream S-box and round equality
    // depends on it, so this must reject.
    let target = input_vars[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "gadget accepted a tampered input[0]");
}

// ── Variable-length sponge hash parity ──────────────────────────────────

fn alloc_var_vec(b: &mut R1csBuilder, vals: &[F]) -> Vec<Var> {
    vals.iter().map(|&v| b.alloc(v)).collect()
}

fn run_hash_and_extract(input: &[F]) -> [F; 4] {
    let mut b = R1csBuilder::new();
    let input_vars = alloc_var_vec(&mut b, input);
    let out_vars = enforce_poseidon2_hash(&mut b, &input_vars);
    assert!(
        b.is_satisfied(),
        "hash gadget unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let mut out = [F::ZERO; 4];
    for (slot, var) in out.iter_mut().zip(out_vars.iter()) {
        *slot = b.witness()[var.col()];
    }
    out
}

#[test]
fn poseidon2_hash_gadget_matches_native_on_empty_input() {
    let input: Vec<F> = vec![];
    let expected = poseidon2_hash(&input);
    let got = run_hash_and_extract(&input);
    assert_eq!(got, expected, "empty-input hash diverges");
}

#[test]
fn poseidon2_hash_gadget_matches_native_on_short_inputs() {
    for len in 1..=8 {
        let input: Vec<F> = (0..len).map(|i| F::from_u64(i as u64 + 1)).collect();
        let expected = poseidon2_hash(&input);
        let got = run_hash_and_extract(&input);
        assert_eq!(got, expected, "hash diverges at len {len}");
    }
}

#[test]
fn poseidon2_hash_gadget_matches_native_on_multiple_chunk_inputs() {
    // RATE = 4. Test lengths that cross multiple chunks.
    for &len in &[9, 11, 16, 17, 32] {
        let input: Vec<F> = (0..len).map(|i| F::from_u64((i * 31 + 7) as u64)).collect();
        let expected = poseidon2_hash(&input);
        let got = run_hash_and_extract(&input);
        assert_eq!(got, expected, "hash diverges at len {len}");
    }
}

#[test]
fn poseidon2_hash_gadget_rejects_tampered_input() {
    let input: Vec<F> = (0..7).map(|i| F::from_u64(i + 100)).collect();

    let mut b = R1csBuilder::new();
    let input_vars = alloc_var_vec(&mut b, &input);
    let _out = enforce_poseidon2_hash(&mut b, &input_vars);
    assert!(b.is_satisfied(), "baseline");

    let target = input_vars[3].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "hash gadget accepted a tampered input");
}
