//! Tests for the Output Sumcheck module.

use neo_math::{K, KExtensions};
use neo_memory::mle::build_chi_table;
use neo_memory::output_check::{
    generate_output_sumcheck_proof, verify_output_sumcheck, OutputCheckError, OutputSumcheckParams, OutputSumcheckProof,
    OutputSumcheckProver, ProgramIO,
};
use neo_reductions::sumcheck::RoundOracle;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

// ============================================================================
// Helpers
// ============================================================================

fn make_test_params(num_bits: usize, claims: &[(u64, u64)]) -> OutputSumcheckParams<F> {
    let r_addr: Vec<K> = (0..num_bits).map(|i| K::from_u64(100 + (i * 37) as u64)).collect();
    let mut program_io = ProgramIO::new();
    for (addr, value) in claims {
        program_io = program_io.with_output(*addr, F::from_u64(*value));
    }
    OutputSumcheckParams::new_for_testing(num_bits, r_addr, program_io).unwrap()
}

fn build_final_state(num_bits: usize, values: &[(u64, u64)]) -> Vec<F> {
    let size = 1usize << num_bits;
    let mut state = vec![F::ZERO; size];
    for (addr, value) in values {
        if (*addr as usize) < size {
            state[*addr as usize] = F::from_u64(*value);
        }
    }
    state
}

fn interpolate(xs: &[K], ys: &[K]) -> Vec<K> {
    use neo_math::KExtensions;
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];
    for i in 0..n {
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE;
        let mut cur_deg = 0;
        let mut denom = K::ONE;
        for j in 0..n {
            if i == j { continue; }
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                next[d + 1] += numer[d];
                next[d] -= xs[j] * numer[d];
            }
            numer = next;
            cur_deg += 1;
            denom *= xs[i] - xs[j];
        }
        let scale = ys[i] * denom.inv();
        for d in 0..n { coeffs[d] += scale * numer[d]; }
    }
    coeffs
}

#[allow(dead_code)]
fn eval_poly(coeffs: &[K], x: K) -> K {
    coeffs.iter().rev().fold(K::ZERO, |acc, &c| acc * x + c)
}

// ============================================================================
// Basic Correctness Tests
// ============================================================================

#[test]
fn test_matching_outputs_claim_is_zero() {
    let claims = [(0, 100), (1, 200), (2, 300)];
    let params = make_test_params(4, &claims);
    let final_state = build_final_state(4, &claims);
    let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    assert_eq!(prover.compute_claim(), K::ZERO);
}

#[test]
fn test_mismatched_outputs_claim_is_nonzero() {
    let params = make_test_params(4, &[(0, 100), (1, 200)]);
    let final_state = build_final_state(4, &[(0, 999), (1, 200)]);
    let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    assert_ne!(prover.compute_claim(), K::ZERO);
}

// ============================================================================
// Soundness Tests
// ============================================================================

#[test]
fn test_verifier_enforces_zero_claim() {
    let num_bits = 4;
    let params = make_test_params(num_bits, &[(0, 100)]);
    let lying_final_state = build_final_state(num_bits, &[(0, 999)]);

    let mut prover = OutputSumcheckProver::new(params.clone(), &lying_final_state).unwrap();
    assert_ne!(prover.compute_claim(), K::ZERO);

    // Generate proof
    let eval_points: Vec<K> = (0..=prover.degree_bound()).map(|i| K::from_u64(i as u64)).collect();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for round in 0..num_bits {
        let coeffs = interpolate(&eval_points, &prover.evals_at(&eval_points));
        round_polys.push(coeffs);
        let r = K::from_u64((round * 13 + 7) as u64);
        challenges.push(r);
        prover.fold(r);
    }

    let proof = OutputSumcheckProof { round_polys };

    // Compute val_final at challenge point
    let chi = build_chi_table(&challenges);
    let val_final_at_r: K = lying_final_state.iter().enumerate()
        .map(|(i, &v)| K::from(v) * chi[i]).sum();

    // Verify using transcript-based verification
    let mut tr = Poseidon2Transcript::new(b"test");
    // Re-derive params from transcript
    let program_io = ProgramIO::new().with_output(0, F::from_u64(100));
    let result = verify_output_sumcheck(&mut tr, num_bits, program_io, val_final_at_r, &proof);
    
    // Should fail because initial claim != 0
    assert!(result.is_err());
}

#[test]
fn test_verifier_rejects_wrong_degree() {
    let num_bits = 4;
    let params = make_test_params(num_bits, &[(0, 100)]);
    let final_state = build_final_state(num_bits, &[(0, 100)]);

    let mut prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    let eval_points: Vec<K> = (0..=prover.degree_bound()).map(|i| K::from_u64(i as u64)).collect();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for round in 0..num_bits {
        let coeffs = interpolate(&eval_points, &prover.evals_at(&eval_points));
        round_polys.push(coeffs);
        let r = K::from_u64((round * 17 + 3) as u64);
        challenges.push(r);
        prover.fold(r);
    }

    // Corrupt: add extra coefficient
    round_polys[0].push(K::from_u64(12345));
    let bad_proof = OutputSumcheckProof { round_polys };

    let chi = build_chi_table(&challenges);
    let val_final_at_r: K = final_state.iter().enumerate()
        .map(|(i, &v)| K::from(v) * chi[i]).sum();

    let mut tr = Poseidon2Transcript::new(b"test");
    let program_io = ProgramIO::new().with_output(0, F::from_u64(100));
    let result = verify_output_sumcheck(&mut tr, num_bits, program_io, val_final_at_r, &bad_proof);

    assert!(matches!(result, Err(OutputCheckError::WrongDegree { round: 0, expected: 4, got: 5 })));
}

// ============================================================================
// Mask Semantics Tests
// ============================================================================

#[test]
fn test_mask_only_constrains_claimed_addresses() {
    let params = make_test_params(4, &[(0, 100)]);
    let mut final_state = build_final_state(4, &[(0, 100)]);
    final_state[1] = F::from_u64(12345); // Unclaimed
    final_state[5] = F::from_u64(99999);
    let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    assert_eq!(prover.compute_claim(), K::ZERO);
}

#[test]
fn test_no_silent_zero_enforcement_on_unclaimed() {
    let params = make_test_params(4, &[(10, 42)]);
    let mut final_state = vec![F::ZERO; 16];
    final_state[10] = F::from_u64(42);
    final_state[11] = F::from_u64(123);
    let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    assert_eq!(prover.compute_claim(), K::ZERO);
}

// ============================================================================
// Validation Tests
// ============================================================================

#[test]
fn test_duplicate_address_detection() {
    let program_io: ProgramIO<F> = ProgramIO::new().with_output(10, F::from_u64(100));
    let result = program_io.try_with_claim(10, F::from_u64(200));
    assert!(matches!(result, Err(OutputCheckError::DuplicateAddress { addr: 10 })));
}

#[test]
fn test_address_out_of_domain_detection() {
    let program_io: ProgramIO<F> = ProgramIO::new()
        .with_output(15, F::from_u64(100))
        .with_output(16, F::from_u64(200));
    assert!(matches!(program_io.validate(4), Err(OutputCheckError::AddressOutOfDomain { addr: 16, .. })));
}

#[test]
fn test_num_bits_too_large() {
    let program_io: ProgramIO<F> = ProgramIO::new();
    assert!(matches!(program_io.validate(64), Err(OutputCheckError::NumBitsTooLarge { .. })));
}

#[test]
fn test_dimension_mismatch_detection() {
    let params = make_test_params(4, &[(0, 100)]);
    let wrong_size_state = vec![F::ZERO; 8];
    assert!(matches!(
        OutputSumcheckProver::new(params, &wrong_size_state),
        Err(OutputCheckError::DimensionMismatch { expected: 16, got: 8 })
    ));
}

// ============================================================================
// RoundOracle Robustness Tests
// ============================================================================

#[test]
fn test_fold_handles_completion() {
    let params = make_test_params(2, &[(0, 42)]);
    let final_state = build_final_state(2, &[(0, 42)]);
    let mut prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    for _ in 0..4 { prover.fold(K::from_u64(7)); } // Extra folds are no-ops
    assert_eq!(prover.num_rounds(), 0);
}

#[test]
fn test_evals_at_handles_completion() {
    let params = make_test_params(2, &[(0, 42)]);
    let final_state = build_final_state(2, &[(0, 42)]);
    let mut prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    for _ in 0..2 { prover.fold(K::from_u64(7)); }
    let evals = prover.evals_at(&[K::ZERO, K::ONE, K::from_u64(5)]);
    assert_eq!(evals[0], evals[1]);
    assert_eq!(evals[1], evals[2]);
}

// ============================================================================
// Full Proof Generation and Verification
// ============================================================================

#[test]
fn test_full_proof_generation_and_verification() {
    let num_bits = 4;
    let claims = [(0, 10), (3, 20), (7, 30)];
    let final_state = build_final_state(num_bits, &claims);

    let mut program_io = ProgramIO::new();
    for (addr, value) in &claims {
        program_io = program_io.with_output(*addr, F::from_u64(*value));
    }

    // Generate proof
    let mut tr_prover = Poseidon2Transcript::new(b"test");
    let proof = generate_output_sumcheck_proof(&mut tr_prover, num_bits, program_io.clone(), &final_state).unwrap();

    // Replay to derive challenges for val_final computation
    let mut tr_replay = Poseidon2Transcript::new(b"test");
    let _params = OutputSumcheckParams::sample_from_transcript(&mut tr_replay, num_bits, program_io.clone()).unwrap();
    let mut challenges = Vec::new();
    for coeffs in &proof.round_polys {
        for &c in coeffs { tr_replay.append_fields(b"output_check/round_coeff", &c.as_coeffs()); }
        challenges.push(neo_math::from_complex(
            tr_replay.challenge_field(b"output_check/chal/re"),
            tr_replay.challenge_field(b"output_check/chal/im"),
        ));
    }

    let chi = build_chi_table(&challenges);
    let val_final_at_r: K = final_state.iter().enumerate().map(|(i, &v)| K::from(v) * chi[i]).sum();

    let mut tr_verify = Poseidon2Transcript::new(b"test");
    let result = verify_output_sumcheck(&mut tr_verify, num_bits, program_io, val_final_at_r, &proof);
    assert!(result.is_ok(), "Verification should succeed: {:?}", result);
}

#[test]
fn test_transcript_based_proof_generation() {
    let num_bits = 4;
    let program_io: ProgramIO<F> = ProgramIO::new()
        .with_output(0, F::from_u64(100))
        .with_output(1, F::from_u64(200));
    let final_state = build_final_state(num_bits, &[(0, 100), (1, 200)]);

    let mut tr = Poseidon2Transcript::new(b"output_check_test");
    let proof = generate_output_sumcheck_proof(&mut tr, num_bits, program_io, &final_state).unwrap();

    assert_eq!(proof.round_polys.len(), num_bits);
}

// ============================================================================
// Multiple Fraud Attempts
// ============================================================================

#[test]
fn test_multiple_fraudulent_values_all_detected() {
    let num_bits = 6;
    let honest_output = 50u64;
    let final_state = build_final_state(num_bits, &[(10, honest_output)]);

    for &fraud in &[0u64, 1, 49, 51, 100, 999, u64::MAX - 1] {
        let params = make_test_params(num_bits, &[(10, fraud)]);
        let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
        assert_ne!(prover.compute_claim(), K::ZERO, "Not detected: {}", fraud);
    }

    let honest_params = make_test_params(num_bits, &[(10, honest_output)]);
    let honest_prover = OutputSumcheckProver::new(honest_params, &final_state).unwrap();
    assert_eq!(honest_prover.compute_claim(), K::ZERO);
}

// ============================================================================
// Bit-Order Consistency Tests
// ============================================================================

#[test]
fn test_bit_order_consistency_small() {
    for num_bits in 2..=4 {
        let params = make_test_params(num_bits, &[(0, 42)]);
        let final_state = build_final_state(num_bits, &[(0, 42)]);
        let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
        assert_eq!(prover.compute_claim(), K::ZERO, "Bit order mismatch at num_bits={}", num_bits);
    }
}

#[test]
fn test_single_bit_flip_detection() {
    let params = make_test_params(4, &[(0, 100)]);
    let final_state = build_final_state(4, &[(0, 101)]);
    let prover = OutputSumcheckProver::new(params, &final_state).unwrap();
    assert_ne!(prover.compute_claim(), K::ZERO);
}

// ============================================================================
// Accept/Reject Regression Tests (LSB-first bit-order validation)
// ============================================================================

/// χ_r(k) = Π_i (r[i] if k_i else 1-r[i])
fn chi_at_point(r: &[K], k: u64, num_bits: usize) -> K {
    (0..num_bits).fold(K::ONE, |acc, i| {
        let bit = ((k >> i) & 1) == 1;
        acc * if bit { r[i] } else { K::ONE - r[i] }
    })
}

fn eval_table_at_point(vals: &[F], r: &[K]) -> K {
    let num_bits = r.len();
    vals.iter()
        .enumerate()
        .map(|(addr, &v)| {
            let w = chi_at_point(r, addr as u64, num_bits);
            let vk: K = v.into();
            vk * w
        })
        .sum()
}

#[test]
fn output_sumcheck_accepts_correct_claims() {
    use neo_memory::output_check::generate_output_sumcheck_proof_and_challenges;
    
    let num_bits = 3usize;
    let k = 1usize << num_bits;

    // Deterministic final state
    let final_memory_state: Vec<F> = (0..k)
        .map(|i| F::from_u64((i as u64).wrapping_mul(17).wrapping_add(3)))
        .collect();

    // Claims match final state at those addresses
    let program_io = ProgramIO::new()
        .with_output(1, final_memory_state[1])
        .with_output(6, final_memory_state[6]);

    // Prove
    let mut tr_p = Poseidon2Transcript::new(b"ob-test");
    let (proof, r_prime) = generate_output_sumcheck_proof_and_challenges(
        &mut tr_p,
        num_bits,
        program_io.clone(),
        &final_memory_state,
    )
    .unwrap();

    // Verifier needs Val_final(r_prime)
    let val_final_at_r_prime = eval_table_at_point(&final_memory_state, &r_prime);

    // Verify
    let mut tr_v = Poseidon2Transcript::new(b"ob-test");
    verify_output_sumcheck(
        &mut tr_v,
        num_bits,
        program_io,
        val_final_at_r_prime,
        &proof,
    )
    .unwrap();
}

#[test]
fn output_sumcheck_rejects_wrong_claims() {
    use neo_memory::output_check::generate_output_sumcheck_proof_and_challenges;
    
    let num_bits = 3usize;
    let k = 1usize << num_bits;

    let final_memory_state: Vec<F> = (0..k)
        .map(|i| F::from_u64((i as u64).wrapping_mul(17).wrapping_add(3)))
        .collect();

    // Make one claim wrong
    let wrong = final_memory_state[6] + F::ONE;
    let program_io = ProgramIO::new()
        .with_output(1, final_memory_state[1])
        .with_output(6, wrong);

    let mut tr_p = Poseidon2Transcript::new(b"ob-test");
    let (proof, r_prime) = generate_output_sumcheck_proof_and_challenges(
        &mut tr_p,
        num_bits,
        program_io.clone(),
        &final_memory_state,
    )
    .unwrap();

    let val_final_at_r_prime = eval_table_at_point(&final_memory_state, &r_prime);

    let mut tr_v = Poseidon2Transcript::new(b"ob-test");
    let err = verify_output_sumcheck(
        &mut tr_v,
        num_bits,
        program_io,
        val_final_at_r_prime,
        &proof,
    )
    .err()
    .expect("must fail");

    // Any failure is acceptable, but it must fail.
    eprintln!("expected failure: {err}");
}
