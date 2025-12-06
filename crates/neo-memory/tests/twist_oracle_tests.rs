//! Tests for Twist oracles with index-bit addressing.

use neo_math::K;
use neo_memory::mle::mle_eval;
use neo_memory::twist_oracle::{
    build_eq_table, compute_eq_from_bits, val_tables_from_inc, TwistReadCheckOracle, TwistValEvalOracle,
    TwistWriteCheckOracle,
};
use neo_reductions::sumcheck::{run_sumcheck_prover, verify_sumcheck_rounds};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn k(u: u64) -> K {
    K::from(Goldilocks::from_u64(u))
}

#[test]
fn val_eval_oracle_matches_identity() {
    // k=2, steps=4 (ell_cycle=2)
    let inc = vec![
        Goldilocks::ONE,
        Goldilocks::from_u64(2),
        Goldilocks::from_u64(3),
        Goldilocks::from_u64(4),
        Goldilocks::from_u64(5),
        Goldilocks::from_u64(6),
        Goldilocks::from_u64(7),
        Goldilocks::from_u64(8),
    ];
    let r_addr = vec![k(3)]; // ell_k = 1
    let r_cycle = vec![k(5), k(7)]; // ell_t = 2

    let (inc_table, lt_table) = val_tables_from_inc(&inc, 2, 4, 2, 4, &r_addr, &r_cycle);
    let initial_sum: K = inc_table
        .iter()
        .zip(lt_table.iter())
        .map(|(a, b)| *a * *b)
        .sum();

    let mut oracle = TwistValEvalOracle::new(&inc, 2, 4, &r_addr, &r_cycle);
    let mut tr = Poseidon2Transcript::new(b"val/oracle/test");
    let (rounds, chals) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).expect("prover");

    let mut tr_v = Poseidon2Transcript::new(b"val/oracle/test");
    let (_chal_out, final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, initial_sum, &rounds);
    assert!(ok);

    // Final value should equal V(r_cycle)
    let r_point = chals;
    let val_at_r = {
        let inc_eval = mle_eval(&inc_table, &r_point);
        let lt_eval_r = neo_memory::mle::lt_eval(&r_point, &r_cycle);
        inc_eval * lt_eval_r
    };
    assert_eq!(final_sum, val_at_r);
}

#[test]
fn read_check_oracle_runs_sumcheck() {
    // Test with bit-decomposed addresses
    // 2 steps (pow2_cycle = 2), 1 address bit
    let r_cycle = vec![k(9)];
    let r_addr = vec![k(11)]; // Single address bit

    // Single bit column: step 0 has addr bit 0, step 1 has addr bit 1
    let ra_bits: Vec<Vec<K>> = vec![
        vec![K::ZERO, K::ONE], // bit 0 values for steps 0 and 1
    ];

    // Val at r_addr (precomputed)
    let val_at_r_addr = vec![k(10), k(30)]; // Val at steps 0, 1
    let rv = vec![k(10), k(30)]; // Read values (matching Val)
    let has_read = vec![K::ONE, K::ONE];

    // Expected sum: Σ eq(r_cycle, t) * has_read(t) * eq(bits_t, r_addr) * (Val - rv) = 0
    // Since rv == val_at_r_addr, diff is zero, so expected is 0
    let expected = K::ZERO;

    let mut oracle = TwistReadCheckOracle::new(&ra_bits, val_at_r_addr, rv, has_read, &r_cycle, &r_addr);

    let mut tr = Poseidon2Transcript::new(b"read/oracle/test");
    let (rounds, _chals) = run_sumcheck_prover(&mut tr, &mut oracle, expected).expect("prover");

    let mut tr_v = Poseidon2Transcript::new(b"read/oracle/test");
    let degree = 3 + ra_bits.len(); // eq_cycle, has_read, diff, bit_eq factors
    let (_c, _final, ok) = verify_sumcheck_rounds(&mut tr_v, degree, expected, &rounds);
    assert!(ok);
}

#[test]
fn write_check_oracle_runs_sumcheck() {
    // Test with bit-decomposed addresses
    let pow2_cycle = 2usize;
    let r_cycle = vec![k(13)];
    let r_addr = vec![k(11)]; // Single address bit

    // Write address bits: step 0 writes to addr 0, step 1 writes to addr 1
    let wa_bits: Vec<Vec<K>> = vec![
        vec![K::ZERO, K::ONE], // bit 0 values
    ];

    let val_at_write_addr = vec![k(5), k(6)]; // Val at steps 0, 1
    let wv = vec![k(7), k(9)]; // Write values
    let inc_at_write_addr = vec![K::ZERO, K::ZERO]; // No increments (test sumcheck mechanics)
    let has_write = vec![K::ONE, K::ONE];

    // Expected sum: Σ eq(r_cycle, t) * has_write(t) * eq(bits_t, r_addr) * (wv - Val - Inc)
    let eq_cycle = build_eq_table(&r_cycle);
    let eq_from_bits = compute_eq_from_bits(&wa_bits, &r_addr);
    let mut expected = K::ZERO;
    for t in 0..pow2_cycle {
        let delta = wv[t] - val_at_write_addr[t] - inc_at_write_addr[t];
        expected += eq_cycle[t] * has_write[t] * eq_from_bits[t] * delta;
    }

    let mut oracle = TwistWriteCheckOracle::new(
        &wa_bits,
        wv,
        val_at_write_addr,
        inc_at_write_addr,
        has_write,
        &r_cycle,
        &r_addr,
    );

    let mut tr = Poseidon2Transcript::new(b"write/oracle/test");
    let (rounds, _chals) = run_sumcheck_prover(&mut tr, &mut oracle, expected).expect("prover");

    let mut tr_v = Poseidon2Transcript::new(b"write/oracle/test");
    let degree = 3 + wa_bits.len();
    let (_c, _final, ok) = verify_sumcheck_rounds(&mut tr_v, degree, expected, &rounds);
    assert!(ok);
}
