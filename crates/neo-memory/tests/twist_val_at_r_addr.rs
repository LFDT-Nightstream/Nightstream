//! Twist time-lane read/write checks: focused sparse-time tests.

use neo_math::K;
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::twist_oracle::{TwistReadCheckOracleSparseTime, TwistWriteCheckOracleSparseTime};
use neo_reductions::sumcheck::{run_sumcheck_prover, verify_sumcheck_rounds, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn k(u: u64) -> K {
    K::from(Goldilocks::from_u64(u))
}

fn dense_to_sparse(v: &[K]) -> SparseIdxVec<K> {
    SparseIdxVec::from_entries(
        v.len(),
        v.iter()
            .enumerate()
            .filter_map(|(i, &x)| (x != K::ZERO).then_some((i, x)))
            .collect(),
    )
}

fn cols_to_sparse(cols: &[Vec<K>]) -> Vec<SparseIdxVec<K>> {
    cols.iter().map(|c| dense_to_sparse(c)).collect()
}

fn addr_to_bits(addr: usize, ell: usize) -> Vec<K> {
    (0..ell)
        .map(|b| if ((addr >> b) & 1) == 1 { K::ONE } else { K::ZERO })
        .collect()
}

fn bits_from_addrs(addrs: &[usize], ell_addr: usize) -> Vec<Vec<K>> {
    let mut cols = vec![vec![K::ZERO; addrs.len()]; ell_addr];
    for (t, &addr) in addrs.iter().enumerate() {
        for b in 0..ell_addr {
            if ((addr >> b) & 1) == 1 {
                cols[b][t] = K::ONE;
            }
        }
    }
    cols
}

fn assert_sumcheck_ok(label: &'static [u8], degree_bound: usize, initial_sum: K, mut oracle: impl RoundOracle) {
    let mut tr_p = Poseidon2Transcript::new(label);
    let (rounds, _chals) = run_sumcheck_prover(&mut tr_p, &mut oracle, initial_sum).expect("prover");

    let mut tr_v = Poseidon2Transcript::new(label);
    let (_chals_v, _final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, degree_bound, initial_sum, &rounds);
    assert!(ok);
}

#[test]
fn read_write_checks_have_zero_claim_with_distractor_writes() {
    let r_cycle = vec![k(2), k(3), k(5)];
    let pow2_cycle = 1usize << r_cycle.len();

    let ell_addr = 2usize;
    let target_addr = 2usize;
    let r_addr = addr_to_bits(target_addr, ell_addr);
    let init_at_r_addr = k(7);

    // Write addresses: include distractors not equal to target_addr.
    let wa_addrs = vec![0usize, 2, 1, 2, 3, 2, 0, 1];
    assert_eq!(wa_addrs.len(), pow2_cycle);
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    // Read addresses: always read target_addr when has_read=1.
    let ra_bits = vec![vec![r_addr[0]; pow2_cycle], vec![r_addr[1]; pow2_cycle]];

    let has_write = vec![K::ONE, K::ONE, K::ONE, K::ONE, K::ONE, K::ONE, K::ZERO, K::ZERO];
    let mut inc_at_write_addr = vec![K::ZERO; pow2_cycle];
    let mut wv = vec![K::ZERO; pow2_cycle];

    // Ensure only writes to `target_addr` affect the expected value used by reads/writes.
    let mut cur = init_at_r_addr;
    let mut val_pre_target = vec![K::ZERO; pow2_cycle];
    for t in 0..pow2_cycle {
        val_pre_target[t] = cur;
        if has_write[t] == K::ONE {
            inc_at_write_addr[t] = k(10 + t as u64);
            if wa_addrs[t] == target_addr {
                wv[t] = cur + inc_at_write_addr[t];
                cur += inc_at_write_addr[t];
            } else {
                // Distractor writes: can be arbitrary (they are gated out by eq(wa_bits, r_addr)).
                wv[t] = k(200 + t as u64);
            }
        }
    }

    let has_read = vec![K::ONE, K::ONE, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ONE];
    let rv = (0..pow2_cycle)
        .map(|t| {
            if has_read[t] == K::ONE {
                val_pre_target[t]
            } else {
                K::ZERO
            }
        })
        .collect::<Vec<_>>();

    let read_check = TwistReadCheckOracleSparseTime::new(
        &r_cycle,
        dense_to_sparse(&has_read),
        dense_to_sparse(&rv),
        cols_to_sparse(&ra_bits),
        dense_to_sparse(&has_write),
        dense_to_sparse(&inc_at_write_addr),
        cols_to_sparse(&wa_bits),
        &r_addr,
        init_at_r_addr,
    );
    assert_sumcheck_ok(
        b"twist/read_check/gating",
        read_check.degree_bound(),
        K::ZERO,
        read_check,
    );

    let write_check = TwistWriteCheckOracleSparseTime::new(
        &r_cycle,
        dense_to_sparse(&has_write),
        dense_to_sparse(&wv),
        dense_to_sparse(&inc_at_write_addr),
        cols_to_sparse(&wa_bits),
        &r_addr,
        init_at_r_addr,
    );
    assert_sumcheck_ok(
        b"twist/write_check/gating",
        write_check.degree_bound(),
        K::ZERO,
        write_check,
    );
}

#[test]
fn corrupting_rv_breaks_read_check_invariant() {
    let r_cycle = vec![k(2), k(3), k(5)];
    let pow2_cycle = 1usize << r_cycle.len();

    let ell_addr = 2usize;
    let addr = 1usize;
    let r_addr = addr_to_bits(addr, ell_addr);
    let init_at_r_addr = k(9);

    // Constant write addr == addr, so eq(wa_bits, r_addr)=1.
    let wa_bits = vec![vec![r_addr[0]; pow2_cycle], vec![r_addr[1]; pow2_cycle]];
    let ra_bits = wa_bits.clone();

    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ZERO, K::ZERO, K::ONE, K::ZERO, K::ZERO];
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| {
            if has_write[t] == K::ONE {
                k(4 + t as u64)
            } else {
                K::ZERO
            }
        })
        .collect::<Vec<_>>();

    // Correct reads: rv[t] = pre-write value at addr.
    let mut cur = init_at_r_addr;
    let mut val_pre = vec![K::ZERO; pow2_cycle];
    for t in 0..pow2_cycle {
        val_pre[t] = cur;
        if has_write[t] == K::ONE {
            cur += inc_at_write_addr[t];
        }
    }
    let has_read = vec![K::ONE; pow2_cycle];
    let mut rv = val_pre.clone();
    rv[1] += K::ONE; // corrupt

    let mut oracle = TwistReadCheckOracleSparseTime::new(
        &r_cycle,
        dense_to_sparse(&has_read),
        dense_to_sparse(&rv),
        cols_to_sparse(&ra_bits),
        dense_to_sparse(&has_write),
        dense_to_sparse(&inc_at_write_addr),
        cols_to_sparse(&wa_bits),
        &r_addr,
        init_at_r_addr,
    );

    let mut tr = Poseidon2Transcript::new(b"twist/read_check/corrupt_rv");
    let err = run_sumcheck_prover(&mut tr, &mut oracle, K::ZERO).expect_err("must fail invariant");
    let _ = err;
}
