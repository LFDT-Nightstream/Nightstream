//! Track A: sparse-in-time oracle tests.
//!
//! Dense-time Route A oracles were removed from production; these tests validate sparse-time
//! claims against naive dense summations (small domains only) and smoke-test sumcheck runs.

use neo_math::K;
use neo_memory::bit_ops::eq_bit_affine;
use neo_memory::mle::{chi_at_index, lt_eval};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::twist_oracle::{
    AddressLookupOracle, IndexAdapterOracleSparseTime, LazyBitnessOracleSparseTime, ShoutValueOracleSparse,
    TwistReadCheckOracleSparseTime, TwistTotalIncOracleSparseTime, TwistValEvalOracleSparseTime,
    TwistWriteCheckOracleSparseTime,
};
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

fn mask_to_bits(mask: usize, ell: usize) -> Vec<K> {
    (0..ell)
        .map(|i| if ((mask >> i) & 1) == 1 { K::ONE } else { K::ZERO })
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
    let (rounds, chals_p) = run_sumcheck_prover(&mut tr_p, &mut oracle, initial_sum).expect("prover");

    let mut tr_v = Poseidon2Transcript::new(label);
    let (chals_v, _final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, degree_bound, initial_sum, &rounds);
    assert!(ok);
    assert_eq!(chals_p, chals_v);
}

#[test]
fn shout_value_oracle_sparse_claim_matches_naive_and_runs_sumcheck() {
    let r_cycle = vec![k(5), k(7), k(11)];
    let pow2_cycle = 1usize << r_cycle.len();

    let has_lookup = vec![K::ONE, K::ZERO, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ONE];
    let val = (0..pow2_cycle).map(|t| k(100 + t as u64)).collect::<Vec<_>>();

    let (oracle, claim) = ShoutValueOracleSparse::new(&r_cycle, dense_to_sparse(&has_lookup), dense_to_sparse(&val));

    let mut naive = K::ZERO;
    for t in 0..pow2_cycle {
        naive += chi_at_index(&r_cycle, t) * has_lookup[t] * val[t];
    }
    assert_eq!(claim, naive);

    assert_sumcheck_ok(b"track_a/shout/value/v1", oracle.degree_bound(), claim, oracle);
}

#[test]
fn shout_adapter_oracle_sparse_claim_matches_naive_and_runs_sumcheck() {
    let r_cycle = vec![k(5), k(7), k(11)];
    let r_addr = vec![k(13), k(17)];
    let pow2_cycle = 1usize << r_cycle.len();
    let ell_addr = r_addr.len();

    let has_lookup = vec![K::ONE, K::ZERO, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ONE];
    let addrs = (0..pow2_cycle).map(|t| (1 + 3 * t) % (1usize << ell_addr)).collect::<Vec<_>>();
    let addr_bits = bits_from_addrs(&addrs, ell_addr);

    let (oracle, claim) = IndexAdapterOracleSparseTime::new_with_gate(
        &r_cycle,
        dense_to_sparse(&has_lookup),
        cols_to_sparse(&addr_bits),
        &r_addr,
    );

    let mut naive = K::ZERO;
    for t in 0..pow2_cycle {
        let mut eq_addr = K::ONE;
        for b in 0..ell_addr {
            eq_addr *= eq_bit_affine(addr_bits[b][t], r_addr[b]);
        }
        naive += chi_at_index(&r_cycle, t) * has_lookup[t] * eq_addr;
    }
    assert_eq!(claim, naive);

    assert_sumcheck_ok(b"track_a/shout/adapter/v1", oracle.degree_bound(), claim, oracle);
}

#[test]
fn lazy_bitness_oracle_sparse_binary_claim_zero_and_runs_sumcheck() {
    let r_cycle = vec![k(5), k(7), k(11)];
    let pow2_cycle = 1usize << r_cycle.len();
    let col = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();

    let oracle = LazyBitnessOracleSparseTime::new_with_cycle(&r_cycle, dense_to_sparse(&col));
    assert_sumcheck_ok(b"track_a/bitness/v1", oracle.degree_bound(), K::ZERO, oracle);
}

#[test]
fn twist_val_eval_and_total_inc_sparse_claims_match_naive_and_run_sumcheck() {
    let r_addr = vec![k(3), k(9)];
    let r_time = vec![k(2), k(4), k(6)];
    let pow2_cycle = 1usize << r_time.len();
    let ell_addr = r_addr.len();

    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ONE];
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { k(5 + t as u64) } else { K::ZERO })
        .collect::<Vec<_>>();
    let wa_addrs = (0..pow2_cycle).map(|t| (3 * t + 1) % (1usize << ell_addr)).collect::<Vec<_>>();
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let (val_oracle, val_claim) = TwistValEvalOracleSparseTime::new(
        cols_to_sparse(&wa_bits),
        dense_to_sparse(&has_write),
        dense_to_sparse(&inc_at_write_addr),
        &r_addr,
        &r_time,
    );
    let mut val_naive = K::ZERO;
    for t in 0..pow2_cycle {
        let mut eq_addr = K::ONE;
        for b in 0..ell_addr {
            eq_addr *= eq_bit_affine(wa_bits[b][t], r_addr[b]);
        }
        let lt = lt_eval(&mask_to_bits(t, r_time.len()), &r_time);
        val_naive += has_write[t] * inc_at_write_addr[t] * eq_addr * lt;
    }
    assert_eq!(val_claim, val_naive);
    assert_sumcheck_ok(b"track_a/twist/val_eval/v1", val_oracle.degree_bound(), val_claim, val_oracle);

    let (total_oracle, total_claim) = TwistTotalIncOracleSparseTime::new(
        cols_to_sparse(&wa_bits),
        dense_to_sparse(&has_write),
        dense_to_sparse(&inc_at_write_addr),
        &r_addr,
    );
    let mut total_naive = K::ZERO;
    for t in 0..pow2_cycle {
        let mut eq_addr = K::ONE;
        for b in 0..ell_addr {
            eq_addr *= eq_bit_affine(wa_bits[b][t], r_addr[b]);
        }
        total_naive += has_write[t] * inc_at_write_addr[t] * eq_addr;
    }
    assert_eq!(total_claim, total_naive);
    assert_sumcheck_ok(
        b"track_a/twist/total_inc/v1",
        total_oracle.degree_bound(),
        total_claim,
        total_oracle,
    );
}

#[test]
fn shout_addr_lookup_oracle_sparse_claim_matches_naive_and_runs_sumcheck() {
    let r_cycle = vec![k(5), k(7), k(11)];
    let pow2_cycle = 1usize << r_cycle.len();
    let ell_addr = 2usize;
    let pow2_addr = 1usize << ell_addr;

    let has_lookup = vec![K::ONE, K::ZERO, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ONE];
    let addrs = (0..pow2_cycle).map(|t| (1 + 3 * t) % pow2_addr).collect::<Vec<_>>();
    let addr_bits = bits_from_addrs(&addrs, ell_addr);
    let table = vec![k(2), k(3), k(5), k(7)];

    let (oracle, claim) = AddressLookupOracle::new(
        &cols_to_sparse(&addr_bits),
        &dense_to_sparse(&has_lookup),
        &table,
        &r_cycle,
        ell_addr,
    );

    let mut weight_table = vec![K::ZERO; pow2_addr];
    for t in 0..pow2_cycle {
        if has_lookup[t] == K::ZERO {
            continue;
        }
        let mut addr_t = 0usize;
        for b in 0..ell_addr {
            if addr_bits[b][t] == K::ONE {
                addr_t |= 1usize << b;
            }
        }
        weight_table[addr_t] += chi_at_index(&r_cycle, t) * has_lookup[t];
    }

    let mut naive = K::ZERO;
    for a in 0..pow2_addr.min(table.len()) {
        naive += table[a] * weight_table[a];
    }
    assert_eq!(claim, naive);

    assert_sumcheck_ok(b"track_a/shout/addr_lookup/v1", oracle.degree_bound(), claim, oracle);
}

#[test]
fn twist_time_read_write_checks_sparse_have_zero_claim_on_consistent_trace() {
    let r_cycle = vec![k(5), k(7), k(11)];
    let pow2_cycle = 1usize << r_cycle.len();

    let addr = 1usize;
    let r_addr = vec![
        if (addr & 1) == 1 { K::ONE } else { K::ZERO },
        if (addr & 2) == 2 { K::ONE } else { K::ZERO },
    ];
    let init_at_r_addr = k(23);

    // Constant read/write address bits = `addr` at all times.
    let wa_bits = vec![vec![r_addr[0]; pow2_cycle], vec![r_addr[1]; pow2_cycle]];
    let ra_bits = wa_bits.clone();

    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ONE];
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| if has_write[t] == K::ONE { k(5 + t as u64) } else { K::ZERO })
        .collect::<Vec<_>>();

    // Simulate a single-address memory so both checks should have 0 claim.
    let mut cur = init_at_r_addr;
    let mut val_pre = vec![K::ZERO; pow2_cycle];
    let mut wv = vec![K::ZERO; pow2_cycle];
    for t in 0..pow2_cycle {
        val_pre[t] = cur;
        if has_write[t] == K::ONE {
            wv[t] = cur + inc_at_write_addr[t];
            cur += inc_at_write_addr[t];
        }
    }

    let has_read = vec![K::ZERO, K::ONE, K::ONE, K::ZERO, K::ONE, K::ZERO, K::ZERO, K::ONE];
    let rv = (0..pow2_cycle)
        .map(|t| if has_read[t] == K::ONE { val_pre[t] } else { K::ZERO })
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
        b"track_a/twist/read_check/v1",
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
        b"track_a/twist/write_check/v1",
        write_check.degree_bound(),
        K::ZERO,
        write_check,
    );
}
