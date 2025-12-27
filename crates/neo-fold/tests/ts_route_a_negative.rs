#![allow(non_snake_case)]

use neo_fold::memory_sidecar::sumcheck_ds::{
    run_batched_sumcheck_prover_ds, run_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds,
    verify_sumcheck_rounds_ds,
};
use neo_math::{F, K};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::twist_oracle::{
    table_mle_eval, AddressLookupOracle, IndexAdapterOracleSparseTime, LazyBitnessOracleSparseTime,
    ShoutValueOracleSparse, TwistReadCheckOracleSparseTime, TwistWriteCheckOracleSparseTime,
};
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

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

fn addr_bits_columns(addrs: &[usize], ell_addr: usize) -> Vec<Vec<K>> {
    let mut cols = vec![vec![K::ZERO; addrs.len()]; ell_addr];
    for (t, &addr) in addrs.iter().enumerate() {
        for b in 0..ell_addr {
            cols[b][t] = if ((addr >> b) & 1) == 1 { K::ONE } else { K::ZERO };
        }
    }
    cols
}

struct ShoutRun {
    addr_claim_sum: K,
    addr_final: K,
    r_addr: Vec<K>,
    value_claim: K,
    adapter_claim: K,
}

fn run_shout_route_a(table: &[K], addr_bits: &[Vec<K>], has_lookup: &[K], val: &[K], r_cycle: &[K]) -> ShoutRun {
    let ell_addr = addr_bits.len();
    let inst_idx = 0usize;

    let addr_bits_sparse = cols_to_sparse(addr_bits);
    let has_lookup_sparse = dense_to_sparse(has_lookup);

    let mut tr_p = Poseidon2Transcript::new(b"shout/route_a_negative");
    let mut tr_v = Poseidon2Transcript::new(b"shout/route_a_negative");

    let (r_addr, addr_claim_sum, addr_final) = {
        let (mut addr_oracle, addr_claim_sum) =
            AddressLookupOracle::new(&addr_bits_sparse, &has_lookup_sparse, table, r_cycle, ell_addr);

        let (addr_rounds, r_addr) = run_sumcheck_prover_ds(
            &mut tr_p,
            b"shout/addr_pre_time",
            inst_idx,
            &mut addr_oracle,
            addr_claim_sum,
        )
        .expect("prover addr sumcheck should succeed");

        let (r_addr_v, addr_final, ok) = verify_sumcheck_rounds_ds(
            &mut tr_v,
            b"shout/addr_pre_time",
            inst_idx,
            2,
            addr_claim_sum,
            &addr_rounds,
        );
        assert!(ok, "addr sumcheck must verify");
        assert_eq!(r_addr_v, r_addr, "addr r mismatch");
        (r_addr, addr_claim_sum, addr_final)
    };

    let (mut value_oracle, value_claim) =
        ShoutValueOracleSparse::new(r_cycle, has_lookup_sparse.clone(), dense_to_sparse(val));

    let (mut adapter_oracle, adapter_claim) = IndexAdapterOracleSparseTime::new_with_gate(
        r_cycle,
        has_lookup_sparse.clone(),
        addr_bits_sparse.clone(),
        &r_addr,
    );

    let mut bitness_oracles: Vec<LazyBitnessOracleSparseTime> = addr_bits_sparse
        .iter()
        .cloned()
        .map(|col| LazyBitnessOracleSparseTime::new_with_cycle(r_cycle, col))
        .collect();
    bitness_oracles.push(LazyBitnessOracleSparseTime::new_with_cycle(r_cycle, has_lookup_sparse));

    let mut claimed_sums: Vec<K> = Vec::with_capacity(2 + bitness_oracles.len());
    let mut round_polys: Vec<Vec<Vec<K>>> = Vec::with_capacity(2 + bitness_oracles.len());
    let mut degree_bounds: Vec<usize> = Vec::with_capacity(2 + bitness_oracles.len());
    let mut labels: Vec<&[u8]> = Vec::with_capacity(2 + bitness_oracles.len());

    let mut claims: Vec<BatchedClaim<'_>> = Vec::with_capacity(2 + bitness_oracles.len());

    claimed_sums.push(value_claim);
    degree_bounds.push(value_oracle.degree_bound());
    labels.push(b"shout/value");
    claims.push(BatchedClaim {
        oracle: &mut value_oracle,
        claimed_sum: value_claim,
        label: b"shout/value",
    });

    claimed_sums.push(adapter_claim);
    degree_bounds.push(adapter_oracle.degree_bound());
    labels.push(b"shout/adapter");
    claims.push(BatchedClaim {
        oracle: &mut adapter_oracle,
        claimed_sum: adapter_claim,
        label: b"shout/adapter",
    });

    for bit_oracle in bitness_oracles.iter_mut() {
        claimed_sums.push(K::ZERO);
        degree_bounds.push(bit_oracle.degree_bound());
        labels.push(b"shout/bitness");
        claims.push(BatchedClaim {
            oracle: bit_oracle,
            claimed_sum: K::ZERO,
            label: b"shout/bitness",
        });
    }

    let (_r_time, per_claim_results) =
        run_batched_sumcheck_prover_ds(&mut tr_p, b"shout/time_batch", inst_idx, claims.as_mut_slice())
            .expect("prover time batch should succeed");
    for res in per_claim_results.iter() {
        round_polys.push(res.round_polys.clone());
    }

    let (_r_time_v, _finals, ok) = verify_batched_sumcheck_rounds_ds(
        &mut tr_v,
        b"shout/time_batch",
        inst_idx,
        &round_polys,
        &claimed_sums,
        &labels,
        &degree_bounds,
    );
    assert!(ok, "time batch must verify");

    ShoutRun {
        addr_claim_sum,
        addr_final,
        r_addr,
        value_claim,
        adapter_claim,
    }
}

#[test]
fn route_a_shout_val_flip_fails() {
    let ell_time = 2usize;
    let ell_addr = 3usize;
    let pow2_time = 1usize << ell_time;
    let pow2_addr = 1usize << ell_addr;

    let r_cycle = vec![K::ZERO; ell_time]; // boolean -> selects t=0

    let table: Vec<K> = (0..pow2_addr)
        .map(|a| K::from(F::from_u64(a as u64)))
        .collect();

    let addrs = vec![3usize, 0, 0, 0];
    let addr_bits = addr_bits_columns(&addrs, ell_addr);
    let has_lookup = vec![K::ONE, K::ZERO, K::ZERO, K::ZERO];

    let mut val = vec![K::ZERO; pow2_time];
    val[0] = table[addrs[0]];

    let good = run_shout_route_a(&table, &addr_bits, &has_lookup, &val, &r_cycle);
    assert_eq!(good.value_claim, good.addr_claim_sum, "sanity: Shout must pass");
    let table_eval = table_mle_eval(&table, &good.r_addr);
    assert_eq!(
        good.addr_final,
        table_eval * good.adapter_claim,
        "sanity: addr terminal identity"
    );

    val[0] += K::ONE;
    let bad = run_shout_route_a(&table, &addr_bits, &has_lookup, &val, &r_cycle);
    assert_ne!(
        bad.value_claim, bad.addr_claim_sum,
        "Shout must fail when a looked-up val is flipped"
    );
}

#[test]
fn route_a_twist_inc_flip_fails() {
    let ell_time = 2usize;
    let ell_addr = 2usize;
    let pow2_time = 1usize << ell_time;

    let r_cycle = vec![K::ZERO; ell_time]; // boolean -> selects t=0
    let r_addr = vec![K::ONE, K::ZERO]; // boolean -> selects addr=1
    let init_at_r_addr = K::ZERO;

    let has_read = vec![K::ZERO; pow2_time];
    let rv = vec![K::ZERO; pow2_time];
    let ra_bits = vec![vec![r_addr[0]; pow2_time], vec![r_addr[1]; pow2_time]];

    let mut has_write = vec![K::ZERO; pow2_time];
    has_write[0] = K::ONE;
    let mut wv = vec![K::ZERO; pow2_time];
    wv[0] = K::from(F::from_u64(5));
    let wa_bits = ra_bits.clone();

    let mut inc = vec![K::ZERO; pow2_time];
    inc[0] = wv[0];

    let mut read_check = TwistReadCheckOracleSparseTime::new(
        &r_cycle,
        dense_to_sparse(&has_read),
        dense_to_sparse(&rv),
        cols_to_sparse(&ra_bits),
        dense_to_sparse(&has_write),
        dense_to_sparse(&inc),
        cols_to_sparse(&wa_bits),
        &r_addr,
        init_at_r_addr,
    );
    let mut write_check = TwistWriteCheckOracleSparseTime::new(
        &r_cycle,
        dense_to_sparse(&has_write),
        dense_to_sparse(&wv),
        dense_to_sparse(&inc),
        cols_to_sparse(&wa_bits),
        &r_addr,
        init_at_r_addr,
    );

    let mut bitness: Vec<LazyBitnessOracleSparseTime> = Vec::with_capacity(2 * ell_addr + 2);
    for col in ra_bits.iter().cloned().chain(wa_bits.iter().cloned()) {
        bitness.push(LazyBitnessOracleSparseTime::new_with_cycle(
            &r_cycle,
            dense_to_sparse(&col),
        ));
    }
    bitness.push(LazyBitnessOracleSparseTime::new_with_cycle(
        &r_cycle,
        dense_to_sparse(&has_read),
    ));
    bitness.push(LazyBitnessOracleSparseTime::new_with_cycle(
        &r_cycle,
        dense_to_sparse(&has_write),
    ));

    let mut claimed_sums = Vec::with_capacity(2 + bitness.len());
    let mut degree_bounds = Vec::with_capacity(2 + bitness.len());
    let mut labels: Vec<&[u8]> = Vec::with_capacity(2 + bitness.len());
    let mut claims: Vec<BatchedClaim<'_>> = Vec::with_capacity(2 + bitness.len());

    claimed_sums.push(K::ZERO);
    degree_bounds.push(read_check.degree_bound());
    labels.push(b"twist/read_check");
    claims.push(BatchedClaim {
        oracle: &mut read_check,
        claimed_sum: K::ZERO,
        label: b"twist/read_check",
    });

    claimed_sums.push(K::ZERO);
    degree_bounds.push(write_check.degree_bound());
    labels.push(b"twist/write_check");
    claims.push(BatchedClaim {
        oracle: &mut write_check,
        claimed_sum: K::ZERO,
        label: b"twist/write_check",
    });

    for bit in bitness.iter_mut() {
        claimed_sums.push(K::ZERO);
        degree_bounds.push(bit.degree_bound());
        labels.push(b"twist/bitness");
        claims.push(BatchedClaim {
            oracle: bit,
            claimed_sum: K::ZERO,
            label: b"twist/bitness",
        });
    }

    let mut tr_p = Poseidon2Transcript::new(b"twist/route_a_negative");
    let (_r_time, per_claim_results) =
        run_batched_sumcheck_prover_ds(&mut tr_p, b"twist/time_batch", 0, claims.as_mut_slice())
            .expect("Twist time batch should succeed on valid witness");

    let round_polys: Vec<Vec<Vec<K>>> = per_claim_results
        .into_iter()
        .map(|r| r.round_polys)
        .collect();

    let mut tr_v = Poseidon2Transcript::new(b"twist/route_a_negative");
    let (_r_time_v, _finals, ok) = verify_batched_sumcheck_rounds_ds(
        &mut tr_v,
        b"twist/time_batch",
        0,
        &round_polys,
        &claimed_sums,
        &labels,
        &degree_bounds,
    );
    assert!(ok, "Twist time batch should verify on valid witness");

    // Now flip one write increment and the prover must fail (claimed sum stays 0).
    inc[0] += K::ONE;
    let mut bad_write_check = TwistWriteCheckOracleSparseTime::new(
        &r_cycle,
        dense_to_sparse(&has_write),
        dense_to_sparse(&wv),
        dense_to_sparse(&inc),
        cols_to_sparse(&wa_bits),
        &r_addr,
        init_at_r_addr,
    );

    let mut bad_claims = vec![BatchedClaim {
        oracle: &mut bad_write_check,
        claimed_sum: K::ZERO,
        label: b"twist/write_check",
    }];

    let mut tr_bad = Poseidon2Transcript::new(b"twist/route_a_negative_bad");
    assert!(
        run_batched_sumcheck_prover_ds(&mut tr_bad, b"twist/time_batch", 0, bad_claims.as_mut_slice()).is_err(),
        "Twist proving must fail when inc_at_write_addr is corrupted"
    );
}

#[test]
fn route_a_proof_tamper_round_poly_fails() {
    let ell_time = 2usize;
    let ell_addr = 2usize;
    let pow2_time = 1usize << ell_time;
    let pow2_addr = 1usize << ell_addr;

    let r_cycle = vec![K::ZERO; ell_time];

    let table: Vec<K> = (0..pow2_addr)
        .map(|a| K::from(F::from_u64((10 + a) as u64)))
        .collect();

    let addrs = vec![1usize, 0, 0, 0];
    let addr_bits = addr_bits_columns(&addrs, ell_addr);
    let has_lookup = vec![K::ONE, K::ZERO, K::ZERO, K::ZERO];

    let mut val = vec![K::ZERO; pow2_time];
    val[0] = table[addrs[0]];

    let r_addr = vec![K::from(F::from_u64(7)); ell_addr];

    let (mut value_oracle, value_claim) =
        ShoutValueOracleSparse::new(&r_cycle, dense_to_sparse(&has_lookup), dense_to_sparse(&val));
    let (mut adapter_oracle, adapter_claim) = IndexAdapterOracleSparseTime::new_with_gate(
        &r_cycle,
        dense_to_sparse(&has_lookup),
        cols_to_sparse(&addr_bits),
        &r_addr,
    );

    let claimed_sums = vec![value_claim, adapter_claim];
    let degree_bounds = vec![value_oracle.degree_bound(), adapter_oracle.degree_bound()];
    let labels: Vec<&[u8]> = vec![b"shout/value", b"shout/adapter"];

    let mut claims = vec![
        BatchedClaim {
            oracle: &mut value_oracle,
            claimed_sum: value_claim,
            label: b"shout/value",
        },
        BatchedClaim {
            oracle: &mut adapter_oracle,
            claimed_sum: adapter_claim,
            label: b"shout/adapter",
        },
    ];

    let mut tr_p = Poseidon2Transcript::new(b"shout/route_a_tamper");
    let (_r_time, per_claim_results) =
        run_batched_sumcheck_prover_ds(&mut tr_p, b"shout/time_batch", 0, claims.as_mut_slice())
            .expect("prover should succeed");
    let mut round_polys: Vec<Vec<Vec<K>>> = per_claim_results
        .into_iter()
        .map(|r| r.round_polys)
        .collect();

    let mut tr_v = Poseidon2Transcript::new(b"shout/route_a_tamper");
    let (_r_time_v, _finals, ok) = verify_batched_sumcheck_rounds_ds(
        &mut tr_v,
        b"shout/time_batch",
        0,
        &round_polys,
        &claimed_sums,
        &labels,
        &degree_bounds,
    );
    assert!(ok, "baseline verify should succeed");

    // Tamper with one coefficient and verification must fail.
    round_polys[0][0][0] += K::ONE;
    let mut tr_v_bad = Poseidon2Transcript::new(b"shout/route_a_tamper");
    let (_r_time_v2, _finals2, ok2) = verify_batched_sumcheck_rounds_ds(
        &mut tr_v_bad,
        b"shout/time_batch",
        0,
        &round_polys,
        &claimed_sums,
        &labels,
        &degree_bounds,
    );
    assert!(!ok2, "verification must fail after tampering");
}
