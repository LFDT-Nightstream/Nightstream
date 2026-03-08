use neo_math::K;
use neo_memory::riscv::exec_table::{Rv32ShoutEventRow, Rv32ShoutEventTable};
use neo_memory::riscv::sparse_access::{
    rv32_shout_event_table_ra_val_mle_eval_chunked, rv32_shout_event_table_to_sparse_ra_and_val,
};
use p3_field::PrimeCharacteristicRing;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn chi_at_index(r: &[K], idx: u128) -> K {
    let mut acc = K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

#[test]
fn shout_event_table_sparse_ra_val_mle_matches_direct_sum() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Small cycle domain so the direct sum is easy to compute.
    let ell_cycle = 3usize;
    let max_cycle = 1usize << ell_cycle;

    // A tiny synthetic Shout event table (keys are arbitrary u64s here).
    let rows = vec![
        Rv32ShoutEventRow {
            row_idx: 0,
            cycle: 0,
            pc: 0,
            shout_id: 1,
            opcode: None,
            key: 0x0123_4567_89ab_cdef,
            lhs: 0,
            rhs: 0,
            value: 11,
        },
        Rv32ShoutEventRow {
            row_idx: 1,
            cycle: 1,
            pc: 4,
            shout_id: 2,
            opcode: None,
            key: 0xfedc_ba98_7654_3210,
            lhs: 0,
            rhs: 0,
            value: 22,
        },
        // Duplicate address at a different cycle (allowed).
        Rv32ShoutEventRow {
            row_idx: 4,
            cycle: 4,
            pc: 16,
            shout_id: 1,
            opcode: None,
            key: 0x0123_4567_89ab_cdef,
            lhs: 0,
            rhs: 0,
            value: 33,
        },
        // Same cycle, different address (also allowed at the event-table layer).
        Rv32ShoutEventRow {
            row_idx: 4,
            cycle: 4,
            pc: 16,
            shout_id: 3,
            opcode: None,
            key: 0x0000_0000_dead_beef,
            lhs: 0,
            rhs: 0,
            value: 44,
        },
    ];
    for r in rows.iter() {
        assert!(r.row_idx < max_cycle, "row_idx must fit ell_cycle");
    }
    let events = Rv32ShoutEventTable { rows };

    let (ra, val) = rv32_shout_event_table_to_sparse_ra_and_val(&events, ell_cycle).expect("sparse mats");

    // Random evaluation points.
    let r_addr: Vec<K> = (0..64).map(|_| K::from_u64(rng.random::<u64>())).collect();
    let r_cycle: Vec<K> = (0..ell_cycle)
        .map(|_| K::from_u64(rng.random::<u64>()))
        .collect();

    // Direct expected sums.
    let mut expected_ra = K::ZERO;
    let mut expected_val = K::ZERO;
    for row in events.rows.iter() {
        let addr = row.key;
        let cycle = row.row_idx as u128;
        let w = chi_at_index(&r_addr, addr) * chi_at_index(&r_cycle, cycle);
        expected_ra += w;
        expected_val += K::from_u64(row.value) * w;
    }

    let got_ra = ra
        .mle_eval_by_folding(&r_addr, &r_cycle)
        .expect("ra mle_eval_by_folding");
    let got_val = val
        .mle_eval_by_folding(&r_addr, &r_cycle)
        .expect("val mle_eval_by_folding");

    assert_eq!(got_ra, expected_ra);
    assert_eq!(got_val, expected_val);

    // Chunked Jolt-style eq-table evaluation (log_k_chunk=16 → 4 chunks).
    let (got_ra_chunked, got_val_chunked) =
        rv32_shout_event_table_ra_val_mle_eval_chunked(&events, &r_addr, &r_cycle, /*log_k_chunk=*/ 16)
            .expect("chunked eval");
    assert_eq!(got_ra_chunked, expected_ra);
    assert_eq!(got_val_chunked, expected_val);
}
