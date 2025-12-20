//! Phase-1 Twist helper/oracle tests.

use neo_math::K;
use neo_memory::twist::compute_val_at_r_addr_pre_write;
use neo_memory::twist_oracle::{build_eq_table, LazyReadCheckOracle, LazyWriteCheckOracle};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn k(u: u64) -> K {
    K::from(Goldilocks::from_u64(u))
}

fn addr_to_bits(addr: usize, ell: usize) -> Vec<K> {
    (0..ell)
        .map(|b| if ((addr >> b) & 1) == 1 { K::ONE } else { K::ZERO })
        .collect()
}

#[test]
fn compute_val_at_r_addr_matches_reference_simulation_on_boolean_r_addr() {
    // ell_cycle=3 => pow2_cycle=8, ell_addr=2 => 4 addresses
    let pow2_cycle = 8usize;
    let ell_addr = 2usize;
    let r_addr_bool = addr_to_bits(2, ell_addr);

    // A small write trace (boolean address bits).
    let mut wa_bits = vec![vec![K::ZERO; pow2_cycle]; ell_addr];
    let mut has_write = vec![K::ZERO; pow2_cycle];
    let mut inc_at_write_addr = vec![K::ZERO; pow2_cycle];

    // Writes:
    // t=1: addr=2, inc=+5
    // t=3: addr=1, inc=+7
    // t=4: addr=2, inc=+11
    for (t, addr, inc) in [(1usize, 2usize, 5u64), (3usize, 1usize, 7u64), (4usize, 2usize, 11u64)] {
        has_write[t] = K::ONE;
        inc_at_write_addr[t] = k(inc);
        let bits = addr_to_bits(addr, ell_addr);
        for b in 0..ell_addr {
            wa_bits[b][t] = bits[b];
        }
    }

    let got = compute_val_at_r_addr_pre_write(&wa_bits, &has_write, &inc_at_write_addr, &r_addr_bool, K::ZERO);

    // Reference simulation over the concrete address r_addr_bool (here: addr=2).
    let mut mem = vec![K::ZERO; 1usize << ell_addr];
    let mut expected = vec![K::ZERO; pow2_cycle];
    let r_addr_int = 2usize;
    for t in 0..pow2_cycle {
        expected[t] = mem[r_addr_int];
        if has_write[t] == K::ONE {
            let mut addr_t = 0usize;
            for b in 0..ell_addr {
                if wa_bits[b][t] == K::ONE {
                    addr_t |= 1 << b;
                }
            }
            mem[addr_t] += inc_at_write_addr[t];
        }
    }

    assert_eq!(got, expected);
}

#[test]
fn read_write_check_oracles_have_zero_claim_on_single_address_trace() {
    // Construct a trace where all reads/writes are to the same boolean address (so eq(bits, r_addr) is an indicator).
    let pow2_cycle = 4usize;
    let ell_addr = 2usize;
    let addr = 1usize;
    let r_addr = addr_to_bits(addr, ell_addr);

    // r_cycle must avoid {0,1} so eq_table weights are non-zero.
    let r_cycle = vec![k(2), k(3)];
    let _eq_cycle = build_eq_table(&r_cycle);

    let mut ra_bits = vec![vec![K::ZERO; pow2_cycle]; ell_addr];
    let mut wa_bits = vec![vec![K::ZERO; pow2_cycle]; ell_addr];
    for b in 0..ell_addr {
        let bit = if ((addr >> b) & 1) == 1 { K::ONE } else { K::ZERO };
        for t in 0..pow2_cycle {
            ra_bits[b][t] = bit;
            wa_bits[b][t] = bit;
        }
    }

    // Memory starts at 0 and evolves only at `addr`.
    // t=0: write new=5 (inc=+5)
    // t=1: read -> 5
    // t=2: write new=9 (inc=+4)
    // t=3: read -> 9
    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ZERO];
    let has_read = vec![K::ZERO, K::ONE, K::ZERO, K::ONE];
    let wv = vec![k(5), K::ZERO, k(9), K::ZERO];
    let rv = vec![K::ZERO, k(5), K::ZERO, k(9)];
    let inc_at_write_addr = vec![k(5), K::ZERO, k(4), K::ZERO];

    let val_at_r_addr = compute_val_at_r_addr_pre_write(&wa_bits, &has_write, &inc_at_write_addr, &r_addr, K::ZERO);

    let read_check =
        LazyReadCheckOracle::new_with_cycle(&r_cycle, &ra_bits, val_at_r_addr.clone(), rv, has_read, &r_addr);
    let write_check = LazyWriteCheckOracle::new_with_cycle(
        &r_cycle,
        &wa_bits,
        wv,
        val_at_r_addr,
        inc_at_write_addr,
        has_write,
        &r_addr,
    );

    assert_eq!(read_check.compute_claim(), K::ZERO);
    assert_eq!(write_check.compute_claim(), K::ZERO);
}

#[test]
fn corrupting_rv_or_inc_breaks_read_or_write_check_claim() {
    let pow2_cycle = 4usize;
    let ell_addr = 2usize;
    let addr = 1usize;
    let r_addr = addr_to_bits(addr, ell_addr);

    let r_cycle = vec![k(2), k(3)];

    let mut ra_bits = vec![vec![K::ZERO; pow2_cycle]; ell_addr];
    let mut wa_bits = vec![vec![K::ZERO; pow2_cycle]; ell_addr];
    for b in 0..ell_addr {
        let bit = if ((addr >> b) & 1) == 1 { K::ONE } else { K::ZERO };
        for t in 0..pow2_cycle {
            ra_bits[b][t] = bit;
            wa_bits[b][t] = bit;
        }
    }

    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ZERO];
    let has_read = vec![K::ZERO, K::ONE, K::ZERO, K::ONE];
    let wv = vec![k(5), K::ZERO, k(9), K::ZERO];
    let mut rv = vec![K::ZERO, k(5), K::ZERO, k(9)];
    let mut inc_at_write_addr = vec![k(5), K::ZERO, k(4), K::ZERO];

    // Corrupt one read value and confirm read_check claim becomes non-zero.
    rv[1] += K::ONE;
    let val_at_r_addr = compute_val_at_r_addr_pre_write(&wa_bits, &has_write, &inc_at_write_addr, &r_addr, K::ZERO);
    let read_check = LazyReadCheckOracle::new_with_cycle(&r_cycle, &ra_bits, val_at_r_addr, rv, has_read, &r_addr);
    assert_ne!(read_check.compute_claim(), K::ZERO);

    // Corrupt one increment and confirm write_check claim becomes non-zero.
    inc_at_write_addr[0] += K::ONE;
    let val_at_r_addr = compute_val_at_r_addr_pre_write(&wa_bits, &has_write, &inc_at_write_addr, &r_addr, K::ZERO);
    let write_check = LazyWriteCheckOracle::new_with_cycle(
        &r_cycle,
        &wa_bits,
        wv,
        val_at_r_addr,
        inc_at_write_addr,
        has_write,
        &r_addr,
    );
    assert_ne!(write_check.compute_claim(), K::ZERO);
}
