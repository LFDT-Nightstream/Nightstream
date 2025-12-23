//! Tests for Twist oracles with index-bit addressing.

use neo_math::K;
use neo_memory::mle::mle_eval;
use neo_memory::twist_oracle::{
    build_eq_table, compute_eq_from_bits, val_tables_from_inc, TwistReadCheckOracle, TwistValEvalOracle,
    TwistReadCheckAddrOracle, TwistWriteCheckAddrOracle, TwistWriteCheckOracle,
};
use neo_reductions::sumcheck::{run_sumcheck_prover, verify_sumcheck_rounds, RoundOracle};
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

fn init_at_addr(init_sparse: &[(usize, K)], addr_bits: &[K]) -> K {
    let one = K::ONE;
    let mut acc = K::ZERO;
    for &(addr, val) in init_sparse.iter() {
        let mut chi = one;
        for (bit_idx, &x) in addr_bits.iter().enumerate() {
            let bit = ((addr >> bit_idx) & 1) as u8;
            chi *= if bit == 1 { x } else { one - x };
        }
        acc += val * chi;
    }
    acc
}

fn eq_bits_at_time(bits: &[Vec<K>], t: usize, addr_bits: &[K]) -> K {
    let one = K::ONE;
    let mut acc = one;
    for (bit_idx, col) in bits.iter().enumerate() {
        let b = col[t];
        let x = addr_bits[bit_idx];
        acc *= b * x + (one - b) * (one - x);
    }
    acc
}

struct NaiveTwistReadCheckAddrOracle {
    ell_addr: usize,
    bit_idx: usize,
    eq_cycle: Vec<K>,
    init_sparse: Vec<(usize, K)>,
    ra_bits: Vec<Vec<K>>,
    wa_bits: Vec<Vec<K>>,
    has_read: Vec<K>,
    rv: Vec<K>,
    has_write: Vec<K>,
    inc_at_write_addr: Vec<K>,
    bound_prefix: Vec<K>,
}

impl NaiveTwistReadCheckAddrOracle {
    fn eval_g(&self, addr_bits: &[K]) -> K {
        let mut mem = init_at_addr(&self.init_sparse, addr_bits);
        let mut sum = K::ZERO;
        for t in 0..self.eq_cycle.len() {
            let eq_ra = eq_bits_at_time(&self.ra_bits, t, addr_bits);
            sum += self.eq_cycle[t] * self.has_read[t] * eq_ra * (mem - self.rv[t]);

            let eq_wa = eq_bits_at_time(&self.wa_bits, t, addr_bits);
            mem += self.has_write[t] * self.inc_at_write_addr[t] * eq_wa;
        }
        sum
    }
}

impl RoundOracle for NaiveTwistReadCheckAddrOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let val = self.eval_g(&self.bound_prefix);
            return vec![val; points.len()];
        }

        let rest = self.ell_addr - self.bit_idx - 1;
        let mut ys = vec![K::ZERO; points.len()];
        for (i, &x) in points.iter().enumerate() {
            let mut addr_bits = vec![K::ZERO; self.ell_addr];
            addr_bits[..self.bit_idx].copy_from_slice(&self.bound_prefix);
            addr_bits[self.bit_idx] = x;

            let mut acc = K::ZERO;
            for rest_mask in 0..(1usize << rest) {
                for j in 0..rest {
                    addr_bits[self.bit_idx + 1 + j] = if ((rest_mask >> j) & 1) == 1 {
                        K::ONE
                    } else {
                        K::ZERO
                    };
                }
                acc += self.eval_g(&addr_bits);
            }
            ys[i] = acc;
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr - self.bit_idx
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn fold(&mut self, r: K) {
        self.bound_prefix.push(r);
        self.bit_idx += 1;
    }
}

struct NaiveTwistWriteCheckAddrOracle {
    ell_addr: usize,
    bit_idx: usize,
    eq_cycle: Vec<K>,
    init_sparse: Vec<(usize, K)>,
    wa_bits: Vec<Vec<K>>,
    has_write: Vec<K>,
    wv: Vec<K>,
    inc_at_write_addr: Vec<K>,
    bound_prefix: Vec<K>,
}

impl NaiveTwistWriteCheckAddrOracle {
    fn eval_g(&self, addr_bits: &[K]) -> K {
        let mut mem = init_at_addr(&self.init_sparse, addr_bits);
        let mut sum = K::ZERO;
        for t in 0..self.eq_cycle.len() {
            let eq_wa = eq_bits_at_time(&self.wa_bits, t, addr_bits);
            sum += self.eq_cycle[t]
                * self.has_write[t]
                * eq_wa
                * (self.wv[t] - mem - self.inc_at_write_addr[t]);

            mem += self.has_write[t] * self.inc_at_write_addr[t] * eq_wa;
        }
        sum
    }
}

impl RoundOracle for NaiveTwistWriteCheckAddrOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let val = self.eval_g(&self.bound_prefix);
            return vec![val; points.len()];
        }

        let rest = self.ell_addr - self.bit_idx - 1;
        let mut ys = vec![K::ZERO; points.len()];
        for (i, &x) in points.iter().enumerate() {
            let mut addr_bits = vec![K::ZERO; self.ell_addr];
            addr_bits[..self.bit_idx].copy_from_slice(&self.bound_prefix);
            addr_bits[self.bit_idx] = x;

            let mut acc = K::ZERO;
            for rest_mask in 0..(1usize << rest) {
                for j in 0..rest {
                    addr_bits[self.bit_idx + 1 + j] = if ((rest_mask >> j) & 1) == 1 {
                        K::ONE
                    } else {
                        K::ZERO
                    };
                }
                acc += self.eval_g(&addr_bits);
            }
            ys[i] = acc;
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr - self.bit_idx
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn fold(&mut self, r: K) {
        self.bound_prefix.push(r);
        self.bit_idx += 1;
    }
}

#[test]
fn read_check_addr_oracle_matches_naive_dense() {
    let ell_addr = 5usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse = vec![(0usize, k(9)), (5usize, k(17)), (31usize, k(23))];

    let has_read = (0..pow2_cycle)
        .map(|t| if t % 3 == 0 || t == 1 { K::ONE } else { K::ZERO })
        .collect::<Vec<_>>();
    let rv = (0..pow2_cycle).map(|t| k(100 + t as u64)).collect::<Vec<_>>();

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(5 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let pow2_addr = 1usize << ell_addr;
    let ra_addrs = (0..pow2_cycle)
        .map(|t| (3 + 4 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let wa_addrs = (0..pow2_cycle)
        .map(|t| (7 + 3 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let ra_bits = bits_from_addrs(&ra_addrs, ell_addr);
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut sparse = TwistReadCheckAddrOracle::new(
        init_sparse.clone(),
        &r_cycle,
        has_read.clone(),
        rv.clone(),
        &ra_bits,
        has_write.clone(),
        &wa_bits,
        inc_at_write_addr.clone(),
    );
    let mut dense = NaiveTwistReadCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse,
        ra_bits,
        wa_bits,
        has_read,
        rv,
        has_write,
        inc_at_write_addr,
        bound_prefix: Vec::new(),
    };

    let points = vec![K::ZERO, K::ONE, k(2)];
    for round in 0..=ell_addr {
        assert_eq!(sparse.evals_at(&points), dense.evals_at(&points));
        if round == ell_addr {
            break;
        }
        let r = k(3 + 2 * round as u64);
        sparse.fold(r);
        dense.fold(r);
    }
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.num_rounds(), 0);
}

#[test]
fn write_check_addr_oracle_matches_naive_dense() {
    let ell_addr = 5usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse = vec![(0usize, k(9)), (5usize, k(17)), (31usize, k(23))];

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let wv = (0..pow2_cycle).map(|t| k(200 + t as u64)).collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(7 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let pow2_addr = 1usize << ell_addr;
    let wa_addrs = (0..pow2_cycle)
        .map(|t| (7 + 3 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut sparse = TwistWriteCheckAddrOracle::new(
        init_sparse.clone(),
        &r_cycle,
        has_write.clone(),
        wv.clone(),
        &wa_bits,
        inc_at_write_addr.clone(),
    );
    let mut dense = NaiveTwistWriteCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse,
        wa_bits,
        has_write,
        wv,
        inc_at_write_addr,
        bound_prefix: Vec::new(),
    };

    let points = vec![K::ZERO, K::ONE, k(2)];
    for round in 0..=ell_addr {
        assert_eq!(sparse.evals_at(&points), dense.evals_at(&points));
        if round == ell_addr {
            break;
        }
        let r = k(3 + 2 * round as u64);
        sparse.fold(r);
        dense.fold(r);
    }
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.num_rounds(), 0);
}

#[test]
fn read_check_addr_oracle_empty_init_matches_naive_dense() {
    let ell_addr = 5usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse: Vec<(usize, K)> = Vec::new();

    let has_read = (0..pow2_cycle)
        .map(|t| if t % 3 == 0 || t == 1 { K::ONE } else { K::ZERO })
        .collect::<Vec<_>>();
    let rv = (0..pow2_cycle).map(|t| k(100 + t as u64)).collect::<Vec<_>>();

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(5 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let pow2_addr = 1usize << ell_addr;
    let ra_addrs = (0..pow2_cycle)
        .map(|t| (3 + 4 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let wa_addrs = (0..pow2_cycle)
        .map(|t| (7 + 3 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let ra_bits = bits_from_addrs(&ra_addrs, ell_addr);
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut sparse = TwistReadCheckAddrOracle::new(
        init_sparse.clone(),
        &r_cycle,
        has_read.clone(),
        rv.clone(),
        &ra_bits,
        has_write.clone(),
        &wa_bits,
        inc_at_write_addr.clone(),
    );
    let mut dense = NaiveTwistReadCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse,
        ra_bits,
        wa_bits,
        has_read,
        rv,
        has_write,
        inc_at_write_addr,
        bound_prefix: Vec::new(),
    };

    let points = vec![K::ZERO, K::ONE, k(2)];
    for round in 0..=ell_addr {
        assert_eq!(sparse.evals_at(&points), dense.evals_at(&points));
        if round == ell_addr {
            break;
        }
        let r = k(3 + 2 * round as u64);
        sparse.fold(r);
        dense.fold(r);
    }
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.num_rounds(), 0);
}

#[test]
fn write_check_addr_oracle_empty_init_matches_naive_dense() {
    let ell_addr = 5usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse: Vec<(usize, K)> = Vec::new();

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let wv = (0..pow2_cycle).map(|t| k(200 + t as u64)).collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(7 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let pow2_addr = 1usize << ell_addr;
    let wa_addrs = (0..pow2_cycle)
        .map(|t| (7 + 3 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut sparse = TwistWriteCheckAddrOracle::new(
        init_sparse.clone(),
        &r_cycle,
        has_write.clone(),
        wv.clone(),
        &wa_bits,
        inc_at_write_addr.clone(),
    );
    let mut dense = NaiveTwistWriteCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse,
        wa_bits,
        has_write,
        wv,
        inc_at_write_addr,
        bound_prefix: Vec::new(),
    };

    let points = vec![K::ZERO, K::ONE, k(2)];
    for round in 0..=ell_addr {
        assert_eq!(sparse.evals_at(&points), dense.evals_at(&points));
        if round == ell_addr {
            break;
        }
        let r = k(3 + 2 * round as u64);
        sparse.fold(r);
        dense.fold(r);
    }
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.num_rounds(), 0);
}

#[test]
fn read_check_addr_oracle_ell_addr_zero_matches_naive_dense() {
    let ell_addr = 0usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse = vec![(0usize, k(9))];

    let has_read = (0..pow2_cycle)
        .map(|t| if t % 3 == 0 || t == 1 { K::ONE } else { K::ZERO })
        .collect::<Vec<_>>();
    let rv = (0..pow2_cycle).map(|t| k(100 + t as u64)).collect::<Vec<_>>();

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(5 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let ra_addrs = vec![0usize; pow2_cycle];
    let wa_addrs = vec![0usize; pow2_cycle];
    let ra_bits = bits_from_addrs(&ra_addrs, ell_addr);
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut sparse = TwistReadCheckAddrOracle::new(
        init_sparse.clone(),
        &r_cycle,
        has_read.clone(),
        rv.clone(),
        &ra_bits,
        has_write.clone(),
        &wa_bits,
        inc_at_write_addr.clone(),
    );
    let mut dense = NaiveTwistReadCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse,
        ra_bits,
        wa_bits,
        has_read,
        rv,
        has_write,
        inc_at_write_addr,
        bound_prefix: Vec::new(),
    };

    let points = vec![K::ZERO, K::ONE, k(2)];
    assert_eq!(sparse.evals_at(&points), dense.evals_at(&points));
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.num_rounds(), 0);
}

#[test]
fn write_check_addr_oracle_ell_addr_zero_matches_naive_dense() {
    let ell_addr = 0usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse = vec![(0usize, k(9))];

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let wv = (0..pow2_cycle).map(|t| k(200 + t as u64)).collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(7 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let wa_addrs = vec![0usize; pow2_cycle];
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut sparse =
        TwistWriteCheckAddrOracle::new(init_sparse.clone(), &r_cycle, has_write.clone(), wv.clone(), &wa_bits, inc_at_write_addr.clone());
    let mut dense = NaiveTwistWriteCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse,
        wa_bits,
        has_write,
        wv,
        inc_at_write_addr,
        bound_prefix: Vec::new(),
    };

    let points = vec![K::ZERO, K::ONE, k(2)];
    assert_eq!(sparse.evals_at(&points), dense.evals_at(&points));
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.num_rounds(), 0);
}

fn sum_over_boolean_addrs(ell_addr: usize, mut eval: impl FnMut(&[K]) -> K) -> K {
    let pow2_addr = 1usize << ell_addr;
    let mut acc = K::ZERO;
    for addr in 0..pow2_addr {
        let mut addr_bits = vec![K::ZERO; ell_addr];
        for b in 0..ell_addr {
            if ((addr >> b) & 1) == 1 {
                addr_bits[b] = K::ONE;
            }
        }
        acc += eval(&addr_bits);
    }
    acc
}

#[test]
fn read_check_addr_oracle_sumcheck_transcript_matches_naive_dense() {
    let ell_addr = 5usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse = vec![(0usize, k(9)), (5usize, k(17)), (31usize, k(23))];

    let has_read = (0..pow2_cycle)
        .map(|t| if t % 3 == 0 || t == 1 { K::ONE } else { K::ZERO })
        .collect::<Vec<_>>();
    let rv = (0..pow2_cycle).map(|t| k(100 + t as u64)).collect::<Vec<_>>();

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(5 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let pow2_addr = 1usize << ell_addr;
    let ra_addrs = (0..pow2_cycle)
        .map(|t| (3 + 4 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let wa_addrs = (0..pow2_cycle)
        .map(|t| (7 + 3 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let ra_bits = bits_from_addrs(&ra_addrs, ell_addr);
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut dense = NaiveTwistReadCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse: init_sparse.clone(),
        ra_bits: ra_bits.clone(),
        wa_bits: wa_bits.clone(),
        has_read: has_read.clone(),
        rv: rv.clone(),
        has_write: has_write.clone(),
        inc_at_write_addr: inc_at_write_addr.clone(),
        bound_prefix: Vec::new(),
    };
    let claim = sum_over_boolean_addrs(ell_addr, |addr_bits| dense.eval_g(addr_bits));

    let mut tr_d = Poseidon2Transcript::new(b"addr_pre/read/transcript_eq/v1");
    let (rounds_d, chals_d) = run_sumcheck_prover(&mut tr_d, &mut dense, claim).expect("dense prover");

    let mut sparse = TwistReadCheckAddrOracle::new(
        init_sparse,
        &r_cycle,
        has_read,
        rv,
        &ra_bits,
        has_write,
        &wa_bits,
        inc_at_write_addr,
    );
    let mut tr_s = Poseidon2Transcript::new(b"addr_pre/read/transcript_eq/v1");
    let (rounds_s, chals_s) = run_sumcheck_prover(&mut tr_s, &mut sparse, claim).expect("sparse prover");

    assert_eq!(rounds_d, rounds_s);
    assert_eq!(chals_d, chals_s);

    let mut tr_v = Poseidon2Transcript::new(b"addr_pre/read/transcript_eq/v1");
    let (_chals_out, final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, claim, &rounds_d);
    assert!(ok);
    assert_eq!(dense.num_rounds(), 0);
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.evals_at(&[K::ZERO])[0], final_sum);
    assert_eq!(sparse.evals_at(&[K::ZERO])[0], final_sum);
}

#[test]
fn write_check_addr_oracle_sumcheck_transcript_matches_naive_dense() {
    let ell_addr = 5usize;
    let ell_cycle = 3usize;
    let pow2_cycle = 1usize << ell_cycle;

    let r_cycle = vec![k(11), k(13), k(17)];
    let eq_cycle = build_eq_table(&r_cycle);

    let init_sparse = vec![(0usize, k(9)), (5usize, k(17)), (31usize, k(23))];

    let has_write = (0..pow2_cycle)
        .map(|t| if t % 2 == 0 { K::ZERO } else { K::ONE })
        .collect::<Vec<_>>();
    let wv = (0..pow2_cycle).map(|t| k(200 + t as u64)).collect::<Vec<_>>();
    let inc_at_write_addr = (0..pow2_cycle)
        .map(|t| k(7 * (t as u64 + 1)))
        .collect::<Vec<_>>();

    let pow2_addr = 1usize << ell_addr;
    let wa_addrs = (0..pow2_cycle)
        .map(|t| (7 + 3 * t) % pow2_addr)
        .collect::<Vec<_>>();
    let wa_bits = bits_from_addrs(&wa_addrs, ell_addr);

    let mut dense = NaiveTwistWriteCheckAddrOracle {
        ell_addr,
        bit_idx: 0,
        eq_cycle,
        init_sparse: init_sparse.clone(),
        wa_bits: wa_bits.clone(),
        has_write: has_write.clone(),
        wv: wv.clone(),
        inc_at_write_addr: inc_at_write_addr.clone(),
        bound_prefix: Vec::new(),
    };
    let claim = sum_over_boolean_addrs(ell_addr, |addr_bits| dense.eval_g(addr_bits));

    let mut tr_d = Poseidon2Transcript::new(b"addr_pre/write/transcript_eq/v1");
    let (rounds_d, chals_d) = run_sumcheck_prover(&mut tr_d, &mut dense, claim).expect("dense prover");

    let mut sparse = TwistWriteCheckAddrOracle::new(init_sparse, &r_cycle, has_write, wv, &wa_bits, inc_at_write_addr);
    let mut tr_s = Poseidon2Transcript::new(b"addr_pre/write/transcript_eq/v1");
    let (rounds_s, chals_s) = run_sumcheck_prover(&mut tr_s, &mut sparse, claim).expect("sparse prover");

    assert_eq!(rounds_d, rounds_s);
    assert_eq!(chals_d, chals_s);

    let mut tr_v = Poseidon2Transcript::new(b"addr_pre/write/transcript_eq/v1");
    let (_chals_out, final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, claim, &rounds_d);
    assert!(ok);
    assert_eq!(dense.num_rounds(), 0);
    assert_eq!(sparse.num_rounds(), 0);
    assert_eq!(dense.evals_at(&[K::ZERO])[0], final_sum);
    assert_eq!(sparse.evals_at(&[K::ZERO])[0], final_sum);
}
