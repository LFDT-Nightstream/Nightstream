use neo_memory::riscv_lookups::{evaluate_opcode_mle, lookup_entry, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn eval_mle_naive(op: RiscvOpcode, r: &[Goldilocks], xlen: usize) -> Goldilocks {
    assert_eq!(r.len(), 2 * xlen);
    assert!(xlen <= 8, "naive MLE test helper supports xlen<=8 only");

    let table_size = 1usize << (2 * xlen);
    let mut result = Goldilocks::ZERO;

    for idx in 0..table_size {
        let mut chi = Goldilocks::ONE;
        for k in 0..(2 * xlen) {
            let bit = ((idx >> k) & 1) as u64;
            let r_k = r[k];
            chi *= if bit == 1 { r_k } else { Goldilocks::ONE - r_k };
        }

        let entry = lookup_entry(op, idx as u128, xlen);
        result += chi * Goldilocks::from_u64(entry);
    }

    result
}

fn sample_r(xlen: usize, seed: u64) -> Vec<Goldilocks> {
    (0..(2 * xlen))
        .map(|i| {
            let i = i as u64;
            // Deterministic, non-boolean values.
            Goldilocks::from_u64(
                seed.wrapping_add(17 * i)
                    .wrapping_mul(31)
                    .wrapping_add(i * (i + 3)),
            )
        })
        .collect()
}

#[test]
fn opcode_mle_matches_naive_for_small_xlen() {
    let xlen = 8usize;
    let seeds = [1u64, 7u64, 123u64];
    let ops = [
        RiscvOpcode::Eq,
        RiscvOpcode::Neq,
        RiscvOpcode::Slt,
        RiscvOpcode::Sltu,
        RiscvOpcode::Sub,
    ];

    for op in ops {
        for seed in seeds {
            let r = sample_r(xlen, seed);
            let got = evaluate_opcode_mle::<Goldilocks>(op, &r, xlen);
            let expected = eval_mle_naive(op, &r, xlen);
            assert_eq!(got, expected, "opcode={op:?}, seed={seed}");
        }
    }
}
