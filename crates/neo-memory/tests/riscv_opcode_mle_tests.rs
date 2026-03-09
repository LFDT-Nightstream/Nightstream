use neo_memory::riscv::instruction::encode_lookup_key;
use neo_memory::riscv::instruction::operand_mode_keys_enabled;
use neo_memory::riscv::lookups::{evaluate_opcode_mle, lookup_entry, RiscvLookupTable, RiscvOpcode};
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

fn eval_low_word_identity(r: &[Goldilocks], xlen: usize) -> Goldilocks {
    let mut out = Goldilocks::ZERO;
    for (i, bit) in r.iter().take(xlen).enumerate() {
        out += Goldilocks::from_u64(1u64 << i) * *bit;
    }
    out
}

fn boolean_point_from_key(key: u128, bits: usize) -> Vec<Goldilocks> {
    (0..bits)
        .map(|i| {
            if ((key >> i) & 1) == 1 {
                Goldilocks::ONE
            } else {
                Goldilocks::ZERO
            }
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
        RiscvOpcode::Mul,
        RiscvOpcode::Mulhu,
    ];

    for op in ops {
        for seed in seeds {
            let r = sample_r(xlen, seed);
            let got = evaluate_opcode_mle::<Goldilocks>(op, &r, xlen);
            let expected = if operand_mode_keys_enabled() {
                match op {
                    RiscvOpcode::Sub => eval_low_word_identity(&r, xlen),
                    _ => eval_mle_naive(op, &r, xlen),
                }
            } else {
                eval_mle_naive(op, &r, xlen)
            };
            assert_eq!(got, expected, "opcode={op:?}, seed={seed}");
        }
    }
}

#[test]
fn word_opcode_mles_match_boolean_point_lookup_entries() {
    let cases = [
        (RiscvOpcode::Addw, 0x1234_5678u64, 0x8765_4321u64),
        (RiscvOpcode::Subw, 0x1234_5678u64, 0x0000_1000u64),
        (RiscvOpcode::Sllw, 0x0000_1234u64, 37u64),
        (RiscvOpcode::Srlw, 0xF000_1234u64, 41u64),
        (RiscvOpcode::Sraw, 0xF000_1234u64, 63u64),
    ];

    for (op, lhs, rhs) in cases {
        let key = encode_lookup_key(op, lhs, rhs, /*xlen=*/ 32);
        let r = boolean_point_from_key(key, /*bits=*/ 64);
        let got = evaluate_opcode_mle::<Goldilocks>(op, &r, /*xlen=*/ 32);
        let expected = RiscvLookupTable::<Goldilocks>::new(op, /*arch_xlen=*/ 64).lookup(key);
        assert_eq!(got, expected, "opcode={op:?} lhs={lhs:#x} rhs={rhs:#x}");
    }
}
