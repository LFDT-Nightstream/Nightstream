//! Regression test: shared CPU bus constraint injection must be multi-instance safe.

use std::marker::PhantomData;

use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::{check_ccs_rowwise_zero, CcsStructure};
use neo_memory::cpu::constraints::{extend_ccs_with_shared_cpu_bus_constraints, ShoutCpuBinding, TwistCpuBinding};
use neo_memory::witness::{LutInstance, MemInstance};
use neo_memory::MemInit;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

fn empty_identity_first_r1cs_ccs(n: usize) -> CcsStructure<F> {
    let i_n = Mat::identity(n);
    let a = Mat::zero(n, n, F::ZERO);
    let b = Mat::zero(n, n, F::ZERO);
    let c = Mat::zero(n, n, F::ZERO);

    // f(I, A, B, C) = A * B - C, with I unused.
    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );
    CcsStructure::new(vec![i_n, a, b, c], f).expect("CCS")
}

fn lut_inst() -> LutInstance<(), F> {
    LutInstance {
        comms: Vec::new(),
        k: 2,
        d: 1,
        n_side: 2,
        steps: 1,
        ell: 1,
        table_spec: None,
        table: vec![F::ZERO, F::ONE],
        _phantom: PhantomData,
    }
}

fn mem_inst() -> MemInstance<(), F> {
    MemInstance {
        comms: Vec::new(),
        k: 2,
        d: 1,
        n_side: 2,
        steps: 1,
        ell: 1,
        init: MemInit::Zero,
        _phantom: PhantomData,
    }
}

#[test]
fn shared_cpu_bus_injection_supports_independent_instances() {
    // Base CCS has no constraints; we will inject binding+padding constraints into trailing rows.
    let n = 64usize;
    let base_ccs = empty_identity_first_r1cs_ccs(n);

    let lut_insts = vec![lut_inst(), lut_inst()];
    let mem_insts = vec![mem_inst(), mem_inst()];

    // CPU columns (all < bus_base) are per-instance.
    let shout_cpu = vec![
        ShoutCpuBinding {
            has_lookup: 1,
            addr: 2,
            val: 3,
        },
        ShoutCpuBinding {
            has_lookup: 4,
            addr: 5,
            val: 6,
        },
    ];
    let twist_cpu = vec![
        TwistCpuBinding {
            has_read: 7,
            has_write: 8,
            read_addr: 9,
            write_addr: 10,
            rv: 11,
            wv: 12,
            inc: None,
        },
        TwistCpuBinding {
            has_read: 13,
            has_write: 14,
            read_addr: 15,
            write_addr: 16,
            rv: 17,
            wv: 18,
            inc: None,
        },
    ];

    let ccs = extend_ccs_with_shared_cpu_bus_constraints(
        &base_ccs, /*m_in=*/ 0, /*const_one_col=*/ 0, &shout_cpu, &twist_cpu, &lut_insts, &mem_insts,
    )
    .expect("inject shared-bus constraints");

    // Bus tail layout (chunk_size==1): [shout0, shout1, twist0, twist1] in canonical order.
    let ell_addr = 1usize;
    let bus_cols_total = lut_insts.len() * (ell_addr + 2) + mem_insts.len() * (2 * ell_addr + 5);
    let bus_base = ccs.m - bus_cols_total;

    let mut z = vec![F::ZERO; ccs.m];
    z[0] = F::ONE; // const-one column

    // CPU witness: make only shout0 and twist1 active.
    z[1] = F::ONE; // shout0.has_lookup
    z[2] = F::ONE; // shout0.addr (packed)
    z[3] = F::from_u64(7); // shout0.val

    z[4] = F::ZERO; // shout1.has_lookup
    z[5] = F::ZERO; // shout1.addr
    z[6] = F::ZERO; // shout1.val

    z[7] = F::ZERO; // twist0.has_read
    z[8] = F::ZERO; // twist0.has_write
    z[9] = F::ZERO; // twist0.read_addr
    z[10] = F::ZERO; // twist0.write_addr
    z[11] = F::ZERO; // twist0.rv
    z[12] = F::ZERO; // twist0.wv

    z[13] = F::ONE; // twist1.has_read
    z[14] = F::ZERO; // twist1.has_write
    z[15] = F::ONE; // twist1.read_addr (packed)
    z[16] = F::ZERO; // twist1.write_addr
    z[17] = F::from_u64(9); // twist1.rv
    z[18] = F::ZERO; // twist1.wv

    // Bus (Shout 0): [addr_bits(1), has_lookup, val]
    z[bus_base + 0] = F::ONE; // addr_bit
    z[bus_base + 1] = F::ONE; // has_lookup
    z[bus_base + 2] = F::from_u64(7); // val

    // Bus (Shout 1): all zeros
    z[bus_base + 3] = F::ZERO;
    z[bus_base + 4] = F::ZERO;
    z[bus_base + 5] = F::ZERO;

    // Bus (Twist 0): [ra_bit, wa_bit, has_read, has_write, wv, rv, inc] all zeros
    for i in 0..7 {
        z[bus_base + 6 + i] = F::ZERO;
    }

    // Bus (Twist 1)
    z[bus_base + 13] = F::ONE; // ra_bit
    z[bus_base + 14] = F::ZERO; // wa_bit
    z[bus_base + 15] = F::ONE; // has_read
    z[bus_base + 16] = F::ZERO; // has_write
    z[bus_base + 17] = F::ZERO; // wv
    z[bus_base + 18] = F::from_u64(9); // rv
    z[bus_base + 19] = F::ZERO; // inc

    check_ccs_rowwise_zero(&ccs, &z, &[]).expect("constraints should be satisfied");

    // Sanity: if we (incorrectly) reuse a single CPU binding across instances (old bug),
    // the same witness must fail because instances become coupled.
    let shout_cpu_bad = vec![shout_cpu[0].clone(); lut_insts.len()];
    let twist_cpu_bad = vec![twist_cpu[0].clone(); mem_insts.len()];
    let ccs_bad = extend_ccs_with_shared_cpu_bus_constraints(
        &base_ccs,
        /*m_in=*/ 0,
        /*const_one_col=*/ 0,
        &shout_cpu_bad,
        &twist_cpu_bad,
        &lut_insts,
        &mem_insts,
    )
    .expect("inject shared-bus constraints");
    assert!(
        check_ccs_rowwise_zero(&ccs_bad, &z, &[]).is_err(),
        "reusing a single CPU binding across instances should be unsatisfiable"
    );
}
