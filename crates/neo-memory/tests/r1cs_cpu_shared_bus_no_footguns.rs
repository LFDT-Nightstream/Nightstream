//! Tests for the shared CPU bus "no footguns" wiring in `R1csCpu`.

use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::{check_ccs_rowwise_zero, CcsStructure};
use neo_ccs::traits::SModuleHomomorphism;
use neo_memory::cpu::constraints::{ShoutCpuBinding, TwistCpuBinding};
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::{CpuArithmetization, R1csCpu, SharedCpuBusConfig};
use neo_params::NeoParams;
use neo_vm_trace::{StepTrace, VmTrace};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

#[derive(Clone, Copy, Default)]
struct NoopCommit;

impl SModuleHomomorphism<F, ()> for NoopCommit {
    fn commit(&self, _z: &Mat<F>) -> () {}

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn empty_identity_first_r1cs_ccs(n: usize) -> CcsStructure<F> {
    let i_n = Mat::identity(n);
    let a = Mat::zero(n, n, F::ZERO);
    let b = Mat::zero(n, n, F::ZERO);
    let c = Mat::zero(n, n, F::ZERO);

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

fn one_empty_step_trace() -> VmTrace<u64, u64> {
    VmTrace {
        steps: vec![StepTrace {
            cycle: 0,
            pc_before: 0,
            pc_after: 0,
            opcode: 0,
            regs_before: Vec::new(),
            regs_after: Vec::new(),
            twist_events: Vec::new(),
            shout_events: Vec::new(),
            halted: false,
        }],
    }
}

#[test]
fn with_shared_cpu_bus_injects_constraints_and_forces_const_one() {
    let n = 64usize;
    let ccs = empty_identity_first_r1cs_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let mut tables: HashMap<u32, LutTable<F>> = HashMap::new();
    tables.insert(
        1,
        LutTable {
            table_id: 1,
            k: 2,
            d: 1,
            n_side: 2,
            content: vec![F::ZERO, F::ONE],
        },
    );

    let cpu = R1csCpu::new(
        ccs.clone(),
        params,
        NoopCommit::default(),
        /*m_in=*/ 1,
        &tables,
        &HashMap::new(),
        Box::new(|_step| vec![F::ZERO]),
    );

    let mut mem_layouts: HashMap<u32, PlainMemLayout> = HashMap::new();
    mem_layouts.insert(2, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    let cfg = SharedCpuBusConfig::<F> {
        mem_layouts,
        initial_mem: HashMap::new(),
        const_one_col: 0,
        shout_cpu: HashMap::from([(
            1,
            ShoutCpuBinding {
                has_lookup: 1,
                addr: 2,
                val: 3,
            },
        )]),
        twist_cpu: HashMap::from([(
            2,
            TwistCpuBinding {
                has_read: 4,
                has_write: 5,
                read_addr: 6,
                write_addr: 7,
                rv: 8,
                wv: 9,
                inc: None,
            },
        )]),
    };

    let cpu = cpu
        .with_shared_cpu_bus(cfg, 1)
        .expect("enable shared_cpu_bus");

    // Base CCS had A/B all-zero; injection should make both non-zero.
    assert!(
        cpu.ccs.matrices[1].as_slice().iter().any(|&v| v != F::ZERO),
        "expected injected constraints in A matrix"
    );
    assert!(
        cpu.ccs.matrices[2].as_slice().iter().any(|&v| v != F::ZERO),
        "expected injected constraints in B matrix"
    );

    let trace = one_empty_step_trace();
    let mcss = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build ccs steps");
    assert_eq!(mcss.len(), 1);
    let (mcs_inst, mcs_wit) = &mcss[0];
    assert_eq!(mcs_inst.x.len(), 1);
    assert_eq!(mcs_inst.x[0], F::ONE, "const_one_col should be forced to 1");

    // The injected constraints should be satisfiable for the constructed witness.
    check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("satisfiable");
}

#[test]
fn with_shared_cpu_bus_rejects_non_public_const_one() {
    let n = 64usize;
    let ccs = empty_identity_first_r1cs_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let tables: HashMap<u32, LutTable<F>> = HashMap::new();
    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        /*m_in=*/ 1,
        &tables,
        &HashMap::new(),
        Box::new(|_step| vec![F::ZERO]),
    );

    let cfg = SharedCpuBusConfig::<F> {
        mem_layouts: HashMap::new(),
        initial_mem: HashMap::new(),
        const_one_col: 1, // not < m_in
        shout_cpu: HashMap::new(),
        twist_cpu: HashMap::new(),
    };

    assert!(
        cpu.with_shared_cpu_bus(cfg, 1).is_err(),
        "const_one_col >= m_in must be rejected"
    );
}

#[test]
fn with_shared_cpu_bus_rejects_bindings_in_bus_tail() {
    let n = 64usize;
    let ccs = empty_identity_first_r1cs_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let mut tables: HashMap<u32, LutTable<F>> = HashMap::new();
    tables.insert(
        1,
        LutTable {
            table_id: 1,
            k: 2,
            d: 1,
            n_side: 2,
            content: vec![F::ZERO, F::ONE],
        },
    );

    let cpu = R1csCpu::new(
        ccs.clone(),
        params,
        NoopCommit::default(),
        /*m_in=*/ 1,
        &tables,
        &HashMap::new(),
        Box::new(|_step| vec![F::ZERO]),
    );

    let mut mem_layouts: HashMap<u32, PlainMemLayout> = HashMap::new();
    mem_layouts.insert(2, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    // One shout + one twist => bus_cols_total = (1*1 + 2) + (2*1*1 + 5) = 10, so bus_base = 64-10 = 54.
    let cfg = SharedCpuBusConfig::<F> {
        mem_layouts,
        initial_mem: HashMap::new(),
        const_one_col: 0,
        shout_cpu: HashMap::from([(
            1,
            ShoutCpuBinding {
                has_lookup: 54, // inside bus tail
                addr: 2,
                val: 3,
            },
        )]),
        twist_cpu: HashMap::from([(
            2,
            TwistCpuBinding {
                has_read: 4,
                has_write: 5,
                read_addr: 6,
                write_addr: 7,
                rv: 8,
                wv: 9,
                inc: None,
            },
        )]),
    };

    assert!(
        cpu.with_shared_cpu_bus(cfg, 1).is_err(),
        "bindings that overlap the bus tail must be rejected"
    );
}
