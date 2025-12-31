use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_memory::builder::{build_shard_witness_shared_cpu_bus, CpuArithmetization};
use neo_memory::plain::PlainMemLayout;
use neo_memory::MemInit;
use neo_vm_trace::{Shout, ShoutId, StepMeta, Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[derive(Default)]
struct MapTwist {
    mem: HashMap<(TwistId, u64), u64>,
}

impl Twist<u64, u64> for MapTwist {
    fn load(&mut self, id: TwistId, addr: u64) -> u64 {
        *self.mem.get(&(id, addr)).unwrap_or(&0)
    }

    fn store(&mut self, id: TwistId, addr: u64, val: u64) {
        self.mem.insert((id, addr), val);
    }
}

#[derive(Default)]
struct NoopShout;

impl Shout<u64> for NoopShout {
    fn lookup(&mut self, _id: ShoutId, _key: u64) -> u64 {
        0
    }
}

struct ScriptCpu {
    pc: u64,
    step: usize,
    halted: bool,
}

impl ScriptCpu {
    fn new() -> Self {
        Self {
            pc: 0,
            step: 0,
            halted: false,
        }
    }
}

impl VmCpu<u64, u64> for ScriptCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![self.step as u64]
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<TW, SH>(&mut self, twist_mem: &mut TW, _shout_tbl: &mut SH) -> Result<StepMeta<u64>, Self::Error>
    where
        TW: Twist<u64, u64>,
        SH: Shout<u64>,
    {
        let mem = TwistId(0);
        match self.step {
            0 => twist_mem.store(mem, 0, 5),
            1 => twist_mem.store(mem, 1, 7),
            2 => twist_mem.store(mem, 0, 9),
            3 => {}
            _ => {}
        }

        self.step += 1;
        self.pc += 4;
        if self.step >= 4 {
            self.halted = true;
        }

        Ok(StepMeta {
            pc_after: self.pc,
            opcode: 0xAA,
        })
    }
}

#[derive(Default)]
struct DummyCpuArith;

impl CpuArithmetization<Goldilocks, ()> for DummyCpuArith {
    type Error = String;

    fn build_ccs_chunks(
        &self,
        trace: &neo_vm_trace::VmTrace<u64, u64>,
        chunk_size: usize,
    ) -> Result<Vec<(McsInstance<(), Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        if chunk_size == 0 {
            return Err("chunk_size must be >= 1".into());
        }
        let chunks_len = trace.steps.len().div_ceil(chunk_size);
        Ok((0..chunks_len)
            .map(|_| {
                (
                    McsInstance {
                        c: (),
                        x: vec![],
                        m_in: 0,
                    },
                    McsWitness {
                        w: vec![],
                        Z: Mat::zero(1, 1, Goldilocks::ZERO),
                    },
                )
            })
            .collect())
    }
}

#[test]
fn build_shard_witness_shared_cpu_bus_sets_init_policy_per_step() {
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    let lut_tables: HashMap<u32, neo_memory::plain::LutTable<Goldilocks>> = HashMap::new();
    let lut_table_specs: HashMap<u32, neo_memory::witness::LutTableSpec> = HashMap::new();
    let initial_mem: HashMap<(u32, u64), Goldilocks> = HashMap::new();

    let bundles = build_shard_witness_shared_cpu_bus::<_, (), neo_math::K, _, _, _>(
        ScriptCpu::new(),
        MapTwist::default(),
        NoopShout::default(),
        16,
        1, // chunk_size=1 => one bundle per step
        &mem_layouts,
        &lut_tables,
        &lut_table_specs,
        &initial_mem,
        &DummyCpuArith::default(),
    )
    .expect("build_shard_witness_shared_cpu_bus should succeed");

    assert_eq!(bundles.len(), 16, "builder pads to max_steps under fixed-length semantics");
    for bundle in &bundles {
        assert_eq!(bundle.mem_instances.len(), 1);
        assert_eq!(bundle.mem_instances[0].0.steps, 1);
        assert!(bundle.mem_instances[0].0.comms.is_empty());
        assert!(bundle.mem_instances[0].1.mats.is_empty());
    }

    let inst0 = &bundles[0].mem_instances[0].0;
    let inst1 = &bundles[1].mem_instances[0].0;
    let inst2 = &bundles[2].mem_instances[0].0;
    let inst3 = &bundles[3].mem_instances[0].0;
    let inst4 = &bundles[4].mem_instances[0].0;
    let inst_last = &bundles.last().expect("non-empty").mem_instances[0].0;

    assert!(matches!(inst0.init, MemInit::Zero));
    assert_eq!(inst1.init, MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(5))]));
    assert_eq!(
        inst2.init,
        MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(5)), (1u64, Goldilocks::from_u64(7))])
    );
    assert_eq!(
        inst3.init,
        MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(9)), (1u64, Goldilocks::from_u64(7))])
    );
    assert_eq!(
        inst4.init,
        MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(9)), (1u64, Goldilocks::from_u64(7))])
    );
    assert_eq!(inst_last.init, inst4.init);
}

#[test]
fn build_shard_witness_shared_cpu_bus_supports_chunk_size_gt_one() {
    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    let lut_tables: HashMap<u32, neo_memory::plain::LutTable<Goldilocks>> = HashMap::new();
    let lut_table_specs: HashMap<u32, neo_memory::witness::LutTableSpec> = HashMap::new();
    let initial_mem: HashMap<(u32, u64), Goldilocks> = HashMap::new();

    let bundles = build_shard_witness_shared_cpu_bus::<_, (), neo_math::K, _, _, _>(
        ScriptCpu::new(),
        MapTwist::default(),
        NoopShout::default(),
        16,
        2, // chunk_size
        &mem_layouts,
        &lut_tables,
        &lut_table_specs,
        &initial_mem,
        &DummyCpuArith::default(),
    )
    .expect("chunk_size>1 should be supported");

    assert_eq!(bundles.len(), 8, "builder pads to max_steps under fixed-length semantics");
    for bundle in &bundles {
        assert_eq!(bundle.mem_instances.len(), 1);
        assert_eq!(bundle.mem_instances[0].0.steps, 2);
    }

    let inst0 = &bundles[0].mem_instances[0].0;
    let inst1 = &bundles[1].mem_instances[0].0;
    let inst2 = &bundles[2].mem_instances[0].0;
    let inst_last = &bundles.last().expect("non-empty").mem_instances[0].0;

    assert!(matches!(inst0.init, MemInit::Zero));
    assert_eq!(
        inst1.init,
        MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(5)), (1u64, Goldilocks::from_u64(7))])
    );
    assert_eq!(
        inst2.init,
        MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(9)), (1u64, Goldilocks::from_u64(7))])
    );
    assert_eq!(inst_last.init, inst2.init);
}
