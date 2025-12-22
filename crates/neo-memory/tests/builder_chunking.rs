use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_memory::builder::{build_shard_witness, CpuArithmetization, ShardBuildError};
use neo_memory::plain::PlainMemLayout;
use neo_memory::MemInit;
use neo_params::NeoParams;
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

fn dummy_commit(_mat: &Mat<Goldilocks>) -> () {}

#[test]
fn build_shard_witness_chunks_memory_and_sets_init_policy() {
    let params = NeoParams::goldilocks_127();

    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    let lut_tables: HashMap<u32, neo_memory::plain::LutTable<Goldilocks>> = HashMap::new();
    let initial_mem: HashMap<(u32, u64), Goldilocks> = HashMap::new();

    let bundles = build_shard_witness::<_, (), _, neo_math::K, _, _, _>(
        ScriptCpu::new(),
        MapTwist::default(),
        NoopShout::default(),
        16,
        2, // chunk_size
        &mem_layouts,
        &lut_tables,
        &initial_mem,
        &params,
        &dummy_commit,
        &DummyCpuArith::default(),
        None,
        0,
    )
    .expect("build_shard_witness should succeed");

    assert_eq!(bundles.len(), 2, "expected 4 steps / chunk_size=2 => 2 chunks");
    assert_eq!(bundles[0].mem_instances.len(), 1);
    assert_eq!(bundles[1].mem_instances.len(), 1);

    let inst0 = &bundles[0].mem_instances[0].0;
    let inst1 = &bundles[1].mem_instances[0].0;
    assert_eq!(inst0.steps, 2);
    assert_eq!(inst1.steps, 2);

    assert!(matches!(inst0.init, MemInit::Zero));
    assert_eq!(
        inst1.init,
        MemInit::Sparse(vec![(0u64, Goldilocks::from_u64(5)), (1u64, Goldilocks::from_u64(7))])
    );

    // Check that inc_at_write_addr encodes the correct write deltas per chunk.
    // Layout: [ra_bits, wa_bits, has_read, has_write, wv, rv, inc_at_write_addr].
    let inc0 = neo_memory::ajtai::decode_vector(&params, &bundles[0].mem_instances[0].1.mats[6]);
    assert_eq!(
        inc0,
        vec![Goldilocks::from_u64(5), Goldilocks::from_u64(7)],
        "chunk 0 inc_at_write_addr should be [5, 7]"
    );
    let inc1 = neo_memory::ajtai::decode_vector(&params, &bundles[1].mem_instances[0].1.mats[6]);
    assert_eq!(
        inc1,
        vec![Goldilocks::from_u64(4), Goldilocks::ZERO],
        "chunk 1 inc_at_write_addr should be [4, 0]"
    );
}

#[test]
fn build_shard_witness_rejects_chunk_size_that_exceeds_ccs_width() {
    let params = NeoParams::goldilocks_127();

    let mut mem_layouts = HashMap::new();
    mem_layouts.insert(0u32, PlainMemLayout { k: 2, d: 1, n_side: 2 });

    let lut_tables: HashMap<u32, neo_memory::plain::LutTable<Goldilocks>> = HashMap::new();
    let initial_mem: HashMap<(u32, u64), Goldilocks> = HashMap::new();

    // With ccs_m=4 and m_in=3, any chunk_len>=2 would overflow embed_vec.
    let err = build_shard_witness::<_, (), _, neo_math::K, _, _, _>(
        ScriptCpu::new(),
        MapTwist::default(),
        NoopShout::default(),
        16,
        2, // chunk_size
        &mem_layouts,
        &lut_tables,
        &initial_mem,
        &params,
        &dummy_commit,
        &DummyCpuArith::default(),
        Some(4),
        3,
    )
    .expect_err("expected InvalidChunkSize error");

    match err {
        ShardBuildError::InvalidChunkSize(_) => {}
        other => panic!("expected InvalidChunkSize, got {other:?}"),
    }
}
