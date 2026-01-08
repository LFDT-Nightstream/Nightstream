#![allow(non_snake_case)]

use std::collections::HashMap;
use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{preprocess_shared_bus_r1cs, witness_layout, FoldingSession, NeoCircuit, SharedBusResources, TwistPort};
use neo_fold::session::{Public, Scalar};
use neo_fold::shard::StepLinkingConfig;
use neo_math::{D, F};
use neo_memory::plain::PlainMemLayout;
use neo_params::NeoParams;
use neo_vm_trace::{Shout, ShoutId, StepMeta, StepTrace, Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const CHUNK_SIZE: usize = 2;
const N_STEPS: usize = CHUNK_SIZE * 2;

witness_layout! {
    #[derive(Clone, Debug)]
    pub MultiWriteCols<const N: usize> {
        pub one: Public<Scalar>,
        pub twist0_lane0: TwistPort<N>,
        pub twist0_lane1: TwistPort<N>,
    }
}

#[derive(Clone, Debug, Default)]
struct MultiWriteCircuit<const N: usize>;

impl<const N: usize> NeoCircuit for MultiWriteCircuit<N> {
    type Layout = MultiWriteCols<N>;

    fn chunk_size(&self) -> usize {
        N
    }

    fn const_one_col(&self, layout: &Self::Layout) -> usize {
        layout.one
    }

    fn resources(&self, resources: &mut SharedBusResources) {
        resources
            .twist(0)
            .layout(PlainMemLayout {
                k: 4,
                d: 2,
                n_side: 2,
                lanes: 2,
            });
    }

    fn cpu_bindings(
        &self,
        layout: &Self::Layout,
    ) -> Result<
        (
            HashMap<u32, Vec<neo_memory::cpu::ShoutCpuBinding>>,
            HashMap<u32, Vec<neo_memory::cpu::TwistCpuBinding>>,
        ),
        String,
    > {
        Ok((
            HashMap::new(),
            HashMap::from([(0u32, vec![layout.twist0_lane0.cpu_binding(), layout.twist0_lane1.cpu_binding()])]),
        ))
    }

    fn define_cpu_constraints(
        &self,
        cs: &mut neo_fold::session::CcsBuilder<F>,
        layout: &Self::Layout,
    ) -> Result<(), String> {
        cs.eq(layout.one, layout.one);
        Ok(())
    }

    fn build_witness_prefix(&self, layout: &Self::Layout, chunk: &[StepTrace<u64, u64>]) -> Result<Vec<F>, String> {
        if chunk.len() != N {
            return Err(format!(
                "MultiWriteCircuit witness builder expects full chunks (len {} != N {})",
                chunk.len(),
                N
            ));
        }

        let mut z = <Self::Layout as neo_fold::session::WitnessLayout>::zero_witness_prefix();
        z[layout.one] = F::ONE;

        TwistPort::fill_lanes_from_trace(
            &[layout.twist0_lane0, layout.twist0_lane1],
            chunk,
            0,
            &mut z,
        )?;

        Ok(z)
    }
}

#[derive(Clone, Debug, Default)]
struct SimpleTwistMem {
    data: HashMap<(TwistId, u64), u64>,
}

impl Twist<u64, u64> for SimpleTwistMem {
    fn load(&mut self, twist_id: TwistId, addr: u64) -> u64 {
        self.data.get(&(twist_id, addr)).copied().unwrap_or(0)
    }

    fn store(&mut self, twist_id: TwistId, addr: u64, value: u64) {
        self.data.insert((twist_id, addr), value);
    }
}

#[derive(Clone, Debug, Default)]
struct DummyShout;

impl Shout<u64> for DummyShout {
    fn lookup(&mut self, _shout_id: ShoutId, _key: u64) -> u64 {
        0
    }
}

#[derive(Clone, Debug, Default)]
struct MultiWriteVm {
    pc: u64,
    step: u64,
}

impl VmCpu<u64, u64> for MultiWriteVm {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        Vec::new()
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        false
    }

    fn step<T, S>(&mut self, twist: &mut T, _shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        twist.store(TwistId(0), 0, self.step + 1);
        twist.store(TwistId(0), 1, self.step + 2);
        self.step += 1;

        self.pc = self.pc.wrapping_add(4);
        Ok(StepMeta { pc_after: self.pc, opcode: 0 })
    }
}

fn setup_ajtai_committer(m: usize, kappa: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, kappa, m).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(pp))
}

#[test]
fn twist_multi_write_two_writes_per_step_prove_verify() {
    let circuit = Arc::new(MultiWriteCircuit::<CHUNK_SIZE>::default());
    let pre = preprocess_shared_bus_r1cs(Arc::clone(&circuit)).expect("preprocess_shared_bus_r1cs");
    let m = pre.m();

    let params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let committer = setup_ajtai_committer(m, params.kappa as usize);
    let prover = pre.into_prover(params.clone(), committer.clone()).expect("into_prover");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params.clone(), committer);
    prover
        .execute_into_session(
            &mut session,
            MultiWriteVm::default(),
            SimpleTwistMem::default(),
            DummyShout::default(),
            N_STEPS,
        )
        .expect("execute_into_session should succeed");

    let run = session.fold_and_prove(prover.ccs()).expect("prove should succeed");
    session.set_step_linking(StepLinkingConfig::new(vec![(0, 0)]));
    let ok = session.verify_collected(prover.ccs(), &run).expect("verify should run");
    assert!(ok, "verification should pass");
}

#[test]
fn twist_multi_write_duplicate_addr_rejected_by_lane_filler() {
    let layout = <MultiWriteCols<CHUNK_SIZE> as neo_fold::session::WitnessLayout>::new_layout();
    let mut z = <MultiWriteCols<CHUNK_SIZE> as neo_fold::session::WitnessLayout>::zero_witness_prefix();

    let mut chunk: Vec<StepTrace<u64, u64>> = Vec::with_capacity(CHUNK_SIZE);
    chunk.push(StepTrace {
        cycle: 0,
        pc_before: 0,
        pc_after: 4,
        opcode: 0,
        regs_before: Vec::new(),
        regs_after: Vec::new(),
        twist_events: vec![
            neo_vm_trace::TwistEvent {
                twist_id: TwistId(0),
                kind: neo_vm_trace::TwistOpKind::Write,
                addr: 0,
                value: 1,
            },
            neo_vm_trace::TwistEvent {
                twist_id: TwistId(0),
                kind: neo_vm_trace::TwistOpKind::Write,
                addr: 0,
                value: 2,
            },
        ],
        shout_events: Vec::new(),
        halted: false,
    });
    for i in 1..CHUNK_SIZE {
        chunk.push(StepTrace {
            cycle: i as u64,
            pc_before: 4 * i as u64,
            pc_after: 4 * (i as u64 + 1),
            opcode: 0,
            regs_before: Vec::new(),
            regs_after: Vec::new(),
            twist_events: Vec::new(),
            shout_events: Vec::new(),
            halted: false,
        });
    }

    let err = TwistPort::fill_lanes_from_trace(
        &[layout.twist0_lane0, layout.twist0_lane1],
        &chunk,
        0,
        &mut z,
    )
    .expect_err("expected duplicate write addr error");
    assert!(err.contains("duplicate twist write addr"), "unexpected error: {err}");
}
