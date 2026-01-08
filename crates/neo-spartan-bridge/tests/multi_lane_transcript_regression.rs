#![allow(non_snake_case)]

//! Regression test: ensure the circuit transcript binding for Route-A memory metadata matches the native verifier
//! when Twist instances use multiple lanes.

use std::collections::HashMap;
use std::sync::Arc;

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{preprocess_shared_bus_r1cs, witness_layout, FoldingSession, NeoCircuit, SharedBusResources};
use neo_fold::session::{Public, Scalar, TwistPort};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{compute_vm_digest_v1, prove_fold_run, setup_fold_run, verify_fold_run};
use neo_vm_trace::{Shout, ShoutId, StepMeta, StepTrace, Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;

const CHUNK_SIZE: usize = 1;
const N_STEPS: usize = CHUNK_SIZE;

witness_layout! {
    #[derive(Clone, Debug)]
    pub MultiLaneCols<const N: usize> {
        pub one: Public<Scalar>,
        pub twist0_lane0: TwistPort<N>,
        pub twist0_lane1: TwistPort<N>,
    }
}

#[derive(Clone, Debug, Default)]
struct MultiLaneCircuit<const N: usize>;

impl<const N: usize> NeoCircuit for MultiLaneCircuit<N> {
    type Layout = MultiLaneCols<N>;

    fn chunk_size(&self) -> usize {
        N
    }

    fn const_one_col(&self, layout: &Self::Layout) -> usize {
        layout.one
    }

    fn resources(&self, resources: &mut SharedBusResources) {
        // Twist instance with 2 lanes (two reads/writes per VM step).
        resources.twist(0).layout(neo_memory::plain::PlainMemLayout {
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
            HashMap::from([(
                0u32,
                vec![layout.twist0_lane0.cpu_binding(), layout.twist0_lane1.cpu_binding()],
            )]),
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
                "MultiLaneCircuit witness builder expects full chunks (len {} != N {})",
                chunk.len(),
                N
            ));
        }

        let mut z = <Self::Layout as neo_fold::session::WitnessLayout>::zero_witness_prefix();
        z[layout.one] = F::ONE;

        TwistPort::fill_lanes_from_trace(&[layout.twist0_lane0, layout.twist0_lane1], chunk, 0, &mut z)?;

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
struct MultiLaneVm {
    pc: u64,
    step: u64,
}

impl VmCpu<u64, u64> for MultiLaneVm {
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

    fn step<T, S>(&mut self, twist: &mut T, shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        // Two Twist writes per step -> requires two lanes.
        twist.store(TwistId(0), 0, self.step + 1);
        twist.store(TwistId(0), 1, self.step + 2);
        let _ = shout;

        self.step += 1;
        self.pc = self.pc.wrapping_add(4);
        Ok(StepMeta { pc_after: self.pc, opcode: 0 })
    }
}

#[test]
fn test_spartan_steps_digest_matches_for_multi_lane_twist() {
    let circuit = Arc::new(MultiLaneCircuit::<CHUNK_SIZE>::default());
    let pre = preprocess_shared_bus_r1cs(Arc::clone(&circuit)).expect("preprocess_shared_bus_r1cs");
    let m = pre.m();

    let params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let seed = [42u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m, seed).expect("set_global_pp_seeded");
    let committer = AjtaiSModule::from_global_for_dims(D, m).expect("committer");

    let prover = pre.into_prover(params.clone(), committer.clone()).expect("into_prover");
    let mut session = FoldingSession::new(FoldingMode::Optimized, params.clone(), committer);

    prover
        .execute_into_session(
            &mut session,
            MultiLaneVm::default(),
            SimpleTwistMem::default(),
            DummyShout::default(),
            N_STEPS,
        )
        .expect("execute_into_session");

    let run = session.fold_and_prove(prover.ccs()).expect("fold_and_prove");

    let steps_public = session.steps_public();
    let initial_accumulator = Vec::new();
    let vm_digest = compute_vm_digest_v1(b"multi_lane_transcript_regression");
    let witness = FoldRunWitness::new(run, steps_public.clone(), initial_accumulator, vm_digest, None);

    let (pk, vk) = setup_fold_run(session.params(), prover.ccs(), &witness).expect("setup_fold_run");
    let spartan = prove_fold_run(&pk, session.params(), prover.ccs(), witness).expect("prove_fold_run");

    assert!(
        verify_fold_run(&vk, session.params(), prover.ccs(), &vm_digest, &steps_public, None, &[], &spartan)
            .expect("verify_fold_run"),
        "Spartan proof should verify for multi-lane instances"
    );
}
