#![allow(non_snake_case)]

use std::collections::HashMap;
use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{preprocess_shared_bus_r1cs, witness_layout, FoldingSession, NeoCircuit, SharedBusResources};
use neo_fold::session::{Public, Scalar, ShoutPort};
use neo_math::{D, F};
use neo_memory::cpu::ShoutCpuBinding;
use neo_memory::riscv::lookups::{compute_op, interleave_bits, uninterleave_bits, RiscvOpcode};
use neo_memory::witness::LutTableSpec;
use neo_params::NeoParams;
use neo_vm_trace::{Shout, ShoutId, StepMeta, StepTrace, Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const CHUNK_SIZE: usize = 1;
const N_STEPS: usize = CHUNK_SIZE;

witness_layout! {
    #[derive(Clone, Debug)]
    pub MultiLookupImplicitSpecCols<const N: usize> {
        pub one: Public<Scalar>,
        pub shout0_lane0: ShoutPort<N>,
        pub shout0_lane1: ShoutPort<N>,
    }
}

#[derive(Clone, Debug, Default)]
struct MultiLookupImplicitSpecCircuit<const N: usize>;

impl<const N: usize> NeoCircuit for MultiLookupImplicitSpecCircuit<N> {
    type Layout = MultiLookupImplicitSpecCols<N>;

    fn chunk_size(&self) -> usize {
        N
    }

    fn const_one_col(&self, layout: &Self::Layout) -> usize {
        layout.one
    }

    fn resources(&self, resources: &mut SharedBusResources) {
        resources
            .shout(0)
            .lanes(2)
            .spec(LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen: 32,
            });
    }

    fn cpu_bindings(
        &self,
        layout: &Self::Layout,
    ) -> Result<(HashMap<u32, Vec<ShoutCpuBinding>>, HashMap<u32, Vec<neo_memory::cpu::TwistCpuBinding>>), String> {
        Ok((
            HashMap::from([(
                0u32,
                vec![layout.shout0_lane0.cpu_binding(), layout.shout0_lane1.cpu_binding()],
            )]),
            HashMap::new(),
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
                "MultiLookupImplicitSpecCircuit witness builder expects full chunks (len {} != N {})",
                chunk.len(),
                N
            ));
        }

        let mut z = <Self::Layout as neo_fold::session::WitnessLayout>::zero_witness_prefix();
        z[layout.one] = F::ONE;

        ShoutPort::fill_lanes_from_trace(
            &[layout.shout0_lane0, layout.shout0_lane1],
            chunk,
            /*shout_id=*/ 0,
            &mut z,
        )?;

        Ok(z)
    }
}

#[derive(Clone, Debug, Default)]
struct DummyTwist;

impl Twist<u64, u64> for DummyTwist {
    fn load(&mut self, _twist_id: TwistId, _addr: u64) -> u64 {
        0
    }

    fn store(&mut self, _twist_id: TwistId, _addr: u64, _value: u64) {}
}

#[derive(Clone, Debug)]
struct RiscvOpcodeShout {
    opcode: RiscvOpcode,
    xlen: usize,
}

impl Shout<u64> for RiscvOpcodeShout {
    fn lookup(&mut self, _shout_id: ShoutId, key: u64) -> u64 {
        let (rs1, rs2) = uninterleave_bits(key as u128);
        compute_op(self.opcode, rs1, rs2, self.xlen)
    }
}

#[derive(Clone, Debug, Default)]
struct MultiLookupImplicitSpecVm {
    pc: u64,
}

impl VmCpu<u64, u64> for MultiLookupImplicitSpecVm {
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

    fn step<T, S>(&mut self, _twist: &mut T, shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        let rs1_a = 0x1234_5678u64;
        let rs2_a = 0x1111_2222u64;
        let rs1_b = 0x0bad_beefu64;
        let rs2_b = 0x00c0_ffeeu64;

        let key0 = interleave_bits(rs1_a, rs2_a) as u64;
        let key1 = interleave_bits(rs1_b, rs2_b) as u64;
        debug_assert_ne!(key0, key1);

        let _ = shout.lookup(ShoutId(0), key0);
        let _ = shout.lookup(ShoutId(0), key1);

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
fn shout_multi_lookup_implicit_table_spec_two_lookups_per_step_prove_verify() {
    let circuit = Arc::new(MultiLookupImplicitSpecCircuit::<CHUNK_SIZE>::default());
    let pre = preprocess_shared_bus_r1cs(Arc::clone(&circuit)).expect("preprocess_shared_bus_r1cs");
    let m = pre.m();

    // Use b=4 so Ajtai digit encoding can represent full Goldilocks values.
    //
    // This matters for implicit RISC-V opcode tables: interleaved (rs1,rs2) keys are 64-bit
    // and can exceed the b=2^54 representable range of the paper params.
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let params = NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        4,              // b
        base_params.k_rho, // keep preset k_rho (can be bumped in heavier tests)
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params");
    let committer = setup_ajtai_committer(m, params.kappa as usize);
    let prover = pre.into_prover(params.clone(), committer.clone()).expect("into_prover");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params.clone(), committer);
    prover
        .execute_into_session(
            &mut session,
            MultiLookupImplicitSpecVm::default(),
            DummyTwist::default(),
            RiscvOpcodeShout {
                opcode: RiscvOpcode::Add,
                xlen: 32,
            },
            N_STEPS,
        )
        .expect("execute_into_session should succeed");

    let run = session.fold_and_prove(prover.ccs()).expect("prove should succeed");
    let ok = session.verify_collected(prover.ccs(), &run).expect("verify should run");
    assert!(ok, "verification should pass");
}
