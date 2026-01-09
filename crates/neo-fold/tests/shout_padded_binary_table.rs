#![allow(non_snake_case)]

use std::collections::HashMap;
use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{preprocess_shared_bus_r1cs, witness_layout, FoldingSession, NeoCircuit, SharedBusResources};
use neo_fold::session::{Public, Scalar, ShoutPort};
use neo_math::{D, F};
use neo_memory::cpu::ShoutCpuBinding;
use neo_params::NeoParams;
use neo_vm_trace::{Shout, ShoutId, StepMeta, StepTrace, Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const CHUNK_SIZE: usize = 4;
const N_STEPS: usize = CHUNK_SIZE;

witness_layout! {
    #[derive(Clone, Debug)]
    pub PaddedBinaryTableCols<const N: usize> {
        pub one: Public<Scalar>,
        pub shout0: ShoutPort<N>,
    }
}

#[derive(Clone, Debug, Default)]
struct PaddedBinaryTableCircuit<const N: usize>;

impl<const N: usize> NeoCircuit for PaddedBinaryTableCircuit<N> {
    type Layout = PaddedBinaryTableCols<N>;

    fn chunk_size(&self) -> usize {
        N
    }

    fn const_one_col(&self, layout: &Self::Layout) -> usize {
        layout.one
    }

    fn resources(&self, resources: &mut SharedBusResources) {
        // 3-entry table gets padded to k=4 with entry 3 = 0.
        resources.shout(0).padded_binary_table(vec![F::from_u64(5), F::from_u64(7), F::from_u64(9)]);
    }

    fn cpu_bindings(
        &self,
        layout: &Self::Layout,
    ) -> Result<(HashMap<u32, Vec<ShoutCpuBinding>>, HashMap<u32, Vec<neo_memory::cpu::TwistCpuBinding>>), String> {
        Ok((HashMap::from([(0u32, vec![layout.shout0.cpu_binding()])]), HashMap::new()))
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
                "PaddedBinaryTableCircuit witness builder expects full chunks (len {} != N {})",
                chunk.len(),
                N
            ));
        }

        let mut z = <Self::Layout as neo_fold::session::WitnessLayout>::zero_witness_prefix();
        z[layout.one] = F::ONE;

        layout.shout0.fill_from_trace(chunk, /*shout_id=*/ 0, &mut z)?;
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

#[derive(Clone, Debug, Default)]
struct PaddedBinaryShoutTable {
    values: Vec<u64>,
}

impl PaddedBinaryShoutTable {
    fn new(values: Vec<u64>) -> Self {
        Self { values }
    }
}

impl Shout<u64> for PaddedBinaryShoutTable {
    fn lookup(&mut self, _shout_id: ShoutId, key: u64) -> u64 {
        self.values.get(key as usize).copied().unwrap_or(0)
    }
}

#[derive(Clone, Debug, Default)]
struct PaddedLookupVm {
    pc: u64,
    step: u64,
}

impl VmCpu<u64, u64> for PaddedLookupVm {
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
        // Cycle through 0..4 (including padded index 3).
        let k = (self.step & 3) as u64;
        let _ = shout.lookup(ShoutId(0), k);
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
fn shout_padded_binary_table_auto_params_and_prove_verify() {
    // Sanity-check the helper: content.len()=3 -> k=4, d=2, padded with 0.
    let circuit = PaddedBinaryTableCircuit::<CHUNK_SIZE>::default();
    let mut resources = SharedBusResources::default();
    circuit.resources(&mut resources);
    let table = resources.lut_tables.get(&0).expect("table_id=0 configured");
    assert_eq!(table.n_side, 2);
    assert_eq!(table.k, 4);
    assert_eq!(table.d, 2);
    assert_eq!(table.content.len(), 4);
    assert_eq!(table.content[3], F::ZERO);

    let circuit = Arc::new(circuit);
    let pre = preprocess_shared_bus_r1cs(Arc::clone(&circuit)).expect("preprocess_shared_bus_r1cs");
    let m = pre.m();

    let params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let committer = setup_ajtai_committer(m, params.kappa as usize);
    let prover = pre.into_prover(params.clone(), committer.clone()).expect("into_prover");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params.clone(), committer);
    prover
        .execute_into_session(
            &mut session,
            PaddedLookupVm::default(),
            DummyTwist::default(),
            // Match the padded table semantics: out-of-range keys return 0.
            PaddedBinaryShoutTable::new(vec![5, 7, 9]),
            N_STEPS,
        )
        .expect("execute_into_session should succeed");

    let run = session.fold_and_prove(prover.ccs()).expect("prove should succeed");
    let ok = session.verify_collected(prover.ccs(), &run).expect("verify should run");
    assert!(ok, "verification should pass");
}
