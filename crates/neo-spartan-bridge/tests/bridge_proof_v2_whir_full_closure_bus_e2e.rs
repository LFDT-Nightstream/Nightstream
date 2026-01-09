#![cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::sync::Arc;

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment as Cmt, s_lincomb, s_mul};
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{preprocess_shared_bus_r1cs, witness_layout, NeoCircuit, SharedBusResources};
use neo_fold::session::{Public, Scalar, TwistPort, WitnessLayout};
use neo_fold::shard::{fold_shard_prove_with_witnesses, CommitMixers};
use neo_math::{D, F, K};
use neo_memory::builder::build_shard_witness_shared_cpu_bus_with_aux;
use neo_memory::witness::StepInstanceBundle;
use neo_params::NeoParams;
use neo_spartan_bridge::bridge_proof_v2::compute_closure_statement_v1;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{prove_fold_run, setup_fold_run, verify_bridge_proof_v2, BridgeProofV2};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_vm_trace::{Shout, ShoutId, StepMeta, StepTrace, Twist, TwistId, VmCpu};
use p3_field::PrimeCharacteristicRing;

const CHUNK_SIZE: usize = 1;
const N_STEPS: usize = CHUNK_SIZE;

witness_layout! {
    #[derive(Clone, Debug)]
    pub BusCircuitCols<const N: usize> {
        pub one: Public<Scalar>,
        pub twist0_lane0: TwistPort<N>,
    }
}

#[derive(Clone, Debug, Default)]
struct BusCircuit<const N: usize>;

impl<const N: usize> NeoCircuit for BusCircuit<N> {
    type Layout = BusCircuitCols<N>;

    fn chunk_size(&self) -> usize {
        N
    }

    fn const_one_col(&self, layout: &Self::Layout) -> usize {
        layout.one
    }

    fn resources(&self, resources: &mut SharedBusResources) {
        // One Twist instance with 1 lane.
        resources.twist(0).layout(neo_memory::plain::PlainMemLayout {
            k: 2,
            d: 1,
            n_side: 2,
            lanes: 1,
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
            HashMap::from([(0u32, vec![layout.twist0_lane0.cpu_binding()])]),
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
            return Err(format!("BusCircuit witness builder expects chunk len {N}, got {}", chunk.len()));
        }

        let mut z = <Self::Layout as WitnessLayout>::zero_witness_prefix();
        z[layout.one] = F::ONE;

        TwistPort::fill_lanes_from_trace(&[layout.twist0_lane0], chunk, 0, &mut z)?;
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
struct OneWriteVm {
    pc: u64,
    step: u64,
}

impl VmCpu<u64, u64> for OneWriteVm {
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
        twist.store(TwistId(0), 0, self.step + 1);
        let _ = shout;
        self.step += 1;
        self.pc = self.pc.wrapping_add(4);
        Ok(StepMeta { pc_after: self.pc, opcode: 0 })
    }
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> neo_math::Rq {
    use neo_math::cf_inv;

    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<neo_math::Rq> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }

    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = neo_math::Rq::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }

    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

#[test]
fn bridge_proof_v2_whir_full_closure_bus_roundtrip() {
    let circuit = Arc::new(BusCircuit::<CHUNK_SIZE>::default());
    let pre = preprocess_shared_bus_r1cs(Arc::clone(&circuit)).expect("preprocess_shared_bus_r1cs");
    let m = pre.m();

    let params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let seed = [42u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m, seed).expect("set_global_pp_seeded");
    let committer = AjtaiSModule::from_global_for_dims(D, m).expect("committer");

    let prover = pre
        .into_prover(params.clone(), committer.clone())
        .expect("into_prover");

    // Build 1-step shard witness with shared CPU bus + Twist instance.
    let (steps_witness, _aux) = build_shard_witness_shared_cpu_bus_with_aux(
        OneWriteVm::default(),
        SimpleTwistMem::default(),
        DummyShout::default(),
        N_STEPS,
        CHUNK_SIZE,
        &prover.resources.mem_layouts,
        &prover.resources.lut_tables,
        &prover.resources.lut_table_specs,
        &prover.resources.lut_lanes,
        &HashMap::new(),
        &prover.cpu,
    )
    .expect("build_shard_witness_shared_cpu_bus_with_aux");

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    // Prove with witnesses to obtain final obligations + Z witnesses.
    let mode = FoldingMode::Optimized;
    let mixers = mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"neo.fold/session");
    let (fold_run, outputs, wits) = fold_shard_prove_with_witnesses(
        mode,
        &mut tr_prove,
        &params,
        prover.ccs(),
        &steps_witness,
        &[],
        &[],
        &committer,
        mixers,
    )
    .expect("fold_shard_prove_with_witnesses");

    let core_t = prover.ccs().t();
    assert!(
        outputs
            .obligations
            .main
            .iter()
            .chain(outputs.obligations.val.iter())
            .any(|me| me.y.len() > core_t),
        "expected bus openings (me.y.len() > core_t) in at least one obligation"
    );

    // Spartan bridge proof (Phase 1).
    let vm_digest = [0u8; 32];
    let witness = FoldRunWitness::new(fold_run, steps_public.clone(), vec![], vm_digest, None);
    let (pk, vk) = setup_fold_run(&params, prover.ccs(), &witness).expect("setup_fold_run");
    let spartan = prove_fold_run(&pk, &params, prover.ccs(), witness).expect("prove_fold_run");

    // Derive a bus layout consistent with the public step instance(s) (same inputs as verifier).
    let first = steps_public.first().expect("step0");
    assert!(
        !first.mem_insts.is_empty(),
        "expected >=1 Twist instance in public step"
    );
    let twist_ell_addrs_and_lanes = first
        .mem_insts
        .iter()
        .map(|inst| (inst.d * inst.ell, inst.lanes));
    let shout_ell_addrs_and_lanes = first
        .lut_insts
        .iter()
        .map(|inst| (inst.d * inst.ell, inst.lanes));
    let chunk_size = first
        .mem_insts
        .first()
        .map(|inst| inst.steps)
        .unwrap_or(CHUNK_SIZE);
    assert_eq!(chunk_size, CHUNK_SIZE);

    let bus = neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes(
        prover.ccs().m,
        first.mcs_inst.m_in,
        chunk_size,
        shout_ell_addrs_and_lanes,
        twist_ell_addrs_and_lanes,
    )
    .expect("build bus layout");
    assert!(bus.bus_cols > 0, "expected nonzero bus cols");

    // Phase-2 closure proof (WHIR full closure, includes bus semantics).
    let closure_stmt = compute_closure_statement_v1(&spartan.statement);
    let closure = neo_closure_proof::prove_whir_p3_full_closure_v1(
        &closure_stmt,
        &params,
        prover.ccs(),
        &outputs.obligations,
        &wits.final_main_wits,
        &wits.val_lane_wits,
        Some(&bus),
    )
    .expect("prove whir full closure");

    let proof = BridgeProofV2 {
        spartan: spartan.clone(),
        closure_stmt,
        closure,
    };

    let ok = verify_bridge_proof_v2(
        &vk,
        &params,
        prover.ccs(),
        &vm_digest,
        &steps_public,
        None,
        &[],
        &proof,
    )
    .expect("verify");
    assert!(ok, "BridgeProofV2 must verify with bus-enabled obligations");
}
