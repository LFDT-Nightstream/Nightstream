//! Nebula lifecycle parity through the CUDA NIFS adapter.

use super::*;

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, LaneCommitments, Mat, SparsePoly};
use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::layout::StepPublicInput;
use neo_fold_clean::lifecycle::{
    self, extend_nebula_open, extend_nebula_open_with_nifs_adapter, extend_with_nifs_adapter,
    finish_uncompressed_with_audit, finish_uncompressed_with_audit_and_nifs_adapter, preprocess,
    verify_uncompressed_audit, Preprocessing, UncompressedAudit,
};
use neo_fold_clean::paper::construction2::{FoldProof, NebulaConfig, ProofState, StackShape};
use neo_fold_clean::paper::digest;

const STEPS_PER_SEGMENT: u64 = 2;
const PUBLIC_INPUT_LEN: usize = 1401;
const LANE_COLS: LaneRanges = LaneRanges {
    ops: 26..27,
    is: 27..28,
    fs: 28..29,
};
const ASSIGNMENT_LEN: usize = 29 * D;

fn lane_bits(step: u64) -> Vec<F> {
    (0..(3 * D) as u64)
        .map(|i| {
            F::from_u64(
                (step
                    .wrapping_mul(0x9E37)
                    .wrapping_add(i)
                    .rotate_left((i % 13) as u32))
                    & 1,
            )
        })
        .collect()
}

fn lane_scheme(prep: &Preprocessing) -> LaneScheme {
    LaneScheme::from_seeds(prep.params.kappa() as usize, LANE_COLS, [0xA7; 32], [0x7A; 32])
        .expect("Nebula lifecycle lane scheme")
}

fn precommit(prep: &Preprocessing, scheme: &LaneScheme) -> (Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]) {
    let mut advs = Vec::new();
    for step in 0..STEPS_PER_SEGMENT {
        let mut z = vec![F::ZERO; ASSIGNMENT_LEN];
        z[26 * D..].copy_from_slice(&lane_bits(step));
        let instance =
            CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, PUBLIC_INPUT_LEN)
                .expect("Nebula lifecycle precommit instance");
        advs.push(
            scheme
                .commit(&instance.witness.Z)
                .expect("Nebula lane commitment"),
        );
    }

    let mem = digest::nebula_chain_mem_header();
    let mut chains = [digest::nebula_chain_ops_header(), mem, mem];
    let tags: [&[u8]; 3] = [
        digest::NEBULA_CHAIN_OPS_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
    ];
    for adv in &advs {
        let leaves = digest::nebula_lane_leaf_digests(adv);
        for lane_id in 0..3 {
            chains[lane_id] = digest::nebula_chain_link(&chains[lane_id], tags[lane_id], &leaves[lane_id]);
        }
    }
    (advs, chains)
}

fn preprocessing() -> (Preprocessing, Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]) {
    let structure = CcsStructure::new(vec![Mat::identity(ASSIGNMENT_LEN)], SparsePoly::new(1, vec![]))
        .expect("Nebula lifecycle structure");
    let params = config::r1cs_params(structure.n, structure.m).expect("Nebula lifecycle params");
    install_seeded_global_pp(params.kappa() as usize, structure.m.div_ceil(D));
    let prep = preprocess(params, structure, Some(PUBLIC_INPUT_LEN)).expect("Nebula lifecycle preprocessing");
    let scheme = lane_scheme(&prep);
    let (advs, d_pre) = precommit(&prep, &scheme);
    let cfg = NebulaConfig {
        scheme,
        steps_per_segment: STEPS_PER_SEGMENT,
        seg_max: 1,
        stacks: StackShape::NONE,
        plan_digest: [F::from_u64(11); 4],
        d_init: d_pre[1],
    };
    (prep.with_nebula(cfg), advs, d_pre)
}

fn derive_gamma(prep: &Preprocessing, audit: &UncompressedAudit, d_pre: [[F; 4]; 3]) -> [K; 2] {
    let state = &audit.proof.state;
    let mut lane = state.nebula.clone().expect("Nebula lane state");
    lane.open_segment(
        prep.nebula().expect("Nebula config"),
        prep.vk.digest(),
        state.z_i,
        state.acc_digest,
        d_pre,
    )
    .expect("Nebula gamma derivation");
    lane.gamma.expect("opened segment gamma")
}

fn step_instance(prep: &Preprocessing, gamma: [K; 2], step: u64, adv: &LaneCommitments<Commitment>) -> CcsInstance {
    let x = StepPublicInput {
        seg_idx: 0,
        idx: step,
        ts_in: step,
        ts_out: step + 1,
        gamma,
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let bits = x
        .encode(StackShape::NONE)
        .expect("Nebula public input encoding");
    let mut z = vec![F::ZERO; ASSIGNMENT_LEN];
    z[0] = F::ONE;
    z[1..1 + bits.len()].copy_from_slice(&bits);
    z[26 * D..].copy_from_slice(&lane_bits(step));
    let mut instance =
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, PUBLIC_INPUT_LEN)
            .expect("Nebula lifecycle instance");
    instance.claim.adv = Some(adv.clone());
    instance
}

fn assert_fold_proofs_identical(gpu: &UncompressedAudit, cpu: &UncompressedAudit) {
    assert_eq!(gpu.steps.len(), cpu.steps.len(), "Nebula lifecycle step count");
    for (step, (gpu_step, cpu_step)) in gpu.steps.iter().zip(&cpu.steps).enumerate() {
        assert_eq!(
            gpu_step.nebula_open, cpu_step.nebula_open,
            "Nebula open payload at step {step}"
        );
        assert_eq!(
            gpu_step.semantic_state_digest, cpu_step.semantic_state_digest,
            "semantic state at step {step}"
        );
        assert_eq!(gpu_step.x_out, cpu_step.x_out, "x_out at step {step}");
        match (&gpu_step.fold, &cpu_step.fold) {
            (FoldProof::NoFold, FoldProof::NoFold) => {}
            (FoldProof::Recursive(gpu_proof), FoldProof::Recursive(cpu_proof)) => {
                super::nifs::assert_nifs_proof_identical(
                    step,
                    &gpu_proof.materialize().expect("GPU step proof"),
                    &cpu_proof.materialize().expect("CPU step proof"),
                );
            }
            _ => panic!("fold kind mismatch at lifecycle step {step}"),
        }
    }

    let gpu_final = gpu.proof.final_fold.as_ref().expect("GPU final fold");
    let cpu_final = cpu.proof.final_fold.as_ref().expect("CPU final fold");
    assert_eq!(gpu_final.x_out, cpu_final.x_out, "final x_out");
    super::nifs::assert_nifs_proof_identical(gpu.steps.len(), &gpu_final.nifs, &cpu_final.nifs);

    let gpu_state = &gpu.proof.state;
    let cpu_state = &cpu.proof.state;
    assert_eq!(gpu_state.chunk_count, cpu_state.chunk_count);
    assert_eq!(gpu_state.step_count, cpu_state.step_count);
    assert_eq!(gpu_state.z_i, cpu_state.z_i);
    assert_eq!(gpu_state.acc_digest, cpu_state.acc_digest);
    assert_eq!(gpu_state.public_trace, cpu_state.public_trace);
    assert_eq!(gpu_state.nebula, cpu_state.nebula);
    let gpu_running = gpu_state
        .proof
        .materialized_running()
        .expect("GPU running materialization")
        .expect("GPU active state");
    let cpu_running = cpu_state
        .proof
        .materialized_running()
        .expect("CPU running materialization")
        .expect("CPU active state");
    assert_eq!(gpu_running.claims, cpu_running.claims, "terminal running claims");
    assert_eq!(
        gpu_running.witnesses, cpu_running.witnesses,
        "terminal running witnesses"
    );
    assert_eq!(
        gpu_running.parent_authority, cpu_running.parent_authority,
        "terminal parent authority"
    );
    assert!(matches!(gpu_state.proof, ProofState::Active { .. }));
}

pub fn nebula_lifecycle() {
    let (prep, advs, d_pre) = preprocessing();
    let cpu = lifecycle::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("CPU Nebula base");
    let gpu = lifecycle::prove(&prep, Vec::<Vec<CcsInstance>>::new()).expect("GPU Nebula base");
    let gamma = derive_gamma(&prep, &cpu, d_pre);
    let step0 = step_instance(&prep, gamma, 0, &advs[0]);
    let step1 = step_instance(&prep, gamma, 1, &advs[1]);
    let mut cuda = CudaNifsProver::new().expect("open CUDA NIFS prover");

    let cpu = extend_nebula_open(&prep, cpu, vec![step0.clone()], d_pre).expect("CPU segment-open extend");
    let gpu = extend_nebula_open_with_nifs_adapter(&prep, &mut cuda, gpu, vec![step0], d_pre)
        .expect("GPU segment-open extend");
    let cpu = lifecycle::extend(&prep, cpu, vec![step1.clone()]).expect("CPU segment-close extend");
    let gpu = extend_with_nifs_adapter(&prep, &mut cuda, gpu, vec![step1]).expect("GPU segment-close extend");
    let cpu = finish_uncompressed_with_audit(&prep, cpu).expect("CPU Nebula final fold");
    let gpu = finish_uncompressed_with_audit_and_nifs_adapter(&prep, &mut cuda, gpu).expect("GPU Nebula final fold");

    verify_uncompressed_audit(&prep, &cpu).expect("CPU Nebula audit verification");
    verify_uncompressed_audit(&prep, &gpu).expect("GPU Nebula audit verification");
    assert_fold_proofs_identical(&gpu, &cpu);
    println!("[parity nebula_lifecycle] OK: CUDA lifecycle and CPU lifecycle are byte-identical");
}
