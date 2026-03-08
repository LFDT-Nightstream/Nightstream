#![allow(non_snake_case)]

#[path = "../../common/twist_low_level_fixtures.rs"]
mod twist_low_level_fixtures;

use neo_ajtai::Commitment as Cmt;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{F, K};
use neo_memory::plain::PlainMemLayout;
use neo_memory::plain::PlainMemTrace;
use neo_memory::witness::StepInstanceBundle;
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

#[test]
fn twist_lane_pinning_allows_writing_lane1_without_lane0() {
    let t = 2usize;
    let mem_layout = PlainMemLayout {
        k: 4,
        d: 2,
        n_side: 2,
        lanes: 2,
    };
    let bus_cols = mem_layout.lanes * (2 * mem_layout.d * 1 + 5);
    let ccs = twist_low_level_fixtures::create_identity_ccs(bus_cols * t);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;
    let l = twist_low_level_fixtures::setup_ajtai_committer(ccs.m, params.kappa as usize);

    let (mem_inst, mem_wit) = twist_low_level_fixtures::make_twist_instance(0, &mem_layout, MemInit::Zero, t);
    let step = twist_low_level_fixtures::create_step_with_bus(
        &params,
        &ccs,
        &l,
        0,
        vec![],
        vec![],
        vec![],
        vec![(
            mem_inst,
            mem_wit,
            vec![
                PlainMemTrace {
                    steps: t,
                    has_read: vec![F::ZERO; t],
                    has_write: vec![F::ZERO, F::ONE],
                    read_addr: vec![0; t],
                    write_addr: vec![0, 0],
                    read_val: vec![F::ZERO; t],
                    write_val: vec![F::ZERO, F::from_u64(3)],
                    inc_at_write_addr: vec![F::ZERO, F::from_u64(3)],
                },
                PlainMemTrace {
                    steps: t,
                    has_read: vec![F::ZERO; t],
                    has_write: vec![F::ONE, F::ZERO],
                    read_addr: vec![0; t],
                    write_addr: vec![1, 0],
                    read_val: vec![F::ZERO; t],
                    write_val: vec![F::from_u64(2), F::ZERO],
                    inc_at_write_addr: vec![F::from_u64(2), F::ZERO],
                },
            ],
        )],
    );

    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let mixers = crate::common_setup::default_mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"twist/lane_pinning");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"twist/lane_pinning");
    let outputs = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
    )
    .expect("verify should succeed");

    assert!(
        !outputs.obligations.val.is_empty(),
        "twist proving should emit val-lane obligations"
    );
}
