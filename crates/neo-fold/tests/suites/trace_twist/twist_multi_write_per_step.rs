#![allow(non_snake_case)]

#[path = "../../common/twist_low_level_fixtures.rs"]
mod twist_low_level_fixtures;

use neo_ajtai::Commitment as Cmt;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{witness_layout, TwistPortWithInc, WitnessLayout};
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{F, K};
use neo_memory::plain::{PlainMemLayout, PlainMemTrace};
use neo_memory::witness::StepInstanceBundle;
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_vm_trace::{StepTrace, TwistEvent, TwistId, TwistOpKind};
use p3_field::PrimeCharacteristicRing;

witness_layout! {
    #[derive(Clone, Debug)]
    pub MultiWriteCols<const N: usize> {
        pub twist0_lane0: TwistPortWithInc<N>,
        pub twist0_lane1: TwistPortWithInc<N>,
    }
}

#[test]
fn twist_multi_write_two_writes_per_step_prove_verify() {
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
                    has_write: vec![F::ONE; t],
                    read_addr: vec![0; t],
                    write_addr: vec![0; t],
                    read_val: vec![F::ZERO; t],
                    write_val: vec![F::ONE, F::from_u64(2)],
                    inc_at_write_addr: vec![F::ONE, F::ONE],
                },
                PlainMemTrace {
                    steps: t,
                    has_read: vec![F::ZERO; t],
                    has_write: vec![F::ONE; t],
                    read_addr: vec![0; t],
                    write_addr: vec![1; t],
                    read_val: vec![F::ZERO; t],
                    write_val: vec![F::from_u64(2), F::from_u64(3)],
                    inc_at_write_addr: vec![F::from_u64(2), F::ONE],
                },
            ],
        )],
    );

    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();
    let mixers = crate::common_setup::default_mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"twist/multi_write");
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

    let mut tr_verify = Poseidon2Transcript::new(b"twist/multi_write");
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

#[test]
fn twist_multi_write_duplicate_addr_rejected_by_lane_filler() {
    const N: usize = 2;
    let layout = <MultiWriteCols<N> as WitnessLayout>::new_layout();
    let mut z = <MultiWriteCols<N> as WitnessLayout>::zero_witness_prefix();

    let chunk: Vec<StepTrace<u64, u64, u128>> = vec![
        StepTrace {
            cycle: 0,
            halted: false,
            pc_before: 0,
            pc_after: 4,
            opcode: 0,
            is_virtual: false,
            virtual_sequence_remaining: None,
            regs_before: Vec::new(),
            regs_after: Vec::new(),
            twist_events: vec![
                TwistEvent {
                    twist_id: TwistId(0),
                    kind: TwistOpKind::Write,
                    addr: 0,
                    value: 1,
                    lane: None,
                },
                TwistEvent {
                    twist_id: TwistId(0),
                    kind: TwistOpKind::Write,
                    addr: 0,
                    value: 2,
                    lane: None,
                },
            ],
            shout_events: Vec::new(),
        },
        StepTrace {
            cycle: 1,
            halted: false,
            pc_before: 4,
            pc_after: 8,
            opcode: 0,
            is_virtual: false,
            virtual_sequence_remaining: None,
            regs_before: Vec::new(),
            regs_after: Vec::new(),
            twist_events: Vec::new(),
            shout_events: Vec::new(),
        },
    ];

    let err = TwistPortWithInc::fill_lanes_from_trace(&[layout.twist0_lane0, layout.twist0_lane1], &chunk, 0, &mut z)
        .expect_err("duplicate addr must fail");
    assert!(err.contains("duplicate twist write addr"), "unexpected error: {err}");
}
