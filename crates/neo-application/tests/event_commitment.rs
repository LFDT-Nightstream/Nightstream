use neo_application::event_commitment::{commit_block, fold_blocks, EventCommitmentState};
use neo_application::poseidon2::{
    apply_full_round, apply_initial_linear, apply_partial_pair, external_linear, full_round_constants, sbox7,
};
use neo_application::{ConstraintTag, Pow7};
use neo_application::{
    EventCommitment, GadgetDescriptor, Poseidon2FullRound12, Poseidon2FullRoundChoice, Poseidon2PartialPair12,
    Poseidon2PartialPairChoice, R1csBuilder, EVENT_COMMITMENT_AUX_COLUMNS,
};
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ONE: usize = 0;

fn assert_satisfied(relation: &neo_application::R1csRelation<&'static str>, assignment: &[F]) {
    check_ccs_rowwise_zero(relation.structure(), &assignment[..1], &assignment[1..])
        .expect("gadget assignment must satisfy the relation");
}

#[test]
fn native_commitment_retains_the_cross_repository_fixture() {
    let f = F::from_u64;
    let previous = [F::ZERO; 4];
    let block = [f(1), f(1), f(2), f(3), f(4), f(5), f(6), f(7)];
    let first = commit_block(previous, block);
    assert_eq!(
        first,
        [
            f(16060384774117980274),
            f(6217562501851223455),
            f(9809238410420041413),
            f(4191298748431046296),
        ]
    );

    let second_block = [
        f(16),
        f(0xffff_ffff),
        f(0xffff_ffff_0000_0000),
        f(0),
        f(42),
        f(7),
        f(0),
        f(1),
    ];
    let second = commit_block(first, second_block);
    assert_eq!(
        second,
        [
            f(2581777910110991851),
            f(4248944502313846729),
            f(3337412769805346927),
            f(12455009736376722043),
        ]
    );
    assert_eq!(
        fold_blocks(EventCommitmentState::default(), &[block, second_block]).into_lanes(),
        second
    );
}

#[test]
fn pow7_assigns_all_powers_and_rejects_tampering() {
    let gadget = Pow7 {
        expression: [(1, F::ONE)],
        powers: [2, 3, 4, 5],
    };
    let mut builder = R1csBuilder::new(6, 1, ONE).unwrap();
    gadget.push_constraints(&mut builder.tagged(ConstraintTag::new("x^7", "test")));
    let relation = builder.build().unwrap();
    let mut assignment = vec![F::ONE, F::from_u64(9), F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    gadget.assign(&mut assignment);
    assert_satisfied(&relation, &assignment);
    assignment[5] += F::ONE;
    assert!(check_ccs_rowwise_zero(relation.structure(), &assignment[..1], &assignment[1..]).is_err());
}

#[test]
fn selectable_rounds_match_the_native_round_functions() {
    const SELECT_FULL: usize = 1;
    const SELECT_OTHER_FULL: usize = 2;
    const SELECT_PARTIAL: usize = 3;
    const STATE_BEFORE: usize = 4;
    const FULL_AFTER: usize = STATE_BEFORE + 12;
    const PARTIAL_AFTER: usize = FULL_AFTER + 12;
    const FULL_POWERS: usize = PARTIAL_AFTER + 12;
    const PARTIAL_POWERS: usize = FULL_POWERS + 48;
    const WIDTH: usize = PARTIAL_POWERS + 8;

    let before = core::array::from_fn(|lane| STATE_BEFORE + lane);
    let full_after = core::array::from_fn(|lane| FULL_AFTER + lane);
    let partial_after = core::array::from_fn(|lane| PARTIAL_AFTER + lane);
    let full = Poseidon2FullRound12 {
        choices: [
            Poseidon2FullRoundChoice::for_round(SELECT_FULL, 0),
            Poseidon2FullRoundChoice::for_round(SELECT_OTHER_FULL, 7),
        ],
        state_before: before,
        state_after: full_after,
        powers: core::array::from_fn(|lane| core::array::from_fn(|power| FULL_POWERS + 4 * lane + power)),
    };
    let partial = Poseidon2PartialPair12 {
        choices: [Poseidon2PartialPairChoice::for_pair(SELECT_PARTIAL, 5)],
        state_before: before,
        state_after: partial_after,
        powers: core::array::from_fn(|offset| PARTIAL_POWERS + offset),
    };
    let mut builder = R1csBuilder::new(WIDTH, 1, ONE).unwrap();
    let mut tagged = builder.tagged(ConstraintTag::new("round", "test"));
    full.push_constraints(&mut tagged);
    partial.push_constraints(&mut tagged);
    let relation = builder.build().unwrap();

    let mut assignment = vec![F::ZERO; WIDTH];
    assignment[ONE] = F::ONE;
    assignment[SELECT_FULL] = F::ONE;
    assignment[SELECT_PARTIAL] = F::ONE;
    for lane in 0..12 {
        assignment[before[lane]] = F::from_u64(100 + lane as u64);
    }
    let mut expected_full = before.map(|column| assignment[column]);
    apply_full_round(0, &mut expected_full);
    let mut expected_partial = before.map(|column| assignment[column]);
    apply_partial_pair(5, &mut expected_partial);
    for lane in 0..12 {
        assignment[full_after[lane]] = expected_full[lane];
        assignment[partial_after[lane]] = expected_partial[lane];
    }
    full.assign_auxiliaries(&mut assignment);
    partial.assign_auxiliaries(&mut assignment);
    assert_satisfied(&relation, &assignment);

    assert_eq!(full_after.map(|column| assignment[column]), expected_full);
    assert_eq!(partial_after.map(|column| assignment[column]), expected_partial);

    let mut inactive = vec![F::ZERO; WIDTH];
    inactive[ONE] = F::ONE;
    for lane in 0..12 {
        inactive[before[lane]] = F::from_u64(200 + lane as u64);
        inactive[full_after[lane]] = F::from_u64(300 + lane as u64);
        inactive[partial_after[lane]] = F::from_u64(400 + lane as u64);
    }
    let full_output = full_after.map(|column| inactive[column]);
    let partial_output = partial_after.map(|column| inactive[column]);
    full.assign_auxiliaries(&mut inactive);
    partial.assign_auxiliaries(&mut inactive);
    assert_satisfied(&relation, &inactive);
    assert_eq!(full_after.map(|column| inactive[column]), full_output);
    assert_eq!(partial_after.map(|column| inactive[column]), partial_output);

    // The selectable round equations alone admit the sum of two round
    // constants. The default API's selector rows must reject that assignment.
    let mut ambiguous = vec![F::ZERO; WIDTH];
    ambiguous[ONE] = F::ONE;
    ambiguous[SELECT_FULL] = F::ONE;
    ambiguous[SELECT_OTHER_FULL] = F::ONE;
    for lane in 0..12 {
        ambiguous[before[lane]] = F::from_u64(500 + lane as u64);
    }
    let mut blended = core::array::from_fn(|lane| {
        sbox7(ambiguous[before[lane]] + full_round_constants(0)[lane] + full_round_constants(7)[lane])
    });
    external_linear(&mut blended);
    for lane in 0..12 {
        ambiguous[full_after[lane]] = blended[lane];
    }
    full.assign_auxiliaries(&mut ambiguous);
    partial.assign_auxiliaries(&mut ambiguous);
    assert!(check_ccs_rowwise_zero(relation.structure(), &ambiguous[..1], &ambiguous[1..]).is_err());
}

#[test]
fn unrolled_commitment_matches_native_and_retains_one_semantic_occurrence() {
    const PREVIOUS: usize = 1;
    const BLOCK: usize = PREVIOUS + 4;
    const OUTPUT: usize = BLOCK + 8;
    const AUXILIARY: usize = OUTPUT + 4;
    const WIDTH: usize = AUXILIARY + EVENT_COMMITMENT_AUX_COLUMNS;

    let gadget = EventCommitment {
        previous: core::array::from_fn(|lane| PREVIOUS + lane),
        block: core::array::from_fn(|word| BLOCK + word),
        output: core::array::from_fn(|lane| OUTPUT + lane),
        auxiliary_start: AUXILIARY,
    };
    let mut builder = R1csBuilder::new(WIDTH, 1, ONE).unwrap();
    gadget.push_constraints(&mut builder.tagged(ConstraintTag::new("event block", "test")));
    let relation = builder.build().unwrap();

    assert_eq!(gadget.auxiliary_range(), AUXILIARY..WIDTH);
    // 12 premix + 8 × (48 S-box + 12 output) + 11 × (8 S-box + 12 output)
    // + 4 feed-forward rows.
    assert_eq!(relation.structure().n, 716);
    let [occurrence] = relation.catalog().gadget_occurrences() else {
        panic!("the composite must retain one semantic occurrence, not its internal rounds")
    };
    assert!(matches!(
        occurrence.descriptor(),
        GadgetDescriptor::EventCommitment { .. }
    ));

    let mut assignment = vec![F::ZERO; WIDTH];
    assignment[ONE] = F::ONE;
    for lane in 0..4 {
        assignment[PREVIOUS + lane] = F::from_u64(20 + lane as u64);
    }
    for word in 0..8 {
        assignment[BLOCK + word] = F::from_u64(100 + word as u64);
    }
    let expected = commit_block(
        core::array::from_fn(|lane| assignment[PREVIOUS + lane]),
        core::array::from_fn(|word| assignment[BLOCK + word]),
    );
    for lane in 0..4 {
        assignment[OUTPUT + lane] = expected[lane];
    }
    gadget.assign_auxiliaries(&mut assignment);
    assert_satisfied(&relation, &assignment);

    assert_eq!(core::array::from_fn(|lane| assignment[OUTPUT + lane]), expected);

    let mut tampered_auxiliary = assignment.clone();
    tampered_auxiliary[AUXILIARY + 12] += F::ONE;
    assert!(check_ccs_rowwise_zero(relation.structure(), &tampered_auxiliary[..1], &tampered_auxiliary[1..]).is_err());

    let mut tampered_output = assignment;
    tampered_output[OUTPUT] += F::ONE;
    assert!(check_ccs_rowwise_zero(relation.structure(), &tampered_output[..1], &tampered_output[1..]).is_err());
}

#[test]
fn grouped_round_order_matches_the_native_permutation() {
    let previous = core::array::from_fn(|lane| F::from_u64(10 + lane as u64));
    let block = core::array::from_fn(|word| F::from_u64(50 + word as u64));
    let mut state = core::array::from_fn(|lane| if lane < 4 { previous[lane] } else { block[lane - 4] });
    apply_initial_linear(&mut state);
    for round in 0..4 {
        apply_full_round(round, &mut state);
    }
    for pair in 0..11 {
        apply_partial_pair(pair, &mut state);
    }
    for round in 4..8 {
        apply_full_round(round, &mut state);
    }
    let compressed: [F; 4] = core::array::from_fn(|lane| state[lane] + previous[lane]);
    assert_eq!(compressed, commit_block(previous, block));
}
