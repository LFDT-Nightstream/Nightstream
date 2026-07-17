//! Exact gate-family snapshots for protocol cost centers.
//!
//! Owns: pinned row-family splits whose individual families matter to the
//! challenge audit. Does not own stage hierarchy or constraint emission.
//!
//! | Snapshot | Mathematical obligation | Emits constraints? |
//! |---|---|---|
//! | PiRLC challenge | Every generic and compact acceptance row family remains visible | no |
//! | Selection binding | The three aggregate families and canonical output rows remain disjoint | no |
//! | Centered-unit children | Ordinary-private and SIS rows remain mechanically disjoint | no |
//! | PiCCS NC terminal | Equality, basis, mixing, range, and final-product source rows exactly partition the physical identity stage | no |

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    GadgetNativeEncodedRowBreakdown, GadgetNativePairTailCount, GadgetNativeStageProfile,
};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::stage as pi_ccs_stage;

const fn pair_tail(coordinates: usize, pair_rows: usize, tail_rows: usize) -> GadgetNativePairTailCount {
    GadgetNativePairTailCount {
        coordinates,
        pair_rows,
        tail_rows,
    }
}

/// Pin the exact common/canonical gate-family split at the expensive PiRLC
/// challenge phase and at its selection-binding leaf.
pub(crate) fn assert_protocol_row_family_snapshots(profile: &GadgetNativeStageProfile) {
    let challenge = profile
        .aggregate_prefix(pi_rlc_challenge_stage::CHALLENGE)
        .expect("PiRLC challenge cost center");
    assert_eq!(
        challenge.encoded_row_breakdown(),
        GadgetNativeEncodedRowBreakdown {
            common_boolean: pair_tail(23_505, 11_745, 15),
            common_centered_unit: pair_tail(318_078, 158_514, 1_050),
            ordinary_private_centered_unit: pair_tail(318_078, 158_514, 1_050),
            fallback: 1_785,
            sbox: 6_708,
            acceptance_tree_bit_pair: 6_720,
            acceptance_product_aggregate: 960,
            acceptance_root_binding: 960,
            packed_mod5_low_bit_pair: 5_760,
            packed_mod5_high_bit_pair: 960,
            packed_mod5_residue_pair: 960,
            selection_accept_aggregate: 810,
            selection_prefix_aggregate: 810,
            selection_symbol_aggregate: 810,
            ..Default::default()
        }
    );

    let selection_bind = profile
        .aggregate_prefix(pi_rlc_challenge_stage::SELECT_BIND)
        .expect("selection-bind aggregate");
    assert_eq!(
        selection_bind.encoded_row_breakdown(),
        GadgetNativeEncodedRowBreakdown {
            common_centered_unit: pair_tail(33_210, 16_200, 810),
            ordinary_private_centered_unit: pair_tail(33_210, 16_200, 810),
            selection_accept_aggregate: 810,
            selection_prefix_aggregate: 810,
            selection_symbol_aggregate: 810,
            ..Default::default()
        }
    );
}

/// Pin the exact recursive NC terminal source-row partition without creating
/// physical encoding-stage boundaries. The latter can alter low-norm pairing;
/// these assurance-only ranges must account for the existing stage exactly.
pub(crate) fn assert_pi_ccs_nc_terminal_row_families(profile: &GadgetNativeStageProfile, ranges: &[RowFamilyRange]) {
    let expected = [
        (pi_ccs_stage::ROW_NC_TERMINAL_EQUALITY_FACTORS, 1, 175),
        (pi_ccs_stage::ROW_NC_TERMINAL_CHI_ALPHA, 1, 632),
        (pi_ccs_stage::ROW_NC_TERMINAL_GAMMA_POWERS, 1, 77),
        (pi_ccs_stage::ROW_NC_TERMINAL_OUTPUT_EVALUATIONS, 15, 4_845),
        (pi_ccs_stage::ROW_NC_TERMINAL_RANGE_PRODUCTS, 15, 180),
        (pi_ccs_stage::ROW_NC_TERMINAL_WEIGHTED_SUM, 1, 78),
        (pi_ccs_stage::ROW_NC_TERMINAL_FINAL_PRODUCT, 1, 5),
    ];
    assert_eq!(
        expected.map(|(name, _, _)| name).as_slice(),
        pi_ccs_stage::ROW_NC_TERMINAL_IDENTITY_CHILDREN
    );
    assert_eq!(
        pi_ccs_stage::ROW_HIERARCHY,
        &[(
            (pi_ccs_stage::NC_TERMINAL_IDENTITY),
            pi_ccs_stage::ROW_NC_TERMINAL_IDENTITY_CHILDREN
        )]
    );

    let mut partition = Vec::new();
    for (name, occurrences, rows) in expected {
        let mut matches = ranges
            .iter()
            .filter(|range| range.name == name)
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            occurrences,
            "unexpected recursive {name} occurrence count"
        );
        let family_rows = matches
            .iter()
            .map(|range| range.row_end - range.row_start)
            .sum::<usize>();
        assert_eq!(family_rows, rows, "unexpected {name} source-row count");
        assert!(
            matches
                .iter()
                .all(|range| range.row_end - range.row_start == rows / occurrences),
            "{name} occurrences must have one fixed row shape"
        );
        partition.append(&mut matches);
    }
    partition.sort_by_key(|range| range.row_start);
    for adjacent in partition.windows(2) {
        assert_eq!(
            adjacent[0].row_end, adjacent[1].row_start,
            "NC terminal row families must be contiguous"
        );
    }

    let physical = profile
        .aggregate_prefix(pi_ccs_stage::NC_TERMINAL_IDENTITY)
        .expect("physical NC terminal identity stage");
    let partition_rows = partition
        .iter()
        .map(|range| range.row_end - range.row_start)
        .sum::<usize>();
    assert_eq!(partition_rows, physical.source_rows);
}
