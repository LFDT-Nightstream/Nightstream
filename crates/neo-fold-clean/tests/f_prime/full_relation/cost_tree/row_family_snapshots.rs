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

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    GadgetNativeEncodedRowBreakdown, GadgetNativePairTailCount, GadgetNativeStageProfile,
};

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
