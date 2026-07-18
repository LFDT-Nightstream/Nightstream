//! Exact fixed-profile cost checks for Π_RLC projection identities.
//!
//! Owns: the six arithmetic phase formulas, claim-family multiplicities,
//! per-`y_zcol`-limb checks, and the fixed 31-identity census.
//!
//! Does not own: constraint emission, stage hierarchy, shared beta/rho costs,
//! padding, or semantic/refinement claims.
//!
//! Emits constraints: no.
//!
//! | Cost owner | Mathematical obligation | Multiplicity | Production owner | Lean owner |
//! |---|---|---:|---|---|
//! | Input evaluations | Evaluate 15 input polynomials in two extension-field limbs | 31 | `ring_action::enforce_eval_at_beta` | bounded evaluation refinement |
//! | Output evaluation | Evaluate one output polynomial in two limbs | 31 | `ring_action::enforce_eval_at_beta` | bounded evaluation refinement |
//! | Quotient evaluation | Evaluate one quotient polynomial in two limbs | 31 | `ring_action::enforce_eval_at_beta` | bounded evaluation refinement |
//! | `rho * input` products | Form the 15 weighted inputs | 31 | `field_ext::enforce_k_mul` | exact Karatsuba refinement |
//! | `quotient * Phi81` product | Form the vanishing-polynomial correction | 31 | `field_ext::enforce_k_mul` | exact Karatsuba refinement |
//! | Final limb checks | Enforce the two extension-field identity limbs | 31 | `ring_action` | exact reduction or `BatchBadRoot` |

use neo_fold_clean::frontends::f_prime::gadget_native::{
    GadgetNativeEncodedRowBreakdown, GadgetNativeStageProfile, ORDINARY_PRIVATE_DIGITS,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

use super::{repeated_canonical_field_rows, repeated_pair_tail};

#[derive(Clone, Copy)]
struct IdentityCostFormula {
    source_rows: usize,
    source_cols: usize,
    canonical_fields: usize,
    ordinary_private_fields: usize,
    linear_fields: usize,
    gadget_fields: usize,
    synthetic_product_sum_fields: usize,
    encoded_rows: usize,
    encoded_cols: usize,
    fallback_rows: usize,
    product_sum_batches: usize,
    product_sum_identities: usize,
    product_sum_rows: usize,
}

const INPUT_EVALUATIONS: IdentityCostFormula = IdentityCostFormula {
    source_rows: 1_620,
    source_cols: 1_620,
    canonical_fields: 0,
    ordinary_private_fields: 30,
    linear_fields: 0,
    gadget_fields: 1_590,
    synthetic_product_sum_fields: 60,
    encoded_rows: 4_515,
    encoded_cols: 6_930,
    fallback_rows: 0,
    product_sum_batches: 1,
    product_sum_identities: 30,
    product_sum_rows: 90,
};
const OUTPUT_EVALUATION: IdentityCostFormula = IdentityCostFormula {
    source_rows: 108,
    source_cols: 108,
    canonical_fields: 0,
    ordinary_private_fields: 2,
    linear_fields: 0,
    gadget_fields: 106,
    synthetic_product_sum_fields: 4,
    encoded_rows: 301,
    encoded_cols: 462,
    fallback_rows: 0,
    product_sum_batches: 0,
    product_sum_identities: 2,
    product_sum_rows: 6,
};
const QUOTIENT_EVALUATION: IdentityCostFormula = IdentityCostFormula {
    source_rows: 106,
    source_cols: 106,
    canonical_fields: 0,
    ordinary_private_fields: 2,
    linear_fields: 0,
    gadget_fields: 104,
    synthetic_product_sum_fields: 4,
    encoded_rows: 301,
    encoded_cols: 462,
    fallback_rows: 0,
    product_sum_batches: 0,
    product_sum_identities: 2,
    product_sum_rows: 6,
};
const RHO_TIMES_INPUT: IdentityCostFormula = IdentityCostFormula {
    source_rows: 75,
    source_cols: 75,
    canonical_fields: 0,
    ordinary_private_fields: 0,
    linear_fields: 0,
    gadget_fields: 75,
    synthetic_product_sum_fields: 0,
    encoded_rows: 0,
    encoded_cols: 0,
    fallback_rows: 0,
    product_sum_batches: 0,
    product_sum_identities: 0,
    product_sum_rows: 0,
};
const QUOTIENT_TIMES_PHI: IdentityCostFormula = IdentityCostFormula {
    source_rows: 5,
    source_cols: 5,
    canonical_fields: 0,
    ordinary_private_fields: 0,
    linear_fields: 0,
    gadget_fields: 5,
    synthetic_product_sum_fields: 0,
    encoded_rows: 0,
    encoded_cols: 0,
    fallback_rows: 0,
    product_sum_batches: 0,
    product_sum_identities: 0,
    product_sum_rows: 0,
};
const FINAL_LIMB_CHECKS: IdentityCostFormula = IdentityCostFormula {
    source_rows: 2,
    source_cols: 0,
    canonical_fields: 0,
    ordinary_private_fields: 0,
    linear_fields: 0,
    gadget_fields: 0,
    synthetic_product_sum_fields: 2,
    encoded_rows: 131,
    encoded_cols: 190,
    fallback_rows: 0,
    product_sum_batches: 0,
    product_sum_identities: 2,
    product_sum_rows: 4,
};
const IDENTITY_TOTAL: IdentityCostFormula = IdentityCostFormula {
    source_rows: 1_916,
    source_cols: 1_914,
    canonical_fields: 0,
    ordinary_private_fields: 34,
    linear_fields: 0,
    gadget_fields: 1_880,
    synthetic_product_sum_fields: 70,
    encoded_rows: 5_248,
    encoded_cols: 8_044,
    fallback_rows: 0,
    product_sum_batches: 1,
    product_sum_identities: 36,
    product_sum_rows: 106,
};

/// Pin the six arithmetic leaves repeated by every production projection
/// identity. The claim-family count is the only multiplier: commitment 18,
/// X 5, y_ring 6, y_zcol 2, and absent Nebula advice 0.
///
/// Lean proves exact trace normal form into the independent Phi81 combination;
/// fixed-profile generated public rows yield exact reduction or the named
/// 29-public-trace `BatchBadRoot`. Whole-attempt list-`Ring` to `RingF`,
/// challenge authority, CE authority, and fresh Rust conformance remain open.
pub(super) fn assert_costs(profile: &GadgetNativeStageProfile) {
    let families = [
        (
            pi_rlc_stage::IDENTITIES_COMMITMENT,
            18,
            [
                (
                    pi_rlc_stage::IDENTITIES_COMMITMENT_EVALUATIONS_INPUTS,
                    INPUT_EVALUATIONS,
                ),
                (
                    pi_rlc_stage::IDENTITIES_COMMITMENT_EVALUATIONS_OUTPUT,
                    OUTPUT_EVALUATION,
                ),
                (
                    pi_rlc_stage::IDENTITIES_COMMITMENT_EVALUATIONS_QUOTIENT,
                    QUOTIENT_EVALUATION,
                ),
                (
                    pi_rlc_stage::IDENTITIES_COMMITMENT_K_PRODUCTS_RHO_TIMES_INPUT,
                    RHO_TIMES_INPUT,
                ),
                (
                    pi_rlc_stage::IDENTITIES_COMMITMENT_K_PRODUCTS_QUOTIENT_TIMES_PHI,
                    QUOTIENT_TIMES_PHI,
                ),
                (pi_rlc_stage::IDENTITIES_COMMITMENT_FINAL_LIMB_CHECKS, FINAL_LIMB_CHECKS),
            ],
        ),
        (
            pi_rlc_stage::IDENTITIES_ADV,
            0,
            [
                (pi_rlc_stage::IDENTITIES_ADV_EVALUATIONS_INPUTS, INPUT_EVALUATIONS),
                (pi_rlc_stage::IDENTITIES_ADV_EVALUATIONS_OUTPUT, OUTPUT_EVALUATION),
                (pi_rlc_stage::IDENTITIES_ADV_EVALUATIONS_QUOTIENT, QUOTIENT_EVALUATION),
                (pi_rlc_stage::IDENTITIES_ADV_K_PRODUCTS_RHO_TIMES_INPUT, RHO_TIMES_INPUT),
                (
                    pi_rlc_stage::IDENTITIES_ADV_K_PRODUCTS_QUOTIENT_TIMES_PHI,
                    QUOTIENT_TIMES_PHI,
                ),
                (pi_rlc_stage::IDENTITIES_ADV_FINAL_LIMB_CHECKS, FINAL_LIMB_CHECKS),
            ],
        ),
        (
            pi_rlc_stage::IDENTITIES_X,
            5,
            [
                (pi_rlc_stage::IDENTITIES_X_EVALUATIONS_INPUTS, INPUT_EVALUATIONS),
                (pi_rlc_stage::IDENTITIES_X_EVALUATIONS_OUTPUT, OUTPUT_EVALUATION),
                (pi_rlc_stage::IDENTITIES_X_EVALUATIONS_QUOTIENT, QUOTIENT_EVALUATION),
                (pi_rlc_stage::IDENTITIES_X_K_PRODUCTS_RHO_TIMES_INPUT, RHO_TIMES_INPUT),
                (
                    pi_rlc_stage::IDENTITIES_X_K_PRODUCTS_QUOTIENT_TIMES_PHI,
                    QUOTIENT_TIMES_PHI,
                ),
                (pi_rlc_stage::IDENTITIES_X_FINAL_LIMB_CHECKS, FINAL_LIMB_CHECKS),
            ],
        ),
        (
            pi_rlc_stage::IDENTITIES_Y_RING,
            6,
            [
                (pi_rlc_stage::IDENTITIES_Y_RING_EVALUATIONS_INPUTS, INPUT_EVALUATIONS),
                (pi_rlc_stage::IDENTITIES_Y_RING_EVALUATIONS_OUTPUT, OUTPUT_EVALUATION),
                (
                    pi_rlc_stage::IDENTITIES_Y_RING_EVALUATIONS_QUOTIENT,
                    QUOTIENT_EVALUATION,
                ),
                (
                    pi_rlc_stage::IDENTITIES_Y_RING_K_PRODUCTS_RHO_TIMES_INPUT,
                    RHO_TIMES_INPUT,
                ),
                (
                    pi_rlc_stage::IDENTITIES_Y_RING_K_PRODUCTS_QUOTIENT_TIMES_PHI,
                    QUOTIENT_TIMES_PHI,
                ),
                (pi_rlc_stage::IDENTITIES_Y_RING_FINAL_LIMB_CHECKS, FINAL_LIMB_CHECKS),
            ],
        ),
        (
            pi_rlc_stage::IDENTITIES_Y_ZCOL,
            2,
            [
                (pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS, INPUT_EVALUATIONS),
                (pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT, OUTPUT_EVALUATION),
                (
                    pi_rlc_stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT,
                    QUOTIENT_EVALUATION,
                ),
                (
                    pi_rlc_stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT,
                    RHO_TIMES_INPUT,
                ),
                (
                    pi_rlc_stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI,
                    QUOTIENT_TIMES_PHI,
                ),
                (pi_rlc_stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS, FINAL_LIMB_CHECKS),
            ],
        ),
    ];

    let mut identity_count = 0;
    for (claim_path, count, leaves) in families {
        identity_count += count;
        assert_identity_cost(profile, claim_path, count, IDENTITY_TOTAL);
        for (leaf_path, unit) in leaves {
            assert_identity_cost(profile, leaf_path, count, unit);
        }
    }
    macro_rules! assert_y_zcol_limb_phase {
        ($field:ident, $unit:expr) => {
            for stages in [
                pi_rlc_stage::Y_ZCOL_LIMB0_IDENTITY_STAGES,
                pi_rlc_stage::Y_ZCOL_LIMB1_IDENTITY_STAGES,
            ] {
                assert_identity_cost(profile, stages.$field, 1, $unit);
            }
        };
    }
    assert_y_zcol_limb_phase!(input_evaluations, INPUT_EVALUATIONS);
    assert_y_zcol_limb_phase!(output_evaluation, OUTPUT_EVALUATION);
    assert_y_zcol_limb_phase!(quotient_evaluation, QUOTIENT_EVALUATION);
    assert_y_zcol_limb_phase!(rho_times_input, RHO_TIMES_INPUT);
    assert_y_zcol_limb_phase!(quotient_times_phi, QUOTIENT_TIMES_PHI);
    assert_y_zcol_limb_phase!(final_limb_checks, FINAL_LIMB_CHECKS);
    assert_eq!(identity_count, 31, "fixed PiRLC identity census");
    assert_identity_cost(profile, pi_rlc_stage::IDENTITIES, identity_count, IDENTITY_TOTAL);
}

fn assert_identity_cost(
    profile: &GadgetNativeStageProfile,
    path: &'static str,
    multiplicity: usize,
    unit: IdentityCostFormula,
) {
    let actual = profile
        .aggregate_prefix(path)
        .unwrap_or_else(|| panic!("missing PiRLC identity cost leaf {path}"));
    let scale = |value: usize| value * multiplicity;
    assert_eq!(actual.source_rows, scale(unit.source_rows), "{path} source rows");
    assert_eq!(actual.source_cols, scale(unit.source_cols), "{path} source columns");
    assert_eq!(
        actual.canonical_binary_field_source_cols,
        scale(unit.canonical_fields),
        "{path} canonical fields"
    );
    assert_eq!(
        actual.ordinary_private_field_source_cols,
        scale(unit.ordinary_private_fields),
        "{path} ordinary-private fields"
    );
    assert_eq!(
        actual.linearly_derived_source_cols,
        scale(unit.linear_fields),
        "{path} linear fields"
    );
    assert_eq!(
        actual.gadget_derived_source_cols,
        scale(unit.gadget_fields),
        "{path} gadget fields"
    );
    assert_eq!(
        actual.synthetic_product_sum_fields,
        scale(unit.synthetic_product_sum_fields),
        "{path} synthetic product-sum fields"
    );
    assert_eq!(actual.encoded_rows, scale(unit.encoded_rows), "{path} encoded rows");
    assert_eq!(actual.encoded_cols, scale(unit.encoded_cols), "{path} encoded columns");
    assert_eq!(
        actual.fallback_source_rows,
        scale(unit.fallback_rows),
        "{path} fallback rows"
    );
    assert_eq!(
        actual.product_sum_batches,
        scale(unit.product_sum_batches),
        "{path} product-sum batches"
    );
    assert_eq!(
        actual.product_sum_identities,
        scale(unit.product_sum_identities),
        "{path} product-sum identities"
    );
    assert_eq!(
        actual.product_sum_rows,
        scale(unit.product_sum_rows),
        "{path} product-sum rows"
    );
    assert_eq!(
        actual.encoded_row_breakdown(),
        GadgetNativeEncodedRowBreakdown {
            common_centered_unit: repeated_pair_tail(
                unit.ordinary_private_fields * ORDINARY_PRIVATE_DIGITS,
                multiplicity,
            ),
            canonical_binary_source_fields: repeated_canonical_field_rows(unit.canonical_fields, multiplicity),
            ordinary_private_centered_unit: repeated_pair_tail(
                unit.ordinary_private_fields * ORDINARY_PRIVATE_DIGITS,
                multiplicity,
            ),
            synthetic_product_sum_fields: repeated_canonical_field_rows(
                unit.synthetic_product_sum_fields,
                multiplicity,
            ),
            fallback: scale(unit.fallback_rows),
            product_sum: scale(unit.product_sum_rows),
            ..GadgetNativeEncodedRowBreakdown::default()
        },
        "{path} encoded row families"
    );
    let unrelated = [
        actual.one_bit_source_cols,
        actual.balanced_ternary_field_source_cols,
        actual.balanced_ternary_alias_source_cols,
        actual.balanced_ternary_binary_source_cols,
        actual.synthetic_ring_fields,
        actual.redundant_boolean_source_rows,
        actual.poseidon_permutations,
        actual.poseidon_hash_permutations,
        actual.poseidon_hashes,
        actual.sboxes,
        actual.k_muls,
        actual.ring_muls,
        actual.selection_accept_aggregate_rows,
        actual.selection_prefix_aggregate_rows,
        actual.selection_symbol_aggregate_rows,
    ];
    assert!(
        unrelated.iter().all(|&value| value == 0),
        "{path} must not hide unrelated gate families: {unrelated:?}"
    );
    assert!(actual.hash_histogram.is_empty(), "{path} must not hide hashes");
    eprintln!(
        "PI_RLC_IDENTITY_LEAF|{path}|{multiplicity}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        actual.source_rows,
        actual.source_cols,
        actual.encoded_rows,
        actual.encoded_cols,
        actual.canonical_binary_field_source_cols,
        actual.linearly_derived_source_cols,
        actual.gadget_derived_source_cols,
        actual.synthetic_product_sum_fields,
        actual.fallback_source_rows,
        actual.product_sum_batches,
        actual.product_sum_identities,
        actual.product_sum_rows,
    );
}
