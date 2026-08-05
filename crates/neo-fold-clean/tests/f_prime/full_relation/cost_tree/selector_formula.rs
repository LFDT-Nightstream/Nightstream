//! Formula-only ownership audit for the fixed selector composition.
//!
//! Owns: disjoint row and column components of the non-executable fixed
//! base/recursive selector estimate. Does not claim trace ownership or
//! selector soundness.
//!
//! | Component | Mathematical obligation | Emits constraints? |
//! |---|---|---|
//! | Coordinate columns | Every shared/private value has one committed coordinate owner | no |
//! | Boolean pair/tail rows | Combined formula uses one explicit family census | no |
//! | Ordinary centered pair/tail rows | All ordinary 41-words share one selector-stage family, disjoint from SIS | no |
//! | SIS centered pair/tail rows | SIS words share a separate selector-stage family | no |
//! | Branch semantics | Active branch rows exclude common coordinate/canonical rows | no |
//! | Inactive ordinary binding | One weighted decoded-field zero row per ordinary 41-word | no |
//! | Inactive one-bit binding | Every inactive ordinary private bit is fixed | no |
//! | Inactive SIS binding | Every inactive SIS centered word is fixed | no |
//! | Inactive canonical binding | Every inactive canonical binary word is fixed | no |
//! | Column-layout ABI | Every estimated fixed column has one ordered, abutting owner; no materializer or authority claim | no |
//! | Direct-selector formula | Every estimator term is explicit; no emitted relation or selector-soundness claim | no |

use neo_fold_clean::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    SelectorGatedGadgetNativeCostLayout, SelectorGatedGadgetNativeEstimate,
};
use neo_fold_clean::frontends::f_prime::low_norm_r1cs::{LowNormR1csEstimate, SelectorGatedR1csEstimate};

const CANONICALITY_RELATIONS_PER_SLOT: usize = 32;
const CANONICALITY_PAIR_ROWS_PER_SLOT: usize = CANONICALITY_RELATIONS_PER_SLOT / 2;

pub(crate) fn assert_direct_selector_cost_formula(
    base: &LowNormR1csEstimate,
    recursive: &LowNormR1csEstimate,
    direct: &SelectorGatedR1csEstimate,
) {
    let public_prefix_columns = direct.public_input_len;
    let selector_columns = 1usize;
    let private_one_bit_columns = direct
        .one_bit_source_cols
        .checked_sub(direct.public_input_len)
        .expect("direct selector one-bit census includes its public prefix");
    let canonical_field_columns = direct
        .canonical_field_source_cols
        .checked_mul(64 + 31)
        .expect("direct selector canonical width overflow");
    let coordinate_bitness_rows = direct.encoded_cols - 1;
    let canonicality_rows = direct
        .canonical_field_source_cols
        .checked_mul(CANONICALITY_RELATIONS_PER_SLOT)
        .expect("direct selector canonicality row overflow");
    let inactive_zero_rows = direct.inactive_zero_rows;
    let base_branch_rows = base
        .source_rows
        .checked_sub(base.linearly_derived_source_cols)
        .expect("base derived-column census cannot exceed its source rows");
    let recursive_branch_rows = recursive
        .source_rows
        .checked_sub(recursive.linearly_derived_source_cols)
        .expect("recursive derived-column census cannot exceed its source rows");
    let branch_rows = base_branch_rows
        .checked_add(recursive_branch_rows)
        .expect("direct selector branch-row formula overflow");

    assert_eq!(
        public_prefix_columns + selector_columns + private_one_bit_columns + canonical_field_columns,
        direct.encoded_cols,
        "direct selector column formula"
    );
    assert_eq!(
        coordinate_bitness_rows + canonicality_rows + inactive_zero_rows + branch_rows,
        direct.encoded_rows,
        "direct selector row formula"
    );
    assert_eq!(
        (
            public_prefix_columns,
            selector_columns,
            private_one_bit_columns,
            direct.canonical_field_source_cols,
            canonical_field_columns,
            coordinate_bitness_rows,
            canonicality_rows,
            inactive_zero_rows,
            branch_rows,
        ),
        (
            257,
            1,
            38_110,
            8_551_728,
            812_414_160,
            812_452_527,
            273_655_296,
            8_589_838,
            8_887_467,
        ),
        "un-audited direct selector estimator component snapshot"
    );

    eprintln!("FPRIME_DIRECT_SELECTOR_FORMULA|component|encoded_rows|encoded_cols");
    for (component, rows, columns) in [
        ("public_prefix", 0, public_prefix_columns),
        ("selector", 0, selector_columns),
        ("private_one_bit", 0, private_one_bit_columns),
        ("canonical_field", 0, canonical_field_columns),
        ("coordinate_bitness", coordinate_bitness_rows, 0),
        ("canonicality", canonicality_rows, 0),
        ("inactive_decoded_zero", inactive_zero_rows, 0),
        ("branch_equations", branch_rows, 0),
    ] {
        eprintln!("FPRIME_DIRECT_SELECTOR_FORMULA|{component}|{rows}|{columns}");
    }
    eprintln!(
        "FPRIME_DIRECT_SELECTOR_FORMULA|total|{}|{}",
        direct.encoded_rows, direct.encoded_cols
    );
}

pub(crate) fn assert_fixed_selector_cost_formula(fixed: &SelectorGatedGadgetNativeEstimate) {
    selector_gated_layout_partitions_every_fixed_column(fixed);

    let public_bits = fixed.public_input_len - 1;
    assert_eq!(fixed.base.public_input_len - 1, public_bits);
    assert_eq!(fixed.recursive.public_input_len - 1, public_bits);

    let base_private_bits = fixed.base.one_bit_source_cols - public_bits;
    let recursive_private_bits = fixed.recursive.one_bit_source_cols - public_bits;
    let base_canonical_slots = fixed
        .base
        .canonical_binary_field_source_cols
        .saturating_add(fixed.base.synthetic_ring_fields)
        .saturating_add(fixed.base.synthetic_product_sum_fields);
    let recursive_canonical_slots = fixed
        .recursive
        .canonical_binary_field_source_cols
        .saturating_add(fixed.recursive.synthetic_ring_fields)
        .saturating_add(fixed.recursive.synthetic_product_sum_fields);
    let source_canonical_slots =
        fixed.base.canonical_binary_field_source_cols + fixed.recursive.canonical_binary_field_source_cols;
    let synthetic_ring_slots = fixed.base.synthetic_ring_fields + fixed.recursive.synthetic_ring_fields;
    let synthetic_product_sum_slots =
        fixed.base.synthetic_product_sum_fields + fixed.recursive.synthetic_product_sum_fields;
    let canonical_slots = base_canonical_slots + recursive_canonical_slots;
    let ordinary_slots =
        fixed.base.ordinary_private_field_source_cols + fixed.recursive.ordinary_private_field_source_cols;
    let ordinary_coordinates =
        ordinary_slots * neo_fold_clean::frontends::f_prime::gadget_native::ORDINARY_PRIVATE_DIGITS;
    let balanced_slots =
        fixed.base.balanced_ternary_field_source_cols + fixed.recursive.balanced_ternary_field_source_cols;
    let balanced_coordinates = balanced_slots * BALANCED_TERNARY_DIGITS;
    let one_bit_slots = public_bits + 1 + base_private_bits + recursive_private_bits;
    let base_packed_low_bits = fixed.base.packed_mod5_low_bit_pair_rows * 2 + fixed.base.packed_mod5_high_bit_pair_rows;
    let recursive_packed_low_bits =
        fixed.recursive.packed_mod5_low_bit_pair_rows * 2 + fixed.recursive.packed_mod5_high_bit_pair_rows;
    let base_packed_residue_coordinates = fixed.base.packed_mod5_encoded_cols - base_packed_low_bits;
    let recursive_packed_residue_coordinates = fixed.recursive.packed_mod5_encoded_cols - recursive_packed_low_bits;
    let packed_low_bits = base_packed_low_bits + recursive_packed_low_bits;
    let packed_residue_coordinates = base_packed_residue_coordinates + recursive_packed_residue_coordinates;
    let acceptance_chunks = fixed.base.acceptance_chunks + fixed.recursive.acceptance_chunks;
    let acceptance_coordinates = fixed.base.acceptance_encoded_cols + fixed.recursive.acceptance_encoded_cols;
    let acceptance_tree_output_coordinates =
        fixed.base.acceptance_tree_output_cols + fixed.recursive.acceptance_tree_output_cols;
    let base_coordinate_rows = fixed
        .base
        .boolean_pairing
        .total_rows()
        .saturating_add(fixed.base.centered_pairing.total_rows());
    let recursive_coordinate_rows = fixed
        .recursive
        .boolean_pairing
        .total_rows()
        .saturating_add(fixed.recursive.centered_pairing.total_rows());
    let base_semantic_rows = fixed
        .base
        .encoded_rows
        .saturating_sub(base_coordinate_rows)
        .saturating_sub(base_canonical_slots * CANONICALITY_PAIR_ROWS_PER_SLOT);
    let recursive_semantic_rows = fixed
        .recursive
        .encoded_rows
        .saturating_sub(recursive_coordinate_rows)
        .saturating_sub(recursive_canonical_slots * CANONICALITY_PAIR_ROWS_PER_SLOT);
    let inactive_one_bit_zero_rows = base_private_bits + recursive_private_bits - packed_low_bits - acceptance_chunks;
    let inactive_sis_word_zero_rows = balanced_slots;
    let inactive_canonical_word_zero_rows = canonical_slots;
    let inactive_other_zero_rows =
        inactive_one_bit_zero_rows + inactive_sis_word_zero_rows + inactive_canonical_word_zero_rows;
    let inactive_binding_rows = inactive_other_zero_rows
        + fixed.ordinary_private_inactive_binding_rows
        + fixed.packed_mod5_inactive_low_bit_rows
        + fixed.packed_mod5_inactive_residue_pair_rows
        + fixed.acceptance_inactive_binding_rows;

    assert_eq!(fixed.one_bit_slots, one_bit_slots);
    assert_eq!(fixed.canonical_binary_field_slots, canonical_slots);
    assert_eq!(fixed.ordinary_private_field_slots, ordinary_slots);
    assert_eq!(fixed.ordinary_private_coordinates, ordinary_coordinates);
    assert_eq!(fixed.ordinary_private_inactive_binding_rows, ordinary_slots);
    assert_eq!(fixed.balanced_ternary_field_slots, balanced_slots);
    assert_eq!(
        fixed.packed_mod5_chunks,
        fixed.base.packed_mod5_chunks + fixed.recursive.packed_mod5_chunks
    );
    assert_eq!(fixed.base.packed_mod5_high_bit_pair_rows, fixed.base.packed_mod5_chunks);
    assert_eq!(
        fixed.recursive.packed_mod5_high_bit_pair_rows,
        fixed.recursive.packed_mod5_chunks
    );
    assert_eq!(fixed.base.packed_mod5_residue_pair_rows, fixed.base.packed_mod5_chunks);
    assert_eq!(
        fixed.recursive.packed_mod5_residue_pair_rows,
        fixed.recursive.packed_mod5_chunks
    );
    assert_eq!(fixed.packed_mod5_low_bit_slots, packed_low_bits);
    assert_eq!(fixed.packed_mod5_residue_coordinates, packed_residue_coordinates);
    assert_eq!(fixed.packed_mod5_inactive_low_bit_rows, packed_low_bits);
    assert_eq!(fixed.packed_mod5_inactive_residue_pair_rows, fixed.packed_mod5_chunks);
    assert_eq!(fixed.acceptance_coordinates, acceptance_coordinates);
    assert_eq!(fixed.acceptance_inactive_binding_rows, acceptance_coordinates);
    assert_eq!(fixed.inactive_binding_rows, inactive_binding_rows);

    let expected_common_coordinates = one_bit_slots
        - fixed.base.balanced_ternary_binary_source_cols
        - fixed.recursive.balanced_ternary_binary_source_cols
        - packed_low_bits
        - acceptance_chunks;
    assert_eq!(fixed.boolean_pairing.common.coordinates, expected_common_coordinates);
    assert_eq!(
        fixed.ordinary_private_centered_pairing.coordinates,
        ordinary_coordinates
    );
    assert_eq!(fixed.sis_centered_pairing.coordinates, balanced_coordinates);
    assert_eq!(
        fixed.centered_pairing.coordinates,
        ordinary_coordinates + balanced_coordinates
    );
    assert_eq!(
        fixed.ordinary_private_centered_pairing.total_rows(),
        ordinary_coordinates.div_ceil(2),
        "ordinary pairing resets once at the combined selector stage"
    );
    assert_eq!(
        fixed.boolean_pairing.source_raw64.coordinates,
        source_canonical_slots * 64
    );
    assert_eq!(
        fixed.boolean_pairing.source_prefix31.coordinates,
        source_canonical_slots * 31
    );
    assert_eq!(
        fixed.boolean_pairing.synthetic_ring_raw64.coordinates,
        synthetic_ring_slots * 64
    );
    assert_eq!(
        fixed.boolean_pairing.synthetic_ring_prefix31.coordinates,
        synthetic_ring_slots * 31
    );
    assert_eq!(
        fixed
            .boolean_pairing
            .synthetic_product_sum_raw64
            .coordinates,
        synthetic_product_sum_slots * 64
    );
    assert_eq!(
        fixed
            .boolean_pairing
            .synthetic_product_sum_prefix31
            .coordinates,
        synthetic_product_sum_slots * 31
    );

    let mut components = vec![
        ("constant_one_column", 0, 1),
        ("one_bit_columns", 0, one_bit_slots),
        ("acceptance_tree_output_columns", 0, acceptance_tree_output_coordinates),
        ("ordinary_private_columns", 0, ordinary_coordinates),
        ("balanced_columns", 0, balanced_coordinates),
        ("packed_residue_columns", 0, packed_residue_coordinates),
        (
            "canonical_source_raw_columns",
            0,
            fixed.boolean_pairing.source_raw64.coordinates,
        ),
        (
            "canonical_source_prefix_columns",
            0,
            fixed.boolean_pairing.source_prefix31.coordinates,
        ),
        (
            "canonical_ring_raw_columns",
            0,
            fixed.boolean_pairing.synthetic_ring_raw64.coordinates,
        ),
        (
            "canonical_ring_prefix_columns",
            0,
            fixed.boolean_pairing.synthetic_ring_prefix31.coordinates,
        ),
        (
            "canonical_product_sum_raw_columns",
            0,
            fixed
                .boolean_pairing
                .synthetic_product_sum_raw64
                .coordinates,
        ),
        (
            "canonical_product_sum_prefix_columns",
            0,
            fixed
                .boolean_pairing
                .synthetic_product_sum_prefix31
                .coordinates,
        ),
        ("common_boolean_pair_rows", fixed.boolean_pairing.common.pair_rows, 0),
        ("common_boolean_tail_rows", fixed.boolean_pairing.common.tail_rows, 0),
        (
            "ordinary_private_centered_pair_rows",
            fixed.ordinary_private_centered_pairing.pair_rows,
            0,
        ),
        (
            "ordinary_private_centered_tail_rows",
            fixed.ordinary_private_centered_pairing.tail_rows,
            0,
        ),
        ("sis_centered_pair_rows", fixed.sis_centered_pairing.pair_rows, 0),
        ("sis_centered_tail_rows", fixed.sis_centered_pairing.tail_rows, 0),
    ];
    for (pair_name, tail_name, count) in [
        (
            "canonical_source_raw_pair_rows",
            "canonical_source_raw_tail_rows",
            fixed.boolean_pairing.source_raw64,
        ),
        (
            "canonical_source_prefix_pair_rows",
            "canonical_source_prefix_tail_rows",
            fixed.boolean_pairing.source_prefix31,
        ),
        (
            "canonical_ring_raw_pair_rows",
            "canonical_ring_raw_tail_rows",
            fixed.boolean_pairing.synthetic_ring_raw64,
        ),
        (
            "canonical_ring_prefix_pair_rows",
            "canonical_ring_prefix_tail_rows",
            fixed.boolean_pairing.synthetic_ring_prefix31,
        ),
        (
            "canonical_product_sum_raw_pair_rows",
            "canonical_product_sum_raw_tail_rows",
            fixed.boolean_pairing.synthetic_product_sum_raw64,
        ),
        (
            "canonical_product_sum_prefix_pair_rows",
            "canonical_product_sum_prefix_tail_rows",
            fixed.boolean_pairing.synthetic_product_sum_prefix31,
        ),
    ] {
        components.push((pair_name, count.pair_rows, 0));
        components.push((tail_name, count.tail_rows, 0));
    }
    components.extend([
        (
            "canonical_relation_pairs",
            canonical_slots * CANONICALITY_PAIR_ROWS_PER_SLOT,
            0,
        ),
        ("base_semantics", base_semantic_rows, 0),
        ("recursive_semantics", recursive_semantic_rows, 0),
        ("inactive_one_bit_zero", inactive_one_bit_zero_rows, 0),
        ("inactive_sis_word_zero", inactive_sis_word_zero_rows, 0),
        ("inactive_canonical_word_zero", inactive_canonical_word_zero_rows, 0),
        (
            "inactive_ordinary_private_weighted_decode",
            fixed.ordinary_private_inactive_binding_rows,
            0,
        ),
        ("inactive_acceptance", fixed.acceptance_inactive_binding_rows, 0),
        ("inactive_packed_low_bits", fixed.packed_mod5_inactive_low_bit_rows, 0),
        (
            "inactive_packed_residue_pair",
            fixed.packed_mod5_inactive_residue_pair_rows,
            0,
        ),
    ]);

    let encoded_rows = components.iter().map(|(_, rows, _)| rows).sum::<usize>();
    let encoded_cols = components.iter().map(|(_, _, cols)| cols).sum::<usize>();
    assert_eq!(encoded_rows, fixed.encoded_rows, "fixed selector row ownership");
    assert_eq!(encoded_cols, fixed.encoded_cols, "fixed selector column ownership");

    eprintln!("FPRIME_FIXED_FORMULA|component|encoded_rows|encoded_cols");
    for (component, rows, cols) in components {
        eprintln!("FPRIME_FIXED_FORMULA|{component}|{rows}|{cols}");
    }
    eprintln!(
        "FPRIME_FIXED_FORMULA|total|{}|{}",
        fixed.encoded_rows, fixed.encoded_cols
    );
}

fn selector_gated_layout_partitions_every_fixed_column(fixed: &SelectorGatedGadgetNativeEstimate) {
    let layout = SelectorGatedGadgetNativeCostLayout::from_estimate(fixed);
    let expected = [
        ("constant_one", 1),
        ("one_bit", fixed.one_bit_slots),
        (
            "acceptance_tree_outputs",
            fixed
                .base
                .acceptance_tree_output_cols
                .checked_add(fixed.recursive.acceptance_tree_output_cols)
                .expect("fixed acceptance-output width overflow"),
        ),
        ("ordinary_private", fixed.ordinary_private_coordinates),
        (
            "balanced_ternary",
            fixed
                .balanced_ternary_field_slots
                .checked_mul(BALANCED_TERNARY_DIGITS)
                .expect("fixed balanced-ternary width overflow"),
        ),
        ("packed_mod5_residues", fixed.packed_mod5_residue_coordinates),
        ("canonical_source_raw", fixed.boolean_pairing.source_raw64.coordinates),
        (
            "canonical_source_prefix",
            fixed.boolean_pairing.source_prefix31.coordinates,
        ),
        (
            "canonical_ring_raw",
            fixed.boolean_pairing.synthetic_ring_raw64.coordinates,
        ),
        (
            "canonical_ring_prefix",
            fixed.boolean_pairing.synthetic_ring_prefix31.coordinates,
        ),
        (
            "canonical_product_sum_raw",
            fixed
                .boolean_pairing
                .synthetic_product_sum_raw64
                .coordinates,
        ),
        (
            "canonical_product_sum_prefix",
            fixed
                .boolean_pairing
                .synthetic_product_sum_prefix31
                .coordinates,
        ),
    ];

    let mut cursor = 0;
    for ((name, range), (expected_name, expected_width)) in layout.ordered_ranges().into_iter().zip(expected) {
        assert_eq!(name, expected_name, "fixed selector column family order");
        assert_eq!(range.start, cursor, "{name} must abut its predecessor");
        assert_eq!(range.end - range.start, expected_width, "{name} width");
        cursor = range.end;
    }
    assert_eq!(cursor, fixed.encoded_cols, "fixed selector column partition end");
    assert_eq!(layout.all_columns(), 0..fixed.encoded_cols);
    assert_eq!(
        layout
            .ordered_ranges()
            .map(|(name, range)| (name, range.clone())),
        [
            ("constant_one", 0..1),
            ("one_bit", 1..5_496_310),
            ("acceptance_tree_outputs", 5_496_310..5_509_750),
            ("ordinary_private", 5_509_750..9_491_752),
            ("balanced_ternary", 9_491_752..12_254_414),
            ("packed_mod5_residues", 12_254_414..12_256_334),
            ("canonical_source_raw", 12_256_334..12_256_462),
            ("canonical_source_prefix", 12_256_462..12_256_524),
            ("canonical_ring_raw", 12_256_524..12_256_524),
            ("canonical_ring_prefix", 12_256_524..12_256_524),
            ("canonical_product_sum_raw", 12_256_524..12_395_404),
            ("canonical_product_sum_prefix", 12_395_404..12_462_674),
        ],
        "production fixed selector column ABI"
    );
}
