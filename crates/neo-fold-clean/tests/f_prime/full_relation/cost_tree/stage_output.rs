//! Machine-readable stage-cost output.
//!
//! Owns: one stable column schema and one renderer for exact stage estimates.
//! Does not own aggregation, constraint emission, or cost formulas.
//!
//! | Output group | Mathematical obligation | Emits constraints? |
//! |---|---|---|
//! | Source roles | Keep ordinary-private, canonical-binary, and SIS words disjoint | no |
//! | Centered rows | Expose ordinary-private and SIS pair/tail children and their parent | no |
//! | Other gate families | Preserve the profiler's exact row-family census | no |

use neo_fold_clean::frontends::f_prime::gadget_native::GadgetNativeStageEstimate;

pub(super) fn print_stage_cost_header(prefix: &str) {
    eprintln!(
        "{prefix}|path|occurrences|source_rows|source_cols|encoded_rows|encoded_cols|bits|canonical_binary_source_fields|ordinary_private_fields|balanced_fields|balanced_aliases|balanced_binary|centered_coords|ordinary_private_coords|sis_centered_coords|synthetic_ring_fields|synthetic_product_sum_fields|acceptance_chunks|acceptance_encoded_cols|acceptance_tree_output_cols|packed_mod5_chunks|packed_mod5_encoded_cols|common_boolean_coordinates|common_boolean_pair_rows|common_boolean_tail_rows|common_centered_unit_pair_rows|common_centered_unit_tail_rows|ordinary_private_centered_pair_rows|ordinary_private_centered_tail_rows|sis_centered_pair_rows|sis_centered_tail_rows|source_raw_coordinates|source_raw_pair_rows|source_raw_tail_rows|source_prefix_coordinates|source_prefix_pair_rows|source_prefix_tail_rows|source_canonicality_relations|source_canonicality_pair_rows|synthetic_ring_raw_coordinates|synthetic_ring_raw_pair_rows|synthetic_ring_raw_tail_rows|synthetic_ring_prefix_coordinates|synthetic_ring_prefix_pair_rows|synthetic_ring_prefix_tail_rows|synthetic_ring_canonicality_relations|synthetic_ring_canonicality_pair_rows|synthetic_product_sum_raw_coordinates|synthetic_product_sum_raw_pair_rows|synthetic_product_sum_raw_tail_rows|synthetic_product_sum_prefix_coordinates|synthetic_product_sum_prefix_pair_rows|synthetic_product_sum_prefix_tail_rows|synthetic_product_sum_canonicality_relations|synthetic_product_sum_canonicality_pair_rows|fallback_rows|redundant_boolean_rows|sbox_rows|k_mul_rows|product_sum_rows|ring_mul_rows|acceptance_tree_bit_pair_rows|acceptance_product_aggregate_rows|acceptance_root_binding_rows|packed_mod5_low_bit_pair_rows|packed_mod5_high_bit_pair_rows|packed_mod5_residue_pair_rows|selection_accept_aggregate_rows|selection_prefix_aggregate_rows|selection_symbol_aggregate_rows|linear|gadget|poseidon_perms|hash_perms|hashes|sboxes|k_muls|product_sum_batches|product_sum_identities|ring_muls|hash_histogram"
    );
}

pub(super) fn print_stage_cost_line(prefix: &str, path: &str, stage: &GadgetNativeStageEstimate) {
    let rows = stage.encoded_row_breakdown();
    eprintln!(
        "{prefix}|{path}|{occurrences}|{source_rows}|{source_cols}|{encoded_rows}|{encoded_cols}|{bits}|{canonical_binary_source_fields}|{ordinary_private_fields}|{balanced_fields}|{balanced_aliases}|{balanced_binary}|{centered_coords}|{ordinary_private_coords}|{sis_centered_coords}|{synthetic_ring_fields}|{synthetic_product_sum_fields}|{acceptance_chunks}|{acceptance_encoded_cols}|{acceptance_tree_output_cols}|{packed_mod5_chunks}|{packed_mod5_encoded_cols}|{common_boolean_coordinates}|{common_boolean_pair_rows}|{common_boolean_tail_rows}|{common_centered_unit_pair_rows}|{common_centered_unit_tail_rows}|{ordinary_private_centered_pair_rows}|{ordinary_private_centered_tail_rows}|{sis_centered_pair_rows}|{sis_centered_tail_rows}|{source_raw_coordinates}|{source_raw_pair_rows}|{source_raw_tail_rows}|{source_prefix_coordinates}|{source_prefix_pair_rows}|{source_prefix_tail_rows}|{source_canonicality_relations}|{source_canonicality_pair_rows}|{synthetic_ring_raw_coordinates}|{synthetic_ring_raw_pair_rows}|{synthetic_ring_raw_tail_rows}|{synthetic_ring_prefix_coordinates}|{synthetic_ring_prefix_pair_rows}|{synthetic_ring_prefix_tail_rows}|{synthetic_ring_canonicality_relations}|{synthetic_ring_canonicality_pair_rows}|{synthetic_product_sum_raw_coordinates}|{synthetic_product_sum_raw_pair_rows}|{synthetic_product_sum_raw_tail_rows}|{synthetic_product_sum_prefix_coordinates}|{synthetic_product_sum_prefix_pair_rows}|{synthetic_product_sum_prefix_tail_rows}|{synthetic_product_sum_canonicality_relations}|{synthetic_product_sum_canonicality_pair_rows}|{fallback_rows}|{redundant_boolean_rows}|{sbox_rows}|{k_mul_rows}|{product_sum_rows}|{ring_mul_rows}|{acceptance_tree_bit_pair_rows}|{acceptance_product_aggregate_rows}|{acceptance_root_binding_rows}|{packed_mod5_low_bit_pair_rows}|{packed_mod5_high_bit_pair_rows}|{packed_mod5_residue_pair_rows}|{selection_accept_aggregate_rows}|{selection_prefix_aggregate_rows}|{selection_symbol_aggregate_rows}|{linear}|{gadget}|{poseidon_perms}|{hash_perms}|{hashes}|{sboxes}|{k_muls}|{product_sum_batches}|{product_sum_identities}|{ring_muls}|{hash_histogram:?}",
        occurrences = stage.occurrences,
        source_rows = stage.source_rows,
        source_cols = stage.source_cols,
        encoded_rows = stage.encoded_rows,
        encoded_cols = stage.encoded_cols,
        bits = stage.one_bit_source_cols,
        canonical_binary_source_fields = stage.canonical_binary_field_source_cols,
        ordinary_private_fields = stage.ordinary_private_field_source_cols,
        balanced_fields = stage.balanced_ternary_field_source_cols,
        balanced_aliases = stage.balanced_ternary_alias_source_cols,
        balanced_binary = stage.balanced_ternary_binary_source_cols,
        centered_coords = stage.centered_encoded_cols,
        ordinary_private_coords = stage.ordinary_private_encoded_cols,
        sis_centered_coords = stage.sis_centered_encoded_cols,
        synthetic_ring_fields = stage.synthetic_ring_fields,
        synthetic_product_sum_fields = stage.synthetic_product_sum_fields,
        acceptance_chunks = stage.acceptance_chunks,
        acceptance_encoded_cols = stage.acceptance_encoded_cols,
        acceptance_tree_output_cols = stage.acceptance_tree_output_cols,
        packed_mod5_chunks = stage.packed_mod5_chunks,
        packed_mod5_encoded_cols = stage.packed_mod5_encoded_cols,
        common_boolean_coordinates = rows.common_boolean.coordinates,
        common_boolean_pair_rows = rows.common_boolean.pair_rows,
        common_boolean_tail_rows = rows.common_boolean.tail_rows,
        common_centered_unit_pair_rows = rows.common_centered_unit.pair_rows,
        common_centered_unit_tail_rows = rows.common_centered_unit.tail_rows,
        ordinary_private_centered_pair_rows = rows.ordinary_private_centered_unit.pair_rows,
        ordinary_private_centered_tail_rows = rows.ordinary_private_centered_unit.tail_rows,
        sis_centered_pair_rows = rows.sis_centered_unit.pair_rows,
        sis_centered_tail_rows = rows.sis_centered_unit.tail_rows,
        source_raw_coordinates = rows.canonical_binary_source_fields.raw_bits.coordinates,
        source_raw_pair_rows = rows.canonical_binary_source_fields.raw_bits.pair_rows,
        source_raw_tail_rows = rows.canonical_binary_source_fields.raw_bits.tail_rows,
        source_prefix_coordinates = rows.canonical_binary_source_fields.prefix_aux.coordinates,
        source_prefix_pair_rows = rows.canonical_binary_source_fields.prefix_aux.pair_rows,
        source_prefix_tail_rows = rows.canonical_binary_source_fields.prefix_aux.tail_rows,
        source_canonicality_relations = rows.canonical_binary_source_fields.canonicality_relations,
        source_canonicality_pair_rows = rows.canonical_binary_source_fields.canonicality_pair_rows,
        synthetic_ring_raw_coordinates = rows.synthetic_ring_fields.raw_bits.coordinates,
        synthetic_ring_raw_pair_rows = rows.synthetic_ring_fields.raw_bits.pair_rows,
        synthetic_ring_raw_tail_rows = rows.synthetic_ring_fields.raw_bits.tail_rows,
        synthetic_ring_prefix_coordinates = rows.synthetic_ring_fields.prefix_aux.coordinates,
        synthetic_ring_prefix_pair_rows = rows.synthetic_ring_fields.prefix_aux.pair_rows,
        synthetic_ring_prefix_tail_rows = rows.synthetic_ring_fields.prefix_aux.tail_rows,
        synthetic_ring_canonicality_relations = rows.synthetic_ring_fields.canonicality_relations,
        synthetic_ring_canonicality_pair_rows = rows.synthetic_ring_fields.canonicality_pair_rows,
        synthetic_product_sum_raw_coordinates = rows.synthetic_product_sum_fields.raw_bits.coordinates,
        synthetic_product_sum_raw_pair_rows = rows.synthetic_product_sum_fields.raw_bits.pair_rows,
        synthetic_product_sum_raw_tail_rows = rows.synthetic_product_sum_fields.raw_bits.tail_rows,
        synthetic_product_sum_prefix_coordinates = rows.synthetic_product_sum_fields.prefix_aux.coordinates,
        synthetic_product_sum_prefix_pair_rows = rows.synthetic_product_sum_fields.prefix_aux.pair_rows,
        synthetic_product_sum_prefix_tail_rows = rows.synthetic_product_sum_fields.prefix_aux.tail_rows,
        synthetic_product_sum_canonicality_relations = rows.synthetic_product_sum_fields.canonicality_relations,
        synthetic_product_sum_canonicality_pair_rows = rows.synthetic_product_sum_fields.canonicality_pair_rows,
        fallback_rows = rows.fallback,
        redundant_boolean_rows = stage.redundant_boolean_source_rows,
        sbox_rows = rows.sbox,
        k_mul_rows = rows.k_mul,
        product_sum_rows = rows.product_sum,
        ring_mul_rows = rows.ring_mul,
        acceptance_tree_bit_pair_rows = rows.acceptance_tree_bit_pair,
        acceptance_product_aggregate_rows = rows.acceptance_product_aggregate,
        acceptance_root_binding_rows = rows.acceptance_root_binding,
        packed_mod5_low_bit_pair_rows = rows.packed_mod5_low_bit_pair,
        packed_mod5_high_bit_pair_rows = rows.packed_mod5_high_bit_pair,
        packed_mod5_residue_pair_rows = rows.packed_mod5_residue_pair,
        selection_accept_aggregate_rows = rows.selection_accept_aggregate,
        selection_prefix_aggregate_rows = rows.selection_prefix_aggregate,
        selection_symbol_aggregate_rows = rows.selection_symbol_aggregate,
        linear = stage.linearly_derived_source_cols,
        gadget = stage.gadget_derived_source_cols,
        poseidon_perms = stage.poseidon_permutations,
        hash_perms = stage.poseidon_hash_permutations,
        hashes = stage.poseidon_hashes,
        sboxes = stage.sboxes,
        k_muls = stage.k_muls,
        product_sum_batches = stage.product_sum_batches,
        product_sum_identities = stage.product_sum_identities,
        ring_muls = stage.ring_muls,
        hash_histogram = stage.hash_histogram,
    );
}
