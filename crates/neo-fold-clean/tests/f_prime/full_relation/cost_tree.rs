//! Structural and quantitative audit of the base and recursive protocol cost trees.
//!
//! Owns: node visibility, parent/child reconciliation, and machine-readable
//! aggregate output for the fixed recursive fixture.
//!
//! Does not own: constraint emission, stage placement, or handwritten cost
//! constants.
//!
//! Emits constraints: no.
//!
//! Authority boundary: production stage ranges and the profiler are inputs;
//! this test rejects missing, overlapping-by-accounting, or hidden cost.
//!
//! | Child surface | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Base/recursive FPrime `ALL/HIERARCHY` | Every complete branch stage has exactly one owner | no | `paper::f_prime::stage` | concrete full bridge open |
//! | PiCCS `ROOT/ALL/HIERARCHY` | Every verifier phase and constraint family reconciles | no | `pi_ccs_circuit::stage` | concrete bridges remain scoped |
//! | Lifecycle `ROOT/LIFECYCLE_ALL` | Challenge, parent/child shape, and algebra reconcile under one PiRLC root | no | `nifs::circuit::pi_rlc` | ownership only |
//! | Challenge `ALL/HIERARCHY` | Every transcript/sampler node, including the three packed Mod-5 leaves, reconciles | no | `alphabet_sampling` plus `gadget_native::mod5` | PiRlcChallenge hierarchy |
//! | Algebra `ALL/HIERARCHY` | Every verifier node reconciles | no | `pi_rlc_circuit::stage` | PiRlcAlgebra hierarchy |
//! | PiRLC identity equation leaves | Every one of 31 identities has the same six exact cost formulas | no | `ring_action.rs` plus validated compact plan | exact Phi81 normal form; generated public rows imply exact reduction or `BatchBadRoot` |
//! | `PI_CCS_TREE` | Exact source and lowered Π_CCS costs remain inspectable | no | gadget-native profiler | no theorem claim |
//! | `PI_RLC_TREE` | Exact source and lowered Π_RLC costs remain inspectable | no | gadget-native profiler | no theorem claim |
//! | Common/canonical gate families | Boolean, centered-unit, and per-origin canonical rows reconcile at every parent | no | `cost_tree::row_family_snapshots` | no theorem claim |
//! | `FPRIME_DIRECT_SELECTOR_FORMULA` | Legacy 258M arithmetic is decomposed without claiming an emitted relation | no | direct selector estimator | no theorem claim |
//! | `FPRIME_FIXED_FORMULA` | Selector cost formula reconciles without claiming trace ownership or selector soundness | no | selector-gated estimator | inactive packed binding proved; combined materializer open |
//! | Dominant SIS snapshots | Pin the two selected generic-lowering cost centers | no | this file | concrete refinement still open |

#[path = "cost_tree/row_family_snapshots.rs"]
mod row_family_snapshots;
#[path = "cost_tree/selector_formula.rs"]
mod selector_formula;
#[path = "cost_tree/stage_output.rs"]
mod stage_output;

pub(super) use row_family_snapshots::assert_protocol_row_family_snapshots;
pub(super) use selector_formula::{assert_direct_selector_cost_formula, assert_fixed_selector_cost_formula};
use stage_output::{print_stage_cost_header, print_stage_cost_line};

use std::collections::{BTreeMap, BTreeSet};

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    GadgetNativeCanonicalBinaryFieldRowBreakdown, GadgetNativeEncodedRowBreakdown, GadgetNativePairTailCount,
    GadgetNativeStageEstimate, GadgetNativeStageProfile, ORDINARY_PRIVATE_DIGITS,
};
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::nifs::circuit::stage as nifs_stage;
use neo_fold_clean::paper::reductions::pi_ccs_circuit::stage as pi_ccs_stage;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;

const CANONICALITY_RELATIONS_PER_SLOT: usize = 32;
const CANONICALITY_PAIR_ROWS_PER_SLOT: usize = CANONICALITY_RELATIONS_PER_SLOT / 2;

fn pair_tail(coordinates: usize) -> GadgetNativePairTailCount {
    GadgetNativePairTailCount {
        coordinates,
        pair_rows: coordinates / 2,
        tail_rows: coordinates % 2,
    }
}

fn add_pair_tail(total: &mut GadgetNativePairTailCount, rows: GadgetNativePairTailCount) {
    total.coordinates += rows.coordinates;
    total.pair_rows += rows.pair_rows;
    total.tail_rows += rows.tail_rows;
}

pub(super) fn assert_f_prime_base_stage_hierarchy(profile: &GadgetNativeStageProfile) {
    assert_stage_hierarchy(
        profile,
        fprime_stage::BASE_ROOT,
        fprime_stage::BASE_ALL,
        fprime_stage::BASE_HIERARCHY,
        fprime_stage::BASE_ALL,
        "FPRIME_BASE_TREE",
        true,
    );
}

pub(super) fn assert_f_prime_recursive_stage_hierarchy(profile: &GadgetNativeStageProfile) {
    let all = fprime_stage::RECURSIVE_ALL
        .iter()
        .chain(nifs_stage::ALL)
        .chain(pi_ccs_stage::ALL)
        .chain(pi_rlc_stage::LIFECYCLE_ALL)
        .chain(pi_rlc_challenge_stage::ALL)
        .chain(pi_rlc_stage::ALL)
        .copied()
        .collect::<Vec<_>>();
    let hierarchy = fprime_stage::RECURSIVE_HIERARCHY
        .iter()
        .chain(nifs_stage::HIERARCHY)
        .chain(pi_ccs_stage::HIERARCHY)
        .chain(pi_rlc_stage::LIFECYCLE_HIERARCHY)
        .chain(pi_rlc_challenge_stage::HIERARCHY)
        .chain(pi_rlc_stage::HIERARCHY)
        .copied()
        .collect::<Vec<_>>();
    let print_paths = fprime_stage::RECURSIVE_ALL
        .iter()
        .copied()
        .chain([
            pi_ccs_stage::ROOT,
            pi_rlc_stage::ROOT,
            nifs_stage::RUNNING_PARENT_PI_DEC,
            nifs_stage::PI_DEC,
            nifs_stage::PI_DEC_VERIFY,
            nifs_stage::POINT_BINDING,
        ])
        .collect::<Vec<_>>();
    assert_stage_hierarchy(
        profile,
        fprime_stage::RECURSIVE_ROOT,
        &all,
        &hierarchy,
        &print_paths,
        "FPRIME_RECURSIVE_TREE",
        true,
    );
}

pub(super) fn assert_pi_ccs_stage_hierarchy(profile: &GadgetNativeStageProfile) {
    assert_stage_hierarchy(
        profile,
        pi_ccs_stage::ROOT,
        pi_ccs_stage::ALL,
        pi_ccs_stage::HIERARCHY,
        pi_ccs_stage::ALL,
        "PI_CCS_TREE",
        false,
    );
}

pub(super) fn assert_pi_rlc_stage_hierarchy(profile: &GadgetNativeStageProfile) {
    let all = pi_rlc_stage::LIFECYCLE_ALL
        .iter()
        .chain(pi_rlc_challenge_stage::ALL)
        .chain(pi_rlc_stage::ALL)
        .copied()
        .collect::<Vec<_>>();
    let hierarchy = pi_rlc_stage::LIFECYCLE_HIERARCHY
        .iter()
        .chain(pi_rlc_challenge_stage::HIERARCHY)
        .chain(pi_rlc_stage::HIERARCHY)
        .copied()
        .collect::<Vec<_>>();
    assert_stage_hierarchy(
        profile,
        pi_rlc_stage::ROOT,
        &all,
        &hierarchy,
        &all,
        "PI_RLC_TREE",
        false,
    );
    assert_packed_mod5_cost_leaves(profile);
}

fn assert_packed_mod5_cost_leaves(profile: &GadgetNativeStageProfile) {
    struct UnitCost {
        path: &'static str,
        source_rows: usize,
        source_cols: usize,
        encoded_rows: usize,
        encoded_cols: usize,
        bits: usize,
        linear: usize,
        gadget: usize,
        chunk_census: usize,
        low_rows: usize,
        high_rows: usize,
        residue_rows: usize,
    }
    let chunks = profile.total.packed_mod5_chunks;
    let expected = [
        UnitCost {
            path: pi_rlc_challenge_stage::LOW_BIT_PAIRS,
            source_rows: 12,
            source_cols: 12,
            encoded_rows: 6,
            encoded_cols: 12,
            bits: 12,
            linear: 0,
            gadget: 0,
            chunk_census: 1,
            low_rows: 6,
            high_rows: 0,
            residue_rows: 0,
        },
        UnitCost {
            path: pi_rlc_challenge_stage::HIGH_BIT_PAIR,
            source_rows: 4,
            source_cols: 3,
            encoded_rows: 1,
            encoded_cols: 1,
            bits: 1,
            linear: 2,
            gadget: 0,
            chunk_census: 0,
            low_rows: 0,
            high_rows: 1,
            residue_rows: 0,
        },
        UnitCost {
            path: pi_rlc_challenge_stage::RESIDUE_PAIR,
            source_rows: 4,
            source_cols: 4,
            encoded_rows: 1,
            encoded_cols: 2,
            bits: 0,
            linear: 1,
            gadget: 3,
            chunk_census: 0,
            low_rows: 0,
            high_rows: 0,
            residue_rows: 1,
        },
    ];
    for unit in expected {
        let actual = profile
            .aggregate_prefix(unit.path)
            .unwrap_or_else(|| panic!("missing packed mod-5 leaf {}", unit.path));
        assert_eq!(actual.occurrences, chunks, "{} occurrences", unit.path);
        assert_eq!(
            actual.source_rows,
            chunks * unit.source_rows,
            "{} source rows",
            unit.path
        );
        assert_eq!(
            actual.source_cols,
            chunks * unit.source_cols,
            "{} source columns",
            unit.path
        );
        assert_eq!(
            actual.encoded_rows,
            chunks * unit.encoded_rows,
            "{} encoded rows",
            unit.path
        );
        assert_eq!(
            actual.encoded_cols,
            chunks * unit.encoded_cols,
            "{} encoded columns",
            unit.path
        );
        assert_eq!(
            actual.one_bit_source_cols,
            chunks * unit.bits,
            "{} Boolean columns",
            unit.path
        );
        assert_eq!(
            actual.linearly_derived_source_cols,
            chunks * unit.linear,
            "{} linear columns",
            unit.path
        );
        assert_eq!(
            actual.gadget_derived_source_cols,
            chunks * unit.gadget,
            "{} gadget columns",
            unit.path
        );
        assert_eq!(actual.packed_mod5_chunks, chunks * unit.chunk_census);
        assert_eq!(actual.packed_mod5_encoded_cols, chunks * unit.encoded_cols);
        assert_eq!(actual.packed_mod5_low_bit_pair_rows, chunks * unit.low_rows);
        assert_eq!(actual.packed_mod5_high_bit_pair_rows, chunks * unit.high_rows);
        assert_eq!(actual.packed_mod5_residue_pair_rows, chunks * unit.residue_rows);
    }
}

fn assert_stage_hierarchy(
    profile: &GadgetNativeStageProfile,
    root: &'static str,
    all: &[&'static str],
    hierarchy: &[(&'static str, &[&'static str])],
    print_paths: &[&'static str],
    tree_prefix: &str,
    complete_profile: bool,
) {
    let exact_stages = profile.aggregate_by_label();
    let exact_by_label = exact_stages
        .iter()
        .map(|stage| (stage.label, stage.clone()))
        .collect::<BTreeMap<_, _>>();
    let expected_nodes = all.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        expected_nodes.len(),
        all.len(),
        "{root} ALL must not contain duplicate nodes"
    );
    let mut child_owners = BTreeMap::<&'static str, usize>::new();
    let children_by_parent = hierarchy.iter().copied().collect::<BTreeMap<_, _>>();
    assert_eq!(
        children_by_parent.len(),
        hierarchy.len(),
        "{root} hierarchy must not repeat a parent"
    );
    for &(parent, children) in hierarchy {
        assert!(
            expected_nodes.contains(parent),
            "{root} hierarchy parent {parent} must be in ALL"
        );
        assert!(
            !children.is_empty(),
            "{root} hierarchy parent {parent} must own children"
        );
        for &child in children {
            assert!(
                expected_nodes.contains(child),
                "{root} hierarchy child {child} must be in ALL"
            );
            *child_owners.entry(child).or_default() += 1;
        }
    }
    assert!(!child_owners.contains_key(root), "{root} must not have a parent");
    for &node in &expected_nodes {
        if node != root {
            assert_eq!(child_owners.get(node), Some(&1), "{node} must have exactly one parent");
        }
    }

    let mut reachable = BTreeSet::new();
    let mut visiting = BTreeSet::new();
    visit_tree(root, &children_by_parent, &mut visiting, &mut reachable);
    assert_eq!(
        reachable, expected_nodes,
        "every {root} node must be reachable from its root"
    );

    let actual_nodes = if complete_profile {
        exact_stages
            .iter()
            .map(|stage| stage.label)
            .collect::<BTreeSet<_>>()
    } else {
        exact_stages
            .iter()
            .filter(|stage| {
                stage.label == root
                    || stage
                        .label
                        .strip_prefix(root)
                        .is_some_and(|suffix| suffix.starts_with('.'))
            })
            .map(|stage| stage.label)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(
        actual_nodes, expected_nodes,
        "every stable node below {root}, including zero-cost owners, must be visible"
    );

    let root_total = aggregate_tree_node(root, &exact_by_label, &children_by_parent);
    if complete_profile {
        assert_complete_profile_total(profile, &root_total);
    }

    print_tree(&exact_by_label, &children_by_parent, print_paths, tree_prefix);

    for &(parent_path, child_paths) in hierarchy {
        let parent = aggregate_tree_node(parent_path, &exact_by_label, &children_by_parent);
        let checkpoint = exact_stages
            .iter()
            .find(|stage| stage.label == parent_path)
            .unwrap_or_else(|| panic!("missing organizational checkpoint {parent_path}"));
        assert_zero_cost_checkpoint(checkpoint);
        let children = child_paths
            .iter()
            .map(|&child_path| aggregate_tree_node(child_path, &exact_by_label, &children_by_parent))
            .collect::<Vec<_>>();

        assert_eq!(
            parent.occurrences,
            checkpoint.occurrences
                + children
                    .iter()
                    .map(|child| child.occurrences)
                    .sum::<usize>(),
            "{parent_path} occurrences must equal checkpoints plus immediate children"
        );
        macro_rules! assert_child_sum {
            ($($field:ident),+ $(,)?) => {
                $(assert_eq!(
                    parent.$field,
                    children.iter().map(|child| child.$field).sum::<usize>(),
                    "{parent_path}.{} must equal its immediate-child sum",
                    stringify!($field)
                );)+
            };
        }
        assert_child_sum!(
            source_rows,
            source_cols,
            one_bit_source_cols,
            canonical_binary_field_source_cols,
            ordinary_private_field_source_cols,
            balanced_ternary_field_source_cols,
            balanced_ternary_alias_source_cols,
            balanced_ternary_binary_source_cols,
            linearly_derived_source_cols,
            gadget_derived_source_cols,
            synthetic_ring_fields,
            synthetic_product_sum_fields,
            acceptance_chunks,
            acceptance_encoded_cols,
            acceptance_tree_output_cols,
            acceptance_tree_bit_pair_rows,
            acceptance_product_aggregate_rows,
            acceptance_root_binding_rows,
            packed_mod5_chunks,
            packed_mod5_encoded_cols,
            packed_mod5_low_bit_pair_rows,
            packed_mod5_high_bit_pair_rows,
            packed_mod5_residue_pair_rows,
            encoded_cols,
            centered_encoded_cols,
            ordinary_private_encoded_cols,
            sis_centered_encoded_cols,
            encoded_rows,
            redundant_boolean_source_rows,
            fallback_source_rows,
            poseidon_permutations,
            poseidon_hash_permutations,
            poseidon_hashes,
            sboxes,
            k_muls,
            product_sum_batches,
            product_sum_identities,
            product_sum_rows,
            ring_muls,
            selection_accept_aggregate_rows,
            selection_prefix_aggregate_rows,
            selection_symbol_aggregate_rows,
        );

        let sum_pairing = |select: fn(&GadgetNativeStageEstimate) -> GadgetNativePairTailCount| {
            children
                .iter()
                .fold(GadgetNativePairTailCount::default(), |mut total, child| {
                    add_pair_tail(&mut total, select(child));
                    total
                })
        };
        assert_eq!(
            parent.ordinary_private_centered_pairing,
            sum_pairing(|child| child.ordinary_private_centered_pairing),
            "{parent_path}.ordinary_private_centered_pairing must equal its immediate-child sum"
        );
        assert_eq!(
            parent.sis_centered_pairing,
            sum_pairing(|child| child.sis_centered_pairing),
            "{parent_path}.sis_centered_pairing must equal its immediate-child sum"
        );
        assert_eq!(
            parent.centered_pairing,
            sum_pairing(|child| child.centered_pairing),
            "{parent_path}.centered_pairing must equal its immediate-child sum"
        );

        let mut child_hash_histogram = BTreeMap::<usize, (usize, usize)>::new();
        for child in &children {
            for (&input_len, &(calls, permutations)) in &child.hash_histogram {
                let total = child_hash_histogram.entry(input_len).or_default();
                total.0 += calls;
                total.1 += permutations;
            }
        }
        assert_eq!(
            parent.hash_histogram, child_hash_histogram,
            "{parent_path}.hash_histogram must equal its immediate-child sum"
        );
        assert_eq!(
            parent.encoded_row_breakdown(),
            sum_row_breakdowns(&children),
            "{parent_path} encoded gate families must equal their immediate-child sum"
        );
    }
}

fn sum_row_breakdowns(stages: &[GadgetNativeStageEstimate]) -> GadgetNativeEncodedRowBreakdown {
    stages
        .iter()
        .map(GadgetNativeStageEstimate::encoded_row_breakdown)
        .fold(GadgetNativeEncodedRowBreakdown::default(), |mut total, rows| {
            add_pair_tail(&mut total.common_boolean, rows.common_boolean);
            add_pair_tail(&mut total.common_centered_unit, rows.common_centered_unit);
            add_pair_tail(
                &mut total.ordinary_private_centered_unit,
                rows.ordinary_private_centered_unit,
            );
            add_pair_tail(&mut total.sis_centered_unit, rows.sis_centered_unit);
            add_pair_tail(
                &mut total.canonical_binary_source_fields.raw_bits,
                rows.canonical_binary_source_fields.raw_bits,
            );
            add_pair_tail(
                &mut total.canonical_binary_source_fields.prefix_aux,
                rows.canonical_binary_source_fields.prefix_aux,
            );
            total.canonical_binary_source_fields.canonicality_relations +=
                rows.canonical_binary_source_fields.canonicality_relations;
            total.canonical_binary_source_fields.canonicality_pair_rows +=
                rows.canonical_binary_source_fields.canonicality_pair_rows;
            add_pair_tail(
                &mut total.synthetic_ring_fields.raw_bits,
                rows.synthetic_ring_fields.raw_bits,
            );
            add_pair_tail(
                &mut total.synthetic_ring_fields.prefix_aux,
                rows.synthetic_ring_fields.prefix_aux,
            );
            total.synthetic_ring_fields.canonicality_relations += rows.synthetic_ring_fields.canonicality_relations;
            total.synthetic_ring_fields.canonicality_pair_rows += rows.synthetic_ring_fields.canonicality_pair_rows;
            add_pair_tail(
                &mut total.synthetic_product_sum_fields.raw_bits,
                rows.synthetic_product_sum_fields.raw_bits,
            );
            add_pair_tail(
                &mut total.synthetic_product_sum_fields.prefix_aux,
                rows.synthetic_product_sum_fields.prefix_aux,
            );
            total.synthetic_product_sum_fields.canonicality_relations +=
                rows.synthetic_product_sum_fields.canonicality_relations;
            total.synthetic_product_sum_fields.canonicality_pair_rows +=
                rows.synthetic_product_sum_fields.canonicality_pair_rows;
            total.fallback += rows.fallback;
            total.sbox += rows.sbox;
            total.k_mul += rows.k_mul;
            total.product_sum += rows.product_sum;
            total.ring_mul += rows.ring_mul;
            total.acceptance_tree_bit_pair += rows.acceptance_tree_bit_pair;
            total.acceptance_product_aggregate += rows.acceptance_product_aggregate;
            total.acceptance_root_binding += rows.acceptance_root_binding;
            total.packed_mod5_low_bit_pair += rows.packed_mod5_low_bit_pair;
            total.packed_mod5_high_bit_pair += rows.packed_mod5_high_bit_pair;
            total.packed_mod5_residue_pair += rows.packed_mod5_residue_pair;
            total.selection_accept_aggregate += rows.selection_accept_aggregate;
            total.selection_prefix_aggregate += rows.selection_prefix_aggregate;
            total.selection_symbol_aggregate += rows.selection_symbol_aggregate;
            total
        })
}

fn visit_tree(
    node: &'static str,
    children_by_parent: &BTreeMap<&'static str, &[&'static str]>,
    visiting: &mut BTreeSet<&'static str>,
    reachable: &mut BTreeSet<&'static str>,
) {
    assert!(visiting.insert(node), "cost hierarchy contains a cycle at {node}");
    assert!(reachable.insert(node), "cost hierarchy reaches {node} more than once");
    if let Some(children) = children_by_parent.get(node) {
        for &child in *children {
            visit_tree(child, children_by_parent, visiting, reachable);
        }
    }
    visiting.remove(node);
}

fn aggregate_tree_node(
    node: &'static str,
    exact_by_label: &BTreeMap<&'static str, GadgetNativeStageEstimate>,
    children_by_parent: &BTreeMap<&'static str, &[&'static str]>,
) -> GadgetNativeStageEstimate {
    let mut total = exact_by_label
        .get(node)
        .unwrap_or_else(|| panic!("missing exact stage {node}"))
        .clone();
    if let Some(children) = children_by_parent.get(node) {
        for &child in *children {
            merge_stage(
                &mut total,
                &aggregate_tree_node(child, exact_by_label, children_by_parent),
            );
        }
    }
    total
}

fn merge_stage(total: &mut GadgetNativeStageEstimate, child: &GadgetNativeStageEstimate) {
    macro_rules! merge_fields {
        ($($field:ident),+ $(,)?) => {
            $(total.$field += child.$field;)+
        };
    }
    merge_fields!(
        occurrences,
        source_rows,
        source_cols,
        one_bit_source_cols,
        canonical_binary_field_source_cols,
        ordinary_private_field_source_cols,
        balanced_ternary_field_source_cols,
        balanced_ternary_alias_source_cols,
        balanced_ternary_binary_source_cols,
        linearly_derived_source_cols,
        gadget_derived_source_cols,
        synthetic_ring_fields,
        synthetic_product_sum_fields,
        acceptance_chunks,
        acceptance_encoded_cols,
        acceptance_tree_output_cols,
        acceptance_tree_bit_pair_rows,
        acceptance_product_aggregate_rows,
        acceptance_root_binding_rows,
        packed_mod5_chunks,
        packed_mod5_encoded_cols,
        packed_mod5_low_bit_pair_rows,
        packed_mod5_high_bit_pair_rows,
        packed_mod5_residue_pair_rows,
        encoded_cols,
        centered_encoded_cols,
        ordinary_private_encoded_cols,
        sis_centered_encoded_cols,
        encoded_rows,
        redundant_boolean_source_rows,
        fallback_source_rows,
        poseidon_permutations,
        poseidon_hash_permutations,
        poseidon_hashes,
        sboxes,
        k_muls,
        product_sum_batches,
        product_sum_identities,
        product_sum_rows,
        ring_muls,
        selection_accept_aggregate_rows,
        selection_prefix_aggregate_rows,
        selection_symbol_aggregate_rows,
    );
    add_pair_tail(&mut total.boolean_pairing.common, child.boolean_pairing.common);
    add_pair_tail(
        &mut total.boolean_pairing.source_raw64,
        child.boolean_pairing.source_raw64,
    );
    add_pair_tail(
        &mut total.boolean_pairing.source_prefix31,
        child.boolean_pairing.source_prefix31,
    );
    add_pair_tail(
        &mut total.boolean_pairing.synthetic_ring_raw64,
        child.boolean_pairing.synthetic_ring_raw64,
    );
    add_pair_tail(
        &mut total.boolean_pairing.synthetic_ring_prefix31,
        child.boolean_pairing.synthetic_ring_prefix31,
    );
    add_pair_tail(
        &mut total.boolean_pairing.synthetic_product_sum_raw64,
        child.boolean_pairing.synthetic_product_sum_raw64,
    );
    add_pair_tail(
        &mut total.boolean_pairing.synthetic_product_sum_prefix31,
        child.boolean_pairing.synthetic_product_sum_prefix31,
    );
    add_pair_tail(&mut total.centered_pairing, child.centered_pairing);
    add_pair_tail(
        &mut total.ordinary_private_centered_pairing,
        child.ordinary_private_centered_pairing,
    );
    add_pair_tail(&mut total.sis_centered_pairing, child.sis_centered_pairing);
    for (&input_len, &(calls, permutations)) in &child.hash_histogram {
        let entry = total.hash_histogram.entry(input_len).or_default();
        entry.0 += calls;
        entry.1 += permutations;
    }
}

fn assert_complete_profile_total(profile: &GadgetNativeStageProfile, root: &GadgetNativeStageEstimate) {
    let total = profile.total;
    assert_eq!(root.source_rows, total.source_rows, "complete source-row ownership");
    assert_eq!(
        root.source_cols + 1,
        total.source_cols,
        "complete source-column ownership"
    );
    assert_eq!(root.encoded_rows, total.encoded_rows, "complete encoded-row ownership");
    assert_eq!(
        root.encoded_cols + 1,
        total.encoded_cols,
        "complete encoded-column ownership"
    );
    macro_rules! assert_total_fields {
        ($($field:ident),+ $(,)?) => {
            $(assert_eq!(root.$field, total.$field, "complete {} ownership", stringify!($field));)+
        };
    }
    assert_total_fields!(
        one_bit_source_cols,
        canonical_binary_field_source_cols,
        ordinary_private_field_source_cols,
        balanced_ternary_field_source_cols,
        balanced_ternary_alias_source_cols,
        balanced_ternary_binary_source_cols,
        centered_encoded_cols,
        ordinary_private_encoded_cols,
        sis_centered_encoded_cols,
        synthetic_ring_fields,
        synthetic_product_sum_fields,
        acceptance_chunks,
        acceptance_encoded_cols,
        acceptance_tree_output_cols,
        acceptance_tree_bit_pair_rows,
        acceptance_product_aggregate_rows,
        acceptance_root_binding_rows,
        packed_mod5_chunks,
        packed_mod5_encoded_cols,
        packed_mod5_low_bit_pair_rows,
        packed_mod5_high_bit_pair_rows,
        packed_mod5_residue_pair_rows,
        linearly_derived_source_cols,
        gadget_derived_source_cols,
        redundant_boolean_source_rows,
        fallback_source_rows,
    );
    assert_eq!(
        root.boolean_pairing, total.boolean_pairing,
        "complete Boolean pairing ownership"
    );
    assert_eq!(
        root.centered_pairing, total.centered_pairing,
        "complete centered pairing ownership"
    );
    assert_eq!(
        root.ordinary_private_centered_pairing, total.ordinary_private_centered_pairing,
        "complete ordinary-private centered pairing ownership"
    );
    assert_eq!(
        root.sis_centered_pairing, total.sis_centered_pairing,
        "complete SIS centered pairing ownership"
    );
    assert_eq!(
        root.encoded_row_breakdown().total(),
        total.encoded_rows,
        "complete encoded row-family ownership"
    );
}

fn assert_zero_cost_checkpoint(stage: &GadgetNativeStageEstimate) {
    macro_rules! assert_zero {
        ($($field:ident),+ $(,)?) => {
            $(assert_eq!(
                stage.$field, 0,
                "{} organizational checkpoint must not own {}",
                stage.label,
                stringify!($field)
            );)+
        };
    }
    assert!(stage.occurrences > 0, "{} checkpoint must be emitted", stage.label);
    assert_zero!(
        source_rows,
        source_cols,
        one_bit_source_cols,
        canonical_binary_field_source_cols,
        ordinary_private_field_source_cols,
        balanced_ternary_field_source_cols,
        balanced_ternary_alias_source_cols,
        balanced_ternary_binary_source_cols,
        linearly_derived_source_cols,
        gadget_derived_source_cols,
        synthetic_ring_fields,
        synthetic_product_sum_fields,
        acceptance_chunks,
        acceptance_encoded_cols,
        acceptance_tree_output_cols,
        acceptance_tree_bit_pair_rows,
        acceptance_product_aggregate_rows,
        acceptance_root_binding_rows,
        packed_mod5_chunks,
        packed_mod5_encoded_cols,
        packed_mod5_low_bit_pair_rows,
        packed_mod5_high_bit_pair_rows,
        packed_mod5_residue_pair_rows,
        encoded_cols,
        centered_encoded_cols,
        ordinary_private_encoded_cols,
        sis_centered_encoded_cols,
        encoded_rows,
        redundant_boolean_source_rows,
        fallback_source_rows,
        poseidon_permutations,
        poseidon_hash_permutations,
        poseidon_hashes,
        sboxes,
        k_muls,
        product_sum_batches,
        product_sum_identities,
        product_sum_rows,
        ring_muls,
        selection_accept_aggregate_rows,
        selection_prefix_aggregate_rows,
        selection_symbol_aggregate_rows,
    );
    assert!(
        stage.centered_pairing == GadgetNativePairTailCount::default(),
        "{} organizational checkpoint centered pairing",
        stage.label
    );
    assert!(
        stage.ordinary_private_centered_pairing == GadgetNativePairTailCount::default(),
        "{} organizational checkpoint ordinary-private centered pairing",
        stage.label
    );
    assert!(
        stage.sis_centered_pairing == GadgetNativePairTailCount::default(),
        "{} organizational checkpoint SIS centered pairing",
        stage.label
    );
    assert!(
        stage.hash_histogram.is_empty(),
        "{} checkpoint hash histogram",
        stage.label
    );
}

/// Pin the two selected SIS-backed stages whose generic canonical-field lowering
/// dominates the complete recursive fixture. These snapshots deliberately
/// separate source arithmetic from encoding tax: a future selective lowering
/// must change the field classes and row families here, not merely the total.
pub(super) fn assert_dominant_sis_snapshots(profile: &GadgetNativeStageProfile) {
    struct Snapshot {
        path: &'static str,
        source_rows: usize,
        source_cols: usize,
        encoded_rows: usize,
        encoded_cols: usize,
        one_bit_fields: usize,
        canonical_fields: usize,
        ordinary_private_fields: usize,
        balanced_fields: usize,
        balanced_aliases: usize,
        balanced_binary: usize,
        centered_coords: usize,
        linear_fields: usize,
        gadget_fields: usize,
        fallback_rows: usize,
        sbox_rows: usize,
        common_boolean: GadgetNativePairTailCount,
        source_raw64: GadgetNativePairTailCount,
        source_prefix31: GadgetNativePairTailCount,
    }

    let expected = [
        Snapshot {
            path: "nifs.pi_ccs.padded_row.output_digest.sis",
            source_rows: 852_526,
            source_cols: 838_944,
            encoded_rows: 728_780,
            encoded_cols: 890_658,
            one_bit_fields: 550_071,
            canonical_fields: 0,
            ordinary_private_fields: 1_516,
            balanced_fields: 6_791,
            balanced_aliases: 278_431,
            balanced_binary: 550_071,
            centered_coords: 340_587,
            linear_fields: 4_432,
            gadget_fields: 4_386,
            fallback_rows: 557_024,
            sbox_rows: 1_462,
            common_boolean: GadgetNativePairTailCount {
                coordinates: 0,
                pair_rows: 0,
                tail_rows: 0,
            },
            source_raw64: GadgetNativePairTailCount {
                coordinates: 0,
                pair_rows: 0,
                tail_rows: 0,
            },
            source_prefix31: GadgetNativePairTailCount {
                coordinates: 0,
                pair_rows: 0,
                tail_rows: 0,
            },
        },
        Snapshot {
            path: "nifs.pi_rlc.verify.projection_binding.sis_digest",
            source_rows: 472_218,
            source_cols: 464_770,
            encoded_rows: 414_412,
            encoded_cols: 516_484,
            one_bit_fields: 301_644,
            canonical_fields: 0,
            ordinary_private_fields: 1_516,
            balanced_fields: 3_724,
            balanced_aliases: 152_684,
            balanced_binary: 301_644,
            centered_coords: 214_840,
            linear_fields: 4_432,
            gadget_fields: 4_386,
            fallback_rows: 305_530,
            sbox_rows: 1_462,
            common_boolean: GadgetNativePairTailCount {
                coordinates: 0,
                pair_rows: 0,
                tail_rows: 0,
            },
            source_raw64: GadgetNativePairTailCount {
                coordinates: 0,
                pair_rows: 0,
                tail_rows: 0,
            },
            source_prefix31: GadgetNativePairTailCount {
                coordinates: 0,
                pair_rows: 0,
                tail_rows: 0,
            },
        },
    ];

    for expected in expected {
        let actual = profile
            .aggregate_prefix(expected.path)
            .unwrap_or_else(|| panic!("missing dominant SIS stage {}", expected.path));
        eprintln!(
            "FPRIME_SIS|{}|source={}x{}|encoded={}x{}|bits={}|canonical={}|ordinary={}|balanced_fields={}|balanced_aliases={}|balanced_binary={}|centered_coords={}|linear={}|gadget={}|rows={:?}",
            expected.path,
            actual.source_rows,
            actual.source_cols,
            actual.encoded_rows,
            actual.encoded_cols,
            actual.one_bit_source_cols,
            actual.canonical_binary_field_source_cols,
            actual.ordinary_private_field_source_cols,
            actual.balanced_ternary_field_source_cols,
            actual.balanced_ternary_alias_source_cols,
            actual.balanced_ternary_binary_source_cols,
            actual.centered_encoded_cols,
            actual.linearly_derived_source_cols,
            actual.gadget_derived_source_cols,
            actual.encoded_row_breakdown(),
        );
        assert_eq!(
            actual.source_rows, expected.source_rows,
            "{} source rows",
            expected.path
        );
        assert_eq!(
            actual.source_cols, expected.source_cols,
            "{} source columns",
            expected.path
        );
        assert_eq!(
            actual.encoded_rows, expected.encoded_rows,
            "{} encoded rows",
            expected.path
        );
        assert_eq!(
            actual.encoded_cols, expected.encoded_cols,
            "{} encoded columns",
            expected.path
        );
        assert_eq!(
            actual.one_bit_source_cols, expected.one_bit_fields,
            "{} one-bit fields",
            expected.path
        );
        assert_eq!(
            actual.canonical_binary_field_source_cols, expected.canonical_fields,
            "{} canonical fields",
            expected.path
        );
        assert_eq!(
            actual.ordinary_private_field_source_cols, expected.ordinary_private_fields,
            "{} ordinary-private fields",
            expected.path
        );
        assert_eq!(
            actual.balanced_ternary_field_source_cols, expected.balanced_fields,
            "{} balanced fields",
            expected.path
        );
        assert_eq!(
            actual.balanced_ternary_alias_source_cols, expected.balanced_aliases,
            "{} balanced aliases",
            expected.path
        );
        assert_eq!(
            actual.balanced_ternary_binary_source_cols, expected.balanced_binary,
            "{} balanced binary auxiliaries",
            expected.path
        );
        assert_eq!(
            actual.centered_encoded_cols, expected.centered_coords,
            "{} centered coordinates",
            expected.path
        );
        assert_eq!(
            actual.linearly_derived_source_cols, expected.linear_fields,
            "{} linear fields",
            expected.path
        );
        assert_eq!(
            actual.gadget_derived_source_cols, expected.gadget_fields,
            "{} gadget fields",
            expected.path
        );
        assert_eq!(
            actual.encoded_row_breakdown(),
            GadgetNativeEncodedRowBreakdown {
                common_boolean: expected.common_boolean,
                common_centered_unit: pair_tail(expected.centered_coords),
                ordinary_private_centered_unit: pair_tail(expected.ordinary_private_fields * ORDINARY_PRIVATE_DIGITS,),
                sis_centered_unit: pair_tail(expected.balanced_aliases),
                canonical_binary_source_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown {
                    raw_bits: expected.source_raw64,
                    prefix_aux: expected.source_prefix31,
                    canonicality_relations: expected.canonical_fields * CANONICALITY_RELATIONS_PER_SLOT,
                    canonicality_pair_rows: expected.canonical_fields * CANONICALITY_PAIR_ROWS_PER_SLOT,
                },
                fallback: expected.fallback_rows,
                sbox: expected.sbox_rows,
                ..GadgetNativeEncodedRowBreakdown::default()
            },
            "{} row-family breakdown",
            expected.path
        );
    }
}

fn print_tree(
    exact_by_label: &BTreeMap<&'static str, GadgetNativeStageEstimate>,
    children_by_parent: &BTreeMap<&'static str, &[&'static str]>,
    paths: &[&'static str],
    tree_prefix: &str,
) {
    print_stage_cost_header(tree_prefix);
    for &path in paths {
        let node = aggregate_tree_node(path, exact_by_label, children_by_parent);
        print_stage_cost_line(tree_prefix, path, &node);
    }
}

pub(super) fn print_stage_cost_families(profile: &GadgetNativeStageProfile) {
    print_stage_cost_header("FPRIME_STAGE");
    for stage in profile.aggregate_by_label() {
        print_stage_cost_line("FPRIME_STAGE", stage.label, &stage);
        assert_eq!(
            stage.encoded_row_breakdown().total(),
            stage.encoded_rows,
            "{} row ownership",
            stage.label
        );
    }
}
