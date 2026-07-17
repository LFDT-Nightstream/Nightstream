//! Π_DEC.V circuit gadget — native vs circuit parity + tamper tests.
//!
//! Drives a real NIFS round trip via `support::toy_preprocessing`, then
//! feeds the resulting `(pi_rlc.combined, pi_dec.children)` pair into
//! the Π_DEC.V circuit and checks constraint satisfaction.

#[path = "../support/mod.rs"]
mod support;

use std::collections::BTreeSet;

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::pi_dec;
use neo_fold_clean::paper::reductions::pi_dec_circuit::{
    alloc_dec_inputs, enforce_dec_v, enforce_dec_v_strict, enforce_r_consistency, enforce_x_bitness,
    stage as pi_dec_stage, DecInputWires,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

#[test]
fn pi_dec_circuit_accepts_honest_decomposition() {
    let (proof, _claims) = drive_nifs(7);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v(&mut builder, &prep.params, &wires).expect("dec_v emit");

    assert!(
        builder.is_satisfied(),
        "circuit must accept the native (parent, children) — first failing row {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn pi_dec_circuit_rejects_tampered_child_commitment() {
    let (proof, _claims) = drive_nifs(11);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v(&mut builder, &prep.params, &wires).expect("dec_v emit");

    // Tamper child 0's first commitment lane (witness column).
    let target_col = wires.children[0].c_data[0].col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        !builder.is_satisfied(),
        "circuit accepted a tampered child commitment lane"
    );
}

#[test]
fn pi_dec_circuit_rejects_tampered_y_ring_lane() {
    let (proof, _claims) = drive_nifs(13);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v(&mut builder, &prep.params, &wires).expect("dec_v emit");

    // Tamper child 0's first y_ring lane (first base-field limb).
    assert!(
        !wires.children[0].y_ring.is_empty() && !wires.children[0].y_ring[0].is_empty(),
        "test fixture must expose child y_ring lanes"
    );
    let target_col = wires.children[0].y_ring[0][0].col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(!builder.is_satisfied(), "circuit accepted a tampered child y_ring lane");
}

#[test]
fn pi_dec_circuit_rejects_tampered_parent_commitment() {
    let (proof, _claims) = drive_nifs(17);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v(&mut builder, &prep.params, &wires).expect("dec_v emit");

    let target_col = wires.parent.c_data[0].col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        !builder.is_satisfied(),
        "circuit accepted a tampered parent commitment lane"
    );
}

#[test]
fn pi_dec_circuit_rejects_wrong_child_count() {
    let (proof, _claims) = drive_nifs(19);

    let parent = &proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children.clone();
    children.pop(); // shrink by one

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, &children);
    let result = enforce_dec_v(&mut builder, &prep.params, &wires);

    assert!(result.is_err(), "enforce_dec_v accepted a child set of the wrong arity");
}

// ── Strict Π_DEC.V: bitness + r-consistency in one call ─────────────────

#[test]
fn pi_dec_circuit_strict_accepts_honest_decomposition() {
    let (proof, _claims) = drive_nifs(23);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");

    assert!(
        builder.is_satisfied(),
        "strict Π_DEC.V must accept native (parent, children) — first failing row {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn pi_dec_strict_leaf_ranges_partition_every_emitted_row() {
    let (proof, _claims) = drive_nifs(25);
    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &proof.pi_dec.children);
    let verifier_start = builder.rows();
    let claim_count = 1 + wires.children.len();
    let allocation_stats = |path| {
        builder
            .row_family_ranges()
            .iter()
            .filter(|range| range.name == path)
            .fold((0usize, 0usize), |(count, rows), range| {
                (count + 1, rows + range.row_end - range.row_start)
            })
    };
    assert_eq!(
        allocation_stats(pi_rlc_stage::ROW_SHAPE_ALLOCATE_INACTIVE_X_SENTINEL),
        (claim_count, claim_count),
        "one inactive-X sentinel row per allocated claim"
    );
    assert_eq!(
        allocation_stats(pi_rlc_stage::ROW_SHAPE_ALLOCATE_FOLD_DIGEST_CANONICALITY),
        (claim_count, 0),
        "canonical honest digests emit no rejection rows"
    );
    assert_eq!(
        allocation_stats(pi_rlc_stage::ROW_SHAPE_ALLOCATE_METADATA),
        (claim_count, 5 * claim_count),
        "five fixed shape-metadata rows per allocated claim"
    );
    assert_eq!(
        verifier_start,
        6 * claim_count,
        "allocation row children must own the entire pre-verifier prefix"
    );
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");

    let mut leaves = pi_dec_stage::LEAVES
        .iter()
        .map(|&path| {
            let matches = builder
                .row_family_ranges()
                .iter()
                .filter(|range| range.name == path)
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1, "{path} must have exactly one row owner");
            *matches[0]
        })
        .collect::<Vec<_>>();
    leaves.sort_by_key(|range| (range.row_start, range.row_end));

    let mut cursor = verifier_start;
    for range in leaves {
        assert_eq!(range.row_start, cursor, "{} leaves a row-ownership gap", range.name);
        assert!(range.row_end >= range.row_start, "{} has a reversed range", range.name);
        cursor = range.row_end;
    }
    assert_eq!(
        cursor,
        builder.rows(),
        "Pi_DEC leaves must own every strict-verifier row"
    );

    let verify = builder
        .row_family_ranges()
        .iter()
        .find(|range| range.name == pi_dec_stage::VERIFY)
        .expect("verify parent range");
    assert_eq!((verify.row_start, verify.row_end), (verifier_start, builder.rows()));
    assert_eq!(
        pi_dec_stage::ROW_ALL
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        pi_dec_stage::ROW_ALL.len(),
        "Pi_DEC row-overlay paths must be unique"
    );
    for &(parent, children) in pi_dec_stage::ROW_HIERARCHY {
        assert!(pi_dec_stage::ROW_ALL.contains(&parent), "missing row parent {parent}");
        assert!(!children.is_empty(), "row parent {parent} must own children");
        for child in children {
            assert!(pi_dec_stage::ROW_ALL.contains(child), "missing row child {child}");
        }
    }

    let recomposition = builder
        .row_family_ranges()
        .iter()
        .find(|range| range.name == pi_dec_stage::RECOMPOSITION)
        .expect("recomposition parent range");
    let recomposition_rows = [
        pi_dec_stage::RECOMPOSITION_COMMITMENT,
        pi_dec_stage::RECOMPOSITION_ADVICE,
        pi_dec_stage::RECOMPOSITION_X,
        pi_dec_stage::RECOMPOSITION_Y_RING,
    ]
    .iter()
    .map(|path| {
        let range = builder
            .row_family_ranges()
            .iter()
            .find(|range| range.name == *path)
            .expect("recomposition child range");
        range.row_end - range.row_start
    })
    .sum::<usize>();
    assert_eq!(
        recomposition.row_end - recomposition.row_start,
        recomposition_rows,
        "recomposition parent must equal its immediate-child sum"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_noncanonical_fold_digest_limb_alias() {
    let (proof, _claims) = drive_nifs(31);

    let mut parent = proof.pi_rlc.combined.clone();
    let mut children = proof.pi_dec.children.clone();
    parent.fold_digest[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());
    for child in &mut children {
        child.fold_digest[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());
    }

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict DEC emit");

    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V accepted a noncanonical fold_digest limb aliasing to zero"
    );
}

#[test]
fn pi_dec_circuit_strict_leaves_only_parent_y_zcol_unconstrained() {
    // Π_DEC owns the b-ary parent→children recomposition for commitment, X,
    // r, y_ring, ct, s_col, and fold_digest. It currently omits y_zcol even
    // though the optimized raw projection is linear and admits the same
    // radix-b recomposition. This test records that known gap: child sidecars
    // are not allocated, while the parent sidecar floats. If any other
    // allocated wire floats, the strict DEC gadget is missing an additional
    // row family.
    let (proof, _claims) = drive_nifs(24);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");

    assert!(
        builder.is_satisfied(),
        "strict Π_DEC.V must accept native (parent, children) before unconstrained-column audit"
    );

    let unconstrained: BTreeSet<_> = builder.unconstrained_columns().into_iter().collect();
    assert!(
        wires.children.iter().all(|child| child.y_zcol.is_empty()),
        "strict Π_DEC children must not allocate y_zcol"
    );
    let allowed = parent_y_zcol_columns(&wires);
    assert!(
        unconstrained == allowed,
        "strict Π_DEC.V left unexpected unconstrained columns: got {unconstrained:?}, \
         expected exactly y_zcol sidecars {allowed:?}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_child_ct_not_derived_from_y_ring() {
    let (proof, _claims) = drive_nifs(24);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline must satisfy");

    assert!(!wires.children[0].ct.is_empty(), "fixture must expose child ct wires");
    let target_col = wires.children[0].ct[0].c0.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V accepted child ct that no longer equals y_ring[j][lane=0]"
    );
}

#[test]
fn pi_dec_native_rejects_child_ct_not_derived_from_y_ring() {
    let (proof, _claims) = drive_nifs(24);

    let parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    assert!(!children[0].ct.is_empty(), "fixture must expose child ct");
    children[0].ct[0] += K::ONE;

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted child ct that no longer equals y_ring[j][lane=0]");
    assert!(
        matches!(err, pi_dec::Error::CtConsistency("child")),
        "expected child ct-consistency rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_native_rejects_child_s_col_relabel() {
    let (proof, _claims) = drive_nifs(47);

    let parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    assert!(!children[0].s_col.is_empty(), "fixture must expose child s_col");
    children[0].s_col[0] += K::ONE;

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted a child s_col that diverges from parent.s_col");
    assert!(
        matches!(err, pi_dec::Error::SColConsistency),
        "expected s_col-consistency rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_native_rejects_noncanonical_fold_digest_limb_alias() {
    let (proof, _claims) = drive_nifs(52);

    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    let mut noncanonical_zero = [0u8; 32];
    noncanonical_zero[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());
    parent.fold_digest = noncanonical_zero;
    for child in &mut children {
        child.fold_digest = noncanonical_zero;
    }

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted a noncanonical fold_digest limb aliasing to zero");
    assert!(
        matches!(
            err,
            pi_dec::Error::FoldDigestCanonicality {
                owner: "parent",
                lane: 0
            }
        ),
        "expected parent fold-digest canonicality rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_native_rejects_extra_self_consistent_s_col_limb() {
    let (proof, _claims) = drive_nifs(48);

    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    parent.s_col.push(K::ZERO);
    for child in &mut children {
        child.s_col.push(K::ZERO);
    }

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted an extra self-consistent s_col limb");
    assert!(
        matches!(
            err,
            pi_dec::Error::SColShape("parent") | pi_dec::Error::SColShape("child")
        ),
        "expected s_col shape rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_native_rejects_extra_self_consistent_r_limb() {
    let (proof, _claims) = drive_nifs(49);

    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    parent.r.push(K::ZERO);
    for child in &mut children {
        child.r.push(K::ZERO);
    }

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted an extra self-consistent r limb");
    assert!(
        matches!(err, pi_dec::Error::RShape("parent") | pi_dec::Error::RShape("child")),
        "expected r shape rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_native_rejects_self_consistent_parent_child_y_ring_padding_lane() {
    let (proof, _claims) = drive_nifs(62);

    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_ring lanes");
    parent.y_ring[0][D] += K::ONE;
    children[0].y_ring[0][D] += K::ONE;

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted self-consistent nonzero y_ring padding");
    assert!(
        matches!(
            err,
            pi_dec::Error::YRingPadding("parent") | pi_dec::Error::YRingPadding("child")
        ),
        "expected y_ring padding rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_native_rejects_extra_self_consistent_y_ring_row() {
    let (proof, _claims) = drive_nifs(63);

    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    let extra_row = vec![K::ZERO; D.next_power_of_two()];
    parent.y_ring.push(extra_row.clone());
    parent.ct.push(K::ZERO);
    for child in &mut children {
        child.y_ring.push(extra_row.clone());
        child.ct.push(K::ZERO);
    }

    let prep = support::toy_preprocessing();
    let err = pi_dec::verify(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        &parent,
        &pi_dec::Proof { children },
    )
    .expect_err("native Π_DEC.V accepted an extra self-consistent y_ring/ct row");
    assert!(
        matches!(
            err,
            pi_dec::Error::YRingShape("parent") | pi_dec::Error::YRingShape("child")
        ),
        "expected y_ring row-count rejection, got {err:?}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_parent_ct_c1_not_derived_from_y_ring() {
    let (proof, _claims) = drive_nifs(25);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline must satisfy");

    assert!(!wires.parent.ct.is_empty(), "fixture must expose parent ct wires");
    let target_col = wires.parent.ct[0].c1.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V accepted parent ct.c1 that no longer equals y_ring[j][lane=0].c1"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_canceling_child_y_ring_padding_lanes() {
    // Plain DEC recomposition only checks the b-weighted sum. Two children can
    // therefore carry nonzero padded y_ring lanes that cancel in the parent.
    // Strict DEC must reject that non-canonical output because children become
    // the next running accumulator.
    let (proof, _claims) = drive_nifs(61);

    let parent = &proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children.clone();
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_ring lanes");
    assert!(children.len() >= 2, "fixture must have at least two DEC children");
    assert!(
        !children[0].y_ring.is_empty() && children[0].y_ring[0].len() == d_pad,
        "fixture must expose full padded child y_ring rows"
    );

    let b_inv = K::from_u64(support::toy_preprocessing().params.b() as u64).inverse();
    children[0].y_ring[0][D] += K::ONE;
    children[1].y_ring[0][D] -= b_inv;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V accepted canceling nonzero child y_ring padding lanes"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_self_consistent_parent_child_y_ring_padding_lane() {
    // Stronger than a one-sided tamper: make the parent and first child
    // agree on a nonzero padded y_ring lane, so plain b-ary recomposition
    // still passes (`b^0 = 1`). Strict DEC must reject the padded lane
    // itself as non-canonical CE data.
    let (proof, _claims) = drive_nifs(62);

    let mut parent = proof.pi_rlc.combined.clone();
    let mut children = proof.pi_dec.children.clone();
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_ring lanes");
    assert!(
        !parent.y_ring.is_empty() && parent.y_ring[0].len() == d_pad,
        "fixture must expose full padded parent y_ring rows"
    );
    assert!(
        !children[0].y_ring.is_empty() && children[0].y_ring[0].len() == d_pad,
        "fixture must expose full padded child y_ring rows"
    );

    parent.y_ring[0][D] += K::ONE;
    children[0].y_ring[0][D] += K::ONE;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V accepted self-consistent nonzero parent/child y_ring padding lanes"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_parent_aux_openings_sidecar() {
    let (proof, _claims) = drive_nifs(25);

    let mut parent = proof.pi_rlc.combined.clone();
    parent.aux_openings.push(K::ONE);
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, children);
    let err = enforce_dec_v_strict(&mut builder, &prep.params, &wires)
        .err()
        .expect("strict DEC must reject unsupported parent aux_openings");
    assert!(
        err.to_string().contains("aux_openings"),
        "expected aux_openings shape error, got {err}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_child_aux_openings_sidecar() {
    let (proof, _claims) = drive_nifs(26);

    let parent = &proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children.clone();
    children[0].aux_openings.push(K::ONE);

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, &children);
    let err = enforce_dec_v_strict(&mut builder, &prep.params, &wires)
        .err()
        .expect("strict DEC must reject unsupported child aux_openings");
    assert!(
        err.to_string().contains("aux_openings"),
        "expected aux_openings shape error, got {err}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_pattern_a_metadata_sidecar() {
    let (proof, _claims) = drive_nifs(27);

    let mut parent = proof.pi_rlc.combined.clone();
    parent.c_step_coords.push(F::ONE);
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, children);
    let err = enforce_dec_v_strict(&mut builder, &prep.params, &wires)
        .err()
        .expect("strict DEC must reject unsupported Pattern-A metadata");
    assert!(
        err.to_string().contains("c_step_coords"),
        "expected c_step_coords shape error, got {err}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_parent_commitment_shape_metadata_drift() {
    let (proof, _claims) = drive_nifs(28);

    let mut parent = proof.pi_rlc.combined.clone();
    parent.c.kappa += 1;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, children);
    let err = enforce_dec_v_strict(&mut builder, &prep.params, &wires)
        .err()
        .expect("strict DEC must reject parent commitment metadata drift");
    assert!(
        err.to_string().contains("parent commitment lane count"),
        "expected parent commitment lane-count shape error, got {err}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_child_commitment_shape_metadata_drift() {
    let (proof, _claims) = drive_nifs(30);

    let parent = &proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children.clone();
    children[0].c.kappa += 1;
    children[0]
        .c
        .data
        .extend(std::iter::repeat(F::ZERO).take(D));

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, &children);
    let err = enforce_dec_v_strict(&mut builder, &prep.params, &wires)
        .err()
        .expect("strict DEC must reject child commitment metadata drift");
    assert!(
        err.to_string().contains("child commitment kappa"),
        "expected child commitment-kappa shape error, got {err}"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_tampered_child_shape_metadata_wire() {
    let (proof, _claims) = drive_nifs(32);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict DEC emit");
    assert!(builder.is_satisfied(), "baseline must satisfy");

    let target_col = wires.children[0].c_kappa_var.col();
    let tampered = builder.witness()[target_col] + F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V accepted a child c.kappa metadata wire that diverges from the parent"
    );
}

#[test]
fn pi_dec_circuit_r_consistency_accepts_native_shared_r() {
    let (proof, _claims) = drive_nifs(29);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    // Sanity: native really does emit children with parent.r.
    for child in children {
        assert_eq!(child.r, parent.r, "native NIFS should emit children sharing parent.r");
    }

    let _prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_r_consistency(&mut builder, &wires).expect("r-consistency emit");

    assert!(
        builder.is_satisfied(),
        "r-consistency must accept native r alignment — first failing row {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn pi_dec_circuit_r_consistency_rejects_tampered_child_r() {
    let (proof, _claims) = drive_nifs(31);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let _prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_r_consistency(&mut builder, &wires).expect("r-consistency emit");
    assert!(builder.is_satisfied(), "baseline");

    // Tamper child 0's first r limb. The equality `parent.r[0].c0 ==
    // child_0.r[0].c0` must fail.
    assert!(
        !wires.children[0].r.is_empty(),
        "test fixture must expose child r lanes"
    );
    let target_col = wires.children[0].r[0].c0.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);
    assert!(
        !builder.is_satisfied(),
        "r-consistency must reject a child whose r diverges from parent"
    );
}

#[test]
fn pi_dec_circuit_x_bitness_rejects_out_of_range_x() {
    let (proof, _claims) = drive_nifs(37);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_x_bitness(&mut builder, &prep.params, &wires);
    assert!(builder.is_satisfied(), "baseline (all child x in {{0..b-1}})");

    // Tamper child 0's first x var to `b` (one above the allowed range).
    let b = prep.params.b();
    let target_col = wires.children[0].x[0].col();
    builder.tamper_witness(target_col, neo_math::F::from_u64(b as u64));
    assert!(
        !builder.is_satisfied(),
        "bitness check must reject child x = b (above {{0..b-1}})"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_tampered_child_r() {
    let (proof, _claims) = drive_nifs(41);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline");

    assert!(
        !wires.children[0].r.is_empty(),
        "test fixture must expose child r lanes"
    );
    let target_col = wires.children[0].r[0].c1.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V must reject a child whose r diverges"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_child_x_recomposition_mismatch() {
    let (proof, _claims) = drive_nifs(43);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline");

    // Mutate one child X without updating the parent. Strict Π_DEC.V does
    // not own unsigned X bitness; this rejection is the b-ary recomposition
    // row. The separate `pi_dec_circuit_x_bitness_rejects_out_of_range_x`
    // test covers callers that opt into unsigned range rows.
    let b = prep.params.b();
    let target_col = wires.children[0].x[0].col();
    builder.tamper_witness(target_col, neo_math::F::from_u64(b as u64));
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V must reject a child X value that no longer recomposes to the parent"
    );
}

// ── SplitNc NC-channel tamper tests ──────────────────────────────────────

#[test]
fn pi_dec_circuit_strict_rejects_tampered_child_s_col() {
    // s_col is shared between parent and all children (NC column-domain
    // point). Strict mode adds `enforce_s_col_consistency`; tampering one
    // lane of one child must break it.
    let (proof, _claims) = drive_nifs(47);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(
        builder.is_satisfied(),
        "baseline (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    assert!(
        !wires.children[0].s_col.is_empty(),
        "test fixture must expose child s_col lanes"
    );
    let target_col = wires.children[0].s_col[0].c0.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(!builder.is_satisfied(), "strict Π_DEC.V accepted tampered child s_col");
}

#[test]
fn pi_dec_circuit_rejects_nonzero_inactive_child_x() {
    // `enforce_dec_v_strict` includes `enforce_inactive_x_zero`, which pins
    // each child's `X[r, c]` to zero for `c >= ceil(m_in / D)`. Tampering
    // inactive slots must break strict Π_DEC.V on its own. The tamper below
    // is recomposition-canceling (`child0 = b`, `child1 = -1`, parent = 0),
    // so the ordinary b-ary X equation still holds; only the inactive-X rows
    // can reject.
    let (proof, _claims) = drive_nifs(59);
    let mut parent = proof.pi_rlc.combined.clone();
    let mut children = proof.pi_dec.children.clone();
    assert!(children.len() >= 2, "fixture must expose at least two DEC children");

    let widen_x = |claim: &mut neo_fold_clean::CeClaim| {
        let mut widened = Mat::zero(D, 2, F::ZERO);
        for row in 0..D {
            widened[(row, 0)] = claim.X[(row, 0)];
        }
        claim.X = widened;
        claim.m_in = 2;
    };
    widen_x(&mut parent);
    for child in &mut children {
        widen_x(child);
    }

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline must satisfy");

    let child = &wires.children[0];
    let m_in = child.x_cols;
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(m_in);
    assert!(
        active_cols < m_in,
        "test fixture must expose inactive child X columns (active={active_cols}, m_in={m_in})"
    );
    let child0_col = child.x[active_cols].col();
    let child1_col = wires.children[1].x[active_cols].col();
    builder.tamper_witness(child0_col, F::from_u64(prep.params.b() as u64));
    builder.tamper_witness(child1_col, F::ZERO - F::ONE);
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V must reject recomposition-canceling non-zero inactive child X"
    );
}

#[test]
fn pi_dec_circuit_does_not_constrain_child_y_zcol() {
    // Native `verify_dec_public` does NOT enforce a parent/child `y_zcol`
    // relation. This test records the current emitted shape; it is not a
    // soundness claim. The NC authority audit demonstrates that terminal child
    // checks do not by themselves justify erasing the state-bound parent
    // old-point projection.
    let (proof, _claims) = drive_nifs(53);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;
    let mut mutated_children = children.clone();
    mutated_children[0].y_zcol[0] += K::ONE;

    let prep = support::toy_preprocessing();
    let mut baseline_builder = R1csBuilder::new();
    let baseline_wires = alloc_dec_inputs(&mut baseline_builder, parent, children);
    enforce_dec_v_strict(&mut baseline_builder, &prep.params, &baseline_wires).expect("baseline strict dec_v emit");
    assert!(
        baseline_builder.is_satisfied(),
        "baseline (first bad row: {:?})",
        baseline_builder.first_unsatisfied_row()
    );
    assert!(
        baseline_wires
            .children
            .iter()
            .all(|child| child.y_zcol.is_empty()),
        "strict Π_DEC must not allocate child y_zcol"
    );

    let mut mutated_builder = R1csBuilder::new();
    let mutated_wires = alloc_dec_inputs(&mut mutated_builder, parent, &mutated_children);
    enforce_dec_v_strict(&mut mutated_builder, &prep.params, &mutated_wires).expect("mutated strict dec_v emit");
    let baseline = baseline_builder.snapshot();
    let mutated = mutated_builder.snapshot();
    assert!(
        baseline.has_same_relation(&mutated),
        "native child y_zcol mutation changed the Π_DEC relation"
    );
    assert_eq!(
        baseline.witness(),
        mutated.witness(),
        "native child y_zcol mutation leaked into the Π_DEC witness"
    );
}

// ── helpers ───────────────────────────────────────────────────────────────

fn drive_nifs(seed: u64) -> (nifs::NifsProof, Vec<neo_fold_clean::CcsInstance>) {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, seed)];
    let claims = fresh.clone();
    let running = RunningInstance::default();

    let mut tr = Transcript::session();
    let (_next_running, proof) = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");
    (proof, claims)
}

fn parent_y_zcol_columns(wires: &DecInputWires) -> BTreeSet<usize> {
    wires.parent.y_zcol.iter().map(|var| var.col()).collect()
}
