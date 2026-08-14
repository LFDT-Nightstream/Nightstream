//! Π_DEC.V circuit gadget — native vs circuit parity + tamper tests.
//!
//! Drives a real NIFS round trip via `support::toy_preprocessing`, then
//! feeds the resulting `(pi_rlc.combined, pi_dec.children)` pair into
//! the Π_DEC.V circuit and checks constraint satisfaction.

#[path = "../support/mod.rs"]
mod support;

use std::collections::BTreeSet;

use neo_ccs::{CcsMatrix, CcsStructure, Mat, SparsePoly};
use neo_fold_clean::engine::r1cs_circuit::builder::{
    PiDecAdvAudit, PiDecClaimAudit, PiDecCommitmentAudit, PiDecStrictAudit,
};
use neo_fold_clean::engine::r1cs_circuit::{CanonicalSparseRow, R1csBuilder};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{strict_radix_accumulator_family_digest, AccumulatorHandle};
use neo_fold_clean::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::pi_dec;
use neo_fold_clean::paper::reductions::pi_dec_circuit::{
    alloc_dec_inputs, enforce_child_x_canonical_split, enforce_dec_v, enforce_dec_v_strict, enforce_r_consistency,
    enforce_x_bitness, stage as pi_dec_stage,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_fold_clean::{preprocess, Params, Preprocessing};
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};

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
fn pi_dec_strict_binary_schedule_has_the_proved_row_counts() {
    let (proof, _claims) = drive_nifs(24);
    let prep = support::toy_preprocessing();
    assert_eq!(prep.params.b(), 2, "fixture must use the production binary radix");

    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &proof.pi_dec.children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");
    assert!(builder.is_satisfied(), "honest binary split must satisfy");

    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(wires.parent.m_in);
    let logical_x = wires.parent.x_rows * active_cols;
    let child_count = wires.children.len();
    let family_rows = |name| {
        builder
            .row_family_ranges()
            .iter()
            .filter(|range| range.name == name)
            .map(|range| range.row_end - range.row_start)
            .sum::<usize>()
    };

    let alphabet_rows = logical_x * (child_count + 2);
    assert_eq!(family_rows(pi_dec_stage::ALPHABET), alphabet_rows);
    assert_eq!(child_count, 14, "production Π_DEC arity changed");
    assert_eq!(
        logical_x * child_count * 2 - alphabet_rows,
        logical_x * 12,
        "uniform-sign schedule must save 12 rows per active coordinate"
    );

    let semantic_y_width = D * <K as BasedVectorSpace<F>>::DIMENSION;
    assert_eq!(
        family_rows(pi_dec_stage::RECOMPOSITION_Y_RING),
        wires.parent.y_ring.len() * semantic_y_width,
        "strict y recomposition must stop at the semantic ring width"
    );
    let full_y_width = wires.parent.y_ring.first().expect("fixture y row").len();
    assert!(full_y_width > semantic_y_width, "fixture must expose padded y lanes");
    assert_eq!(
        wires.parent.y_ring.len() * (full_y_width - semantic_y_width),
        wires.parent.y_ring.len() * 20,
        "strict schedule must remove exactly 20 padded recomposition rows per matrix"
    );

    let audit = builder.pi_dec_strict_audits().last().expect("strict audit");
    assert_eq!(audit.x_sign_traces.len(), logical_x);
    assert!(
        audit
            .x_sign_traces
            .iter()
            .all(|[sign, product]| sign != product),
        "each active coordinate must expose distinct sign and centered-product columns"
    );
}

#[test]
fn pi_dec_strict_canonical_x_receipt_matches_every_emitted_row() {
    let (proof, _claims) = drive_nifs(0xCA11_0A1C);
    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    parent.m_in = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
    parent.X = Mat::zero(
        D,
        neo_fold_clean::paper::relations::superneo_public_x_cols(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN),
        F::ZERO,
    );
    for child in &mut children {
        child.m_in = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
        child.X = Mat::zero(
            D,
            neo_fold_clean::paper::relations::superneo_public_x_cols(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN),
            F::ZERO,
        );
    }
    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    let receipt = enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict PiDEC receipt");
    assert!(builder.is_satisfied(), "explicit 270-coordinate strict PiDEC fixture");
    let snapshot = builder.snapshot();
    let program = receipt.program();
    let plan = program.plan();

    assert_eq!((plan.x_rows(), plan.active_columns(), plan.child_count()), (D, 5, 14));
    assert_eq!(plan.logical_coordinates(), 270);
    assert_eq!(plan.recomposition_rows(), 270);
    assert_eq!(plan.canonicality_rows(), 4_320);
    assert_eq!(program.row_count(), 4_590);
    assert_eq!(plan.canonical_column_count(), 4_591);
    assert_eq!(plan.active_index(53, 4), Some(269));
    assert_eq!(plan.public_column(53, 4), Some(269));
    assert_eq!(plan.active_index(1, 0), Some(5));
    assert_eq!(plan.public_column(1, 0), Some(1));

    let mut physical_rows = BTreeSet::new();
    for relative_row in 0..program.row_count() {
        let physical_row = receipt
            .physical_row(relative_row)
            .expect("every indexed row has a physical owner");
        assert!(physical_rows.insert(physical_row), "physical row is multiply owned");
        let actual = CanonicalSparseRow {
            a: snapshot.a_row(physical_row).to_vec(),
            b: snapshot.b_row(physical_row).to_vec(),
            c: snapshot.c_row(physical_row).to_vec(),
        };
        assert_eq!(
            receipt.actual_row_at(relative_row),
            Some(actual),
            "indexed canonical-X row {relative_row} differs from production emission"
        );
    }
    assert_eq!(physical_rows.len(), 4_590);
    assert_eq!(receipt.recomposition_rows().len(), 270);
    assert_eq!(receipt.canonicality_rows().len(), 4_320);
}

#[test]
fn pi_dec_strict_audit_survives_public_column_normalization() {
    let (proof, _claims) = drive_nifs(29);
    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &proof.pi_dec.children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");

    let public_outputs = [
        wires.children[0].x[0],
        wires.parent.fold_digest_fields[2],
        wires.parent.y_ring[0][0],
    ];
    let mut old_to_new = vec![0; builder.witness().len()];
    let mut selected = vec![false; builder.witness().len()];
    selected[0] = true;
    let mut old_columns = vec![0];
    for output in public_outputs {
        assert!(!selected[output.col()], "fixture public columns must be distinct");
        selected[output.col()] = true;
        old_columns.push(output.col());
    }
    old_columns.extend((1..builder.witness().len()).filter(|&col| !selected[col]));
    for (new_col, old_col) in old_columns.into_iter().enumerate() {
        old_to_new[old_col] = new_col;
    }

    let mut expected = builder.pi_dec_strict_audits()[0].clone();
    remap_pi_dec_audit_for_test(&mut expected, &old_to_new);
    let lowered = lower_field_r1cs(builder, &public_outputs).expect("lower strict PiDEC relation");
    assert_eq!(
        lowered.shape().pi_dec_strict_audits(),
        &[expected],
        "lowering must preserve every strict-PiDEC row field and remap every column field"
    );
}

#[test]
fn pi_dec_strict_rejects_a_rewitnessed_binary_sign_tamper() {
    let (proof, _claims) = drive_nifs(26);
    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    parent.X.set(0, 0, F::ONE);
    for child in &mut children {
        child.X.set(0, 0, F::ZERO);
    }
    children[0].X.set(0, 0, F::ONE);

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");
    assert!(builder.is_satisfied(), "baseline");

    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(wires.parent.m_in);
    let coordinate = (0..wires.parent.x_rows * active_cols)
        .find(|coordinate| {
            let row = coordinate / active_cols;
            let col = coordinate % active_cols;
            wires
                .children
                .iter()
                .any(|child| builder.witness()[child.x[row * child.x_cols + col].col()] != F::ZERO)
        })
        .expect("honest fixture must contain a non-zero public split digit");
    let [sign_col, product_col] = builder.pi_dec_strict_audits()[0].x_sign_traces[coordinate];
    let old_sign = builder.witness()[sign_col];
    assert!(old_sign == F::ONE || old_sign == F::ZERO - F::ONE);
    let new_sign = F::ZERO - old_sign;
    let new_product = (new_sign + F::ONE) * new_sign;

    // Re-witness the centered-unit intermediate as well as its input, so the
    // centered-unit rows continue to hold. A child digit/sign row must reject.
    builder.tamper_witness(sign_col, new_sign);
    builder.tamper_witness(product_col, new_product);
    assert!(
        !builder.is_satisfied(),
        "binary digit/sign equations accepted a different shared sign"
    );
}

#[test]
fn pi_dec_strict_rejects_recomposition_preserving_mixed_binary_signs() {
    let (proof, _claims) = drive_nifs(28);
    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    assert!(children.len() >= 2);

    // Both children remain in {-1,0,1}, and -1 + 2*1 = 1, so the old
    // independent alphabet plus recomposition schedule accepted this alias.
    parent.X.set(0, 0, F::ONE);
    for child in &mut children {
        child.X.set(0, 0, F::ZERO);
    }
    children[0].X.set(0, 0, F::ZERO - F::ONE);
    children[1].X.set(0, 0, F::ONE);

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict dec_v emit");
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC accepted a recomposition-valid noncanonical binary split"
    );
}

#[test]
fn pi_dec_strict_radix_four_accepts_the_native_canonical_split() {
    let (prep, proof) = drive_radix_four_nifs(0x4401);
    assert_eq!(
        proof.pi_dec.children.len(),
        7,
        "radix-four profile must emit seven children"
    );

    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &proof.pi_dec.children);
    let receipt = enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict radix-four PiDEC");

    assert_eq!(receipt.program().plan().radix(), 4);
    assert_eq!(receipt.program().plan().child_count(), 7);
    assert!(
        builder.is_satisfied(),
        "radix-four circuit rejected the native split at row {:?}",
        builder.first_unsatisfied_row()
    );
    let snapshot = builder.snapshot();
    for relative_row in 0..receipt.program().row_count() {
        let physical_row = receipt
            .physical_row(relative_row)
            .expect("radix-four physical row");
        let actual = CanonicalSparseRow {
            a: snapshot.a_row(physical_row).to_vec(),
            b: snapshot.b_row(physical_row).to_vec(),
            c: snapshot.c_row(physical_row).to_vec(),
        };
        assert_eq!(receipt.actual_row_at(relative_row), Some(actual));
    }
}

#[test]
fn radix_four_accumulator_handle_selects_the_compact_family_codec() {
    let (_prep, proof) = drive_radix_four_nifs(0x4405);
    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    parent.X.set(0, 0, F::from_u64(2));
    let split =
        neo_reductions::common::split_b_matrix_k(&parent.X, children.len(), 4).expect("radix-four canonical split");
    for (child, expected_x) in children.iter_mut().zip(split) {
        child.X = expected_x;
    }

    let compact = strict_radix_accumulator_family_digest(4, &children, &parent)
        .expect("native radix-four family must pass canonical recomposition");
    let handle = AccumulatorHandle::from_running_parts(4, &children, Some(&parent));

    assert_eq!(handle.digest_fields(), compact);
    assert_ne!(handle, AccumulatorHandle::from_claims(&children));
    assert!(
        strict_radix_accumulator_family_digest(2, &children, &parent).is_none(),
        "the verifier radix must control canonical child recomposition"
    );
}

#[test]
fn pi_dec_strict_radix_four_rejects_a_mixed_sign_recomposition_alias() {
    let (prep, proof) = drive_radix_four_nifs(0x4402);
    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;

    parent.X.set(0, 0, F::ONE);
    for child in &mut children {
        child.X.set(0, 0, F::ZERO);
    }
    children[0].X.set(0, 0, F::ZERO - F::from_u64(3));
    children[1].X.set(0, 0, F::ONE);

    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict radix-four PiDEC");
    assert!(
        !builder.is_satisfied(),
        "radix-four PiDEC accepted -3 + 4·1 as the canonical split of one"
    );
}

#[test]
fn pi_dec_strict_radix_four_rejects_a_rewitnessed_limb() {
    let (prep, proof) = drive_radix_four_nifs(0x4403);
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &proof.pi_dec.children);
    let receipt = enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict radix-four PiDEC");
    assert!(builder.is_satisfied(), "baseline radix-four witness must satisfy");

    let canonical_limb = receipt
        .program()
        .limb_canonical_column(0, 0, 0)
        .expect("first radix-four limb");
    let physical_limb = receipt
        .columns()
        .actual_column(canonical_limb)
        .expect("mapped first radix-four limb");
    builder.tamper_witness(physical_limb, builder.witness()[physical_limb] + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "radix-four PiDEC accepted a rewitnessed signed limb"
    );
}

#[test]
fn pi_dec_radix_four_selective_lowering_uses_two_exact_signed_limbs() {
    let (prep, proof) = drive_radix_four_nifs(0x4404);
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &proof.pi_dec.children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict radix-four PiDEC");
    let (shape, assignment) = lower_field_r1cs(builder, &[])
        .expect("lower radix-four source R1CS")
        .into_parts();
    let trace = shape.pi_dec_strict_audits()[0].x_radix_four_decompositions[0];

    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(
        &[shape.clone(), shape.clone()],
        0,
        D,
        shape.m_in % D,
    )
    .expect("compile radix-four selective relation");
    assert_eq!(relation.field_slot(0, trace.value_col), None);
    assert!(trace.limb_cols.iter().all(|&column| relation
        .field_slot(0, column)
        .is_some_and(|slot| slot.1 == 1)));
    let encoded = relation
        .encode(0, &assignment)
        .expect("encode radix-four assignment");
    assert!(relation.is_satisfied(&encoded), "honest compact radix-four assignment");

    let mut invalid_assignment = assignment.clone();
    invalid_assignment[trace.limb_cols[0]] = F::from_u64(2);
    let invalid = relation
        .encode(0, &invalid_assignment)
        .expect("encode hostile radix-four limb");
    assert!(
        !relation.is_satisfied(&invalid),
        "joint norm must reject a non-unit limb"
    );

    let mut drifted = shape.clone();
    let csc = match &mut drifted.a {
        CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => csc,
        other => panic!("unexpected radix-four A matrix: {other:?}"),
    };
    let entry = csc
        .column_range(trace.value_col)
        .find(|&entry| csc.row_index(entry) == trace.row)
        .expect("radix-four value coefficient");
    csc.vals[entry] += F::ONE;
    assert!(
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&[drifted.clone(), drifted], 0, D, shape.m_in % D,)
            .is_err(),
        "selective lowering accepted a drifted decomposition row"
    );
}

#[test]
fn pi_dec_nonbinary_alphabet_schedule_is_unchanged() {
    let (proof, _claims) = drive_nifs(30);
    let base = neo_params::NeoParams::goldilocks_paper_b2();
    let radix_three = neo_params::NeoParams::new(
        base.q,
        base.eta,
        base.d,
        base.kappa,
        base.m,
        3,
        base.k_rho,
        base.T,
        base.s,
        base.lambda,
    )
    .expect("valid test-only radix-three profile");
    let params = neo_fold_clean::Params::test_only_from_neo_params(radix_three);
    let mut children = proof.pi_dec.children;
    children[0].X.set(0, 0, F::from_u64(2));

    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &proof.pi_rlc.combined, &children);
    let alphabet_start = builder.rows();
    let sign_traces =
        enforce_child_x_canonical_split(&mut builder, &params, &wires).expect("radix-three alphabet emit");
    assert!(sign_traces.is_empty(), "nonbinary schedules must not emit binary signs");
    assert!(
        builder.is_satisfied(),
        "radix-three centered alphabet must retain digit 2"
    );

    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(wires.parent.m_in);
    assert_eq!(
        builder.rows() - alphabet_start,
        wires.children.len() * wires.parent.x_rows * active_cols * 4,
        "five-point centered alphabet must retain its four-row polynomial schedule"
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
        5 * claim_count,
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
#[test]
fn pi_dec_circuit_rejects_noncanonical_x_width() {
    // SuperNeo stores exactly the coefficient embedding. An extra physical X
    // column is a malformed claim shape, not an inactive carrier.
    let (proof, _claims) = drive_nifs(59);
    let mut parent = proof.pi_rlc.combined.clone();
    let mut children = proof.pi_dec.children.clone();
    let widen_x = |claim: &mut neo_fold_clean::CeClaim| {
        let mut widened = Mat::zero(D, 2, F::ZERO);
        for row in 0..D {
            widened[(row, 0)] = claim.X[(row, 0)];
        }
        claim.X = widened;
    };
    widen_x(&mut parent);
    for child in &mut children {
        widen_x(child);
    }

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    assert!(
        enforce_dec_v_strict(&mut builder, &prep.params, &wires).is_err(),
        "strict Π_DEC.V accepted an X matrix wider than the coefficient embedding"
    );
}

#[test]
fn pi_dec_circuit_rejects_partial_ring_public_input() {
    let (proof, _claims) = drive_nifs(61);
    let mut parent = proof.pi_rlc.combined;
    let mut children = proof.pi_dec.children;
    parent.m_in = 1;
    for child in &mut children {
        child.m_in = 1;
    }

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &parent, &children);
    assert!(
        enforce_dec_v_strict(&mut builder, &prep.params, &wires).is_err(),
        "strict PiDEC accepted a partial-ring public input"
    );
}

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

fn drive_radix_four_nifs(seed: u64) -> (Preprocessing, nifs::NifsProof) {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        neo_params::goldilocks_paper_b2::KAPPA,
        neo_params::goldilocks_paper_b2::M,
        4,
        7,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        114,
    )
    .expect("radix-four test profile");
    let params = Params::test_only_from_neo_params(inner);
    let structure =
        CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, vec![])).expect("toy radix-four CCS structure");
    support::install_ajtai_module(&params, &structure);
    let prep = preprocess(params, structure, Some(D)).expect("radix-four preprocessing");
    let fresh = vec![support::toy_instance(&prep, seed)];
    let running = RunningInstance::default();
    let mut transcript = Transcript::session();
    let (_, proof) = nifs::prove(
        &mut transcript,
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
    .expect("radix-four NIFS proof");
    (prep, proof)
}

fn remap_pi_dec_commitment_for_test(audit: &mut PiDecCommitmentAudit, old_to_new: &[usize]) {
    audit.d_col = old_to_new[audit.d_col];
    audit.kappa_col = old_to_new[audit.kappa_col];
    for col in &mut audit.data_cols {
        *col = old_to_new[*col];
    }
}

fn remap_pi_dec_adv_for_test(audit: &mut PiDecAdvAudit, old_to_new: &[usize]) {
    remap_pi_dec_commitment_for_test(&mut audit.ops, old_to_new);
    remap_pi_dec_commitment_for_test(&mut audit.is, old_to_new);
    remap_pi_dec_commitment_for_test(&mut audit.fs, old_to_new);
}

fn remap_pi_dec_claim_for_test(audit: &mut PiDecClaimAudit, old_to_new: &[usize]) {
    remap_pi_dec_commitment_for_test(&mut audit.commitment, old_to_new);
    if let Some(adv) = &mut audit.adv {
        remap_pi_dec_adv_for_test(adv, old_to_new);
    }
    for col in &mut audit.x_cols {
        *col = old_to_new[*col];
    }
    audit.x_rows_col = old_to_new[audit.x_rows_col];
    audit.x_width_col = old_to_new[audit.x_width_col];
    audit.m_in_col = old_to_new[audit.m_in_col];
    for row in &mut audit.y_ring_cols {
        for col in row {
            *col = old_to_new[*col];
        }
    }
    for pair in audit.ct_cols.iter_mut().chain(&mut audit.r_cols) {
        *pair = pair.map(|col| old_to_new[col]);
    }
    audit.fold_digest_cols = audit.fold_digest_cols.map(|col| old_to_new[col]);
}

fn remap_pi_dec_audit_for_test(audit: &mut PiDecStrictAudit, old_to_new: &[usize]) {
    audit.first_allocated_column = old_to_new[audit.first_allocated_column];
    remap_pi_dec_claim_for_test(&mut audit.parent, old_to_new);
    for child in &mut audit.children {
        remap_pi_dec_claim_for_test(child, old_to_new);
    }
    for pair in &mut audit.x_sign_traces {
        *pair = pair.map(|col| old_to_new[col]);
    }
    for trace in &mut audit.x_radix_four_decompositions {
        trace.value_col = old_to_new[trace.value_col];
        trace.limb_cols = trace.limb_cols.map(|col| old_to_new[col]);
    }
}
