//! Π_DEC.V circuit gadget — native vs circuit parity + tamper tests.
//!
//! Drives a real NIFS round trip via `support::toy_preprocessing`, then
//! feeds the resulting `(pi_rlc.combined, pi_dec.children)` pair into
//! the Π_DEC.V circuit and checks constraint satisfaction.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::reductions::pi_dec_circuit::{
    alloc_dec_inputs, enforce_dec_v, enforce_dec_v_strict, enforce_r_consistency, enforce_x_bitness,
};
use p3_field::PrimeCharacteristicRing;

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
    if wires.children[0].y_ring.is_empty() || wires.children[0].y_ring[0].is_empty() {
        eprintln!("test fixture has no y_ring lanes — skipping");
        return;
    }
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
    if wires.children[0].r.is_empty() {
        eprintln!("test fixture has empty r — skipping");
        return;
    }
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

    if wires.children[0].r.is_empty() {
        return;
    }
    let target_col = wires.children[0].r[0].c1.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V must reject a child whose r diverges"
    );
}

#[test]
fn pi_dec_circuit_strict_rejects_out_of_range_x() {
    let (proof, _claims) = drive_nifs(43);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline");

    // Set a child x to `b`. The b-ary recomposition would technically allow
    // larger values (it just sums), but bitness narrows it to {0..b-1}.
    let b = prep.params.b();
    let target_col = wires.children[0].x[0].col();
    builder.tamper_witness(target_col, neo_math::F::from_u64(b as u64));
    assert!(!builder.is_satisfied(), "strict Π_DEC.V must reject child x = b");
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

    if wires.children[0].s_col.is_empty() {
        eprintln!("test fixture has no s_col lanes — skipping");
        return;
    }
    let target_col = wires.children[0].s_col[0].c0.col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(!builder.is_satisfied(), "strict Π_DEC.V accepted tampered child s_col");
}

#[test]
fn pi_dec_circuit_rejects_nonzero_inactive_child_x() {
    // `enforce_dec_v_strict` includes `enforce_inactive_x_zero`, which pins
    // each child's `X[r, c]` to zero for `c >= ceil(m_in / D)`. Tampering
    // an inactive slot must break strict Π_DEC.V on its own — without this
    // guard the next running accumulator could carry non-canonical data
    // that no downstream Π_CCS would re-validate at a terminal state.
    let (proof, _claims) = drive_nifs(59);
    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v_strict(&mut builder, &prep.params, &wires).expect("strict emit");
    assert!(builder.is_satisfied(), "baseline must satisfy");

    let child = &wires.children[0];
    let m_in = child.x_cols;
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(m_in);
    if active_cols >= m_in {
        eprintln!("toy fixture has no inactive cols (active={active_cols}, m_in={m_in}) — skipping");
        return;
    }
    let target_col = child.x[0 * m_in + active_cols].col();
    builder.tamper_witness(target_col, neo_math::F::ONE);
    assert!(
        !builder.is_satisfied(),
        "strict Π_DEC.V must reject non-zero inactive child X"
    );
}

#[test]
fn pi_dec_circuit_does_not_constrain_child_y_zcol() {
    // Native `verify_dec_public` does NOT enforce `parent.y_zcol = Σ b^{i-1}
    // · child_i.y_zcol`. Π_CCS outputs mix MCS digit-decomposed and ME
    // linear y_zcols, and after Π_RLC the parent y_zcol doesn't telescope
    // under b-ary split. Children's y_zcol values are free relative to
    // Π_DEC and only re-bound by the next step's Π_CCS NC terminal identity.
    // This test pins that contract: tampering a child's y_zcol must NOT
    // break Π_DEC.V on its own.
    let (proof, _claims) = drive_nifs(53);

    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;

    let prep = support::toy_preprocessing();
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, parent, children);
    enforce_dec_v(&mut builder, &prep.params, &wires).expect("dec_v emit");
    assert!(
        builder.is_satisfied(),
        "baseline (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    if wires.children[0].y_zcol.is_empty() {
        eprintln!("test fixture has no y_zcol lanes — skipping");
        return;
    }
    let target_col = wires.children[0].y_zcol[0].col();
    let tampered = builder.witness()[target_col] + neo_math::F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        builder.is_satisfied(),
        "Π_DEC.V must not constrain child y_zcol (mirrors native `verify_dec_public`)"
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
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        fresh,
        &running,
    )
    .expect("NIFS.P");
    (proof, claims)
}
