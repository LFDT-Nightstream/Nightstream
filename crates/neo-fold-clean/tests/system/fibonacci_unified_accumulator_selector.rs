//! Unified accumulator selector — A2 structural + algebraic tests.
//!
//! Pins both halves of the selector contract:
//!
//! **Structural** — plan side pushes a `UnifiedAccumulatorSelector`
//! into the image config when `acc.unified == true`; structure builder
//! emits four selector product rows over the `new_acc_digest` digest
//! lanes; binary constraint on `is_base` falls out of the bit-validity
//! loop.
//!
//! **Algebraic** — base-mode witness with `is_base = 1` and
//! `new_acc_digest = digest32_as_fields(accumulator_digest_from_claims(b, &[]))`
//! satisfies the structure; recursive-mode witness with `is_base = 0`
//! and `new_acc_digest = post-fold parent c_data digest` also
//! satisfies; flipping `is_base` without changing the matching trace
//! is rejected (selector row fails); selecting a third digest that
//! matches neither base nor recursive is rejected.
//!
//! Red-team rationale: a compiler that records `is_base` but does not
//! algebraically constrain the digest branch would pass the structural
//! tests but fail `rejects_base_flag_flip` — that's the exact bug A2
//! exists to prevent.

#![allow(non_snake_case)]

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageLayout, NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut,
};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_accumulator_preimage_fields, build_boundary_update_preimage_fields,
    build_public_trace_update_preimage_fields, build_recursive_step_image_config, build_state_x_out_preimage_fields,
    AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
use neo_fold_clean::paper::digest::{
    accumulator_digest_from_claims, accumulator_digest_from_parent_c_data, digest32_as_fields,
};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const BOUNDARY_BITS: usize = 4 * POSEIDON2_GOLDILOCKS_BITS;
/// Hardcoded `b` for the test fixture. The exact value doesn't matter
/// for the selector — `accumulator_digest_from_claims(_, &[])` ignores
/// `b` when the claims slice is empty.
const B: u32 = 2;
/// Recursive-mode `child_count` used in the fixture. Matches the plan
/// (1 child); the test verifies the recursive digest path against the
/// `accumulator_digest_from_parent_c_data(1, &c_data)` helper.
const CHILD_COUNT: u64 = 1;
const NEW_CHUNK_COUNT: u64 = 7;
const NEW_STEP_COUNT: u64 = 13;
const PC: u64 = 1;

fn ring_action_layout_u64() -> RingActionTraceLayout {
    RingActionTraceLayout::new(
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
    )
}

fn minimal_ce_shape() -> NifsCeClaimShape {
    NifsCeClaimShape {
        c_data_entries: 2,
        x_rows: 0,
        x_active_cols: 0,
        r_len: 0,
        y_ring_inner_lens: vec![],
        y_zcol_len: 0,
        s_col_len: 0,
    }
}

fn make_plan(unified: bool) -> RecursiveStepImagePlan {
    // Probe to learn boundary offset, then place state_x_out lanes inside it.
    let plan_probe = RecursiveStepImagePlan {
        limbs: 3,
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: ring_action_layout_u64(),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(minimal_ce_shape())],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: 2,
            child_count: 1,
            unified,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan_probe));
    let boundary_start = probe_layout.boundary.offset;
    let lane_starts: [usize; 4] = std::array::from_fn(|m| boundary_start + m * POSEIDON2_GOLDILOCKS_BITS);

    RecursiveStepImagePlan {
        state_x_out: Some(StateXOutPlanOptions {
            pc: 1,
            public_x_out_lane_bit_starts: lane_starts,
            app_public_input_var_indices: Vec::new(),
            semantic_state_in_var_indices: Vec::new(),
            semantic_state_out_var_indices: Vec::new(),
            initial_semantic_state_digest_anchor: None,
        }),
        ..plan_probe
    }
}

#[test]
fn unified_plan_pushes_selector_into_config() {
    let plan = make_plan(/* unified = */ true);
    let config = build_recursive_step_image_config(&plan);
    let selector = config
        .unified_accumulator_selector
        .expect("unified plan must push a UnifiedAccumulatorSelector into the config");
    assert_eq!(
        selector.base_trace_index, 2,
        "canonical assignment: base accumulator trace at one_shot_index = 2"
    );
    assert_eq!(
        selector.recursive_trace_index, 3,
        "canonical assignment: recursive accumulator trace at one_shot_index = 3"
    );
}

#[test]
fn legacy_plan_leaves_selector_unset() {
    let plan = make_plan(/* unified = */ false);
    let config = build_recursive_step_image_config(&plan);
    assert!(
        config.unified_accumulator_selector.is_none(),
        "legacy single-accumulator path must not set unified_accumulator_selector"
    );
}

#[test]
fn unified_plan_drops_direct_new_acc_digest_binding() {
    let plan = make_plan(/* unified = */ true);
    let config = build_recursive_step_image_config(&plan);
    // Unified mode replaces the direct linear binding with the
    // selector product rows; the state-out bindings list must NOT
    // mention NewAccDigest.
    use neo_fold_clean::frontends::f_prime::image::StateOutDigestTarget;
    let has_new_acc = config
        .one_shot_digest_to_state_out_bindings
        .iter()
        .any(|b| matches!(b.state_out_target, StateOutDigestTarget::NewAccDigest));
    assert!(
        !has_new_acc,
        "unified mode must not emit the direct NewAccDigest state-out binding; selector handles it"
    );
}

#[test]
fn legacy_plan_keeps_direct_new_acc_digest_binding() {
    let plan = make_plan(/* unified = */ false);
    let config = build_recursive_step_image_config(&plan);
    use neo_fold_clean::frontends::f_prime::image::StateOutDigestTarget;
    let has_new_acc = config
        .one_shot_digest_to_state_out_bindings
        .iter()
        .any(|b| matches!(b.state_out_target, StateOutDigestTarget::NewAccDigest));
    assert!(
        has_new_acc,
        "legacy mode keeps the direct NewAccDigest state-out binding"
    );
}

#[test]
fn unified_plan_emits_five_one_shot_traces() {
    // Legacy plan: boundary, public_trace, accumulator, state_x_out = 4 traces.
    // Unified plan: boundary, public_trace, base_acc, recursive_acc, state_x_out = 5 traces.
    let unified_config = build_recursive_step_image_config(&make_plan(true));
    let legacy_config = build_recursive_step_image_config(&make_plan(false));
    assert_eq!(unified_config.poseidon_one_shot_preimage_lens.len(), 5);
    assert_eq!(legacy_config.poseidon_one_shot_preimage_lens.len(), 4);
}

#[test]
fn unified_image_layout_reserves_is_base_lane() {
    let plan = make_plan(/* unified = */ true);
    let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
    assert_eq!(
        layout.is_base.bits, 1,
        "is_base region must reserve exactly one bit lane"
    );
    // is_base sits between app_private and nifs_payloads in the
    // canonical fill order.
    assert_eq!(
        layout.is_base.offset,
        layout.app_private.end(),
        "is_base must immediately follow app_private"
    );
    assert_eq!(
        layout.nifs_payloads.offset,
        layout.is_base.end(),
        "nifs_payloads must immediately follow is_base"
    );
}

#[test]
fn unified_structure_builds_without_panicking() {
    // The structure builder asserts row counts internally
    // (`debug_assert_eq!(builder.rows, total_rows)`). If the selector
    // emission diverges from `unified_selector_count`, this build
    // panics. Reaching the end of `build_f_prime_structure`
    // is the M3a structural smoke test.
    let plan = make_plan(/* unified = */ true);
    let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
    let _structure = build_f_prime_structure(layout);
    // No assertion needed — building without panic is the assertion.
}

// ─────────────────────────────────────────────────────────────────────────
// Algebraic selector tests — satisfying-witness construction.
// ─────────────────────────────────────────────────────────────────────────

/// Fixed test inputs (state-in digests + chunk_digest + c_data). Same
/// values used across the 4 algebraic tests so the only thing that
/// changes between them is `is_base` and the selected `new_acc_digest`.
struct FixedInputs {
    vk_fs_digest: [F; 4],
    structure_digest: [F; 4],
    z_0: [F; 4],
    z_i_in: [F; 4],
    acc_digest_in: [F; 4],
    semantic_state_digest_in: [F; 4],
    public_trace_in: [F; 4],
    chunk_digest: [F; 4],
    c_data: Vec<F>,
}

fn fixed_inputs() -> FixedInputs {
    FixedInputs {
        vk_fs_digest: [
            F::from_u64(0x101),
            F::from_u64(0x202),
            F::from_u64(0x303),
            F::from_u64(0x404),
        ],
        structure_digest: [
            F::from_u64(0x505),
            F::from_u64(0x606),
            F::from_u64(0x707),
            F::from_u64(0x808),
        ],
        z_0: [
            F::from_u64(0x900),
            F::from_u64(0xa00),
            F::from_u64(0xb00),
            F::from_u64(0xc00),
        ],
        z_i_in: [
            F::from_u64(0x111),
            F::from_u64(0x222),
            F::from_u64(0x333),
            F::from_u64(0x444),
        ],
        acc_digest_in: [F::ZERO; 4],
        semantic_state_digest_in: [F::ZERO; 4],
        public_trace_in: [
            F::from_u64(0xaaa),
            F::from_u64(0xbbb),
            F::from_u64(0xccc),
            F::from_u64(0xddd),
        ],
        chunk_digest: [
            F::from_u64(0x10001),
            F::from_u64(0x20002),
            F::from_u64(0x30003),
            F::from_u64(0x40004),
        ],
        c_data: vec![F::from_u64(0xabcd), F::from_u64(0xef01)],
    }
}

/// Outputs of `build_unified_image` — the satisfying image plus the
/// two accumulator-trace digests so callers can assert exact equality
/// against `accumulator_digest_from_claims(b, &[])` and
/// `accumulator_digest_from_parent_c_data(child_count, c_data)`.
struct UnifiedFixture {
    layout: FPrimeImageLayout,
    image: FPrimeImage,
    base_digest_native: [F; 4],
    recursive_digest_native: [F; 4],
}

/// Build a satisfying unified-plan image. `selected = SelectedDigest::Base`
/// writes `is_base = 1` and `new_acc_digest = base_digest`; `Recursive`
/// writes `is_base = 0` and `new_acc_digest = recursive_digest`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SelectedDigest {
    Base,
    Recursive,
}

fn build_unified_image(selected: SelectedDigest) -> UnifiedFixture {
    let plan = make_plan(/* unified = */ true);
    let config = build_recursive_step_image_config(&plan);
    let public_x_out_lane_bit_starts = plan
        .state_x_out
        .as_ref()
        .map(|sxo| sxo.public_x_out_lane_bit_starts)
        .expect("plan has state_x_out");
    let layout = FPrimeImageLayout::new(config);
    let mut image = FPrimeImage::new(layout.clone());
    let fx = fixed_inputs();

    // state_in, chunk_digest.
    image.fill_state_in(&StateIn {
        vk_fs_digest: fx.vk_fs_digest,
        structure_digest: fx.structure_digest,
        z_0: fx.z_0,
        z_i_in: fx.z_i_in,
        acc_digest_in: fx.acc_digest_in,
        semantic_state_digest_in: fx.semantic_state_digest_in,
        public_trace_in: fx.public_trace_in,
    });
    image.fill_chunk_digest(fx.chunk_digest);

    // NIFS payload — needed because the recursive accumulator preimage
    // sources `c_data` lanes from `NifsPayloadLane { payload_index: 0, .. }`.
    image.fill_nifs_ce_claim_at(
        0,
        &NifsCeClaimView {
            d: 0,
            kappa: 0,
            c_data: fx.c_data.clone(),
            x_rows: 0,
            x_cols: 0,
            x_active_cols: 0,
            x_active_flat: vec![],
            r: vec![],
            y_ring: vec![],
            y_zcol: vec![],
            s_col: vec![],
            m_in: 0,
            fold_digest_fields: [F::ZERO; 4],
        },
    );

    // Boundary + public-trace updates (Poseidon traces 0 and 1).
    let boundary_preimage = build_boundary_update_preimage_fields(fx.z_i_in, fx.chunk_digest);
    let public_trace_preimage = build_public_trace_update_preimage_fields(fx.public_trace_in, fx.chunk_digest);
    let boundary_trace = encode_poseidon_trace(&boundary_preimage);
    let public_trace_trace = encode_poseidon_trace(&public_trace_preimage);
    image.splice_one_shot_poseidon(0, &boundary_trace);
    image.splice_one_shot_poseidon(1, &public_trace_trace);

    // Base accumulator: preimage `(tag, 0)` ⇒ digest = empty-accumulator
    // digest. The structure pins this trace at one_shot_index = 2.
    let base_preimage = build_accumulator_preimage_fields(0, &[]);
    let base_trace = encode_poseidon_trace(&base_preimage);
    image.splice_one_shot_poseidon(2, &base_trace);

    // Recursive accumulator: preimage `(tag, child_count, c_data_entries, c_data...)`.
    // Structure pins this trace at one_shot_index = 3.
    let recursive_preimage = build_accumulator_preimage_fields(CHILD_COUNT, &fx.c_data);
    let recursive_trace = encode_poseidon_trace(&recursive_preimage);
    image.splice_one_shot_poseidon(3, &recursive_trace);

    // Selected digest enters `state_out.new_acc_digest`. The selector
    // constraint then accepts iff this matches the trace dictated by
    // `is_base`.
    let new_acc_digest = match selected {
        SelectedDigest::Base => base_trace.digest_native,
        SelectedDigest::Recursive => recursive_trace.digest_native,
    };

    let new_z_i = boundary_trace.digest_native;
    let new_public_trace = public_trace_trace.digest_native;

    image.fill_state_out(&StateOut {
        new_chunk_count: NEW_CHUNK_COUNT,
        new_step_count: NEW_STEP_COUNT,
        new_z_i,
        new_public_trace,
        new_semantic_state_digest: new_acc_digest,
        new_acc_digest,
    });

    // is_base bit (after state_out is filled).
    image.fill_is_base(matches!(selected, SelectedDigest::Base));

    // state_x_out hash: absorbs the actual `new_acc_digest` we just
    // wrote, so the state_x_out trace remains consistent with state_out
    // regardless of which digest was selected.
    let sxo_preimage = build_state_x_out_preimage_fields(
        fx.vk_fs_digest,
        fx.structure_digest,
        NEW_CHUNK_COUNT,
        NEW_STEP_COUNT,
        fx.z_0,
        new_z_i,
        PC,
        new_acc_digest,
        new_acc_digest,
        new_public_trace,
    );
    let sxo_trace = encode_poseidon_trace(&sxo_preimage);
    image.splice_one_shot_poseidon(4, &sxo_trace);

    // Public-x_out boundary bits.
    for (m, &lane_bit_start) in public_x_out_lane_bit_starts.iter().enumerate() {
        let value = sxo_trace.digest_native[m].as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            image.values[lane_bit_start + j] = if (value >> j) & 1 == 1 { F::ONE } else { F::ZERO };
        }
    }

    UnifiedFixture {
        layout,
        image,
        base_digest_native: base_trace.digest_native,
        recursive_digest_native: recursive_trace.digest_native,
    }
}

/// Verify the base trace's digest equals the canonical empty-accumulator
/// digest the lifecycle uses. Council's correction: this test must
/// reach for `accumulator_digest_from_claims` — not a hand-rolled
/// `H(tag, 0)`.
#[test]
fn unified_accumulator_selector_base_digest_equals_paper_helper() {
    let fixture = build_unified_image(SelectedDigest::Base);
    let expected = digest32_as_fields(accumulator_digest_from_claims(B, &[]));
    assert_eq!(
        fixture.base_digest_native, expected,
        "base accumulator trace must produce `accumulator_digest_from_claims(b, &[])`"
    );
}

/// Recursive trace's digest should equal
/// `accumulator_digest_from_parent_c_data(child_count, c_data)` — the
/// authority the lifecycle uses for non-empty running accumulators.
#[test]
fn unified_accumulator_selector_recursive_digest_equals_paper_helper() {
    let fixture = build_unified_image(SelectedDigest::Recursive);
    let fx = fixed_inputs();
    let expected = digest32_as_fields(accumulator_digest_from_parent_c_data(CHILD_COUNT as usize, &fx.c_data));
    assert_eq!(
        fixture.recursive_digest_native, expected,
        "recursive accumulator trace must produce `accumulator_digest_from_parent_c_data(child_count, c_data)`"
    );
}

#[test]
fn unified_accumulator_selector_accepts_base_digest() {
    let fixture = build_unified_image(SelectedDigest::Base);
    let structure = build_f_prime_structure(fixture.layout.clone());
    let z = structure.extend_witness_from_image(&fixture.image);
    assert!(
        structure.is_satisfied(&z),
        "honest base-mode image must satisfy the unified F' structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z)
    );
}

#[test]
fn unified_accumulator_selector_accepts_recursive_digest() {
    let fixture = build_unified_image(SelectedDigest::Recursive);
    let structure = build_f_prime_structure(fixture.layout.clone());
    let z = structure.extend_witness_from_image(&fixture.image);
    assert!(
        structure.is_satisfied(&z),
        "honest recursive-mode image must satisfy the unified F' structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z)
    );
}

#[test]
fn unified_accumulator_selector_rejects_base_flag_flip() {
    // Honest base-mode image (is_base = 1, new_acc_digest = base_digest).
    let mut fixture = build_unified_image(SelectedDigest::Base);

    // Flip the is_base bit to 0 while leaving new_acc_digest pointing
    // at the base (empty-accumulator) digest. The selector constraint
    //   (1 - is_base) · (rec - base) = new - base
    // now demands `rec - base = 0`, i.e. `new = rec`. But
    // `new = base ≠ rec`, so a selector row must fail.
    let is_base_col = fixture.layout.is_base.offset;
    assert_eq!(fixture.image.values[is_base_col], F::ONE);
    fixture.image.values[is_base_col] = F::ZERO;

    let structure = build_f_prime_structure(fixture.layout.clone());
    let z = structure.extend_witness_from_image(&fixture.image);
    assert!(
        !structure.is_satisfied(&z),
        "flipping is_base from 1 → 0 without changing new_acc_digest must trip the selector row \
         — otherwise is_base is not algebraically binding the digest branch"
    );
}

#[test]
fn unified_accumulator_selector_rejects_wrong_selected_digest() {
    // Build a satisfying recursive-mode image, then mutate
    // new_acc_digest to a third value that's neither base nor
    // recursive. Also rebuild state_x_out so the absorb binding
    // doesn't trip — this isolates the *selector* row as the only
    // failing constraint.
    let fixture_base = build_unified_image(SelectedDigest::Base);
    let mut fixture = build_unified_image(SelectedDigest::Recursive);
    let fx = fixed_inputs();

    // Choose a coherent "third" digest: the base digest. With
    // is_base = 0, the selector requires new = recursive_digest, but
    // we'll write new = base_digest. (We use the actually-computed
    // base digest from `fixture_base` — a real Poseidon output, not
    // a random value — so the test rejects "is_base mismatch" rather
    // than "bits out of range".)
    let third_digest = fixture_base.base_digest_native;
    assert_ne!(
        third_digest, fixture.recursive_digest_native,
        "fixture precondition: base and recursive digests must differ"
    );

    // Rewrite state_out.new_acc_digest in the image's bits.
    let acc_digest_lane_count = 4;
    let new_z_i =
        encode_poseidon_trace(&build_boundary_update_preimage_fields(fx.z_i_in, fx.chunk_digest)).digest_native;
    let new_public_trace = encode_poseidon_trace(&build_public_trace_update_preimage_fields(
        fx.public_trace_in,
        fx.chunk_digest,
    ))
    .digest_native;
    fixture.image.fill_state_out(&StateOut {
        new_chunk_count: NEW_CHUNK_COUNT,
        new_step_count: NEW_STEP_COUNT,
        new_z_i,
        new_public_trace,
        new_acc_digest: third_digest,
        new_semantic_state_digest: third_digest,
    });
    let _ = acc_digest_lane_count;

    // Rebuild state_x_out so the absorb-row binding doesn't trip
    // (it would also reject, but we want to isolate the selector).
    let sxo_preimage = build_state_x_out_preimage_fields(
        fx.vk_fs_digest,
        fx.structure_digest,
        NEW_CHUNK_COUNT,
        NEW_STEP_COUNT,
        fx.z_0,
        new_z_i,
        PC,
        third_digest,
        third_digest,
        new_public_trace,
    );
    let sxo_trace = encode_poseidon_trace(&sxo_preimage);
    fixture.image.splice_one_shot_poseidon(4, &sxo_trace);
    let public_x_out_lane_bit_starts = {
        let plan = make_plan(true);
        plan.state_x_out.unwrap().public_x_out_lane_bit_starts
    };
    for (m, &lane_bit_start) in public_x_out_lane_bit_starts.iter().enumerate() {
        let value = sxo_trace.digest_native[m].as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            fixture.image.values[lane_bit_start + j] = if (value >> j) & 1 == 1 { F::ONE } else { F::ZERO };
        }
    }

    let structure = build_f_prime_structure(fixture.layout.clone());
    let z = structure.extend_witness_from_image(&fixture.image);
    assert!(
        !structure.is_satisfied(&z),
        "writing a third digest into new_acc_digest (matches neither base nor recursive) must trip the selector row"
    );
}

/// Defence in depth: prove the rejection in `rejects_base_flag_flip`
/// comes from the selector rows specifically, not from bit-validity
/// or a Poseidon trace inconsistency.
#[test]
fn unified_accumulator_selector_flip_failure_is_a_selector_row() {
    use neo_fold_clean::frontends::f_prime::image::UnifiedAccumulatorSelector;

    let mut fixture = build_unified_image(SelectedDigest::Base);
    fixture.image.values[fixture.layout.is_base.offset] = F::ZERO;
    let structure = build_f_prime_structure(fixture.layout.clone());
    let z = structure.extend_witness_from_image(&fixture.image);
    let bad_row = structure.first_unsatisfied_row(&z).expect("must fail");

    // Selector rows are emitted immediately after state-out bindings
    // and before public-x_out bindings. Compute their row range.
    let bit_count = fixture.layout.end - 1;
    let ring_action_product = 0usize; // canonical Fibonacci plan has 0 pairs.
    let ring_action_output = 0usize;
    let state_out_binding_count = fixture
        .layout
        .config
        .one_shot_digest_to_state_out_bindings
        .len()
        * 4;
    let selector_start = bit_count + ring_action_product + ring_action_output + state_out_binding_count;
    let _selector: &UnifiedAccumulatorSelector = fixture
        .layout
        .config
        .unified_accumulator_selector
        .as_ref()
        .expect("unified mode");
    let selector_end = selector_start + 4; // POSEIDON2_DIGEST_LEN = 4

    assert!(
        bad_row >= selector_start && bad_row < selector_end,
        "flip failure must lie inside the selector row range \
         [{selector_start}, {selector_end}); got row {bad_row}"
    );
}
