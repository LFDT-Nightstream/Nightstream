//! Static F' layout budget for the canonical Fibonacci recursive step.
//!
//! This test deliberately stops at `FPrimeImageLayout`: no preprocessing,
//! no Ajtai setup, no proof, and no giant `FPrimeStructure` matrix build.
//! It gives us a cheap guardrail for the committed F' image width.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::ccs_native::poseidon2::{
    build_bit_backed_poseidon2_hash, BITS_PER_PERMUTATION, POSEIDON2_GOLDILOCKS_BITS,
};
use neo_fold_clean::engine::r1cs_circuit::builder::{Lc, R1csBuilder};
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_hash;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, PoseidonPreimageLaneSource, StateOutDigestTarget};
use neo_fold_clean::frontends::f_prime::recursive_plan::build_recursive_step_image_config;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use support::fibonacci_f_prime::canonical_threaded_plan;

const LEGACY_ACCUMULATOR_TAG: &[u8] = b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1";

const CURRENT_IMAGE_WIDTH: usize = 134_788;
const REMOVED_RECURSIVE_ACCUMULATOR_TRACE_DIGITS: usize = 5_406_336;
const REMOVED_PUBLIC_TRACE_UPDATE_DIGITS: usize = 109_440;
const REMOVED_BOUNDARY_UPDATE_TRACE_DIGITS: usize = 87_552;
const REMOVED_STATE_X_OUT_STRUCTURE_DIGEST_DIRECT_ABSORB_DIGITS: usize = 22_912;
const REMOVED_STATE_X_OUT_INITIAL_BOUNDARY_DIRECT_ABSORB_DIGITS: usize = 21_888;
const REMOVED_STATE_X_OUT_PUBLIC_TRACE_DUPLICATE_ABSORB_DIGITS: usize = 21_888;
const REMOVED_STATE_X_OUT_CHUNK_COUNT_PC_ABSORB_DIGITS: usize = 21_888;
const REMOVED_HOT_F_PRIME_DOMAIN_TAG_ABSORB_DIGITS: usize = 43_776;
const REMOVED_UNIFIED_SOURCE_NIFS_PAYLOAD_DIGITS: usize = 158_784;
const PREVIOUS_IMAGE_WIDTH: usize = CURRENT_IMAGE_WIDTH
    + REMOVED_RECURSIVE_ACCUMULATOR_TRACE_DIGITS
    + REMOVED_PUBLIC_TRACE_UPDATE_DIGITS
    + REMOVED_BOUNDARY_UPDATE_TRACE_DIGITS
    + REMOVED_STATE_X_OUT_STRUCTURE_DIGEST_DIRECT_ABSORB_DIGITS
    + REMOVED_STATE_X_OUT_INITIAL_BOUNDARY_DIRECT_ABSORB_DIGITS
    + REMOVED_STATE_X_OUT_PUBLIC_TRACE_DUPLICATE_ABSORB_DIGITS
    + REMOVED_STATE_X_OUT_CHUNK_COUNT_PC_ABSORB_DIGITS
    + REMOVED_HOT_F_PRIME_DOMAIN_TAG_ABSORB_DIGITS
    + REMOVED_UNIFIED_SOURCE_NIFS_PAYLOAD_DIGITS;

#[test]
fn fibonacci_f_prime_layout_budget_confirms_recursive_accumulator_trace_removed() {
    let plan = canonical_threaded_plan();
    let config = build_recursive_step_image_config(&plan);
    let layout = FPrimeImageLayout::new(config);

    assert_eq!(layout.end, CURRENT_IMAGE_WIDTH, "canonical F' image width changed");
    assert_eq!(
        layout.one_shot_poseidon_layouts.len(),
        1,
        "expected only the state_x_out trace",
    );

    let state_x_out = layout.one_shot_poseidon_layouts[0].trace_len;

    assert_eq!(state_x_out, 131_328);
    assert_eq!(
        layout.nifs_payloads.bits, 0,
        "unified delayed-handle mode must not reserve dead source-image NIFS payload columns",
    );

    eprintln!(
        "[fib-layout] previous image width              {}",
        PREVIOUS_IMAGE_WIDTH
    );
    eprintln!(
        "[fib-layout] removed accumulator trace         {}",
        REMOVED_RECURSIVE_ACCUMULATOR_TRACE_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed public_trace trace        {}",
        REMOVED_PUBLIC_TRACE_UPDATE_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed boundary_update trace     {}",
        REMOVED_BOUNDARY_UPDATE_TRACE_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed structure direct absorb   {}",
        REMOVED_STATE_X_OUT_STRUCTURE_DIGEST_DIRECT_ABSORB_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed z_0 direct absorb         {}",
        REMOVED_STATE_X_OUT_INITIAL_BOUNDARY_DIRECT_ABSORB_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed public_trace direct absorb {}",
        REMOVED_STATE_X_OUT_PUBLIC_TRACE_DUPLICATE_ABSORB_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed chunk_count/pc absorb    {}",
        REMOVED_STATE_X_OUT_CHUNK_COUNT_PC_ABSORB_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed hot domain tag absorbs   {}",
        REMOVED_HOT_F_PRIME_DOMAIN_TAG_ABSORB_DIGITS,
    );
    eprintln!(
        "[fib-layout] removed unified NIFS payload     {}",
        REMOVED_UNIFIED_SOURCE_NIFS_PAYLOAD_DIGITS,
    );
    eprintln!("[fib-layout] current image width               {}", layout.end);
}

#[test]
fn fibonacci_f_prime_layout_budget_breaks_down_remaining_width() {
    let plan = canonical_threaded_plan();
    let config = build_recursive_step_image_config(&plan);
    let layout = FPrimeImageLayout::new(config.clone());

    let state_bits = layout.state_in.bits + layout.state_out.bits + layout.chunk_digest.bits;
    let control_bits = layout.boundary.bits + layout.app_private.bits + layout.is_base.bits;
    let non_poseidon_bits = layout.end - layout.poseidon.bits;

    eprintln!("[fib-layout] current image width               {}", layout.end);
    eprintln!("[fib-layout]   boundary/app/control bits       {}", control_bits);
    eprintln!("[fib-layout]   state/chunk bits                {}", state_bits);
    eprintln!(
        "[fib-layout]   NIFS payload bits               {}",
        layout.nifs_payloads.bits
    );
    eprintln!("[fib-layout]   kmul trace bits                 {}", layout.kmul.bits);
    eprintln!(
        "[fib-layout]   ring-action trace bits          {}",
        layout.ring_action.bits
    );
    eprintln!(
        "[fib-layout]   Poseidon trace bits             {}",
        layout.poseidon.bits
    );
    eprintln!("[fib-layout]   non-Poseidon subtotal           {}", non_poseidon_bits);
    eprintln!(
        "[fib-layout]   Poseidon bits / total            {:.2}%",
        layout.poseidon.bits as f64 * 100.0 / layout.end as f64
    );

    let mut non_bit_poseidon_rows = 0usize;
    for (idx, &preimage_len) in config.poseidon_one_shot_preimage_lens.iter().enumerate() {
        let trace_layout = layout.one_shot_poseidon_layouts[idx];
        let dummy_preimage = vec![F::ZERO; preimage_len];
        let native = build_bit_backed_poseidon2_hash(&dummy_preimage);
        let non_bit_rows = native.structure.n - trace_layout.trace_len;
        non_bit_poseidon_rows += non_bit_rows;
        eprintln!(
            "[fib-layout]   one-shot[{idx}] preimage={preimage_len} absorbs={} bits={} non-bit rows={}",
            trace_layout.absorbs, trace_layout.trace_len, non_bit_rows,
        );
    }

    let sponge_permutations = layout.sponge_transcript_bits / BITS_PER_PERMUTATION;
    let sponge_non_bit_rows = sponge_permutations * (86 + 248 + 8);
    non_bit_poseidon_rows += sponge_non_bit_rows;
    eprintln!(
        "[fib-layout]   sponge permutations={} bits={} estimated non-bit rows={}",
        sponge_permutations, layout.sponge_transcript_bits, sponge_non_bit_rows,
    );
    eprintln!(
        "[fib-layout]   lifted Poseidon non-bit rows    {}",
        non_bit_poseidon_rows
    );

    assert_eq!(layout.end, CURRENT_IMAGE_WIDTH);
    assert_eq!(layout.poseidon.bits, 131_328);
    assert_eq!(layout.ring_action.bits % POSEIDON2_GOLDILOCKS_BITS, 0);
    assert_eq!(
        layout.poseidon.bits,
        layout
            .one_shot_poseidon_layouts
            .iter()
            .map(|trace| trace.trace_len)
            .sum::<usize>()
            + layout.sponge_transcript_bits,
    );
}

#[test]
fn fibonacci_f_prime_layout_has_no_producer_side_accumulator_hash_trace() {
    let plan = canonical_threaded_plan();
    let config = build_recursive_step_image_config(&plan);
    let layout = FPrimeImageLayout::new(config.clone());

    assert_eq!(
        layout.one_shot_poseidon_layouts.len(),
        1,
        "canonical unified mode should only emit the state_x_out hash",
    );
    assert!(
        config.unified_accumulator_selector.is_none(),
        "canonical unified mode uses delayed accumulator-handle binding, not a producer-side selector",
    );

    assert!(
        !config
            .one_shot_digest_to_state_out_bindings
            .iter()
            .any(|binding| binding.state_out_target == StateOutDigestTarget::NewAccDigest),
        "producer side must not bind new_acc_digest to a local accumulator hash trace",
    );

    assert_eq!(
        config.one_shot_digest_to_public_x_out_bindings[0].one_shot_index, 0,
        "state_x_out is one-shot index 0 after removing accumulator, public_trace, and boundary hash traces",
    );

    let variable_parent_lanes = config
        .poseidon_transition_enforcements
        .iter()
        .flat_map(|enforcement| enforcement.preimage_lanes.iter())
        .filter(|lane| matches!(lane, PoseidonPreimageLaneSource::NifsPayloadLane { .. }))
        .count();
    assert_eq!(
        variable_parent_lanes, 0,
        "no producer-side Poseidon hash may absorb parent.c_data lanes from the NIFS payload",
    );
}

#[test]
fn fibonacci_f_prime_layout_budget_rejects_naive_field_poseidon_replacement() {
    let plan = canonical_threaded_plan();
    let acc = plan
        .accumulator
        .as_ref()
        .expect("canonical plan has accumulator");

    let mut builder = R1csBuilder::new();
    let parent_c_data: Vec<_> = (0..acc.c_data_entries)
        .map(|_| builder.alloc(F::ZERO))
        .collect();
    let input_cols = builder.cols();
    enforce_legacy_parent_accumulator_digest(&mut builder, acc.child_count as usize, &parent_c_data);

    let extra_poseidon_vars = builder.cols() - input_cols;
    let encoded_extra_width = extra_poseidon_vars * 64;

    eprintln!("[fib-layout] naive field Poseidon rows       {}", builder.rows(),);
    eprintln!(
        "[fib-layout] naive field Poseidon extra vars {} ({} encoded bits)",
        extra_poseidon_vars, encoded_extra_width,
    );

    assert!(
        encoded_extra_width > REMOVED_RECURSIVE_ACCUMULATOR_TRACE_DIGITS,
        "under today's R1CS-F' 64-bit variable encoding, the field-var Poseidon gadget is not a width fix",
    );
}

fn enforce_legacy_parent_accumulator_digest(
    builder: &mut R1csBuilder,
    child_count: usize,
    parent_c_data: &[neo_fold_clean::engine::r1cs_circuit::Var],
) {
    let mut preimage = alloc_const_tag(builder, LEGACY_ACCUMULATOR_TAG);
    preimage.push(alloc_constant(builder, F::from_u64(child_count as u64)));
    if child_count > 0 {
        preimage.push(alloc_constant(builder, F::from_u64(parent_c_data.len() as u64)));
        preimage.extend_from_slice(parent_c_data);
    }
    let _ = enforce_poseidon2_hash(builder, &preimage);
}

fn alloc_const_tag(builder: &mut R1csBuilder, tag: &'static [u8]) -> Vec<neo_fold_clean::engine::r1cs_circuit::Var> {
    const BYTES_PER_LIMB: usize = 7;
    let mut out = Vec::with_capacity(1 + tag.len().div_ceil(BYTES_PER_LIMB));
    out.push(alloc_constant(builder, F::from_u64(tag.len() as u64)));
    for chunk in tag.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(alloc_constant(builder, F::from_u64(u64::from_le_bytes(limb))));
    }
    out
}

fn alloc_constant(builder: &mut R1csBuilder, c: F) -> neo_fold_clean::engine::r1cs_circuit::Var {
    let v = builder.alloc(c);
    builder.enforce_eq(&Lc::from_var(v), &Lc::from_const(c));
    v
}

#[test]
fn fibonacci_f_prime_layout_budget_rejects_field_var_state_x_out_replacement() {
    let plan = canonical_threaded_plan();
    let config = build_recursive_step_image_config(&plan);
    let layout = FPrimeImageLayout::new(config.clone());
    let state_x_out_preimage_len = config.poseidon_one_shot_preimage_lens[0];
    let state_x_out_bits = layout.one_shot_poseidon_layouts[0].trace_len;

    let mut builder = R1csBuilder::new();
    let input_vars: Vec<_> = (0..state_x_out_preimage_len)
        .map(|i| builder.alloc(F::from_u64(i as u64 + 1)))
        .collect();
    let input_cols = builder.cols();
    let _digest = enforce_poseidon2_hash(&mut builder, &input_vars);
    let extra_poseidon_vars = builder.cols() - input_cols;
    let optimistic_encoded_width = extra_poseidon_vars * POSEIDON2_GOLDILOCKS_BITS;

    eprintln!(
        "[fib-layout] state_x_out field-var Poseidon rows {} extra vars {} (optimistic {} encoded bits)",
        builder.rows(),
        extra_poseidon_vars,
        optimistic_encoded_width,
    );

    assert!(
        optimistic_encoded_width > state_x_out_bits,
        "under today's R1CS-F' 64-bit variable encoding, field-var Poseidon is not a width fix for state_x_out",
    );
}
