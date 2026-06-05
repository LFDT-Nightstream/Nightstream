//! Phase 1.3d — targeted parity against the F' R1CS emitter.
//!
//! Builds a real Fibonacci F' recursive step using the same machinery as
//! `tests/f_prime/r1cs.rs::f_prime_recursive_step_accepts_real_native_nifs_proof`,
//! then validates that the bit-backed `FPrimeImage`'s decoded
//! values match BOTH the in-circuit witness wires the F' R1CS emitter
//! produces AND the production native digest functions in
//! `paper::digest::*`.
//!
//! Mini-1 cross-validates post-step values three ways (native ↔ R1CS wires
//! ↔ image-decoded):
//!
//! - `new_chunk_count`, `new_step_count` (state_out counters).
//! - `new_z_i` digest (state_out z_i lanes mirror the chunk digest).
//! - `new_public_trace` mirrors `new_z_i` in the canonical F' state.
//! - `new_acc_digest` (state_out acc_digest carried from the accumulator handle).
//! - `x_out` digest (poseidon state_x_out trace).
//!
//! Out of scope:
//! - Lifecycle migration (Phase 1.5).
//! - CCS structure (Phase 1.4).
//! - Generic AppStep / Spartan / anything that turns `ivc_invariants` green.
//! - kmul/ring_action cross-validation. Their local round-trip tests exist, but
//!   wire-by-wire parity against the F' emitter needs additional internal
//!   wire exposure.
//!
//! Mini-2 cross-validates the nifs_payloads `running_parent_authority` payload against
//! the actual `SplitNcPiCcsOutputWires` exposed by the embedded NIFS verifier.

#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::engine::r1cs_circuit::builder::RingMulAuditEntry;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, KMulView, StateIn, StateOut,
};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, state_x_out_digest_with_mode, AccumulatorHandle,
    StateXOutDigestMode, F_PRIME_STATE_X_OUT_DOMAIN,
};
use neo_fold_clean::paper::f_prime::poseidon_trace::{
    assert_committed_coords_are_bits, decode_digest_lanes, encode_poseidon_trace,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_public_input, enforce_f_prime_recursive_step_circuit, FPrimeRecursiveInputs, FPrimeStateIn,
    FPrimeStepConfig, F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::ring::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const TRANSCRIPT_LABEL: &[u8] = b"neo.test.f_prime/step/v1";
const EXPECTED_COVERAGE_K_MULS: usize = 7_100;
const EXPECTED_COVERAGE_RING_MULS: usize = 465;

// ── Fixture (mirrors tests/f_prime/r1cs.rs::build_fixture) ──────────────

struct Fixture {
    prep: neo_fold_clean::Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    combined: CeClaim,
    children: Vec<CeClaim>,
    state: FPrimeStateIn,
    chunk_digest: [F; 4],
}

struct RecursiveSourceFixture {
    image: FPrimeSourceImage,
    chunk_count_in_word: Word64Image,
    step_count_in_word: Word64Image,
    pc_word: Word64Image,
    prior_x_out_bits: BitRange,
    public_x_out_bits: BitRange,
}

fn bit_carrier_r1cs() -> R1cs {
    let m = F_PRIME_PUBLIC_INPUT_LEN;
    R1cs {
        a: Mat::zero(1, m, F::ZERO),
        b: Mat::zero(1, m, F::ZERO),
        c: Mat::zero(1, m, F::ZERO),
        m_in: F_PRIME_PUBLIC_INPUT_LEN,
    }
}

fn rand_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|i| F::from_u64(seed.wrapping_mul(31).wrapping_add(i as u64 + 1)))
}

fn native_prior_x_out(state: &FPrimeStateIn) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(state.vk_fs_digest),
        &state.structure_digest,
        state.chunk_count_in,
        state.step_count_in,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(state.z_i_in),
        state.pc,
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.public_trace_in),
    ))
}

fn build_fixture() -> Fixture {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    // First fold: seed the running accumulator.
    let zero_assignment = vec![F::ZERO; prep.structure().m];
    let first = direct_ccs::build_instance(&prep, &r1cs, &zero_assignment).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _) = neo_fold_clean::paper::nifs::prove(
        &mut first_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let acc_digest_in =
        AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest_fields();
    let state = FPrimeStateIn {
        vk_fs_digest: rand_digest(0x10),
        structure_digest: rand_digest(0x20),
        chunk_count_in: 1,
        step_count_in: 1,
        z_0: rand_digest(0x100),
        z_i_in: rand_digest(0x101),
        pc: 1,
        semantic_state_digest_in: acc_digest_in,
        acc_digest_in,
        public_trace_in: rand_digest(0x40),
    };

    let prior_x_out = native_prior_x_out(&state);
    let mut z = encode_f_prime_public_input(prior_x_out);
    assert_eq!(z.len(), F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(prep.structure().m, F_PRIME_PUBLIC_INPUT_LEN);
    z.resize(prep.structure().m, F::ZERO);

    let second = direct_ccs::build_instance(&prep, &r1cs, &z).expect("second instance");
    let fresh_claims = vec![second.claim.clone()];

    let chunk_digest = rand_digest(0x50);
    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/structure", &state.structure_digest);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    tr.append_fields(b"f_prime/z_0", &state.z_0);
    tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    tr.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    tr.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second],
        &running,
    )
    .expect("second NIFS.P");

    let combined = proof.pi_rlc.combined.clone();
    let children: Vec<_> = next_running.claims.clone();

    Fixture {
        prep,
        fresh_claims,
        running,
        proof,
        combined,
        children,
        state,
        chunk_digest,
    }
}

fn split_nc_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'a> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params reconstruction");
    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure()).expect("engine dims");
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &mat_digest,
    )
    .expect("header bundle digest");
    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

fn make_step_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        state_x_out_digest_mode: match prep.semantic_state_mode() {
            neo_fold_clean::paper::construction2::SemanticStateMode::Stateless => {
                neo_fold_clean::paper::digest::StateXOutDigestMode::Stateless
            }
            neo_fold_clean::paper::construction2::SemanticStateMode::Stateful => {
                neo_fold_clean::paper::digest::StateXOutDigestMode::Stateful
            }
        },
    }
}

fn msg_from_fixture<'a>(f: &'a Fixture) -> NifsVCircuitMessages<'a> {
    NifsVCircuitMessages {
        fresh: &f.fresh_claims,
        running: &f.running.claims,
        running_parent_authority: f.running.parent_authority.as_ref(),
        pi_ccs: &f.proof.pi_ccs,
        combined: &f.combined,
        children: &f.children,
    }
}

fn native_x_out(
    state: &FPrimeStateIn,
    chunk_digest: [F; 4],
    new_acc_digest: [F; 4],
    new_semantic_state_digest: [F; 4],
    new_chunk_count: u64,
    new_step_count: u64,
) -> [F; 4] {
    let new_z_i = digest_fields_as_digest32(chunk_digest);
    let new_public_trace = new_z_i;
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(state.vk_fs_digest),
        &state.structure_digest,
        new_chunk_count,
        new_step_count,
        digest_fields_as_digest32(state.z_0),
        new_z_i,
        state.pc,
        digest_fields_as_digest32(new_semantic_state_digest),
        digest_fields_as_digest32(new_acc_digest),
        new_public_trace,
    ))
}

fn recursive_x_out(fixture: &Fixture) -> [F; 4] {
    let new_acc = recursive_acc_digest(fixture);
    native_x_out(
        &fixture.state,
        fixture.chunk_digest,
        new_acc,
        new_acc,
        fixture.state.chunk_count_in + 1,
        fixture.state.step_count_in + fixture.fresh_claims.len() as u64,
    )
}

fn recursive_acc_digest(fixture: &Fixture) -> [F; 4] {
    AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields()
}

fn recursive_source_image(fixture: &Fixture) -> RecursiveSourceFixture {
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(fixture.state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(fixture.state.step_count_in);
    let pc_word = image.push_u64_le(fixture.state.pc);
    let prior_public = image.push_f_prime_public_input(native_prior_x_out(&fixture.state));
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(recursive_x_out(fixture));
    RecursiveSourceFixture {
        image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    }
}

// ── Image config (minimal — only the regions touched by this test) ──────

fn image_config_for_one_step(poseidon_one_shot_preimage_lens: Vec<usize>) -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 704,
        nifs_payload_shapes: vec![], // not exercised here
        kmul_count: 0,               // not exercised here
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::SignedDigit { bits: 5 },
            LowNormEncoding::SignedDigit { bits: 8 },
            LowNormEncoding::SignedDigit { bits: 12 },
            LowNormEncoding::SignedDigit { bits: 20 },
        ),
        // One Poseidon one-shot for state_x_out. The old boundary_update
        // trace is intentionally absent: `new_z_i` mirrors the chunk
        // digest directly, and the accumulator handle is carried in
        // state_out until consumed.
        poseidon_one_shot_preimage_lens,
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    }
}

// ── The parity test ─────────────────────────────────────────────────────

#[test]
fn phase_1_3d_state_out_and_x_out_three_way_parity() {
    let fixture = build_fixture();

    // ── 1. Compute the post-step values natively. ─────────────────────
    let new_chunk_count = fixture.state.chunk_count_in + 1;
    let new_step_count = fixture.state.step_count_in + fixture.fresh_claims.len() as u64;
    let new_z_i = fixture.chunk_digest;
    let new_public_trace = new_z_i;
    let new_acc_digest = recursive_acc_digest(&fixture);
    let native_x_out_value = recursive_x_out(&fixture);

    // ── 2. Run the F' R1CS emitter and read state_out wires. ──────────
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    let out = enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        builder.is_satisfied(),
        "F' R1CS must be satisfied by the honest fixture (first bad row {:?})",
        builder.first_unsatisfied_row()
    );

    let witness = builder.witness();
    let read_lane = |v: neo_fold_clean::engine::r1cs_circuit::Var| witness[v.col()];
    let read_digest =
        |arr: [neo_fold_clean::engine::r1cs_circuit::Var; 4]| -> [F; 4] { std::array::from_fn(|i| read_lane(arr[i])) };

    let wire_new_chunk_count = read_lane(out.state_out.chunk_count).as_canonical_u64();
    let wire_new_step_count = read_lane(out.state_out.step_count).as_canonical_u64();
    let wire_new_z_i = read_digest(out.state_out.z_i);
    let wire_new_public_trace = read_digest(out.state_out.public_trace);
    let wire_new_acc_digest = read_digest(out.state_out.acc_digest);
    let wire_x_out = read_digest(out.x_out);

    // First parity gate: F' R1CS emitter computes the same values the
    // production native digest path does.
    assert_eq!(wire_new_chunk_count, new_chunk_count, "chunk_count: wires ↔ native");
    assert_eq!(wire_new_step_count, new_step_count, "step_count: wires ↔ native");
    assert_eq!(wire_new_z_i, new_z_i, "new_z_i: wires ↔ native");
    assert_eq!(
        wire_new_public_trace, new_public_trace,
        "new_public_trace: wires ↔ native"
    );
    assert_eq!(wire_new_acc_digest, new_acc_digest, "new_acc_digest: wires ↔ native");
    assert_eq!(wire_x_out, native_x_out_value, "x_out: wires ↔ native");

    // ── 3. Construct the producer-side Poseidon preimage. ──────────────
    // Layout config below uses this ordering: index 0 = state_x_out.
    let x_out_preimage =
        build_state_x_out_preimage_from_fixture(&fixture, new_chunk_count, new_step_count, new_acc_digest);

    // ── 4. Build a bit-backed image and fill state_in/state_out/chunk_digest from the same fixture. ─
    let layout = FPrimeImageLayout::new(image_config_for_one_step(vec![x_out_preimage.len()]));
    let mut image = FPrimeImage::new(layout);

    image.fill_state_in(&StateIn {
        vk_fs_digest: fixture.state.vk_fs_digest,
        structure_digest: fixture.state.structure_digest,
        z_0: fixture.state.z_0,
        z_i_in: fixture.state.z_i_in,
        acc_digest_in: fixture.state.acc_digest_in,
        semantic_state_digest_in: fixture.state.acc_digest_in,
        public_trace_in: fixture.state.public_trace_in,
    });
    image.fill_state_out(&StateOut {
        new_chunk_count,
        new_step_count,
        new_z_i,
        new_public_trace,
        new_semantic_state_digest: new_acc_digest,
        new_acc_digest,
    });
    image.fill_chunk_digest(fixture.chunk_digest);

    // ── 5. Splice the producer-side Poseidon trace into poseidon. ──────
    let x_out_trace = encode_poseidon_trace(&x_out_preimage);

    image.splice_one_shot_poseidon(0, &x_out_trace);

    // ── 6. Decode image and three-way assert. ───────────────────────────
    assert_committed_coords_are_bits(&image.values);

    let decoded_state_in = image.decode_state_in();
    assert_eq!(decoded_state_in.z_i_in, fixture.state.z_i_in, "state_in z_i_in decode");
    assert_eq!(
        decoded_state_in.acc_digest_in, fixture.state.acc_digest_in,
        "state_in acc_digest_in decode"
    );

    let decoded_state_out = image.decode_state_out();
    assert_eq!(
        decoded_state_out.new_chunk_count, new_chunk_count,
        "state_out chunk_count decode"
    );
    assert_eq!(
        decoded_state_out.new_step_count, new_step_count,
        "state_out step_count decode"
    );
    assert_eq!(decoded_state_out.new_z_i, new_z_i, "state_out z_i decode");
    assert_eq!(
        decoded_state_out.new_public_trace, new_public_trace,
        "state_out public_trace decode"
    );
    assert_eq!(
        decoded_state_out.new_acc_digest, new_acc_digest,
        "state_out acc_digest decode"
    );

    let decoded_chunk_digest = image.decode_chunk_digest();
    assert_eq!(decoded_chunk_digest, fixture.chunk_digest, "chunk_digest decode");

    let decoded_x_out = image.decode_one_shot_poseidon_digest(0);

    // Three-way parity: image ↔ wires ↔ native.
    assert_eq!(fixture.chunk_digest, wire_new_z_i, "chunk_digest ↔ wire z_i");
    assert_eq!(fixture.chunk_digest, new_z_i, "chunk_digest ↔ native z_i");
    assert_eq!(
        wire_new_public_trace, wire_new_z_i,
        "public_trace wire mirrors z_i wire"
    );
    assert_eq!(new_public_trace, new_z_i, "public_trace native mirrors z_i native");
    assert_eq!(wire_new_acc_digest, new_acc_digest, "acc_digest: wire ↔ native");
    assert_eq!(decoded_x_out, wire_x_out, "poseidon[0] x_out ↔ wire");
    assert_eq!(decoded_x_out, native_x_out_value, "poseidon[0] x_out ↔ native");

    eprintln!(
        "phase_1_3d parity: x_out lanes {:?} match across native/wire/image",
        decoded_x_out.map(|f| f.as_canonical_u64())
    );
}

// ── Preimage builders (mirror `paper::digest::*`) ────────────────────────

fn u64_halves(value: u64) -> [F; 2] {
    [F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]
}

fn build_state_x_out_preimage_from_fixture(
    fixture: &Fixture,
    new_chunk_count: u64,
    new_step_count: u64,
    new_acc_digest: [F; 4],
) -> Vec<F> {
    let new_z_i = digest_fields_as_digest32(fixture.chunk_digest);
    let mut p = vec![F::from_u64(F_PRIME_STATE_X_OUT_DOMAIN)];
    p.extend(digest32_as_fields(digest_fields_as_digest32(
        fixture.state.vk_fs_digest,
    )));
    p.extend(u64_halves(new_chunk_count));
    p.extend(u64_halves(new_step_count));
    p.extend(u64_halves(fixture.state.pc));
    p.extend(digest32_as_fields(new_z_i));
    p.extend(digest32_as_fields(digest_fields_as_digest32(new_acc_digest)));
    p
}

// Mark imports used only inside specific blocks to avoid unused-import warnings.
#[allow(dead_code)]
fn _suppress_unused() {
    let _ = POSEIDON2_GOLDILOCKS_BITS;
    let _ = decode_digest_lanes;
}

// ── Phase 1.3d-mini-2: nifs_payloads parent_authority wire parity ───────────────────

use neo_fold_clean::engine::r1cs_circuit::Var;
use neo_fold_clean::frontends::f_prime::image::{NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires;
use neo_fold_clean::paper::relations::superneo_public_x_cols;
use p3_field::BasedVectorSpace;

/// Convert a production CeClaim into a `NifsCeClaimView`. Mirrors
/// `tests/system/phase_1_3b_nifs_payloads.rs::ce_claim_to_view`.
fn ce_claim_to_nifs_view(claim: &CeClaim) -> NifsCeClaimView {
    let x_rows = claim.X.rows();
    let x_cols = claim.X.cols();
    let x_active = superneo_public_x_cols(claim.m_in);
    let x_active_flat: Vec<F> = (0..x_rows)
        .flat_map(|r| (0..x_active).map(move |c| claim.X[(r, c)]))
        .collect();
    let k_pair = |k: &neo_math::K| -> [F; 2] {
        let limbs = k.as_basis_coefficients_slice();
        [limbs[0], limbs[1]]
    };
    NifsCeClaimView {
        d: claim.c.d as u64,
        kappa: claim.c.kappa as u64,
        c_data: claim.c.data.clone(),
        x_rows: x_rows as u64,
        x_cols: x_cols as u64,
        x_active_cols: x_active as u64,
        x_active_flat,
        r: claim.r.iter().map(k_pair).collect(),
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| row.iter().map(k_pair).collect())
            .collect(),
        y_zcol: claim.y_zcol.iter().map(k_pair).collect(),
        s_col: claim.s_col.iter().map(k_pair).collect(),
        m_in: claim.m_in as u64,
        fold_digest_fields: digest32_as_fields(claim.fold_digest),
    }
}

fn ce_view_shape(view: &NifsCeClaimView) -> NifsCeClaimShape {
    NifsCeClaimShape {
        c_data_entries: view.c_data.len(),
        x_rows: view.x_rows as usize,
        x_active_cols: view.x_active_cols as usize,
        r_len: view.r.len(),
        y_ring_inner_lens: view.y_ring.iter().map(|row| row.len()).collect(),
        y_zcol_len: view.y_zcol.len(),
        s_col_len: view.s_col.len(),
    }
}

/// Read a `SplitNcPiCcsOutputWires` bundle's witness values into a
/// `NifsCeClaimView`. Slices the X row-major wire layout down to the
/// active-cols-only F sequence that `ce_claim_digest` would hash.
fn wires_to_nifs_view(wires: &SplitNcPiCcsOutputWires, witness: &[F]) -> NifsCeClaimView {
    let lane = |v: Var| witness[v.col()];
    let kvar_pair = |k: &neo_fold_clean::engine::r1cs_circuit::field_ext::KVar| [lane(k.c0), lane(k.c1)];

    let x_rows = wires.x_rows;
    let x_cols = wires.x_cols;
    let x_active = superneo_public_x_cols(wires.m_in);
    // Wire allocation is `x[r * x_cols + c] = X[(r, c)]` for FULL x_cols
    // (see `alloc_ce_wires`); slice to active columns to match the
    // production digest's preimage shape.
    let x_active_flat: Vec<F> = (0..x_rows)
        .flat_map(|r| (0..x_active).map(move |c| lane(wires.x[r * x_cols + c])))
        .collect();

    NifsCeClaimView {
        d: wires.c_d as u64,
        kappa: wires.c_kappa as u64,
        c_data: wires.c_data.iter().map(|v| lane(*v)).collect(),
        x_rows: x_rows as u64,
        x_cols: x_cols as u64,
        x_active_cols: x_active as u64,
        x_active_flat,
        r: wires.r.iter().map(kvar_pair).collect(),
        y_ring: wires
            .y_ring
            .iter()
            .map(|row| row.iter().map(kvar_pair).collect())
            .collect(),
        y_zcol: wires.y_zcol.iter().map(kvar_pair).collect(),
        s_col: wires.s_col.iter().map(kvar_pair).collect(),
        m_in: wires.m_in as u64,
        fold_digest_fields: std::array::from_fn(|i| lane(wires.fold_digest_fields[i])),
    }
}

/// Image config sized for one parent_authority nifs_payloads CeClaim — no other
/// regions exercised (boundary/state_out/chunk_digest/app_private left empty, kmul/ring_action/poseidon all zero).
fn nifs_only_image_config(shapes: Vec<NifsPayloadShape>) -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 0,
        nifs_payload_shapes: shapes,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::SignedDigit { bits: 5 },
            LowNormEncoding::SignedDigit { bits: 8 },
            LowNormEncoding::SignedDigit { bits: 12 },
            LowNormEncoding::SignedDigit { bits: 20 },
        ),
        poseidon_one_shot_preimage_lens: vec![],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    }
}

#[test]
fn phase_1_3d_nifs_parent_authority_wire_parity_three_way() {
    let fixture = build_fixture();

    // ── 1. Native view from fixture (the same CeClaim the verifier consumes). ─
    let native_parent = fixture
        .running
        .parent_authority
        .as_ref()
        .expect("parent_authority present");
    let native_view = ce_claim_to_nifs_view(native_parent);

    // ── 2. Run F' R1CS emitter; pull parent_authority wires from FPrimeStepOutput. ─
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    let out = enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        builder.is_satisfied(),
        "F' R1CS must be satisfied (first bad row {:?})",
        builder.first_unsatisfied_row()
    );

    let parent_wires = out
        .nifs_running_parent_authority
        .as_ref()
        .expect("running parent_authority wires exposed");
    let wire_view = wires_to_nifs_view(parent_wires, builder.witness());

    // ── 3. Wire-derived view must match the native CeClaim, field for field. ─
    assert_eq!(wire_view.d, native_view.d, "d");
    assert_eq!(wire_view.kappa, native_view.kappa, "kappa");
    assert_eq!(wire_view.c_data, native_view.c_data, "c_data");
    assert_eq!(wire_view.x_rows, native_view.x_rows, "x_rows");
    assert_eq!(wire_view.x_cols, native_view.x_cols, "x_cols");
    assert_eq!(wire_view.x_active_cols, native_view.x_active_cols, "x_active_cols");
    assert_eq!(wire_view.x_active_flat, native_view.x_active_flat, "x_active_flat");
    assert_eq!(wire_view.r, native_view.r, "r");
    assert_eq!(wire_view.y_ring, native_view.y_ring, "y_ring");
    assert_eq!(wire_view.y_zcol, native_view.y_zcol, "y_zcol");
    assert_eq!(wire_view.s_col, native_view.s_col, "s_col");
    assert_eq!(wire_view.m_in, native_view.m_in, "m_in");
    assert_eq!(
        wire_view.fold_digest_fields, native_view.fold_digest_fields,
        "fold_digest"
    );
    assert_eq!(wire_view, native_view, "wire ↔ native (combined)");

    // ── 4. Build a nifs_payloads-sized image, fill with the native view, decode. ─
    let shape = ce_view_shape(&native_view);
    let layout = FPrimeImageLayout::new(nifs_only_image_config(vec![NifsPayloadShape::CeClaim(shape.clone())]));
    let mut image = FPrimeImage::new(layout);
    let next_offset = image.fill_nifs_ce_claim_at(0, &native_view);
    assert_eq!(next_offset, shape.bits());

    let decoded = image.decode_nifs_ce_claim_at(0, &shape);
    assert_committed_coords_are_bits(&image.values);

    // ── 5. Three-way nifs_payloads parity: native ↔ wire ↔ image-decoded. ──────────
    assert_eq!(decoded, native_view, "image decode ↔ native CeClaim");
    assert_eq!(decoded, wire_view, "image decode ↔ F' emitter wires");

    eprintln!(
        "phase_1_3d-mini-2 nifs_payloads parent_authority parity: c_data {} entries, x active {}×{}, r {}, y_ring {} rows, y_zcol {}, s_col {} — all three sources agree",
        native_view.c_data.len(),
        native_view.x_rows,
        native_view.x_active_cols,
        native_view.r.len(),
        native_view.y_ring.len(),
        native_view.y_zcol.len(),
        native_view.s_col.len(),
    );
}

fn signed_repr(value: F) -> i128 {
    let p: u128 = (1u128 << 64) - (1u128 << 32) + 1;
    let v = value.as_canonical_u64() as u128;
    if v <= p / 2 {
        v as i128
    } else {
        -((p - v) as i128)
    }
}

fn fits_signed_digit(value: F, bits: u8) -> bool {
    let signed = signed_repr(value);
    let half = 1i128 << (bits - 1);
    signed >= -half && signed < half
}

fn first_ring_action_signed_digit_overflow(ring_muls: &[RingMulAuditEntry], witness: &[F]) -> Option<String> {
    for (i, entry) in ring_muls.iter().enumerate() {
        for k in 0..D {
            let value = witness[entry.rho[k].col()];
            if !fits_signed_digit(value, 5) {
                return Some(format!("ring_mul[{i}].rho[{k}] = {}", signed_repr(value)));
            }
        }
        for k in 0..D {
            let value = witness[entry.c[k].col()];
            if !fits_signed_digit(value, 8) {
                return Some(format!("ring_mul[{i}].c[{k}] = {}", signed_repr(value)));
            }
        }
        for r in 0..D {
            for c in 0..D {
                let value = witness[entry.products[r][c].col()];
                if !fits_signed_digit(value, 12) {
                    return Some(format!("ring_mul[{i}].prod[{r}][{c}] = {}", signed_repr(value)));
                }
            }
        }
        for m in 0..D {
            let value = witness[entry.output[m].col()];
            if !fits_signed_digit(value, 20) {
                return Some(format!("ring_mul[{i}].output[{m}] = {}", signed_repr(value)));
            }
        }
    }
    None
}

// ── Phase 1.3d-coverage: full F' step kmul/ring_action accounting + image fill ──────
//
// Runs the F' R1CS emitter once with the K-mul / ring-mul audit trail
// enabled, then asserts that every K-mul and every ring-mul the emitter
// actually invoked round-trips through a `FPrimeImage` sized to
// those observed counts. This is the load-bearing coverage gate: it
// fails if a future emitter change adds a K-mul or ring-mul that the
// bit-backed image config does not account for, OR if the image
// fill/decode loses fidelity for a wire value the emitter committed to.
//
// Three checks per K-mul: native = Karatsuba on a/b inputs, wire =
// witness values at the recorded (p, q, r) intermediates, image =
// decoded bits after fill. We only check wire ↔ image here (the wire
// values ARE the native values by construction — the K-mul gadget
// allocates them as mults). Mini-3 covers the native-↔-wire parity
// surface in isolation.
//
// Three checks per ring-mul: ρ/c inputs read from witness, products
// decoded from image vs witness, output decoded from image vs witness
// vs `Rq::mul` native answer.

#[test]
fn phase_1_3d_kmul_ring_action_coverage_full_step_three_way_parity() {
    let fixture = build_fixture();

    // ── 1. Run F' emitter with audit trail enabled. ──────────────────────
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    builder.enable_audit_trail();
    let _out = enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        builder.is_satisfied(),
        "F' R1CS must be satisfied (first bad row {:?})",
        builder.first_unsatisfied_row()
    );

    let k_muls: Vec<[Var; 3]> = builder.audit_k_muls().to_vec();
    let ring_muls: Vec<RingMulAuditEntry> = builder.audit_ring_muls().to_vec();
    let witness: Vec<F> = builder.witness().to_vec();

    assert!(
        !k_muls.is_empty(),
        "F' emitter must invoke at least one K-mul (audit_k_muls empty)"
    );
    assert!(
        !ring_muls.is_empty(),
        "F' emitter must invoke at least one ring-mul (audit_ring_muls empty)"
    );
    assert_eq!(
        k_muls.len(),
        EXPECTED_COVERAGE_K_MULS,
        "full-step K-mul count changed; update the kmul layout/accounting deliberately"
    );
    assert_eq!(
        ring_muls.len(),
        EXPECTED_COVERAGE_RING_MULS,
        "full-step ring-mul count changed; update the ring_action layout/accounting deliberately"
    );

    // ── 2. Size the image to match observed counts. ──────────────────────
    // The old SignedDigit{5/8/12/20} cost model does not fit the actual
    // full-step F' ring-action witness. Keep this guard so a future protocol
    // change that restores signed bounds forces us to revisit ring_action sizing.
    let overflow = first_ring_action_signed_digit_overflow(&ring_muls, &witness);
    assert!(
        overflow.is_some(),
        "full-step ring_action values now fit SignedDigit{{5/8/12/20}}; revisit U64 layout/accounting"
    );
    eprintln!(
        "phase_1_3d coverage: using U64 ring_action layout because SignedDigit{{5/8/12/20}} overflows at {}",
        overflow.expect("checked above")
    );
    let pair_layout = RingActionTraceLayout::new(
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
    );
    let image_config = FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: k_muls.len(),
        ring_action_pair_count: ring_muls.len(),
        ring_action_pair_layout: pair_layout,
        poseidon_one_shot_preimage_lens: vec![],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    };
    assert_eq!(
        image_config.kmul_count,
        k_muls.len(),
        "image kmul slot count must match observed K-muls"
    );
    assert_eq!(
        image_config.ring_action_pair_count,
        ring_muls.len(),
        "image ring_action slot count must match observed ring-muls"
    );
    let mut image = FPrimeImage::new(FPrimeImageLayout::new(image_config));

    // ── 3. kmul — per K-mul, wire ↔ image parity. ──────────────────────────
    for (i, intermediates) in k_muls.iter().enumerate() {
        let view = KMulView {
            p: [witness[intermediates[0].col()], F::ZERO],
            q: [witness[intermediates[1].col()], F::ZERO],
            r: [witness[intermediates[2].col()], F::ZERO],
        };
        image.fill_kmul_at(i, &view);
        assert_eq!(
            image.decode_kmul_at(i),
            view,
            "kmul K-mul[{i}]: image decode ↔ wire view"
        );
    }

    // ── 4. ring_action — per ring-mul, wire ↔ image parity for ρ, c, every product, and output. ─
    for (i, entry) in ring_muls.iter().enumerate() {
        let rho_vals: [F; D] = std::array::from_fn(|k| witness[entry.rho[k].col()]);
        let c_vals: [F; D] = std::array::from_fn(|k| witness[entry.c[k].col()]);
        let trace = encode_ring_action_trace(&rho_vals, &c_vals, pair_layout);
        image.splice_ring_action_pair(i, &trace);

        // Output: image ↔ wire ↔ `Rq::mul` native (encoded inside trace).
        let wire_out: [F; D] = std::array::from_fn(|m| witness[entry.output[m].col()]);
        let decoded_out = image.decode_ring_action_pair_output(i);
        assert_eq!(decoded_out, wire_out, "ring_action ring-mul[{i}] output: image ↔ wire");
        assert_eq!(
            decoded_out, trace.output_native,
            "ring_action ring-mul[{i}] output: image ↔ Rq::mul native"
        );

        // Products: every `ρ[r]·c[c]` cell, decoded from the image, matches the
        // wire-side `prods[r][c]`. This is the full D² = 2916 coverage check.
        let splice = image.layout.ring_action_pair_splices[i];
        let layout = image.layout.config.ring_action_pair_layout;
        for r in 0..D {
            for c in 0..D {
                let lane_start = splice + layout.prod_limb_start(r, c) - 1;
                let mut acc = F::ZERO;
                for l in 0..layout.prod_enc.limb_count() {
                    let bit = image.values[lane_start + l];
                    assert!(bit == F::ZERO || bit == F::ONE);
                    if bit == F::ONE {
                        acc += layout.prod_enc.limb_coef(l);
                    }
                }
                let wire_prod = witness[entry.products[r][c].col()];
                assert_eq!(acc, wire_prod, "ring_action ring-mul[{i}] prod[{r}][{c}]: image ↔ wire");
            }
        }
    }

    assert_committed_coords_are_bits(&image.values);

    eprintln!(
        "phase_1_3d coverage: full F' step exercises {} K-muls and {} ring-muls; kmul/ring_action image ↔ wire parity holds for every slot",
        k_muls.len(),
        ring_muls.len(),
    );
}
