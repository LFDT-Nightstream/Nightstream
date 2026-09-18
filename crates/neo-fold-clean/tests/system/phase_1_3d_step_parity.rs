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
use neo_fold_clean::frontends::nebula::layout::encode_delayed_f_prime_suffix;
use neo_fold_clean::frontends::r1cs_f_prime::lower_field_r1cs;
use neo_fold_clean::paper::construction2::{NebulaConfig, NebulaLane, NebulaStepX, RunningInstance, StackShape};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest, nebula_lane_chains,
    state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode, F_PRIME_STATE_X_OUT_DOMAIN,
};
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use neo_fold_clean::paper::f_prime::poseidon_trace::{
    assert_committed_coords_are_bits, decode_digest_lanes, encode_poseidon_trace,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_public_input, enforce_f_prime_recursive_step_circuit, FPrimePublicInputLayout,
    FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig, F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim, LaneRanges, LaneScheme};
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const TRANSCRIPT_LABEL: &[u8] = b"neo.test.f_prime/step/v1";
// Every authoritative Pi_RLC ring-action client is projection-checked:
// one shared D-step beta ladder, then one aggregate identity per output.
const EXPECTED_COVERAGE_K_MULS: usize = 7_650;
const EXPECTED_COVERAGE_RING_MULS: usize = 0;

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

fn bit_carrier_r1cs(public_input_len: usize) -> R1cs {
    let m = public_input_len;
    R1cs {
        a: Mat::zero(1, m, F::ZERO),
        b: Mat::zero(1, m, F::ZERO),
        c: Mat::zero(1, m, F::ZERO),
        m_in: public_input_len,
    }
}

fn rand_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|i| F::from_u64(seed.wrapping_mul(31).wrapping_add(i as u64 + 1)))
}

fn native_prior_x_out(state: &FPrimeStateIn) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(state.vk_fs_digest),
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        state.chunk_count_in,
        state.step_count_in,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(state.z_i_in),
        state.pc,
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.public_trace_in),
        None,
    ))
}

fn build_fixture() -> Fixture {
    build_fixture_with_public_suffix(&[])
}

fn build_fixture_with_public_suffix(public_suffix: &[F]) -> Fixture {
    let public_input_len = F_PRIME_PUBLIC_INPUT_LEN + public_suffix.len();
    let r1cs = bit_carrier_r1cs(public_input_len);
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
        None,
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
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: 1,
        step_count_in: 1,
        z_0: rand_digest(0x100),
        z_i_in: rand_digest(0x101),
        pc: 1,
        semantic_state_digest_in: acc_digest_in,
        acc_digest_in,
        public_trace_in: rand_digest(0x40),
        nebula: None,
    };

    let prior_x_out = native_prior_x_out(&state);
    let mut z = encode_f_prime_public_input(prior_x_out);
    z.extend_from_slice(public_suffix);
    assert_eq!(z.len(), public_input_len);
    assert_eq!(prep.structure().m, public_input_len);
    z.resize(prep.structure().m, F::ZERO);

    let second = direct_ccs::build_instance(&prep, &r1cs, &z).expect("second instance");
    let fresh_claims = vec![second.claim.clone()];

    let chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &fresh_claims);
    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/pi_ccs_header", &state.pi_ccs_header_bundle);
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
        None,
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
        structure: prep.structure().into(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

fn make_step_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> FPrimeStepConfig<'a> {
    make_step_config_with_suffix(prep, 0)
}

fn make_step_config_with_suffix<'a>(
    prep: &'a neo_fold_clean::Preprocessing,
    suffix_len: usize,
) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::with_suffix(suffix_len),
        nebula: None,
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
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        new_chunk_count,
        new_step_count,
        digest_fields_as_digest32(state.z_0),
        new_z_i,
        state.pc,
        digest_fields_as_digest32(new_semantic_state_digest),
        digest_fields_as_digest32(new_acc_digest),
        new_public_trace,
        None,
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
        projection_batches: Vec::new(),
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
        structure_digest: fixture.state.pi_ccs_header_bundle,
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

#[test]
fn authoritative_recursive_f_prime_lowers_with_exact_public_prefix() {
    let fixture = build_fixture();
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
    assert!(builder.is_satisfied(), "authoritative F' fixture must satisfy");

    let rows = builder.rows();
    let cols = builder.cols();
    let expected_public: Vec<F> = out
        .x_out_bits
        .iter()
        .map(|wire| builder.witness()[wire.col()])
        .collect();
    let lowered = lower_field_r1cs(builder, &out.x_out_bits).expect("lower authoritative recursive F'");

    assert_eq!(lowered.shape().n, rows, "lowering must preserve every F' row");
    assert_eq!(lowered.shape().m, cols, "lowering must preserve every F' column");
    assert_eq!(
        lowered.shape().m_in,
        F_PRIME_PUBLIC_INPUT_LEN,
        "authoritative public prefix is [1 || enc_inst(x_out)]"
    );
    assert_eq!(lowered.assignment()[0], F::ONE);
    assert_eq!(&lowered.assignment()[1..F_PRIME_PUBLIC_INPUT_LEN], &expected_public);
    assert!(
        expected_public
            .iter()
            .all(|value| *value == F::ZERO || *value == F::ONE),
        "enc_inst(x_out) must remain bit-valued at the lowering boundary"
    );
    lowered
        .shape()
        .is_satisfied_by(lowered.assignment())
        .expect("lowered authoritative F' relation must satisfy");

    let mut tampered = lowered.assignment().to_vec();
    tampered[1] = F::ONE - tampered[1];
    assert!(
        lowered.shape().is_satisfied_by(&tampered).is_err(),
        "a public x_out bit flip must violate the lowered authoritative relation"
    );
}

#[test]
fn recursive_f_prime_surfaces_transcript_bound_fresh_public_suffix() {
    let suffix = [F::ONE, F::ZERO, F::ONE, F::ONE];
    let fixture = build_fixture_with_public_suffix(&suffix);
    let cfg = make_step_config_with_suffix(&fixture.prep, suffix.len());
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
    assert!(builder.is_satisfied(), "suffix-bearing F' fixture must satisfy");
    assert_eq!(out.fresh_public_suffixes.len(), 1);
    let surfaced: Vec<F> = out.fresh_public_suffixes[0]
        .iter()
        .map(|wire| builder.witness()[wire.col()])
        .collect();
    assert_eq!(surfaced, suffix, "NIFS.V must expose the exact claim-bound suffix");
}

#[test]
fn authoritative_recursive_f_prime_enforces_delayed_nebula_transition() {
    let stacks = StackShape::NONE;
    let suffix_len = delayed_nebula_public_suffix_len(stacks);
    let public_input_len = F_PRIME_PUBLIC_INPUT_LEN + suffix_len;
    let first_lane_col = public_input_len.div_ceil(D);
    let m = (first_lane_col + 3) * D;
    let r1cs = R1cs {
        a: Mat::zero(1, m, F::ZERO),
        b: Mat::zero(1, m, F::ZERO),
        c: Mat::zero(1, m, F::ZERO),
        m_in: public_input_len,
    };
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0xD8).expect("preprocess Nebula carrier");
    let scheme = LaneScheme::from_seeds(
        prep.params.kappa() as usize,
        LaneRanges {
            ops: first_lane_col..first_lane_col + 1,
            is: first_lane_col + 1..first_lane_col + 2,
            fs: first_lane_col + 2..first_lane_col + 3,
        },
        [0xA5; 32],
        [0x5A; 32],
    )
    .expect("lane scheme");

    let mut first_assignment = vec![F::ZERO; m];
    first_assignment[0] = F::ONE;
    let mut first = direct_ccs::build_instance(&prep, &r1cs, &first_assignment).expect("first instance");
    first.claim.adv = Some(scheme.commit(&first.witness.Z).expect("first adv"));
    let mut first_tr = Transcript::session();
    let (running, _) = neo_fold_clean::paper::nifs::prove(
        &mut first_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        Some(&scheme),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("seed running");

    // Lane commitments cover only the three aligned private columns, so
    // their values are independent of the public suffix we derive below.
    let mut provisional = direct_ccs::build_instance(&prep, &r1cs, &first_assignment).expect("provisional fresh");
    provisional.claim.adv = Some(
        scheme
            .commit(&provisional.witness.Z)
            .expect("provisional adv"),
    );
    let fresh_adv = provisional.claim.adv.clone().expect("fresh adv");
    let d_pre = nebula_lane_chains([&fresh_adv]);
    let nebula_cfg = NebulaConfig {
        scheme: scheme.clone(),
        steps_per_segment: 1,
        seg_max: 1,
        stacks,
        plan_digest: rand_digest(0xD800),
        d_init: d_pre[1],
    };

    let acc_digest_in =
        AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest_fields();
    let lane_in = NebulaLane::base(&nebula_cfg);
    let state = FPrimeStateIn {
        vk_fs_digest: rand_digest(0xD810),
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: 1,
        step_count_in: 1,
        z_0: rand_digest(0xD830),
        z_i_in: rand_digest(0xD840),
        pc: 1,
        semantic_state_digest_in: acc_digest_in,
        acc_digest_in,
        public_trace_in: rand_digest(0xD850),
        nebula: Some(lane_in.clone()),
    };
    let vk_bytes = digest_fields_as_digest32(state.vk_fs_digest);
    let z_i_bytes = digest_fields_as_digest32(state.z_i_in);
    let acc_bytes = digest_fields_as_digest32(state.acc_digest_in);
    let mut opened = lane_in.clone();
    opened
        .open_segment(&nebula_cfg, vk_bytes, z_i_bytes, acc_bytes, d_pre)
        .expect("open segment");
    let step = NebulaStepX {
        seg_idx: 0,
        idx: 0,
        ts_in: 0,
        ts_out: 1,
        gamma: opened.gamma.expect("open gamma"),
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let suffix = encode_delayed_f_prime_suffix(&step, stacks, Some(d_pre)).expect("delayed suffix");

    let prior_x_out = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        vk_bytes,
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        state.chunk_count_in,
        state.step_count_in,
        digest_fields_as_digest32(state.z_0),
        z_i_bytes,
        state.pc,
        acc_bytes,
        acc_bytes,
        digest_fields_as_digest32(state.public_trace_in),
        Some(lane_in.digest()),
    ));
    let mut second_assignment = encode_f_prime_public_input(prior_x_out);
    second_assignment.extend_from_slice(&suffix);
    second_assignment.resize(m, F::ZERO);
    let mut second = direct_ccs::build_instance(&prep, &r1cs, &second_assignment).expect("fresh instance");
    second.claim.adv = Some(scheme.commit(&second.witness.Z).expect("fresh adv"));
    assert_eq!(second.claim.adv.as_ref(), Some(&fresh_adv));
    let fresh_claims = vec![second.claim.clone()];

    let chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &fresh_claims);
    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/pi_ccs_header", &state.pi_ccs_header_bundle);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    tr.append_fields(b"f_prime/z_0", &state.z_0);
    tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    tr.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    tr.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    tr.append_fields(b"f_prime/nebula_lane_in", &lane_in.digest());
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        Some(&scheme),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second],
        &running,
    )
    .expect("delayed Nebula fold");
    let combined = proof.pi_rlc.combined.clone();
    let children = next_running.claims.clone();
    let new_acc_digest = AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields();

    let mut lane_out = lane_in.clone();
    lane_out
        .open_segment(&nebula_cfg, vk_bytes, z_i_bytes, acc_bytes, d_pre)
        .expect("native open");
    lane_out
        .advance(&nebula_cfg, &step, Some(&fresh_adv))
        .expect("native delayed transition");
    assert!(lane_out.is_closed(), "single-step segment must close");
    let expected_x_out = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        vk_bytes,
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        state.chunk_count_in + 1,
        state.step_count_in + 1,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(chunk_digest),
        state.pc,
        digest_fields_as_digest32(new_acc_digest),
        digest_fields_as_digest32(new_acc_digest),
        digest_fields_as_digest32(chunk_digest),
        Some(lane_out.digest()),
    ));

    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(state.step_count_in);
    let pc_word = image.push_u64_le(state.pc);
    let prior_public = image.push_f_prime_public_input(prior_x_out);
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(expected_x_out);
    let cfg = FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(&prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::with_suffix(suffix_len),
        nebula: Some(&nebula_cfg),
        state_x_out_digest_mode: StateXOutDigestMode::Stateless,
    };
    let doubled_fresh = vec![fresh_claims[0].clone(), fresh_claims[0].clone()];
    let doubled_messages = NifsVCircuitMessages {
        fresh: &doubled_fresh,
        running: &running.claims,
        running_parent_authority: running.parent_authority.as_ref(),
        pi_ccs: &proof.pi_ccs,
        combined: &combined,
        children: &children,
    };
    let doubled_inputs = FPrimeRecursiveInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: new_acc_digest,
        acc_digest_out: new_acc_digest,
        nifs_msg: doubled_messages,
        rows_in_chunk: 1,
        source_image: &image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    };
    let arity_error =
        match enforce_f_prime_recursive_step_circuit(&mut R1csBuilder::new(), &prep.params, &cfg, &doubled_inputs) {
            Ok(_) => panic!("Nebula F' must reject K != 1"),
            Err(error) => error,
        };
    assert!(arity_error
        .to_string()
        .contains("exactly one delayed fresh claim"));

    let messages = NifsVCircuitMessages {
        fresh: &fresh_claims,
        running: &running.claims,
        running_parent_authority: running.parent_authority.as_ref(),
        pi_ccs: &proof.pi_ccs,
        combined: &combined,
        children: &children,
    };
    let inputs = FPrimeRecursiveInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: new_acc_digest,
        acc_digest_out: new_acc_digest,
        nifs_msg: messages,
        rows_in_chunk: 1,
        source_image: &image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    let out =
        enforce_f_prime_recursive_step_circuit(&mut builder, &prep.params, &cfg, &inputs).expect("emit Nebula F'");
    assert!(
        builder.is_satisfied(),
        "integrated Nebula F' relation must satisfy: {:?}",
        builder.first_unsatisfied_row()
    );
    let witness = builder.witness();
    assert_eq!(out.x_out.map(|wire| witness[wire.col()]), expected_x_out);
    let lane_wires = out.state_out.nebula.expect("Nebula state-out wires");
    assert_eq!(witness[lane_wires.open.col()], F::ZERO);
    assert_eq!(witness[lane_wires.seg_idx.col()], F::ONE);
    assert_eq!(witness[lane_wires.idx.col()], F::ZERO);
    assert_eq!(witness[lane_wires.ts.col()], F::ONE);
    assert_eq!(lane_wires.d_mem.map(|wire| witness[wire.col()]), lane_out.d_mem);

    let tamper_columns = [
        out.fresh_public_suffixes[0][0].col(),
        out.fresh_adv[0].as_ref().expect("adv wires").ops.data[0].col(),
        out.state_in.nebula.expect("Nebula state-in wires").d_mem[0].col(),
        lane_wires.d_mem[0].col(),
    ];
    for column in tamper_columns {
        let original = builder.witness()[column];
        builder.tamper_witness(column, original + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "Nebula relation accepted tampered column {column}"
        );
        builder.tamper_witness(column, original);
        assert!(
            builder.is_satisfied(),
            "restoring column {column} must restore satisfaction"
        );
    }
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
    p.extend(fixture.state.pi_ccs_header_bundle);
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
        projection_batches: Vec::new(),
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
        ring_muls.is_empty(),
        "authoritative F' must not materialize D-squared ring products after full projection adoption"
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
        projection_batches: Vec::new(),
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
