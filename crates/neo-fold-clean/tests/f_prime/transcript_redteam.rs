//! Red-team tests for F' transcript authority.
//!
//! These tests avoid the public prover entrypoint and hand-roll a NIFS proof
//! under one F' transcript, then try to replay it in the R1CS verifier under a
//! self-consistent but different F' state. They are meant to catch missing
//! pre-NIFS transcript absorbs, not ordinary shape errors.

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_public_input, enforce_f_prime_recursive_step_circuit, FPrimeRecursiveInputs, FPrimeStateIn,
    FPrimeStepConfig, F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const TRANSCRIPT_LABEL: &[u8] = b"neo.test.f_prime/step/v1";

struct RedteamFixture {
    prep: neo_fold_clean::Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    combined: CeClaim,
    children: Vec<CeClaim>,
    state_x_out_digest_mode: StateXOutDigestMode,
    forged_state: FPrimeStateIn,
    semantic_state_digest_out: [F; 4],
    chunk_digest: [F; 4],
}

struct SourceFixture {
    image: FPrimeSourceImage,
    chunk_count_in_word: Word64Image,
    step_count_in_word: Word64Image,
    pc_word: Word64Image,
    prior_x_out_bits: BitRange,
    public_x_out_bits: BitRange,
}

#[derive(Clone, Copy)]
enum ProverTranscriptShape {
    Full,
    OmitVkFs,
    OmitStructure,
    OmitChunkCount,
    OmitStepCount,
    OmitZ0,
    OmitZiIn,
    OmitPc,
    OmitSemanticState,
    OmitAccDigest,
    OmitPublicTrace,
    OmitChunkDigest,
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

fn append_f_prime_step_context(tr: &mut Transcript, state: &FPrimeStateIn, chunk_digest: [F; 4]) {
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
}

fn append_f_prime_step_context_without_semantic_state(
    tr: &mut Transcript,
    state: &FPrimeStateIn,
    chunk_digest: [F; 4],
) {
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/structure", &state.structure_digest);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    tr.append_fields(b"f_prime/z_0", &state.z_0);
    tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    tr.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
}

fn append_f_prime_step_context_without_z_0(tr: &mut Transcript, state: &FPrimeStateIn, chunk_digest: [F; 4]) {
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/structure", &state.structure_digest);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    tr.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    tr.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
}

fn append_f_prime_step_context_without_acc_digest(tr: &mut Transcript, state: &FPrimeStateIn, chunk_digest: [F; 4]) {
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/structure", &state.structure_digest);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    tr.append_fields(b"f_prime/z_0", &state.z_0);
    tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    tr.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
}

fn append_f_prime_step_context_without_public_trace(tr: &mut Transcript, state: &FPrimeStateIn, chunk_digest: [F; 4]) {
    tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    tr.append_fields(b"f_prime/structure", &state.structure_digest);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    tr.append_fields(b"f_prime/z_0", &state.z_0);
    tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    tr.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
}

fn append_f_prime_step_context_without_chunk_digest(tr: &mut Transcript, state: &FPrimeStateIn, _chunk_digest: [F; 4]) {
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
}

fn append_f_prime_step_context_with_omission(
    tr: &mut Transcript,
    state: &FPrimeStateIn,
    chunk_digest: [F; 4],
    omitted: ProverTranscriptShape,
) {
    if !matches!(omitted, ProverTranscriptShape::OmitVkFs) {
        tr.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitStructure) {
        tr.append_fields(b"f_prime/structure", &state.structure_digest);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitChunkCount) {
        tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitStepCount) {
        tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitZ0) {
        tr.append_fields(b"f_prime/z_0", &state.z_0);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitZiIn) {
        tr.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitPc) {
        tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitSemanticState) {
        tr.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitAccDigest) {
        tr.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitPublicTrace) {
        tr.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    }
    if !matches!(omitted, ProverTranscriptShape::OmitChunkDigest) {
        tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
    }
}

fn running_acc_digest(running: &RunningInstance) -> [F; 4] {
    AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest_fields()
}

fn native_prior_x_out(mode: StateXOutDigestMode, state: &FPrimeStateIn) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        mode,
        digest_fields_as_digest32(state.vk_fs_digest),
        &state.structure_digest,
        state.chunk_count_in,
        state.step_count_in,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(state.z_i_in),
        state.pc,
        digest_fields_as_digest32(state.semantic_state_digest_in),
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.public_trace_in),
    ))
}

fn recursive_step_x_out(
    mode: StateXOutDigestMode,
    state: &FPrimeStateIn,
    chunk_digest: [F; 4],
    new_semantic_state_digest: [F; 4],
    new_acc_digest: [F; 4],
    rows_in_chunk: u64,
) -> [F; 4] {
    let new_z_i = digest_fields_as_digest32(chunk_digest);
    digest32_as_fields(state_x_out_digest_with_mode(
        mode,
        digest_fields_as_digest32(state.vk_fs_digest),
        &state.structure_digest,
        state.chunk_count_in + 1,
        state.step_count_in + rows_in_chunk,
        digest_fields_as_digest32(state.z_0),
        new_z_i,
        state.pc,
        digest_fields_as_digest32(new_semantic_state_digest),
        digest_fields_as_digest32(new_acc_digest),
        new_z_i,
    ))
}

fn make_step_config<'a>(prep: &'a neo_fold_clean::Preprocessing, mode: StateXOutDigestMode) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        state_x_out_digest_mode: mode,
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

fn msg_from_fixture<'a>(fixture: &'a RedteamFixture) -> NifsVCircuitMessages<'a> {
    NifsVCircuitMessages {
        fresh: &fixture.fresh_claims,
        running: &fixture.running.claims,
        running_parent_authority: fixture.running.parent_authority.as_ref(),
        pi_ccs: &fixture.proof.pi_ccs,
        combined: &fixture.combined,
        children: &fixture.children,
    }
}

fn semantic_state_digest_out_for(fixture: &RedteamFixture, new_acc_digest: [F; 4]) -> [F; 4] {
    match fixture.state_x_out_digest_mode {
        StateXOutDigestMode::Stateless => new_acc_digest,
        StateXOutDigestMode::Stateful => fixture.semantic_state_digest_out,
    }
}

fn source_image_for(fixture: &RedteamFixture) -> SourceFixture {
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let new_semantic_state_digest = semantic_state_digest_out_for(fixture, new_acc_digest);
    let public_x_out = recursive_step_x_out(
        fixture.state_x_out_digest_mode,
        &fixture.forged_state,
        fixture.chunk_digest,
        new_semantic_state_digest,
        new_acc_digest,
        1,
    );
    source_image_for_public_x_out(fixture, public_x_out)
}

fn source_image_for_public_x_out(fixture: &RedteamFixture, public_x_out: [F; 4]) -> SourceFixture {
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(fixture.forged_state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(fixture.forged_state.step_count_in);
    let pc_word = image.push_u64_le(fixture.forged_state.pc);
    let prior_public = image.push_f_prime_public_input(native_prior_x_out(
        fixture.state_x_out_digest_mode,
        &fixture.forged_state,
    ));
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(public_x_out);
    SourceFixture {
        image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    }
}

fn build_honest_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|_| {})
}

fn build_semantic_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.semantic_state_digest_in[0] += F::ONE;
    })
}

fn build_stateful_semantic_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture_with_mode(
        StateXOutDigestMode::Stateful,
        rand_digest(0x60),
        rand_digest(0x61),
        |state| {
            state.semantic_state_digest_in[0] += F::ONE;
        },
    )
}

fn build_stateful_semantic_omitted_transcript_fixture() -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        StateXOutDigestMode::Stateful,
        rand_digest(0x70),
        rand_digest(0x71),
        |_| {},
        ProverTranscriptShape::OmitSemanticState,
    )
}

fn build_z_0_omitted_transcript_fixture() -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        StateXOutDigestMode::Stateful,
        rand_digest(0x76),
        rand_digest(0x77),
        |_| {},
        ProverTranscriptShape::OmitZ0,
    )
}

fn build_acc_digest_omitted_transcript_fixture() -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        StateXOutDigestMode::Stateful,
        rand_digest(0x78),
        rand_digest(0x79),
        |_| {},
        ProverTranscriptShape::OmitAccDigest,
    )
}

fn build_public_trace_omitted_transcript_fixture() -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        StateXOutDigestMode::Stateful,
        rand_digest(0x72),
        rand_digest(0x73),
        |_| {},
        ProverTranscriptShape::OmitPublicTrace,
    )
}

fn build_chunk_digest_omitted_transcript_fixture() -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        StateXOutDigestMode::Stateful,
        rand_digest(0x74),
        rand_digest(0x75),
        |_| {},
        ProverTranscriptShape::OmitChunkDigest,
    )
}

fn build_omitted_transcript_fixture(shape: ProverTranscriptShape, seed: u64) -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        StateXOutDigestMode::Stateful,
        rand_digest(seed),
        rand_digest(seed + 1),
        |_| {},
        shape,
    )
}

fn build_step_count_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.step_count_in += 1;
    })
}

fn build_chunk_count_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.chunk_count_in += 1;
    })
}

fn build_vk_fs_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.vk_fs_digest[0] += F::ONE;
    })
}

fn build_structure_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.structure_digest[0] += F::ONE;
    })
}

fn build_z_0_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.z_0[0] += F::ONE;
    })
}

fn build_z_i_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.z_i_in[0] += F::ONE;
    })
}

fn build_public_trace_replay_fixture() -> RedteamFixture {
    build_transcript_replay_fixture(|state| {
        state.public_trace_in[0] += F::ONE;
    })
}

fn build_chunk_digest_replay_fixture() -> RedteamFixture {
    let mut fixture = build_honest_fixture();
    fixture.chunk_digest[0] += F::ONE;
    fixture
}

fn build_transcript_replay_fixture(forge_state: impl FnOnce(&mut FPrimeStateIn)) -> RedteamFixture {
    build_transcript_replay_fixture_with_mode(StateXOutDigestMode::Stateless, [F::ZERO; 4], [F::ZERO; 4], forge_state)
}

fn build_transcript_replay_fixture_with_mode(
    mode: StateXOutDigestMode,
    semantic_state_digest_in_override: [F; 4],
    semantic_state_digest_out_override: [F; 4],
    forge_state: impl FnOnce(&mut FPrimeStateIn),
) -> RedteamFixture {
    build_transcript_replay_fixture_with_mode_and_context(
        mode,
        semantic_state_digest_in_override,
        semantic_state_digest_out_override,
        forge_state,
        ProverTranscriptShape::Full,
    )
}

fn build_transcript_replay_fixture_with_mode_and_context(
    mode: StateXOutDigestMode,
    semantic_state_digest_in_override: [F; 4],
    semantic_state_digest_out_override: [F; 4],
    forge_state: impl FnOnce(&mut FPrimeStateIn),
    prover_transcript_shape: ProverTranscriptShape,
) -> RedteamFixture {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    let zero_assignment = vec![F::ZERO; prep.structure().m];
    let first = direct_ccs::build_instance(&prep, &r1cs, &zero_assignment).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _first_proof) = neo_fold_clean::paper::nifs::prove(
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
    .expect("seed running");

    let acc_digest_in = running_acc_digest(&running);
    let semantic_state_digest_in = match mode {
        StateXOutDigestMode::Stateless => acc_digest_in,
        StateXOutDigestMode::Stateful => semantic_state_digest_in_override,
    };
    let semantic_state_digest_out = match mode {
        StateXOutDigestMode::Stateless => acc_digest_in,
        StateXOutDigestMode::Stateful => semantic_state_digest_out_override,
    };
    let honest_state = FPrimeStateIn {
        vk_fs_digest: rand_digest(0x10),
        structure_digest: rand_digest(0x20),
        chunk_count_in: 1,
        step_count_in: 1,
        z_0: rand_digest(0x100),
        z_i_in: rand_digest(0x101),
        pc: 1,
        semantic_state_digest_in,
        acc_digest_in,
        public_trace_in: rand_digest(0x40),
    };
    let mut forged_state = honest_state.clone();
    forge_state(&mut forged_state);

    let mut forged_z = encode_f_prime_public_input(native_prior_x_out(mode, &forged_state));
    forged_z.resize(prep.structure().m, F::ZERO);
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &forged_z).expect("forged fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let chunk_digest = rand_digest(0x50);

    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    match prover_transcript_shape {
        ProverTranscriptShape::Full => append_f_prime_step_context(&mut tr, &honest_state, chunk_digest),
        ProverTranscriptShape::OmitVkFs
        | ProverTranscriptShape::OmitStructure
        | ProverTranscriptShape::OmitChunkCount
        | ProverTranscriptShape::OmitStepCount
        | ProverTranscriptShape::OmitZiIn
        | ProverTranscriptShape::OmitPc => {
            append_f_prime_step_context_with_omission(&mut tr, &honest_state, chunk_digest, prover_transcript_shape)
        }
        ProverTranscriptShape::OmitZ0 => append_f_prime_step_context_without_z_0(&mut tr, &honest_state, chunk_digest),
        ProverTranscriptShape::OmitSemanticState => {
            append_f_prime_step_context_without_semantic_state(&mut tr, &honest_state, chunk_digest)
        }
        ProverTranscriptShape::OmitAccDigest => {
            append_f_prime_step_context_without_acc_digest(&mut tr, &honest_state, chunk_digest)
        }
        ProverTranscriptShape::OmitPublicTrace => {
            append_f_prime_step_context_without_public_trace(&mut tr, &honest_state, chunk_digest)
        }
        ProverTranscriptShape::OmitChunkDigest => {
            append_f_prime_step_context_without_chunk_digest(&mut tr, &honest_state, chunk_digest)
        }
    }
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("NIFS.P under honest state transcript");

    let combined = proof.pi_rlc.combined.clone();
    let children = next_running.claims.clone();
    RedteamFixture {
        prep,
        fresh_claims,
        running,
        proof,
        combined,
        children,
        state_x_out_digest_mode: mode,
        forged_state,
        semantic_state_digest_out,
        chunk_digest,
    }
}

fn assert_replay_fixture_rejected(fixture: RedteamFixture, message: &str) {
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(!builder.is_satisfied(), "{message}");
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_vk_fs() {
    let fixture = build_vk_fs_replay_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F' recursive step accepted an NIFS proof generated under a different vk_fs transcript \
         while fresh.x/source bits were self-consistent with the forged verifier-key digest"
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_structure_digest() {
    let fixture = build_structure_replay_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F' recursive step accepted an NIFS proof generated under a different structure digest \
         transcript while fresh.x/source bits were self-consistent with the forged structure"
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_chunk_count() {
    assert_replay_fixture_rejected(
        build_chunk_count_replay_fixture(),
        "F' recursive step accepted an NIFS proof generated under a different chunk_count_in \
         transcript while fresh.x/source bits were self-consistent with the forged counter",
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_z_0() {
    assert_replay_fixture_rejected(
        build_z_0_replay_fixture(),
        "F' recursive step accepted an NIFS proof generated under a different z_0 transcript \
         while fresh.x/source bits were self-consistent with the forged initial boundary",
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_z_i() {
    assert_replay_fixture_rejected(
        build_z_i_replay_fixture(),
        "F' recursive step accepted an NIFS proof generated under a different z_i transcript \
         while fresh.x/source bits were self-consistent with the forged current boundary",
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_public_trace() {
    assert_replay_fixture_rejected(
        build_public_trace_replay_fixture(),
        "F' recursive step accepted an NIFS proof generated under a different public_trace transcript \
         while all source-image/public-link bits were self-consistent",
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_chunk_digest() {
    assert_replay_fixture_rejected(
        build_chunk_digest_replay_fixture(),
        "F' recursive step accepted an NIFS proof generated under a different chunk_digest transcript \
         while the output public bits were self-consistent with the forged chunk",
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_semantic_state() {
    let fixture = build_semantic_replay_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F' recursive step accepted an NIFS proof generated under a different semantic state-in \
         transcript while fresh.x/source bits were self-consistent with the forged state"
    );
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_bound_to_different_semantic_state() {
    let fixture = build_stateful_semantic_replay_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "stateful F' recursive step accepted an NIFS proof generated under a different semantic_state_in \
         transcript while source bits and state_x_out were self-consistent with the forged semantic state"
    );
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_that_omits_semantic_state_from_transcript() {
    let fixture = build_stateful_semantic_omitted_transcript_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "stateful F' recursive step accepted a NIFS proof whose prover transcript omitted semantic_state_in; \
         this would disconnect HyperNova's state hash from the NIFS Fiat-Shamir challenges"
    );
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_that_omits_z_0_from_transcript() {
    let fixture = build_z_0_omitted_transcript_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "stateful F' recursive step accepted a NIFS proof whose prover transcript omitted z_0; \
         HyperNova's recursive link includes the initial boundary in the verifier state"
    );
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_that_omits_remaining_context_fields() {
    for (shape, name, seed) in [
        (ProverTranscriptShape::OmitVkFs, "vk_fs", 0x80),
        (ProverTranscriptShape::OmitStructure, "structure", 0x82),
        (ProverTranscriptShape::OmitChunkCount, "chunk_count_in", 0x84),
        (ProverTranscriptShape::OmitStepCount, "step_count_in", 0x86),
        (ProverTranscriptShape::OmitZiIn, "z_i_in", 0x88),
        (ProverTranscriptShape::OmitPc, "pc", 0x8A),
    ] {
        let fixture = build_omitted_transcript_fixture(shape, seed);
        let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
        let source = source_image_for(&fixture);
        let new_acc_digest =
            AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
        let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
        let inputs = FPrimeRecursiveInputs {
            state: fixture.forged_state.clone(),
            chunk_digest: fixture.chunk_digest,
            semantic_state_digest_out,
            acc_digest_out: new_acc_digest,
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
        enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs)
            .unwrap_or_else(|err| panic!("emit omitted-{name} F' R1CS: {err}"));
        assert!(
            !builder.is_satisfied(),
            "stateful F' recursive step accepted a NIFS proof whose prover transcript omitted {name}; \
             every HyperNova/F' verifier-context field must drive NIFS.V Fiat-Shamir challenges"
        );
    }
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_that_omits_acc_digest_from_transcript() {
    let fixture = build_acc_digest_omitted_transcript_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "stateful F' recursive step accepted a NIFS proof whose prover transcript omitted acc_digest_in; \
         HyperNova's running accumulator U_i must drive the verifier challenges, not only the x_out hash"
    );
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_that_omits_public_trace_from_transcript() {
    let fixture = build_public_trace_omitted_transcript_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "stateful F' recursive step accepted a NIFS proof whose prover transcript omitted public_trace_in; \
         the public trace is part of the verifier state consumed before NIFS.V challenges"
    );
}

#[test]
fn f_prime_recursive_stateful_rejects_nifs_proof_that_omits_chunk_digest_from_transcript() {
    let fixture = build_chunk_digest_omitted_transcript_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "stateful F' recursive step accepted a NIFS proof whose prover transcript omitted chunk_digest; \
         the deposited chunk must be in the verifier-driven transcript before NIFS.V challenges"
    );
}

#[test]
fn f_prime_recursive_rejects_nifs_proof_bound_to_different_step_count() {
    let fixture = build_step_count_replay_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F' recursive step accepted an NIFS proof generated under a different step_count_in \
         transcript while fresh.x/source bits were self-consistent with the forged counter"
    );
}

#[test]
fn f_prime_recursive_rejects_coherent_wrong_acc_digest_out() {
    let fixture = build_honest_fixture();
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let mut forged_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    forged_acc_digest[0] += F::ONE;
    let forged_x_out = recursive_step_x_out(
        fixture.state_x_out_digest_mode,
        &fixture.forged_state,
        fixture.chunk_digest,
        forged_acc_digest,
        forged_acc_digest,
        1,
    );
    let source = source_image_for_public_x_out(&fixture, forged_x_out);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: forged_acc_digest,
        acc_digest_out: forged_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F' recursive accepted a coherent forged acc_digest_out; the producer step must bind \
         state_out.acc_digest to the NIFS.V output accumulator it just computed"
    );
}

#[test]
fn f_prime_recursive_rejects_pc_not_trivial_even_if_source_word_matches() {
    let mut fixture = build_honest_fixture();
    fixture.forged_state.pc = 2;
    let cfg = make_step_config(&fixture.prep, fixture.state_x_out_digest_mode);
    let source = source_image_for(&fixture);
    let new_acc_digest =
        AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields();
    let semantic_state_digest_out = semantic_state_digest_out_for(&fixture, new_acc_digest);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.forged_state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out,
        acc_digest_out: new_acc_digest,
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
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F' recursive accepted pc != TRIVIAL_PC even though the source-image pc word matched; \
         the single-program build must constrain pc exactly"
    );
}
