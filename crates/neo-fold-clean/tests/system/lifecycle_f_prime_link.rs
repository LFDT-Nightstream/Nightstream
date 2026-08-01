#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{self, FoldProof, ProofState, State, StepProof};
use neo_fold_clean::paper::digest::{
    ccs_claim_digest, chunk_public_digest, digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest,
    initial_boundary_digest, pi_ccs_instance_digest, public_trace_seed_digest, state_x_out_digest_with_mode,
    structure_digest, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_superneo_public_input, encode_x_out_public_bits, enforce_f_prime_base_step_circuit,
    enforce_f_prime_recursive_step_circuit, FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs,
    FPrimeStateIn, FPrimeStepConfig, F_PRIME_ENC_INST_BITS, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance, CeClaim};
use neo_fold_clean::{Uncompressed, UncompressedAudit};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;
fn bit_carrier_r1cs() -> R1cs {
    R1cs {
        a: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        b: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        c: Mat::zero(1, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    }
}

fn k_c1_one() -> K {
    K::from_coeffs([F::ZERO, F::ONE])
}
fn compute_x_out_native(prep: &neo_fold_clean::Preprocessing, state: &State) -> [F; 4] {
    let mode = match prep.semantic_state_mode() {
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    digest32_as_fields(state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        &structure_digest(prep.structure()),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    ))
}

fn split_nc_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'a> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params");
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
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::plain(),
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

fn f_prime_state_in(state: &State, prep: &neo_fold_clean::Preprocessing) -> FPrimeStateIn {
    FPrimeStateIn {
        vk_fs_digest: digest32_as_fields(prep.vk.digest()),
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: state.chunk_count,
        step_count_in: state.step_count,
        z_0: digest32_as_fields(state.z_0),
        z_i_in: digest32_as_fields(state.z_i),
        pc: state.pc,
        acc_digest_in: digest32_as_fields(state.acc_digest),
        semantic_state_digest_in: digest32_as_fields(state.semantic_state_digest),
        public_trace_in: digest32_as_fields(state.public_trace),
        nebula: None,
    }
}

fn base_state(prep: &neo_fold_clean::Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = AccumulatorHandle::empty().digest();
    State::base(z_0, public_trace, acc_digest, acc_digest)
}
fn build_link_instance(prep: &neo_fold_clean::Preprocessing, r1cs: &R1cs, x_out_target: [F; 4]) -> CcsInstance {
    let z = encode_f_prime_superneo_public_input(x_out_target);
    direct_ccs::build_instance(prep, r1cs, &z).expect("recursive-link instance")
}
fn peek_next_state(prep: &neo_fold_clean::Preprocessing, state: &State, batch: &[CcsInstance]) -> State {
    let (next, _) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        state.clone(),
        batch.to_vec(),
    )
    .expect("peek step");
    next
}
struct StepSnapshot {
    state_in: State,
    state_out: State,
    public_batch: Vec<CcsClaim>,
    step_proof: StepProof,
}

struct ChainFixture {
    prep: neo_fold_clean::Preprocessing,
    snapshots: Vec<StepSnapshot>,
}
fn build_f_prime_honest_chain(len: usize) -> ChainFixture {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy_inst = || direct_ccs::build_instance(&prep, &r1cs, &placeholder_z).expect("dummy");

    let mut state = base_state(&prep);
    let mut snapshots = Vec::with_capacity(len);

    for _ in 0..len {
        let predicted = peek_next_state(&prep, &state, &[dummy_inst()]);
        let target_x_out = compute_x_out_native(&prep, &predicted);
        let batch = build_link_instance(&prep, &r1cs, target_x_out);

        let state_in = state.clone();
        let public_batch = vec![batch.claim.clone()];

        let (next_state, step_proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &prep.vk,
            state,
            vec![batch],
        )
        .expect("step");
        debug_assert_eq!(next_state.z_i, predicted.z_i);
        debug_assert_eq!(next_state.public_trace, predicted.public_trace);
        debug_assert_eq!(next_state.semantic_state_digest, predicted.semantic_state_digest);
        debug_assert_eq!(next_state.acc_digest, predicted.acc_digest);

        snapshots.push(StepSnapshot {
            state_in,
            state_out: next_state.clone(),
            public_batch,
            step_proof,
        });

        state = next_state;
    }

    ChainFixture { prep, snapshots }
}
struct RecursiveStepView<'a> {
    prep: &'a neo_fold_clean::Preprocessing,
    state_in: &'a State,
    state_out: &'a State,
    fresh: Vec<CcsClaim>,
    running_claims: Vec<CeClaim>,
    running_parent_authority: Option<CeClaim>,
    running_pending_projection: Option<neo_fold_clean::paper::construction2::PendingProjectionState>,
    nifs: NifsProof,
    chunk_digest: [F; 4],
    prior_x_out: [F; 4],
    post_step_x_out: [F; 4],
}

impl ChainFixture {
    fn recursive_step(&self, idx: usize) -> RecursiveStepView<'_> {
        let snapshot = &self.snapshots[idx];
        let FoldProof::Recursive(nifs) = &snapshot.step_proof.fold else {
            panic!("step {idx} is not FoldProof::Recursive");
        };
        let nifs = nifs
            .materialize()
            .expect("recursive NIFS proof materialization");
        let (running_claims, running_parent_authority, running_pending_projection, fresh) =
            match &snapshot.state_in.proof {
                ProofState::Active { running, latest } => {
                    let running = running
                        .materialize()
                        .expect("recursive step running materialization");
                    let pending_projection = running.pending_projection().cloned();
                    (
                        running.claims,
                        running.parent_authority,
                        pending_projection,
                        latest.claims(),
                    )
                }
                ProofState::Initial => panic!("step {idx} state-in is Initial; can't be recursive"),
            };
        let chunk_digest = f_prime_chunk_public_digest(snapshot.state_in.step_count, &snapshot.public_batch);
        let prior_x_out = compute_x_out_native(&self.prep, &snapshot.state_in);
        let post_step_x_out = compute_x_out_native(&self.prep, &snapshot.state_out);
        RecursiveStepView {
            prep: &self.prep,
            state_in: &snapshot.state_in,
            state_out: &snapshot.state_out,
            fresh,
            running_claims,
            running_parent_authority,
            running_pending_projection,
            nifs,
            chunk_digest,
            prior_x_out,
            post_step_x_out,
        }
    }

    fn is_recursive(&self, idx: usize) -> bool {
        matches!(self.snapshots[idx].step_proof.fold, FoldProof::Recursive(_))
    }
}

struct BaseSourceImage {
    image: FPrimeSourceImage,
    chunk_count_in_word: Word64Image,
    step_count_in_word: Word64Image,
    pc_word: Word64Image,
    public_x_out_bits: BitRange,
}

fn build_base_source_image(post_step_x_out: [F; 4], f_state: &FPrimeStateIn) -> BaseSourceImage {
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(f_state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(f_state.step_count_in);
    let pc_word = image.push_u64_le(f_state.pc);
    let public_x_out_bits = image.push_enc_inst(post_step_x_out);
    BaseSourceImage {
        image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    }
}

fn run_base_check_with_semantic(
    prep: &neo_fold_clean::Preprocessing,
    snapshot: &StepSnapshot,
    tweak: impl FnOnce(&mut FPrimeStateIn, &mut [F; 4], &mut [F; 4], &mut BaseSourceImage),
) -> R1csBuilder {
    let mut f_state = f_prime_state_in(&snapshot.state_in, prep);
    let mut chunk_digest = f_prime_chunk_public_digest(snapshot.state_in.step_count, &snapshot.public_batch);
    let mut semantic_state_digest_out = digest32_as_fields(snapshot.state_out.semantic_state_digest);
    let post_step_x_out = compute_x_out_native(prep, &snapshot.state_out);
    let mut source = build_base_source_image(post_step_x_out, &f_state);

    tweak(
        &mut f_state,
        &mut chunk_digest,
        &mut semantic_state_digest_out,
        &mut source,
    );

    let cfg = make_step_config(prep);
    let inputs = FPrimeBaseInputs {
        state: f_state,
        chunk_digest,
        semantic_state_digest_out,
        rows_in_chunk: snapshot.public_batch.len() as u64,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit base F' R1CS");
    b
}

struct RecursiveSourceImage {
    image: FPrimeSourceImage,
    chunk_count_in_word: Word64Image,
    step_count_in_word: Word64Image,
    pc_word: Word64Image,
    prior_x_out_bits: BitRange,
    public_x_out_bits: BitRange,
}

fn build_source_image(view: &RecursiveStepView<'_>, f_state: &FPrimeStateIn) -> RecursiveSourceImage {
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(f_state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(f_state.step_count_in);
    let pc_word = image.push_u64_le(f_state.pc);
    let prior_public = image.push_f_prime_public_input(view.prior_x_out);
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(view.post_step_x_out);
    RecursiveSourceImage {
        image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    }
}

fn run_recursive_check(
    view: &RecursiveStepView<'_>,
    tweak: impl FnOnce(&mut FPrimeStateIn, &mut [F; 4], &mut [F; 4], &mut RecursiveSourceImage, &mut Vec<CcsClaim>),
) -> R1csBuilder {
    run_recursive_check_with_semantic(
        view,
        |f_state, chunk_digest, acc_digest_out, _semantic_state_digest_out, source, fresh| {
            tweak(f_state, chunk_digest, acc_digest_out, source, fresh);
        },
    )
}

fn run_recursive_check_with_semantic(
    view: &RecursiveStepView<'_>,
    tweak: impl FnOnce(
        &mut FPrimeStateIn,
        &mut [F; 4],
        &mut [F; 4],
        &mut [F; 4],
        &mut RecursiveSourceImage,
        &mut Vec<CcsClaim>,
    ),
) -> R1csBuilder {
    let mut f_state = f_prime_state_in(view.state_in, view.prep);
    let mut chunk_digest = view.chunk_digest;
    let mut acc_digest_out = digest32_as_fields(view.state_out.acc_digest);
    let mut semantic_state_digest_out = digest32_as_fields(view.state_out.semantic_state_digest);
    let mut source = build_source_image(view, &f_state);
    let mut fresh = view.fresh.clone();

    tweak(
        &mut f_state,
        &mut chunk_digest,
        &mut acc_digest_out,
        &mut semantic_state_digest_out,
        &mut source,
        &mut fresh,
    );

    let cfg = make_step_config(view.prep);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out,
        acc_digest_out,
        state: f_state,
        chunk_digest,
        nifs_msg: NifsVCircuitMessages {
            fresh: &fresh,
            running: view.running_claims.as_slice(),
            running_parent_authority: view.running_parent_authority.as_ref(),
            running_pending_projection: view.running_pending_projection.as_ref(),
            pi_ccs: &view.nifs.pi_ccs,
            combined: &view.nifs.pi_rlc.combined,
            children: &view.nifs.pi_dec.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &view.prep.params, &cfg, &inputs).expect("emit F' R1CS");
    b
}

fn run_recursive_check_with_output_authority(
    view: &RecursiveStepView<'_>,
    combined: &CeClaim,
    children: &[CeClaim],
    acc_digest_out: [F; 4],
    semantic_state_digest_out: [F; 4],
    source: RecursiveSourceImage,
) -> R1csBuilder {
    let cfg = make_step_config(view.prep);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out,
        acc_digest_out,
        state: f_prime_state_in(view.state_in, view.prep),
        chunk_digest: view.chunk_digest,
        nifs_msg: NifsVCircuitMessages {
            fresh: &view.fresh,
            running: view.running_claims.as_slice(),
            running_parent_authority: view.running_parent_authority.as_ref(),
            running_pending_projection: view.running_pending_projection.as_ref(),
            pi_ccs: &view.nifs.pi_ccs,
            combined,
            children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &view.prep.params, &cfg, &inputs).expect("emit F' R1CS");
    b
}

fn overwrite_enc_inst_bits(source: &mut FPrimeSourceImage, range: BitRange, digest: [F; 4]) {
    for (offset, bit) in encode_x_out_public_bits(digest).into_iter().enumerate() {
        source.set_bit(range.start() + offset, bit == F::ONE);
    }
}

#[test]
fn lifecycle_base_step_rejects_semantic_digest_out_not_equal_empty_acc() {
    let chain = build_f_prime_honest_chain(1);
    let snapshot = &chain.snapshots[0];
    assert!(
        matches!(snapshot.step_proof.fold, FoldProof::NoFold),
        "first lifecycle step must be the F' base case"
    );
    let b = run_base_check_with_semantic(&chain.prep, snapshot, |_, _, semantic_state_digest_out, _| {
        semantic_state_digest_out[0] += F::ONE;
    });
    assert!(
        !b.is_satisfied(),
        "base F' R1CS accepted semantic_state_digest_out != empty accumulator digest"
    );
}

#[test]
fn lifecycle_base_step_rejects_pc_not_trivial_even_if_source_word_matches() {
    let chain = build_f_prime_honest_chain(1);
    let snapshot = &chain.snapshots[0];
    assert!(
        matches!(snapshot.step_proof.fold, FoldProof::NoFold),
        "first lifecycle step must be the F' base case"
    );
    let b = run_base_check_with_semantic(&chain.prep, snapshot, |f_state, _, _, source| {
        let bad_pc = neo_fold_clean::paper::construction2::TRIVIAL_PC + 1;
        f_state.pc = bad_pc;
        let start = source.pc_word.bits().start();
        for i in 0..64 {
            source.image.set_bit(start + i, ((bad_pc >> i) & 1) == 1);
        }
        let mut forged_state_out = snapshot.state_out.clone();
        forged_state_out.pc = bad_pc;
        let forged_x_out = compute_x_out_native(&chain.prep, &forged_state_out);
        overwrite_enc_inst_bits(&mut source.image, source.public_x_out_bits, forged_x_out);
    });
    assert!(
        !b.is_satisfied(),
        "base F' R1CS accepted pc != TRIVIAL_PC even though the source-image pc word matched"
    );
}

#[test]
fn native_f_prime_verify_rejects_nofold_when_chunk_count_nonzero() {
    let chain = build_f_prime_honest_chain(1);
    let snapshot = &chain.snapshots[0];
    assert!(
        matches!(snapshot.step_proof.fold, FoldProof::NoFold),
        "fixture's first lifecycle step must be the F' NoFold branch"
    );
    let mut forged_state_in = snapshot.state_in.clone();
    forged_state_in.chunk_count = 1;
    forged_state_in.step_count = 1;
    forged_state_in.proof = ProofState::Initial;

    let chunk_digest = f_prime_chunk_public_digest(forged_state_in.step_count, &snapshot.public_batch);
    let empty_acc = AccumulatorHandle::empty().digest();
    let mut forged_state_out = forged_state_in.clone();
    forged_state_out.chunk_count += 1;
    forged_state_out.step_count += snapshot.public_batch.len() as u64;
    forged_state_out.z_i = digest_fields_as_digest32(chunk_digest);
    forged_state_out.public_trace = forged_state_out.z_i;
    forged_state_out.acc_digest = empty_acc;
    forged_state_out.semantic_state_digest = empty_acc;
    let forged_x_out = compute_x_out_native(&chain.prep, &forged_state_out);
    let forged_step = StepProof {
        nebula_open: None,
        fold: FoldProof::NoFold,
        semantic_state_digest: empty_acc,
        x_out: construction2::EncInst::from_digest(digest_fields_as_digest32(forged_x_out)),
    };

    let err = construction2::verify_step(
        &chain.prep.params,
        chain.prep.structure(),
        chain.prep.optimized_cache(),
        chain.prep.structure_digest(),
        chain.prep.mix_rhos_commits(),
        chain.prep.combine_b_pows(),
        &chain.prep.vk,
        forged_state_in,
        &snapshot.public_batch,
        &forged_step,
        chain.prep.semantic_state_mode(),
        None,
    )
    .err()
    .expect("native F' verify_step accepted NoFold with chunk_count > 0");
    assert!(
        matches!(err, construction2::Error::BaseCaseMismatch),
        "wrong rejection for nonzero-counter NoFold branch: {err:?}"
    );
}

#[test]
fn native_f_prime_verify_rejects_empty_nofold_step() {
    let chain = build_f_prime_honest_chain(1);
    let state_in = base_state(&chain.prep);
    let empty_acc = AccumulatorHandle::empty().digest();
    let chunk_digest = f_prime_chunk_public_digest(state_in.step_count, &[]);

    let mut state_out = state_in.clone();
    state_out.chunk_count += 1;
    state_out.z_i = digest_fields_as_digest32(chunk_digest);
    state_out.public_trace = state_out.z_i;
    state_out.acc_digest = empty_acc;
    state_out.semantic_state_digest = empty_acc;
    let x_out = compute_x_out_native(&chain.prep, &state_out);
    let step = StepProof {
        nebula_open: None,
        fold: FoldProof::NoFold,
        semantic_state_digest: empty_acc,
        x_out: construction2::EncInst::from_digest(digest_fields_as_digest32(x_out)),
    };

    let err = construction2::verify_step(
        &chain.prep.params,
        chain.prep.structure(),
        chain.prep.optimized_cache(),
        chain.prep.structure_digest(),
        chain.prep.mix_rhos_commits(),
        chain.prep.combine_b_pows(),
        &chain.prep.vk,
        state_in,
        &[],
        &step,
        chain.prep.semantic_state_mode(),
        None,
    )
    .err()
    .expect("native F' verify_step accepted an empty NoFold step");
    assert!(
        matches!(err, construction2::Error::EmptyStep),
        "wrong rejection for empty F' step: {err:?}"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_zero_step_count_even_with_matching_fresh_and_nifs_proof() {
    let chain = build_f_prime_honest_chain(3);
    let snapshot = &chain.snapshots[2];
    let ProofState::Active { running, .. } = &snapshot.state_in.proof else {
        panic!("step 2 must enter the recursive branch");
    };
    let running = running
        .materialize()
        .expect("recursive step running materialization");
    assert!(
        !running.claims.is_empty(),
        "fixture must carry a non-empty running accumulator"
    );
    let mut forged_state_in = snapshot.state_in.clone();
    forged_state_in.step_count = 0;
    let forged_prior_x_out = compute_x_out_native(&chain.prep, &forged_state_in);
    let forged_fresh = build_link_instance(&chain.prep, &bit_carrier_r1cs(), forged_prior_x_out);
    let fresh_claims = vec![forged_fresh.claim.clone()];

    let forged_chunk_digest = f_prime_chunk_public_digest(forged_state_in.step_count, &snapshot.public_batch);
    let mut tr = neo_fold_clean::paper::f_prime::native::f_prime_step_transcript(
        &chain.prep.vk,
        chain.prep.structure_digest(),
        &forged_state_in,
        forged_chunk_digest,
    );
    let (forged_running_out, forged_nifs) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &chain.prep.params,
        chain.prep.structure(),
        chain.prep.optimized_cache(),
        &chain.prep.log,
        None,
        chain.prep.mix_rhos_commits(),
        chain.prep.combine_b_pows(),
        vec![forged_fresh],
        &running,
    )
    .expect("forge internally consistent NIFS proof under zero step_count");
    let forged_parent = forged_running_out
        .parent_authority
        .as_ref()
        .expect("recursive output must carry parent authority");
    let forged_acc_digest =
        AccumulatorHandle::from_running_parts(&forged_running_out.claims, Some(forged_parent)).digest();

    let mut forged_state_out = forged_state_in.clone();
    forged_state_out.chunk_count += 1;
    forged_state_out.step_count += snapshot.public_batch.len() as u64;
    forged_state_out.z_i = digest_fields_as_digest32(forged_chunk_digest);
    forged_state_out.public_trace = forged_state_out.z_i;
    forged_state_out.acc_digest = forged_acc_digest;
    forged_state_out.semantic_state_digest = forged_acc_digest;
    let forged_public_x_out = compute_x_out_native(&chain.prep, &forged_state_out);

    let f_state = f_prime_state_in(&forged_state_in, &chain.prep);
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(f_state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(f_state.step_count_in);
    let pc_word = image.push_u64_le(f_state.pc);
    let prior_public = image.push_f_prime_public_input(forged_prior_x_out);
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(forged_public_x_out);

    let cfg = make_step_config(&chain.prep);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: digest32_as_fields(forged_acc_digest),
        acc_digest_out: digest32_as_fields(forged_acc_digest),
        state: f_state,
        chunk_digest: forged_chunk_digest,
        nifs_msg: NifsVCircuitMessages {
            fresh: &fresh_claims,
            running: &running.claims,
            running_parent_authority: running.parent_authority.as_ref(),
            running_pending_projection: running.pending_projection(),
            pi_ccs: &forged_nifs.pi_ccs,
            combined: &forged_nifs.pi_rlc.combined,
            children: &forged_nifs.pi_dec.children,
        },
        rows_in_chunk: snapshot.public_batch.len() as u64,
        source_image: &image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &chain.prep.params, &cfg, &inputs)
        .expect("emit forged recursive F' R1CS");
    assert!(
        !b.is_satisfied(),
        "F' recursive R1CS accepted Active branch with step_count_in = 0 after the folded fresh \
         instance, NIFS proof, and x_out links were rebuilt around that impossible coordinate"
    );
}

#[test]
fn lifecycle_recursive_step_satisfies_f_prime_r1cs() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    assert!(
        !view.running_claims.is_empty(),
        "step 2 must see a non-empty running accumulator"
    );
    let b = run_recursive_check(&view, |_, _, _, _, _| ());
    assert!(
        b.is_satisfied(),
        "F' R1CS rejected a real lifecycle chain step (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn lifecycle_recursive_step_rejects_tampered_chunk_digest() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check(&view, |_, chunk_digest, _, _, _| {
        chunk_digest[0] += F::ONE;
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a step with a tampered chunk digest"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_tampered_prior_source_image_bit() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check(&view, |_, _, _, source, _| {
        let idx = source.prior_x_out_bits.start();
        let original = source.image.values()[idx];
        source.image.set_bit(idx, original == F::ZERO);
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a step with a tampered prior-x_out source-image bit"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_tampered_pc_source_word_bit() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check(&view, |_, _, _, source, _| {
        let idx = source.pc_word.bits().start();
        let original = source.image.values()[idx];
        source.image.set_bit(idx, original == F::ZERO);
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a step whose pc source-image word diverged from the state wire"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_tampered_acc_digest_in() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check(&view, |f_state, _, _, _, _| {
        f_state.acc_digest_in[0] += F::ONE;
    });
    assert!(!b.is_satisfied(), "F' R1CS accepted a step with tampered acc_digest_in");
}

#[test]
fn lifecycle_recursive_step_rejects_tampered_acc_digest_in_even_if_prior_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check(&view, |f_state, _, _, source, _| {
        f_state.acc_digest_in[0] += F::ONE;
        let mut tampered_state = view.state_in.clone();
        tampered_state.acc_digest = digest_fields_as_digest32(f_state.acc_digest_in);
        let tampered_prior_x_out = compute_x_out_native(view.prep, &tampered_state);
        overwrite_enc_inst_bits(&mut source.image, source.prior_x_out_bits, tampered_prior_x_out);
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a tampered incoming accumulator handle even after prior_x_out was rebuilt"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_running_child_field_tamper_even_if_handle_and_prior_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    assert!(
        !view.running_claims.is_empty(),
        "test setup needs a recursive step with non-empty running"
    );

    let mut running_claims = view.running_claims.to_vec();
    let parent_authority = view.running_parent_authority.clone();
    running_claims[0].y_ring[0][0] += k_c1_one();
    let forged_acc_digest =
        AccumulatorHandle::from_running_parts(&running_claims, parent_authority.as_ref()).digest_fields();
    let mut f_state = f_prime_state_in(view.state_in, view.prep);
    f_state.acc_digest_in = forged_acc_digest;
    f_state.semantic_state_digest_in = forged_acc_digest;
    let mut source = build_source_image(&view, &f_state);
    let mut forged_state_in = view.state_in.clone();
    forged_state_in.acc_digest = digest_fields_as_digest32(forged_acc_digest);
    forged_state_in.semantic_state_digest = digest_fields_as_digest32(forged_acc_digest);
    let forged_prior_x_out = compute_x_out_native(view.prep, &forged_state_in);
    overwrite_enc_inst_bits(&mut source.image, source.prior_x_out_bits, forged_prior_x_out);

    let cfg = make_step_config(view.prep);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: digest32_as_fields(view.state_out.semantic_state_digest),
        acc_digest_out: digest32_as_fields(view.state_out.acc_digest),
        state: f_state,
        chunk_digest: view.chunk_digest,
        nifs_msg: NifsVCircuitMessages {
            fresh: &view.fresh,
            running: &running_claims,
            running_parent_authority: parent_authority.as_ref(),
            running_pending_projection: view.running_pending_projection.as_ref(),
            pi_ccs: &view.nifs.pi_ccs,
            combined: &view.nifs.pi_rlc.combined,
            children: &view.nifs.pi_dec.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &view.prep.params, &cfg, &inputs).expect("emit F' R1CS");
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a mutated running child CE field after the parent handle \
         and prior_x_out source bits were rebuilt around that mutation"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_running_child_fold_digest_tamper_even_if_handle_and_prior_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let mut running_claims = view.running_claims.to_vec();
    running_claims[0].fold_digest[0] ^= 0x80;
    let parent_authority = view.running_parent_authority.clone();
    let forged_acc_digest =
        AccumulatorHandle::from_running_parts(&running_claims, parent_authority.as_ref()).digest_fields();
    let mut f_state = f_prime_state_in(view.state_in, view.prep);
    f_state.acc_digest_in = forged_acc_digest;
    f_state.semantic_state_digest_in = forged_acc_digest;
    let mut source = build_source_image(&view, &f_state);
    let mut forged_state_in = view.state_in.clone();
    forged_state_in.acc_digest = digest_fields_as_digest32(forged_acc_digest);
    forged_state_in.semantic_state_digest = digest_fields_as_digest32(forged_acc_digest);
    let forged_prior_x_out = compute_x_out_native(view.prep, &forged_state_in);
    overwrite_enc_inst_bits(&mut source.image, source.prior_x_out_bits, forged_prior_x_out);

    let cfg = make_step_config(view.prep);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: digest32_as_fields(view.state_out.semantic_state_digest),
        acc_digest_out: digest32_as_fields(view.state_out.acc_digest),
        state: f_state,
        chunk_digest: view.chunk_digest,
        nifs_msg: NifsVCircuitMessages {
            fresh: &view.fresh,
            running: &running_claims,
            running_parent_authority: parent_authority.as_ref(),
            running_pending_projection: view.running_pending_projection.as_ref(),
            pi_ccs: &view.nifs.pi_ccs,
            combined: &view.nifs.pi_rlc.combined,
            children: &view.nifs.pi_dec.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &view.prep.params, &cfg, &inputs).expect("emit F' R1CS");
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a running child fold_digest tamper after the parent handle \
         and prior_x_out source bits were rebuilt around that mutation"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_running_parent_field_tamper_even_if_handle_and_prior_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let mut parent_authority = view
        .running_parent_authority
        .clone()
        .expect("test setup needs running parent authority");
    parent_authority.y_ring[0][0] += k_c1_one();
    let forged_acc_digest =
        AccumulatorHandle::from_running_parts(view.running_claims.as_slice(), Some(&parent_authority)).digest_fields();
    let mut f_state = f_prime_state_in(view.state_in, view.prep);
    f_state.acc_digest_in = forged_acc_digest;
    f_state.semantic_state_digest_in = forged_acc_digest;
    let mut source = build_source_image(&view, &f_state);
    let mut forged_state_in = view.state_in.clone();
    forged_state_in.acc_digest = digest_fields_as_digest32(forged_acc_digest);
    forged_state_in.semantic_state_digest = digest_fields_as_digest32(forged_acc_digest);
    let forged_prior_x_out = compute_x_out_native(view.prep, &forged_state_in);
    overwrite_enc_inst_bits(&mut source.image, source.prior_x_out_bits, forged_prior_x_out);

    let cfg = make_step_config(view.prep);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: digest32_as_fields(view.state_out.semantic_state_digest),
        acc_digest_out: digest32_as_fields(view.state_out.acc_digest),
        state: f_state,
        chunk_digest: view.chunk_digest,
        nifs_msg: NifsVCircuitMessages {
            fresh: &view.fresh,
            running: view.running_claims.as_slice(),
            running_parent_authority: Some(&parent_authority),
            running_pending_projection: view.running_pending_projection.as_ref(),
            pi_ccs: &view.nifs.pi_ccs,
            combined: &view.nifs.pi_rlc.combined,
            children: &view.nifs.pi_dec.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &view.prep.params, &cfg, &inputs).expect("emit F' R1CS");
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a mutated running parent-authority CE field after the parent \
         handle and prior_x_out source bits were rebuilt around that mutation"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_tampered_acc_digest_out_without_matching_x_out() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check(&view, |_, _, acc_digest_out, _, _| {
        acc_digest_out[0] += F::ONE;
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a tampered outgoing accumulator handle that was not reflected in state_x_out"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_coherent_wrong_acc_digest_out() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check_with_semantic(&view, |_, _, acc_digest_out, semantic_state_digest_out, source, _| {
        acc_digest_out[0] += F::ONE;
        *semantic_state_digest_out = *acc_digest_out;
        let mut forged_state_out = view.state_out.clone();
        forged_state_out.acc_digest = digest_fields_as_digest32(*acc_digest_out);
        forged_state_out.semantic_state_digest = digest_fields_as_digest32(*semantic_state_digest_out);
        let forged_x_out = compute_x_out_native(view.prep, &forged_state_out);
        overwrite_enc_inst_bits(&mut source.image, source.public_x_out_bits, forged_x_out);
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a coherent forged outgoing accumulator handle in a real lifecycle step"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_output_child_field_tamper_even_if_handle_and_public_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let mut children = view.nifs.pi_dec.children.clone();
    assert!(
        !children.is_empty(),
        "test setup needs NIFS output children in the recursive step"
    );
    children[0].y_ring[0][0] += k_c1_one();
    let combined = view.nifs.pi_rlc.combined.clone();
    let forged_acc_digest = AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields();
    let mut source = build_source_image(&view, &f_prime_state_in(view.state_in, view.prep));
    let mut forged_state_out = view.state_out.clone();
    forged_state_out.acc_digest = digest_fields_as_digest32(forged_acc_digest);
    forged_state_out.semantic_state_digest = digest_fields_as_digest32(forged_acc_digest);
    let forged_x_out = compute_x_out_native(view.prep, &forged_state_out);
    overwrite_enc_inst_bits(&mut source.image, source.public_x_out_bits, forged_x_out);

    let b = run_recursive_check_with_output_authority(
        &view,
        &combined,
        &children,
        forged_acc_digest,
        forged_acc_digest,
        source,
    );
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a mutated NIFS output child after acc_digest_out and public x_out \
         were rebuilt around that mutated child"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_output_child_fold_digest_tamper_even_if_handle_and_public_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let mut children = view.nifs.pi_dec.children.clone();
    children[0].fold_digest[0] ^= 0x80;
    let combined = view.nifs.pi_rlc.combined.clone();
    let forged_acc_digest = AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields();
    let mut source = build_source_image(&view, &f_prime_state_in(view.state_in, view.prep));
    let mut forged_state_out = view.state_out.clone();
    forged_state_out.acc_digest = digest_fields_as_digest32(forged_acc_digest);
    forged_state_out.semantic_state_digest = digest_fields_as_digest32(forged_acc_digest);
    let forged_x_out = compute_x_out_native(view.prep, &forged_state_out);
    overwrite_enc_inst_bits(&mut source.image, source.public_x_out_bits, forged_x_out);

    let b = run_recursive_check_with_output_authority(
        &view,
        &combined,
        &children,
        forged_acc_digest,
        forged_acc_digest,
        source,
    );
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a NIFS output child fold_digest tamper after acc_digest_out \
         and public x_out were rebuilt around that mutation"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_output_parent_field_tamper_even_if_handle_and_public_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let children = view.nifs.pi_dec.children.clone();
    let mut combined = view.nifs.pi_rlc.combined.clone();
    combined.y_ring[0][0] += k_c1_one();
    let forged_acc_digest = AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields();
    let mut source = build_source_image(&view, &f_prime_state_in(view.state_in, view.prep));
    let mut forged_state_out = view.state_out.clone();
    forged_state_out.acc_digest = digest_fields_as_digest32(forged_acc_digest);
    forged_state_out.semantic_state_digest = digest_fields_as_digest32(forged_acc_digest);
    let forged_x_out = compute_x_out_native(view.prep, &forged_state_out);
    overwrite_enc_inst_bits(&mut source.image, source.public_x_out_bits, forged_x_out);

    let b = run_recursive_check_with_output_authority(
        &view,
        &combined,
        &children,
        forged_acc_digest,
        forged_acc_digest,
        source,
    );
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted a mutated NIFS output parent after acc_digest_out and public x_out \
         were rebuilt around that mutated parent"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_transcript_prefix_tamper_after_x_out_repair() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);

    for anchor in [
        "vk_fs_digest",
        "pi_ccs_header_bundle",
        "z_0",
        "z_i_in",
        "public_trace_in",
    ] {
        let b = run_recursive_check_with_semantic(
            &view,
            |f_state, _, acc_digest_out, semantic_state_digest_out, source, _| {
                match anchor {
                    "vk_fs_digest" => f_state.vk_fs_digest[0] += F::ONE,
                    "pi_ccs_header_bundle" => f_state.pi_ccs_header_bundle[0] += F::ONE,
                    "z_0" => f_state.z_0[0] += F::ONE,
                    "z_i_in" => f_state.z_i_in[0] += F::ONE,
                    "public_trace_in" => f_state.public_trace_in[0] += F::ONE,
                    _ => unreachable!(),
                }
                // Repair both visible x_out links using the tampered
                // anchor fields. The verifier header is now directly in the
                // state_x_out preimage; the remaining repairs keep this test
                // focused on transcript and state binding.
                let forged_prior_x_out = digest32_as_fields(state_x_out_digest_with_mode(
                    StateXOutDigestMode::Stateless,
                    digest_fields_as_digest32(f_state.vk_fs_digest),
                    f_state.pi_ccs_header_bundle,
                    &f_state.pi_ccs_header_bundle,
                    f_state.chunk_count_in,
                    f_state.step_count_in,
                    digest_fields_as_digest32(f_state.z_0),
                    digest_fields_as_digest32(f_state.z_i_in),
                    f_state.pc,
                    digest_fields_as_digest32(f_state.semantic_state_digest_in),
                    digest_fields_as_digest32(f_state.acc_digest_in),
                    digest_fields_as_digest32(f_state.public_trace_in),
                    None,
                ));
                overwrite_enc_inst_bits(&mut source.image, source.prior_x_out_bits, forged_prior_x_out);

                let forged_public_x_out = digest32_as_fields(state_x_out_digest_with_mode(
                    StateXOutDigestMode::Stateless,
                    digest_fields_as_digest32(f_state.vk_fs_digest),
                    f_state.pi_ccs_header_bundle,
                    &f_state.pi_ccs_header_bundle,
                    view.state_out.chunk_count,
                    view.state_out.step_count,
                    view.state_out.z_0,
                    view.state_out.z_i,
                    view.state_out.pc,
                    digest_fields_as_digest32(*semantic_state_digest_out),
                    digest_fields_as_digest32(*acc_digest_out),
                    view.state_out.public_trace,
                    None,
                ));
                overwrite_enc_inst_bits(&mut source.image, source.public_x_out_bits, forged_public_x_out);
            },
        );
        assert!(
            !b.is_satisfied(),
            "F' R1CS accepted a coherent {anchor} relabel; the NIFS.V transcript prefix must bind it"
        );
    }
}

#[test]
fn lifecycle_recursive_step_rejects_semantic_digest_out_not_equal_acc_digest_out() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let b = run_recursive_check_with_semantic(&view, |_, _, _, semantic_state_digest_out, _, _| {
        semantic_state_digest_out[0] += F::ONE;
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted semantic_state_digest_out != acc_digest_out in stateless mode"
    );
}

#[test]
fn lifecycle_all_recursive_steps_satisfy_f_prime_r1cs() {
    let chain = build_f_prime_honest_chain(6);

    let mut recursive_count = 0;
    let mut saw_nonempty_running = false;
    for idx in 0..chain.snapshots.len() {
        if !chain.is_recursive(idx) {
            continue;
        }
        recursive_count += 1;

        let view = chain.recursive_step(idx);
        assert!(
            !view.running_claims.is_empty(),
            "every recursive step must consume the paper default or a folded accumulator",
        );
        saw_nonempty_running = true;

        let b = run_recursive_check(&view, |_, _, _, _, _| ());
        assert!(
            b.is_satisfied(),
            "F' R1CS rejected step {idx} of a real lifecycle chain (first bad row: {:?})",
            b.first_unsatisfied_row()
        );
    }
    assert_eq!(
        recursive_count, 5,
        "chain(6) should contain 5 FoldProof::Recursive steps (indices 1..5)"
    );
    assert!(
        saw_nonempty_running,
        "full-chain replay should cover recursive steps with non-empty running"
    );
}

#[test]
fn lifecycle_verify_uncompressed_rejects_multi_chunk_f_prime_terminal_only_scope() {
    let chain = build_f_prime_honest_chain(2);
    let audit = UncompressedAudit {
        proof: Uncompressed {
            state: chain
                .snapshots
                .last()
                .expect("linked chain has a final state")
                .state_out
                .clone(),
            final_fold: None,
        },
        steps: chain
            .snapshots
            .iter()
            .map(|s| s.step_proof.clone())
            .collect(),
        public_batches: chain
            .snapshots
            .iter()
            .map(|s| s.public_batch.clone())
            .collect(),
    };
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&chain.prep, audit).expect("finish linked chain");

    let err = neo_fold_clean::verify_uncompressed(&chain.prep, &finished.proof)
        .expect_err("terminal-only verifier accepted a multi-chunk proof without replaying the induction");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::TerminalOnlyMultiChunkUnsupported { chunk_count: 2 }
        ),
        "expected TerminalOnlyMultiChunkUnsupported(2), got {err:?}"
    );

    neo_fold_clean::verify_uncompressed_audit(&chain.prep, &finished)
        .expect("audit verifier accepts the same honest multi-chunk history");
}

#[test]
fn lifecycle_compress_uses_audit_path_for_multi_chunk_f_prime_until_decider_lands() {
    let chain = build_f_prime_honest_chain(2);
    let audit = UncompressedAudit {
        proof: Uncompressed {
            state: chain
                .snapshots
                .last()
                .expect("linked chain has a final state")
                .state_out
                .clone(),
            final_fold: None,
        },
        steps: chain
            .snapshots
            .iter()
            .map(|s| s.step_proof.clone())
            .collect(),
        public_batches: chain
            .snapshots
            .iter()
            .map(|s| s.public_batch.clone())
            .collect(),
    };

    let err = neo_fold_clean::compress(&chain.prep, audit)
        .err()
        .expect("compress must fail only at the unsupported decider layer for honest multi-chunk F'");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Decider(neo_fold_clean::paper::decider::Error::Unsupported)
        ),
        "compress must use the audit replay path for multi-chunk F' before hitting the decider placeholder; got {err:?}"
    );
}

#[test]
fn f_prime_chunk_public_digest_is_independent_of_recursive_link_x() {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    let z_a: Vec<F> = (0..prep.structure().m)
        .map(|i| F::from_u64((i as u64) & 1))
        .collect();
    let z_b: Vec<F> = (0..prep.structure().m)
        .map(|i| F::from_u64(((i + 1) as u64) & 1))
        .collect();
    let inst_a = direct_ccs::build_instance(&prep, &r1cs, &z_a).expect("inst a");
    let inst_b = direct_ccs::build_instance(&prep, &r1cs, &z_b).expect("inst b");
    assert_ne!(inst_a.claim.x, inst_b.claim.x, "claims must differ in x");
    assert_ne!(
        inst_a.claim.c.data, inst_b.claim.c.data,
        "in direct-CCS, distinct x implies distinct commitments"
    );
    assert_eq!(inst_a.claim.m_in, inst_b.claim.m_in);
    assert_eq!(inst_a.claim.c.d, inst_b.claim.c.d);
    assert_eq!(inst_a.claim.c.kappa, inst_b.claim.c.kappa);
    let f_prime_a = f_prime_chunk_public_digest(0, &[inst_a.claim.clone()]);
    let f_prime_b = f_prime_chunk_public_digest(0, &[inst_b.claim.clone()]);
    assert_eq!(
        f_prime_a, f_prime_b,
        "f_prime_chunk_public_digest must ignore both claim.x and claim.c.data — \
         otherwise the recursive-link fixed point reappears"
    );
    let ord_a = chunk_public_digest(0, &[inst_a.claim.clone()]);
    let ord_b = chunk_public_digest(0, &[inst_b.claim.clone()]);
    assert_ne!(
        ord_a, ord_b,
        "ordinary chunk_public_digest must still bind claim.x for non-F' CCS identity uses"
    );
    let claim_a = ccs_claim_digest(&inst_a.claim);
    let claim_b = ccs_claim_digest(&inst_b.claim);
    assert_ne!(claim_a, claim_b, "ordinary ccs_claim_digest must remain x-sensitive");
}

#[test]
fn nifs_transcript_binds_chunk_contents_even_though_f_prime_digest_is_shape_only() {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let z_a: Vec<F> = (0..prep.structure().m)
        .map(|i| F::from_u64((i as u64) & 1))
        .collect();
    let z_b: Vec<F> = (0..prep.structure().m)
        .map(|i| F::from_u64(((i + 1) as u64) & 1))
        .collect();
    let inst_a = direct_ccs::build_instance(&prep, &r1cs, &z_a).expect("inst a");
    let inst_b = direct_ccs::build_instance(&prep, &r1cs, &z_b).expect("inst b");
    assert_ne!(inst_a.claim.x, inst_b.claim.x, "same shape, different x");
    assert_ne!(
        inst_a.claim.c.data, inst_b.claim.c.data,
        "Ajtai commitment over full z makes c.data differ too"
    );
    assert_eq!(inst_a.claim.m_in, inst_b.claim.m_in);
    assert_eq!(
        f_prime_chunk_public_digest(0, &[inst_a.claim.clone()]),
        f_prime_chunk_public_digest(0, &[inst_b.claim.clone()]),
        "F' chunk digest must collapse same-shape claims (regression for the recursive-link fixed point)"
    );
    let empty_running: Vec<CeClaim> = Vec::new();
    assert_ne!(
        pi_ccs_instance_digest(&[inst_a.claim.clone()], &empty_running),
        pi_ccs_instance_digest(&[inst_b.claim.clone()], &empty_running),
        "Π_CCS instance digest must distinguish A vs B; otherwise NIFS FS challenges would not depend on the folded claim's contents"
    );
    let chain = build_f_prime_honest_chain(2);
    let audit = UncompressedAudit {
        proof: Uncompressed {
            state: chain
                .snapshots
                .last()
                .expect("linked chain has a final state")
                .state_out
                .clone(),
            final_fold: None,
        },
        steps: chain
            .snapshots
            .iter()
            .map(|s| s.step_proof.clone())
            .collect(),
        public_batches: chain
            .snapshots
            .iter()
            .map(|s| s.public_batch.clone())
            .collect(),
    };
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&chain.prep, audit).expect("finish linked chain");
    assert!(
        matches!(
            neo_fold_clean::verify_uncompressed(&chain.prep, &finished.proof),
            Err(neo_fold_clean::Error::TerminalOnlyMultiChunkUnsupported { chunk_count: 2 })
        ),
        "terminal-only verifier must reject multi-chunk direct-CCS history"
    );
    neo_fold_clean::verify_uncompressed_audit(&chain.prep, &finished).expect("untampered audit proof verifies");
    let untampered_statement = neo_fold_clean::build_decider_statement(&chain.prep, &finished);
    neo_fold_clean::paper::decider::validate_witness(
        &chain.prep.params,
        chain.prep.structure(),
        chain.prep.optimized_cache(),
        chain.prep.structure_digest(),
        &chain.prep.log,
        chain.prep.mix_rhos_commits(),
        chain.prep.combine_b_pows(),
        &chain.prep.vk,
        chain.prep.public_input_len,
        chain.prep.enforces_f_prime_recursive_link(),
        chain.prep.enforces_terminal_induction(),
        chain.prep.semantic_state_mode(),
        chain.prep.initial_semantic_state_digest(),
        None,
        &untampered_statement,
    )
    .expect("untampered statement passes validate_witness");
    let mut tampered_statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    assert_eq!(tampered_statement.witness.public_batches[0].len(), 1);
    tampered_statement.witness.public_batches[0][0] = inst_b.claim.clone();

    assert!(
        neo_fold_clean::paper::decider::validate_witness(
            &chain.prep.params,
            chain.prep.structure(),
            chain.prep.optimized_cache(),
            chain.prep.structure_digest(),
            &chain.prep.log,
            chain.prep.mix_rhos_commits(),
            chain.prep.combine_b_pows(),
            &chain.prep.vk,
            chain.prep.public_input_len,
            chain.prep.enforces_f_prime_recursive_link(),
            chain.prep.enforces_terminal_induction(),
            chain.prep.semantic_state_mode(),
            chain.prep.initial_semantic_state_digest(),
            None,
            &tampered_statement,
        )
        .is_err(),
        "validate_witness accepted a statement whose folded batch's full claim was swapped for a \
         same-shape distinct-content one — NIFS / Π_CCS transcript would not be binding chunk contents \
         before content-checking challenges"
    );
}
