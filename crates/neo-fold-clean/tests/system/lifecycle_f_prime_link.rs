//! Lifecycle ↔ F' R1CS contract.
//!
//! Drives a real `construction2::step` chain (the same path
//! `lifecycle::extend` calls internally), with each fresh batch's public
//! `x` set to `enc_inst(prior_x_out)` for the *next* step. Snapshots every
//! step's `(state_in, public_batch, state_out, step_proof)` and feeds each
//! recursive step into `enforce_f_prime_recursive_step_circuit`.
//!
//! Two layers of coverage:
//!
//! 1. **Single-step hard gate + tamper**: a 3-step chain, verify the
//!    `running.claims`-non-empty step (index 2) under several tampers.
//! 2. **Full-chain replay gate**: a 6-step chain, verify *every* recursive
//!    step. This is the pre-decider invariant: if every native lifecycle
//!    step can be replayed by F' R1CS, the remaining decider work is just
//!    packaging the same relation.
//!
//! ## Why this can be a real chain, not a hand-crafted state
//!
//! F' state advance uses `f_prime_chunk_public_digest`, which absorbs only
//! `(commitment shape: d, kappa)` and `m_in` from each fresh claim —
//! **neither `claim.x` nor `claim.c.data`**, both of which depend on the
//! recursive-link `x` in direct-CCS (where the Ajtai commitment covers
//! the full assignment `z = [x | w]`). That breaks the recursive-link
//! fixed point
//!
//!   `x_i = enc_inst(state_x_out(state_{i+1}(chunk_digest(x_i))))`,
//!
//! so the chain advances deterministically from the protocol shape and
//! the caller is free to set `x_i = enc_inst(prior_x_out_{i+1})` after
//! observing `state_{i+1}`. Commitments are still bound to the chain via
//! the running-accumulator digest and re-checked inside NIFS.V.
//!
//! The ordinary `chunk_public_digest` (which still binds `claim.x` and
//! `claim.c.data`) is unchanged and continues to be used for non-F'
//! CCS-identity purposes.

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
    encode_f_prime_public_input, encode_x_out_public_bits, enforce_f_prime_base_step_circuit,
    enforce_f_prime_recursive_step_circuit, FPrimeBaseInputs, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig,
    F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance, CeClaim};
use neo_fold_clean::{Uncompressed, UncompressedAudit};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

// ── Fixture helpers ─────────────────────────────────────────────────────────

/// All-zero R1CS with `m_in = F_PRIME_PUBLIC_INPUT_LEN`. Any low-norm
/// assignment satisfies it.
fn bit_carrier_r1cs() -> R1cs {
    R1cs {
        a: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        b: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        c: Mat::zero(1, F_PRIME_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_PUBLIC_INPUT_LEN,
    }
}

fn k_c1_one() -> K {
    K::from_coeffs([F::ZERO, F::ONE])
}

/// Native mirror of `paper::construction2::compute_x_out` (which is
/// `pub(crate)`). Same absorb sequence as `state_x_out_digest`.
fn compute_x_out_native(prep: &neo_fold_clean::Preprocessing, state: &State) -> [F; 4] {
    let mode = match prep.semantic_state_mode() {
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    digest32_as_fields(state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        &structure_digest(prep.structure()),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
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
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
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
        structure_digest: structure_digest(prep.structure()),
        chunk_count_in: state.chunk_count,
        step_count_in: state.step_count,
        z_0: digest32_as_fields(state.z_0),
        z_i_in: digest32_as_fields(state.z_i),
        pc: state.pc,
        acc_digest_in: digest32_as_fields(state.acc_digest),
        semantic_state_digest_in: digest32_as_fields(state.semantic_state_digest),
        public_trace_in: digest32_as_fields(state.public_trace),
    }
}

fn base_state(prep: &neo_fold_clean::Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = AccumulatorHandle::empty().digest();
    State::base(z_0, public_trace, acc_digest, acc_digest)
}

/// CCS instance whose public-input bits encode `enc_inst(x_out_target)`.
/// The Ajtai commitment over the resulting `z` depends on `x_out_target`,
/// but `f_prime_chunk_public_digest` absorbs only the commitment shape and
/// `m_in`, so the chain advance is unaffected.
fn build_link_instance(prep: &neo_fold_clean::Preprocessing, r1cs: &R1cs, x_out_target: [F; 4]) -> CcsInstance {
    let mut z = encode_f_prime_public_input(x_out_target);
    z.resize(prep.structure().m, F::ZERO);
    direct_ccs::build_instance(prep, r1cs, &z).expect("recursive-link instance")
}

/// Run one F'-step on a cloned state, returning the would-be post-step
/// state. Used to peek at `state_{i+1}` so we can pick the next batch's
/// `x` before committing to it.
fn peek_next_state(prep: &neo_fold_clean::Preprocessing, state: &State, batch: &[CcsInstance]) -> State {
    let (next, _) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state.clone(),
        batch.to_vec(),
    )
    .expect("peek step");
    next
}

// ── Chain fixture ───────────────────────────────────────────────────────────

/// One step's full record from a real `construction2::step` chain.
struct StepSnapshot {
    /// State going into this step (before its NIFS runs).
    state_in: State,
    /// State coming out of this step (after `advance_state`).
    state_out: State,
    /// Claims of the batch deposited as `latest` at this step.
    public_batch: Vec<CcsClaim>,
    /// The `StepProof` this step produced.
    step_proof: StepProof,
}

struct ChainFixture {
    prep: neo_fold_clean::Preprocessing,
    snapshots: Vec<StepSnapshot>,
}

/// Build an F'-honest chain of `len` steps. Each batch's `x` is
/// `enc_inst(prior_x_out)` for that step's recursive-link verifier check,
/// so every `FoldProof::Recursive` snapshot is independently F'-replayable.
fn build_f_prime_honest_chain(len: usize) -> ChainFixture {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy_inst = || direct_ccs::build_instance(&prep, &r1cs, &placeholder_z).expect("dummy");

    let mut state = base_state(&prep);
    let mut snapshots = Vec::with_capacity(len);

    for _ in 0..len {
        // Peek at the post-step state to recover the prior_x_out the F'
        // R1CS will recompute, then set this batch's `x` accordingly.
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
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &prep.vk,
            state,
            vec![batch],
        )
        .expect("step");

        // Sanity: real step matches the peek (since the chunk digest is
        // shape-only, batch.x doesn't change state advance).
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

/// Borrowed view of one recursive step's F' R1CS inputs.
struct RecursiveStepView<'a> {
    prep: &'a neo_fold_clean::Preprocessing,
    state_in: &'a State,
    state_out: &'a State,
    fresh: Vec<CcsClaim>,
    running_claims: &'a [CeClaim],
    running_parent_authority: Option<&'a CeClaim>,
    nifs: &'a NifsProof,
    chunk_digest: [F; 4],
    prior_x_out: [F; 4],
    post_step_x_out: [F; 4],
}

impl ChainFixture {
    /// View of the recursive step at `idx`. Panics if the step is not
    /// `FoldProof::Recursive` (use `is_recursive(idx)` first if unsure).
    fn recursive_step(&self, idx: usize) -> RecursiveStepView<'_> {
        let snapshot = &self.snapshots[idx];
        let FoldProof::Recursive(nifs) = &snapshot.step_proof.fold else {
            panic!("step {idx} is not FoldProof::Recursive");
        };
        let (running_claims, running_parent_authority, fresh) = match &snapshot.state_in.proof {
            ProofState::Active { running, latest } => (
                running.claims.as_slice(),
                running.parent_authority.as_ref(),
                latest.claims(),
            ),
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

// ── F' R1CS checks, parameterized by real lifecycle snapshots ───────────────

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
            running: view.running_claims,
            running_parent_authority: view.running_parent_authority,
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
            running: view.running_claims,
            running_parent_authority: view.running_parent_authority,
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

// ── Single-step hard gate + tampers (uses chain of 3, step index 2) ─────────
//
// Step 2 is the first recursive step with a non-empty running accumulator,
// the most interesting case for tampering.

#[test]
fn lifecycle_base_step_rejects_semantic_digest_out_not_equal_empty_acc() {
    let chain = build_f_prime_honest_chain(1);
    let snapshot = &chain.snapshots[0];
    assert!(
        matches!(snapshot.step_proof.fold, FoldProof::NoFold),
        "first lifecycle step must be the F' base case"
    );
    let b = run_base_check_with_semantic(&chain.prep, snapshot, |_, _, semantic_state_digest_out, _| {
        // Base F' sets U_1 to the default empty accumulator. In
        // stateless mode, semantic state is the accumulator digest, while
        // state_x_out omits the duplicate semantic lane. Mutating only
        // this side field should be rejected by the same
        // `semantic_state_digest_out == acc_digest_out` row used by the
        // recursive path.
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
        // Make the state wire, source-image word, and visible x_out bits
        // agree on the same wrong pc. The explicit `pc == TRIVIAL_PC` row
        // must still reject this coherent relabel.
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
        // Flip only the low source bit; the state wire still carries
        // TRIVIAL_PC. The source-image word must be bound directly to the
        // in-circuit state var before that var is absorbed into x_out.
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

        // Rebuild the prior-x_out source bits from the tampered
        // accumulator handle so this is not merely the public-link bit
        // check failing. The recursive step must still reject because
        // the consumed running accumulator recomputes to the honest
        // handle, not this tampered one.
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
    let parent_authority = view.running_parent_authority.cloned();
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
        "F' R1CS accepted a mutated running child CE field after the full-running handle \
         and prior_x_out source bits were rebuilt around that mutation"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_running_child_fold_digest_tamper_even_if_handle_and_prior_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let mut running_claims = view.running_claims.to_vec();
    running_claims[0].fold_digest[0] ^= 0x80;
    let parent_authority = view.running_parent_authority.cloned();
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
        "F' R1CS accepted a running child fold_digest tamper after the full-running handle \
         and prior_x_out source bits were rebuilt around that mutation"
    );
}

#[test]
fn lifecycle_recursive_step_rejects_running_parent_field_tamper_even_if_handle_and_prior_x_out_rebuilt() {
    let chain = build_f_prime_honest_chain(3);
    let view = chain.recursive_step(2);
    let mut parent_authority = view
        .running_parent_authority
        .cloned()
        .expect("test setup needs running parent authority");
    parent_authority.y_ring[0][0] += k_c1_one();
    let forged_acc_digest =
        AccumulatorHandle::from_running_parts(view.running_claims, Some(&parent_authority)).digest_fields();
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
            running: view.running_claims,
            running_parent_authority: Some(&parent_authority),
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
        "F' R1CS accepted a mutated running parent-authority CE field after the full-running \
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

        // Rebuild the public output bits from the forged post-state.
        // This bypasses the ordinary "x_out bits don't match the
        // supplied digest" failure mode and isolates the producer-side
        // equation that must bind acc_digest_out to this step's NIFS.V
        // output accumulator.
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

    for anchor in ["vk_fs_digest", "structure_digest", "z_0", "z_i_in", "public_trace_in"] {
        let b = run_recursive_check_with_semantic(
            &view,
            |f_state, _, acc_digest_out, semantic_state_digest_out, source, _| {
                match anchor {
                    "vk_fs_digest" => f_state.vk_fs_digest[0] += F::ONE,
                    "structure_digest" => f_state.structure_digest[0] += F::ONE,
                    "z_0" => f_state.z_0[0] += F::ONE,
                    "z_i_in" => f_state.z_i_in[0] += F::ONE,
                    "public_trace_in" => f_state.public_trace_in[0] += F::ONE,
                    _ => unreachable!(),
                }

                // Repair both visible x_out links using the tampered
                // anchor fields. For `structure_digest`, this is a no-op
                // for the optimized state_x_out preimage. The same is true
                // for `z_0` and `public_trace_in`; `z_i_in` is repaired
                // through the prior-x_out bits below. Those cases make the
                // test specifically exercise the NIFS transcript prefix and
                // local state rows instead of relying on the public x_out
                // bit links to fail first.
                let forged_prior_x_out = digest32_as_fields(state_x_out_digest_with_mode(
                    StateXOutDigestMode::Stateless,
                    digest_fields_as_digest32(f_state.vk_fs_digest),
                    &f_state.structure_digest,
                    f_state.chunk_count_in,
                    f_state.step_count_in,
                    digest_fields_as_digest32(f_state.z_0),
                    digest_fields_as_digest32(f_state.z_i_in),
                    f_state.pc,
                    digest_fields_as_digest32(f_state.semantic_state_digest_in),
                    digest_fields_as_digest32(f_state.acc_digest_in),
                    digest_fields_as_digest32(f_state.public_trace_in),
                ));
                overwrite_enc_inst_bits(&mut source.image, source.prior_x_out_bits, forged_prior_x_out);

                let forged_public_x_out = digest32_as_fields(state_x_out_digest_with_mode(
                    StateXOutDigestMode::Stateless,
                    digest_fields_as_digest32(f_state.vk_fs_digest),
                    &f_state.structure_digest,
                    view.state_out.chunk_count,
                    view.state_out.step_count,
                    view.state_out.z_0,
                    view.state_out.z_i,
                    view.state_out.pc,
                    digest_fields_as_digest32(*semantic_state_digest_out),
                    digest_fields_as_digest32(*acc_digest_out),
                    view.state_out.public_trace,
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
        // In stateless mode, `state_x_out` intentionally omits the
        // semantic-state lane and Construction 2 carries the accumulator
        // digest as the semantic state. Mutate only this side field while
        // leaving the NIFS output accumulator and public x_out bits honest.
        // The dedicated `semantic_state_digest_out == acc_digest_out` row
        // is the load-bearing rejection here.
        semantic_state_digest_out[0] += F::ONE;
    });
    assert!(
        !b.is_satisfied(),
        "F' R1CS accepted semantic_state_digest_out != acc_digest_out in stateless mode"
    );
}

// ── Full-chain replay hard gate ─────────────────────────────────────────────
//
// Pre-decider invariant: every recursive step in a real lifecycle chain
// must be independently replayable by `enforce_f_prime_recursive_step_circuit`.
// If this passes for chain length 6, the remaining decider work is
// packaging the same relation, not discovering another lifecycle/F' mismatch.

#[test]
fn lifecycle_all_recursive_steps_satisfy_f_prime_r1cs() {
    let chain = build_f_prime_honest_chain(6);

    let mut recursive_count = 0;
    let mut saw_empty_running = false;
    let mut saw_nonempty_running = false;
    for idx in 0..chain.snapshots.len() {
        if !chain.is_recursive(idx) {
            continue;
        }
        recursive_count += 1;

        let view = chain.recursive_step(idx);
        if view.running_claims.is_empty() {
            saw_empty_running = true;
        } else {
            saw_nonempty_running = true;
        }

        let b = run_recursive_check(&view, |_, _, _, _, _| ());
        assert!(
            b.is_satisfied(),
            "F' R1CS rejected step {idx} of a real lifecycle chain (first bad row: {:?})",
            b.first_unsatisfied_row()
        );
    }

    // chain[0] is base (NoFold); all later snapshots are Recursive in this build.
    assert_eq!(
        recursive_count, 5,
        "chain(6) should contain 5 FoldProof::Recursive steps (indices 1..5)"
    );
    // The base-to-recursive transition: step 1 still has empty running.
    assert!(
        saw_empty_running,
        "full-chain replay should cover the first recursive step with empty running"
    );
    // Steady-state recursive: step 2+ has non-empty running.
    assert!(
        saw_nonempty_running,
        "full-chain replay should cover recursive steps with non-empty running"
    );
}

#[test]
fn lifecycle_verify_uncompressed_rejects_terminal_latest_public_input_not_linked_to_state() {
    let r1cs = bit_carrier_r1cs();
    let chain = build_f_prime_honest_chain(1);
    let state = chain
        .snapshots
        .last()
        .expect("chain has at least one step")
        .state_out
        .clone();
    let expected_x_out = compute_x_out_native(&chain.prep, &state);
    let mut wrong_x_out = expected_x_out;
    wrong_x_out[0] += F::ONE;
    let wrong_latest = build_link_instance(&chain.prep, &r1cs, wrong_x_out);

    let mut state_with_bad_latest = state.clone();
    match &mut state_with_bad_latest.proof {
        ProofState::Active { latest, .. } => {
            assert_eq!(latest.instances.len(), 1, "fixture uses one latest instance");
            latest.instances[0] = wrong_latest;
        }
        ProofState::Initial => panic!("fixture must be active after two steps"),
    }

    let audit = UncompressedAudit {
        proof: Uncompressed {
            state: state_with_bad_latest,
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
    let finished = neo_fold_clean::finish_uncompressed(&chain.prep, audit)
        .expect("wrong latest is still terminal-fold provable for the zero carrier CCS");

    let err = neo_fold_clean::verify_uncompressed(&chain.prep, &finished)
        .expect_err("verify_uncompressed accepted a terminal latest not linked to pre-final x_out");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::TerminalLatestPublicInputMismatch { index: 0 }
        ),
        "expected terminal latest link mismatch, got {err:?}"
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
        .expect_err("terminal-only verifier accepted a multi-chunk F' proof without replaying F' induction");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::FPrimeNonReplayUnsupported { chunk_count: 2 }
        ),
        "expected FPrimeNonReplayUnsupported(2), got {err:?}"
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
fn audit_replay_rejects_intermediate_latest_public_input_not_linked_to_state() {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let placeholder_z = vec![F::ZERO; prep.structure().m];
    let dummy_inst = || direct_ccs::build_instance(&prep, &r1cs, &placeholder_z).expect("dummy");

    let state0 = base_state(&prep);
    let predicted0 = peek_next_state(&prep, &state0, &[dummy_inst()]);
    let mut wrong_step0_x = compute_x_out_native(&prep, &predicted0);
    wrong_step0_x[0] += F::ONE;
    let wrong_step0_latest = build_link_instance(&prep, &r1cs, wrong_step0_x);
    let wrong_step0_claim = wrong_step0_latest.claim.clone();

    // Prove step 0 with a deliberately wrong public input. Because the
    // F' chunk digest is shape-only, this still advances to the same
    // verifier state as an honest step; the wrong claim becomes the
    // pending `latest` that step 1 will fold.
    let (state1, step0_proof) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state0,
        vec![wrong_step0_latest],
    )
    .expect("malicious step 0 is internally provable");

    // Now prove step 1 honestly over the malformed pending latest. The
    // NIFS proof is self-consistent with the bad claim, so a replay that
    // only checks the final trailing latest would accept this history.
    let predicted1 = peek_next_state(&prep, &state1, &[dummy_inst()]);
    let step1_x = compute_x_out_native(&prep, &predicted1);
    let step1_latest = build_link_instance(&prep, &r1cs, step1_x);
    let step1_claim = step1_latest.claim.clone();
    let (state2, step1_proof) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        state1,
        vec![step1_latest],
    )
    .expect("step 1 folds the malformed latest with a matching proof");

    let audit = UncompressedAudit {
        proof: Uncompressed {
            state: state2,
            final_fold: None,
        },
        steps: vec![step0_proof, step1_proof],
        public_batches: vec![vec![wrong_step0_claim], vec![step1_claim]],
    };
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit)
        .expect("malformed intermediate latest is terminal-fold provable");

    let err = neo_fold_clean::verify_uncompressed_audit(&prep, &finished)
        .expect_err("audit replay must reject an intermediate latest not linked to prior x_out");
    assert!(
        matches!(
            err,
            neo_fold_clean::Error::Decider(
                neo_fold_clean::paper::decider::Error::TerminalLatestPublicInputMismatch { index: 0 }
            )
        ),
        "expected intermediate latest link mismatch, got {err:?}"
    );
}

// ── Fixed-point-gone regression ─────────────────────────────────────────────

#[test]
fn f_prime_chunk_public_digest_is_independent_of_recursive_link_x() {
    // Build two real CCS instances with distinct low-norm assignments.
    // In direct-CCS the Ajtai commitment covers the full z, so both
    // `claim.x` and `claim.c.data` differ between the two — yet
    // `f_prime_chunk_public_digest` must yield the same value, because
    // it absorbs only the commitment **shape** (`d`, `kappa`) and `m_in`.
    // That insensitivity is what breaks the recursive-link fixed point.
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

    // F' chunk digest must collapse the two claims to the same value...
    let f_prime_a = f_prime_chunk_public_digest(0, &[inst_a.claim.clone()]);
    let f_prime_b = f_prime_chunk_public_digest(0, &[inst_b.claim.clone()]);
    assert_eq!(
        f_prime_a, f_prime_b,
        "f_prime_chunk_public_digest must ignore both claim.x and claim.c.data — \
         otherwise the recursive-link fixed point reappears"
    );

    // ...while ordinary CCS-identity digests still distinguish them.
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

// ── Layered Fiat–Shamir invariant ───────────────────────────────────────────
//
// Security claim: F'-chunk-digest is shape-only and Π_CCS / NIFS is
// content-bound. The two together are sound iff no challenge that checks
// chunk *content* is sampled before NIFS / Π_CCS has absorbed the full
// claim. This test pins the separation:
//
//   1. `f_prime_chunk_public_digest` collapses two same-shape distinct-
//       content batches (F' transcript prefix is intentionally shape-only).
//   2. `pi_ccs_instance_digest` distinguishes them (NIFS transcript
//       absorbs full `ccs_claim_digest`, which binds `claim.x` and
//       `claim.c.data`).
//   3. A real lifecycle proof generated for batch A, then verified with
//       A's deposited claim replaced wholesale by B's (same shape,
//       different content), rejects at the next step's NIFS.V — exactly
//       because that step's transcript absorbs the actual folded claim.

#[test]
fn nifs_transcript_binds_chunk_contents_even_though_f_prime_digest_is_shape_only() {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    // Two same-shape distinct-content low-norm instances.
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

    // (1) F' chunk digest is intentionally shape-only.
    assert_eq!(
        f_prime_chunk_public_digest(0, &[inst_a.claim.clone()]),
        f_prime_chunk_public_digest(0, &[inst_b.claim.clone()]),
        "F' chunk digest must collapse same-shape claims (regression for the recursive-link fixed point)"
    );

    // (2) Π_CCS instance digest binds full claim contents — the NIFS
    //     transcript absorbs this before any content-checking Fiat-Shamir
    //     challenge, so A vs B yield different challenges in the fold.
    let empty_running: Vec<CeClaim> = Vec::new();
    assert_ne!(
        pi_ccs_instance_digest(&[inst_a.claim.clone()], &empty_running),
        pi_ccs_instance_digest(&[inst_b.claim.clone()], &empty_running),
        "Π_CCS instance digest must distinguish A vs B; otherwise NIFS FS challenges would not depend on the folded claim's contents"
    );

    // (3) End-to-end: build a real proof folding A at step 1, then swap
    //     A's claim in the audit trail (`statement.witness.public_batches[0]`)
    //     for B's claim. At step 1, the **chain-replay verifier**
    //     (`paper::decider::validate_witness`) absorbs B's claim into Π_CCS
    //     before sampling NIFS challenges; the proof was generated for A, so
    //     the algebraic checks fail. (After Phase 1.7 `verify_uncompressed`
    //     is non-replay — it authenticates only the final running
    //     accumulator — so audit-trail tampers like this one are caught by
    //     `validate_witness`, the gatekeeper for the Spartan SNARK
    //     statement.)
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

    // Sanity: the original proof verifies under audit replay. The
    // terminal-only projection must fail closed for multi-chunk F'
    // histories because it has dropped the intermediate F' witnesses
    // and NIFS.V messages that HyperNova's induction needs.
    assert!(
        matches!(
            neo_fold_clean::verify_uncompressed(&chain.prep, &finished.proof),
            Err(neo_fold_clean::Error::FPrimeNonReplayUnsupported { chunk_count: 2 })
        ),
        "terminal-only verifier must reject multi-chunk F' history"
    );
    neo_fold_clean::verify_uncompressed_audit(&chain.prep, &finished).expect("untampered audit proof verifies");
    let untampered_statement = neo_fold_clean::build_decider_statement(&chain.prep, &finished);
    neo_fold_clean::paper::decider::validate_witness(
        &chain.prep.params,
        chain.prep.structure(),
        chain.prep.optimized_cache(),
        chain.prep.structure_digest(),
        &chain.prep.log,
        chain.prep.mix_rhos_commits,
        chain.prep.combine_b_pows,
        &chain.prep.vk,
        chain.prep.public_input_len,
        chain.prep.semantic_state_mode(),
        chain.prep.initial_semantic_state_digest(),
        &untampered_statement,
    )
    .expect("untampered statement passes validate_witness");

    // Whole-claim swap (same shape, different content) on the decider
    // statement's witness side. The chain replay catches it.
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
            chain.prep.mix_rhos_commits,
            chain.prep.combine_b_pows,
            &chain.prep.vk,
            chain.prep.public_input_len,
            chain.prep.semantic_state_mode(),
            chain.prep.initial_semantic_state_digest(),
            &tampered_statement,
        )
        .is_err(),
        "validate_witness accepted a statement whose folded batch's full claim was swapped for a \
         same-shape distinct-content one — NIFS / Π_CCS transcript would not be binding chunk contents \
         before content-checking challenges"
    );
}
