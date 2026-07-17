//! F' R1CS step composition — wiring + strict-mode + real-proof tests.
//!
//! Two entry points are tested:
//!   - `enforce_f_prime_base_step_circuit` (i = 0; no NIFS.V).
//!   - `enforce_f_prime_recursive_step_circuit` (i ≥ 1; runs NIFS.V).
//!
//! ## Scope
//!
//! - Base-step satisfaction with honest state (no NIFS.V).
//! - Base-step rejection of `chunk_count_in != 0` and `z_i_in != z_0`.
//! - Recursive-step verifier gate: a real native `nifs::prove` proof,
//!   paired with an F' state-in whose `prior_x_out` is bit-encoded into the
//!   fresh CCS instance's public input and an `acc_digest_in` that matches
//!   `digest(running)`, must satisfy the F' R1CS verifier shell.
//! - Recursive-step batch shape-rejection: empty fresh batch (`fresh.is_empty()`),
//!   `fresh[i].m_in != F_PRIME_PUBLIC_INPUT_LEN`, and `chunk_count_in == 0`.
//! - Recursive-step batched-link: a K=3 fresh batch with every public
//!   input rooted at one shared prior `x_out` must satisfy; an
//!   independently-proved fresh whose public input encodes a *different*
//!   `enc_inst` than the source-image `prior_x_out_bits` must be rejected
//!   by the per-fresh recursive-link gate (not just by NIFS.V).
//!
//! ## Caveat (Phase 6g' follow-up)
//!
//! This test only fixes the **public** recursive link and proves the
//! verifier shell accepts a real native NIFS proof. A full `u_i = F'`
//! instance chain also needs the F' R1CS itself to be emitted as a CCS
//! relation and its private witness encoded low-norm; the raw
//! `R1csBuilder` witness contains Poseidon2 outputs, challenges, and
//! sumcheck values that do not fit `b = 2`.

#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest, state_x_out_digest_with_mode,
    AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_public_input, enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit,
    FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig,
    F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const TRANSCRIPT_LABEL: &[u8] = b"neo.test.f_prime/step/v1";

// ── Fixture ──────────────────────────────────────────────────────────────

/// Real two-step `nifs::prove` run wired for F' consumption:
///   - The carrier R1CS has `m_in = F_PRIME_PUBLIC_INPUT_LEN` so the fresh
///     CCS instance can hold the bit-encoded `prior_x_out`.
///   - `state` is built so `acc_digest_in == digest(first_running)` and
///     `bit_encode(state_x_out(state)) == second_fresh.x`.
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

/// All-zero R1CS with `m_in = F_PRIME_PUBLIC_INPUT_LEN`. The constraint
/// `0·z * 0·z = 0·z` is trivially satisfied by any assignment, so we can
/// hand the fresh CCS instance any low-norm public input we want.
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
}

/// Native counterpart of [`enforce_state_x_out_digest_circuit`]. Same
/// absorb sequence as [`state_x_out_digest`], driven from the F' state-in
/// fields.
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
    build_fixture_with_k_fresh(1)
}

/// Construct a fixture whose recursive-step NIFS proof folds K=3 fresh
/// CCS instances **of which `divergent_index` encodes a different
/// `enc_inst` than the honest `prior_x_out`**. The proof is otherwise
/// honest, so NIFS.V on its own accepts (sumcheck/header/etc. are valid
/// for the actual fresh claims handed in). The F' R1CS recursive-link
/// gate is what catches the mismatch — the per-fresh loop must equate
/// every `fresh[i].x[1..]` to the *shared* `source_image[prior_x_out_bits]`,
/// which still encode `enc_inst(honest_prior_x_out)`.
fn build_fixture_with_divergent_fresh(divergent_index: usize) -> Fixture {
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
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let acc_digest_in = running_acc_digest(&running);
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
    let mut honest_z = encode_f_prime_public_input(prior_x_out);
    honest_z.resize(prep.structure().m, F::ZERO);

    // `divergent_x_out` differs from `prior_x_out` in at least one lane —
    // so its `enc_inst` body differs from the honest source-image bits.
    let mut divergent_x_out = prior_x_out;
    divergent_x_out[0] += F::ONE;
    let mut divergent_z = encode_f_prime_public_input(divergent_x_out);
    divergent_z.resize(prep.structure().m, F::ZERO);

    let k_fresh = 3;
    assert!(divergent_index < k_fresh);
    let fresh_instances: Vec<_> = (0..k_fresh)
        .map(|i| {
            let z = if i == divergent_index { &divergent_z } else { &honest_z };
            direct_ccs::build_instance(&prep, &r1cs, z).expect("fresh instance")
        })
        .collect();
    let fresh_claims: Vec<_> = fresh_instances.iter().map(|i| i.claim.clone()).collect();

    let chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &fresh_claims);
    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    append_f_prime_step_context(&mut tr, &state, chunk_digest);
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh_instances,
        &running,
    )
    .expect("divergent NIFS.P");

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

/// Construct a fixture whose recursive-step NIFS proof folds `k_fresh`
/// fresh CCS instances into the running accumulator. All `k_fresh`
/// fresh instances encode the **same** `enc_inst(prior_x_out)` — the
/// "one chunk rooted at a single prior Construction-2 state"
/// semantics F' R1CS expects.
fn build_fixture_with_k_fresh(k_fresh: usize) -> Fixture {
    build_fixture_with_k_fresh_and_public_one(k_fresh, F::ONE)
}

fn build_fixture_with_k_fresh_and_public_one(k_fresh: usize, public_one: F) -> Fixture {
    assert!(k_fresh >= 1, "build_fixture_with_k_fresh: k_fresh must be \u{2265} 1");
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    // First fold: seed the running accumulator. Any low-norm assignment
    // works; all zeros is the simplest.
    let zero_assignment = vec![F::ZERO; prep.structure().m];
    let first = direct_ccs::build_instance(&prep, &r1cs, &zero_assignment).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _first_proof) = neo_fold_clean::paper::nifs::prove(
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

    // Pin F' state-in so the recursive link is honest:
    //   acc_digest_in = digest(strict Pi_DEC parent authority)
    //   fresh[i].x   = enc_inst(state_x_out(state))   for every i in 0..k_fresh
    let acc_digest_in = running_acc_digest(&running);
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
    assert_eq!(z.len(), F_PRIME_PUBLIC_INPUT_LEN);
    z[0] = public_one;
    // Carrier R1CS has m = m_in, so the assignment is exactly
    // [public_one, bits…].
    assert_eq!(prep.structure().m, F_PRIME_PUBLIC_INPUT_LEN);
    // Defensive: pad/truncate to structure.m in case those ever diverge.
    z.resize(prep.structure().m, F::ZERO);

    let fresh_instances: Vec<_> = (0..k_fresh)
        .map(|_| direct_ccs::build_instance(&prep, &r1cs, &z).expect("fresh instance"))
        .collect();
    let fresh_claims: Vec<_> = fresh_instances.iter().map(|i| i.claim.clone()).collect();

    // Mirror F' R1CS's transcript exactly: same init label, same pre-NIFS
    // state absorbs, in the same order. Without this, in-circuit and
    // native challenges diverge and the NIFS.V verifier's algebraic
    // checks (sumcheck challenges, ρ, β_m, …) fail at the satisfaction
    // boundary.
    let chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &fresh_claims);
    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    append_f_prime_step_context(&mut tr, &state, chunk_digest);
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh_instances,
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

fn rebuild_recursive_fixture_for_state(mut fixture: Fixture, state: FPrimeStateIn) -> Fixture {
    let prior_x_out = native_prior_x_out(&state);
    let mut z = encode_f_prime_public_input(prior_x_out);
    z.resize(fixture.prep.structure().m, F::ZERO);

    let fresh_instances: Vec<_> = (0..fixture.fresh_claims.len())
        .map(|_| direct_ccs::build_instance(&fixture.prep, &bit_carrier_r1cs(), &z).expect("fresh instance"))
        .collect();
    let fresh_claims: Vec<_> = fresh_instances.iter().map(|i| i.claim.clone()).collect();

    let mut tr = Transcript::with_label(TRANSCRIPT_LABEL);
    append_f_prime_step_context(&mut tr, &state, fixture.chunk_digest);
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut tr,
        &fixture.prep.params,
        fixture.prep.structure(),
        fixture.prep.optimized_cache(),
        &fixture.prep.log,
        None,
        fixture.prep.mix_rhos_commits(),
        fixture.prep.combine_b_pows(),
        fresh_instances,
        &fixture.running,
    )
    .expect("rebuilt NIFS.P");

    fixture.fresh_claims = fresh_claims;
    fixture.proof = proof;
    fixture.combined = fixture.proof.pi_rlc.combined.clone();
    fixture.children = next_running.claims;
    fixture.state = state;
    fixture
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
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
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

fn running_acc_digest(running: &RunningInstance) -> [F; 4] {
    AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest_fields()
}

fn base_state(b: u32, z_0: [F; 4]) -> FPrimeStateIn {
    let _ = b;
    let empty_acc = AccumulatorHandle::empty().digest_fields();
    FPrimeStateIn {
        vk_fs_digest: rand_digest(0x10),
        pi_ccs_header_bundle: rand_digest(0x20),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0,
        z_i_in: z_0,
        pc: 1,
        acc_digest_in: empty_acc,
        semantic_state_digest_in: empty_acc,
        public_trace_in: rand_digest(0x40),
        nebula: None,
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

/// Native counterpart of [`build_x_out`] inside the F' R1CS, given the
/// state-in fields and the *post-step* `(acc, chunk_count, step_count)`.
/// Returns the four raw Goldilocks digest lanes (pre-`enc_inst`).
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

/// Raw `x_out` (`[F; 4]`) for the **base** step: post-state is `(empty_acc,
/// 1, rows_in_chunk)`. Callers wrap this in an `FPrimeSourceImage` via
/// `push_enc_inst` to get the bit-encoded `BitRange`.
fn base_step_x_out(b: u32, state: &FPrimeStateIn, chunk_digest: [F; 4], rows_in_chunk: u64) -> [F; 4] {
    let _ = b;
    let empty_acc = AccumulatorHandle::empty().digest_fields();
    native_x_out(state, chunk_digest, empty_acc, empty_acc, 1, rows_in_chunk)
}

/// Raw `x_out` (`[F; 4]`) for the **recursive** step: post-state's
/// `acc_digest` is `digest(children)`, `chunk_count' = chunk_count + 1`,
/// `step_count' = step_count + |fresh|`.
fn recursive_step_x_out(state: &FPrimeStateIn, chunk_digest: [F; 4], new_acc: [F; 4], fresh_count: u64) -> [F; 4] {
    native_x_out(
        state,
        chunk_digest,
        new_acc,
        new_acc,
        state.chunk_count_in + 1,
        state.step_count_in + fresh_count,
    )
}

fn repeated_shape_chunk_digest(start_index: u64, fresh_len: u64, template: &CcsClaim) -> [F; 4] {
    let claims = vec![template.clone(); fresh_len as usize];
    f_prime_chunk_public_digest(start_index, &claims)
}

// ── Base-step tests ──────────────────────────────────────────────────────

#[test]
fn f_prime_base_step_emits_and_satisfies() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let state = base_state(cfg.b, z_0);
    let rows_in_chunk = 3;
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, rows_in_chunk, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, rows_in_chunk);
    let source = base_source_image(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let rows_before = b.rows();
    let out = enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    let rows_added = b.rows() - rows_before;
    assert!(
        rows_added < 25_000,
        "F' base step should be lightweight (no NIFS.V); got {rows_added}"
    );
    assert!(
        rows_added > 1_000,
        "F' base step should still emit non-trivial constraints"
    );
    assert_eq!(out.x_out.len(), 4);
    assert_eq!(out.x_out_bits.len(), F_PRIME_ENC_INST_BITS);
    assert!(
        b.is_satisfied(),
        "F' base step must accept honest base witness (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let unconstrained = b.unconstrained_columns();
    let mut allowed = Vec::new();
    allowed.extend(out.state_in.semantic_state_digest.iter().map(|v| v.col()));
    allowed.extend(out.state_in.public_trace.iter().map(|v| v.col()));
    allowed.sort_unstable();
    assert!(
        unconstrained == allowed,
        "base F' step left unexpected unconstrained columns: got {unconstrained:?}, \
         expected only decider-owned base seed pins {allowed:?}"
    );
}

#[test]
fn f_prime_base_step_rejects_zero_rows_in_chunk() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let state = base_state(cfg.b, z_0);
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 0, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 0);
    let source = base_source_image(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 0,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let result = enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs);
    assert!(
        result.is_err(),
        "base F' R1CS must reject a zero-row first chunk; lifecycle forbids empty batches, \
         and the audit circuit should fail closed on the same shape"
    );
}

#[test]
fn f_prime_base_step_rejects_nonzero_chunk_count_in() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let mut state = base_state(cfg.b, z_0);
    state.chunk_count_in = 1;
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let source = base_source_image(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(!b.is_satisfied(), "base step must reject chunk_count_in != 0");
}

#[test]
fn f_prime_base_rejects_noncanonical_zero_chunk_count_source_word() {
    // `p = 0xFFFF_FFFF_0000_0001` decodes to zero in Goldilocks. Without
    // the source-word canonicality row, the `chunk_count_in == 0` equality
    // would accept this noncanonical low-norm source image.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let state = base_state(cfg.b, z_0);
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let mut source = base_source_image(&state, expected_x_out);
    let start = source.chunk_count_in_word.bits().start();
    let noncanonical: u64 = 0xFFFF_FFFF_0000_0001;
    for i in 0..64 {
        source
            .image
            .set_bit(start + i, ((noncanonical >> i) & 1) == 1);
    }
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "base source-image chunk_count word must be canonical Goldilocks (< p)"
    );
}

#[test]
fn f_prime_base_step_rejects_source_image_step_count_mismatch() {
    // Base F' also binds `step_count_in` to the source image before
    // forcing the state field to zero. Mutate only the source-image word:
    // the base state remains honest, so the source-word equality is the
    // row that must catch this.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let state = base_state(cfg.b, z_0);
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let mut source = base_source_image(&state, expected_x_out);
    let idx = source.step_count_in_word.bits().start();
    source
        .image
        .set_bit(idx, source.image.values()[idx] == F::ZERO);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "base source-image step_count word must match the in-circuit state var"
    );
}

#[test]
fn f_prime_base_step_rejects_tampered_public_x_out_bits() {
    // HyperNova's F' output is the hash `x_out`. The base step computes it
    // in-circuit and must pin the source-image/public `enc_inst(x_out)`
    // bits to that digest. Mutate only the carried output bits while
    // leaving the base state and chunk digest honest.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let state = base_state(cfg.b, z_0);
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let mut source = base_source_image(&state, expected_x_out);
    let idx = source.public_x_out_bits.start();
    source
        .image
        .set_bit(idx, source.image.values()[idx] == F::ZERO);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "base F' accepted tampered enc_inst(x_out) output bits"
    );
}

#[test]
fn f_prime_base_step_rejects_noncanonical_source_image_pc_word() {
    // The single-program build fixes pc = TRIVIAL_PC = 1. Encode `p + 1`
    // in the source-image pc word: as a field element it equals 1, so the
    // pc equality alone would accept. The canonical Goldilocks word row
    // must reject it.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let state = base_state(cfg.b, z_0);
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let mut source = base_source_image(&state, expected_x_out);
    let start = source.pc_word.bits().start();
    let noncanonical: u64 = 0xFFFF_FFFF_0000_0002;
    for i in 0..64 {
        source
            .image
            .set_bit(start + i, ((noncanonical >> i) & 1) == 1);
    }
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "base source-image pc word must be canonical Goldilocks (< p)"
    );
}

#[test]
fn f_prime_base_step_rejects_nonzero_step_count_in() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let mut state = base_state(cfg.b, z_0);
    state.step_count_in = 1;
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let source = base_source_image(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(!b.is_satisfied(), "base step must reject step_count_in != 0");
}

#[test]
fn f_prime_base_step_rejects_nonempty_acc_digest_in() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let mut state = base_state(cfg.b, z_0);
    state.acc_digest_in[0] += F::ONE;
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let source = base_source_image(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "base step must reject an incoming accumulator digest other than u_perp"
    );
}

#[test]
fn f_prime_base_step_rejects_z_i_neq_z_0() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let z_0 = rand_digest(0x100);
    let mut state = base_state(cfg.b, z_0);
    state.z_i_in = rand_digest(0x101);
    let chunk_digest = repeated_shape_chunk_digest(state.step_count_in, 3, &fixture.fresh_claims[0]);
    let expected_x_out = base_step_x_out(cfg.b, &state, chunk_digest, 3);
    let source = base_source_image(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: state.semantic_state_digest_in,
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_base_step_circuit(&mut b, &cfg, &inputs).expect("emit");
    assert!(!b.is_satisfied(), "base step must reject z_i_in != z_0");
}

// ── Recursive-step tests ─────────────────────────────────────────────────

fn recursive_x_out(fixture: &Fixture) -> [F; 4] {
    recursive_step_x_out(
        &fixture.state,
        fixture.chunk_digest,
        recursive_acc_digest(fixture),
        fixture.fresh_claims.len() as u64,
    )
}

fn recursive_acc_digest(fixture: &Fixture) -> [F; 4] {
    AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields()
}

/// Bundle of the source image + every handle a base/recursive F' input
/// needs. The image owns the buffer and the handles index into it; both
/// live alongside the inputs that borrow the image.
struct BaseSourceFixture {
    image: FPrimeSourceImage,
    chunk_count_in_word: Word64Image,
    step_count_in_word: Word64Image,
    pc_word: Word64Image,
    public_x_out_bits: BitRange,
}

struct RecursiveSourceFixture {
    image: FPrimeSourceImage,
    chunk_count_in_word: Word64Image,
    step_count_in_word: Word64Image,
    pc_word: Word64Image,
    prior_x_out_bits: BitRange,
    public_x_out_bits: BitRange,
}

/// Build a base-step source image: counters first, then output enc_inst.
fn base_source_image(state: &FPrimeStateIn, x_out: [F; 4]) -> BaseSourceFixture {
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(state.step_count_in);
    let pc_word = image.push_u64_le(state.pc);
    let public_x_out_bits = image.push_enc_inst(x_out);
    BaseSourceFixture {
        image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    }
}

/// Build a recursive-step source image: counters, then prior public
/// input (with the leading constant-one), then output enc_inst.
fn recursive_source_image(fixture: &Fixture) -> RecursiveSourceFixture {
    recursive_source_image_with_public_x_out(fixture, recursive_x_out(fixture))
}

fn recursive_source_image_with_public_x_out(fixture: &Fixture, public_x_out: [F; 4]) -> RecursiveSourceFixture {
    recursive_source_image_for_state(&fixture.state, public_x_out)
}

fn recursive_source_image_for_state(state: &FPrimeStateIn, public_x_out: [F; 4]) -> RecursiveSourceFixture {
    let mut image = FPrimeSourceImage::new();
    let chunk_count_in_word = image.push_u64_le(state.chunk_count_in);
    let step_count_in_word = image.push_u64_le(state.step_count_in);
    let pc_word = image.push_u64_le(state.pc);
    let prior_public = image.push_f_prime_public_input(native_prior_x_out(state));
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = image.push_enc_inst(public_x_out);
    RecursiveSourceFixture {
        image,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    }
}

/// **Hard gate**: a real native `nifs::prove` proof, with both halves of
/// the recursive link wired honestly:
///   - fresh.x        == enc_inst(prior_x_out)         (input link)
///   - public_x_out   == enc_inst(this step's x_out)   (output link)
/// must satisfy the F' R1CS end-to-end.
#[test]
fn f_prime_recursive_step_accepts_real_native_nifs_proof() {
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

    let mut b = R1csBuilder::new();
    let rows_before = b.rows();
    let out = enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    let rows_added = b.rows() - rows_before;
    assert!(
        rows_added > 100_000,
        "F' recursive step should emit >100k rows; got {rows_added}"
    );
    assert_eq!(out.x_out.len(), 4);
    assert_eq!(out.x_out_bits.len(), F_PRIME_ENC_INST_BITS);
    assert!(
        b.is_satisfied(),
        "real recursive F' step must satisfy (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let unconstrained = b.unconstrained_columns();
    let mut allowed = Vec::new();
    if let Some(running) = &out.nifs_running {
        allowed.extend(running.iter().flat_map(|claim| {
            claim
                .y_zcol
                .iter()
                .take(D)
                .flat_map(|v| [v.c0.col(), v.c1.col()])
        }));
    }
    if let Some(parent) = &out.nifs_running_parent_authority {
        allowed.extend(parent.y_zcol.iter().flat_map(|v| [v.c0.col(), v.c1.col()]));
    }
    if let Some(children) = &out.nifs_children {
        allowed.extend(
            children
                .iter()
                .flat_map(|child| child.y_zcol.iter().map(|v| v.col())),
        );
    }
    allowed.sort_unstable();
    assert!(
        unconstrained == allowed,
        "recursive F' step left unexpected unconstrained columns: got {unconstrained:?}, \
         expected only currently unbound y_zcol sidecar limbs {allowed:?}"
    );
}

#[test]
fn f_prime_recursive_rejects_chunk_counter_field_modulus_wrap() {
    let fixture = build_fixture();
    let mut state = fixture.state.clone();
    state.chunk_count_in = F::ORDER_U64 - 1;
    state.step_count_in = 1;
    let fixture = rebuild_recursive_fixture_for_state(fixture, state);
    let cfg = make_step_config(&fixture.prep);

    let acc = recursive_acc_digest(&fixture);
    let wrapped_x_out = native_x_out(
        &fixture.state,
        fixture.chunk_digest,
        acc,
        acc,
        0,
        fixture.state.step_count_in + 1,
    );
    let source = recursive_source_image_with_public_x_out(&fixture, wrapped_x_out);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: acc,
        acc_digest_out: acc,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' must reject chunk_count transition p - 1 -> 0; Construction 2 counters are integers, not field elements"
    );
}

#[test]
fn f_prime_recursive_rejects_step_counter_field_modulus_wrap() {
    let fixture = build_fixture();
    let mut state = fixture.state.clone();
    state.chunk_count_in = 1;
    state.step_count_in = F::ORDER_U64 - 1;
    let fixture = rebuild_recursive_fixture_for_state(fixture, state);
    let cfg = make_step_config(&fixture.prep);

    let acc = recursive_acc_digest(&fixture);
    let wrapped_x_out = native_x_out(
        &fixture.state,
        fixture.chunk_digest,
        acc,
        acc,
        fixture.state.chunk_count_in + 1,
        0,
    );
    let source = recursive_source_image_with_public_x_out(&fixture, wrapped_x_out);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: acc,
        acc_digest_out: acc,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' must reject step_count transition p - 1 -> 0; Construction 2 counters are integers, not field elements"
    );
}

#[test]
fn f_prime_recursive_rejects_chunk_count_in_zero() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut state = fixture.state.clone();
    state.chunk_count_in = 0;
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        state,
        chunk_digest: fixture.chunk_digest,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let result = enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs);
    assert!(result.is_err(), "recursive must reject chunk_count_in == 0");
}

#[test]
fn f_prime_recursive_rejects_empty_fresh_batch() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let empty: Vec<CcsClaim> = Vec::new();
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &empty,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
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
    let result = enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs);
    assert!(
        result.is_err(),
        "recursive must reject empty fresh batch (SuperNeo K \u{2265} 1)"
    );
}

/// Multi-fresh chunk: a real NIFS proof with K=3 fresh CCS instances
/// (the whole batch rooted at one prior Construction-2 state) must
/// satisfy the F' R1CS recursive verifier — proving the in-circuit
/// public-link gate runs over **every** fresh `u_i`, not just `fresh[0]`.
#[test]
fn recursive_step_accepts_multi_fresh_batch() {
    let fixture = build_fixture_with_k_fresh(3);
    assert_eq!(fixture.fresh_claims.len(), 3);
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        b.is_satisfied(),
        "multi-fresh recursive F' step must satisfy when every fresh.x is linked to the shared prior x_out (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

/// Negative companion to [`recursive_step_accepts_multi_fresh_batch`]:
/// the K=3 NIFS proof is generated honestly over a batch where
/// `fresh[2].x` encodes a **different** `enc_inst` than the honest prior
/// `x_out`. NIFS.V on its own accepts (the proof matches the actual
/// fresh handed in). The F' R1CS recursive-link gate must still reject,
/// because `source_image[prior_x_out_bits]` is the *honest* enc_inst and
/// only the per-fresh loop catches the divergent `fresh[2].x[1..]`.
///
/// This isolates the per-fresh recursive-link gate from NIFS.V's own
/// sumcheck/header binding: if the loop bailed after `fresh[0]`, this
/// test would (wrongly) pass.
#[test]
fn f_prime_recursive_rejects_multi_fresh_with_divergent_non_first_link() {
    let fixture = build_fixture_with_divergent_fresh(2);
    assert_eq!(fixture.fresh_claims.len(), 3);
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' R1CS must run the per-fresh recursive-link gate at every index — fresh[2].x diverged from \
         source_image[prior_x_out_bits] (honest), so the loop at i=2 must fail"
    );
}

#[test]
fn f_prime_recursive_rejects_fresh_m_in_mismatch() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    // Hand-craft a fresh claim with `m_in != F_PRIME_PUBLIC_INPUT_LEN`.
    let mut bad_fresh = fixture.fresh_claims.clone();
    bad_fresh[0].m_in = 4;
    bad_fresh[0].x = vec![F::ZERO; 4];
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &bad_fresh,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
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
    let result = enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs);
    assert!(
        result.is_err(),
        "recursive must reject fresh m_in != F_PRIME_PUBLIC_INPUT_LEN"
    );
}

// ── Recursive-step tamper tests ──────────────────────────────────────────
//
// Each tamper test flips ONE F'-side input field and asserts the circuit
// stops satisfying. NIFS.V-internal tampers (sumcheck round, header
// digest, combined.y_ring, child.s_col) are already covered by the
// L-gate in `tests/reductions/nifs_v.rs`.

#[test]
fn f_prime_recursive_rejects_tampered_public_x_out_bits() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    // Flip the first bit of the enc_inst body in the source image itself —
    // SourceImageWires::alloc will pick up the tampered value.
    let idx = source.public_x_out_bits.start();
    let original = source.image.values()[idx];
    source.image.set_bit(idx, original == F::ZERO);
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject tampered enc_inst(x_out) public output bits"
    );
}

#[test]
fn f_prime_recursive_rejects_tampered_acc_digest_in() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut state = fixture.state.clone();
    state.acc_digest_in[0] += F::ONE;
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        state,
        chunk_digest: fixture.chunk_digest,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject acc_digest_in that doesn't match digest(running)"
    );
}

#[test]
fn f_prime_recursive_rejects_tampered_chunk_digest() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut bad_chunk_digest = fixture.chunk_digest;
    bad_chunk_digest[0] += F::ONE;
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: bad_chunk_digest,
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject chunk_digest that diverges from native pre-NIFS transcript absorb"
    );
}

#[test]
fn f_prime_recursive_rejects_tampered_fresh_x_bit() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut bad_fresh = fixture.fresh_claims.clone();
    // Flip one enc_inst bit of fresh.x (the first body bit, index 1 —
    // index 0 is the CCS constant-one slot). enc_inst(prior_x_out) check
    // breaks immediately; sumcheck/header challenges diverge as well.
    let bit_idx = 1; // F_PRIME_ENC_INST_OFFSET
    let v = bad_fresh[0].x[bit_idx];
    bad_fresh[0].x[bit_idx] = if v == F::ZERO { F::ONE } else { F::ZERO };
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &bad_fresh,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
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
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject fresh.x that doesn't encode prior_x_out"
    );
}

#[test]
fn f_prime_recursive_rejects_fresh_public_one_slot_not_one_even_when_nifs_proof_matches() {
    // Build the NIFS proof honestly for a fresh public input whose body is
    // the correct enc_inst(prior_x_out), but whose CCS constant-one slot is
    // zero. NIFS.V itself is consistent with that fresh claim; the F'
    // recursive-link shell must reject via the explicit x[0] == 1 row.
    let fixture = build_fixture_with_k_fresh_and_public_one(1, F::ZERO);
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive accepted a fresh public input with x[0] != 1 even though the NIFS proof matched it"
    );
}

#[test]
fn f_prime_recursive_rejects_nonbinary_source_image_public_bit() {
    // SourceImageWires::alloc enforces bitness on every source-image
    // coordinate. Tamper one of the public-x_out source-image bits to a
    // non-{0,1} value and verify F' rejects it — independent of the
    // enc_inst(x_out) algebraic check.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    source
        .image
        .set_raw(source.public_x_out_bits.start(), F::from_u64(2));
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "non-binary source-image bit must be rejected by bitness constraint"
    );
}

// ── Step 3: input-link source-image tampers ──────────────────────────────

#[test]
fn f_prime_recursive_rejects_tampered_prior_source_image_bit() {
    // F' constrains `source_image[prior_x_out_bits] == enc_inst(prior_x_out)`.
    // Flipping one prior-image bit while leaving prior_x_out (computed
    // in-circuit from honest state-in) untouched must fail.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let idx = source.prior_x_out_bits.start();
    let original = source.image.values()[idx];
    source.image.set_bit(idx, original == F::ZERO);
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "tampered prior source-image bit must break recursive input link"
    );
}

#[test]
fn f_prime_recursive_rejects_fresh_x_not_matching_prior_source_image() {
    // F' wire-to-wire-equates `fresh[0].x[1..] == source_image[prior_x_out_bits]`.
    // Tampering fresh.x[1] inside the NIFS proof — but leaving the
    // source image honest — must fail at that equality (the NIFS algebraic
    // checks already reject it too, but this test exercises the
    // source-image binding specifically).
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let mut bad_fresh = fixture.fresh_claims.clone();
    let bit_idx = 1; // F_PRIME_ENC_INST_OFFSET
    let v = bad_fresh[0].x[bit_idx];
    bad_fresh[0].x[bit_idx] = if v == F::ZERO { F::ONE } else { F::ZERO };
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &bad_fresh,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
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
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(!b.is_satisfied(), "fresh.x must match source-image prior enc_inst bits");
}

// ── Step 4: counter source-image tampers ─────────────────────────────────

#[test]
fn f_prime_recursive_rejects_source_image_chunk_count_mismatch() {
    // F' constrains `sw.chunk_count_in == decode(source_image.chunk_count_in_word)`.
    // Flip one bit of the chunk_count word in the source image while the
    // `state.chunk_count_in` field stays honest — the wire-to-LC equality
    // must fail.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let idx = source.chunk_count_in_word.bits().start();
    source
        .image
        .set_bit(idx, source.image.values()[idx] == F::ZERO);
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "source-image chunk_count word must match in-circuit state var"
    );
}

#[test]
fn f_prime_recursive_rejects_source_image_step_count_mismatch() {
    // Same Step-4 source-image binding as chunk_count, but for
    // `step_count_in`. HyperNova's state hash absorbs the step counter;
    // a prover must not be able to route a stale or forged source-image
    // word while keeping the in-circuit state field honest.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let idx = source.step_count_in_word.bits().start();
    source
        .image
        .set_bit(idx, source.image.values()[idx] == F::ZERO);
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "source-image step_count word must match in-circuit state var"
    );
}

#[test]
fn f_prime_recursive_rejects_noncanonical_source_image_pc_word() {
    // Overwrite pc's source-image word with `p + 1`. This is noncanonical
    // but field-decodes to the honest single-program pc (`TRIVIAL_PC = 1`).
    // Without the Step 4 canonicality row, the pc equality row alone would
    // accept this source image.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let start = source.pc_word.bits().start();
    let noncanonical: u64 = 0xFFFF_FFFF_0000_0002;
    for i in 0..64 {
        source
            .image
            .set_bit(start + i, ((noncanonical >> i) & 1) == 1);
    }
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

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, &fixture.prep.params, &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "source-image pc word must be canonical Goldilocks (< p)"
    );
}
