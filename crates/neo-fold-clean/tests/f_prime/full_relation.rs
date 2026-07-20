//! Complete Construction-2 `F'` relation and connectivity tests.
//!
//! | Audit branch | Evidence |
//! |---|---|
//! | Semantic relation | Native/circuit acceptance and mutation rejection |
//! | Encoding | Oracle, derived, and gadget-native reconciliation |
//! | Cost tree | Exact parent/leaf rows, columns, and gate-family ownership |
//! | Source selector relation | Materialized base/recursive selector behavior and dimensions |
//! | Low-norm selector formulas | Explicit component arithmetic; no materializer or soundness claim |

use std::collections::BTreeMap;

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    audit_r1cs_gadget_native_canonical_u64, profile_r1cs_gadget_native_stages,
};
use neo_fold_clean::frontends::f_prime::low_norm_r1cs::{
    encode_r1cs_derived, encode_r1cs_oracle, estimate_r1cs_encoding, estimate_selector_gated_r1cs_encoding,
    LowNormR1csEncodingKind,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    semantic_state_digest_fields, FullFPrimeBranchExecution, FullFPrimeContext, FullFPrimeError, FullFPrimeRelation,
    FullFPrimeShape, R1csShape,
};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest,
    f_prime_chunk_public_digest_for_uniform_shape, state_x_out_digest_with_mode, AccumulatorHandle,
    StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_superneo_public_input, FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs,
    FPrimeStateIn, FPrimeStepConfig, F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
    F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::{prove_fixed, FixedNifsAccumulator};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

#[path = "full_relation/context.rs"]
mod context;
#[path = "full_relation/cost_tree.rs"]
mod cost_tree;
#[path = "full_relation/source_role_manifest.rs"]
mod source_role_manifest;
use cost_tree::{
    assert_direct_selector_cost_formula, assert_dominant_sis_snapshots, assert_f_prime_base_stage_hierarchy,
    assert_f_prime_recursive_stage_hierarchy, assert_fixed_selector_cost_formula,
    assert_pi_ccs_nc_terminal_row_families, assert_pi_ccs_stage_hierarchy, assert_pi_rlc_stage_hierarchy,
    assert_protocol_row_family_snapshots, print_stage_cost_families,
};

const TRANSCRIPT_LABEL: &[u8] = b"neo.test.full_f_prime/step/v1";

struct BaseSource {
    image: FPrimeSourceImage,
    chunk_count: Word64Image,
    step_count: Word64Image,
    pc: Word64Image,
    x_out: BitRange,
}

struct RecursiveSource {
    image: FPrimeSourceImage,
    chunk_count: Word64Image,
    step_count: Word64Image,
    pc: Word64Image,
    prior_x_out: BitRange,
    x_out: BitRange,
}

fn rand_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|lane| F::from_u64(seed.wrapping_mul(31).wrapping_add(lane as u64 + 1)))
}

fn bit_carrier_r1cs() -> R1cs {
    let padding = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN - F_PRIME_PUBLIC_INPUT_LEN;
    let mut a = Mat::zero(padding, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
    let mut b = Mat::zero(padding, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
    for row in 0..padding {
        a[(row, F_PRIME_PUBLIC_INPUT_LEN + row)] = F::ONE;
        b[(row, 0)] = F::ONE;
    }
    R1cs {
        a,
        b,
        c: Mat::zero(padding, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    }
}

fn distinct_bit_carrier_r1cs() -> R1cs {
    let mut carrier = bit_carrier_r1cs();
    carrier.a[(0, 0)] = F::ONE;
    carrier
}

fn fibonacci_step_r1cs() -> R1csShape {
    // z = [1, a, b, next_a, next_b]
    // b * 1 = next_a
    // (a + b) * 1 = next_b
    let mut a = Mat::zero(2, 5, F::ZERO);
    let mut b = Mat::zero(2, 5, F::ZERO);
    let mut c = Mat::zero(2, 5, F::ZERO);
    a[(0, 2)] = F::ONE;
    b[(0, 0)] = F::ONE;
    c[(0, 3)] = F::ONE;
    a[(1, 1)] = F::ONE;
    a[(1, 2)] = F::ONE;
    b[(1, 0)] = F::ONE;
    c[(1, 4)] = F::ONE;
    R1cs { a, b, c, m_in: 1 }.into()
}

fn split_nc_config(prep: &neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'_> {
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

fn step_config(prep: &neo_fold_clean::Preprocessing) -> FPrimeStepConfig<'_> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::plain(),
        nebula: None,
        state_x_out_digest_mode: StateXOutDigestMode::Stateful,
    }
}

fn full_context(prep: &neo_fold_clean::Preprocessing, initial: [F; 4]) -> FullFPrimeContext {
    FullFPrimeContext::derive(&prep.params, prep.structure(), &prep.log, initial).expect("full F' verifier key")
}

fn construction2_zero_digest(prep: &neo_fold_clean::Preprocessing) -> [F; 4] {
    let zero = RunningInstance::canonical_zero(&prep.params, prep.structure(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN)
        .expect("canonical fixed-k accumulator");
    AccumulatorHandle::from_running_parts(&zero.claims, zero.parent_authority.as_ref()).digest_fields()
}

fn uniform_chunk_digest(prep: &neo_fold_clean::Preprocessing, start_index: u64) -> [F; 4] {
    f_prime_chunk_public_digest_for_uniform_shape(
        start_index,
        1,
        D,
        prep.params.kappa() as usize,
        F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    )
}

fn native_x_out(
    state: &FPrimeStateIn,
    chunk_digest: [F; 4],
    semantic_state: [F; 4],
    construction2_acc: [F; 4],
    chunk_count: u64,
    step_count: u64,
) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateful,
        digest_fields_as_digest32(state.vk_fs_digest),
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        chunk_count,
        step_count,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(chunk_digest),
        state.pc,
        digest_fields_as_digest32(semantic_state),
        digest_fields_as_digest32(construction2_acc),
        digest_fields_as_digest32(chunk_digest),
        None,
    ))
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

fn base_source(state: &FPrimeStateIn, x_out: [F; 4]) -> BaseSource {
    let mut image = FPrimeSourceImage::new();
    let chunk_count = image.push_u64_le(state.chunk_count_in);
    let step_count = image.push_u64_le(state.step_count_in);
    let pc = image.push_u64_le(state.pc);
    let x_out = image.push_enc_inst(x_out);
    BaseSource {
        image,
        chunk_count,
        step_count,
        pc,
        x_out,
    }
}

fn recursive_source(state: &FPrimeStateIn, prior_x_out: [F; 4], x_out: [F; 4]) -> RecursiveSource {
    let mut image = FPrimeSourceImage::new();
    let chunk_count = image.push_u64_le(state.chunk_count_in);
    let step_count = image.push_u64_le(state.step_count_in);
    let pc = image.push_u64_le(state.pc);
    let prior_public = image.push_f_prime_public_input(prior_x_out);
    let prior_x_out = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let x_out = image.push_enc_inst(x_out);
    RecursiveSource {
        image,
        chunk_count,
        step_count,
        pc,
        prior_x_out,
        x_out,
    }
}

fn build_base_branch(
    prep: &neo_fold_clean::Preprocessing,
    application: &R1csShape,
    application_assignment: &[F],
) -> FullFPrimeBranchExecution {
    let semantic_in = semantic_state_digest_fields(&application_assignment[1..3]);
    let semantic_out = semantic_state_digest_fields(&application_assignment[3..5]);
    let context = full_context(prep, semantic_in);
    let relation = FullFPrimeRelation::new(context, step_config(prep), application, vec![1, 2], vec![3, 4])
        .expect("fixed full F' relation");
    let state = FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest(),
        pi_ccs_header_bundle: context.pi_ccs_header_bundle(),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0: semantic_in,
        z_i_in: semantic_in,
        pc: 1,
        semantic_state_digest_in: semantic_in,
        acc_digest_in: AccumulatorHandle::empty().digest_fields(),
        public_trace_in: semantic_in,
        nebula: None,
    };
    let chunk_digest = uniform_chunk_digest(prep, 0);
    let expected_x_out = native_x_out(
        &state,
        chunk_digest,
        semantic_out,
        construction2_zero_digest(prep),
        1,
        1,
    );
    let source = base_source(&state, expected_x_out);
    relation
        .build_base(
            &FPrimeBaseInputs {
                state,
                chunk_digest,
                semantic_state_digest_out: semantic_out,
                rows_in_chunk: 1,
                source_image: &source.image,
                chunk_count_in_word: source.chunk_count,
                step_count_in_word: source.step_count,
                pc_word: source.pc,
                public_x_out_bits: source.x_out,
            },
            application_assignment,
        )
        .expect("complete base F'")
}

#[test]
fn full_relation_shape_is_independent_of_matrix_dependent_verifier_key_values() {
    let first_carrier = bit_carrier_r1cs();
    let second_carrier = distinct_bit_carrier_r1cs();
    let first_prep = direct_ccs::preprocess_seeded(&first_carrier, 45).expect("first carrier preprocessing");
    let second_prep = direct_ccs::preprocess_seeded(&second_carrier, 45).expect("second carrier preprocessing");
    assert_eq!(first_prep.structure().n, second_prep.structure().n);
    assert_eq!(first_prep.structure().m, second_prep.structure().m);

    let application = fibonacci_step_r1cs();
    let assignment = [F::ONE, F::from_u64(3), F::from_u64(5), F::from_u64(5), F::from_u64(8)];
    let first = build_base_branch(&first_prep, &application, &assignment);
    let second = build_base_branch(&second_prep, &application, &assignment);
    assert!(first.is_satisfied());
    assert!(second.is_satisfied());
    assert!(
        first.snapshot().has_same_relation(second.snapshot()),
        "same-capacity verifier keys must change only witness values, never the F' matrix"
    );
    assert_ne!(
        first
            .verifier_key_columns()
            .iter()
            .map(|&column| first.snapshot().witness()[column])
            .collect::<Vec<_>>(),
        second
            .verifier_key_columns()
            .iter()
            .map(|&column| second.snapshot().witness()[column])
            .collect::<Vec<_>>(),
        "test carriers must actually produce different verifier-key data"
    );
}

#[test]
fn complete_base_relation_executes_then_encodes_and_rejects_disconnected_state() {
    let carrier = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&carrier, 42).expect("carrier preprocessing");
    let cfg = step_config(&prep);
    let application_relation = fibonacci_step_r1cs();
    let application_assignment = [F::ONE, F::from_u64(3), F::from_u64(5), F::from_u64(5), F::from_u64(8)];
    let semantic_in = semantic_state_digest_fields(&application_assignment[1..3]);
    let semantic_out = semantic_state_digest_fields(&application_assignment[3..5]);
    let context = full_context(&prep, semantic_in);
    let relation = FullFPrimeRelation::new(context, cfg, &application_relation, vec![1, 2], vec![3, 4])
        .expect("fixed full F' relation");
    let state = FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest(),
        pi_ccs_header_bundle: context.pi_ccs_header_bundle(),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0: semantic_in,
        z_i_in: semantic_in,
        pc: 1,
        semantic_state_digest_in: semantic_in,
        acc_digest_in: AccumulatorHandle::empty().digest_fields(),
        public_trace_in: semantic_in,
        nebula: None,
    };
    let zero_acc = construction2_zero_digest(&prep);
    let chunk_digest = uniform_chunk_digest(&prep, 0);
    let expected_x_out = native_x_out(&state, chunk_digest, semantic_out, zero_acc, 1, 1);
    let source = base_source(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state,
        chunk_digest,
        semantic_state_digest_out: semantic_out,
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count,
        step_count_in_word: source.step_count,
        pc_word: source.pc,
        public_x_out_bits: source.x_out,
    };
    let execution = relation
        .build_base(&inputs, &application_assignment)
        .expect("complete base F'");
    assert!(
        execution.is_satisfied(),
        "complete F' source relation failed at {:?}",
        execution.first_unsatisfied_row()
    );
    assert_eq!(execution.public_bit_columns().len(), 256);

    let mut forged_key_witness = execution.snapshot().witness().to_vec();
    let header_column = execution.verifier_key_columns()[4];
    forged_key_witness[header_column] += F::ONE;
    assert!(
        execution
            .snapshot()
            .first_unsatisfied_row(&forged_key_witness)
            .is_some(),
        "the Pi_CCS header used by NIFS must be bound into vk_fs"
    );

    let oracle =
        encode_r1cs_oracle(execution.snapshot(), execution.public_bit_columns()).expect("canonical oracle encoding");
    assert!(oracle.is_satisfied());
    assert_eq!(oracle.decode().expect("oracle inverse"), execution.snapshot().witness());
    let oracle_cols = oracle.assignment.len();
    drop(oracle);

    let mut derived =
        encode_r1cs_derived(execution.snapshot(), execution.public_bit_columns()).expect("derived encoding");
    assert!(derived.is_satisfied());
    assert_eq!(
        derived.decode().expect("derived inverse"),
        execution.snapshot().witness()
    );
    assert_eq!(derived.plan.public_input_len(), F_PRIME_PUBLIC_INPUT_LEN);
    assert!(derived.assignment.len() < oracle_cols);

    // Change the application input while keeping the claimed F' state and
    // every NIFS/state-transition field fixed. This used to be the dangerous
    // disconnected seam: a valid shell could describe an unrelated app step.
    let app_input_column = execution.application_columns()[1];
    let app_input_range = derived
        .plan
        .encoded_range_for_column(app_input_column)
        .expect("application input encoding");
    derived.assignment[app_input_range.start] = F::ZERO;
    assert!(
        !derived.is_satisfied(),
        "changing app state without changing the F' input digest must fail"
    );

    // A different internally valid first transition cannot choose its own
    // initial state under the same verifier context.
    let alternate_assignment = [
        F::ONE,
        F::from_u64(13),
        F::from_u64(21),
        F::from_u64(21),
        F::from_u64(34),
    ];
    let alternate_in = semantic_state_digest_fields(&alternate_assignment[1..3]);
    let alternate_out = semantic_state_digest_fields(&alternate_assignment[3..5]);
    let alternate_state = FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest(),
        pi_ccs_header_bundle: context.pi_ccs_header_bundle(),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0: alternate_in,
        z_i_in: alternate_in,
        pc: 1,
        semantic_state_digest_in: alternate_in,
        acc_digest_in: AccumulatorHandle::empty().digest_fields(),
        public_trace_in: alternate_in,
        nebula: None,
    };
    let alternate_x_out = native_x_out(&alternate_state, chunk_digest, alternate_out, zero_acc, 1, 1);
    let alternate_source = base_source(&alternate_state, alternate_x_out);
    let alternate_inputs = FPrimeBaseInputs {
        state: alternate_state,
        chunk_digest,
        semantic_state_digest_out: alternate_out,
        rows_in_chunk: 1,
        source_image: &alternate_source.image,
        chunk_count_in_word: alternate_source.chunk_count,
        step_count_in_word: alternate_source.step_count,
        pc_word: alternate_source.pc,
        public_x_out_bits: alternate_source.x_out,
    };
    let alternate = relation
        .build_base(&alternate_inputs, &alternate_assignment)
        .expect("emit alternate base witness");
    assert!(
        !alternate.is_satisfied(),
        "the verifier-owned initial semantic state must reject a substituted base transition"
    );
}

#[test]
fn complete_base_relation_rejects_caller_chosen_chunk_digest() {
    let carrier = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&carrier, 43).expect("carrier preprocessing");
    let cfg = step_config(&prep);
    let application_relation = fibonacci_step_r1cs();
    let application_assignment = [
        F::ONE,
        F::from_u64(13),
        F::from_u64(21),
        F::from_u64(21),
        F::from_u64(34),
    ];
    let semantic_in = semantic_state_digest_fields(&application_assignment[1..3]);
    let semantic_out = semantic_state_digest_fields(&application_assignment[3..5]);
    let context = full_context(&prep, semantic_in);
    let relation = FullFPrimeRelation::new(context, cfg, &application_relation, vec![1, 2], vec![3, 4])
        .expect("fixed full F' relation");
    let state = FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest(),
        pi_ccs_header_bundle: context.pi_ccs_header_bundle(),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0: semantic_in,
        z_i_in: semantic_in,
        pc: 1,
        semantic_state_digest_in: semantic_in,
        acc_digest_in: AccumulatorHandle::empty().digest_fields(),
        public_trace_in: semantic_in,
        nebula: None,
    };
    let false_chunk = rand_digest(0xdead);
    let zero_acc = construction2_zero_digest(&prep);
    let false_x_out = native_x_out(&state, false_chunk, semantic_out, zero_acc, 1, 1);
    let source = base_source(&state, false_x_out);
    let inputs = FPrimeBaseInputs {
        state,
        chunk_digest: false_chunk,
        semantic_state_digest_out: semantic_out,
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count,
        step_count_in_word: source.step_count,
        pc_word: source.pc,
        public_x_out_bits: source.x_out,
    };
    let execution = relation
        .build_base(&inputs, &application_assignment)
        .expect("emit complete F'");
    assert!(
        !execution.is_satisfied(),
        "the caller cannot choose a chunk digest unrelated to the verifier-owned step shape"
    );
}

#[test]
fn complete_recursive_relation_folds_one_fresh_instance_and_binds_the_application() {
    let carrier = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&carrier, 44).expect("carrier preprocessing");
    let cfg = step_config(&prep);
    let application_relation = fibonacci_step_r1cs();
    let application_assignment = [F::ONE, F::from_u64(5), F::from_u64(8), F::from_u64(8), F::from_u64(13)];
    let semantic_in = semantic_state_digest_fields(&application_assignment[1..3]);
    let semantic_out = semantic_state_digest_fields(&application_assignment[3..5]);
    let context = full_context(&prep, semantic_in);
    let relation = FullFPrimeRelation::new(context, cfg, &application_relation, vec![1, 2], vec![3, 4])
        .expect("fixed full F' relation");

    let zero = FixedNifsAccumulator::canonical_zero(
        &prep.params,
        prep.structure(),
        prep.combine_b_pows(),
        F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    )
    .expect("fixed-k zero accumulator");
    let zero_digest =
        AccumulatorHandle::from_running_parts(zero.claims(), zero.running().parent_authority.as_ref()).digest_fields();
    let prior_chunk_digest = uniform_chunk_digest(&prep, 0);
    let state = FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest(),
        pi_ccs_header_bundle: context.pi_ccs_header_bundle(),
        chunk_count_in: 1,
        step_count_in: 1,
        z_0: rand_digest(0x70),
        z_i_in: prior_chunk_digest,
        pc: 1,
        semantic_state_digest_in: semantic_in,
        acc_digest_in: zero_digest,
        public_trace_in: prior_chunk_digest,
        nebula: None,
    };
    let prior_x_out = native_x_out(&state, prior_chunk_digest, semantic_in, zero_digest, 1, 1);
    let fresh_assignment = encode_f_prime_superneo_public_input(prior_x_out);
    assert_eq!(fresh_assignment.len(), prep.structure().m);
    let fresh = direct_ccs::build_instance(&prep, &carrier, &fresh_assignment).expect("fresh CCS instance");
    let fresh_claim = fresh.claim.clone();
    let fresh_claims = [fresh_claim];
    let chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &fresh_claims);

    let mut prover_transcript = Transcript::with_label(TRANSCRIPT_LABEL);
    append_f_prime_step_context(&mut prover_transcript, &state, chunk_digest);
    let (next, proof) = prove_fixed(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &zero,
    )
    .expect("fixed NIFS prover");
    let next_digest =
        AccumulatorHandle::from_running_parts(next.claims(), next.running().parent_authority.as_ref()).digest_fields();
    let expected_x_out = native_x_out(&state, chunk_digest, semantic_out, next_digest, 2, 2);
    let source = recursive_source(&state, prior_x_out, expected_x_out);
    let messages = NifsVCircuitMessages {
        fresh: &fresh_claims,
        running: zero.claims(),
        running_parent_authority: zero.running().parent_authority.as_ref(),
        pi_ccs: &proof.pi_ccs,
        combined: &proof.pi_rlc.combined,
        children: next.claims(),
    };
    let inputs = FPrimeRecursiveInputs {
        state: state.clone(),
        chunk_digest,
        semantic_state_digest_out: semantic_out,
        acc_digest_out: next_digest,
        nifs_msg: messages,
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count,
        step_count_in_word: source.step_count,
        pc_word: source.pc,
        prior_x_out_bits: source.prior_x_out,
        public_x_out_bits: source.x_out,
    };
    let execution = relation
        .build_recursive(&inputs, &application_assignment)
        .expect("complete recursive F'");
    assert!(
        execution.is_satisfied(),
        "complete recursive F' failed at {:?}",
        execution.first_unsatisfied_row()
    );

    // Every authority-bearing recursive input must be connected to the
    // relation independently. Keep all other inputs honest for each case.
    {
        let mut tampered_fresh = fresh_claims.clone();
        tampered_fresh[0].x[1] += F::ONE;
        let tampered_messages = NifsVCircuitMessages {
            fresh: &tampered_fresh,
            running: zero.claims(),
            running_parent_authority: zero.running().parent_authority.as_ref(),
            pi_ccs: &proof.pi_ccs,
            combined: &proof.pi_rlc.combined,
            children: next.claims(),
        };
        let tampered_inputs = FPrimeRecursiveInputs {
            state: state.clone(),
            chunk_digest,
            semantic_state_digest_out: semantic_out,
            acc_digest_out: next_digest,
            nifs_msg: tampered_messages,
            rows_in_chunk: 1,
            source_image: &source.image,
            chunk_count_in_word: source.chunk_count,
            step_count_in_word: source.step_count,
            pc_word: source.pc,
            prior_x_out_bits: source.prior_x_out,
            public_x_out_bits: source.x_out,
        };
        let tampered = relation
            .build_recursive(&tampered_inputs, &application_assignment)
            .expect("emit latest-CCS tamper witness");
        assert!(!tampered.is_satisfied(), "mutating the latest CCS claim must fail");
    }

    {
        let mut tampered_proof = proof.clone();
        let first_coefficient = tampered_proof
            .pi_ccs
            .sumcheck
            .sumcheck_rounds
            .first_mut()
            .and_then(|round| round.first_mut())
            .expect("non-empty Π_CCS sumcheck transcript");
        *first_coefficient += K::ONE;
        let tampered_messages = NifsVCircuitMessages {
            fresh: &fresh_claims,
            running: zero.claims(),
            running_parent_authority: zero.running().parent_authority.as_ref(),
            pi_ccs: &tampered_proof.pi_ccs,
            combined: &tampered_proof.pi_rlc.combined,
            children: next.claims(),
        };
        let tampered_inputs = FPrimeRecursiveInputs {
            state: state.clone(),
            chunk_digest,
            semantic_state_digest_out: semantic_out,
            acc_digest_out: next_digest,
            nifs_msg: tampered_messages,
            rows_in_chunk: 1,
            source_image: &source.image,
            chunk_count_in_word: source.chunk_count,
            step_count_in_word: source.step_count,
            pc_word: source.pc,
            prior_x_out_bits: source.prior_x_out,
            public_x_out_bits: source.x_out,
        };
        let tampered = relation
            .build_recursive(&tampered_inputs, &application_assignment)
            .expect("emit NIFS-proof tamper witness");
        assert!(!tampered.is_satisfied(), "mutating the NIFS proof must fail");
    }

    {
        let mut tampered_running = zero.claims().to_vec();
        tampered_running[0].c.data[0] += F::ONE;
        let tampered_messages = NifsVCircuitMessages {
            fresh: &fresh_claims,
            running: &tampered_running,
            running_parent_authority: zero.running().parent_authority.as_ref(),
            pi_ccs: &proof.pi_ccs,
            combined: &proof.pi_rlc.combined,
            children: next.claims(),
        };
        let tampered_inputs = FPrimeRecursiveInputs {
            state: state.clone(),
            chunk_digest,
            semantic_state_digest_out: semantic_out,
            acc_digest_out: next_digest,
            nifs_msg: tampered_messages,
            rows_in_chunk: 1,
            source_image: &source.image,
            chunk_count_in_word: source.chunk_count,
            step_count_in_word: source.step_count,
            pc_word: source.pc,
            prior_x_out_bits: source.prior_x_out,
            public_x_out_bits: source.x_out,
        };
        let tampered = relation
            .build_recursive(&tampered_inputs, &application_assignment)
            .expect("emit running-accumulator tamper witness");
        assert!(!tampered.is_satisfied(), "mutating the running accumulator must fail");
    }

    {
        let mut tampered_state = state.clone();
        tampered_state.chunk_count_in += 1;
        let tampered_inputs = FPrimeRecursiveInputs {
            state: tampered_state,
            chunk_digest,
            semantic_state_digest_out: semantic_out,
            acc_digest_out: next_digest,
            nifs_msg: NifsVCircuitMessages {
                fresh: &fresh_claims,
                running: zero.claims(),
                running_parent_authority: zero.running().parent_authority.as_ref(),
                pi_ccs: &proof.pi_ccs,
                combined: &proof.pi_rlc.combined,
                children: next.claims(),
            },
            rows_in_chunk: 1,
            source_image: &source.image,
            chunk_count_in_word: source.chunk_count,
            step_count_in_word: source.step_count,
            pc_word: source.pc,
            prior_x_out_bits: source.prior_x_out,
            public_x_out_bits: source.x_out,
        };
        let tampered = relation
            .build_recursive(&tampered_inputs, &application_assignment)
            .expect("emit iteration tamper witness");
        assert!(!tampered.is_satisfied(), "mutating the iteration must fail");
    }

    {
        let mut tampered_state = state.clone();
        tampered_state.z_i_in[0] += F::ONE;
        let tampered_inputs = FPrimeRecursiveInputs {
            state: tampered_state,
            chunk_digest,
            semantic_state_digest_out: semantic_out,
            acc_digest_out: next_digest,
            nifs_msg: NifsVCircuitMessages {
                fresh: &fresh_claims,
                running: zero.claims(),
                running_parent_authority: zero.running().parent_authority.as_ref(),
                pi_ccs: &proof.pi_ccs,
                combined: &proof.pi_rlc.combined,
                children: next.claims(),
            },
            rows_in_chunk: 1,
            source_image: &source.image,
            chunk_count_in_word: source.chunk_count,
            step_count_in_word: source.step_count,
            pc_word: source.pc,
            prior_x_out_bits: source.prior_x_out,
            public_x_out_bits: source.x_out,
        };
        let tampered = relation
            .build_recursive(&tampered_inputs, &application_assignment)
            .expect("emit current-state tamper witness");
        assert!(
            !tampered.is_satisfied(),
            "mutating the current application state must fail"
        );
    }

    {
        let mut tampered_image = source.image.clone();
        let public_bit = source.x_out.start();
        let flipped = tampered_image.values()[public_bit] == F::ZERO;
        tampered_image.set_bit(public_bit, flipped);
        let tampered_inputs = FPrimeRecursiveInputs {
            state: state.clone(),
            chunk_digest,
            semantic_state_digest_out: semantic_out,
            acc_digest_out: next_digest,
            nifs_msg: NifsVCircuitMessages {
                fresh: &fresh_claims,
                running: zero.claims(),
                running_parent_authority: zero.running().parent_authority.as_ref(),
                pi_ccs: &proof.pi_ccs,
                combined: &proof.pi_rlc.combined,
                children: next.claims(),
            },
            rows_in_chunk: 1,
            source_image: &tampered_image,
            chunk_count_in_word: source.chunk_count,
            step_count_in_word: source.step_count,
            pc_word: source.pc,
            prior_x_out_bits: source.prior_x_out,
            public_x_out_bits: source.x_out,
        };
        let tampered = relation
            .build_recursive(&tampered_inputs, &application_assignment)
            .expect("emit public-digest tamper witness");
        assert!(!tampered.is_satisfied(), "mutating the public x_out digest must fail");
    }

    // Construction 2 has one relation with an internal base/recursive
    // branch. Build a base witness under this same verifier context, then
    // freeze both branch shapes behind one constrained selector.
    let base_assignment = application_assignment;
    let base_semantic_in = semantic_state_digest_fields(&base_assignment[1..3]);
    let base_semantic_out = semantic_state_digest_fields(&base_assignment[3..5]);
    let base_state = FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest(),
        pi_ccs_header_bundle: context.pi_ccs_header_bundle(),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0: base_semantic_in,
        z_i_in: base_semantic_in,
        pc: 1,
        semantic_state_digest_in: base_semantic_in,
        acc_digest_in: AccumulatorHandle::empty().digest_fields(),
        public_trace_in: base_semantic_in,
        nebula: None,
    };
    let base_chunk_digest = uniform_chunk_digest(&prep, 0);
    let base_x_out = native_x_out(&base_state, base_chunk_digest, base_semantic_out, zero_digest, 1, 1);
    let base_source = base_source(&base_state, base_x_out);
    let base_inputs = FPrimeBaseInputs {
        state: base_state,
        chunk_digest: base_chunk_digest,
        semantic_state_digest_out: base_semantic_out,
        rows_in_chunk: 1,
        source_image: &base_source.image,
        chunk_count_in_word: base_source.chunk_count,
        step_count_in_word: base_source.step_count,
        pc_word: base_source.pc,
        public_x_out_bits: base_source.x_out,
    };
    let base_branch = relation
        .build_base(&base_inputs, &base_assignment)
        .expect("complete base branch");
    source_role_manifest::check_source_role_manifest(&base_branch, &execution);

    // This is another valid Fibonacci transition, but it is unrelated to
    // the state absorbed by the already-valid NIFS proof. Its values must not
    // change the verifier-owned recursive matrix, and it must not be accepted
    // as the template used to freeze the unified relation.
    let disconnected_assignment = [
        F::ONE,
        F::from_u64(13),
        F::from_u64(21),
        F::from_u64(21),
        F::from_u64(34),
    ];
    let disconnected_branch = relation
        .build_recursive(&inputs, &disconnected_assignment)
        .expect("emit disconnected recursive F'");
    assert!(
        execution
            .snapshot()
            .has_same_relation(disconnected_branch.snapshot()),
        "recursive relation shape must not depend on application witness values"
    );
    assert!(matches!(
        FullFPrimeShape::new(&base_branch, &disconnected_branch),
        Err(FullFPrimeError::UnsatisfiedTemplate { branch: "recursive" })
    ));

    let base_estimate = estimate_r1cs_encoding(
        base_branch.snapshot(),
        base_branch.public_bit_columns(),
        LowNormR1csEncodingKind::Derived,
    )
    .expect("base branch estimate");
    let recursive_estimate = estimate_r1cs_encoding(
        execution.snapshot(),
        execution.public_bit_columns(),
        LowNormR1csEncodingKind::Derived,
    )
    .expect("recursive branch estimate");
    let direct_estimate = estimate_selector_gated_r1cs_encoding(
        base_branch.snapshot(),
        base_branch.public_bit_columns(),
        execution.snapshot(),
        execution.public_bit_columns(),
        LowNormR1csEncodingKind::Derived,
    )
    .expect("direct selector-gated estimate");
    assert_direct_selector_cost_formula(&base_estimate, &recursive_estimate, &direct_estimate);
    // Branch source dimensions are materialized above. The direct selector
    // number is only a regression snapshot of an un-audited cost formula: no
    // selector-gated relation is materialized or proved sound by this test.
    assert_eq!((base_estimate.source_rows, base_estimate.source_cols), (22_812, 22_353));
    assert_eq!(
        (recursive_estimate.source_rows, recursive_estimate.source_cols),
        (2_576_416, 2_399_107)
    );
    assert_eq!(
        (direct_estimate.encoded_rows, direct_estimate.encoded_cols),
        (258_444_060, 190_149_709)
    );
    eprintln!(
        "full F' branches: base={}x{} ({} field), recursive={}x{} ({} field, {} linear definitions), un-audited direct CCS estimate={}x{}",
        base_estimate.source_rows,
        base_estimate.source_cols,
        base_estimate.canonical_field_source_cols,
        recursive_estimate.source_rows,
        recursive_estimate.source_cols,
        recursive_estimate.canonical_field_source_cols,
        recursive_estimate.linearly_derived_source_cols,
        direct_estimate.encoded_rows,
        direct_estimate.encoded_cols,
    );
    let shape = FullFPrimeShape::new(&base_branch, &execution).expect("single full F' shape");
    let (base_gadget, recursive_gadget) = shape
        .gadget_native_branch_estimates()
        .expect("gadget-native branch estimates");
    let fixed_gadget = shape
        .gadget_native_fixed_estimate()
        .expect("fixed gadget-native estimate");
    let base_u64 = audit_r1cs_gadget_native_canonical_u64(
        base_branch.snapshot(),
        base_branch.encoding_trace(),
        base_branch.public_bit_columns(),
    )
    .expect("base canonical-u64 census");
    let recursive_u64 = audit_r1cs_gadget_native_canonical_u64(
        execution.snapshot(),
        execution.encoding_trace(),
        execution.public_bit_columns(),
    )
    .expect("recursive canonical-u64 census");
    assert_eq!(
        (
            base_u64.census.total,
            base_u64.census.direct,
            base_u64.census.equality_linked,
            base_u64.census.linear,
            base_u64.census.field_linearly_derived,
        ),
        (7, 0, 4, 3, 7)
    );
    assert_eq!(
        (
            recursive_u64.census.total,
            recursive_u64.census.direct,
            recursive_u64.census.equality_linked,
            recursive_u64.census.linear,
            recursive_u64.census.field_linearly_derived,
        ),
        (253, 2, 8, 243, 251)
    );
    eprintln!("CANONICAL_U64|base|{:?}", base_u64.census);
    eprintln!("CANONICAL_U64|recursive|{:?}", recursive_u64.census);
    for stage in &base_u64.stages {
        eprintln!("CANONICAL_U64_STAGE|{}|{:?}", stage.stage, stage.census);
    }
    for stage in &recursive_u64.stages {
        eprintln!("CANONICAL_U64_STAGE|{}|{:?}", stage.stage, stage.census);
    }
    assert_eq!((base_gadget.encoded_rows, base_gadget.encoded_cols), (66_358, 125_695));
    assert_eq!(
        (recursive_gadget.encoded_rows, recursive_gadget.encoded_cols),
        (4_933_049, 8_137_378)
    );
    assert_eq!(
        (fixed_gadget.encoded_rows, fixed_gadget.encoded_cols),
        (6_184_892, 8_262_817)
    );
    assert_eq!(base_gadget.public_input_len, F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(recursive_gadget.public_input_len, F_PRIME_PUBLIC_INPUT_LEN);
    assert!(base_gadget.encoded_cols < base_estimate.encoded_cols);
    assert!(recursive_gadget.encoded_cols < recursive_estimate.encoded_cols);
    assert!(recursive_gadget.gadget_derived_source_cols > 0);
    assert_eq!(fixed_gadget.public_input_len, F_PRIME_PUBLIC_INPUT_LEN);
    assert!(fixed_gadget.encoded_cols > recursive_gadget.encoded_cols);
    assert_fixed_selector_cost_formula(&fixed_gadget);
    eprintln!(
        "full F' gadget-native: base={}x{}, recursive={}x{}, fixed={}x{} ({} projected gadget fields, {} synthetic ring fields, {} synthetic product-sum fields)",
        base_gadget.encoded_rows,
        base_gadget.encoded_cols,
        recursive_gadget.encoded_rows,
        recursive_gadget.encoded_cols,
        fixed_gadget.encoded_rows,
        fixed_gadget.encoded_cols,
        recursive_gadget.gadget_derived_source_cols,
        recursive_gadget.synthetic_ring_fields,
        recursive_gadget.synthetic_product_sum_fields,
    );
    let base_stage_profile = profile_r1cs_gadget_native_stages(
        base_branch.snapshot(),
        base_branch.encoding_trace(),
        base_branch.public_bit_columns(),
    )
    .expect("base F' stage profile");
    assert_eq!(base_stage_profile.total, base_gadget);
    assert_f_prime_base_stage_hierarchy(&base_stage_profile);
    let stage_profile = profile_r1cs_gadget_native_stages(
        execution.snapshot(),
        execution.encoding_trace(),
        execution.public_bit_columns(),
    )
    .expect("recursive F' stage profile");
    assert_eq!(stage_profile.total, recursive_gadget);
    assert_f_prime_recursive_stage_hierarchy(&stage_profile);
    assert_pi_ccs_stage_hierarchy(&stage_profile);
    assert_pi_rlc_stage_hierarchy(&stage_profile);
    assert_dominant_sis_snapshots(&stage_profile);
    assert_protocol_row_family_snapshots(&stage_profile);
    assert_pi_ccs_nc_terminal_row_families(&stage_profile, execution.row_family_ranges());
    print_stage_cost_families(&stage_profile);
    let challenge = stage_profile
        .aggregate_prefix("nifs.pi_rlc.challenge")
        .expect("Pi_RLC challenge cost center");
    assert_eq!(challenge.source_rows, 127_611);
    assert_eq!(challenge.source_cols, 121_566);
    assert_eq!(challenge.one_bit_source_cols, 36_945);
    assert_eq!(challenge.canonical_binary_field_source_cols, 0);
    assert_eq!(challenge.ordinary_private_field_source_cols, 7_758);
    assert_eq!(challenge.linearly_derived_source_cols, 26_169);
    assert_eq!(challenge.gadget_derived_source_cols, 50_694);
    assert_eq!(challenge.encoded_cols, 370_383);
    assert_eq!(challenge.encoded_rows, 198_567);
    assert_eq!(challenge.redundant_boolean_source_rows, 23_505);
    #[derive(Debug)]
    struct ChallengeLeaf {
        path: &'static str,
        occurrences: usize,
        source_rows: usize,
        source_cols: usize,
        encoded_rows: usize,
        encoded_cols: usize,
        ordinary_fields: usize,
        bits: usize,
        linear: usize,
        gadget: usize,
        fallback: usize,
        redundant_boolean_rows: usize,
        selection_aggregate_rows: usize,
        poseidon_permutations: usize,
        sboxes: usize,
    }
    let expected_challenge_leaves = [
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST,
            occurrences: 1,
            source_rows: 1_206,
            source_cols: 1_206,
            encoded_rows: 3_698,
            encoded_cols: 7_052,
            ordinary_fields: 172,
            bits: 0,
            linear: 518,
            gadget: 516,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 2,
            sboxes: 172,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR,
            occurrences: 15,
            source_rows: 645,
            source_cols: 645,
            encoded_rows: 1_849,
            encoded_cols: 3_526,
            ordinary_fields: 86,
            bits: 0,
            linear: 301,
            gadget: 258,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 1,
            sboxes: 86,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SAMPLE_INITIALIZE,
            occurrences: 15,
            source_rows: 15,
            source_cols: 15,
            encoded_rows: 0,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 15,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::TRANSCRIPT_DIGEST,
            occurrences: 60,
            source_rows: 45_240,
            source_cols: 45_240,
            encoded_rows: 138_675,
            encoded_cols: 264_450,
            ordinary_fields: 6_450,
            bits: 0,
            linear: 19_440,
            gadget: 19_350,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 75,
            sboxes: 6_450,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION,
            occurrences: 240,
            source_rows: 16_560,
            source_cols: 15_840,
            encoded_rows: 13_680,
            encoded_cols: 25_200,
            ordinary_fields: 240,
            bits: 15_360,
            linear: 240,
            gadget: 0,
            fallback: 960,
            redundant_boolean_rows: 15_360,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::ACCEPT_TREE_BIT_PAIRS,
            occurrences: 960,
            source_rows: 0,
            source_cols: 0,
            encoded_rows: 6_720,
            encoded_cols: 13_440,
            ordinary_fields: 0,
            bits: 0,
            linear: 0,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::ACCEPT_PRODUCT_AGGREGATE,
            occurrences: 960,
            source_rows: 0,
            source_cols: 0,
            encoded_rows: 960,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 0,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::ACCEPT_ROOT_BINDING,
            occurrences: 960,
            source_rows: 3_840,
            source_cols: 1_920,
            encoded_rows: 960,
            encoded_cols: 960,
            ordinary_fields: 0,
            bits: 960,
            linear: 0,
            gadget: 960,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::CHUNK_SYMBOL_AND_PREFIX,
            occurrences: 960,
            source_rows: 1_920,
            source_cols: 1_920,
            encoded_rows: 0,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 1_920,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::ACCEPTANCE_BOUND,
            occurrences: 15,
            source_rows: 90,
            source_cols: 75,
            encoded_rows: 45,
            encoded_cols: 45,
            ordinary_fields: 0,
            bits: 45,
            linear: 30,
            gadget: 0,
            fallback: 15,
            redundant_boolean_rows: 45,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SELECT_INITIALIZE,
            occurrences: 15,
            source_rows: 15,
            source_cols: 15,
            encoded_rows: 0,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 15,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SELECT_ONE_HOT,
            occurrences: 810,
            source_rows: 9_720,
            source_cols: 8_910,
            encoded_rows: 4_860,
            encoded_cols: 8_100,
            ordinary_fields: 0,
            bits: 8_100,
            linear: 810,
            gadget: 0,
            fallback: 810,
            redundant_boolean_rows: 8_100,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SELECT_PRODUCTS,
            occurrences: 810,
            source_rows: 26_730,
            source_cols: 26_730,
            encoded_rows: 0,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 0,
            gadget: 26_730,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 0,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SELECT_BIND_ACCEPT,
            occurrences: 810,
            source_rows: 810,
            source_cols: 0,
            encoded_rows: 810,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 0,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 810,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SELECT_BIND_PREFIX,
            occurrences: 810,
            source_rows: 810,
            source_cols: 0,
            encoded_rows: 810,
            encoded_cols: 0,
            ordinary_fields: 0,
            bits: 0,
            linear: 0,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 810,
            poseidon_permutations: 0,
            sboxes: 0,
        },
        ChallengeLeaf {
            path: pi_rlc_challenge_stage::SELECT_BIND_SYMBOL,
            occurrences: 810,
            source_rows: 810,
            source_cols: 810,
            encoded_rows: 17_820,
            encoded_cols: 33_210,
            ordinary_fields: 810,
            bits: 0,
            linear: 0,
            gadget: 0,
            fallback: 0,
            redundant_boolean_rows: 0,
            selection_aggregate_rows: 810,
            poseidon_permutations: 0,
            sboxes: 0,
        },
    ];
    let actual_challenge_nodes = stage_profile.aggregate_by_label();
    for expected in &expected_challenge_leaves {
        let actual = actual_challenge_nodes
            .iter()
            .find(|stage| stage.label == expected.path)
            .unwrap_or_else(|| panic!("missing challenge leaf {}", expected.path));
        assert_eq!(actual.label, expected.path);
        assert_eq!(
            actual.occurrences, expected.occurrences,
            "{} occurrences",
            expected.path
        );
        assert_eq!(
            actual.source_rows, expected.source_rows,
            "{} source rows",
            expected.path
        );
        assert_eq!(
            actual.source_cols, expected.source_cols,
            "{} source columns",
            expected.path
        );
        assert_eq!(
            actual.encoded_rows, expected.encoded_rows,
            "{} encoded rows",
            expected.path
        );
        assert_eq!(
            actual.encoded_cols, expected.encoded_cols,
            "{} encoded columns",
            expected.path
        );
        assert_eq!(
            actual.canonical_binary_field_source_cols, 0,
            "{} canonical fields",
            expected.path
        );
        assert_eq!(
            actual.ordinary_private_field_source_cols, expected.ordinary_fields,
            "{} ordinary-private fields",
            expected.path
        );
        assert_eq!(actual.one_bit_source_cols, expected.bits, "{} bits", expected.path);
        assert_eq!(
            actual.linearly_derived_source_cols, expected.linear,
            "{} linear columns",
            expected.path
        );
        assert_eq!(
            actual.gadget_derived_source_cols, expected.gadget,
            "{} gadget columns",
            expected.path
        );
        assert_eq!(
            actual.fallback_source_rows, expected.fallback,
            "{} fallback rows",
            expected.path
        );
        assert_eq!(
            actual.redundant_boolean_source_rows, expected.redundant_boolean_rows,
            "{} exact duplicate Boolean rows",
            expected.path
        );
        assert_eq!(
            actual.selection_accept_aggregate_rows
                + actual.selection_prefix_aggregate_rows
                + actual.selection_symbol_aggregate_rows,
            expected.selection_aggregate_rows,
            "{} selection aggregate rows",
            expected.path
        );
        assert_eq!(
            actual.poseidon_permutations, expected.poseidon_permutations,
            "{} Poseidon permutations",
            expected.path
        );
        assert_eq!(actual.sboxes, expected.sboxes, "{} S-boxes", expected.path);
    }
    let selection_bind = stage_profile
        .aggregate_prefix(pi_rlc_challenge_stage::SELECT_BIND)
        .expect("selection-bind aggregate");
    assert_eq!(selection_bind.source_rows, 2_430);
    assert_eq!(selection_bind.source_cols, 810);
    assert_eq!(selection_bind.encoded_rows, 19_440);
    assert_eq!(selection_bind.encoded_cols, 33_210);
    assert_eq!(selection_bind.canonical_binary_field_source_cols, 0);
    assert_eq!(selection_bind.ordinary_private_field_source_cols, 810);
    let trace = execution.encoding_trace();
    let hash_permutations = trace
        .poseidon_hashes()
        .iter()
        .map(|hash| hash.permutation_range.len())
        .sum::<usize>();
    let mut hash_histogram = BTreeMap::<usize, (usize, usize)>::new();
    for hash in trace.poseidon_hashes() {
        let entry = hash_histogram.entry(hash.input_len).or_default();
        entry.0 += 1;
        entry.1 += hash.permutation_range.len();
    }
    eprintln!(
        "full F' nonlinear trace: {} Poseidon permutations ({} inside {} one-shot hashes, {} transcript), {} sboxes, {} K-muls, {} ring-muls; hash histogram {:?}",
        trace.poseidon_permutations().len(),
        hash_permutations,
        trace.poseidon_hashes().len(),
        trace.poseidon_permutations().len() - hash_permutations,
        trace.sbox7().len(),
        trace.k_muls().len(),
        trace.ring_muls_toom3().len(),
        hash_histogram,
    );
    let full_base = shape
        .execute_base(&base_branch)
        .expect("base witness in unified F'");
    assert!(full_base.is_satisfied());
    assert!(full_base.snapshot().unconstrained_columns().is_empty());

    let full_recursive = shape
        .execute_recursive(&execution)
        .expect("recursive witness in unified F'");
    assert!(full_recursive.is_satisfied());
    assert!(full_recursive.snapshot().unconstrained_columns().is_empty());
    assert_eq!(
        (full_recursive.snapshot().rows(), full_recursive.snapshot().cols()),
        (7_575_344, 4_998_209),
        "materialized selector-composed source-R1CS dimensions"
    );
    let estimate = estimate_r1cs_encoding(
        full_recursive.snapshot(),
        full_recursive.public_bit_columns(),
        LowNormR1csEncodingKind::Derived,
    )
    .expect("full F' derived-encoding estimate");
    assert_eq!(estimate.public_input_len, F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(
        (estimate.encoded_rows, estimate.encoded_cols),
        (568_647_410, 420_130_158),
        "generic low-norm estimate of the materialized source selector relation"
    );
    assert_eq!(direct_estimate.public_input_len, F_PRIME_PUBLIC_INPUT_LEN);
    assert!(direct_estimate.encoded_cols < estimate.encoded_cols);
    assert!(direct_estimate.encoded_rows < estimate.encoded_rows);
    eprintln!(
        "full F' reference shape: source={}x{}, derived={}x{}",
        estimate.source_rows, estimate.source_cols, estimate.encoded_rows, estimate.encoded_cols
    );
    assert!(
        full_base
            .snapshot()
            .has_same_relation(full_recursive.snapshot()),
        "base and recursive executions must have one identical folded relation"
    );
    drop(full_base);
    drop(full_recursive);

    let disconnected = shape
        .execute_recursive(&disconnected_branch)
        .expect("disconnected witness has the same unified relation shape");
    assert!(
        !disconnected.is_satisfied(),
        "a valid but unrelated application transition must not satisfy the recursive shell"
    );
}
