//! Red-team regression for the recursive F' encoding boundary.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::matrix::Mat as NeoMat;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::compiler::{
    assemble_shared_chunk_traces, assemble_step_from_shared, nifs_payload_inputs_for_source_image, perp_nifs_ce_view,
};
use neo_fold_clean::frontends::f_prime::image::NifsPayloadShape;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::r1cs_f_prime::{
    self, assignment_to_bits, compile_step, encode_r1cs_f_prime_step, start_chain, R1csEncoderInput,
    R1csFPrimeStepInput,
};
use neo_fold_clean::paper::construction2::{
    LatestInstance, ProofState, SemanticStateAdvance, SemanticStateMode, State,
};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, initial_boundary_digest, public_trace_seed_digest, AccumulatorHandle,
};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::relations::CcsClaim;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, assignment_one_product_with_extras, make_small_plan, make_stateful_plan_with_anchor,
    make_tiny_lifecycle_plan, make_tiny_stateful_lifecycle_plan_with_anchor, one_product_r1cs, tiny_params,
};

/// Full-lane direct R1CS permits variable zero to be an ordinary public
/// input. An output-only semantic plan must therefore bind it just like every
/// other public coordinate; the Bellpepper constant-one convention is not an
/// invariant of this frontend.
#[test]
fn r1cs_f_prime_plan_binds_public_variable_zero_in_explicit_semantic_output() {
    let r1cs = one_product_r1cs();
    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("small plan carries state_x_out");
    state_x_out.semantic_state_out_var_indices = vec![1];
    assert_eq!(
        state_x_out.app_public_input_var_indices,
        vec![0],
        "fixture exposes variable zero as the R1CS public input"
    );

    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6054)
        .expect("current validation treats public variable zero as an implicit constant");
    assert!(
        !prep.anchors().constant_lane_pinned,
        "attack requires full-lane variable zero to remain an ordinary app value"
    );

    let first_assignment = assignment_one_product(2, 3);
    let second_assignment = assignment_one_product(2, 4);
    assert_ne!(first_assignment[0], second_assignment[0]);
    assert_eq!(first_assignment[1], second_assignment[1]);

    let compile = |assignment| {
        let mut ctx = start_chain(&prep).expect("fresh compiler context");
        compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment }).expect("compile satisfying base step")
    };
    let first = compile(first_assignment);
    let second = compile(second_assignment);
    assert!(first.encoded.structure.is_satisfied(&first.encoded.witness));
    assert!(second
        .encoded
        .structure
        .is_satisfied(&second.encoded.witness));
    assert_eq!(
        first.semantic_state_digest_out, second.semantic_state_digest_out,
        "attack keeps the selected explicit semantic output fixed"
    );

    assert_ne!(
        first.public_output_digest, second.public_output_digest,
        "soundness failure: two satisfying R1CS-F' steps with different public x[0] have the same verifier-visible state_x_out because plan validation exempted variable zero without pinning it"
    );
}

/// Typed-width preprocessing must preserve the direct R1CS language.  Merely
/// selecting a narrower encoding for variable zero currently imports the
/// Bellpepper constant-one convention and rejects otherwise-valid assignments
/// whose public `z[0]` is an ordinary value.
#[test]
fn typed_r1cs_widths_preserve_ordinary_public_variable_zero() {
    let r1cs = one_product_r1cs();
    let honest_assignment = assignment_one_product(2, 3);
    assert_eq!(honest_assignment[0], F::from_u64(6));
    r1cs.is_satisfied_by(&honest_assignment)
        .expect("direct R1CS accepts ordinary public variable zero");

    let mut plan = make_small_plan(r1cs.m(), r1cs.m_in);
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[0] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;

    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6055)
        .expect("preprocessing accepts the claimed one-bit width for ordinary z[0]");
    assert!(
        prep.anchors().constant_lane_pinned,
        "typed layout silently changes the relation by pinning ordinary z[0]"
    );

    let mut ctx = start_chain(&prep).expect("fresh compiler context");
    compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: honest_assignment,
        },
    )
    .expect(
        "completeness failure: typed encoding must accept every assignment accepted by the verifier-owned direct R1CS",
    );
}

/// The semantic-state digest is the authoritative cross-step link for a
/// stateful F' chain. It must therefore bind the tuple arity as well as the
/// field values. The current Poseidon2 sponge cannot distinguish a missing
/// final lane from an absorbed zero lane, and plan validation permits the
/// input and output semantic tuples to have different arities.
#[test]
fn semantic_state_digest_binds_declared_tuple_arity() {
    let r1cs = one_product_r1cs();
    let state = F::from_u64(42);
    let input_preimage = build_semantic_state_preimage_fields(&[state]);
    let output_preimage = build_semantic_state_preimage_fields(&[state, F::ZERO]);
    assert_ne!(input_preimage, output_preimage, "attack uses distinct state tuples");

    let input_digest = encode_poseidon_trace(&input_preimage).digest_native;
    let plan = make_stateful_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![6],
        vec![6, 7],
        Some(digest_fields_as_digest32(input_digest)),
    );
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6027)
        .expect("unequal-arity semantic-state plan is accepted");
    let mut ctx = start_chain(&prep).expect("start compiler context");
    let compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: assignment_one_product_with_extras(1, 1, &[(6, 42), (7, 0)]),
        },
    )
    .expect("compile state transition [42] -> [42, 0]");

    assert!(
        compiled
            .encoded
            .structure
            .is_satisfied(&compiled.encoded.witness),
        "the colliding semantic-state transition must satisfy the verifier-owned F' rows"
    );
    assert_ne!(
        compiled.semantic_state_digest_in, compiled.semantic_state_digest_out,
        "soundness failure: F' treats distinct semantic states [42] and [42, 0] as the same cross-step authority"
    );
}

/// The semantic-state encoding must also separate a full-field coordinate
/// from a packed Boolean vector.  Those are different application-state
/// types even when their packed field values happen to agree.  At present
/// both paths use the same tag and the packed-bit path omits the bit count,
/// so the field tuple `[1]` and Boolean tuple `[1, 0]` have the exact same
/// Poseidon preimage.  This collision survives merely fixing the sponge's
/// missing-length padding demonstrated above.
#[test]
fn semantic_state_digest_binds_field_and_packed_bit_domains() {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::ZERO);
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::ZERO);
    b[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ZERO - F::ONE;
    let c = NeoMat::zero(1, m, F::ZERO);
    let r1cs = R1cs { a, b, c, m_in: 2 };

    let semantic_in = F::ONE;
    let input_digest = encode_poseidon_trace(&build_semantic_state_preimage_fields(&[semantic_in])).digest_native;
    let mut plan = make_stateful_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![2],
        Vec::new(),
        Some(digest_fields_as_digest32(input_digest)),
    );
    plan.app_private_var_widths = vec![POSEIDON2_GOLDILOCKS_BITS; r1cs.m()];
    plan.app_private_var_widths[0] = 1;
    plan.app_private_var_widths[1] = 1;
    plan.limbs = plan.app_private_var_widths.iter().sum::<usize>() + 1;
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("stateful plan carries state_x_out");
    state_x_out.app_public_input_var_indices.clear();
    state_x_out.app_public_input_bit_var_indices = vec![0, 1];

    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6028)
        .expect("field-input/packed-bit-output plan is accepted");
    let mut assignment = vec![F::ZERO; r1cs.m()];
    assignment[0] = F::ONE;
    assignment[2] = semantic_in;
    let mut ctx = start_chain(&prep).expect("start compiler context");
    let compiled = compile_step(&prep, &mut ctx, R1csFPrimeStepInput { assignment })
        .expect("compile field-to-packed-bit semantic transition");

    assert!(
        compiled
            .encoded
            .structure
            .is_satisfied(&compiled.encoded.witness),
        "the type-confused semantic-state transition must satisfy the verifier-owned F' rows"
    );
    assert_ne!(
        compiled.semantic_state_digest_in, compiled.semantic_state_digest_out,
        "soundness failure: F' gives the full-field tuple [1] and packed Boolean tuple [1, 0] the same semantic-state authority digest"
    );
}

/// Every 64-bit trace word is documented as a canonical Goldilocks encoding.
/// The optimized F' shell, however, uses the bits only through their field
/// recomposition. Replacing an all-zero internal Poseidon word with the binary
/// representation of the Goldilocks modulus therefore leaves every CCS row
/// unchanged while producing a second low-norm opening for the same trace.
#[test]
fn lifecycle_rejects_noncanonical_poseidon_trace_word() {
    const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

    let r1cs = fibonacci_transition_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6022)
        .expect("F' preprocessing");
    let mut ctx = start_chain(&prep).expect("start compiler context");
    let mut compiled = compile_step(
        &prep,
        &mut ctx,
        R1csFPrimeStepInput {
            assignment: fibonacci_assignment(1, 1),
        },
    )
    .expect("compile honest base step");

    let lane = compiled
        .encoded
        .structure
        .lane_slots
        .poseidon_trace_lanes
        .iter()
        .flatten()
        .copied()
        .find(|lane| {
            compiled.encoded.witness[lane.bit_start..lane.bit_start + POSEIDON2_GOLDILOCKS_BITS]
                .iter()
                .all(|&bit| bit == F::ZERO)
        })
        .expect("honest Poseidon trace contains an all-zero word");

    for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
        compiled.encoded.witness[lane.bit_start + bit] = if (GOLDILOCKS_MODULUS >> bit) & 1 == 1 {
            F::ONE
        } else {
            F::ZERO
        };
    }
    assert_ne!(compiled.encoded.witness, compiled.encoded.image.values);
    assert!(
        compiled
            .encoded
            .structure
            .is_satisfied(&compiled.encoded.witness),
        "modulus alias must preserve every verifier-owned CCS row"
    );

    let forged = r1cs_f_prime::build_instance(&prep, &compiled.encoded)
        .expect("build noncanonical but locally satisfying instance");
    let audit = neo_fold_clean::lifecycle::prove::prove_one_with_semantic_state(
        &prep.prep,
        vec![forged],
        digest_fields_as_digest32(compiled.semantic_state_digest_in),
        digest_fields_as_digest32(compiled.semantic_state_digest_out),
    )
    .expect("prove noncanonical F' instance");
    let proof = neo_fold_clean::finish_uncompressed(&prep.prep, audit).expect("finalize noncanonical F' instance");
    let accepted = neo_fold_clean::verify_uncompressed(&prep.prep, &proof).is_ok();

    assert!(
        !accepted,
        "soundness failure: the lifecycle accepted a noncanonical active Poseidon trace word equal to the Goldilocks modulus"
    );
}

/// A semantic digest is carried as bytes but absorbed as four Goldilocks
/// elements.  Verification must reject noncanonical byte words instead of
/// reducing them modulo the field and giving two public state images the same
/// recursive hash-chain meaning.
fn stateful_zero_base_step_fixture() -> (
    r1cs_f_prime::R1csFPrimePreprocessing,
    State,
    Vec<CcsClaim>,
    neo_fold_clean::paper::construction2::StepProof,
) {
    let r1cs = one_product_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(r1cs.m(), r1cs.m_in, vec![6], vec![7], Some([0u8; 32]));
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6036)
        .expect("stateful preprocessing with canonical zero anchor");
    assert!(matches!(prep.prep.semantic_state_mode(), SemanticStateMode::Stateful));

    let empty_acc = AccumulatorHandle::empty().digest();
    let base = State::base(
        initial_boundary_digest(prep.prep.structure_digest(), prep.prep.public_input_len),
        public_trace_seed_digest(prep.prep.structure_digest()),
        empty_acc,
        [0u8; 32],
    );
    let assignment = vec![F::ZERO; prep.prep.structure().m];
    let instance = neo_fold_clean::CcsInstance::from_low_norm_assignment(
        &prep.prep.params,
        &prep.prep.log,
        prep.prep.structure(),
        &assignment,
        prep.prep
            .public_input_len
            .expect("F' preprocessing fixes public-input width"),
    )
    .expect("shape-valid base deposit");
    let claims = vec![instance.claim.clone()];
    let (_next, mut step) = neo_fold_clean::paper::construction2::step_with_semantic_state(
        &prep.prep.params,
        prep.prep.structure(),
        prep.prep.optimized_cache(),
        prep.prep.structure_digest(),
        &prep.prep.log,
        prep.prep.mix_rhos_commits(),
        prep.prep.combine_b_pows(),
        &prep.prep.vk,
        base.clone(),
        vec![instance],
        SemanticStateAdvance::Stateful([0u8; 32]),
        None,
        None,
    )
    .expect("produce canonical zero semantic step");
    step.semantic_state_digest = [0u8; 32];
    (prep, base, claims, step)
}

fn verify_stateful_base_step(
    prep: &r1cs_f_prime::R1csFPrimePreprocessing,
    base: State,
    claims: &[CcsClaim],
    step: &neo_fold_clean::paper::construction2::StepProof,
) -> Result<State, neo_fold_clean::paper::construction2::Error> {
    neo_fold_clean::paper::construction2::verify_step(
        &prep.prep.params,
        prep.prep.structure(),
        prep.prep.optimized_cache(),
        prep.prep.structure_digest(),
        prep.prep.mix_rhos_commits(),
        prep.prep.combine_b_pows(),
        &prep.prep.vk,
        base,
        claims,
        step,
        SemanticStateMode::Stateful,
        None,
    )
}

#[test]
fn native_stateful_step_rejects_noncanonical_semantic_digest_bytes() {
    const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

    let (prep, base, claims, mut step) = stateful_zero_base_step_fixture();
    verify_stateful_base_step(&prep, base.clone(), &claims, &step).expect("canonical step verifies");

    let mut modulus_alias = [0u8; 32];
    modulus_alias[..8].copy_from_slice(&GOLDILOCKS_MODULUS.to_le_bytes());
    assert_ne!(modulus_alias, [0u8; 32]);
    assert_eq!(
        digest32_as_fields(modulus_alias),
        digest32_as_fields([0u8; 32]),
        "attack precondition: raw modulus word aliases canonical zero in the field"
    );
    step.semantic_state_digest = modulus_alias;
    let result = verify_stateful_base_step(&prep, base, &claims, &step);

    assert!(
        result.is_err(),
        "public-state malleability: stateful verify_step accepted the noncanonical semantic word p as the canonical zero digest ({result:?})"
    );
}

/// The Initial/NoFold branch has one canonical verifier-owned base state.
/// Native `verify_step` must reject alternate anchors and a nonempty incoming
/// accumulator instead of silently preserving or overwriting them.
#[test]
fn native_verify_step_rejects_noncanonical_base_state_anchors() {
    let (prep, base, claims, step) = stateful_zero_base_step_fixture();
    verify_stateful_base_step(&prep, base.clone(), &claims, &step).expect("canonical base state verifies");

    let mut wrong_acc = base.clone();
    wrong_acc.acc_digest = [1u8; 32];
    let acc_result = verify_stateful_base_step(&prep, wrong_acc, &claims, &step);

    let mut wrong_z_0 = base.clone();
    wrong_z_0.z_0 = [1u8; 32];
    wrong_z_0.z_i = wrong_z_0.z_0;
    let z_0_result = verify_stateful_base_step(&prep, wrong_z_0, &claims, &step);

    let mut wrong_initial_semantic = base.clone();
    wrong_initial_semantic.initial_semantic_state_digest = [1u8; 32];
    let semantic_result = verify_stateful_base_step(&prep, wrong_initial_semantic, &claims, &step);

    let mut wrong_current_semantic = base.clone();
    wrong_current_semantic.semantic_state_digest = [1u8; 32];
    let current_semantic_result = verify_stateful_base_step(&prep, wrong_current_semantic, &claims, &step);

    let mut wrong_public_trace = base;
    wrong_public_trace.public_trace = [1u8; 32];
    let public_trace_result = verify_stateful_base_step(&prep, wrong_public_trace, &claims, &step);
    let accepted_wrong_acc = acc_result.is_ok();
    let accepted_wrong_z_0 = z_0_result.is_ok();
    let accepted_wrong_initial_semantic = semantic_result.is_ok();
    let accepted_wrong_current_semantic = current_semantic_result.is_ok();
    let accepted_wrong_public_trace = public_trace_result.is_ok();

    assert!(
        !accepted_wrong_acc
            && !accepted_wrong_z_0
            && !accepted_wrong_initial_semantic
            && !accepted_wrong_current_semantic
            && !accepted_wrong_public_trace,
        "native/recursive base-language mismatch: verify_step accepted noncanonical Initial/NoFold inputs (acc={accepted_wrong_acc}, z_0={accepted_wrong_z_0}, initial_semantic={accepted_wrong_initial_semantic}, current_semantic={accepted_wrong_current_semantic}, public_trace={accepted_wrong_public_trace})"
    );
}

fn stateless_zero_active_step_fixture() -> (R1cs, neo_fold_clean::Preprocessing, State) {
    let r1cs = R1cs {
        a: NeoMat::zero(1, neo_math::D, F::ZERO),
        b: NeoMat::zero(1, neo_math::D, F::ZERO),
        c: NeoMat::zero(1, neo_math::D, F::ZERO),
        m_in: 1,
    };
    let prep =
        neo_fold_clean::frontends::direct_ccs::preprocess_seeded(&r1cs, 0xF17E_6042).expect("zero-R1CS preprocessing");
    let empty_acc = AccumulatorHandle::empty().digest();
    let base = State::base(
        initial_boundary_digest(prep.structure_digest(), prep.public_input_len),
        public_trace_seed_digest(prep.structure_digest()),
        empty_acc,
        empty_acc,
    );
    let first = neo_fold_clean::frontends::direct_ccs::build_instance(&prep, &r1cs, &vec![F::ZERO; neo_math::D])
        .expect("satisfying zero-R1CS instance");
    let (active, _) = neo_fold_clean::paper::construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        base,
        vec![first],
    )
    .expect("establish an Active state");
    (r1cs, prep, active)
}

/// Construction-2 counters are exact integer state, while the recursive F'
/// language requires their 64-bit words to be canonical Goldilocks values.
/// The native transition must enforce a compatible range and checked advance
/// instead of wrapping in an optimized build and returning an Active state
/// with a zero counter that the recursive language cannot represent.
#[test]
fn native_verify_step_rejects_step_count_wraparound() {
    let (r1cs, prep, mut near_wrap) = stateless_zero_active_step_fixture();
    near_wrap.step_count = u64::MAX;

    let next = neo_fold_clean::frontends::direct_ccs::build_instance(&prep, &r1cs, &vec![F::ZERO; neo_math::D])
        .expect("satisfying zero-R1CS instance");
    let next_claims = vec![next.claim.clone()];
    let (wrapped_post, step) = neo_fold_clean::paper::construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        near_wrap.clone(),
        vec![next],
    )
    .expect("native prover accepted the near-wrap pre-state");
    assert_eq!(
        wrapped_post.step_count, 0,
        "attack precondition: release addition wrapped"
    );

    let result = neo_fold_clean::paper::construction2::verify_step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        near_wrap,
        &next_claims,
        &step,
        SemanticStateMode::Stateless,
        None,
    );
    let accepted_counters = result
        .as_ref()
        .ok()
        .map(|state| (state.chunk_count, state.step_count));
    assert!(
        result.is_err(),
        "native/recursive language mismatch: verify_step accepted a transition whose step_count wrapped from u64::MAX to zero; accepted counters={accepted_counters:?}"
    );
}

/// The NIFS Fiat-Shamir prefix and F' chunk digest absorb counters as one
/// Goldilocks element, while the public x_out hash retains their full 64-bit
/// halves. Consequently, a recursive fold proof can be replayed at counters
/// shifted by the field modulus after changing only the public deterministic
/// x_out. Native verification must reject the noncanonical source state.
#[test]
fn native_recursive_step_rejects_counter_modulus_fold_replay() {
    let (r1cs, prep, honest_pre) = stateless_zero_active_step_fixture();
    let build_instance = || {
        neo_fold_clean::frontends::direct_ccs::build_instance(&prep, &r1cs, &vec![F::ZERO; neo_math::D])
            .expect("satisfying zero-R1CS instance")
    };

    let honest_next = build_instance();
    let next_claims = vec![honest_next.claim.clone()];
    let (_honest_post, honest_step) = neo_fold_clean::paper::construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        honest_pre.clone(),
        vec![honest_next],
    )
    .expect("honest recursive step");

    let mut aliased_pre = honest_pre;
    aliased_pre.chunk_count = aliased_pre
        .chunk_count
        .checked_add(F::ORDER_U64)
        .expect("small counter plus Goldilocks modulus fits u64");
    aliased_pre.step_count = aliased_pre
        .step_count
        .checked_add(F::ORDER_U64)
        .expect("small counter plus Goldilocks modulus fits u64");
    let (_aliased_post, aliased_step) = neo_fold_clean::paper::construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        aliased_pre.clone(),
        vec![build_instance()],
    )
    .expect("native prover accepts field-modulus-shifted counters");

    let replayed = neo_fold_clean::paper::construction2::StepProof {
        fold: honest_step.fold,
        semantic_state_digest: honest_step.semantic_state_digest,
        x_out: aliased_step.x_out,
    };
    let result = neo_fold_clean::paper::construction2::verify_step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        aliased_pre,
        &next_claims,
        &replayed,
        SemanticStateMode::Stateless,
        None,
    );
    let accepted_counters = result
        .as_ref()
        .ok()
        .map(|state| (state.chunk_count, state.step_count));
    assert!(
        result.is_err(),
        "Fiat-Shamir replay failure: native verify_step accepted an honest recursive fold proof at counters shifted by the Goldilocks modulus after only recomputing x_out; accepted counters={accepted_counters:?}"
    );
}

/// In an Active state, `acc_digest` is a derived handle for the carried
/// running accumulator. Recursive F' constrains that equality before replaying
/// NIFS. The exported native step prover/verifier must not accept an arbitrary
/// incoming handle merely because both parties absorb the same lie into their
/// outer transcript and overwrite it after the fold.
#[test]
fn native_recursive_step_rejects_incoming_accumulator_handle_mismatch() {
    let (r1cs, prep, mut forged_pre) = stateless_zero_active_step_fixture();
    let actual_pre_acc = forged_pre.acc_digest;
    forged_pre.acc_digest = [1u8; 32];
    forged_pre.semantic_state_digest = forged_pre.acc_digest;
    assert_ne!(
        forged_pre.acc_digest, actual_pre_acc,
        "attack requires a false incoming accumulator handle"
    );

    let next = neo_fold_clean::frontends::direct_ccs::build_instance(&prep, &r1cs, &vec![F::ZERO; neo_math::D])
        .expect("satisfying zero-R1CS instance");
    let next_claims = vec![next.claim.clone()];
    let (_post, step) = neo_fold_clean::paper::construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        forged_pre.clone(),
        vec![next],
    )
    .expect("native prover accepted the false incoming handle");

    let result = neo_fold_clean::paper::construction2::verify_step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        forged_pre,
        &next_claims,
        &step,
        SemanticStateMode::Stateless,
        None,
    );
    let accepted_post_acc = result.as_ref().ok().map(|state| state.acc_digest);
    assert!(
        result.is_err(),
        "native/recursive language mismatch: verify_step accepted a recursive state whose incoming acc_digest did not match its running accumulator, then returned post acc={accepted_post_acc:?}"
    );
}

/// Stateless recursive F' represents one accumulator coordinate twice on the
/// native State surface. The recursive circuit enforces equality on input;
/// native step/verify_step must enforce the same pre-state invariant rather
/// than accepting a lie and normalizing it away after NIFS.V.
#[test]
fn native_recursive_step_rejects_incoming_stateless_semantic_acc_mismatch() {
    let (r1cs, prep, mut forged_pre) = stateless_zero_active_step_fixture();
    assert_eq!(
        forged_pre.semantic_state_digest, forged_pre.acc_digest,
        "fixture must start from the canonical stateless invariant"
    );
    forged_pre.semantic_state_digest = [1u8; 32];
    assert_ne!(
        forged_pre.semantic_state_digest, forged_pre.acc_digest,
        "attack requires distinct incoming semantic/accumulator coordinates"
    );

    let next = neo_fold_clean::frontends::direct_ccs::build_instance(&prep, &r1cs, &vec![F::ZERO; neo_math::D])
        .expect("satisfying zero-R1CS instance");
    let next_claims = vec![next.claim.clone()];
    let (_post, step) = neo_fold_clean::paper::construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        forged_pre.clone(),
        vec![next],
    )
    .expect("native prover accepted the false incoming stateless semantic coordinate");

    let result = neo_fold_clean::paper::construction2::verify_step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        forged_pre,
        &next_claims,
        &step,
        SemanticStateMode::Stateless,
        None,
    );
    let accepted_post = result
        .as_ref()
        .ok()
        .map(|state| (state.semantic_state_digest, state.acc_digest));
    assert!(
        result.is_err(),
        "native/recursive language mismatch: verify_step accepted semantic_state_digest_in != acc_digest_in, then normalized both post-state coordinates to the derived accumulator; accepted post={accepted_post:?}"
    );
}

/// Verifier-policy bits that change the accepted language must be bound by
/// the verifier-key digest. Otherwise a generic CCS preprocessing and an
/// R1CS-F' preprocessing can share one apparent verifier key even though the
/// former omits the recursive-link check required by HyperNova Construction 2.
#[test]
fn verifier_key_digest_binds_f_prime_recursive_link_policy() {
    let mut r1cs = fibonacci_transition_r1cs();
    r1cs.m_in = 0;
    let mut plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let state_x_out = plan
        .state_x_out
        .as_mut()
        .expect("tiny lifecycle plan carries state_x_out");
    state_x_out.app_public_input_var_indices.clear();
    state_x_out.app_public_input_bit_var_indices.clear();
    state_x_out.semantic_state_in_var_indices.clear();
    state_x_out.semantic_state_out_var_indices.clear();
    state_x_out.initial_semantic_state_digest_anchor = None;

    let strong = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6021)
        .expect("F' preprocessing");
    assert!(strong.prep.enforces_f_prime_recursive_link());
    assert!(matches!(
        strong.prep.semantic_state_mode(),
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateless
    ));

    let weak = neo_fold_clean::lifecycle::preprocess(
        strong.prep.params.clone(),
        strong.prep.structure().clone(),
        strong.prep.public_input_len,
    )
    .expect("generic preprocessing over the same verifier-owned artifacts");
    assert!(!weak.enforces_f_prime_recursive_link());
    assert_eq!(
        weak.vk.digest(),
        strong.prep.vk.digest(),
        "attack requires a verifier-key digest collision across policies"
    );

    // Build a locally satisfying recursive F' source image rooted at an
    // invented prior state. The generic CCS context has no recursive-link
    // policy and therefore treats this public input as an ordinary valid CCS
    // instance. The F' context correctly rejects it because it does not encode
    // the lifecycle verifier's actual pre-final state x_out.
    let mut forged_ctx = start_chain(&strong).expect("start malicious low-level compiler context");
    forged_ctx.chain_state.chunk_count = 7;
    forged_ctx.chain_state.step_count = 7;
    let shared = assemble_shared_chunk_traces(
        &forged_ctx,
        false,
        neo_fold_clean::paper::digest::AccumulatorHandle::empty().digest_fields(),
        1,
    );
    let assembly = assemble_step_from_shared(&shared, &forged_ctx, &[], None);
    let ce_shape = match strong
        .plan()
        .nifs_payload_shapes
        .first()
        .expect("plan carries canonical CE shape")
    {
        NifsPayloadShape::CeClaim(shape) => shape.clone(),
        NifsPayloadShape::CcsClaim(_) => panic!("tiny lifecycle plan must carry a CE shape"),
    };
    let forged_encoded = encode_r1cs_f_prime_step(
        R1csEncoderInput {
            plan: strong.plan().clone(),
            boundary_bits: assembly.boundary_bits,
            state_in: assembly.state_in,
            state_out: assembly.state_out,
            chunk_digest: assembly.chunk_digest,
            assignment_bits: assignment_to_bits(&fibonacci_assignment(1, 1)),
            is_base: false,
            nifs_payloads: nifs_payload_inputs_for_source_image(strong.plan(), perp_nifs_ce_view(&ce_shape)),
            kmul_views: vec![],
            ring_action_pairs: vec![],
            one_shot_traces: vec![assembly.traces.state_x_out],
            sponge_trace: None,
        },
        std::sync::Arc::clone(strong.structure()),
    );
    let forged_instance =
        r1cs_f_prime::build_instance(&strong, &forged_encoded).expect("build locally valid forged instance");

    let audit = neo_fold_clean::lifecycle::prove(&weak, [vec![forged_instance]])
        .expect("generic context proves the locally valid CCS instance");
    let proof = neo_fold_clean::finish_uncompressed(&weak, audit).expect("generic context finalizes proof");
    assert!(
        neo_fold_clean::verify_uncompressed(&weak, &proof).is_ok(),
        "generic context must accept the proof to demonstrate the language difference"
    );
    assert!(
        neo_fold_clean::verify_uncompressed(&strong.prep, &proof).is_err(),
        "F' context must reject the invented recursive link"
    );

    assert_ne!(
        weak.vk.digest(),
        strong.prep.vk.digest(),
        "soundness vulnerability: verifier contexts with different accepted languages share one verifier-key digest"
    );
}

/// A third-party verifier must reject a recursive F' image whose private
/// semantic input is disconnected from the prior step's public output.
///
/// The honest compiler rejects this construction, so the test uses the
/// public low-level image encoder and lifecycle APIs exactly as a malicious
/// proof producer can. Both app transitions satisfy the verifier-owned R1CS;
/// only their missing recursive link is false:
///
/// ```text
/// (1, 1) -> (1, 2)
/// (10, 10) -> (10, 20)
/// ```
#[test]
fn audit_verifier_rejects_disconnected_recursive_f_prime_semantic_input() {
    let r1cs = fibonacci_transition_r1cs();
    let initial = semantic_digest_for_pair(1, 1);
    let first_out = semantic_digest_for_pair(1, 2);
    let forged_in = semantic_digest_for_pair(10, 10);
    let forged_out = semantic_digest_for_pair(10, 20);
    assert_ne!(first_out, forged_in, "attack requires a disconnected state link");

    let plan =
        make_tiny_stateful_lifecycle_plan_with_anchor(r1cs.m(), r1cs.m_in, vec![1, 2], vec![3, 4], Some(initial));
    let prep =
        r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF17E_6009).expect("preprocess");

    // Produce the honest base image and deposit it as the first pending
    // instance in the Construction-2 state.
    let mut base_ctx = start_chain(&prep).expect("start base compiler");
    let first = compile_step(
        &prep,
        &mut base_ctx,
        R1csFPrimeStepInput {
            assignment: fibonacci_assignment(1, 1),
        },
    )
    .expect("compile base transition");
    let first_instance = r1cs_f_prime::build_instance(&prep, &first.encoded).expect("build base instance");
    let audit_after_first = neo_fold_clean::lifecycle::prove::prove_one_with_semantic_state(
        &prep.prep,
        vec![first_instance.clone()],
        initial,
        first_out,
    )
    .expect("prove base step");

    // Derive the legitimate recursive fold over the *previous* latest.
    // The next deposit is a same-shape placeholder because the F' chunk
    // digest intentionally binds only shape/count, not claim contents.
    let mut forged_audit = neo_fold_clean::lifecycle::prove::extend_with_semantic_state(
        &prep.prep,
        audit_after_first.clone(),
        vec![first_instance],
        forged_out,
    )
    .expect("derive recursive fold with same-shape placeholder");

    let pre_state = &audit_after_first.proof.state;
    let post_state = &forged_audit.proof.state;

    // Hand-roll a locally satisfying recursive source image. Its public
    // post-state agrees with the lifecycle walk, but its private state-in
    // lanes bind H(10,10), not the prior step's H(1,2).
    let mut forged_ctx = start_chain(&prep).expect("start malicious low-level compiler context");
    forged_ctx.chain_state = r1cs_f_prime::R1csChainState {
        chunk_count: pre_state.chunk_count,
        step_count: pre_state.step_count,
        z_i: digest32_as_fields(pre_state.z_i),
        semantic_state_digest: digest32_as_fields(forged_in),
        acc_digest: digest32_as_fields(pre_state.acc_digest),
        public_trace: digest32_as_fields(pre_state.public_trace),
    };

    let shared = assemble_shared_chunk_traces(&forged_ctx, false, digest32_as_fields(post_state.acc_digest), 1);
    let assembly = assemble_step_from_shared(&shared, &forged_ctx, &[], Some(digest32_as_fields(forged_out)));
    let assignment = fibonacci_assignment(10, 10);
    let ce_shape = match prep
        .plan()
        .nifs_payload_shapes
        .first()
        .expect("plan carries canonical CE shape")
    {
        NifsPayloadShape::CeClaim(shape) => shape.clone(),
        NifsPayloadShape::CcsClaim(_) => panic!("stateful test plan must carry a CE shape"),
    };
    let forged_encoded = encode_r1cs_f_prime_step(
        R1csEncoderInput {
            plan: prep.plan().clone(),
            boundary_bits: assembly.boundary_bits,
            state_in: assembly.state_in,
            state_out: assembly.state_out,
            chunk_digest: assembly.chunk_digest,
            assignment_bits: assignment_to_bits(&assignment),
            is_base: false,
            nifs_payloads: nifs_payload_inputs_for_source_image(prep.plan(), perp_nifs_ce_view(&ce_shape)),
            kmul_views: vec![],
            ring_action_pairs: vec![],
            one_shot_traces: vec![
                encode_poseidon_trace(&build_semantic_state_preimage_fields(&[
                    F::from_u64(10),
                    F::from_u64(10),
                ])),
                encode_poseidon_trace(&build_semantic_state_preimage_fields(&[
                    F::from_u64(10),
                    F::from_u64(20),
                ])),
                assembly.traces.state_x_out,
            ],
            sponge_trace: None,
        },
        std::sync::Arc::clone(prep.structure()),
    );
    let forged_instance =
        r1cs_f_prime::build_instance(&prep, &forged_encoded).expect("build locally valid disconnected instance");

    // Replace only the placeholder deposit. This is the same operation the
    // production chain builder performs after deriving a shape-only fold.
    match &mut forged_audit.proof.state.proof {
        ProofState::Active { latest, .. } => {
            *latest = LatestInstance::from_instances(vec![forged_instance.clone()]);
        }
        ProofState::Initial => panic!("recursive extend must produce an active state"),
    }
    *forged_audit
        .public_batches
        .last_mut()
        .expect("recursive extend appends one public batch") = vec![forged_instance.claim];

    let finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep.prep, forged_audit)
        .expect("a locally valid disconnected image reaches finalization");
    let result = neo_fold_clean::verify_uncompressed_audit(&prep.prep, &finalized);
    assert!(
        result.is_err(),
        "soundness vulnerability: the audit verifier accepted two locally valid F' transitions with a disconnected semantic-state link"
    );
}

fn fibonacci_transition_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(2, m, F::default());
    a[(0, 2)] = F::ONE;
    a[(1, 1)] = F::ONE;
    a[(1, 2)] = F::ONE;
    let mut b = NeoMat::zero(2, m, F::default());
    b[(0, 0)] = F::ONE;
    b[(1, 0)] = F::ONE;
    let mut c = NeoMat::zero(2, m, F::default());
    c[(0, 3)] = F::ONE;
    c[(1, 4)] = F::ONE;
    R1cs { a, b, c, m_in: 5 }
}

fn fibonacci_assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; neo_math::D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(b);
    z[4] = F::from_u64(a + b);
    z
}

fn semantic_digest_for_pair(a: u64, b: u64) -> [u8; 32] {
    let fields = [F::from_u64(a), F::from_u64(b)];
    digest_fields_as_digest32(encode_poseidon_trace(&build_semantic_state_preimage_fields(&fields)).digest_native)
}
