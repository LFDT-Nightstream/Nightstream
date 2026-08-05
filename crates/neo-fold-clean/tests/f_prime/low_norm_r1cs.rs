//! Generic `enc(R1CS)` oracle and derived-encoding differential tests.
//!
//! | Boundary | Evidence owned here |
//! |---|---|
//! | Generic derived encoding | Exact source-witness round trip and tamper rejection |
//! | Traced nonlinear gadgets | Exact source-witness round trip through custom CCS gates |
//! | Product-sum batches | Nested K-mul supersession, aggregate rows, and deterministic reconstruction |
//! | First-accepted selection | Product projection, deterministic reconstruction, and output binding |
//! | Boolean row dedup | Exact singleton duplicate removal and same-column near-miss retention |

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_ccs::SparsePoly;
use neo_fold_clean::config;
use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::{enforce_alphabet_sample_5_d, pi_rlc_challenge_stage};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::encoding_trace::{
    BalancedTernaryTraceTestMutation, ProductSumTraceTestMutation,
};
use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_dot_product, enforce_k_mul, KLc, KVar};
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::ring_action::enforce_ring_mul_toom3;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, R1csEncodingTrace, R1csSnapshot, TranscriptGadget, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, estimate_r1cs_gadget_native, estimate_selector_gated_r1cs_gadget_native,
    profile_r1cs_gadget_native_stages, GadgetNativeError, ORDINARY_PRIVATE_DIGITS,
};
use neo_fold_clean::frontends::f_prime::low_norm_r1cs::{
    encode_r1cs_derived, encode_r1cs_oracle, estimate_r1cs_encoding, LowNormR1csEncodingKind, LowNormR1csError,
};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{enforce_commit_fields, SisAccumulatorConfig};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn source_relation() -> (neo_fold_clean::engine::r1cs_circuit::R1csSnapshot, usize, usize, usize) {
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(9));
    let public_bit = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public_bit);
    let private_bit = builder.alloc(F::ONE);
    enforce_bit(&mut builder, private_bit);
    let product = builder.alloc_mul(&Lc::from_var(field), &Lc::from_var(public_bit));
    builder.enforce_eq(&Lc::from_var(product), &Lc::from_var(field));
    assert!(builder.is_satisfied());
    (builder.snapshot(), field.col(), public_bit.col(), private_bit.col())
}

#[test]
fn gadget_native_removes_only_exact_singleton_boolean_source_rows() {
    let mut builder = R1csBuilder::new();
    let bit = builder.alloc(F::ZERO);
    enforce_bit(&mut builder, bit);
    let swapped_bit = builder.alloc(F::ZERO);
    let swapped_minus_one = Lc::from_var(swapped_bit).add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    builder.enforce(&swapped_minus_one, &Lc::from_var(swapped_bit), &Lc::zero());

    // This row references the same relation-proven Boolean column but is not
    // the exact `v * (v - 1) = 0` shape. It is satisfied by the honest zero
    // witness and must remain an ordinary fallback row.
    let near_bit = Lc::from_var(bit).add_scaled(&Lc::from_const(F::from_u64(2)), -F::ONE);
    builder.enforce(&Lc::from_var(bit), &near_bit, &Lc::zero());
    let twice_bit = Lc::zero().add_scaled(&Lc::from_var(bit), F::from_u64(2));
    let bit_minus_one = Lc::from_var(bit).add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    builder.enforce(&twice_bit, &bit_minus_one, &Lc::zero());
    builder.enforce(&Lc::from_var(bit), &bit_minus_one, &Lc::from_var(bit));
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let trace = R1csEncodingTrace::default();
    let public_bits = [bit.col(), swapped_bit.col()];
    let estimate = estimate_r1cs_gadget_native(&source, &trace, &public_bits).expect("Boolean dedup estimate");
    assert_eq!(estimate.source_rows, 5);
    assert_eq!(estimate.redundant_boolean_source_rows, 2);
    assert_eq!(estimate.fallback_source_rows, 3);
    assert_eq!(estimate.encoded_cols, 3);
    assert_eq!(estimate.encoded_rows, 4);

    let mut encoded = encode_r1cs_gadget_native(&source, &trace, &public_bits).expect("Boolean dedup lowering");
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.decode_source().expect("Boolean inverse"), source.witness());

    let bit_slot = encoded
        .plan
        .encoded_range_for_source_column(bit.col())
        .expect("singleton Boolean slot");
    assert_eq!(bit_slot.len(), 1);
    encoded.assignment[bit_slot.start] = F::ONE;
    assert!(
        !encoded.is_satisfied(),
        "same-column near-bit row must remain after exact duplicate removal"
    );
}

#[test]
fn oracle_and_derived_encoding_decode_to_the_same_full_witness() {
    let (source, _field, public_bit, private_bit) = source_relation();
    let oracle = encode_r1cs_oracle(&source, &[public_bit]).expect("oracle encoding");
    let derived = encode_r1cs_derived(&source, &[public_bit]).expect("derived encoding");

    assert!(oracle.is_satisfied());
    assert!(derived.is_satisfied());
    assert_eq!(oracle.decode().expect("oracle inverse"), source.witness());
    assert_eq!(derived.decode().expect("derived inverse"), source.witness());
    assert_eq!(oracle.public_input(), derived.public_input());
    assert_eq!(derived.plan.encoded_bits_for_column(private_bit), Some(1));
    assert_eq!(oracle.plan.encoded_bits_for_column(private_bit), Some(64));
    assert!(derived.plan.is_linearly_derived(source.cols() - 1));
    assert_eq!(derived.plan.encoded_bits_for_column(source.cols() - 1), None);
    assert!(derived.assignment.len() < oracle.assignment.len());
    let estimate =
        estimate_r1cs_encoding(&source, &[public_bit], LowNormR1csEncodingKind::Derived).expect("derived estimate");
    assert_eq!(estimate.encoded_cols, derived.structure.m);
    assert_eq!(estimate.encoded_rows, derived.structure.n);
    assert_eq!(estimate.linearly_derived_source_cols, 1);
}

#[test]
fn encoding_rejects_a_disconnected_source_constraint() {
    let (source, _field, public_bit, _private_bit) = source_relation();
    let mut encoded = encode_r1cs_derived(&source, &[public_bit]).expect("derived encoding");
    let public_range = encoded
        .plan
        .encoded_range_for_column(public_bit)
        .expect("public bit slot");
    encoded.assignment[public_range.start] = F::ZERO;

    assert!(encoded.first_unsatisfied_row().is_some());
    assert!(matches!(
        encoded.decode(),
        Err(LowNormR1csError::UnsatisfiedEncoding { .. })
    ));
}

#[test]
fn oracle_rejects_the_goldilocks_modulus_alias() {
    let (source, field, public_bit, _private_bit) = source_relation();
    let mut encoded = encode_r1cs_oracle(&source, &[public_bit]).expect("oracle encoding");
    let field_range = encoded
        .plan
        .encoded_range_for_column(field)
        .expect("field slot");
    for bit in 0..64 {
        encoded.assignment[field_range.start + bit] = F::from_u64((F::ORDER_U64 >> bit) & 1);
    }
    let aux_range = encoded
        .plan
        .canonical_aux_range_for_column(field)
        .expect("canonical prefix auxiliaries");
    for auxiliary in aux_range {
        encoded.assignment[auxiliary] = F::ONE;
    }

    assert!(
        encoded.first_unsatisfied_row().is_some(),
        "the noncanonical bit string for p must not alias canonical zero"
    );
}

#[test]
fn enc_inst_columns_must_be_relation_proven_bits() {
    let (source, field, _public_bit, _private_bit) = source_relation();
    let error = encode_r1cs_oracle(&source, &[field]).expect_err("raw field cannot be enc_inst");
    assert!(matches!(
        error,
        LowNormR1csError::PublicColumnNotBoolean { column } if column == field
    ));
}

#[test]
fn encoded_relation_crosses_the_superneo_low_norm_commitment_boundary() {
    let (source, _field, public_bit, _private_bit) = source_relation();
    let encoded = encode_r1cs_derived(&source, &[public_bit]).expect("derived encoding");
    let params = neo_fold_clean::config::ccs_params(
        encoded.structure.n,
        encoded.structure.m,
        encoded.structure.t(),
        encoded.structure.max_degree(),
    )
    .expect("shape-compatible SuperNeo params");
    let columns = encoded.structure.m.div_ceil(D);
    if !has_global_pp_for_dims(D, columns) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x454e_435f_5231_4353u64.to_le_bytes());
        set_global_pp_seeded(D, params.kappa() as usize, columns, seed).expect("Ajtai setup");
    }
    let log = AjtaiSModule::from_global_for_dims(D, columns).expect("Ajtai module");
    let instance = encoded
        .to_ccs_instance(&params, &log)
        .expect("all encoded coordinates satisfy b=2");

    assert_eq!(instance.claim.x, encoded.public_input());
    let private = instance
        .witness
        .private_values(instance.claim.m_in, encoded.assignment.len())
        .expect("packed low-norm private witness");
    assert_eq!(private.as_ref(), encoded.private_witness());
}

fn traced_gadget_relation() -> (R1csSnapshot, R1csEncodingTrace, usize, usize) {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();

    let public_bit = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public_bit);
    let poseidon_input: [Var; 8] = std::array::from_fn(|lane| {
        if lane == 0 {
            public_bit
        } else {
            builder.alloc(F::from_u64((lane as u64) * 17 + 3))
        }
    });
    let _poseidon_output = enforce_poseidon2_permutation(&mut builder, &poseidon_input);

    let a = KVar::alloc(&mut builder, F::from_u64(3), F::from_u64(5));
    let b = KVar::alloc(&mut builder, F::from_u64(7), F::from_u64(11));
    let k_output = enforce_k_mul(&mut builder, &KLc::from_var(a), &KLc::from_var(b));

    let rho: [Var; D] = std::array::from_fn(|index| builder.alloc(F::from_u64((index % 7 + 1) as u64)));
    let c: [Var; D] = std::array::from_fn(|index| builder.alloc(F::from_u64((index % 11 + 2) as u64)));
    let _ring_output = enforce_ring_mul_toom3(&mut builder, &rho, &c);

    assert!(builder.is_satisfied());
    (
        builder.snapshot(),
        builder.encoding_trace().clone(),
        public_bit.col(),
        k_output.c0.col(),
    )
}

fn balanced_ternary_relation(value: F) -> (R1csSnapshot, R1csEncodingTrace, usize) {
    const CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
        seed: [0xA7; 32],
        kappa: 1,
        domain: 0x5445_524e_4152_595f,
    };

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let field = builder.alloc(value);
    enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("one-field SIS commitment");
    assert!(builder.is_satisfied());
    (builder.snapshot(), builder.encoding_trace().clone(), field.col())
}

#[test]
fn gadget_native_balanced_ternary_shares_field_digits_and_preserves_the_source_witness() {
    for value in [F::ZERO, F::ONE, -F::ONE, F::from_u64(F::ORDER_U64 / 2)] {
        let (source, trace, field) = balanced_ternary_relation(value);
        assert_eq!(trace.balanced_ternary_openings().len(), 1);

        let estimate = estimate_r1cs_gadget_native(&source, &trace, &[]).expect("balanced-ternary estimate");
        assert_eq!(estimate.balanced_ternary_field_source_cols, 1);
        assert_eq!(estimate.balanced_ternary_alias_source_cols, 41);
        assert_eq!(estimate.balanced_ternary_binary_source_cols, 81);
        assert_eq!(estimate.sis_centered_encoded_cols, 41);
        assert_eq!(
            estimate.centered_encoded_cols,
            estimate.ordinary_private_encoded_cols + estimate.sis_centered_encoded_cols
        );

        let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("balanced-ternary lowering");
        assert!(encoded.is_satisfied());
        assert_eq!(encoded.structure.m, estimate.encoded_cols);
        assert_eq!(encoded.structure.n, estimate.encoded_rows);
        assert_eq!(encoded.decode_source().expect("balanced inverse"), source.witness());
        assert_eq!(
            encoded
                .plan
                .encoded_range_for_source_column(field)
                .expect("balanced source field")
                .len(),
            41
        );

        let opening = &trace.balanced_ternary_openings()[0];
        let field_range = encoded
            .plan
            .encoded_range_for_source_column(field)
            .expect("balanced source field");
        for (digit, &source_digit) in opening.digit_cols.iter().enumerate() {
            assert_eq!(
                encoded.plan.encoded_range_for_source_column(source_digit),
                Some(field_range.start + digit..field_range.start + digit + 1),
            );
        }
    }
}

#[test]
fn gadget_native_balanced_ternary_trace_mutations_fail_closed() {
    let (source, trace, _field) = balanced_ternary_relation(F::from_u64(19));
    let opening = &trace.balanced_ternary_openings()[0];

    let rejected = |mutation| {
        let mut corrupted = trace.clone();
        corrupted.apply_balanced_ternary_trace_test_mutation(0, mutation);
        estimate_r1cs_gadget_native(&source, &corrupted, &[]).expect_err("corrupted balanced-ternary trace")
    };
    for error in [
        rejected(BalancedTernaryTraceTestMutation::FieldColumn { column: 0 }),
        rejected(BalancedTernaryTraceTestMutation::DigitColumn {
            index: 0,
            column: opening.field_col,
        }),
        rejected(BalancedTernaryTraceTestMutation::NegativeColumn { index: 0, column: 0 }),
        rejected(BalancedTernaryTraceTestMutation::BorrowColumn { index: 0, column: 0 }),
        rejected(BalancedTernaryTraceTestMutation::DigitRows { rows: 0..1 }),
        rejected(BalancedTernaryTraceTestMutation::ReconstructionRow { row: 0 }),
        rejected(BalancedTernaryTraceTestMutation::TransitionRows { rows: 0..1 }),
    ] {
        assert!(matches!(error, GadgetNativeError::BalancedTernaryGeometry { .. }));
    }

    let mut swapped = trace.clone();
    swapped.apply_balanced_ternary_trace_test_mutation(
        0,
        BalancedTernaryTraceTestMutation::DigitColumn {
            index: 0,
            column: opening.digit_cols[1],
        },
    );
    swapped.apply_balanced_ternary_trace_test_mutation(
        0,
        BalancedTernaryTraceTestMutation::DigitColumn {
            index: 1,
            column: opening.digit_cols[0],
        },
    );
    assert!(matches!(
        estimate_r1cs_gadget_native(&source, &swapped, &[]).expect_err("row-identity drift"),
        GadgetNativeError::TraceRowMismatch { .. }
    ));

    let mut duplicate = trace.clone();
    duplicate.duplicate_balanced_ternary_trace_for_test(0);
    assert!(matches!(
        estimate_r1cs_gadget_native(&source, &duplicate, &[]).expect_err("duplicate opening"),
        GadgetNativeError::BalancedTernaryGeometry { .. }
    ));
}

#[test]
fn gadget_native_balanced_ternary_omitted_families_are_entailed_or_fail_closed() {
    let (source, trace, field) = balanced_ternary_relation(F::from_u64(23));
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("balanced-ternary lowering");
    let opening = &trace.balanced_ternary_openings()[0];

    // The retained centered-unit gate rejects a non-centered digit.
    let mut digit_tamper = encoded.clone();
    let digit = digit_tamper
        .plan
        .encoded_range_for_source_column(field)
        .expect("balanced field")
        .start;
    digit_tamper.assignment[digit] = F::from_u64(2);
    assert!(digit_tamper.first_unsatisfied_row().is_some());

    // Omitted family 1: negative-indicator bitness follows from the retained
    // centered-unit and negative-definition gates.
    let negative_column = encoded
        .plan
        .encoded_range_for_source_column(opening.negative_cols[0])
        .expect("negative indicator")
        .start;
    let mut negative_bitness_tamper = encoded.clone();
    negative_bitness_tamper.assignment[negative_column] = F::from_u64(2);
    assert!(negative_bitness_tamper.first_unsatisfied_row().is_some());

    // Omitted family 2: internal-borrow bitness follows from the retained
    // transition chain and its fixed zero sentinels.
    let borrow_column = encoded
        .plan
        .encoded_range_for_source_column(opening.borrow_cols[0])
        .expect("borrow indicator")
        .start;
    let mut borrow_bitness_tamper = encoded.clone();
    borrow_bitness_tamper.assignment[borrow_column] = F::from_u64(2);
    assert!(borrow_bitness_tamper.first_unsatisfied_row().is_some());

    // Omitted family 3: choose an honest zero digit and set its negative
    // indicator to one. Bitness still holds, but n(d+1)=0 does not; the
    // retained definition row rejects the assignment.
    let support_index = opening
        .digit_cols
        .iter()
        .position(|&column| source.witness()[column] == F::ZERO)
        .expect("sample has a zero balanced digit");
    assert_eq!(source.witness()[opening.negative_cols[support_index]], F::ZERO);
    let support_digit_column = encoded
        .plan
        .encoded_range_for_source_column(opening.digit_cols[support_index])
        .expect("support digit")
        .start;
    let support_negative_column = encoded
        .plan
        .encoded_range_for_source_column(opening.negative_cols[support_index])
        .expect("support negative indicator")
        .start;
    let mut support_tamper = encoded.clone();
    support_tamper.assignment[support_negative_column] = F::ONE;
    assert_ne!(
        support_tamper.assignment[support_negative_column] * (support_tamper.assignment[support_digit_column] + F::ONE),
        F::ZERO
    );
    assert!(support_tamper.first_unsatisfied_row().is_some());

    // Omitted family 4: reconstruction has no independent target witness.
    // Reassigning its field role to a digit source column makes the structural
    // alias overlap and is rejected before any row can be omitted.
    let mut reconstruction_tamper = trace.clone();
    reconstruction_tamper.apply_balanced_ternary_trace_test_mutation(
        0,
        BalancedTernaryTraceTestMutation::FieldColumn {
            column: opening.digit_cols[0],
        },
    );
    assert!(matches!(
        estimate_r1cs_gadget_native(&source, &reconstruction_tamper, &[]).expect_err("malformed reconstruction alias"),
        GadgetNativeError::BalancedTernaryGeometry { .. }
    ));
}

#[test]
fn gadget_native_balanced_ternary_row_reader_binds_gate_polynomial() {
    let (source, trace, _) = balanced_ternary_relation(F::from_u64(23));
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("balanced-ternary lowering");
    encoded
        .balanced_ternary_rows(0)
        .expect("exact matrix rows and gate polynomial");

    let arity = encoded.structure.f.arity();
    let mut missing_terms = encoded.clone();
    missing_terms.structure.f = SparsePoly::new(arity, Vec::new());
    assert!(matches!(
        missing_terms
            .balanced_ternary_rows(0)
            .expect_err("missing centered/product polynomial terms"),
        GadgetNativeError::BalancedTernaryGeometry { .. }
    ));

    let mut wrong_arity = encoded;
    wrong_arity.structure.f = SparsePoly::new(arity + 1, Vec::new());
    assert!(matches!(
        wrong_arity
            .balanced_ternary_rows(0)
            .expect_err("wrong gate polynomial arity"),
        GadgetNativeError::BalancedTernaryGeometry { .. }
    ));
}

#[test]
fn gadget_native_lowering_is_differentially_equal_to_the_source_r1cs() {
    let (source, trace, public_bit, k_output) = traced_gadget_relation();
    assert_eq!(trace.sbox7().len(), 86);
    assert_eq!(trace.k_muls().len(), 1);
    assert_eq!(trace.ring_muls_toom3().len(), 1);

    let estimate = estimate_r1cs_gadget_native(&source, &trace, &[public_bit]).expect("gadget-native estimate");
    let generic = estimate_r1cs_encoding(&source, &[public_bit], LowNormR1csEncodingKind::Derived)
        .expect("generic derived estimate");
    let fixed =
        estimate_selector_gated_r1cs_gadget_native(&source, &trace, &[public_bit], &source, &trace, &[public_bit])
            .expect("selector-gated estimate");
    assert!(estimate.encoded_cols < generic.encoded_cols);
    assert_eq!(fixed.encoded_cols, 2 * estimate.encoded_cols - 1);
    assert!(fixed.encoded_rows > estimate.encoded_rows);
    assert_eq!(estimate.synthetic_ring_fields, 5 * 35);
    assert_eq!(estimate.max_degree, 8);

    let encoded = encode_r1cs_gadget_native(&source, &trace, &[public_bit]).expect("gadget-native encoding");
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.structure.m, estimate.encoded_cols);
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert_eq!(encoded.structure.max_degree(), estimate.max_degree);
    let params = config::ccs_params(
        encoded.structure.n,
        encoded.structure.m,
        encoded.structure.t(),
        encoded.structure.max_degree(),
    )
    .expect("selected relation parameters");
    let joint_dims = neo_reductions::engines::pi_ccs_joint::build_joint_dims(params.inner(), &encoded.structure, 1, 0)
        .expect("selected one-joint dimensions");
    assert_eq!(
        joint_dims.degree, 9,
        "degree-eight relation needs ten SumCheck coefficients"
    );
    assert_eq!(encoded.decode_source().expect("inverse"), source.witness());
    assert!(encoded
        .plan
        .encoded_range_for_source_column(k_output)
        .is_some());
    for intermediate in trace.sbox7()[0].intermediates {
        assert!(encoded.plan.is_gadget_derived(intermediate.col()));
        assert!(encoded
            .plan
            .encoded_range_for_source_column(intermediate.col())
            .is_none());
    }
}

#[test]
fn gadget_native_constraints_reject_output_and_synthetic_ring_tamper() {
    let (source, trace, public_bit, k_output) = traced_gadget_relation();
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[public_bit]).expect("gadget-native encoding");

    let mut output_tamper = encoded.clone();
    let output_bit = output_tamper
        .plan
        .encoded_range_for_source_column(k_output)
        .expect("K output field slot")
        .start;
    output_tamper.assignment[output_bit] = F::ONE - output_tamper.assignment[output_bit];
    assert!(output_tamper.first_unsatisfied_row().is_some());

    let mut ring_tamper = encoded;
    let ring_bit = ring_tamper
        .plan
        .synthetic_ring_coefficient_range(0, 2, 17)
        .expect("synthetic convolution slot")
        .start;
    ring_tamper.assignment[ring_bit] = F::ONE - ring_tamper.assignment[ring_bit];
    assert!(ring_tamper.first_unsatisfied_row().is_some());
}

#[test]
fn gadget_native_product_sum_batch_supersedes_nested_k_muls_exactly() {
    const WIDTH: usize = 20;

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let lhs = (0..WIDTH)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(index as u64 + 2),
                F::from_u64(index as u64 + 3),
            )
        })
        .collect::<Vec<_>>();
    let rhs = (0..WIDTH)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(index as u64 + 5),
                F::from_u64(index as u64 + 7),
            )
        })
        .collect::<Vec<_>>();
    let output = enforce_k_dot_product(&mut builder, &lhs, &rhs);
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let trace = builder.encoding_trace().clone();
    assert_eq!(trace.k_muls().len(), WIDTH);
    let estimate = estimate_r1cs_gadget_native(&source, &trace, &[]).expect("product-sum estimate");
    assert_eq!(estimate.synthetic_product_sum_fields, 3);

    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("product-sum lowering");
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.structure.m, estimate.encoded_cols);
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert_eq!(encoded.decode_source().expect("product-sum inverse"), source.witness());

    let nested = &trace.k_muls()[0];
    for column in nested
        .intermediates
        .iter()
        .chain(nested.output.iter())
        .map(|variable| variable.col())
    {
        assert!(encoded.plan.is_gadget_derived(column));
        assert!(encoded
            .plan
            .encoded_range_for_source_column(column)
            .is_none());
    }
    assert!(encoded
        .plan
        .encoded_range_for_source_column(output.c0.col())
        .is_some());

    let mut tampered = encoded;
    let output_bit = tampered
        .plan
        .encoded_range_for_source_column(output.c0.col())
        .expect("retained product-sum output")
        .start;
    tampered.assignment[output_bit] = F::ONE - tampered.assignment[output_bit];
    assert!(tampered.first_unsatisfied_row().is_some());
    assert!(matches!(
        tampered.decode_source(),
        Err(GadgetNativeError::UnsatisfiedEncoding { .. })
    ));
}

#[test]
fn gadget_native_product_sum_width_37_uses_two_carries_per_identity() {
    const WIDTH: usize = 37;

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let lhs = (0..WIDTH)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(index as u64 + 2),
                F::from_u64(index as u64 + 3),
            )
        })
        .collect::<Vec<_>>();
    let rhs = (0..WIDTH)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(index as u64 + 41),
                F::from_u64(index as u64 + 43),
            )
        })
        .collect::<Vec<_>>();
    let _output = enforce_k_dot_product(&mut builder, &lhs, &rhs);
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let trace = builder.encoding_trace().clone();
    let estimate = estimate_r1cs_gadget_native(&source, &trace, &[]).expect("three-group product-sum estimate");
    assert_eq!(estimate.synthetic_product_sum_fields, 6);

    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("three-group product-sum lowering");
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.structure.m, estimate.encoded_cols);
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert_eq!(
        encoded
            .decode_source()
            .expect("three-group product-sum inverse"),
        source.witness()
    );
}

#[test]
fn gadget_native_product_sum_trace_mutations_fail_closed() {
    const WIDTH: usize = 20;

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let lhs = (0..WIDTH)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(index as u64 + 11),
                F::from_u64(index as u64 + 13),
            )
        })
        .collect::<Vec<_>>();
    let rhs = (0..WIDTH)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(index as u64 + 17),
                F::from_u64(index as u64 + 19),
            )
        })
        .collect::<Vec<_>>();
    let _output = enforce_k_dot_product(&mut builder, &lhs, &rhs);
    let source = builder.snapshot();
    let trace = builder.encoding_trace().clone();

    let rejected = |mutation| {
        let mut corrupted = trace.clone();
        corrupted.apply_product_sum_trace_test_mutation(0, mutation);
        estimate_r1cs_gadget_native(&source, &corrupted, &[]).expect_err("corrupted product-sum trace")
    };

    assert!(matches!(
        rejected(ProductSumTraceTestMutation::RowEnd { row_end: 0 }),
        GadgetNativeError::ProductSumGeometry { .. }
    ));
    assert!(matches!(
        rejected(ProductSumTraceTestMutation::AllocatedColumn { offset: 0, column: 0 }),
        GadgetNativeError::ProductSumGeometry { .. }
    ));
    assert!(matches!(
        rejected(ProductSumTraceTestMutation::RetainedColumns { columns: Vec::new() }),
        GadgetNativeError::ProductSumGeometry { .. }
    ));
    assert!(matches!(
        rejected(ProductSumTraceTestMutation::CopyIdentity { from: 0, to: 1 }),
        GadgetNativeError::ProductSumRetainedRank { batch: 0 }
    ));
    assert!(matches!(
        rejected(ProductSumTraceTestMutation::FactorCoefficient {
            identity: 0,
            factor: 0,
            coefficient: F::TWO,
        }),
        GadgetNativeError::ProductSumIdentityMismatch { batch: 0, identity: 0 }
    ));
    assert!(matches!(
        rejected(ProductSumTraceTestMutation::ClearResult { identity: 0 }),
        GadgetNativeError::ProductSumIdentityMismatch { batch: 0, identity: 0 }
    ));
}

#[test]
fn gadget_native_product_sum_rejects_a_removed_temporary_escape() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let lhs = [KVar::alloc(&mut builder, F::from_u64(2), F::from_u64(3))];
    let rhs = [KVar::alloc(&mut builder, F::from_u64(5), F::from_u64(7))];
    let _output = enforce_k_dot_product(&mut builder, &lhs, &rhs);
    let removed = builder.encoding_trace().k_muls()[0].output[0];
    let removed_value = builder.witness()[removed.col()];
    builder.enforce_eq(&Lc::from_var(removed), &Lc::from_const(removed_value));
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let error = estimate_r1cs_gadget_native(&source, builder.encoding_trace(), &[])
        .expect_err("a removed product-sum temporary escaped its batch");
    assert!(matches!(
        error,
        GadgetNativeError::GadgetTemporaryEscapes { column } if column == removed.col()
    ));
}

#[test]
fn gadget_native_product_sum_estimator_rejects_a_cross_batch_projected_input() {
    // Before the global dependency check, estimation accepted this exact trace
    // while materialization failed when translating the second batch's LCs.
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let first_lhs = [KVar::alloc(&mut builder, F::from_u64(2), F::from_u64(3))];
    let first_rhs = [KVar::alloc(&mut builder, F::from_u64(5), F::from_u64(7))];
    let _first_output = enforce_k_dot_product(&mut builder, &first_lhs, &first_rhs);
    let first_products = builder.encoding_trace().k_muls()[0].intermediates;

    let projected_input = KVar::new(first_products[0], first_products[1]);
    let second_rhs = [KVar::alloc(&mut builder, F::from_u64(11), F::from_u64(13))];
    let _second_output = enforce_k_dot_product(&mut builder, &[projected_input], &second_rhs);
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let trace = builder.encoding_trace().clone();
    let expected_column = first_products[1].col();
    for error in [
        estimate_r1cs_gadget_native(&source, &trace, &[]).expect_err("estimator must reject unavailable input"),
        encode_r1cs_gadget_native(&source, &trace, &[]).expect_err("materializer must reject unavailable input"),
    ] {
        assert!(matches!(
            error,
            GadgetNativeError::ProductSumUnavailableDependency { batch: 1, column }
                if column == expected_column
        ));
    }
}

#[test]
fn first_accepted_selection_lowering_projects_products_and_reconstructs_the_source() {
    const APP: &[u8] = b"neo.test.selection.lowering/v1";

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.selection");
    let mut transcript = TranscriptGadget::new(&mut builder, APP);
    let outputs = enforce_alphabet_sample_5_d(&mut builder, &mut transcript, 0x5e1ec7);
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let trace = builder.encoding_trace().clone();
    assert_eq!(trace.first_accepted_selections().len(), D);
    for event in trace.first_accepted_selections() {
        assert_eq!(event.one_hot.len(), 11);
        assert_eq!(event.one_hot_rows.len(), 12);
        assert_eq!(event.product_rows.len(), 33);
        assert_eq!(event.bind_rows.len(), 3);
        assert_eq!(event.products.len(), 11);
    }

    let estimate = estimate_r1cs_gadget_native(&source, &trace, &[]).expect("selection estimate");
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("selection lowering");
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.structure.m, estimate.encoded_cols);
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert_eq!(encoded.decode_source().expect("selection inverse"), source.witness());

    let profile = profile_r1cs_gadget_native_stages(&source, &trace, &[]).expect("selection stage profile");
    let binding = profile
        .aggregate_prefix(pi_rlc_challenge_stage::SELECT_BIND)
        .expect("selection binding subtree");
    assert_eq!(binding.selection_accept_aggregate_rows, D);
    assert_eq!(binding.selection_prefix_aggregate_rows, D);
    assert_eq!(binding.selection_symbol_aggregate_rows, D);
    assert_eq!(binding.encoded_cols, D * ORDINARY_PRIVATE_DIGITS);
    assert_eq!(binding.encoded_rows, D * ORDINARY_PRIVATE_DIGITS.div_ceil(2) + 3 * D);
    let stages = profile.aggregate_by_label();
    for (label, encoded_cols, encoded_rows) in [
        (pi_rlc_challenge_stage::SELECT_BIND_ACCEPT, 0, D),
        (pi_rlc_challenge_stage::SELECT_BIND_PREFIX, 0, D),
        (
            pi_rlc_challenge_stage::SELECT_BIND_SYMBOL,
            D * ORDINARY_PRIVATE_DIGITS,
            D * ORDINARY_PRIVATE_DIGITS.div_ceil(2) + D,
        ),
    ] {
        let stage = stages
            .iter()
            .find(|stage| stage.label == label)
            .expect("selection aggregate leaf");
        assert_eq!(stage.encoded_cols, encoded_cols, "{label} columns");
        assert_eq!(stage.encoded_rows, encoded_rows, "{label} rows");
    }

    let first_binding_row = trace.first_accepted_selections()[0].bind_rows.start;
    let mut corrupted_source = source.clone();
    corrupted_source.apply_a_row_test_mutation(first_binding_row, Var::ONE.col(), F::ONE);
    assert!(matches!(
        estimate_r1cs_gadget_native(&corrupted_source, &trace, &[]),
        Err(GadgetNativeError::TraceRowMismatch {
            gadget: "first-accepted selection",
            row,
        }) if row == first_binding_row
    ));

    for event in trace.first_accepted_selections() {
        for products in &event.products {
            for product in [products.symbol, products.accepted, products.prefix] {
                assert!(encoded.plan.is_gadget_derived(product.col()));
                assert!(encoded
                    .plan
                    .encoded_range_for_source_column(product.col())
                    .is_none());
            }
        }
    }

    let mut tampered = encoded;
    let output = outputs[0].col();
    let output_bit = tampered
        .plan
        .encoded_range_for_source_column(output)
        .expect("selected output remains an encoded downstream input")
        .start;
    tampered.assignment[output_bit] = F::ONE - tampered.assignment[output_bit];
    assert!(tampered.first_unsatisfied_row().is_some());
}

#[test]
fn gadget_native_compiler_rejects_a_traced_temporary_that_escapes() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let input: [Var; 8] = std::array::from_fn(|lane| builder.alloc(F::from_u64(lane as u64 + 1)));
    let _output = enforce_poseidon2_permutation(&mut builder, &input);
    let escaped = builder.encoding_trace().sbox7()[0].intermediates[0];
    let escaped_value = builder.witness()[escaped.col()];
    builder.enforce_eq(&Lc::from_var(escaped), &Lc::from_const(escaped_value));
    assert!(builder.is_satisfied());

    let source = builder.snapshot();
    let error = estimate_r1cs_gadget_native(&source, builder.encoding_trace(), &[])
        .expect_err("escaped temporary must stop compilation");
    assert!(matches!(
        error,
        GadgetNativeError::GadgetTemporaryEscapes { column } if column == escaped.col()
    ));
}

#[path = "low_norm_r1cs/acceptance.rs"]
mod acceptance;
#[path = "low_norm_r1cs/coordinate_gates.rs"]
mod coordinate_gates;
#[path = "low_norm_r1cs/mod5.rs"]
mod mod5;

#[path = "low_norm_r1cs/ordinary_private_field.rs"]
mod ordinary_private_field;
