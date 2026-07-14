//! Generic `enc(R1CS)` oracle and derived-encoding differential tests.

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_mul, KLc, KVar};
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::ring_action::enforce_ring_mul_toom3;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, R1csEncodingTrace, R1csSnapshot, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, estimate_r1cs_gadget_native, estimate_selector_gated_r1cs_gadget_native,
    GadgetNativeError,
};
use neo_fold_clean::frontends::f_prime::low_norm_r1cs::{
    encode_r1cs_derived, encode_r1cs_oracle, estimate_r1cs_encoding, LowNormR1csEncodingKind, LowNormR1csError,
};
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
    assert_eq!(instance.witness.w, encoded.private_witness());
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
