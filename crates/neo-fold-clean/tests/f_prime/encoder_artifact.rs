//! Receipt and corruption checks for the persistent low-norm encoder.

use std::io::Cursor;

use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs, LowNormEncoderArtifactLimits,
    LowNormEncoderArtifactReceipt, MultiBranchLowNormR1cs,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

fn relation() -> MultiBranchLowNormR1cs {
    let mut builder = R1csBuilder::new();
    let value = builder.alloc(F::from_u64(37));
    let bits = decompose_var_to_u64_bits(&mut builder, value);
    let shape = lower_field_r1cs(builder, &[bits[0]])
        .expect("lower artifact fixture")
        .into_parts()
        .0;
    build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("compile artifact fixture")
}

fn limits(receipt: &LowNormEncoderArtifactReceipt) -> LowNormEncoderArtifactLimits {
    let shape = receipt.relation_shape().expect("receipt shape");
    LowNormEncoderArtifactLimits::new(
        receipt.artifact_bytes(),
        shape.0,
        shape.1,
        shape.2,
        receipt.arm_field_counts().len(),
        receipt
            .arm_field_counts()
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize,
        receipt
            .arm_derived_counts()
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize,
    )
}

#[test]
fn encoder_artifact_round_trip_checks_receipt_and_bounds() {
    let relation = relation();
    let matrix_digest = [F::from_u64(11), F::from_u64(12), F::from_u64(13), F::from_u64(14)];
    let mut bytes = Vec::new();
    let receipt = relation
        .write_encoder_artifact(&mut bytes, matrix_digest)
        .expect("write encoder artifact");
    assert_eq!(receipt.artifact_bytes(), bytes.len() as u64);

    let verified =
        MultiBranchLowNormR1cs::read_verified_encoder_artifact(Cursor::new(&bytes), &receipt, limits(&receipt))
            .expect("load exact encoder artifact");
    assert_eq!(verified.receipt(), &receipt);

    let shape = receipt.relation_shape().expect("receipt shape");
    let too_small = MultiBranchLowNormR1cs::read_verified_encoder_artifact(
        Cursor::new(&bytes),
        &receipt,
        LowNormEncoderArtifactLimits::new(
            receipt.artifact_bytes() - 1,
            shape.0,
            shape.1,
            shape.2,
            receipt.arm_field_counts().len(),
            receipt
                .arm_field_counts()
                .iter()
                .copied()
                .max()
                .unwrap_or(0) as usize,
            receipt
                .arm_derived_counts()
                .iter()
                .copied()
                .max()
                .unwrap_or(0) as usize,
        ),
    );
    assert!(too_small.is_err(), "the byte cap must apply before decoding");

    let max_fields = receipt
        .arm_field_counts()
        .iter()
        .copied()
        .max()
        .expect("arm fields") as usize;
    let expansion_too_large = MultiBranchLowNormR1cs::read_verified_encoder_artifact(
        Cursor::new(&bytes),
        &receipt,
        LowNormEncoderArtifactLimits::new(
            receipt.artifact_bytes(),
            shape.0,
            shape.1,
            shape.2,
            receipt.arm_field_counts().len(),
            max_fields - 1,
            receipt
                .arm_derived_counts()
                .iter()
                .copied()
                .max()
                .unwrap_or(0) as usize,
        ),
    );
    assert!(
        expansion_too_large.is_err(),
        "run expansion must be bounded before allocation"
    );

    let mut corrupted = bytes.clone();
    let last = corrupted.last_mut().expect("nonempty artifact");
    *last ^= 1;
    let corrupted =
        MultiBranchLowNormR1cs::read_verified_encoder_artifact(Cursor::new(corrupted), &receipt, limits(&receipt));
    assert!(corrupted.is_err(), "changed bytes must fail the receipt digest");
}

#[test]
fn encoder_artifact_rejects_a_different_matrix_identity() {
    let relation = relation();
    let matrix_digest = [F::from_u64(21), F::from_u64(22), F::from_u64(23), F::from_u64(24)];
    let mut bytes = Vec::new();
    let receipt = relation
        .write_encoder_artifact(&mut bytes, matrix_digest)
        .expect("write encoder artifact");
    let shape = receipt.relation_shape().expect("receipt shape");
    let wrong = LowNormEncoderArtifactReceipt::from_parts(
        receipt.artifact_bytes(),
        (shape.0 as u64, shape.1 as u64, shape.2 as u32),
        [31, 32, 33, 34],
        receipt.encoder_digest(),
        receipt.arm_field_counts().to_vec(),
        receipt.arm_derived_counts().to_vec(),
    )
    .expect("well-formed wrong receipt");
    let loaded = MultiBranchLowNormR1cs::read_verified_encoder_artifact(Cursor::new(bytes), &wrong, limits(&wrong));
    assert!(loaded.is_err(), "matrix substitution must fail");
}
