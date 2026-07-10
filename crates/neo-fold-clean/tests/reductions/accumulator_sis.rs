//! Native/circuit parity and cost pins for candidate C14.

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::{lower_field_r1cs, lower_sparse_r1cs_to_low_norm};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest, commit_fields, enforce_accumulator_digest, enforce_commit_fields, SisAccumulatorConfig,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xA7; 32],
    kappa: 1,
};

#[test]
fn sis_accumulator_matches_native_and_rejects_tampering() {
    let values = [F::from_u64(3), F::from_u64(0x1_0000_0001), -F::ONE];
    let native = commit_fields(CONFIG, &values).expect("native SIS accumulator");
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &fields).expect("SIS accumulator circuit");
    let circuit_data: Vec<F> = commitment
        .data
        .iter()
        .map(|wire| builder.witness()[wire.col()])
        .collect();

    assert_eq!(circuit_data, native.data);
    assert!(builder.is_satisfied());
    assert_eq!(
        builder.cols(),
        258,
        "three fields, canonical bits, and one D-wide output"
    );
    assert_eq!(builder.rows(), 263, "canonical decomposition plus D output equations");
    assert!(
        builder.nonzero_entries() > 10_000,
        "the dense-linear cost must remain visible"
    );

    let input_col = fields[1].col();
    let input = builder.witness()[input_col];
    builder.tamper_witness(input_col, input + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "input mutation must break canonical recomposition"
    );
    builder.tamper_witness(input_col, input);

    let output_col = commitment.data[0].col();
    let output = builder.witness()[output_col];
    builder.tamper_witness(output_col, output + F::ONE);
    assert!(!builder.is_satisfied(), "output mutation must break the Ajtai equation");
}

#[test]
fn sis_accumulator_hash_then_fs_matches_native_and_exposes_cost() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let native = accumulator_digest(CONFIG, &values).expect("native SIS digest");
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let wires = enforce_accumulator_digest(&mut builder, CONFIG, &fields).expect("SIS digest circuit");
    let circuit_digest: [F; 4] = std::array::from_fn(|lane| builder.witness()[wires.digest[lane].col()]);

    assert_eq!(circuit_digest, native);
    assert!(builder.is_satisfied());
    eprintln!(
        "SIS accumulator (3 fields, kappa=1): rows={}, cols={}, nnz={}",
        builder.rows(),
        builder.cols(),
        builder.nonzero_entries()
    );
}

#[test]
fn sis_accumulator_reuses_source_bits_in_low_norm_lowering() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let wires = enforce_accumulator_digest(&mut builder, CONFIG, &fields).expect("SIS digest circuit");
    let lowered = lower_field_r1cs(builder, &wires.digest).expect("field lowering");
    let (shape, assignment) = lowered.into_parts();
    let encoded = lower_sparse_r1cs_to_low_norm(&shape, &assignment).expect("low-norm lowering");

    assert!(encoded.is_satisfied(encoded.assignment()));
    assert_eq!(encoded.structure().n, 671_981, "pin the complete low-norm row cost");
    assert_eq!(
        encoded.structure().m,
        661_445,
        "pin the complete low-norm committed width"
    );
    eprintln!(
        "low-norm SIS accumulator (3 fields, kappa=1): rows={}, committed_bits={}",
        encoded.structure().n,
        encoded.structure().m
    );
}
