//! Rectangular SplitNc regression for the selective low-norm compiler.

use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_width_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn selective_poseidon_lowering_is_rectangular_and_binds_alignment_tail() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for arm in 0..3u64 {
        let mut builder = R1csBuilder::new();
        let input = core::array::from_fn(|lane| builder.alloc(F::from_u64(arm * 17 + lane as u64 + 1)));
        let output = enforce_poseidon2_permutation(&mut builder, &input);
        let output_bits = decompose_var_to_u64_bits(&mut builder, output[0]);
        let lowered = lower_field_r1cs(builder, &[output_bits[0]]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let audit =
        audit_multi_branch_selective_low_norm_width_with_alignment(&shapes, 0, D, 0).expect("selective width audit");
    assert!(audit
        .arms
        .iter()
        .all(|arm| arm.traces.poseidon2_columns == 87));

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("selective relation");
    let structure = relation.structure();
    assert!(
        structure.n < structure.m,
        "SplitNc must preserve the semantic-row domain"
    );
    assert_eq!(structure.t(), 13, "only semantic matrices belong in the relation");
    assert!(
        !structure.matrices[0].is_identity(),
        "the NC identity matrix is obsolete"
    );
    assert_eq!(structure.m, audit.total_coordinates.next_multiple_of(D));

    let alignment_tail = audit.total_coordinates..structure.m;
    assert!(!alignment_tail.is_empty(), "fixture must exercise ring alignment");
    for arm in 0..3 {
        let encoded = relation.encode(arm, &assignments[arm]).expect("encode arm");
        assert!(relation.is_satisfied(&encoded), "arm {arm} must satisfy the relation");
        assert!(encoded[alignment_tail.clone()]
            .iter()
            .all(|value| *value == F::ZERO));

        let mut padding_tamper = encoded.clone();
        padding_tamper[alignment_tail.start] = F::ONE;
        assert!(
            !relation.is_satisfied(&padding_tamper),
            "alignment tail must be zero-bound"
        );

        let mut output_tamper = encoded;
        output_tamper[1] = if output_tamper[1] == F::ZERO { F::ONE } else { F::ZERO };
        assert!(
            !relation.is_satisfied(&output_tamper),
            "public Poseidon output must be bound"
        );
    }
}
