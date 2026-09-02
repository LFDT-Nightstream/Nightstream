use super::*;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::frontends::r1cs_f_prime::lower_field_r1cs;
use p3_field::PrimeCharacteristicRing;

fn bit_relation() -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    let public = builder.alloc(F::ONE);
    let first_private = builder.alloc(F::ZERO);
    let second_private = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    enforce_bit(&mut builder, first_private);
    enforce_bit(&mut builder, second_private);
    lower_field_r1cs(builder, &[public])
        .expect("field lowering")
        .into_parts()
        .0
}

#[test]
fn lightweight_shape_matches_complete_audit_exactly() {
    let arm = bit_relation();
    let arms = [arm.clone(), arm.clone(), arm];

    for shared_private_fields in [1, 2] {
        let prepared = super::super::prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms.to_vec(),
            shared_private_fields,
            1,
            D,
            7,
            2,
        )
        .expect("owned prepared relation");
        let summary = prepared.shape_summary();
        let shape =
            audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(&arms, shared_private_fields, 1, D, 7)
                .expect("complete shape audit");

        assert!(summary.matches(&shape));
        assert_eq!(summary.rows, shape.compiler_audit.rows().total_rows());
        assert_eq!(summary.columns, summary.total_coordinates.next_multiple_of(D));

        let relation = prepared.finish().expect("emit prepared relation");
        assert_eq!(
            (relation.structure().n, relation.structure().m),
            (summary.rows, summary.columns)
        );
        assert_eq!(relation.public_input_len(), summary.public_input_len);
        assert_eq!(
            relation
                .selective_compiler_audit()
                .expect("compiler audit")
                .width()
                .total_coordinates,
            summary.total_coordinates,
        );

        let mut wrong_summary = summary;
        wrong_summary.total_coordinates += 1;
        assert!(!wrong_summary.matches(&shape));
    }
}
