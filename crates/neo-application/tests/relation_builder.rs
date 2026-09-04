use neo_application::{
    ApplicationRelation, ApplicationRelationError, ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth,
    ConstraintTag, R1csBuildError, R1csBuilder, R1csSide,
};
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ONE: usize = 0;
const BIT: usize = 1;
const FACTOR: usize = 2;
const PRODUCT: usize = 3;

neo_application::define_column_region! {
    region: "macro_test",
    start: 0,
    width: pub MACRO_TEST_WIDTH,
    families: pub MACRO_TEST_FAMILIES,
    indices: pub,
    columns: [
        MACRO_FLAG: Boolean => "test flag",
        MACRO_BYTES: [Byte; 3] => "test bytes",
        MACRO_TWO_BITS: (Bits(2)) => "test two-bit value",
        MACRO_NIBBLES: [(Bits(4)); 2] => "test nibbles",
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Domain,
    Transition,
}

fn test_columns() -> ColumnRegistry {
    ColumnRegistry::new([
        ColumnFamilySpec {
            region: "system",
            start: ONE,
            len: 1,
            name: "ONE",
            role: "verifier-supplied constant",
            width: ColumnWidth::Field,
        },
        ColumnFamilySpec {
            region: "state",
            start: BIT,
            len: 1,
            name: "BIT",
            role: "transition selector",
            width: ColumnWidth::Boolean,
        },
        ColumnFamilySpec {
            region: "input",
            start: FACTOR,
            len: 1,
            name: "FACTOR",
            role: "private factor",
            width: ColumnWidth::Field,
        },
        ColumnFamilySpec {
            region: "state",
            start: PRODUCT,
            len: 1,
            name: "PRODUCT",
            role: "selected product",
            width: ColumnWidth::Field,
        },
    ])
    .expect("valid test column registry")
}

#[test]
fn exported_macro_declares_contiguous_indices_and_families() {
    assert_eq!(MACRO_TEST_WIDTH, 7);
    assert_eq!(MACRO_FLAG, 0);
    assert_eq!(MACRO_BYTES, [1, 2, 3]);
    assert_eq!(MACRO_TWO_BITS, 4);
    assert_eq!(MACRO_NIBBLES, [5, 6]);
    assert_eq!(MACRO_TEST_FAMILIES[1].start, 1);
    assert_eq!(MACRO_TEST_FAMILIES[1].len, 3);
    assert_eq!(MACRO_TEST_FAMILIES[1].width, ColumnWidth::Byte);
    assert_eq!(MACRO_TEST_FAMILIES[2].width, ColumnWidth::Bits(2));
    assert_eq!(MACRO_TEST_FAMILIES[3].len, 2);
    assert_eq!(MACRO_TEST_FAMILIES[3].width, ColumnWidth::Bits(4));

    let registry = ColumnRegistry::new(MACRO_TEST_FAMILIES.iter().copied()).expect("valid generated registry");
    assert_eq!(registry.column_count(), MACRO_TEST_WIDTH);
    assert_eq!(
        registry.family_for_column(0).map(|family| family.name),
        Some("MACRO_FLAG")
    );
    assert_eq!(
        registry.family_for_column(1).map(|family| family.name),
        Some("MACRO_BYTES")
    );
    assert_eq!(
        registry.family_for_column(3).map(|family| family.name),
        Some("MACRO_BYTES")
    );
    assert_eq!(
        registry.family_for_column(4).map(|family| family.name),
        Some("MACRO_TWO_BITS")
    );
    assert_eq!(
        registry.family_for_column(6).map(|family| family.name),
        Some("MACRO_NIBBLES")
    );
    assert_eq!(registry.family_for_column(7), None);
}

#[test]
fn column_registry_rejects_invalid_custom_bit_widths() {
    for bits in [0, 64] {
        let error = ColumnRegistry::new([ColumnFamilySpec {
            region: "invalid",
            start: 0,
            len: 1,
            name: "INVALID_BITS",
            role: "invalid custom-width value",
            width: ColumnWidth::Bits(bits),
        }])
        .expect_err("custom widths outside 1..=63 must be rejected");
        assert_eq!(
            error,
            ColumnRegistryError::InvalidBitWidth {
                name: "INVALID_BITS",
                bits,
            }
        );
    }
}

#[test]
fn builds_ccs_and_catalog_from_the_same_tagged_rows() {
    let columns = test_columns();
    let mut builder = R1csBuilder::new(columns.column_count(), 1, ONE).expect("valid builder layout");
    builder.with_tag(ConstraintTag::new("selector is boolean", Owner::Domain), |domain| {
        domain.push_boolean(BIT);
        domain.with_tag(
            ConstraintTag::new("selected multiplication", Owner::Transition),
            |transition| {
                transition.push_row([(BIT, F::ONE)], [(FACTOR, F::ONE)], [(PRODUCT, F::ONE)]);
            },
        );
    });
    {
        let mut equality = builder.tagged(ConstraintTag::new("selected value", Owner::Domain));
        equality.push_linear_zero([(PRODUCT, F::ONE), (FACTOR, -F::ONE)]);
    }

    let r1cs = builder.build().expect("valid R1CS relation");
    let relation = ApplicationRelation::new(r1cs, columns).expect("complete application relation");
    assert_eq!(relation.r1cs().structure().n, 3);
    assert_eq!(relation.r1cs().structure().m, 4);
    assert!(relation
        .r1cs()
        .structure()
        .matrices
        .iter()
        .all(|matrix| matrix.as_csc().is_some()));
    assert_eq!(relation.columns().column_count(), 4);
    assert_eq!(relation.r1cs().public_input_count(), 1);
    assert_eq!(relation.r1cs().const_one_column(), ONE);
    assert_eq!(relation.r1cs().catalog().len(), relation.r1cs().structure().n);

    let rows = relation.r1cs().catalog().rows();
    assert_eq!(rows[0].tag().label(), "selector is boolean");
    assert_eq!(rows[0].tag().owner(), &Owner::Domain);
    assert_eq!(rows[1].tag().owner(), &Owner::Transition);
    assert_eq!(rows[1].row().a_terms(), &[(BIT, F::ONE)]);
    assert_eq!(rows[1].row().b_terms(), &[(FACTOR, F::ONE)]);
    assert_eq!(rows[1].row().c_terms(), &[(PRODUCT, F::ONE)]);
    assert_eq!(rows[2].tag().label(), "selected value");
    assert_eq!(rows[2].tag().owner(), &Owner::Domain);

    let public = [F::ONE];
    let witness = [F::ONE, F::from_u64(7), F::from_u64(7)];
    check_ccs_rowwise_zero(relation.r1cs().structure(), &public, &witness).expect("satisfied relation");

    let bad_witness = [F::ONE, F::from_u64(7), F::from_u64(6)];
    assert!(check_ccs_rowwise_zero(relation.r1cs().structure(), &public, &bad_witness).is_err());
}

#[test]
fn gated_linear_zero_only_enforces_the_expression_when_active() {
    let columns = test_columns();
    let mut builder = R1csBuilder::new(columns.column_count(), 1, ONE).expect("valid builder layout");
    builder.with_tag(ConstraintTag::new("gated equality", Owner::Transition), |b| {
        b.push_boolean(BIT);
        b.push_gated_linear_zero(BIT, [(PRODUCT, F::ONE), (FACTOR, -F::ONE)]);
    });

    let relation = ApplicationRelation::new(builder.build().expect("valid R1CS relation"), columns)
        .expect("complete application relation");
    let public = [F::ONE];

    check_ccs_rowwise_zero(
        relation.r1cs().structure(),
        &public,
        &[F::ZERO, F::from_u64(7), F::from_u64(6)],
    )
    .expect("an inactive gate leaves the expression unconstrained");
    check_ccs_rowwise_zero(
        relation.r1cs().structure(),
        &public,
        &[F::ONE, F::from_u64(7), F::from_u64(7)],
    )
    .expect("an active gate accepts a zero expression");
    assert!(
        check_ccs_rowwise_zero(
            relation.r1cs().structure(),
            &public,
            &[F::ONE, F::from_u64(7), F::from_u64(6)],
        )
        .is_err(),
        "an active gate must reject a nonzero expression"
    );
}

#[test]
fn rejects_ambiguous_layouts_and_out_of_range_terms() {
    let layout_error = ColumnRegistry::new([
        ColumnFamilySpec {
            region: "system",
            start: 0,
            len: 1,
            name: "ONE",
            role: "",
            width: ColumnWidth::Field,
        },
        ColumnFamilySpec {
            region: "state",
            start: 2,
            len: 1,
            name: "VALUE",
            role: "",
            width: ColumnWidth::Field,
        },
    ])
    .expect_err("column gaps must be rejected");
    assert_eq!(
        layout_error,
        ColumnRegistryError::NonContiguous {
            name: "VALUE",
            expected_start: 1,
            actual_start: 2,
        }
    );

    let public_constant_error = R1csBuilder::<Owner>::new(test_columns().column_count(), 0, ONE)
        .expect_err("the constant-one column must be verifier-supplied");
    assert_eq!(
        public_constant_error,
        R1csBuildError::ConstantOneNotPublic {
            column: ONE,
            public_input_count: 0,
        }
    );

    let mut builder = R1csBuilder::new(test_columns().column_count(), 1, ONE).expect("valid builder layout");
    builder
        .tagged(ConstraintTag::new("invalid column", Owner::Domain))
        .push_linear_zero([(4, F::ONE)]);
    assert_eq!(
        builder
            .build()
            .expect_err("out-of-range term must be rejected"),
        R1csBuildError::TermOutOfRange {
            row: 0,
            side: R1csSide::A,
            column: 4,
            column_count: 4,
        }
    );

    let mut wider_builder = R1csBuilder::new(5, 1, ONE).expect("valid wider R1CS");
    wider_builder
        .tagged(ConstraintTag::new("fifth-column relation", Owner::Domain))
        .push_linear_zero([(4, F::ONE)]);
    let coverage_error = ApplicationRelation::new(
        wider_builder.build().expect("valid wider R1CS relation"),
        test_columns(),
    )
    .expect_err("application metadata must cover every R1CS column");
    assert_eq!(
        coverage_error,
        ApplicationRelationError::ColumnCountMismatch {
            r1cs_column_count: 5,
            registry_column_count: 4,
        }
    );
}
