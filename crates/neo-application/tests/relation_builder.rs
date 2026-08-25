use neo_application::{
    ApplicationRelation, ApplicationRelationError, ColumnRegistry, ColumnRegistryError, ColumnSpec, ColumnWidth,
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
    specs: pub MACRO_TEST_SPECS,
    indices: pub,
    columns: [
        MACRO_FLAG: Boolean => "test flag",
        MACRO_BYTES: [Byte; 3] => "test bytes",
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Domain,
    Transition,
}

fn test_columns() -> ColumnRegistry {
    ColumnRegistry::new([
        ColumnSpec {
            region: "system",
            start: ONE,
            len: 1,
            name: "ONE",
            role: "verifier-supplied constant",
            width: ColumnWidth::Field,
        },
        ColumnSpec {
            region: "state",
            start: BIT,
            len: 1,
            name: "BIT",
            role: "transition selector",
            width: ColumnWidth::Boolean,
        },
        ColumnSpec {
            region: "input",
            start: FACTOR,
            len: 1,
            name: "FACTOR",
            role: "private factor",
            width: ColumnWidth::Field,
        },
        ColumnSpec {
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
fn exported_macro_declares_contiguous_indices_and_specs() {
    assert_eq!(MACRO_TEST_WIDTH, 4);
    assert_eq!(MACRO_FLAG, 0);
    assert_eq!(MACRO_BYTES, [1, 2, 3]);
    assert_eq!(MACRO_TEST_SPECS[1].start, 1);
    assert_eq!(MACRO_TEST_SPECS[1].len, 3);
    assert_eq!(MACRO_TEST_SPECS[1].width, ColumnWidth::Byte);

    let registry = ColumnRegistry::new(MACRO_TEST_SPECS.iter().cloned()).expect("valid generated registry");
    assert_eq!(registry.column_count(), MACRO_TEST_WIDTH);
    assert_eq!(registry.spec_for_column(0).map(|spec| spec.name), Some("MACRO_FLAG"));
    assert_eq!(registry.spec_for_column(1).map(|spec| spec.name), Some("MACRO_BYTES"));
    assert_eq!(registry.spec_for_column(3).map(|spec| spec.name), Some("MACRO_BYTES"));
    assert_eq!(registry.spec_for_column(4), None);
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
fn rejects_ambiguous_layouts_and_out_of_range_terms() {
    let layout_error = ColumnRegistry::new([
        ColumnSpec {
            region: "system",
            start: 0,
            len: 1,
            name: "ONE",
            role: "",
            width: ColumnWidth::Field,
        },
        ColumnSpec {
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
