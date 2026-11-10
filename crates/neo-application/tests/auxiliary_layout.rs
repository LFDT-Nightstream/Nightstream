use neo_application::{
    AuxiliaryBuilder, ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth, ConstraintTag,
    R1csBuildError, R1csBuilder, ZeroTest,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn base() -> ColumnRegistry {
    ColumnRegistry::new([ColumnFamilySpec {
        region: "base",
        start: 0,
        len: 2,
        name: "INPUT".into(),
        role: "public inputs",
        width: ColumnWidth::Field,
    }])
    .unwrap()
}

#[test]
fn assignment_is_fixed_and_reads_only_prior_families() {
    let mut builder = AuxiliaryBuilder::new(&base());
    let product = builder
        .allocate(
            "test",
            format!("product.{}", 0),
            "product",
            ColumnWidth::U32,
            1,
            |input, output| {
                assert_eq!(input.len(), 2);
                assert_eq!(output.len(), 1);
                output[0] = input[1] * input[1];
                Ok(())
            },
        )
        .unwrap();
    builder
        .allocate(
            "test",
            "derived",
            "derived value",
            ColumnWidth::U32,
            1,
            move |input, output| {
                assert_eq!(input.len(), product.end);
                output[0] = input[product.start] + input[1];
                Ok(())
            },
        )
        .unwrap();
    let layout = builder.finish().unwrap();
    let columns = layout.columns().clone();
    for value in [0, 1, 17, 65535] {
        let mut witness = vec![F::ONE, F::from_u64(value), F::from_u64(999), F::from_u64(999)];
        layout.assign(&mut witness).unwrap();
        assert_eq!(&witness[..2], &[F::ONE, F::from_u64(value)]);
        assert_eq!(witness[2], F::from_u64(value * value));
        assert_eq!(witness[3], F::from_u64(value * value + value));
        assert_eq!(layout.columns(), &columns);
    }
    assert!(layout.assign(&mut [F::ZERO; 3]).is_err());
    assert!(layout.assign(&mut [F::ZERO; 5]).is_err());
}

#[test]
fn named_allocation_rejects_duplicate_and_empty_families() {
    let mut builder = AuxiliaryBuilder::new(&base());
    builder
        .allocate("test", "INPUT", "collision", ColumnWidth::Boolean, 1, |_, out| {
            out[0] = F::ZERO;
            Ok(())
        })
        .unwrap();
    assert!(matches!(
        builder.finish(),
        Err(ColumnRegistryError::DuplicateName { name }) if name == "INPUT"
    ));
    let mut builder = AuxiliaryBuilder::new(&base());
    builder
        .allocate("test", "empty", "empty", ColumnWidth::Boolean, 0, |_, _| Ok(()))
        .unwrap();
    assert!(builder.finish().is_err());
}

#[test]
fn extension_preserves_gadget_catalog_and_rejects_insufficient_columns() {
    let mut rows = R1csBuilder::new(2, 0).unwrap();
    ZeroTest::column(1, 2, 3).push_constraints(&mut rows.tagged(ConstraintTag::new("zero", ())));
    let base = rows.build(&test_columns(4)).unwrap();
    let columns = ColumnRegistry::new([ColumnFamilySpec {
        region: "test",
        start: 0,
        len: 5,
        name: "ALL".into(),
        role: "test",
        width: ColumnWidth::Field,
    }])
    .unwrap();
    let mut rows = R1csBuilder::from_relation(&base);
    rows.tagged(ConstraintTag::new("derived", ()))
        .push_row([(3, F::ONE)], [(0, F::ONE)], [(4, F::ONE)]);
    let extended = rows.build(&columns).unwrap();
    assert_eq!(
        extended.catalog().gadget_occurrences(),
        base.catalog().gadget_occurrences()
    );
    assert_eq!(&extended.catalog().rows()[..2], base.catalog().rows());
    extended
        .check_rows(&[F::ONE, F::ZERO, F::ZERO, F::ONE, F::ONE])
        .unwrap();
    let small = ColumnRegistry::new([ColumnFamilySpec {
        region: "test",
        start: 0,
        len: 3,
        name: "SMALL".into(),
        role: "test",
        width: ColumnWidth::Field,
    }])
    .unwrap();
    assert!(matches!(
        R1csBuilder::from_relation(&base).build(&small),
        Err(R1csBuildError::InsufficientColumns { required: 4, actual: 3 })
    ));
}

fn test_columns(len: usize) -> neo_application::ColumnRegistry {
    neo_application::ColumnRegistry::new([neo_application::ColumnFamilySpec {
        region: "test",
        start: 0,
        len,
        name: "ALL".into(),
        role: "test columns",
        width: neo_application::ColumnWidth::Field,
    }])
    .unwrap()
}

#[test]
fn final_registry_must_cover_public_inputs() {
    let mut rows = R1csBuilder::new(2, 0).unwrap();
    rows.tagged(ConstraintTag::new("identity", ()))
        .push_row([(0, F::ONE)], [(0, F::ONE)], [(0, F::ONE)]);
    assert!(matches!(
        rows.build(&test_columns(1)),
        Err(R1csBuildError::PublicInputCount {
            count: 2,
            column_count: 1
        })
    ));
}

#[test]
fn extension_retains_unreferenced_base_columns() {
    let mut rows = R1csBuilder::new(1, 0).unwrap();
    rows.tagged(ConstraintTag::new("identity", ()))
        .push_row([(0, F::ONE)], [(0, F::ONE)], [(0, F::ONE)]);
    let base = rows.build(&test_columns(4)).unwrap();
    assert!(matches!(
        R1csBuilder::from_relation(&base).build(&test_columns(3)),
        Err(R1csBuildError::InsufficientColumns { required: 4, actual: 3 })
    ));
}
