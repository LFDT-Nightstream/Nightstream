use neo_application::{
    range_checked_variable_widths, ColumnFamilySpec, ColumnWidth, R1csBuilder, RangeCheckAssignmentError,
    RangeCheckBitFamily, RangeCheckLayout,
};
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ONE: usize = 0;
const BYTE: usize = 1;
const WORD: usize = 2;
const TWO_BITS: usize = 3;
const FIELD: usize = 4;
const BASE_WIDTH: usize = 5;

fn range_layout() -> RangeCheckLayout {
    RangeCheckLayout::new(
        [
            ColumnFamilySpec {
                region: "base",
                start: ONE,
                len: 1,
                name: "ONE",
                role: "constant one",
                width: ColumnWidth::Boolean,
            },
            ColumnFamilySpec {
                region: "base",
                start: BYTE,
                len: 1,
                name: "BYTE",
                role: "byte value",
                width: ColumnWidth::Byte,
            },
            ColumnFamilySpec {
                region: "base",
                start: WORD,
                len: 1,
                name: "WORD",
                role: "32-bit word",
                width: ColumnWidth::U32,
            },
            ColumnFamilySpec {
                region: "base",
                start: TWO_BITS,
                len: 1,
                name: "TWO_BITS",
                role: "two-bit value",
                width: ColumnWidth::Bits(2),
            },
            ColumnFamilySpec {
                region: "base",
                start: FIELD,
                len: 1,
                name: "FIELD",
                role: "unrestricted field value",
                width: ColumnWidth::Field,
            },
        ],
        RangeCheckBitFamily {
            region: "range_bits",
            name: "RANGE_BITS",
            role: "explicit decomposition bits",
        },
    )
    .expect("valid range-check layout")
}

#[test]
fn range_check_layout_allocates_and_assigns_explicit_bits() {
    let layout = range_layout();
    assert_eq!(layout.base_column_count(), BASE_WIDTH);
    assert_eq!(layout.bit_columns(), BASE_WIDTH..BASE_WIDTH + 42);
    assert_eq!(layout.bit_columns_for(ONE), None);
    assert_eq!(layout.bit_columns_for(BYTE), Some(BASE_WIDTH..BASE_WIDTH + 8));
    assert_eq!(layout.bit_columns_for(WORD), Some(BASE_WIDTH + 8..BASE_WIDTH + 40));
    assert_eq!(layout.bit_columns_for(TWO_BITS), Some(BASE_WIDTH + 40..BASE_WIDTH + 42));
    assert_eq!(layout.bit_columns_for(FIELD), None);
    assert_eq!(layout.bit_columns_for(layout.column_count()), None);

    let bit_family = layout
        .columns()
        .family_for_column(BASE_WIDTH)
        .expect("generated bits must belong to the completed registry");
    assert_eq!(bit_family.region, "range_bits");
    assert_eq!(bit_family.width, ColumnWidth::Boolean);
    assert_eq!(
        range_checked_variable_widths(layout.columns()),
        [1, 8, 32, 2, 64]
            .into_iter()
            .chain([1; 42])
            .collect::<Vec<_>>()
    );

    let mut witness = vec![
        F::ONE,
        F::from_u64(0xa5),
        F::from_u64(0x89ab_cdef),
        F::from_u64(3),
        -F::ONE,
    ];
    layout
        .assign_bits(&mut witness)
        .expect("base witness must extend");
    assert_eq!(witness.len(), layout.column_count());
    for (index, bit) in layout.bit_columns_for(BYTE).unwrap().enumerate() {
        assert_eq!(witness[bit], F::from_u64((0xa5 >> index) & 1));
    }
    for (index, bit) in layout.bit_columns_for(WORD).unwrap().enumerate() {
        assert_eq!(witness[bit], F::from_u64((0x89ab_cdef >> index) & 1));
    }
    let two_bits = layout.bit_columns_for(TWO_BITS).unwrap();
    assert_eq!(witness[two_bits.start], F::ONE);
    assert_eq!(witness[two_bits.start + 1], F::ONE);
}

#[test]
fn range_check_constraints_accept_valid_values_and_reject_overflow() {
    let layout = range_layout();
    let mut builder = R1csBuilder::new(layout.column_count(), 1, ONE).unwrap();
    layout.push_constraints(
        &mut builder.tagged(neo_application::ConstraintTag::new("range checks", "test")),
        "range check",
    );
    let relation = builder.build().unwrap();

    let mut witness = vec![
        F::ONE,
        F::from_u64(u8::MAX.into()),
        F::from_u64(u32::MAX.into()),
        F::from_u64(3),
        -F::ONE,
    ];
    layout.assign_bits(&mut witness).unwrap();
    check_ccs_rowwise_zero(relation.structure(), &witness[..1], &witness[1..])
        .expect("valid declared widths must satisfy the generated rows");

    witness[BYTE] = F::from_u64(1 << 8);
    layout.assign_bits(&mut witness).unwrap();
    check_ccs_rowwise_zero(relation.structure(), &witness[..1], &witness[1..])
        .expect_err("a byte overflow must fail recomposition");

    witness[BYTE] = F::from_u64(u8::MAX.into());
    witness[TWO_BITS] = F::from_u64(4);
    layout.assign_bits(&mut witness).unwrap();
    check_ccs_rowwise_zero(relation.structure(), &witness[..1], &witness[1..])
        .expect_err("a two-bit overflow must fail recomposition");

    assert_eq!(
        layout.assign_bits(&mut vec![F::ZERO; 2]).unwrap_err(),
        RangeCheckAssignmentError::WitnessWidth {
            base: BASE_WIDTH,
            range_checked: BASE_WIDTH + 42,
            actual: 2,
        }
    );
}
