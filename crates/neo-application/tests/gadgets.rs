use neo_application::{ConstraintTag, GadgetDescriptor, R1csBuilder, ZeroTest};
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ONE: usize = 0;
const VALUE: usize = 1;
const INVERSE: usize = 2;
const IS_ZERO: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Predicate,
}

const ZERO_TEST: ZeroTest = ZeroTest {
    value: VALUE,
    inverse: INVERSE,
    is_zero: IS_ZERO,
};

#[test]
fn zero_test_retains_semantics_and_assigns_satisfying_auxiliaries() {
    let mut builder = R1csBuilder::new(4, 1, ONE).expect("valid test layout");
    let tag = ConstraintTag::new("input is zero", Owner::Predicate);
    ZERO_TEST.push_constraints(&mut builder.tagged(tag.clone()));
    let relation = builder.build().expect("valid zero-test relation");

    let [occurrence] = relation.catalog().gadget_occurrences() else {
        panic!("expected one retained gadget occurrence");
    };
    assert_eq!(occurrence.tag(), &tag);
    assert_eq!(occurrence.descriptor(), &GadgetDescriptor::ZeroTest(ZERO_TEST));
    assert_eq!(occurrence.row_range(), &(0..2));

    for value in [F::ZERO, F::from_u64(9)] {
        let mut assignment = [F::ONE, value, F::ZERO, F::ZERO];
        ZERO_TEST.assign(&mut assignment);
        check_ccs_rowwise_zero(relation.structure(), &assignment[..1], &assignment[1..])
            .expect("descriptor assignment satisfies its emitted rows");
    }

    let mut tampered = [F::ONE, F::from_u64(9), F::ZERO, F::ZERO];
    ZERO_TEST.assign(&mut tampered);
    tampered[IS_ZERO] = F::ONE;
    assert!(check_ccs_rowwise_zero(relation.structure(), &tampered[..1], &tampered[1..]).is_err());
}
