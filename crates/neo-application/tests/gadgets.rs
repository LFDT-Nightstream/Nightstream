use neo_application::{ConditionalSelect, ConstraintTag, GadgetDescriptor, R1csBuilder, ZeroTest};
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ONE: usize = 0;
const LHS: usize = 1;
const RHS: usize = 2;
const INVERSE: usize = 3;
const IS_ZERO: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Predicate,
}

const ZERO_TEST: ZeroTest<2> = ZeroTest {
    expression: [(LHS, F::ONE), (RHS, F::NEG_ONE)],
    inverse: INVERSE,
    is_zero: IS_ZERO,
};

#[test]
fn zero_test_retains_semantics_and_assigns_satisfying_auxiliaries() {
    let mut builder = R1csBuilder::new(5, 1, ONE).expect("valid test layout");
    let tag = ConstraintTag::new("inputs are equal", Owner::Predicate);
    ZERO_TEST.push_constraints(&mut builder.tagged(tag.clone()));
    let relation = builder.build().expect("valid zero-test relation");

    let [occurrence] = relation.catalog().gadget_occurrences() else {
        panic!("expected one retained gadget occurrence");
    };
    assert_eq!(occurrence.tag(), &tag);
    assert_eq!(
        occurrence.descriptor(),
        &GadgetDescriptor::ZeroTest {
            expression: ZERO_TEST.expression.to_vec(),
            inverse: INVERSE,
            is_zero: IS_ZERO,
        }
    );
    assert_eq!(occurrence.row_range(), &(0..2));

    for (lhs, rhs) in [(F::from_u64(9), F::from_u64(9)), (F::from_u64(9), F::from_u64(4))] {
        let mut assignment = [F::ONE, lhs, rhs, F::ZERO, F::ZERO];
        ZERO_TEST.assign(&mut assignment);
        check_ccs_rowwise_zero(relation.structure(), &assignment[..1], &assignment[1..])
            .expect("descriptor assignment satisfies its emitted rows");
    }

    let mut tampered = [F::ONE, F::from_u64(9), F::from_u64(4), F::ZERO, F::ZERO];
    ZERO_TEST.assign(&mut tampered);
    tampered[IS_ZERO] = F::ONE;
    assert!(check_ccs_rowwise_zero(relation.structure(), &tampered[..1], &tampered[1..]).is_err());
}

const ACTIVATION: usize = 1;
const CONDITION: usize = 2;
const SELECT_LHS: usize = 3;
const SELECT_RHS: usize = 4;
const OUTPUT: usize = 5;
const DELTA: usize = 6;

const SELECT: ConditionalSelect = ConditionalSelect {
    activation: ACTIVATION,
    condition: [(CONDITION, F::ONE)],
    lhs: SELECT_LHS,
    rhs: SELECT_RHS,
    output: OUTPUT,
    delta: DELTA,
};

#[test]
fn conditional_select_retains_semantics_and_assigns_only_its_delta() {
    let mut builder = R1csBuilder::new(7, 1, ONE).expect("valid test layout");
    let tag = ConstraintTag::new("select input", Owner::Predicate);
    SELECT.push_constraints(&mut builder.tagged(tag.clone()));
    let relation = builder.build().expect("valid conditional-select relation");

    let [occurrence] = relation.catalog().gadget_occurrences() else {
        panic!("expected one retained gadget occurrence");
    };
    assert_eq!(occurrence.tag(), &tag);
    assert_eq!(
        occurrence.descriptor(),
        &GadgetDescriptor::ConditionalSelect {
            activation: ACTIVATION,
            condition: SELECT.condition.to_vec(),
            lhs: SELECT_LHS,
            rhs: SELECT_RHS,
            output: OUTPUT,
            delta: DELTA,
        }
    );
    assert_eq!(occurrence.row_range(), &(0..2));

    for (activation, condition, output) in [
        (F::ONE, F::ZERO, F::from_u64(4)),
        (F::ONE, F::ONE, F::from_u64(9)),
        (F::ZERO, F::ONE, F::from_u64(17)),
    ] {
        let mut assignment = [
            F::ONE,
            activation,
            condition,
            F::from_u64(9),
            F::from_u64(4),
            output,
            F::ZERO,
        ];
        SELECT.assign_delta(&mut assignment);
        assert_eq!(
            assignment[OUTPUT], output,
            "the gadget must not overwrite semantic output"
        );
        check_ccs_rowwise_zero(relation.structure(), &assignment[..1], &assignment[1..])
            .expect("descriptor assignment satisfies its emitted rows");
    }

    let mut tampered = [
        F::ONE,
        F::ZERO,
        F::ONE,
        F::from_u64(9),
        F::from_u64(4),
        F::ZERO,
        F::ZERO,
    ];
    SELECT.assign_delta(&mut tampered);
    tampered[DELTA] += F::ONE;
    assert!(check_ccs_rowwise_zero(relation.structure(), &tampered[..1], &tampered[1..]).is_err());
}
