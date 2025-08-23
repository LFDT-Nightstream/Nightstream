use neo_arithmetize::*;
use neo_ccs::{CcsInstance, CcsWitness, check_satisfiability};
use neo_fields::{embed_base_to_ext, F, ExtF};
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_fibonacci_ccs_small() {
    let length = 5;
    let ccs = fibonacci_ccs(length);

    // Expected Fibonacci sequence: [0, 1, 1, 2, 3]
    let fib_sequence = vec![0u64, 1, 1, 2, 3];
    let z: Vec<ExtF> = fib_sequence
        .iter()
        .map(|&x| embed_base_to_ext(F::from_u64(x)))
        .collect();

    let witness = CcsWitness { z };

    // Create a dummy instance for satisfiability check
    let dummy_instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    // Check satisfiability - should pass for correct Fibonacci sequence
    assert!(check_satisfiability(&ccs, &dummy_instance, &witness));
}

#[test]
fn test_fibonacci_ccs_wrong_sequence() {
    let length = 5;
    let ccs = fibonacci_ccs(length);

    // Wrong sequence: [0, 1, 2, 2, 3] (x_2 should be 1, not 2)
    let wrong_sequence = vec![0u64, 1, 2, 2, 3];
    let z: Vec<ExtF> = wrong_sequence
        .iter()
        .map(|&x| embed_base_to_ext(F::from_u64(x)))
        .collect();

    let witness = CcsWitness { z };

    // Create a dummy instance for satisfiability check
    let dummy_instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    // Check satisfiability - should fail for incorrect sequence
    assert!(!check_satisfiability(&ccs, &dummy_instance, &witness));
}

#[test]
fn test_fibonacci_ccs_trivial() {
    let ccs_0 = fibonacci_ccs(0);
    assert_eq!(ccs_0.num_constraints, 0);

    let ccs_1 = fibonacci_ccs(1);
    assert_eq!(ccs_1.num_constraints, 0);

    let ccs_2 = fibonacci_ccs(2);
    assert_eq!(ccs_2.num_constraints, 0); // No constraints for length 2
}
