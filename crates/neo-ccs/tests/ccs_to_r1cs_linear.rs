// neo-ccs/tests/ccs_to_r1cs_linear.rs
use neo_ccs::{CcsInstance, CcsStructure, CcsWitness, mv_poly};
use neo_ccs::integration::convert_ccs_for_spartan2;
use neo_fields::{from_base, ExtF, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

#[test]
fn ccs_to_r1cs_linear_simple() {
    println!("🧪 Testing CCS → R1CS conversion on simple addition");
    
    // Create a simple CCS: a + b = c (3 variables, 1 constraint)
    let mats = vec![
        // Matrix A: selects first variable (a)
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO], 3),
        // Matrix B: selects second variable (b)  
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO], 3),
        // Matrix C: selects third variable (c)
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE], 3),
    ];

    // Constraint: a + b - c = 0, so f(x0, x1, x2) = x0 + x1 - x2
    let f = mv_poly(|inputs: &[ExtF]| {
        if inputs.len() != 3 {
            ExtF::ZERO
        } else {
            inputs[0] + inputs[1] - inputs[2]
        }
    }, 1);

    let structure = CcsStructure::new(mats, f);

    // Create witness: a=2, b=3, c=5
    let witness_values = vec![
        from_base(F::from_u64(2)),
        from_base(F::from_u64(3)),
        from_base(F::from_u64(5)),
    ];
    let witness = CcsWitness { z: witness_values };

    // Create instance with no public inputs
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    #[allow(non_snake_case)]
    let ((A, B, C), x, w) = convert_ccs_for_spartan2(&structure, &instance, &witness)
        .expect("CCS → R1CS conversion should succeed");
    
    assert!(x.is_empty(), "No public inputs expected");
    assert_eq!(w.len(), 3, "Witness should have 3 variables");

    // Check each row: (A_row · v) * (B_row · v) == (C_row · v)
    let mut v = Vec::with_capacity(1 + w.len());
    v.push(F::ONE); // The constant "1" wire
    v.extend_from_slice(&w);

    let dot = |row: &Vec<F>| row.iter().zip(v.iter()).fold(F::ZERO, |acc, (a, b)| acc + *a * *b);

    for i in 0..A.len() {
        let lhs = dot(&A[i]) * dot(&B[i]);
        let rhs = dot(&C[i]);
        assert_eq!(lhs, rhs, "R1CS constraint {} failed: {} ≠ {}", i, lhs.as_canonical_u64(), rhs.as_canonical_u64());
    }

    println!("✅ CCS → R1CS conversion test passed");
    println!("   Converted {} CCS constraints to {} R1CS constraints", structure.num_constraints, A.len());
    println!("   R1CS matrix dimensions: {} × {} (including constant wire)", A.len(), A[0].len());
    println!("   Witness variables: {}", w.len());
    println!("   Witness values: {:?}", w.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
}
