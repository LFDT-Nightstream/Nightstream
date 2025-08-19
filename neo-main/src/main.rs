use neo_fold::FoldState;
use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use neo_decomp::decomp_b;
use p3_matrix::dense::RowMajorMatrix;
use p3_field::PrimeCharacteristicRing;

fn setup_test_structure() -> CcsStructure {
    let a = RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO], 3);
    let b = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO], 3);
    let c = RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE], 3);
    let mats = vec![a, b, c];
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() != 3 {
                ExtF::ZERO
            } else {
                inputs[0] * inputs[1] - inputs[2]
            }
        },
        2,
    );
    CcsStructure::new(mats, f)
}

fn main() {
    println!("Generating and verifying a proof...");

    let structure = setup_test_structure();
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);

    let z1_base = vec![F::ONE, F::from_u64(3), F::from_u64(3)];
    let z1 = z1_base.iter().copied().map(from_base).collect();
    let witness1 = CcsWitness { z: z1 };
    let z1_mat = decomp_b(&z1_base, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&z1_mat, &params);
    let mut t1 = Vec::new();
    let (commit1, _, _, _) = committer.commit(&w1, &mut t1).expect("commit");
    let instance1 = CcsInstance {
        commitment: commit1,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    let z2_base = vec![F::from_u64(2), F::from_u64(2), F::from_u64(4)];
    let z2 = z2_base.iter().copied().map(from_base).collect();
    let witness2 = CcsWitness { z: z2 };
    let z2_mat = decomp_b(&z2_base, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&z2_mat, &params);
    let mut t2 = Vec::new();
    let (commit2, _, _, _) = committer.commit(&w2, &mut t2).expect("commit");
    let instance2 = CcsInstance {
        commitment: commit2,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };

    let mut state = FoldState::new(structure.clone());
    let proof = state.generate_proof((instance1.clone(), witness1.clone()), (instance2.clone(), witness2.clone()), &committer).unwrap();

    let verifier_state = FoldState::new(structure);
    let verify_result = verifier_state.verify(&proof.transcript, &committer);

    if verify_result {
        println!("Proof verification succeeded!");
    } else {
        println!("Proof verification failed!");
    }
}
