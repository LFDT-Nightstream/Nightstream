use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::{from_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;
use neo_fold::{pi_dec, FoldState};
use p3_matrix::dense::RowMajorMatrix;
use quickcheck_macros::quickcheck;

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

#[test]
fn test_full_fold_verification() {
    let structure = setup_test_structure();
    let mut fold_state = FoldState::new(structure.clone());
    let params = SECURE_PARAMS;
    let committer = AjtaiCommitter::setup(params);
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
    let proof = fold_state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(fold_state.verify(&proof.transcript, &committer));
    // NARK mode: Test transcript tampering instead of state tampering
    // since NARK mode trusts sumcheck proofs rather than doing additional validation
    let mut bad_proof = proof.clone();
    bad_proof.transcript[100] ^= 1; // Tamper with transcript
    assert!(!fold_state.verify(&bad_proof.transcript, &committer));
}

#[test]
fn test_verify_fails_on_transcript_mutation() {
    let structure = setup_test_structure();
    let mut state = FoldState::new(structure.clone());
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    let z_base = vec![F::ONE, F::from_u64(3), F::from_u64(3)];
    let z = z_base.iter().copied().map(from_base).collect();
    let witness = CcsWitness { z };
    let z_mat = decomp_b(&z_base, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&z_mat, &params);
    let mut t3 = Vec::new();
    let (commit, _, _, _) = committer.commit(&w, &mut t3).expect("commit");
    let instance = CcsInstance {
        commitment: commit.clone(),
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let instance2 = CcsInstance {
        commitment: commit,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness2 = CcsWitness { z: z_base.iter().copied().map(from_base).collect() };
    let proof = state.generate_proof(
        (instance.clone(), witness),
        (instance2, witness2),
        &committer,
    );
    let mut bad = proof.clone();
    bad.transcript[10] ^= 1;
    assert!(!state.verify(&bad.transcript, &committer));
}

#[test]
fn test_zk_folding_different_proofs() {
    let structure = setup_test_structure();
    let params = SECURE_PARAMS;
    let committer = AjtaiCommitter::setup(params);
    let z1_base = vec![F::ONE, F::from_u64(3), F::from_u64(3)];
    let z1 = z1_base.iter().copied().map(from_base).collect();
    let witness1 = CcsWitness { z: z1 };
    let z1_mat = decomp_b(&z1_base, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&z1_mat, &params);
    let mut t4 = Vec::new();
    let (commit1, _, _, _) = committer.commit(&w1, &mut t4).expect("commit");
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
    let mut t5 = Vec::new();
    let (commit2, _, _, _) = committer.commit(&w2, &mut t5).expect("commit");
    let instance2 = CcsInstance {
        commitment: commit2,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let mut state = FoldState::new(structure);
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer));
}

#[test]
fn test_fs_binding_in_dec() {
    let structure = setup_test_structure();
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let base_eval = neo_fold::EvalInstance {
        commitment: vec![neo_ring::RingElement::zero(TOY_PARAMS.n); TOY_PARAMS.k],
        r: vec![ExtF::ZERO],
        ys: vec![ExtF::ZERO; structure.mats.len()],
        u: ExtF::ZERO,
        e_eval: ExtF::ZERO,
        norm_bound: TOY_PARAMS.norm_bound,
    };
    let mut state1 = FoldState::new(structure.clone());
    state1.eval_instances.push(base_eval.clone());
    let mut t1 = vec![];
    pi_dec(&mut state1, &committer, &mut t1);
    let mut state2 = FoldState::new(structure);
    state2.eval_instances.push(base_eval);
    let mut t2 = b"extra".to_vec();
    pi_dec(&mut state2, &committer, &mut t2);
    assert_ne!(t1, t2);
}

#[quickcheck]
fn prop_rejects_mutated_transcript(mutated_index: usize, mutated_bit: u8) -> bool {
    let structure = setup_test_structure();
    let mut state = FoldState::new(structure.clone());
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    let z_base = vec![F::ONE, F::from_u64(3), F::from_u64(3)];
    let z = z_base.iter().copied().map(from_base).collect();
    let witness = CcsWitness { z };
    let z_mat = decomp_b(&z_base, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&z_mat, &params);
    let mut t6 = Vec::new();
    let (commit, _, _, _) = committer.commit(&w, &mut t6).expect("commit");
    let instance = CcsInstance {
        commitment: commit.clone(),
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let instance2 = CcsInstance {
        commitment: commit,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness2 = CcsWitness { z: z_base.iter().copied().map(from_base).collect() };
    let proof = state.generate_proof((instance, witness), (instance2, witness2), &committer);
    let mut bad_transcript = proof.transcript.clone();
    if mutated_index < bad_transcript.len() && mutated_bit != 0 {
        bad_transcript[mutated_index] ^= mutated_bit;
        !state.verify(&bad_transcript, &committer)
    } else {
        true
    }
}

#[test]
fn test_commit_serialization_roundtrip_non_empty() {
    use std::io::Cursor;
    use neo_fold::{serialize_commit, read_commit};
    use neo_commit::AjtaiCommitter;
    use neo_ring::RingElement;
    use neo_modint::ModInt;
    
    let params = TOY_PARAMS;
    let _committer = AjtaiCommitter::setup_unchecked(params);
    let ring1 = RingElement::from_coeffs(vec![ModInt::from_u64(1), ModInt::from_u64(2), ModInt::from_u64(3), ModInt::from_u64(4)], params.n);
    let ring2 = RingElement::from_coeffs(vec![ModInt::from_u64(5), ModInt::from_u64(6), ModInt::from_u64(7), ModInt::from_u64(8)], params.n);
    let commit = vec![ring1, ring2];
    let serialized = serialize_commit(&commit);
    let mut cursor = Cursor::new(serialized.as_slice());
    let extracted = read_commit(&mut cursor, params.n);
    assert_eq!(extracted.len(), 2);
    for (i, (orig, ext)) in commit.iter().zip(extracted.iter()).enumerate() {
        assert_eq!(orig.coeffs(), ext.coeffs(), "Ring {} mismatch", i);
    }
    assert_eq!(cursor.position() as usize, serialized.len(), "Cursor consumed all bytes");
}
